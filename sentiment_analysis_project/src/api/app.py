"""
FastAPI application for sentiment analysis.
Provides REST API endpoints for predictions, health checks, and metrics.
"""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from src.inference.predict import SentimentPredictor, load_predictor
from src.monitoring.metrics import get_metrics_collector, RequestTimer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready NLP Sentiment Analysis API using Bidirectional LSTM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[SentimentPredictor] = None
metrics_collector = get_metrics_collector()


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")
        if len(v) > 100:
            raise ValueError("Maximum 100 texts allowed per batch")
        return [text.strip() for text in v if text.strip()]


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    text: str
    sentiment: str
    confidence: float
    probabilities: dict


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    metrics: dict


@app.on_event("startup")
async def startup_event():
    """Initialize model on application startup."""
    global predictor
    
    logger.info("Starting up Sentiment Analysis API...")
    
    # Get model paths from environment variables or use defaults
    model_path = os.getenv(
        'MODEL_PATH',
        'models/checkpoints/sentiment_model_best.h5'
    )
    tokenizer_path = os.getenv(
        'TOKENIZER_PATH',
        'models/tokenizer.pkl'
    )
    
    try:
        # Check if model files exist
        if not Path(model_path).exists():
            logger.warning(f"Model file not found at {model_path}. API will start but predictions will fail.")
            logger.info("To train a model, run: python -m src.training.train")
            return
        
        if not Path(tokenizer_path).exists():
            logger.warning(f"Tokenizer file not found at {tokenizer_path}. API will start but predictions will fail.")
            return
        
        # Load predictor
        predictor = load_predictor(
            model_path=model_path,
            tokenizer_path=tokenizer_path
        )
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model info: {predictor.get_model_info()}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("API will start but predictions will fail until model is loaded.")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint.
    Returns API status and model availability.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Get API metrics and performance statistics.
    Returns request counts, latency, error rates, and prediction statistics.
    """
    metrics = metrics_collector.get_metrics()
    return MetricsResponse(metrics=metrics)


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment for a single text.
    
    Args:
        request: Prediction request with text
    
    Returns:
        Prediction result with sentiment, confidence, and probabilities
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    with RequestTimer(metrics_collector, "/predict"):
        try:
            # Get prediction
            result = predictor.predict_single(request.text)
            
            # Record prediction metrics
            metrics_collector.record_prediction(
                sentiment=result['sentiment'],
                confidence=result['confidence']
            )
            
            return PredictionResponse(**result)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch_sentiment(request: BatchPredictionRequest):
    """
    Predict sentiment for multiple texts in batch.
    
    Args:
        request: Batch prediction request with list of texts
    
    Returns:
        List of prediction results
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )
    
    with RequestTimer(metrics_collector, "/batch_predict"):
        try:
            # Get predictions
            results = predictor.predict_batch(request.texts)
            
            # Record prediction metrics
            for result in results:
                metrics_collector.record_prediction(
                    sentiment=result['sentiment'],
                    confidence=result['confidence']
                )
            
            # Convert to response format
            prediction_responses = [
                PredictionResponse(**result) for result in results
            ]
            
            return BatchPredictionResponse(
                predictions=prediction_responses,
                total=len(prediction_responses)
            )
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch prediction failed: {str(e)}"
            )


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded."
        )
    
    return predictor.get_model_info()


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

