"""
FastAPI Backend for Real Estate Price Prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import pickle
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import uuid
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "ml-pipeline"))
from data.preprocessing import DataPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Real Estate Price Predictor API",
    description="Production-ready API for predicting real estate prices in INR",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made')
BATCH_PREDICTION_COUNTER = Counter('batch_predictions_total', 'Total batch predictions made')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
ERROR_COUNTER = Counter('prediction_errors_total', 'Total prediction errors')

# Load model and preprocessor
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "production_model.pkl"
PREPROCESSOR_PATH = Path(__file__).parent.parent.parent / "models" / "preprocessor.pkl"

model = None
preprocessor = None

def load_model():
    """Load the production model and preprocessor"""
    global model, preprocessor
    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info("✅ Model loaded successfully")
        else:
            logger.warning("⚠️ Model file not found, predictions will fail")
        
        if PREPROCESSOR_PATH.exists():
            preprocessor = DataPreprocessor.load(str(PREPROCESSOR_PATH))
            logger.info("✅ Preprocessor loaded successfully")
        else:
            logger.warning("⚠️ Preprocessor file not found, predictions will fail")
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

# Request/Response models
class PropertyRequest(BaseModel):
    area_sqft: float = Field(..., gt=0, description="Built-up area in square feet")
    bedrooms: int = Field(..., ge=1, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=1, description="Number of bathrooms")
    floor: int = Field(..., ge=1, description="Floor number")
    total_floors: int = Field(..., ge=1, description="Total floors in building")
    property_age: int = Field(..., ge=0, description="Property age in years")
    location: str = Field(..., description="Locality / city zone")
    city: str = Field(..., description="City name")
    property_type: str = Field(..., description="Apartment / Villa / Independent")
    facing: str = Field(..., description="East / West / North / South")
    furnishing: str = Field(..., description="Furnished / Semi-Furnished / Unfurnished")
    parking: int = Field(..., ge=0, description="Parking slots")
    amenities_score: float = Field(..., ge=0, le=1, description="Amenities score (0-1)")
    distance_city_center_km: float = Field(..., ge=0, description="Distance from city center in km")

class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    predicted_price: float
    currency: str
    confidence_interval: dict
    model_version: str
    metadata: dict

class BatchPropertyRequest(BaseModel):
    properties: List[PropertyRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Real Estate Price Predictor API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    preprocessor_status = "loaded" if preprocessor is not None else "not_loaded"
    
    return {
        "status": "healthy" if model and preprocessor else "degraded",
        "model_status": model_status,
        "preprocessor_status": preprocessor_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

@app.post("/api/v1/predict", response_model=PredictionResponse)
@PREDICTION_LATENCY.time()
async def predict_price(request: PropertyRequest):
    """Single prediction endpoint"""
    try:
        PREDICTION_COUNTER.inc()
        
        if model is None or preprocessor is None:
            ERROR_COUNTER.inc()
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please ensure model files exist."
            )
        
        # Validate input
        if request.area_sqft <= 0:
            ERROR_COUNTER.inc()
            raise HTTPException(status_code=400, detail="Area must be positive")
        
        # Prepare input DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Preprocess
        try:
            processed_data = preprocessor.transform(input_data)
        except Exception as e:
            ERROR_COUNTER.inc()
            logger.error(f"Preprocessing error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
        
        # Predict
        try:
            prediction = model.predict(processed_data)[0]
        except Exception as e:
            ERROR_COUNTER.inc()
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        # Generate confidence interval (10% margin)
        margin = prediction * 0.1
        
        return PredictionResponse(
            prediction_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            predicted_price=round(prediction, 2),
            currency="INR",
            confidence_interval={
                "lower_bound": round(prediction - margin, 2),
                "upper_bound": round(prediction + margin, 2)
            },
            model_version="1.0.0",
            metadata={
                "area_sqft": request.area_sqft,
                "location": request.location,
                "property_type": request.property_type,
                "city": request.city
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPropertyRequest):
    """Batch prediction endpoint"""
    try:
        BATCH_PREDICTION_COUNTER.inc()
        
        if model is None or preprocessor is None:
            ERROR_COUNTER.inc()
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please ensure model files exist."
            )
        
        predictions = []
        
        for prop in request.properties:
            try:
                # Prepare input DataFrame
                input_data = pd.DataFrame([prop.dict()])
                
                # Preprocess
                processed_data = preprocessor.transform(input_data)
                
                # Predict
                prediction = model.predict(processed_data)[0]
                
                # Generate confidence interval
                margin = prediction * 0.1
                
                predictions.append(PredictionResponse(
                    prediction_id=str(uuid.uuid4()),
                    timestamp=datetime.now().isoformat(),
                    predicted_price=round(prediction, 2),
                    currency="INR",
                    confidence_interval={
                        "lower_bound": round(prediction - margin, 2),
                        "upper_bound": round(prediction + margin, 2)
                    },
                    model_version="1.0.0",
                    metadata={
                        "area_sqft": prop.area_sqft,
                        "location": prop.location,
                        "property_type": prop.property_type,
                        "city": prop.city
                    }
                ))
            except Exception as e:
                logger.error(f"Error processing property in batch: {str(e)}")
                ERROR_COUNTER.inc()
                # Continue with other predictions
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    print("Starting Real Estate Price Predictor API...")
    print("API will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

