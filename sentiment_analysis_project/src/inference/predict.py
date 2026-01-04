"""
Inference pipeline for sentiment analysis predictions.
Handles single and batch predictions with confidence scores.
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Union, Optional
from pathlib import Path

from src.data_processing.preprocess import TextPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SentimentPredictor:
    """
    Sentiment analysis predictor class.
    Handles model loading, text preprocessing, and prediction.
    """
    
    def __init__(
        self,
        model_path: str,
        preprocessor: Optional[TextPreprocessor] = None,
        tokenizer_path: Optional[str] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved Keras model
            preprocessor: Preprocessor instance (optional)
            tokenizer_path: Path to saved tokenizer (optional)
        """
        logger.info(f"Loading model from {model_path}")
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Initialize or load preprocessor
        if preprocessor is not None:
            self.preprocessor = preprocessor
        elif tokenizer_path is not None:
            self.preprocessor = TextPreprocessor()
            self.preprocessor.load_tokenizer(tokenizer_path)
        else:
            raise ValueError("Either preprocessor or tokenizer_path must be provided")
        
        # Class names (default, can be overridden)
        self.class_names = ['negative', 'neutral', 'positive']
        
        logger.info("SentimentPredictor initialized")
    
    def predict_single(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text string
        
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        # Preprocess text
        processed = self.preprocessor.preprocess_texts([text], fit=False)
        
        # Get prediction
        prediction = self.model.predict(processed, verbose=0)
        
        # Get predicted class
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class_idx])
        predicted_class = self.class_names[predicted_class_idx]
        
        # Get probabilities for all classes
        probabilities = {
            self.class_names[i]: float(prediction[0][i])
            for i in range(len(self.class_names))
        }
        
        result = {
            'text': text,
            'sentiment': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input text strings
        
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Predicting sentiment for {len(texts)} texts")
        
        # Preprocess all texts
        processed = self.preprocessor.preprocess_texts(texts, fit=False)
        
        # Get predictions
        predictions = self.model.predict(processed, verbose=0)
        
        # Process results
        results = []
        for i, text in enumerate(texts):
            predicted_class_idx = np.argmax(predictions[i])
            confidence = float(predictions[i][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            probabilities = {
                self.class_names[j]: float(predictions[i][j])
                for j in range(len(self.class_names))
            }
            
            results.append({
                'text': text,
                'sentiment': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities
            })
        
        logger.info(f"Completed predictions for {len(texts)} texts")
        return results
    
    def predict_with_confidence_threshold(
        self,
        text: str,
        threshold: float = 0.6
    ) -> Dict[str, Union[str, float, bool]]:
        """
        Predict sentiment with confidence threshold.
        Returns 'uncertain' if confidence is below threshold.
        
        Args:
            text: Input text string
            threshold: Minimum confidence threshold
        
        Returns:
            Dictionary with prediction and uncertainty flag
        """
        result = self.predict_single(text)
        
        if result['confidence'] < threshold:
            result['sentiment'] = 'uncertain'
            result['is_uncertain'] = True
        else:
            result['is_uncertain'] = False
        
        return result
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_parameters': self.model.count_params(),
            'vocab_size': self.preprocessor.get_vocab_size(),
            'max_length': self.preprocessor.max_length
        }


def load_predictor(
    model_path: str,
    tokenizer_path: str,
    class_names: Optional[List[str]] = None
) -> SentimentPredictor:
    """
    Convenience function to load a predictor.
    
    Args:
        model_path: Path to saved model
        tokenizer_path: Path to saved tokenizer
        class_names: Optional list of class names
    
    Returns:
        Initialized SentimentPredictor
    """
    predictor = SentimentPredictor(
        model_path=model_path,
        tokenizer_path=tokenizer_path
    )
    
    if class_names is not None:
        predictor.class_names = class_names
    
    return predictor

