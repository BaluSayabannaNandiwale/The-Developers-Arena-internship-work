"""
Main training script for sentiment analysis model.
Run this script to train the model from scratch.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.preprocess import load_and_prepare_data
from src.training.train import train_model, evaluate_model
from src.data_processing.preprocess import TextPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("SENTIMENT ANALYSIS MODEL TRAINING")
    logger.info("=" * 60)
    
    # Configuration
    data_path = "data/raw/sample_sentiment_data.csv"
    checkpoint_dir = "models/checkpoints"
    tokenizer_path = "models/tokenizer.pkl"
    
    # Check if data exists
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run: python scripts/generate_sample_data.py first")
        return
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare_data(
        data_path=data_path,
        text_column='text',
        label_column='label',
        test_size=0.2,
        random_state=42
    )
    
    # Save tokenizer
    Path(tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save_tokenizer(tokenizer_path)
    logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    # Split training data into train and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Get vocabulary size
    vocab_size = preprocessor.get_vocab_size()
    max_length = preprocessor.max_length
    num_classes = len(preprocessor.label_encoder.classes_)
    
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Max sequence length: {max_length}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Train model
    model, history = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        vocab_size=vocab_size,
        max_length=max_length,
        num_classes=num_classes,
        batch_size=32,
        epochs=50,
        embedding_dim=128,
        lstm_units=64,
        checkpoint_dir=checkpoint_dir,
        model_name='sentiment_model',
        use_gpu=True
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save evaluation metrics
    import json
    metrics_path = os.path.join(checkpoint_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Test metrics saved to {metrics_path}")
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {checkpoint_dir}")
    logger.info(f"Tokenizer saved to: {tokenizer_path}")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()

