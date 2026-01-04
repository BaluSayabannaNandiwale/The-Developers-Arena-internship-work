"""
Training pipeline for sentiment analysis model.
Includes early stopping, model checkpoints, metrics tracking, and GPU support.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard
)

from src.models.lstm_model import build_bidirectional_lstm_model
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def check_gpu_availability() -> bool:
    """
    Check if GPU is available for training.
    
    Returns:
        True if GPU is available, False otherwise
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            logger.info(f"  - {gpu}")
        return True
    else:
        logger.info("No GPU available. Training will use CPU.")
        return False


def setup_gpu(gpu_memory_growth: bool = True) -> None:
    """
    Configure GPU settings for optimal performance.
    
    Args:
        gpu_memory_growth: Enable memory growth to avoid allocating all GPU memory
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if gpu_memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU configured successfully")
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")


def create_callbacks(
    checkpoint_dir: str,
    model_name: str = 'sentiment_model',
    patience: int = 5,
    monitor: str = 'val_loss',
    mode: str = 'min'
) -> list:
    """
    Create training callbacks for model optimization.
    
    Callbacks:
    1. EarlyStopping: Stop training if no improvement
    2. ModelCheckpoint: Save best model
    3. ReduceLROnPlateau: Reduce learning rate on plateau
    4. CSVLogger: Log training metrics
    5. TensorBoard: Visualize training (optional)
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        model_name: Name for saved model files
        patience: Number of epochs to wait before early stopping
        monitor: Metric to monitor
        mode: 'min' or 'max' for the monitored metric
    
    Returns:
        List of callbacks
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Early Stopping: Stop training if validation loss doesn't improve
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            verbose=1,
            restore_best_weights=True
        ),
        
        # Model Checkpoint: Save best model during training
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'{model_name}_best.h5'),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        ),
        
        # Reduce Learning Rate: Reduce LR when loss plateaus
        ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # CSV Logger: Save training history to CSV
        CSVLogger(
            filename=os.path.join(checkpoint_dir, 'training_history.csv'),
            append=False
        )
    ]
    
    # TensorBoard callback (optional, for visualization)
    try:
        tensorboard_dir = os.path.join(checkpoint_dir, 'tensorboard_logs')
        callbacks.append(
            TensorBoard(
                log_dir=tensorboard_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        )
        logger.info(f"TensorBoard logs will be saved to {tensorboard_dir}")
    except Exception as e:
        logger.warning(f"TensorBoard callback not added: {e}")
    
    logger.info(f"Created {len(callbacks)} callbacks")
    return callbacks


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    vocab_size: int,
    max_length: int = 100,
    num_classes: int = 3,
    batch_size: int = 32,
    epochs: int = 50,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    checkpoint_dir: str = 'models/checkpoints',
    model_name: str = 'sentiment_model',
    use_gpu: bool = True
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Train the sentiment analysis model.
    
    Args:
        X_train: Training sequences
        y_train: Training labels
        X_val: Validation sequences
        y_val: Validation labels
        vocab_size: Vocabulary size
        max_length: Maximum sequence length
        num_classes: Number of classes
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        embedding_dim: Embedding dimension
        lstm_units: Number of LSTM units
        checkpoint_dir: Directory to save checkpoints
        model_name: Name for saved model
        use_gpu: Whether to use GPU if available
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    logger.info("=" * 60)
    logger.info("Starting Model Training")
    logger.info("=" * 60)
    
    # Setup GPU if requested
    if use_gpu:
        setup_gpu()
    check_gpu_availability()
    
    # Build model
    logger.info("Building model architecture...")
    model = build_bidirectional_lstm_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        max_length=max_length,
        num_classes=num_classes
    )
    
    # Create callbacks
    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        model_name=model_name,
        patience=5,
        monitor='val_loss',
        mode='min'
    )
    
    # Training parameters
    logger.info(f"Training parameters:")
    logger.info(f"  - Training samples: {len(X_train)}")
    logger.info(f"  - Validation samples: {len(X_val)}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Max epochs: {epochs}")
    logger.info(f"  - Vocabulary size: {vocab_size}")
    logger.info(f"  - Sequence length: {max_length}")
    
    # Train model
    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Training completed!")
    
    # Load best model (saved by ModelCheckpoint)
    best_model_path = os.path.join(checkpoint_dir, f'{model_name}_best.h5')
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        model = keras.models.load_model(best_model_path)
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, f'{model_name}_final.h5')
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save training configuration
    config = {
        'vocab_size': vocab_size,
        'max_length': max_length,
        'num_classes': num_classes,
        'embedding_dim': embedding_dim,
        'lstm_units': lstm_units,
        'batch_size': batch_size,
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1]
    }
    
    config_path = os.path.join(checkpoint_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Training configuration saved to {config_path}")
    
    return model, history


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32
) -> Dict:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        X_test: Test sequences
        y_test: Test labels
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model on test set...")
    
    # Get predictions
    predictions = model.predict(X_test, batch_size=batch_size, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_test, predicted_classes)
    precision = precision_score(y_test, predicted_classes, average='weighted')
    recall = recall_score(y_test, predicted_classes, average='weighted')
    f1 = f1_score(y_test, predicted_classes, average='weighted')
    cm = confusion_matrix(y_test, predicted_classes)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    logger.info("Test Set Evaluation:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  Confusion Matrix:\n{cm}")
    
    return metrics

