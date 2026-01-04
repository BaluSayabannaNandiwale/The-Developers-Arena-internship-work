"""
Bidirectional LSTM model architecture for sentiment analysis.
Implements deep learning model with embedding, LSTM layers, and dropout.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    GlobalMaxPooling1D
)
from typing import Optional, Dict

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_bidirectional_lstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    max_length: int = 100,
    num_classes: int = 3,
    dropout_rate: float = 0.5,
    lstm_dropout: float = 0.3,
    recurrent_dropout: float = 0.3
) -> keras.Model:
    """
    Build a Bidirectional LSTM model for sentiment analysis.
    
    Architecture:
    1. Embedding Layer: Maps word indices to dense vectors
    2. Bidirectional LSTM Layer 1: Processes sequences in both directions
    3. Dropout: Prevents overfitting
    4. Bidirectional LSTM Layer 2: Second layer for deeper understanding
    5. Global Max Pooling: Extracts most important features
    6. Dense Layer: Feature transformation
    7. Output Layer: Classification (softmax for multi-class)
    
    Args:
        vocab_size: Size of vocabulary (from tokenizer)
        embedding_dim: Dimension of word embeddings
        lstm_units: Number of LSTM units in each direction
        max_length: Maximum sequence length
        num_classes: Number of output classes (3: negative, neutral, positive)
        dropout_rate: Dropout rate for dense layers
        lstm_dropout: Dropout rate for LSTM inputs
        recurrent_dropout: Dropout rate for LSTM recurrent connections
    
    Returns:
        Compiled Keras model
    """
    logger.info(f"Building Bidirectional LSTM model with vocab_size={vocab_size}, "
                f"embedding_dim={embedding_dim}, lstm_units={lstm_units}")
    
    model = Sequential([
        # Embedding Layer
        # Converts word indices to dense vectors of fixed size
        # Input: (batch_size, max_length)
        # Output: (batch_size, max_length, embedding_dim)
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            name='embedding_layer'
        ),
        
        # First Bidirectional LSTM Layer
        # Processes sequence in both forward and backward directions
        # return_sequences=True: Returns full sequence for next layer
        Bidirectional(
            LSTM(
                units=lstm_units,
                return_sequences=True,
                dropout=lstm_dropout,
                recurrent_dropout=recurrent_dropout,
                name='lstm_layer_1'
            ),
            name='bidirectional_lstm_1'
        ),
        
        # Dropout for regularization
        Dropout(rate=dropout_rate, name='dropout_1'),
        
        # Second Bidirectional LSTM Layer
        # return_sequences=False: Returns only final output
        Bidirectional(
            LSTM(
                units=lstm_units // 2,  # Half the units for second layer
                return_sequences=False,
                dropout=lstm_dropout,
                recurrent_dropout=recurrent_dropout,
                name='lstm_layer_2'
            ),
            name='bidirectional_lstm_2'
        ),
        
        # Dropout for regularization
        Dropout(rate=dropout_rate, name='dropout_2'),
        
        # Dense layer for feature transformation
        Dense(
            units=64,
            activation='relu',
            name='dense_layer'
        ),
        
        # Final dropout
        Dropout(rate=dropout_rate * 0.6, name='dropout_3'),
        
        # Output layer
        # Softmax activation for multi-class classification
        Dense(
            units=num_classes,
            activation='softmax',
            name='output_layer'
        )
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',  # For integer labels
        metrics=['accuracy']
    )
    
    logger.info("Model compiled successfully")
    
    # Print model summary
    model.summary(print_fn=logger.info)
    
    return model


def build_alternative_model(
    vocab_size: int,
    embedding_dim: int = 128,
    max_length: int = 100,
    num_classes: int = 3
) -> keras.Model:
    """
    Alternative model architecture with Global Max Pooling.
    Simpler architecture, faster training.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        max_length: Maximum sequence length
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    logger.info("Building alternative model with Global Max Pooling")
    
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ),
        
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        
        # Global Max Pooling: Takes maximum value across sequence
        GlobalMaxPooling1D(),
        
        Dense(32, activation='relu'),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_info(model: keras.Model) -> Dict:
    """
    Get information about the model.
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary with model information
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }

