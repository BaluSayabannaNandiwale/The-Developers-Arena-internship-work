"""
Text preprocessing pipeline for sentiment analysis.
Handles text cleaning, tokenization, padding, and label encoding.
"""

import re
import string
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessing class for sentiment analysis.
    Handles cleaning, tokenization, and sequence preparation.
    """
    
    def __init__(self, max_words: int = 10000, max_length: int = 100, oov_token: str = '<OOV>'):
        """
        Initialize the preprocessor.
        
        Args:
            max_words: Maximum vocabulary size
            max_length: Maximum sequence length for padding/truncation
            oov_token: Token for out-of-vocabulary words
        """
        self.max_words = max_words
        self.max_length = max_length
        self.oov_token = oov_token
        self.tokenizer = None
        self.label_encoder = None
        
        logger.info(f"Initialized TextPreprocessor with max_words={max_words}, max_length={max_length}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove email addresses
        4. Remove special characters (keep basic punctuation)
        5. Remove extra whitespace
        
        Args:
            text: Raw text string
        
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove user mentions and hashtags (for social media data)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove numbers (optional - can be kept for some use cases)
        text = re.sub(r'\d+', '', text)
        
        # Remove special characters but keep basic punctuation
        # Keep: . , ! ? - ' "
        text = re.sub(r'[^a-zA-Z\s.,!?\'"-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_texts(self, texts: List[str]) -> List[str]:
        """
        Clean a list of text strings.
        
        Args:
            texts: List of raw text strings
        
        Returns:
            List of cleaned text strings
        """
        logger.info(f"Cleaning {len(texts)} text samples")
        cleaned_texts = [self.clean_text(text) for text in texts]
        logger.info("Text cleaning completed")
        return cleaned_texts
    
    def fit_tokenizer(self, texts: List[str]) -> None:
        """
        Fit the tokenizer on training texts.
        Creates vocabulary from the most frequent words.
        
        Args:
            texts: List of text strings to fit on
        """
        logger.info(f"Fitting tokenizer on {len(texts)} texts")
        
        self.tokenizer = Tokenizer(
            num_words=self.max_words,
            oov_token=self.oov_token,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        
        self.tokenizer.fit_on_texts(texts)
        
        vocab_size = len(self.tokenizer.word_index) + 1  # +1 for padding token
        logger.info(f"Tokenizer fitted. Vocabulary size: {vocab_size}")
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert texts to sequences of integers.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of sequences (list of lists)
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call fit_tokenizer() first.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences
    
    def pad_sequences(self, sequences: List[List[int]]) -> np.ndarray:
        """
        Pad or truncate sequences to fixed length.
        
        Args:
            sequences: List of sequences (list of lists)
        
        Returns:
            Padded sequences array
        """
        padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )
        return padded
    
    def preprocess_texts(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        Complete preprocessing pipeline: clean -> tokenize -> pad.
        
        Args:
            texts: List of raw text strings
            fit: Whether to fit tokenizer (True for training data)
        
        Returns:
            Preprocessed sequences array
        """
        # Step 1: Clean texts
        cleaned_texts = self.clean_texts(texts)
        
        # Step 2: Fit tokenizer if needed
        if fit:
            self.fit_tokenizer(cleaned_texts)
        
        # Step 3: Convert to sequences
        sequences = self.texts_to_sequences(cleaned_texts)
        
        # Step 4: Pad sequences
        padded_sequences = self.pad_sequences(sequences)
        
        logger.info(f"Preprocessed {len(texts)} texts. Shape: {padded_sequences.shape}")
        return padded_sequences
    
    def encode_labels(self, labels: List[str], fit: bool = False) -> np.ndarray:
        """
        Encode string labels to integers.
        
        Args:
            labels: List of label strings (e.g., ['positive', 'negative', 'neutral'])
            fit: Whether to fit label encoder (True for training data)
        
        Returns:
            Encoded labels array
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        
        if fit:
            encoded = self.label_encoder.fit_transform(labels)
            logger.info(f"Label encoder fitted. Classes: {self.label_encoder.classes_}")
        else:
            encoded = self.label_encoder.transform(labels)
        
        return np.array(encoded)
    
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """
        Decode integer labels back to strings.
        
        Args:
            encoded_labels: Array of encoded labels
        
        Returns:
            List of label strings
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted.")
        
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size.
        
        Returns:
            Vocabulary size
        """
        if self.tokenizer is None:
            return 0
        return len(self.tokenizer.word_index) + 1  # +1 for padding token
    
    def save_tokenizer(self, filepath: str) -> None:
        """
        Save tokenizer to file (using pickle).
        
        Args:
            filepath: Path to save tokenizer
        """
        import pickle
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        logger.info(f"Tokenizer saved to {filepath}")
    
    def load_tokenizer(self, filepath: str) -> None:
        """
        Load tokenizer from file.
        
        Args:
            filepath: Path to load tokenizer from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        logger.info(f"Tokenizer loaded from {filepath}")


def load_and_prepare_data(
    data_path: str,
    text_column: str = 'text',
    label_column: str = 'label',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TextPreprocessor]:
    """
    Load data from CSV and prepare train/test splits.
    
    Args:
        data_path: Path to CSV file
        text_column: Name of text column
        label_column: Name of label column
        test_size: Proportion of test data
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Extract texts and labels
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess texts (fit on all data first)
    X = preprocessor.preprocess_texts(texts, fit=True)
    
    # Encode labels
    y = preprocessor.encode_labels(labels, fit=True)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, preprocessor

