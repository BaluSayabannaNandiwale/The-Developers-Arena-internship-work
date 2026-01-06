"""
Data Preprocessing Pipeline
Handles missing values, encoding, scaling, and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
from typing import Dict, List, Tuple

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering transformer"""
    
    def __init__(self):
        self.price_per_sqft_mean = None
        
    def fit(self, X, y=None):
        if y is not None:
            self.price_per_sqft_mean = (y / X['area_sqft']).mean()
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        # Price per sqft (if target available)
        if y is not None:
            X['price_per_sqft'] = y / X['area_sqft']
        elif self.price_per_sqft_mean is not None:
            # Use mean for inference
            X['price_per_sqft'] = self.price_per_sqft_mean
        
        # Age buckets
        X['age_bucket'] = pd.cut(
            X['property_age'],
            bins=[0, 5, 10, 20, 50, 100],
            labels=['New', 'Recent', 'Moderate', 'Old', 'Very Old']
        ).astype(str)
        
        # Location popularity score (based on frequency)
        if hasattr(self, 'location_scores'):
            X['location_popularity_score'] = X['location'].map(self.location_scores).fillna(0.5)
        else:
            # Calculate during fit
            location_counts = X['location'].value_counts()
            max_count = location_counts.max()
            self.location_scores = (location_counts / max_count).to_dict()
            X['location_popularity_score'] = X['location'].map(self.location_scores).fillna(0.5)
        
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features"""
    
    def __init__(self):
        self.encoders = {}
        self.categorical_cols = ['location', 'city', 'property_type', 'facing', 'furnishing']
        
    def fit(self, X, y=None):
        # Get all categorical columns that exist in X
        available_cats = [col for col in self.categorical_cols if col in X.columns]
        # Also check for age_bucket if it exists
        if 'age_bucket' in X.columns:
            available_cats.append('age_bucket')
        
        for col in available_cats:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        for col in self.encoders.keys():
            if col in X.columns:
                # Handle unseen values
                try:
                    X[col] = self.encoders[col].transform(X[col].astype(str))
                except ValueError:
                    # If unseen value, use most common class
                    X[col] = 0
        return X

class DataPreprocessor:
    """Complete data preprocessing pipeline"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.categorical_encoder = CategoricalEncoder()
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.numerical_cols = [
            'area_sqft', 'bedrooms', 'bathrooms', 'floor', 'total_floors',
            'property_age', 'parking', 'amenities_score', 'distance_city_center_km'
        ]
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit and transform training data"""
        X = X.copy()
        
        # Feature engineering
        X = self.feature_engineer.fit_transform(X, y)
        
        # Get numerical columns that exist
        available_numerical = [col for col in self.numerical_cols if col in X.columns]
        
        # Handle missing values in numerical columns
        if available_numerical:
            X[available_numerical] = self.numerical_imputer.fit_transform(X[available_numerical])
        
        # Encode categoricals
        X = self.categorical_encoder.fit_transform(X)
        
        # Scale numerical features
        if available_numerical:
            X[available_numerical] = self.scaler.fit_transform(X[available_numerical])
        
        return X, y
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data"""
        X = X.copy()
        
        # Feature engineering
        X = self.feature_engineer.transform(X)
        
        # Get numerical columns that exist
        available_numerical = [col for col in self.numerical_cols if col in X.columns]
        
        # Handle missing values
        if available_numerical:
            X[available_numerical] = self.numerical_imputer.transform(X[available_numerical])
        
        # Encode categoricals
        X = self.categorical_encoder.transform(X)
        
        # Scale numerical features
        if available_numerical:
            X[available_numerical] = self.scaler.transform(X[available_numerical])
        
        return X
    
    def save(self, path: str):
        """Save preprocessor"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str):
        """Load preprocessor"""
        with open(path, 'rb') as f:
            return pickle.load(f)

