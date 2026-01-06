"""
Unit tests for data preprocessing
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "ml-pipeline"))
from data.preprocessing import DataPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'area_sqft': [1200, 1500, 2000],
        'bedrooms': [2, 3, 4],
        'bathrooms': [2, 2, 3],
        'floor': [3, 5, 7],
        'total_floors': [10, 12, 15],
        'property_age': [5, 10, 15],
        'location': ['City Center', 'Suburb', 'Rural'],
        'city': ['Bangalore', 'Mumbai', 'Delhi'],
        'property_type': ['Apartment', 'Villa', 'House'],
        'facing': ['East', 'West', 'North'],
        'furnishing': ['Furnished', 'Semi-Furnished', 'Unfurnished'],
        'parking': [1, 2, 2],
        'amenities_score': [0.75, 0.65, 0.55],
        'distance_city_center_km': [5.0, 15.0, 25.0]
    })

@pytest.fixture
def sample_target():
    """Create sample target values"""
    return pd.Series([5000000, 7500000, 10000000])

def test_preprocessor_fit_transform(sample_data, sample_target):
    """Test preprocessor fit and transform"""
    preprocessor = DataPreprocessor()
    X_processed, y = preprocessor.fit_transform(sample_data, sample_target)
    
    assert X_processed.shape[0] == sample_data.shape[0]
    assert X_processed.shape[1] >= sample_data.shape[1]  # Should have more features after engineering

def test_preprocessor_transform(sample_data, sample_target):
    """Test preprocessor transform on new data"""
    preprocessor = DataPreprocessor()
    preprocessor.fit_transform(sample_data, sample_target)
    
    new_data = sample_data.copy()
    X_processed = preprocessor.transform(new_data)
    
    assert X_processed.shape[0] == new_data.shape[0]

def test_preprocessor_save_load(sample_data, sample_target, tmp_path):
    """Test saving and loading preprocessor"""
    preprocessor = DataPreprocessor()
    preprocessor.fit_transform(sample_data, sample_target)
    
    save_path = tmp_path / "preprocessor.pkl"
    preprocessor.save(str(save_path))
    
    loaded_preprocessor = DataPreprocessor.load(str(save_path))
    X_processed = loaded_preprocessor.transform(sample_data)
    
    assert X_processed.shape[0] == sample_data.shape[0]

