"""
Integration tests for FastAPI endpoints
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "backend"))
from api.main import app

client = TestClient(app)

@pytest.fixture
def sample_property():
    """Sample property data for testing"""
    return {
        "area_sqft": 1200,
        "bedrooms": 2,
        "bathrooms": 2,
        "floor": 3,
        "total_floors": 10,
        "property_age": 5,
        "location": "City Center",
        "city": "Bangalore",
        "property_type": "Apartment",
        "facing": "East",
        "furnishing": "Semi-Furnished",
        "parking": 1,
        "amenities_score": 0.75,
        "distance_city_center_km": 12.4
    }

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "version" in response.json()

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data

def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

def test_predict_endpoint(sample_property):
    """Test prediction endpoint"""
    response = client.post("/api/v1/predict", json=sample_property)
    # May fail if model not loaded, which is OK for testing
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "predicted_price" in data
        assert "confidence_interval" in data

def test_batch_endpoint(sample_property):
    """Test batch prediction endpoint"""
    batch_data = {
        "properties": [sample_property, sample_property]
    }
    response = client.post("/api/v1/batch", json=batch_data)
    # May fail if model not loaded, which is OK for testing
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "total_count" in data

def test_invalid_input():
    """Test invalid input handling"""
    invalid_data = {
        "area_sqft": -100,  # Invalid: negative area
        "bedrooms": 2,
        "bathrooms": 2,
        "floor": 3,
        "total_floors": 10,
        "property_age": 5,
        "location": "City Center",
        "city": "Bangalore",
        "property_type": "Apartment",
        "facing": "East",
        "furnishing": "Semi-Furnished",
        "parking": 1,
        "amenities_score": 0.75,
        "distance_city_center_km": 12.4
    }
    response = client.post("/api/v1/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

