"""
Test suite for the Sentiment Analysis API.
Tests all endpoints and error handling.
"""

import pytest
import requests
import json
from fastapi.testclient import TestClient

from src.api.app import app
from src.inference.predict import SentimentPredictor
from src.monitoring.metrics import MetricsCollector

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check(self):
        """Test health check endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"
    
    def test_health_check_structure(self):
        """Test health check response structure."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["version"], str)


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        metrics = data["metrics"]
        assert "requests" in metrics
        assert "latency" in metrics
        assert "predictions" in metrics
    
    def test_metrics_structure(self):
        """Test metrics response structure."""
        response = client.get("/metrics")
        data = response.json()
        metrics = data["metrics"]
        
        # Check requests structure
        assert "total" in metrics["requests"]
        assert "successful" in metrics["requests"]
        assert "failed" in metrics["requests"]
        
        # Check latency structure
        assert "average_ms" in metrics["latency"]
        assert "current_ms" in metrics["latency"]


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""
    
    def test_predict_endpoint_structure(self):
        """Test /predict endpoint request/response structure."""
        # This will fail if model is not loaded, which is expected
        response = client.post(
            "/predict",
            json={"text": "This is a great product!"}
        )
        
        # Should return either 200 (if model loaded) or 503 (if not)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "text" in data
            assert "sentiment" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert data["sentiment"] in ["positive", "negative", "neutral"]
            assert 0 <= data["confidence"] <= 1
    
    def test_predict_empty_text(self):
        """Test /predict with empty text should return 422."""
        response = client.post(
            "/predict",
            json={"text": ""}
        )
        assert response.status_code == 422
    
    def test_predict_missing_text(self):
        """Test /predict with missing text field."""
        response = client.post(
            "/predict",
            json={}
        )
        assert response.status_code == 422
    
    def test_batch_predict_endpoint_structure(self):
        """Test /batch_predict endpoint structure."""
        response = client.post(
            "/batch_predict",
            json={"texts": ["Great!", "Bad product", "It's okay"]}
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total" in data
            assert isinstance(data["predictions"], list)
            assert data["total"] == len(data["predictions"])
    
    def test_batch_predict_empty_list(self):
        """Test /batch_predict with empty list."""
        response = client.post(
            "/batch_predict",
            json={"texts": []}
        )
        assert response.status_code == 422
    
    def test_batch_predict_too_many_texts(self):
        """Test /batch_predict with too many texts."""
        texts = ["text"] * 101
        response = client.post(
            "/batch_predict",
            json={"texts": texts}
        )
        assert response.status_code == 422


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""
    
    def test_model_info_endpoint(self):
        """Test /model/info endpoint."""
        response = client.get("/model/info")
        
        # Should return either 200 (if model loaded) or 503 (if not)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "input_shape" in data
            assert "output_shape" in data
            assert "total_parameters" in data


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_invalid_endpoint(self):
        """Test 404 for invalid endpoint."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404


class TestMetricsCollector:
    """Tests for metrics collector."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector()
        assert collector.total_requests == 0
        assert collector.successful_requests == 0
        assert collector.failed_requests == 0
    
    def test_record_request(self):
        """Test recording requests."""
        collector = MetricsCollector()
        collector.record_request("/predict", 0.1, success=True)
        assert collector.total_requests == 1
        assert collector.successful_requests == 1
        assert collector.failed_requests == 0
    
    def test_record_error(self):
        """Test recording errors."""
        collector = MetricsCollector()
        collector.record_request("/predict", 0.1, success=False, error_type="ValueError")
        assert collector.failed_requests == 1
        assert collector.error_count["ValueError"] == 1
    
    def test_get_metrics(self):
        """Test getting metrics."""
        collector = MetricsCollector()
        collector.record_request("/predict", 0.1, success=True)
        metrics = collector.get_metrics()
        assert "requests" in metrics
        assert "latency" in metrics
        assert metrics["requests"]["total"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

