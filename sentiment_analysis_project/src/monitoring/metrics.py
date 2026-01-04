"""
Monitoring and metrics tracking for the sentiment analysis API.
Tracks request count, latency, error rates, and model performance.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict, deque
from threading import Lock
import json
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MetricsCollector:
    """
    Thread-safe metrics collector for API monitoring.
    Tracks requests, latency, errors, and predictions.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of records to keep in history
        """
        self.max_history = max_history
        self.lock = Lock()
        
        # Request metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Latency tracking (using deque for efficient sliding window)
        self.latency_history = deque(maxlen=max_history)
        self.current_latency = 0.0
        
        # Error tracking
        self.error_count = defaultdict(int)
        self.error_history = deque(maxlen=100)
        
        # Prediction metrics
        self.prediction_counts = defaultdict(int)
        self.confidence_scores = deque(maxlen=max_history)
        
        # Timestamps
        self.start_time = datetime.now()
        self.last_request_time = None
        
        logger.info("MetricsCollector initialized")
    
    def record_request(
        self,
        endpoint: str,
        latency: float,
        success: bool = True,
        error_type: Optional[str] = None
    ) -> None:
        """
        Record a request with its metrics.
        
        Args:
            endpoint: API endpoint name
            latency: Request latency in seconds
            success: Whether request was successful
            error_type: Type of error if request failed
        """
        with self.lock:
            self.total_requests += 1
            self.last_request_time = datetime.now()
            
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
                if error_type:
                    self.error_count[error_type] += 1
                    self.error_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'endpoint': endpoint,
                        'error_type': error_type
                    })
            
            # Record latency
            self.latency_history.append(latency)
            self.current_latency = latency
            
            logger.debug(f"Recorded request: {endpoint}, latency: {latency:.3f}s, success: {success}")
    
    def record_prediction(
        self,
        sentiment: str,
        confidence: float
    ) -> None:
        """
        Record a prediction result.
        
        Args:
            sentiment: Predicted sentiment class
            confidence: Prediction confidence score
        """
        with self.lock:
            self.prediction_counts[sentiment] += 1
            self.confidence_scores.append(confidence)
    
    def get_metrics(self) -> Dict:
        """
        Get current metrics summary.
        
        Returns:
            Dictionary with all current metrics
        """
        with self.lock:
            # Calculate average latency
            avg_latency = (
                sum(self.latency_history) / len(self.latency_history)
                if self.latency_history else 0.0
            )
            
            # Calculate min/max latency
            min_latency = min(self.latency_history) if self.latency_history else 0.0
            max_latency = max(self.latency_history) if self.latency_history else 0.0
            
            # Calculate error rate
            error_rate = (
                (self.failed_requests / self.total_requests * 100)
                if self.total_requests > 0 else 0.0
            )
            
            # Calculate average confidence
            avg_confidence = (
                sum(self.confidence_scores) / len(self.confidence_scores)
                if self.confidence_scores else 0.0
            )
            
            # Calculate uptime
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600
            uptime_days = uptime_hours / 24
            
            metrics = {
                'requests': {
                    'total': self.total_requests,
                    'successful': self.successful_requests,
                    'failed': self.failed_requests,
                    'error_rate_percent': round(error_rate, 2)
                },
                'latency': {
                    'current_ms': round(self.current_latency * 1000, 2),
                    'average_ms': round(avg_latency * 1000, 2),
                    'min_ms': round(min_latency * 1000, 2),
                    'max_ms': round(max_latency * 1000, 2)
                },
                'predictions': {
                    'total': sum(self.prediction_counts.values()),
                    'by_sentiment': dict(self.prediction_counts),
                    'average_confidence': round(avg_confidence, 3)
                },
                'errors': {
                    'by_type': dict(self.error_count),
                    'recent_count': len(self.error_history)
                },
                'uptime': {
                    'seconds': int(uptime_seconds),
                    'hours': round(uptime_hours, 2),
                    'days': round(uptime_days, 2),
                    'started_at': self.start_time.isoformat(),
                    'last_request': self.last_request_time.isoformat() if self.last_request_time else None
                }
            }
            
            return metrics
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict]:
        """
        Get recent error records.
        
        Args:
            limit: Maximum number of errors to return
        
        Returns:
            List of recent error records
        """
        with self.lock:
            return list(self.error_history)[-limit:]
    
    def reset_metrics(self) -> None:
        """
        Reset all metrics (use with caution).
        """
        with self.lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.latency_history.clear()
            self.error_count.clear()
            self.error_history.clear()
            self.prediction_counts.clear()
            self.confidence_scores.clear()
            self.start_time = datetime.now()
            self.last_request_time = None
        
        logger.info("Metrics reset")
    
    def save_metrics(self, filepath: str) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            filepath: Path to save metrics
        """
        metrics = self.get_metrics()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get or create the global metrics collector instance.
    
    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


class RequestTimer:
    """
    Context manager for timing API requests.
    """
    
    def __init__(self, metrics_collector: MetricsCollector, endpoint: str):
        """
        Initialize request timer.
        
        Args:
            metrics_collector: MetricsCollector instance
            endpoint: Endpoint name
        """
        self.metrics_collector = metrics_collector
        self.endpoint = endpoint
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time
        success = exc_type is None
        error_type = exc_type.__name__ if exc_type else None
        
        self.metrics_collector.record_request(
            endpoint=self.endpoint,
            latency=latency,
            success=success,
            error_type=error_type
        )

