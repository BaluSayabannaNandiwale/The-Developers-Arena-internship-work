from src.validators import validate_weather
from datetime import datetime

def test_valid_weather():
    data = {
        "temperature": 25,
        "humidity": 60,
        "pressure": 1012,
        "timestamp": datetime.now()
    }
    assert validate_weather(data) is True

def test_invalid_temperature():
    data = {
        "temperature": 100,
        "humidity": 60,
        "pressure": 1012
    }
    assert validate_weather(data) is False
