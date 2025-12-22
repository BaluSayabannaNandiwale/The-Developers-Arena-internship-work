from src.api_client import fetch_weather
import os

def test_api_call():
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        # skip live API test when no key provided
        return
    data = fetch_weather(api_key, "Mumbai")
    assert "temperature" in data
    assert "humidity" in data
