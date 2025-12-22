import requests
from datetime import datetime

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def fetch_weather(api_key, city):
    if not api_key:
        raise ValueError("OPENWEATHER API key is required")

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    return {
        "city": city,
        "timestamp": datetime.now(),
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data.get("wind", {}).get("speed", 0),
        "condition": data["weather"][0]["description"]
    }
