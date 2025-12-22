import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
# Use environment variable; keep secrets out of source code.
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Database
DB_PATH = "database/weather_data.db"

# Cities to track
CITIES = [
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Chennai",
    "Kolkata"
]

# Alert thresholds
TEMP_THRESHOLD = 30.0
HUMIDITY_THRESHOLD = 75
