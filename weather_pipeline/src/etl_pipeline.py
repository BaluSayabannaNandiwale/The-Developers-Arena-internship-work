import logging
from src.api_client import fetch_weather
from src.validators import validate_weather
from src.database import insert_weather, insert_alert, get_or_create_city
from src.config import TEMP_THRESHOLD, HUMIDITY_THRESHOLD

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_pipeline(api_key, city):
    try:
        data = fetch_weather(api_key, city)

        if validate_weather(data):
            insert_weather(data)
            logging.info(f"Inserted weather data for {city}")

            city_id = get_or_create_city(city)

            if data["temperature"] > TEMP_THRESHOLD:
                insert_alert(city_id, "HIGH_TEMPERATURE", data["temperature"])

            if data["humidity"] > HUMIDITY_THRESHOLD:
                insert_alert(city_id, "HIGH_HUMIDITY", data["humidity"])
        else:
            logging.warning(f"Validation failed for {city}")

    except Exception as e:
        logging.error(f"Pipeline failed for {city}: {e}")
