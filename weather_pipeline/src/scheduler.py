import schedule
import time
from src.etl_pipeline import run_pipeline
from src.config import OPENWEATHER_API_KEY, CITIES
from src.database import setup_database

def job():
    for city in CITIES:
        run_pipeline(OPENWEATHER_API_KEY, city)

if __name__ == "__main__":
    setup_database()
    schedule.every(1).hour.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)
