import sqlite3
from src.config import DB_PATH

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def setup_database():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cities (
        city_id INTEGER PRIMARY KEY AUTOINCREMENT,
        city_name TEXT UNIQUE,
        country TEXT,
        latitude REAL,
        longitude REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS weather_data (
        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        city_id INTEGER,
        timestamp TIMESTAMP,
        temperature_c REAL,
        humidity INTEGER,
        pressure_hpa REAL,
        wind_speed_mps REAL,
        weather_condition TEXT,
        FOREIGN KEY (city_id) REFERENCES cities(city_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
        city_id INTEGER,
        alert_type TEXT,
        alert_value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (city_id) REFERENCES cities(city_id)
    )
    """)

    conn.commit()
    conn.close()

def get_or_create_city(city):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT city_id FROM cities WHERE city_name=?", (city,))
    row = cur.fetchone()

    if row:
        city_id = row["city_id"]
    else:
        cur.execute("INSERT INTO cities (city_name) VALUES (?)", (city,))
        conn.commit()
        city_id = cur.lastrowid

    conn.close()
    return city_id

def insert_weather(data):
    city_id = get_or_create_city(data["city"])

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO weather_data 
    (city_id, timestamp, temperature_c, humidity, pressure_hpa, wind_speed_mps, weather_condition)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        city_id,
        data["timestamp"],
        data["temperature"],
        data["humidity"],
        data["pressure"],
        data["wind_speed"],
        data["condition"]
    ))

    conn.commit()
    conn.close()

def insert_alert(city_id, alert_type, value):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO alerts (city_id, alert_type, alert_value)
    VALUES (?, ?, ?)
    """, (city_id, alert_type, value))

    conn.commit()
    conn.close()
