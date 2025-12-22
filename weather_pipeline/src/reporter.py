import sqlite3
from src.config import DB_PATH

def generate_daily_report():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    SELECT c.city_name, AVG(w.temperature_c), AVG(w.humidity)
    FROM weather_data w
    JOIN cities c ON w.city_id = c.city_id
    GROUP BY c.city_name
    """)

    rows = cur.fetchall()
    conn.close()

    # ensure reports directory exists
    import os
    os.makedirs('reports', exist_ok=True)

    with open("reports/daily_report.txt", "w", encoding='utf-8') as f:
        f.write("DAILY WEATHER REPORT\n")
        f.write("====================\n\n")
        for r in rows:
            temp = r[1] if r[1] is not None else 0.0
            hum = r[2] if r[2] is not None else 0.0
            f.write(f"{r[0]} → Avg Temp: {temp:.2f}°C | Avg Humidity: {hum:.2f}%\n")
