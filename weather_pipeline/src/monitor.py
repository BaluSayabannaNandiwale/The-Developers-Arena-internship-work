import os
import sqlite3
from src.config import DB_PATH

def health_check():
    status = {}

    status["database_exists"] = os.path.exists(DB_PATH)

    if status["database_exists"]:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM weather_data")
            status["records"] = cur.fetchone()[0]
        except sqlite3.OperationalError:
            status["records"] = 0
        finally:
            conn.close()
    else:
        status["records"] = 0

    return status

if __name__ == "__main__":
    print("SYSTEM HEALTH CHECK")
    print(health_check())
