import os
import io
import sqlite3
from datetime import datetime
import pandas as pd

import streamlit as st

from src import config, database, etl_pipeline, reporter, monitor

st.set_page_config(page_title="Weather Pipeline UI", layout="wide")

def run_etl(use_mock=False):
    database.setup_database()
    results = {"success": [], "failed": []}

    if use_mock:
        # monkeypatch fetch to avoid showing key or hitting API
        from src import api_client
        def mock_fetch(api_key, city):
            return {
                "city": city,
                "timestamp": datetime.now(),
                "temperature": 20.0 + (hash(city) % 10),
                "humidity": 50 + (hash(city) % 40),
                "pressure": 1010,
                "wind_speed": 3.5,
                "condition": "clear sky",
            }
        api_client.fetch_weather = mock_fetch

    for city in config.CITIES:
        try:
            etl_pipeline.run_pipeline(config.OPENWEATHER_API_KEY, city)
            results["success"].append(city)
        except Exception as e:
            results["failed"].append((city, str(e)))
    return results

def get_db_summary():
    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.cursor()
    out = {}
    try:
        cur.execute("SELECT COUNT(*) FROM cities")
        out['cities'] = cur.fetchone()[0]
    except Exception:
        out['cities'] = None
    try:
        cur.execute("SELECT COUNT(*) FROM weather_data")
        out['weather'] = cur.fetchone()[0]
    except Exception:
        out['weather'] = None
    conn.close()
    return out

def tail_log(path, lines=50):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().splitlines()[-lines:]

def show_table(query, limit=100):
    conn = sqlite3.connect(config.DB_PATH)
    try:
        df = pd.read_sql_query(query + f" LIMIT {limit}", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

st.title("Weather Data Pipeline — Control Panel")

with st.sidebar:
    st.header("Actions")
    use_mock = st.checkbox("Use mock API (no external calls)", value=False)
    if st.button("Run ETL now"):
        with st.spinner("Running ETL..."):
            res = run_etl(use_mock=use_mock)
        st.success("ETL finished")
        st.write(res)

    if st.button("Generate report"):
        reporter.generate_daily_report()
        st.success("Report generated")
        st.markdown("[Download report](reports/daily_report.txt)")

    if st.button("Run Health Check"):
        st.json(monitor.health_check())

    if st.button("Run Tests"):
        import pytest
        st.text("Running pytest...")
        # run tests programmatically and capture output
        ret = pytest.main(["-q"])
        st.write(f"pytest exit code: {ret}")

    st.markdown("---")
    st.markdown("**Config**")
    st.write({
        'DB_PATH': config.DB_PATH,
        'Cities': config.CITIES,
        'Temp threshold': config.TEMP_THRESHOLD,
        'Humidity threshold': config.HUMIDITY_THRESHOLD,
        'API key present': bool(config.OPENWEATHER_API_KEY)
    })

st.header("Database Summary")
summary = get_db_summary()
st.metric("Cities rows", summary.get('cities'))
st.metric("Weather rows", summary.get('weather'))

st.header("Recent Weather Data")
df = show_table("SELECT * FROM weather_data ORDER BY record_id DESC", limit=50)
if not df.empty:
    st.dataframe(df)
else:
    st.info("No weather rows available")

st.header("Cities")
dfc = show_table("SELECT * FROM cities ORDER BY city_id DESC", limit=50)
if not dfc.empty:
    st.dataframe(dfc)
else:
    st.info("No cities available")

st.header("Pipeline Log")
logs = tail_log('logs/pipeline.log', lines=200)
if logs:
    st.text('\n'.join(logs[-200:]))
else:
    st.info('No pipeline log found')

st.markdown("---")
st.caption("Run this app with: `streamlit run streamlit_app.py`")
