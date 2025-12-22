# Weather Pipeline 🌦️

**Short description:** A lightweight ETL pipeline to fetch weather data from OpenWeather, validate and store it in a local SQLite database, generate alerts and simple reports, and provide a Streamlit UI.

---

## 🚀 Quick start

Prerequisites:
- Python 3.8+
- Git (optional)

Commands:

1. Clone the repo (if needed):

   ```bash
   git clone <repo-url>
   cd weather_pipeline
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenWeather API key:

   ```env
   OPENWEATHER_API_KEY=your_api_key_here
   ```

4. Initialize the local database:

   ```bash
   python database/init_db.py
   ```

5. Run the Streamlit app (UI):

   ```bash
   streamlit run streamlit_app.py
   ```

6. Run tests:

   ```bash
   pytest
   ```

---

## 🧭 Project structure

- `streamlit_app.py` — Streamlit frontend for viewing reports
- `debug_run.py` — simple runner used for quick debugging/test execution
- `database/` — DB helpers and initialization script
- `src/` — core modules:
  - `api_client.py` — fetch data from OpenWeather
  - `config.py` — configuration and constants (loads `.env`)
  - `database.py` — SQLite helper functions
  - `etl_pipeline.py` — main ETL logic and alert insertion
  - `monitor.py`, `reporter.py`, `scheduler.py`, `validators.py` — support
- `tests/` — unit tests (pytest)
- `logs/` — pipeline logs
- `reports/` — generated reports

---

## ⚙️ Environment variables

- `OPENWEATHER_API_KEY` (required) — API key for OpenWeather
- `DB_PATH` (optional) — path to SQLite DB (defaults to `database/weather_data.db`)

> Tip: store secrets in `.env` (this repo uses `python-dotenv`, see `src/config.py`).

---

## 🧪 Usage examples

Run the pipeline programmatically for a city (example):

```bash
python -c "import os; from src.etl_pipeline import run_pipeline; run_pipeline(os.getenv('OPENWEATHER_API_KEY'), 'Mumbai')"
```

Log files are written to `logs/pipeline.log` and a simple daily report may be found in `reports/daily_report.txt`.

---

## ✅ Testing

- Run tests with `pytest`.
- API tests will be skipped if `OPENWEATHER_API_KEY` is not set (see `tests/test_api_client.py`).

---

## 🤝 Contributing

Contributions welcome — please open issues or PRs. Keep changes small and add/adjust tests when behavior changes.

---

## 📄 License

This project is available under the MIT License. See `LICENSE` (add one if you want to include a license file).

---

If you'd like, I can add a `CONTRIBUTING.md`, a `LICENSE`, or badges to the README. 🔧
