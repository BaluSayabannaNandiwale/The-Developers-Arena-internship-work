# Scripts Directory

This directory contains helper scripts for running the Real Estate Price Prediction System.

## Available Scripts

### `start_backend.py`
Starts the FastAPI backend server on port 8000.

```bash
python scripts/start_backend.py
```

### `start_frontend.py`
Starts the Streamlit frontend on port 8501.

```bash
python scripts/start_frontend.py
```

### `start_all.py`
Starts both backend and frontend together.

```bash
python scripts/start_all.py
```

### `train_models.sh` / `train_models.py`
Trains all ML models. Run this after enriching the data.

```bash
python ml-pipeline/training/train_models.py
```

### `run_tests.sh`
Runs the test suite with coverage reporting.

```bash
pytest tests/ -v --cov
```

## Quick Start

The easiest way to start the system is from the project root:

```bash
python run.py
```

This automatically starts both backend and frontend services.

