# Deployment Guide

## Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- Git

## Local Deployment

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data

```bash
python ml-pipeline/data/enrich_data.py
```

### Step 3: Train Models

```bash
python ml-pipeline/training/train_models.py
```

### Step 4: Start Backend

```bash
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 5: Start Frontend

```bash
streamlit run frontend/app.py
```

## Quick Start Scripts

### Start All Services

```bash
python scripts/start_all.py
```

This starts both backend and frontend automatically.

### Start Services Separately

**Backend:**
```bash
python scripts/start_backend.py
```

**Frontend (in new terminal):**
```bash
python scripts/start_frontend.py
```

### Services

- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

## Environment Variables

### Backend

- `ENV`: Environment (development/production)
- `MODEL_PATH`: Path to model file
- `PREPROCESSOR_PATH`: Path to preprocessor file

### Frontend

- `API_URL`: Backend API URL

## Production Checklist

- [ ] Models trained and validated
- [ ] Environment variables configured
- [ ] Monitoring set up
- [ ] Logging configured
- [ ] Security measures in place
- [ ] Backup strategy defined
- [ ] Documentation updated

