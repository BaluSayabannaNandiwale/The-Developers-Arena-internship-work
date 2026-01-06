# Setup Instructions

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

## Step-by-Step Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

The dataset needs to be enriched with additional fields:

```bash
python ml-pipeline/data/enrich_data.py
```

This creates `data/house_prices_enriched.csv` with all required fields.

### 3. Train Models

Train all ML models:

```bash
python ml-pipeline/training/train_models.py
```

This will:
- Train 4 different models
- Compare their performance
- Save the best model to `models/production_model.pkl`
- Log experiments to MLflow

**Note**: Training takes 5-10 minutes depending on your system.

### 4. Start the Application

**Option 1: Start everything together (Recommended)**
```bash
python run.py
```

**Option 2: Start services separately**

Terminal 1 - Backend:
```bash
python scripts/start_backend.py
```

Terminal 2 - Frontend:
```bash
python scripts/start_frontend.py
```

### 5. Access the Application

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

## Verification

### Test Backend

```bash
curl http://localhost:8000/api/v1/health
```

Should return:
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "preprocessor_status": "loaded",
  "timestamp": "..."
}
```

### Test Frontend

1. Open http://localhost:8501 in your browser
2. Fill in the property details form
3. Click "Predict Price"
4. View the prediction result

## Troubleshooting

### Model Not Found

If you see "Model not loaded" error:
1. Ensure you've completed step 3 (train models)
2. Check that `models/production_model.pkl` exists
3. Verify the model path in `backend/api/main.py`

### Port Already in Use

If port 8000 or 8501 is already in use:

**Backend**: Edit `scripts/start_backend.py` and change port
**Frontend**: Edit `scripts/start_frontend.py` and change port

### Import Errors

1. Make sure you're in the project root directory
2. Install all dependencies: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.9+)

## Next Steps

- Read the [Quick Start Guide](docs/QUICKSTART.md)
- Explore the [API Documentation](http://localhost:8000/docs)
- Check the [Architecture Documentation](docs/ARCHITECTURE.md)
- Review [ML Design Decisions](docs/ML_DESIGN.md)

