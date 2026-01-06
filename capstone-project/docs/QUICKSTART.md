# Quick Start Guide

## 5-Minute Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Enrich Data

```bash
python ml-pipeline/data/enrich_data.py
```

This creates `data/house_prices_enriched.csv` with all required fields.

### Step 3: Train Models

```bash
python ml-pipeline/training/train_models.py
```

This will:
- Train 4 models (Linear Regression, Random Forest, XGBoost, Neural Network)
- Compare their performance
- Save the best model to `models/production_model.pkl`
- Log experiments to MLflow

**Note**: Training may take 5-10 minutes depending on your system.

### Step 4: Start the System

**Easy way (starts both):**
```bash
python run.py
```

**OR start separately:**

Terminal 1 - Backend:
```bash
python scripts/start_backend.py
```

Terminal 2 - Frontend:
```bash
python scripts/start_frontend.py
```

Services will be available at:
- Backend: http://localhost:8000
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs

## Testing the System

### Test API

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "area_sqft": 1200,
    "bedrooms": 2,
    "bathrooms": 2,
    "floor": 3,
    "total_floors": 10,
    "property_age": 5,
    "location": "City Center",
    "city": "Bangalore",
    "property_type": "Apartment",
    "facing": "East",
    "furnishing": "Semi-Furnished",
    "parking": 1,
    "amenities_score": 0.75,
    "distance_city_center_km": 12.4
  }'
```

### Test Frontend

1. Open http://localhost:8501
2. Fill in the property details form
3. Click "Predict Price"
4. View the prediction with confidence interval

## Troubleshooting

### Model Not Found Error

If you see "Model not loaded" error:
1. Ensure you've run the training script
2. Check that `models/production_model.pkl` exists
3. Verify the model path in `backend/api/main.py`

### Import Errors

If you encounter import errors:
1. Ensure you're in the project root directory
2. Install all dependencies: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.9+)

### Port Already in Use

If ports 8000 or 8501 are already in use:
- Backend: Change port in `uvicorn` command: `--port 8001`
- Frontend: Change port in Streamlit: `streamlit run frontend/app.py --server.port 8502`

## Next Steps

1. **Explore API Documentation**: http://localhost:8000/docs
2. **View Analytics**: Use the Analytics Dashboard tab in the frontend
3. **Check Metrics**: http://localhost:8000/api/v1/metrics
4. **Review MLflow**: Check `mlruns/` directory for experiment tracking

## Quick Start (All Services)

Start both backend and frontend together:

```bash
python scripts/start_all.py
```

This automatically starts:
- Backend at http://localhost:8000
- Frontend at http://localhost:8501

