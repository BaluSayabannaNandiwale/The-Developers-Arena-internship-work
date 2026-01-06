# Real Estate Price Prediction System

A production-ready machine learning system for predicting real estate prices in INR, featuring end-to-end ML lifecycle, full-stack deployment, monitoring, and comprehensive documentation.

## 🎯 Project Overview

This capstone project implements a complete ML system that:
- Predicts property prices using multiple ML models (Linear Regression, Random Forest, XGBoost, Neural Network)
- Provides RESTful API for real-time and batch predictions
- Features a modern Streamlit frontend with analytics dashboard
- Includes comprehensive monitoring with Prometheus and Grafana
- Supports containerized deployment with Docker
- Implements CI/CD pipeline with GitHub Actions

## 📁 Project Structure

```
capstone-project/
├── backend/              # FastAPI backend application
│   ├── api/             # API endpoints
│   ├── models/          # ML models and pipelines
│   ├── services/        # Business logic
│   └── database/        # Database models
├── frontend/            # Streamlit frontend
│   └── app.py          # Main Streamlit application
├── ml-pipeline/        # ML training pipeline
│   ├── data/           # Data processing and enrichment
│   ├── training/       # Model training scripts
│   └── evaluation/     # Model evaluation
├── monitoring/         # Monitoring setup
│   ├── prometheus/     # Prometheus configuration
│   └── grafana/        # Grafana dashboards
├── infrastructure/     # Deployment files
│   ├── docker/         # Docker configurations
│   └── kubernetes/     # K8s manifests
├── tests/              # Test suite
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   └── performance/   # Performance tests
├── docs/              # Documentation
├── data/              # Datasets
├── models/            # Trained models
├── docker-compose.yml # Docker Compose configuration
└── requirements.txt   # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (optional)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd capstone-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Enrich the dataset**
   ```bash
   python ml-pipeline/data/enrich_data.py
   ```

4. **Train the models**
   ```bash
   python ml-pipeline/training/train_models.py
   ```

5. **Start the system** (easiest way)
   ```bash
   python run.py
   ```
   
   This starts both backend and frontend automatically!

   **OR start separately:**
   
   Terminal 1 - Backend:
   ```bash
   python scripts/start_backend.py
   ```
   
   Terminal 2 - Frontend:
   ```bash
   python scripts/start_frontend.py
   ```

### Using Startup Scripts

**Option 1: Start both services together**
```bash
python scripts/start_all.py
```

**Option 2: Start separately**

Terminal 1 - Backend:
```bash
python scripts/start_backend.py
```

Terminal 2 - Frontend:
```bash
python scripts/start_frontend.py
```

This will start:
- Backend API at http://localhost:8000
- Frontend at http://localhost:8501
- API Documentation at http://localhost:8000/docs

## 📊 Data Schema

### Input Features

| Field | Type | Description |
|-------|------|-------------|
| area_sqft | float | Built-up area in square feet |
| bedrooms | int | Number of bedrooms |
| bathrooms | int | Number of bathrooms |
| floor | int | Floor number |
| total_floors | int | Total floors in building |
| property_age | int | Property age in years |
| location | categorical | Locality / city zone |
| city | categorical | City name |
| property_type | categorical | Apartment / Villa / Independent |
| facing | categorical | East / West / North / South |
| furnishing | categorical | Furnished / Semi-Furnished / Unfurnished |
| parking | int | Parking slots |
| amenities_score | float | Amenities score (0-1) |
| distance_city_center_km | float | Distance from city center in km |

### Target Variable

| Field | Type | Description |
|-------|------|-------------|
| price_inr | float | Property price in INR |

## 🤖 ML Models

The system trains and compares multiple models:

1. **Linear Regression** - Baseline model
2. **Random Forest** - Ensemble method with 200 trees
3. **XGBoost** - Gradient boosting with optimized hyperparameters
4. **Neural Network** - Deep learning model with TensorFlow

### Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

All experiments are tracked using MLflow.

## 🌐 API Endpoints

### Single Prediction
```bash
POST /api/v1/predict
Content-Type: application/json

{
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
}
```

### Batch Prediction
```bash
POST /api/v1/batch
Content-Type: application/json

{
  "properties": [
    { ... property 1 ... },
    { ... property 2 ... }
  ]
}
```

### Health Check
```bash
GET /api/v1/health
```

### Metrics
```bash
GET /api/v1/metrics
```

### API Documentation
```bash
GET /docs
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v --cov=backend --cov=ml-pipeline --cov-report=html
```

Test coverage target: ≥80%

## 📈 Monitoring

### Prometheus Metrics

- `predictions_total` - Total predictions made
- `prediction_latency_seconds` - Prediction latency histogram
- `prediction_errors_total` - Total prediction errors

### Grafana Dashboards

Access Grafana at http://localhost:3000 (admin/admin) to view:
- Prediction rate
- Latency percentiles
- Error rates
- System health

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Architecture Diagram** - System architecture overview
- **API Documentation** - Complete API reference
- **Deployment Guide** - Step-by-step deployment instructions
- **ML Design Decisions** - Model selection and rationale
- **Monitoring Guide** - Monitoring setup and usage
- **Business Impact Analysis** - ROI and business value

## 🔧 Configuration

### Environment Variables

- `API_URL` - Backend API URL (default: http://localhost:8000)
- `ENV` - Environment (development/production)

### Model Configuration

Models are stored in the `models/` directory:
- `production_model.pkl` - Production model
- `preprocessor.pkl` - Data preprocessor

## 🚢 Deployment

### Docker Deployment

```bash
docker-compose up -d
```

### Kubernetes Deployment

Kubernetes manifests are available in `infrastructure/kubernetes/`.

## 📊 Performance Benchmarks

- **Average Latency**: <200ms
- **Throughput**: 100+ requests/second
- **Model Accuracy**: R² > 0.85
- **Uptime**: 99.9%+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

Built as a capstone project for portfolio and interview preparation.

## 🙏 Acknowledgments

- MLflow for experiment tracking
- FastAPI for the API framework
- Streamlit for the frontend
- Prometheus and Grafana for monitoring

