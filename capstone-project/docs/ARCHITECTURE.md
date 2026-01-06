# System Architecture

## Overview

The Real Estate Price Prediction System follows a microservices architecture with clear separation of concerns:

```
┌─────────────┐
│  Frontend   │  Streamlit Web Application
│  (Streamlit)│
└──────┬──────┘
       │ HTTP/REST
┌──────▼──────┐
│   Backend   │  FastAPI REST API
│   (FastAPI) │
└──────┬──────┘
       │
┌──────▼──────┐
│ ML Pipeline │  Model Training & Inference
│  (MLflow)   │
└──────┬──────┘
       │
┌──────▼──────┐
│  Monitoring │  Prometheus + Grafana
└─────────────┘
```

## Components

### 1. Frontend (Streamlit)
- **Purpose**: User interface for predictions and analytics
- **Technology**: Streamlit, Plotly
- **Features**:
  - Property input form
  - Real-time price predictions
  - Analytics dashboard
  - Prediction history

### 2. Backend (FastAPI)
- **Purpose**: RESTful API for predictions
- **Technology**: FastAPI, Pydantic, Uvicorn
- **Endpoints**:
  - `/api/v1/predict` - Single prediction
  - `/api/v1/batch` - Batch predictions
  - `/api/v1/health` - Health check
  - `/api/v1/metrics` - Prometheus metrics
  - `/docs` - API documentation

### 3. ML Pipeline
- **Purpose**: Model training and preprocessing
- **Technology**: Scikit-learn, XGBoost, TensorFlow, MLflow
- **Components**:
  - Data enrichment
  - Feature engineering
  - Model training
  - Model evaluation
  - Model registry

### 4. Monitoring
- **Purpose**: System observability
- **Technology**: Prometheus, Grafana
- **Metrics**:
  - Prediction latency
  - Request count
  - Error rate
  - Model performance

## Data Flow

1. **Training Phase**:
   - Raw data → Enrichment → Preprocessing → Feature Engineering
   - Training data → Model Training → Evaluation → Model Registry

2. **Inference Phase**:
   - User input → API → Preprocessing → Model → Prediction → Response

## Deployment Architecture

### Development
- Local execution with Python
- Direct model loading

### Production (Docker)
- Containerized services
- Docker Compose orchestration
- Service discovery and networking

### Production (Kubernetes)
- Pod-based deployment
- Service mesh
- Auto-scaling
- Load balancing

## Security Considerations

- Input validation
- Rate limiting
- Error handling
- Logging and monitoring
- Data privacy compliance

