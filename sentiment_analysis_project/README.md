# Sentiment Analysis System - Production-Ready NLP Project

A complete, production-ready Natural Language Processing (NLP) system for sentiment analysis using Deep Learning (Bidirectional LSTM), containerized with Docker, and served via FastAPI REST API.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Monitoring](#monitoring)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements a complete sentiment analysis system that classifies text into three categories:
- **Positive**: Expressing favorable opinions, satisfaction, or happiness
- **Neutral**: Expressing neutral or factual information
- **Negative**: Expressing unfavorable opinions, dissatisfaction, or criticism

### Key Technologies

- **Deep Learning**: TensorFlow/Keras with Bidirectional LSTM
- **API Framework**: FastAPI
- **Containerization**: Docker with multi-stage builds
- **Monitoring**: Custom metrics tracking
- **Testing**: Pytest

## ✨ Features

- ✅ **Bidirectional LSTM Model**: State-of-the-art deep learning architecture
- ✅ **Complete Preprocessing Pipeline**: Text cleaning, tokenization, padding
- ✅ **RESTful API**: FastAPI with automatic documentation
- ✅ **Docker Support**: Multi-stage optimized Docker builds
- ✅ **Monitoring & Metrics**: Request tracking, latency monitoring, error logging
- ✅ **Production-Ready**: Error handling, logging, health checks
- ✅ **Scalable**: Designed for horizontal scaling
- ✅ **Well-Documented**: Comprehensive documentation and code comments

## 🏗️ Architecture

### System Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   FastAPI API   │
│   (Port 8000)   │
└──────┬──────────┘
       │
       ├──► Preprocessing
       ├──► Model Inference
       └──► Metrics Collection
       │
       ▼
┌─────────────────┐
│  LSTM Model     │
│  (TensorFlow)   │
└─────────────────┘
```

### Model Architecture

1. **Embedding Layer**: Converts word indices to dense vectors (128 dimensions)
2. **Bidirectional LSTM Layer 1**: 64 units, processes sequences in both directions
3. **Dropout**: 0.5 rate for regularization
4. **Bidirectional LSTM Layer 2**: 32 units
5. **Dense Layer**: 64 units with ReLU activation
6. **Output Layer**: 3 units with softmax activation (3 classes)

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)
- 4GB+ RAM recommended
- GPU (optional, for faster training)

### Local Installation

1. **Clone the repository** (or navigate to project directory):
```bash
cd sentiment_analysis_project
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🏃 Quick Start

### Step 1: Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates a sample dataset at `data/raw/sample_sentiment_data.csv`.

### Step 2: Train the Model

```bash
python scripts/train_model.py
```

This will:
- Load and preprocess the data
- Train the Bidirectional LSTM model
- Save the model to `models/checkpoints/`
- Save the tokenizer to `models/tokenizer.pkl`
- Evaluate on test set

### Step 3: Start the API

```bash
# Option 1: Direct Python
python -m src.api.app

# Option 2: Using uvicorn
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Step 4: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing!"}'

# Batch prediction
curl -X POST "http://localhost:8000/batch_predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great product!", "Terrible service", "It's okay"]}'
```

## 📖 Usage

### Python API Usage

```python
from src.inference.predict import load_predictor

# Load predictor
predictor = load_predictor(
    model_path="models/checkpoints/sentiment_model_best.h5",
    tokenizer_path="models/tokenizer.pkl"
)

# Single prediction
result = predictor.predict_single("This is a great product!")
print(result)
# Output: {'text': '...', 'sentiment': 'positive', 'confidence': 0.95, ...}

# Batch prediction
results = predictor.predict_batch([
    "Great product!",
    "Terrible service",
    "It's okay"
])
```

### API Endpoints

#### 1. Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### 2. Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "text": "This product is amazing!"
}
```

Response:
```json
{
  "text": "This product is amazing!",
  "sentiment": "positive",
  "confidence": 0.95,
  "probabilities": {
    "negative": 0.02,
    "neutral": 0.03,
    "positive": 0.95
  }
}
```

#### 3. Batch Prediction
```http
POST /batch_predict
Content-Type: application/json

{
  "texts": ["Great!", "Bad product", "It's okay"]
}
```

#### 4. Metrics
```http
GET /metrics
```

Returns request counts, latency statistics, error rates, and prediction metrics.

#### 5. Model Info
```http
GET /model/info
```

Returns model architecture information.

## 🐳 Docker Deployment

### Build Docker Image

```bash
cd docker
docker build -f Dockerfile -t sentiment-analysis-api ..
```

### Run with Docker Compose

```bash
cd docker
docker-compose up -d
```

### Environment Variables

Set these in `docker-compose.yml` or as environment variables:

- `MODEL_PATH`: Path to model file (default: `/app/models/checkpoints/sentiment_model_best.h5`)
- `TOKENIZER_PATH`: Path to tokenizer file (default: `/app/models/tokenizer.pkl`)

### Access API

Once running, access at http://localhost:8000

## 📁 Project Structure

```
sentiment_analysis_project/
│
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data
│
├── notebooks/
│   └── eda_and_experiments.ipynb  # EDA and experimentation
│
├── src/
│   ├── data_processing/
│   │   └── preprocess.py  # Text preprocessing pipeline
│   │
│   ├── models/
│   │   └── lstm_model.py  # LSTM model architecture
│   │
│   ├── training/
│   │   └── train.py       # Training pipeline
│   │
│   ├── inference/
│   │   └── predict.py     # Prediction logic
│   │
│   ├── api/
│   │   └── app.py         # FastAPI application
│   │
│   ├── monitoring/
│   │   └── metrics.py     # Metrics tracking
│   │
│   └── utils/
│       └── logger.py      # Logging utility
│
├── docker/
│   ├── Dockerfile         # Docker image definition
│   └── docker-compose.yml # Docker Compose configuration
│
├── tests/
│   └── test_api.py        # API tests
│
├── scripts/
│   ├── generate_sample_data.py  # Data generation script
│   └── train_model.py           # Training script
│
├── models/                # Saved models (created after training)
│   ├── checkpoints/
│   └── tokenizer.pkl
│
├── docs/
│   ├── step1_problem_understanding.md
│   └── project_report.md
│
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🧪 Testing

Run tests:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v
```

## 📊 Monitoring

The API includes built-in monitoring:

- **Request Tracking**: Total, successful, failed requests
- **Latency Monitoring**: Average, min, max response times
- **Error Tracking**: Error types and counts
- **Prediction Metrics**: Sentiment distribution, confidence scores
- **Uptime Tracking**: System uptime and last request time

Access metrics via:
```bash
curl http://localhost:8000/metrics
```

## 🔧 Configuration

### Model Parameters

Edit `scripts/train_model.py` to adjust:
- `embedding_dim`: Embedding dimension (default: 128)
- `lstm_units`: LSTM units (default: 64)
- `batch_size`: Training batch size (default: 32)
- `epochs`: Maximum epochs (default: 50)
- `dropout_rate`: Dropout rate (default: 0.5)

### API Configuration

Edit `src/api/app.py` or set environment variables:
- `MODEL_PATH`: Model file path
- `TOKENIZER_PATH`: Tokenizer file path

## 📈 Performance

### Expected Metrics

- **Accuracy**: > 85% on test set
- **Latency**: < 200ms per prediction
- **Throughput**: > 100 requests/second (with proper scaling)
- **Model Size**: ~5-10 MB

### Optimization Tips

1. **GPU Training**: Use GPU for faster training
2. **Model Quantization**: Reduce model size for production
3. **Caching**: Implement caching for frequent predictions
4. **Load Balancing**: Use multiple API instances for scaling

## 🚀 Deployment

### Production Deployment

1. **Build optimized Docker image**:
```bash
docker build -f docker/Dockerfile -t sentiment-api:latest .
```

2. **Deploy to cloud** (AWS, GCP, Azure):
   - Use container services (ECS, Cloud Run, Container Instances)
   - Set up load balancer
   - Configure auto-scaling

3. **Set environment variables**:
   - `MODEL_PATH`
   - `TOKENIZER_PATH`
   - `LOG_LEVEL`

### Kubernetes Deployment

See `deployment/` directory for Kubernetes manifests (if created).

## 📝 Documentation

- **Problem Understanding**: See `docs/step1_problem_understanding.md`
- **Project Report**: See `docs/project_report.md`
- **API Docs**: Available at http://localhost:8000/docs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is for educational purposes.

## 👤 Author

Created as part of Month 5 Advanced Data Science Project.

## 🎓 Academic Use

This project is suitable for:
- Academic submissions
- Portfolio projects
- Technical interviews
- Learning MLOps and production ML

## 📞 Support

For issues or questions:
1. Check the documentation
2. Review error logs
3. Check API health endpoint

---

**Built with ❤️ using TensorFlow, FastAPI, and Docker**

