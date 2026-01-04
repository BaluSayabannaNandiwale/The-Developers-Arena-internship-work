# Quick Start Guide - Sentiment Analysis System

Get up and running in 5 minutes!

## Prerequisites

- Python 3.9+
- pip

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates `data/raw/sample_sentiment_data.csv` with 1000 sample texts.

### 3. Train the Model

```bash
python scripts/train_model.py
```

This will:
- Load and preprocess data
- Train the Bidirectional LSTM model
- Save model to `models/checkpoints/sentiment_model_best.h5`
- Save tokenizer to `models/tokenizer.pkl`
- Evaluate on test set

**Note**: Training takes 10-30 minutes depending on your hardware.

### 4. Start the API

```bash
# Option 1: Direct
python -m src.api.app

# Option 2: Uvicorn
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test the API

Open your browser:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

Or use curl:

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing!"}'
```

## Docker Quick Start

### Build and Run

```bash
cd docker
docker-compose up -d
```

Access at http://localhost:8000

## Troubleshooting

### Model Not Found Error

If you see "Model not loaded" error:
1. Make sure you've run `python scripts/train_model.py`
2. Check that files exist:
   - `models/checkpoints/sentiment_model_best.h5`
   - `models/tokenizer.pkl`

### Import Errors

If you see import errors:
```bash
pip install -r requirements.txt
```

### Port Already in Use

If port 8000 is busy:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8001
```

## Next Steps

- Read the full [README.md](README.md)
- Check [docs/project_report.md](docs/project_report.md) for detailed documentation
- Explore the API at http://localhost:8000/docs
- Run tests: `pytest tests/ -v`

## Need Help?

1. Check the logs in `logs/` directory
2. Review error messages in the terminal
3. Check API health: `curl http://localhost:8000/health`
4. Review the comprehensive README.md

Happy coding! 🚀

