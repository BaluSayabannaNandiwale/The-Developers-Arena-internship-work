#!/bin/bash

# Model Training Script
# This script trains all models and saves the best one

echo "Starting model training pipeline..."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Enrich data if not already done
if [ ! -f "data/house_prices_enriched.csv" ]; then
    echo "Enriching dataset..."
    python ml-pipeline/data/enrich_data.py
fi

# Train models
echo "Training models..."
python ml-pipeline/training/train_models.py

echo "Training complete!"
echo "Check the models/ directory for trained models."

