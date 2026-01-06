#!/bin/bash

# Test Runner Script

echo "Running test suite..."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run tests with coverage
pytest tests/ -v --cov=backend --cov=ml-pipeline --cov-report=html --cov-report=term-missing

echo "Tests complete! Coverage report generated in htmlcov/"

