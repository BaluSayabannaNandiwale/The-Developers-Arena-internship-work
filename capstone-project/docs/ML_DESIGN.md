# ML Design Decisions

## Model Selection

### Linear Regression
- **Purpose**: Baseline model for comparison
- **Advantages**: Simple, interpretable, fast
- **Use Case**: Quick estimates, feature importance analysis

### Random Forest
- **Purpose**: Ensemble method for robust predictions
- **Advantages**: Handles non-linear relationships, feature importance
- **Hyperparameters**: 200 trees, max_depth=15

### XGBoost
- **Purpose**: State-of-the-art gradient boosting
- **Advantages**: High accuracy, handles missing values
- **Hyperparameters**: 
  - n_estimators: 200
  - learning_rate: 0.1
  - max_depth: 6

### Neural Network
- **Purpose**: Deep learning for complex patterns
- **Architecture**: 4 layers (128-64-32-1)
- **Advantages**: Captures non-linear interactions

## Feature Engineering

### Engineered Features

1. **price_per_sqft**: Price divided by area
   - Captures area efficiency
   - Normalizes price by size

2. **age_bucket**: Property age categories
   - New (0-5 years)
   - Recent (5-10 years)
   - Moderate (10-20 years)
   - Old (20-50 years)
   - Very Old (50+ years)

3. **location_popularity_score**: Normalized location frequency
   - Indicates market demand
   - Range: 0-1

## Preprocessing Pipeline

1. **Missing Value Handling**: Median imputation for numerical features
2. **Categorical Encoding**: Label encoding for categorical variables
3. **Feature Scaling**: StandardScaler for numerical features
4. **Feature Engineering**: Applied before encoding and scaling

## Model Evaluation

### Metrics
- **MAE**: Mean Absolute Error (primary metric)
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error

### Validation Strategy
- Train/Validation/Test split: 64%/16%/20%
- Cross-validation for hyperparameter tuning
- MLflow for experiment tracking

## Model Selection Criteria

Best model selected based on:
1. Lowest validation MAE
2. Good generalization (low overfitting)
3. Reasonable inference time
4. Model interpretability (for business use)

## Production Considerations

1. **Model Versioning**: MLflow model registry
2. **A/B Testing**: Support for multiple model versions
3. **Monitoring**: Track prediction drift and performance
4. **Retraining**: Automated retraining pipeline
5. **Fallback**: Linear regression as fallback model

