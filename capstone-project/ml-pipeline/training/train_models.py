"""
Model Training Script
Trains and compares multiple ML models with MLflow tracking
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import DataPreprocessor
from evaluation.metrics import calculate_metrics

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_linear_regression(X_train, y_train, X_val, y_val):
    """Train Linear Regression model"""
    print("\n[INFO] Training Linear Regression...")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    metrics = {
        'train_mae': mean_absolute_error(y_train, train_pred),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'val_r2': r2_score(y_val, val_pred),
        'train_mape': mean_absolute_percentage_error(y_train, train_pred),
        'val_mape': mean_absolute_percentage_error(y_val, val_pred)
    }
    
    return model, metrics

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest model"""
    print("\n[INFO] Training Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    metrics = {
        'train_mae': mean_absolute_error(y_train, train_pred),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'val_r2': r2_score(y_val, val_pred),
        'train_mape': mean_absolute_percentage_error(y_train, train_pred),
        'val_mape': mean_absolute_percentage_error(y_val, val_pred)
    }
    
    return model, metrics

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    print("\n[INFO] Training XGBoost...")
    
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    metrics = {
        'train_mae': mean_absolute_error(y_train, train_pred),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'val_r2': r2_score(y_val, val_pred),
        'train_mape': mean_absolute_percentage_error(y_train, train_pred),
        'val_mape': mean_absolute_percentage_error(y_val, val_pred)
    }
    
    return model, metrics

def train_neural_network(X_train, y_train, X_val, y_val):
    """Train Neural Network model"""
    print("\n[INFO] Training Neural Network...")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Build model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        train_pred = model.predict(X_train, verbose=0).flatten()
        val_pred = model.predict(X_val, verbose=0).flatten()
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'train_mape': mean_absolute_percentage_error(y_train, train_pred),
            'val_mape': mean_absolute_percentage_error(y_val, val_pred)
        }
        
        return model, metrics
        
    except ImportError:
        print("[WARN] TensorFlow not available, skipping Neural Network")
        return None, None

def train_all_models(data_path: str = None):
    """Train all models and log to MLflow"""
    
    # Set default data path
    if data_path is None:
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / 'data' / 'house_prices_enriched.csv'
    
    # Load data
    print("[INFO] Loading data...")
    print(f"[INFO] Data path: {data_path}")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    feature_cols = [
        'area_sqft', 'bedrooms', 'bathrooms', 'floor', 'total_floors',
        'property_age', 'location', 'city', 'property_type', 'facing',
        'furnishing', 'parking', 'amenities_score', 'distance_city_center_km'
    ]
    
    X = df[feature_cols]
    y = df['price_inr']
    
    # Train/validation/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )
    
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Preprocess
    print("[INFO] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train_processed, y_train = preprocessor.fit_transform(X_train, y_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    preprocessor.save(str(models_dir / 'preprocessor.pkl'))
    
    # Set up MLflow
    mlflow.set_experiment("real_estate_pricing")
    
    models = {}
    all_metrics = {}
    
    # Train Linear Regression
    with mlflow.start_run(run_name="linear_regression"):
        model, metrics = train_linear_regression(X_train_processed, y_train, X_val_processed, y_val)
        if model:
            models['linear_regression'] = model
            all_metrics['linear_regression'] = metrics
            mlflow.log_params({'model_type': 'linear_regression'})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            print(f"[OK] Linear Regression - Val MAE: Rs {metrics['val_mae']:,.2f}, R²: {metrics['val_r2']:.3f}")
    
    # Train Random Forest
    with mlflow.start_run(run_name="random_forest"):
        model, metrics = train_random_forest(X_train_processed, y_train, X_val_processed, y_val)
        if model:
            models['random_forest'] = model
            all_metrics['random_forest'] = metrics
            mlflow.log_params({
                'model_type': 'random_forest',
                'n_estimators': 200,
                'max_depth': 15
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            print(f"[OK] Random Forest - Val MAE: Rs {metrics['val_mae']:,.2f}, R²: {metrics['val_r2']:.3f}")
    
    # Train XGBoost
    with mlflow.start_run(run_name="xgboost"):
        model, metrics = train_xgboost(X_train_processed, y_train, X_val_processed, y_val)
        if model:
            models['xgboost'] = model
            all_metrics['xgboost'] = metrics
            mlflow.log_params({
                'model_type': 'xgboost',
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            print(f"[OK] XGBoost - Val MAE: Rs {metrics['val_mae']:,.2f}, R²: {metrics['val_r2']:.3f}")
    
    # Train Neural Network
    with mlflow.start_run(run_name="neural_network"):
        model, metrics = train_neural_network(X_train_processed, y_train, X_val_processed, y_val)
        if model:
            models['neural_network'] = model
            all_metrics['neural_network'] = metrics
            mlflow.log_params({'model_type': 'neural_network'})
            mlflow.log_metrics(metrics)
            # Save NN model separately
            model.save('../../models/neural_network.h5')
            print(f"[OK] Neural Network - Val MAE: Rs {metrics['val_mae']:,.2f}, R²: {metrics['val_r2']:.3f}")
    
    # Find best model
    best_model_name = min(all_metrics.keys(), key=lambda k: all_metrics[k]['val_mae'])
    best_model = models[best_model_name]
    best_metrics = all_metrics[best_model_name]
    
    print(f"\n[BEST] Best Model: {best_model_name}")
    print(f"   Val MAE: Rs {best_metrics['val_mae']:,.2f}")
    print(f"   Val RMSE: Rs {best_metrics['val_rmse']:,.2f}")
    print(f"   Val R²: {best_metrics['val_r2']:.3f}")
    print(f"   Val MAPE: {best_metrics['val_mape']:.2f}%")
    
    # Evaluate on test set
    test_pred = best_model.predict(X_test_processed)
    test_metrics = {
        'test_mae': mean_absolute_error(y_test, test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'test_r2': r2_score(y_test, test_pred),
        'test_mape': mean_absolute_percentage_error(y_test, test_pred)
    }
    
    print(f"\n[TEST] Test Set Performance:")
    print(f"   Test MAE: Rs {test_metrics['test_mae']:,.2f}")
    print(f"   Test RMSE: Rs {test_metrics['test_rmse']:,.2f}")
    print(f"   Test R²: {test_metrics['test_r2']:.3f}")
    print(f"   Test MAPE: {test_metrics['test_mape']:.2f}%")
    
    # Save best model
    import pickle
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / 'models'
    
    if best_model_name == 'neural_network':
        best_model.save(str(models_dir / 'production_model.h5'))
    else:
        with open(str(models_dir / f'{best_model_name}_production.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
        
        # Also save as production_model.pkl for easy loading
        with open(str(models_dir / 'production_model.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
    
    return models, all_metrics, best_model_name, preprocessor

if __name__ == "__main__":
    train_all_models()

