# 03 - MLOps with MLflow: Experiment Tracking & Model Registry
# Demonstrates MLOps skills required by Solutech

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully")

# ============================================================================
# 1. SETUP MLFLOW
# ============================================================================

print("\n" + "="*60)
print("MLFLOW SETUP")
print("="*60)

# Set tracking URI (local for now, can be remote in production)
mlflow.set_tracking_uri("file:../mlops/mlruns")
mlflow.set_experiment("sales_forecasting_experiment")

print("✓ MLflow experiment initialized")
print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
print(f"   Experiment: {mlflow.get_experiment_by_name('sales_forecasting_experiment').name}")

# ============================================================================
# 2. DATA PREPARATION
# ============================================================================

print("\n" + "="*60)
print("DATA PREPARATION")
print("="*60)

# Load and prepare data
df = pd.read_csv('../data/sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Feature engineering
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['week_of_year'] = df['date'].dt.isocalendar().week
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Encode categoricals
le_product = LabelEncoder()
le_city = LabelEncoder()
df['product_encoded'] = le_product.fit_transform(df['product'])
df['city_encoded'] = le_city.fit_transform(df['city'])

# Features
feature_cols = [
    'quantity', 'unit_price', 'month', 'day', 'day_of_week',
    'quarter', 'week_of_year', 'is_weekend', 'is_holiday',
    'product_encoded', 'city_encoded'
]

X = df[feature_cols]
y = df['revenue']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Data prepared")
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")
print(f"   Features: {len(feature_cols)}")

# ============================================================================
# 3. EXPERIMENT 1: RANDOM FOREST WITH HYPERPARAMETER TUNING
# ============================================================================

print("\n" + "="*60)
print("EXPERIMENT 1: RANDOM FOREST")
print("="*60)

# Test different hyperparameters
rf_configs = [
    {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 10},
    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
    {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2}
]

for i, config in enumerate(rf_configs, 1):
    print(f"\n🔄 Training Random Forest Config {i}...")
    
    with mlflow.start_run(run_name=f"RandomForest_Config_{i}"):
        # Log parameters
        mlflow.log_params(config)
        mlflow.log_param("model_type", "RandomForest")
        
        # Train model
        model = RandomForestRegressor(**config, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Log metrics
        mlflow.log_metrics({
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        
        print(f"✓ Config {i} complete - Test R²: {test_r2:.4f}, MAE: {test_mae:,.2f}")

# ============================================================================
# 4. EXPERIMENT 2: XGBOOST
# ============================================================================

print("\n" + "="*60)
print("EXPERIMENT 2: XGBOOST")
print("="*60)

xgb_configs = [
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
    {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05},
    {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.01}
]

for i, config in enumerate(xgb_configs, 1):
    print(f"\n🔄 Training XGBoost Config {i}...")
    
    with mlflow.start_run(run_name=f"XGBoost_Config_{i}"):
        # Log parameters
        mlflow.log_params(config)
        mlflow.log_param("model_type", "XGBoost")
        
        # Train model
        model = xgb.XGBRegressor(**config, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Log metrics
        mlflow.log_metrics({
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse
        })
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        print(f"✓ Config {i} complete - Test R²: {test_r2:.4f}, MAE: {test_mae:,.2f}")

# ============================================================================
# 5. EXPERIMENT 3: LIGHTGBM
# ============================================================================

print("\n" + "="*60)
print("EXPERIMENT 3: LIGHTGBM")
print("="*60)

lgb_configs = [
    {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'num_leaves': 31},
    {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.05, 'num_leaves': 50},
]

for i, config in enumerate(lgb_configs, 1):
    print(f"\n🔄 Training LightGBM Config {i}...")
    
    with mlflow.start_run(run_name=f"LightGBM_Config_{i}"):
        # Log parameters
        mlflow.log_params(config)
        mlflow.log_param("model_type", "LightGBM")
        
        # Train model
        model = lgb.LGBMRegressor(**config, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Log metrics
        mlflow.log_metrics({
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse
        })
        
        # Log model
        mlflow.lightgbm.log_model(model, "model")
        
        print(f"✓ Config {i} complete - Test R²: {test_r2:.4f}, MAE: {test_mae:,.2f}")

# ============================================================================
# 6. COMPARE ALL EXPERIMENTS
# ============================================================================

print("\n" + "="*60)
print("COMPARING ALL EXPERIMENTS")
print("="*60)

# Get all runs from current experiment
experiment = mlflow.get_experiment_by_name("sales_forecasting_experiment")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Display results
print("\nAll Experiment Runs:")
comparison = runs[['run_id', 'params.model_type', 'metrics.test_r2', 
                    'metrics.test_mae', 'metrics.test_rmse']].copy()
comparison = comparison.sort_values('metrics.test_r2', ascending=False)
print(comparison.to_string(index=False))

# Get best run
best_run_id = comparison.iloc[0]['run_id']
best_model_type = comparison.iloc[0]['params.model_type']
best_r2 = comparison.iloc[0]['metrics.test_r2']

print(f"\n🏆 BEST MODEL:")
print(f"   Run ID: {best_run_id}")
print(f"   Model Type: {best_model_type}")
print(f"   Test R²: {best_r2:.4f}")

# ============================================================================
# 7. MODEL REGISTRY (Production Deployment)
# ============================================================================

print("\n" + "="*60)
print("MODEL REGISTRY & DEPLOYMENT")
print("="*60)

# Register best model
model_name = "sales_forecasting_production"

# Load the best model
best_model_uri = f"runs:/{best_run_id}/model"
model_version = mlflow.register_model(best_model_uri, model_name)

print(f"✓ Model registered: {model_name}")
print(f"   Version: {model_version.version}")

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)

print(f"✓ Model transitioned to Production stage")

# ============================================================================
# 8. LOAD PRODUCTION MODEL FOR INFERENCE
# ============================================================================

print("\n" + "="*60)
print("PRODUCTION INFERENCE")
print("="*60)

# Load production model
model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

# Make sample predictions
sample_data = X_test.head(5)
predictions = model.predict(sample_data)

print("\nSample Predictions:")
results_df = pd.DataFrame({
    'Actual': y_test.head(5).values,
    'Predicted': predictions,
    'Error': abs(y_test.head(5).values - predictions)
})
print(results_df.to_string(index=False))

print("\n" + "="*60)
print("✓ MLOps PIPELINE COMPLETE!")
print("="*60)
print("\nTo view experiments in MLflow UI, run:")
print("  cd ../mlops")
print("  mlflow ui")
print("  Then open: http://localhost:5000")