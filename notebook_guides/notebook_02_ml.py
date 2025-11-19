# 02 - Machine Learning Models for Sales Forecasting
# Demonstrates ML skills: scikit-learn, XGBoost, LightGBM, model evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb

# MLOps
import mlflow
import mlflow.sklearn
import joblib

import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported successfully")

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================

print("\n" + "="*60)
print("DATA PREPARATION")
print("="*60)

# Load data
df = pd.read_csv('../data/sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Feature Engineering
print("\n📊 Engineering features...")

# Time-based features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['week_of_year'] = df['date'].dt.isocalendar().week
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

# Lag features (previous sales patterns)
df = df.sort_values('date')
df['revenue_lag_1'] = df.groupby('product')['revenue'].shift(1)
df['revenue_lag_7'] = df.groupby('product')['revenue'].shift(7)
df['revenue_rolling_7'] = df.groupby('product')['revenue'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

# Encode categorical variables
le_product = LabelEncoder()
le_city = LabelEncoder()
le_salesperson = LabelEncoder()

df['product_encoded'] = le_product.fit_transform(df['product'])
df['city_encoded'] = le_city.fit_transform(df['city'])
df['salesperson_encoded'] = le_salesperson.fit_transform(df['salesperson_id'])

# Drop rows with NaN from lag features
df = df.dropna()

print(f"✓ Features engineered: {df.shape}")
print(f"\nNew features: {[col for col in df.columns if col not in ['transaction_id', 'date', 'product', 'city', 'salesperson_id', 'customer_id', 'day_of_week']]}")

# ============================================================================
# 2. TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*60)
print("TRAIN-TEST SPLIT")
print("="*60)

# Define features and target
feature_cols = [
    'quantity', 'unit_price', 'month', 'day', 'day_of_week', 
    'quarter', 'week_of_year', 'is_weekend', 'is_holiday',
    'is_month_start', 'is_month_end', 'product_encoded', 
    'city_encoded', 'salesperson_encoded',
    'revenue_lag_1', 'revenue_lag_7', 'revenue_rolling_7'
]

X = df[feature_cols]
y = df['revenue']

# Time-based split (80-20)
split_date = df['date'].quantile(0.8)
train_mask = df['date'] <= split_date
test_mask = df['date'] > split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Training set: {X_train.shape[0]:,} samples ({train_mask.sum()/len(df)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]:,} samples ({test_mask.sum()/len(df)*100:.1f}%)")
print(f"Features: {len(feature_cols)}")

# ============================================================================
# 3. MODEL TRAINING & EVALUATION
# ============================================================================

print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

# Store results
results = {}

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a model"""
    print(f"\n🔄 Training {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results[name] = {
        'model': model,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'predictions': y_test_pred
    }
    
    print(f"✓ {name} trained successfully")
    print(f"   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"   Train MAE: {train_mae:,.2f} | Test MAE: {test_mae:,.2f}")
    
    return model

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)
}

for name, model in models.items():
    evaluate_model(name, model, X_train, X_test, y_train, y_test)

# ============================================================================
# 4. MODEL COMPARISON
# ============================================================================

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

# Create comparison dataframe
comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Train R²': [results[m]['train_r2'] for m in results.keys()],
    'Test R²': [results[m]['test_r2'] for m in results.keys()],
    'Train MAE': [results[m]['train_mae'] for m in results.keys()],
    'Test MAE': [results[m]['test_mae'] for m in results.keys()],
    'Test RMSE': [results[m]['test_rmse'] for m in results.keys()]
})

comparison = comparison.sort_values('Test R²', ascending=False)
print("\n" + comparison.to_string(index=False))

# Visualize comparison
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('R² Score Comparison', 'MAE Comparison')
)

# R² comparison
fig.add_trace(
    go.Bar(name='Train R²', x=comparison['Model'], y=comparison['Train R²']),
    row=1, col=1
)
fig.add_trace(
    go.Bar(name='Test R²', x=comparison['Model'], y=comparison['Test R²']),
    row=1, col=1
)

# MAE comparison
fig.add_trace(
    go.Bar(name='Train MAE', x=comparison['Model'], y=comparison['Train MAE']),
    row=1, col=2
)
fig.add_trace(
    go.Bar(name='Test MAE', x=comparison['Model'], y=comparison['Test MAE']),
    row=1, col=2
)

fig.update_layout(height=500, title_text="Model Performance Comparison")
fig.show()

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get best model (highest test R²)
best_model_name = comparison.iloc[0]['Model']
best_model = results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test R²: {results[best_model_name]['test_r2']:.4f}")

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Visualize
    fig = px.bar(importance_df.head(15), 
                 x='Importance', y='Feature',
                 orientation='h',
                 title=f'Feature Importance - {best_model_name}')
    fig.show()

# ============================================================================
# 6. PREDICTION VISUALIZATION
# ============================================================================

print("\n" + "="*60)
print("PREDICTION VISUALIZATION")
print("="*60)

# Actual vs Predicted
y_pred_best = results[best_model_name]['predictions']

fig = go.Figure()

# Scatter plot
fig.add_trace(go.Scatter(
    x=y_test, 
    y=y_pred_best,
    mode='markers',
    name='Predictions',
    marker=dict(size=5, opacity=0.5)
))

# Perfect prediction line
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Perfect Prediction',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title=f'Actual vs Predicted Revenue - {best_model_name}',
    xaxis_title='Actual Revenue',
    yaxis_title='Predicted Revenue',
    height=600
)
fig.show()

# ============================================================================
# 7. SAVE BEST MODEL
# ============================================================================

print("\n" + "="*60)
print("SAVING BEST MODEL")
print("="*60)

# Create models directory
import os
os.makedirs('../models', exist_ok=True)

# Save model
model_path = f'../models/best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
joblib.dump(best_model, model_path)
print(f"✓ Model saved: {model_path}")

# Save encoders
joblib.dump(le_product, '../models/product_encoder.pkl')
joblib.dump(le_city, '../models/city_encoder.pkl')
joblib.dump(le_salesperson, '../models/salesperson_encoder.pkl')
print("✓ Encoders saved")

# Save feature names
joblib.dump(feature_cols, '../models/feature_names.pkl')
print("✓ Feature names saved")

# ============================================================================
# 8. BUSINESS INSIGHTS
# ============================================================================

print("\n" + "="*60)
print("KEY BUSINESS INSIGHTS")
print("="*60)

print(f"\n📊 Model Performance:")
print(f"   Best Model: {best_model_name}")
print(f"   Accuracy (R²): {results[best_model_name]['test_r2']:.2%}")
print(f"   Average Error: KES {results[best_model_name]['test_mae']:,.2f}")
print(f"   RMSE: KES {results[best_model_name]['test_rmse']:,.2f}")

print(f"\n💡 Business Impact:")
avg_revenue = y_test.mean()
mape = (results[best_model_name]['test_mae'] / avg_revenue) * 100
print(f"   Average Transaction: KES {avg_revenue:,.2f}")
print(f"   Error Rate: {mape:.1f}%")
print(f"   Model can predict sales within ±{mape:.1f}% accuracy")

print("\n" + "="*60)
print("✓ MACHINE LEARNING PIPELINE COMPLETE!")
print("="*60)