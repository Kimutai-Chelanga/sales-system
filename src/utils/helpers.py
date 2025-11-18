"""
Utility helper functions
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty print metrics"""
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.upper():15s}: {value:,.4f}")
    print("="*50 + "\n")
