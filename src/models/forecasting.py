import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn,model_selection import TimeSeriesSplit, cross_val_score
import mlflow
import joblib

class SalesForecaster:
    def __init__(self):
        self.model = GradientBoosting