import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .base import Evaluator, EvalResult


class RegressionEvaluator(Evaluator):
    """Evaluates regressors with MSE, RMSE, MAE, R²."""
    
    def evaluate(self, model, X, y_true):
        y_pred = model.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
        return EvalResult(metrics, y_pred)
    
    def plot(self, result, y_true, title=''):
        plt.figure(figsize=(6, 5))
        plt.scatter(y_true, result.predictions, alpha=0.5)
        min_val = min(y_true.min(), result.predictions.min())
        max_val = max(y_true.max(), result.predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Predicted vs Actual - {title}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
