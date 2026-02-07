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
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Predicted vs Actual
        axes[0].scatter(y_true, result.predictions, alpha=0.5)
        min_val = min(y_true.min(), result.predictions.min())
        max_val = max(y_true.max(), result.predictions.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title(f'Predicted vs Actual - {title}')
        axes[0].grid(True)
        
        # Residual distribution
        residuals = np.array(y_true) - result.predictions
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residual')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        
        plt.tight_layout()
        plt.show()
