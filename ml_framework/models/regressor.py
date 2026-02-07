from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from .base import Model


class Regressor(Model):
    """Model for regression tasks."""

    model_type = 'regressor'
    ESTIMATORS = {
        'ridge': Ridge,
        'svr': SVR,
        'rf': RandomForestRegressor,
    }

    def predict(self, X):
        """Predict continuous values."""
        return self.model.predict(X)
