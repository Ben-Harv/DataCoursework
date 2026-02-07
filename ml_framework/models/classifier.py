from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from .base import Model


class Classifier(Model):
    """Model for classification tasks."""

    model_type = 'classifier'
    ESTIMATORS = {
        'knn': KNeighborsClassifier,
        'logistic': LogisticRegression,
        'svc': SVC,
        'rf': RandomForestClassifier,
        'mlp': MLPClassifier,
    }

    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probability for positive class."""
        return self.model.predict_proba(X)[:, 1]
