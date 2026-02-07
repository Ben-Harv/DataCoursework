from sklearn.preprocessing import StandardScaler
from .base import Transformer


class ScalerTransformer(Transformer):
    """Standardizes features using z-score normalization."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        return self.scaler.transform(X)
