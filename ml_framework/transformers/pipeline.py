from .base import Transformer


class TransformerPipeline(Transformer):
    """Chains multiple transformers together."""
    
    def __init__(self, transformers):
        """
        Args:
            transformers: List of Transformer instances
        """
        self.transformers = transformers
    
    def fit(self, X, y=None):
        for t in self.transformers:
            X = t.fit_transform(X, y)
        return self
    
    def transform(self, X):
        for t in self.transformers:
            X = t.transform(X)
        return X

    def __repr__(self):
        return f"TransformerPipeline({self.transformers!r})"
