import numpy as np
from sklearn.decomposition import PCA
from .base import Transformer


class PCATransformer(Transformer):
    """PCA with automatic component selection based on variance threshold."""
    
    def __init__(self, variance_threshold=0.95):
        """
        Args:
            variance_threshold: Proportion of variance to retain (default 0.95)
        """
        self.variance_threshold = variance_threshold
        self.pca = None
        self.n_components = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X, y=None):
        # First pass: find optimal number of components
        pca_full = PCA(n_components=min(X.shape))
        pca_full.fit(X)
        
        self.explained_variance_ratio_ = pca_full.explained_variance_ratio_
        cumvar = np.cumsum(self.explained_variance_ratio_)
        self.n_components = np.argmax(cumvar >= self.variance_threshold) + 1
        
        # Second pass: fit with optimal components
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        
        print(f"PCA: {self.n_components} components for {self.variance_threshold*100}% variance")
        return self
    
    def transform(self, X):
        return self.pca.transform(X)
