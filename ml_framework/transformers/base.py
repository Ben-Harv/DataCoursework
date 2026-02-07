class Transformer:
    """Base class for all transformers. Follows sklearn fit/transform pattern."""
    
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        
        Args:
            X: Feature matrix
            y: Target vector (optional, unused by most transformers)
            
        Returns:
            self
        """
        raise NotImplementedError
    
    def transform(self, X):
        """
        Transform the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        raise NotImplementedError
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def __repr__(self):
        # Show class name and public config attributes (simple types only)
        attrs = {k: v for k, v in vars(self).items()
                 if not k.startswith('_') and isinstance(v, (int, float, str, bool, tuple, list))}
        params = ', '.join(f'{k}={v!r}' for k, v in attrs.items())
        return f"{self.__class__.__name__}({params})" if params else f"{self.__class__.__name__}()"
