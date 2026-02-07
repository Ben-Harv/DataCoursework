from sklearn.model_selection import GridSearchCV


class Model:
    """Base class for all models with built-in grid search cross-validation."""

    model_type = None  # Override in subclasses: 'classifier' or 'regressor'
    ESTIMATORS = {}    # Override in subclasses with name -> sklearn class mappings

    def __init__(self, name, param_grid=None, cv=5, **estimator_kwargs):
        """
        Args:
            name: String key from ESTIMATORS dict (e.g. 'knn', 'ridge')
            param_grid: Dict of hyperparameters for grid search
            cv: Number of cross-validation folds
            **estimator_kwargs: Passed directly to the sklearn estimator constructor
        """
        if name not in self.ESTIMATORS:
            raise ValueError(
                f"Unknown {self.model_type} '{name}'. "
                f"Available: {', '.join(self.ESTIMATORS)}"
            )
        self._name = name
        self.estimator = self.ESTIMATORS[name](**estimator_kwargs)
        self.param_grid = param_grid or {}
        self.cv = cv
        self.model = None
        self.best_params = None

    @property
    def name(self):
        """Read-only model name."""
        return self._name

    @property
    def is_fitted(self):
        """Whether the model has been trained."""
        return self.model is not None

    def fit(self, X, y):
        """
        Fit the model, using grid search if param_grid is provided.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self
        """
        if self.param_grid:
            search = GridSearchCV(
                self.estimator,
                self.param_grid,
                cv=self.cv,
                n_jobs=-1
            )
            search.fit(X, y)
            self.model = search.best_estimator_
            self.best_params = search.best_params_
        else:
            self.model = self.estimator.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError

    def summary(self):
        """Return a human-readable summary string."""
        fitted = 'fitted' if self.is_fitted else 'unfitted'
        params = f', best_params={self.best_params}' if self.best_params else ''
        return f"{self.__class__.__name__}('{self._name}', {fitted}{params})"

    def __repr__(self):
        return self.summary()
