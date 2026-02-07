class EvalResult:
    """Container for evaluation results."""
    
    def __init__(self, metrics, predictions, probabilities=None):
        """
        Args:
            metrics: Dict of metric name -> value
            predictions: Array of predictions
            probabilities: Array of probability predictions (classifiers only)
        """
        self.metrics = metrics
        self.predictions = predictions
        self.probabilities = probabilities

    def format_metrics(self, decimals=4):
        """Return numeric metrics rounded to the given number of decimal places."""
        return {k: round(v, decimals) for k, v in self.metrics.items()
                if isinstance(v, (int, float))}


class Evaluator:
    """Base class for all evaluators."""
    
    def evaluate(self, model, X, y_true):
        """
        Evaluate model performance.
        
        Args:
            model: Trained Model instance
            X: Feature matrix
            y_true: True target values
            
        Returns:
            EvalResult
        """
        raise NotImplementedError
    
    def plot(self, result, y_true, title=''):
        """
        Plot evaluation results.
        
        Args:
            result: EvalResult instance
            y_true: True target values
            title: Plot title
        """
        raise NotImplementedError


def get_evaluator(model):
    """
    Returns appropriate evaluator for the model type.
    
    Args:
        model: Model instance
        
    Returns:
        Evaluator instance
    """
    from .classification import ClassificationEvaluator
    from .regression import RegressionEvaluator
    
    evaluators = {
        'classifier': ClassificationEvaluator(),
        'regressor': RegressionEvaluator(),
    }
    
    if model.model_type not in evaluators:
        raise ValueError(f"No evaluator for model type: {model.model_type}")
    
    return evaluators[model.model_type]
