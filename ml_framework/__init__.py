"""
ML Framework - A simple, reusable data science toolkit.

Example usage:
    from ml_framework import Experiment, MatLoader, ScalerTransformer, Classifier

    exp = Experiment(
        loader=MatLoader('QSAR_data'),
        transformers=[ScalerTransformer()]
    )
    exp.load_data('data.mat', target_col=41)
    exp.transform()
    exp.add_model(Classifier('knn', {'n_neighbors': [1,3,5,7]}))
    exp.train_all()
    exp.evaluate_all()
"""

from .loaders import DataLoader, CSVLoader, MatLoader
from .transformers import Transformer, ScalerTransformer, PCATransformer, TransformerPipeline
from .models import Model, Classifier, Regressor
from .evaluation import Evaluator, EvalResult, ClassificationEvaluator, RegressionEvaluator, get_evaluator
from .experiment import Experiment

__all__ = [
    # Loaders
    'DataLoader', 'CSVLoader', 'MatLoader',
    # Transformers
    'Transformer', 'ScalerTransformer', 'PCATransformer', 'TransformerPipeline',
    # Models
    'Model', 'Classifier', 'Regressor',
    # Evaluation
    'Evaluator', 'EvalResult', 'ClassificationEvaluator', 'RegressionEvaluator', 'get_evaluator',
    # Pipeline
    'Experiment',
]
