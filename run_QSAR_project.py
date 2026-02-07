"""
Example: Using ml_framework with QSAR data

This shows how to use the framework for a classification task.
"""

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

from ml_framework import (
    Experiment,
    MatLoader,
    ScalerTransformer,
    PCATransformer,
    Classifier,
)
from ml_framework.visualization import MultiROCPlotter, CorrelationPlotter


# Create experiment with loader and transformers
exp = Experiment(
    loader=MatLoader('QSAR_data'),
    transformers=[
        ScalerTransformer(),
        PCATransformer(variance_threshold=0.95)
    ]
)

# Load and prepare data
exp.load_data('QSAR_data.mat', target_col=41)

# Optional: visualize correlations before transformation
CorrelationPlotter().plot(exp.df.drop(41, axis=1))

# Apply transformations
exp.transform()

# Add models -- each constructor is explicit and self-documenting
exp.add_model(Classifier('knn', {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}, metric='euclidean'))
exp.add_model(Classifier('logistic', {'C': [0.01, 0.1, 1, 10, 25, 50, 75, 100], 'penalty': ['l1', 'l2']}, solver='liblinear', max_iter=1000))
exp.add_model(Classifier('svc', {'C': [0.1, 1, 5, 8, 10, 12, 15], 'gamma': ['auto', 'scale']}, probability=True, kernel='rbf', random_state=1))
exp.add_model(Classifier('mlp', {'hidden_layer_sizes': [(10,), (20,), (30,), (20, 20)], 'activation': ['tanh', 'relu'], 'solver': ['lbfgs'], 'alpha': [0.1, 0.5, 1, 2, 3, 4]}, max_iter=7000, random_state=1))

# Train and evaluate
exp.train_all()
exp.evaluate_all(show_plots=True)

# Compare all models
exp.compare()

# Plot combined ROC curves
MultiROCPlotter().plot(exp.results)
