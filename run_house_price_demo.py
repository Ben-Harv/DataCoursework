"""
Example: Using ml_framework for house price regression

This shows how to use the framework for a regression task.
"""

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

from ml_framework import (
    Experiment,
    CSVLoader,
    ScalerTransformer,
    Regressor,
)


# Create experiment with loader and transformer
exp = Experiment(
    loader=CSVLoader(),
    transformers=[ScalerTransformer()]
)

# Load and prepare data
exp.load_data('house_price_regression_dataset.csv', target_col='House_Price')

# Apply transformations
exp.transform()

# Add models with grid search parameters
exp.add_model(Regressor('ridge', {'alpha': [0.1, 1, 10, 100]}))
exp.add_model(Regressor('rf', {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}))

# Train and evaluate
exp.train_all()
exp.evaluate_all(show_plots=True)

# Compare all models
exp.compare()
