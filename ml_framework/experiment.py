from sklearn.model_selection import train_test_split

from .transformers import TransformerPipeline
from .evaluation import get_evaluator


class Experiment:
    """
    Main orchestrator that combines data loading, transformation,
    model training, and evaluation.
    """

    def __init__(self, loader, transformers=None):
        """
        Args:
            loader: DataLoader instance
            transformers: List of Transformer instances (optional)
        """
        self.loader = loader
        self.transformer_pipeline = TransformerPipeline(transformers or [])
        self.models = {}
        self.results = {}

        # Data containers
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, path, target_col, test_size=0.2, random_state=1):
        """
        Load and split data into train/test sets.

        Args:
            path: Path to data file
            target_col: Name or index of target column
            test_size: Proportion for test set
            random_state: Random seed for reproducibility

        Returns:
            self (for chaining)
        """
        self.df = self.loader.load(path)
        X = self.df.drop(target_col, axis=1)
        y = self.df[target_col]

        # Only stratify for classification (few unique values)
        stratify = y if len(set(y)) < 20 else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        print(f"Data loaded: {len(self.X_train)} train, {len(self.X_test)} test samples")
        return self

    def transform(self):
        """
        Apply transformers to train and test data.

        Returns:
            self (for chaining)
        """
        self.X_train = self.transformer_pipeline.fit_transform(self.X_train)
        self.X_test = self.transformer_pipeline.transform(self.X_test)
        return self

    def add_model(self, model, name=None):
        """
        Add a model to the experiment.

        Args:
            model: Model instance
            name: Model identifier (defaults to model.name)

        Returns:
            self (for chaining)
        """
        name = name or model.name
        self.models[name] = model
        return self

    def train_all(self):
        """
        Train all added models.

        Returns:
            self (for chaining)
        """
        for _, model in self.models.items():
            
            model.fit(self.X_train, self.y_train)
            print(f"\nTraining {model.summary()}...")
            if model.best_params:
                print(f"  Best params: {model.best_params}")
        return self

    def evaluate_all(self, show_plots=True):
        """
        Evaluate all trained models.

        Args:
            show_plots: Whether to display plots

        Returns:
            self (for chaining)
        """
        for name, model in self.models.items():
            evaluator = get_evaluator(model)
            result = evaluator.evaluate(model, self.X_test, self.y_test)
            self.results[name] = result

            #print(f"\n{name}: {result.format_metrics()}")

            if show_plots:
                evaluator.plot(result, self.y_test, title=name)

        return self

    def compare(self):
        """Print summary comparison of all models."""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        for name, result in self.results.items():
            print(f"{name}: {result.format_metrics()}")

        return self
