class DataLoader:
    """Base class for all data loaders."""

    def __init__(self):
        self._last_loaded = None

    def load(self, path):
        """
        Load data from the given path.

        Args:
            path: Path to the data source

        Returns:
            pandas DataFrame
        """
        raise NotImplementedError("Subclasses must implement load()")

    def describe(self):
        """Return a summary of the last loaded DataFrame."""
        if self._last_loaded is None:
            return "No data loaded yet."
        df = self._last_loaded
        return (
            f"Shape: {df.shape}, "
            f"Dtypes: {dict(df.dtypes.value_counts())}, "
            f"Missing: {int(df.isna().sum().sum())}"
        )
