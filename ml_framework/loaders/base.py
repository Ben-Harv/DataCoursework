class DataLoader:
    """Base class for all data loaders."""

    def load(self, path):
        """
        Load data from the given path.

        Args:
            path: Path to the data source

        Returns:
            pandas DataFrame
        """
        raise NotImplementedError("Subclasses must implement load()")
