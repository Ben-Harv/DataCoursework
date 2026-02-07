import pandas as pd
from .base import DataLoader


class CSVLoader(DataLoader):
    """Loads data from CSV files."""

    def __init__(self, **read_kwargs):
        """
        Args:
            **read_kwargs: Arguments passed to pandas.read_csv()
        """
        super().__init__()
        self.read_kwargs = read_kwargs

    def load(self, path):
        df = pd.read_csv(path, **self.read_kwargs)
        self._last_loaded = df
        return df
