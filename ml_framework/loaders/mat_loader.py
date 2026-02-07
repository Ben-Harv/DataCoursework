import pandas as pd
from scipy.io import loadmat
from .base import DataLoader


class MatLoader(DataLoader):
    """Loads data from MATLAB .mat files."""

    def __init__(self, key):
        """
        Args:
            key: The key/variable name in the .mat file to extract
        """
        super().__init__()
        self.key = key

    def load(self, path):
        data = loadmat(path)[self.key]
        df = pd.DataFrame(data)
        return df
