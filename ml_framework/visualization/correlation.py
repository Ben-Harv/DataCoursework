import matplotlib.pyplot as plt
from .base import Plotter


class CorrelationPlotter(Plotter):
    """Plots correlation matrix heatmap."""
    
    def plot(self, df, title='Correlation Matrix'):
        """
        Plot correlation matrix for a DataFrame.
        
        Args:
            df: pandas DataFrame
            title: Plot title
        """
        corr = df.corr()
        
        plt.figure(figsize=(10, 8))
        plt.matshow(corr, fignum=1)
        plt.title(title, pad=20)
        plt.colorbar(label='Correlation Coefficient', shrink=0.8)
        plt.tight_layout()
        plt.show()
