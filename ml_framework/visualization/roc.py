import matplotlib.pyplot as plt
from .base import Plotter


class MultiROCPlotter(Plotter):
    """Plots multiple ROC curves for model comparison."""
    
    def plot(self, results, title='ROC Comparison'):
        """
        Plot ROC curves for multiple models.
        
        Args:
            results: Dict of model_name -> EvalResult
            title: Plot title
        """
        plt.figure(figsize=(8, 6))
        
        for name, result in results.items():
            if 'fpr' in result.metrics and 'tpr' in result.metrics:
                plt.plot(
                    result.metrics['fpr'],
                    result.metrics['tpr'],
                    lw=2,
                    label=f"{name} (AUC = {result.metrics['auc']:.3f})"
                )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
