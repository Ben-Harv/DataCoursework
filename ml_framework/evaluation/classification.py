import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

from .base import Evaluator, EvalResult


class ClassificationEvaluator(Evaluator):
    """Evaluates classifiers with accuracy, sensitivity, specificity, AUC."""
    
    def evaluate(self, model, X, y_true):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'auc': auc(fpr, tpr),
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
        }
        return EvalResult(metrics, y_pred, y_proba)
    
    def plot(self, result, y_true, title=''):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Confusion matrix
        cm = result.metrics['confusion_matrix']
        im = axes[0].imshow(cm, cmap='Blues')
        axes[0].set_title(f'Confusion Matrix - {title}')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0].text(j, i, str(cm[i, j]), ha='center', va='center')
        
        # ROC curve
        axes[1].plot(
            result.metrics['fpr'], 
            result.metrics['tpr'],
            color='darkorange',
            lw=2,
            label=f"AUC = {result.metrics['auc']:.3f}"
        )
        axes[1].plot([0, 1], [0, 1], 'k--', lw=2)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'ROC - {title}')
        axes[1].legend(loc='lower right')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
