import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def _plot_roc_pr_curves(eval_results):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    true_labels = eval_results.get('val_true_labels', np.array([]))
    val_probabilities = eval_results.get('val_probabilities', None)

    # Validasi data buat plot
    valid_data = (
        val_probabilities is not None and
        val_probabilities.shape[1] > 1 and
        len(np.unique(true_labels)) > 1
    )
    
    if valid_data:
        probs = val_probabilities[:, 1]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve - Insider Threat Detection', fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(true_labels, probs)
        avg_precision = np.trapz(precision, recall)
        
        ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
    else:
        for ax, title in zip([ax1, ax2], ['ROC Curve', 'Precision-Recall Curve']):
            ax.text(0.5, 0.5, 'Curve not available\n(insufficient class diversity or probabilities)', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('result/visualizations/roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()