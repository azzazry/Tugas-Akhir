import matplotlib.pyplot as plt
import numpy as np

def _plot_performance_dashboard(eval_results):
    """Dashboard performa model sesuai struktur data"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Metrics comparison
    metrics = ['Accuracy', 'F1-Score', 'AUC']
    auc_value = eval_results.get('val_auc', 0)
    if np.isnan(auc_value):
        auc_value = 0
    
    values = [eval_results['val_accuracy'], eval_results['val_f1_score'], auc_value]
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim([0, 1])
    
    for bar, value in zip(bars, values):
        label = f'{value:.3f}' if not np.isnan(value) else 'N/A'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                label, ha='center', va='bottom', fontweight='bold')
    
    # 2. Class distribution
    true_labels = eval_results['val_true_labels']
    pred_labels = eval_results['val_predictions']
    
    class_names = ['Normal', 'Insider']
    true_counts = [np.sum(true_labels == 0), np.sum(true_labels == 1)]
    pred_counts = [np.sum(pred_labels == 0), np.sum(pred_labels == 1)]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax2.bar(x - width/2, true_counts, width, label='True Labels', alpha=0.8, color='blue')
    ax2.bar(x + width/2, pred_counts, width, label='Predictions', alpha=0.8, color='orange')
    ax2.set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    
    # Add count labels
    for i, (true_c, pred_c) in enumerate(zip(true_counts, pred_counts)):
        ax2.text(i - width/2, true_c + 0.5, str(true_c), ha='center', va='bottom')
        ax2.text(i + width/2, pred_c + 0.5, str(pred_c), ha='center', va='bottom')
    
    # 3. Prediction confidence distribution
    if 'val_probabilities' in eval_results and eval_results['val_probabilities'].shape[1] > 1:
        insider_probs = eval_results['val_probabilities'][:, 1]
        ax3.hist(insider_probs[true_labels == 0], alpha=0.7, label='True Normal', bins=20, color='blue')
        if np.sum(true_labels == 1) > 0:
            ax3.hist(insider_probs[true_labels == 1], alpha=0.7, label='True Insider', bins=20, color='red')
        ax3.set_title('Insider Probability Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Insider Probability')
        ax3.set_ylabel('Frequency')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No probability\ndata available', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Insider Probability Distribution', fontsize=14, fontweight='bold')
    
    # 4. Model architecture info
    ax4.axis('off')
    model_params = eval_results.get('model_parameters', {})
    training_info = eval_results.get('training_info', {})
    
    info_text = f"""GraphSAGE Model Summary:

Architecture:
• Hidden Dimension: {model_params.get('hidden_dim', 64)}
• Output Dimension: {model_params.get('out_dim', 2)}
• Number of Layers: {model_params.get('num_layers', 2)}
• Edge Types Used: {len(model_params.get('used_edges', []))}

Training Configuration:
• Epochs: {training_info.get('epochs', 100)}
• Final Loss: {training_info.get('final_loss', 0):.4f}
• Final Train Acc: {training_info.get('final_acc', 0):.4f}

Validation Performance:
• Accuracy: {eval_results['val_accuracy']:.4f}
• F1-Score: {eval_results['val_f1_score']:.4f}
• AUC: {'N/A' if np.isnan(auc_value) else f'{auc_value:.4f}'}"""
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_title('Model Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('result/visualizations/performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()