import matplotlib.pyplot as plt
import numpy as np
import os

def _plot_performance_dashboard(eval_results, output_dir):

    if isinstance(output_dir, dict):
        output_dir = output_dir["visualization_dir"]
    
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
    val_accuracy = eval_results.get('val_accuracy', 0)
    val_f1_score = eval_results.get('val_f1_score', 0)
    auc_value = eval_results.get('val_auc', 0) or 0

    val_true_labels = eval_results.get('val_true_labels', np.array([]))
    val_predictions = eval_results.get('val_predictions', np.array([]))
    val_probabilities = eval_results.get('val_probabilities', None)

    # === 1. Metrics ===
    metrics = ['Accuracy', 'F1-Score', 'AUC']
    values = [val_accuracy, val_f1_score, auc_value]
    colors = ['skyblue', 'lightgreen', 'salmon']

    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_title('Model Performance Metrics', fontsize=14)
    ax1.set_ylabel('Score')
    ax1.set_ylim([0, min(1.2, max(values) + 0.2)])
    ax1.grid(axis='y', alpha=0.3)
    for bar, value in zip(bars, values):
        label = f'{value:.3f}' if not np.isnan(value) else 'N/A'
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, 
                 label, ha='center', va='bottom')

    # === 2. Class distribution ===
    class_names = ['Normal', 'Insider']
    true_counts = [np.sum(val_true_labels == 0), np.sum(val_true_labels == 1)]
    pred_counts = [np.sum(val_predictions == 0), np.sum(val_predictions == 1)]

    x = np.arange(len(class_names))
    width = 0.35
    ax2.bar(x - width / 2, true_counts, width, label='True Labels', alpha=0.8, color='blue')
    ax2.bar(x + width / 2, pred_counts, width, label='Predictions', alpha=0.8, color='orange')
    ax2.set_title('Class Distribution Comparison', fontsize=14)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    for i, (true_c, pred_c) in enumerate(zip(true_counts, pred_counts)):
        ax2.text(i - width / 2, true_c + 0.5, str(true_c), ha='center')
        ax2.text(i + width / 2, pred_c + 0.5, str(pred_c), ha='center')

    # === 3. Probability distribution ===
    ax3.set_title('Insider Probability Distribution', fontsize=14)
    if val_probabilities is not None and val_probabilities.shape[1] > 1:
        insider_probs = val_probabilities[:, 1]
        if np.sum(val_true_labels == 0) > 0:
            ax3.hist(insider_probs[val_true_labels == 0], alpha=0.6, label='True Normal', bins=20, color='blue')
        if np.sum(val_true_labels == 1) > 0:
            ax3.hist(insider_probs[val_true_labels == 1], alpha=0.6, label='True Insider', bins=20, color='red')
        ax3.set_xlabel('Insider Probability')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No probability\ndata available', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=12)
        ax3.axis('off')

    # === Save plot ===
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'performance_dashboard.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[✔] Performance dashboard")