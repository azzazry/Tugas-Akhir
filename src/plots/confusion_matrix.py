import os
import matplotlib.pyplot as plt
import seaborn as sns

def _plot_confusion_matrix(eval_results, output_dir):
    vis_path = output_dir["visualization_dir"]
    os.makedirs(vis_path, exist_ok=True)
    cm = eval_results.get('confusion_matrix', None)
    
    plt.style.use('default')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Insider'],
                yticklabels=['Normal', 'Insider'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    save_path = os.path.join(vis_path, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[✔] Confusion matrix")