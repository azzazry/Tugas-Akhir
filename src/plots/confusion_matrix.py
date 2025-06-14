import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os

def _plot_detailed_confusion_matrix(eval_results, output_dir):

    vis_path = output_dir["visualization_dir"]
    os.makedirs(vis_path, exist_ok=True)

    cm = eval_results.get('confusion_matrix', None)
    if cm is None:
        print("Confusion matrix tidak ditemukan di eval_results.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))

    # Custom colormap yang keren
    colors = ['#ffffff', '#e3f2fd', '#90caf9', '#1976d2', '#1976d2']
    custom_cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=256)

    sns.heatmap(cm, annot=False, fmt='d', cmap=custom_cmap,
                xticklabels=['Normal', 'Insider'],
                yticklabels=['Normal', 'Insider'],
                cbar_kws={'label': 'Count', 'shrink': 0.8},
                linewidths=2, linecolor='white',
                square=True, cbar=True)

    plt.title('Confusion Matrix',
              fontsize=16, fontweight='medium', pad=20, color="#1b1b1d")
    plt.ylabel('Label Asli', fontsize=13, fontweight='medium')
    plt.xlabel('Label Prediksi', fontsize=13, fontweight='medium')

    total = np.sum(cm)
    if total > 0:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Count dengan font besar
                plt.text(j + 0.5, i + 0.4, f'{cm[i,j]}',
                         ha='center', va='center', fontsize=18,
                         fontweight='medium', color='white' if cm[i,j] > cm.max()/2 else '#1976d2')

                # Percentage dengan font sedang
                plt.text(j + 0.5, i + 0.65, f'({cm[i,j]/total*100:.1f}%)',
                         ha='center', va='center', fontsize=12,
                         fontweight='medium', color='white' if cm[i,j] > cm.max()/2 else '#424242')

    plt.xticks(fontsize=11, fontweight='medium')
    plt.yticks(fontsize=11, fontweight='medium', rotation=0)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    save_path = os.path.join(vis_path, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[✔] Confusion matrix")
