import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def _plot_detailed_confusion_matrix(eval_results):
    """Plot confusion matrix yang detailed dengan style menarik"""
    cm = eval_results['confusion_matrix']
    
    # Set style yang lebih modern
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    
    # Custom colormap yang lebih menarik
    colors = ['#ffffff', '#e3f2fd', '#90caf9', '#1976d2', '#0d47a1']
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=256)
    
    # Create heatmap dengan style yang lebih menarik
    sns.heatmap(cm, annot=False, fmt='d', cmap=custom_cmap,
                xticklabels=['Normal', 'Insider'],
                yticklabels=['Normal', 'Insider'],
                cbar_kws={'label': 'Count', 'shrink': 0.8},
                linewidths=2, linecolor='white',
                square=True, cbar=True)
    
    # Title dengan styling yang lebih menarik
    plt.title('Confusion Matrix - GraphSAGE Insider Threat Detection', 
              fontsize=16, fontweight='bold', pad=20, color='#1a237e')
    plt.ylabel('Actual Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    
    # Add custom annotations dengan styling yang lebih baik
    total = np.sum(cm)
    if total > 0:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Count dengan font besar
                plt.text(j + 0.5, i + 0.4, f'{cm[i,j]}', 
                        ha='center', va='center', fontsize=18, 
                        fontweight='bold', color='white' if cm[i,j] > cm.max()/2 else '#1a237e')
                
                # Percentage dengan font sedang
                plt.text(j + 0.5, i + 0.65, f'({cm[i,j]/total*100:.1f}%)', 
                        ha='center', va='center', fontsize=12, 
                        fontweight='medium', color='white' if cm[i,j] > cm.max()/2 else '#424242')
    
    # Styling untuk ticks
    plt.xticks(fontsize=11, fontweight='medium')
    plt.yticks(fontsize=11, fontweight='medium', rotation=0)
    
    # Remove top and right spines untuk tampilan yang lebih clean
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('result/visualizations/confusion_matrix.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()