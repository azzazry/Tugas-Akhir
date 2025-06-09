import matplotlib.pyplot as plt
import numpy as np

def _plot_prediction_analysis(eval_results):
    """Analisis prediksi model yang lebih detail"""
    true_labels = eval_results.get('val_true_labels')
    pred_labels = eval_results.get('val_predictions')

    if true_labels is None or pred_labels is None:
        print("True labels atau prediksi tidak ditemukan di eval_results.")
        return
    
    # Hitung kategori prediksi
    correct_normal = np.sum((true_labels == 0) & (pred_labels == 0))
    correct_insider = np.sum((true_labels == 1) & (pred_labels == 1))
    false_positive = np.sum((true_labels == 0) & (pred_labels == 1))
    false_negative = np.sum((true_labels == 1) & (pred_labels == 0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart breakdown
    categories = ['True\nNormal', 'True\nInsider', 'False\nPositive', 'False\nNegative']
    values = [correct_normal, correct_insider, false_positive, false_negative]
    colors = ['green', 'darkgreen', 'orange', 'red']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_title('Prediction Outcomes Breakdown', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Users')
    
    for bar, value in zip(bars, values):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                     str(value), ha='center', va='bottom', fontweight='bold')
    
    # Pie chart distribusi prediksi
    total_correct = correct_normal + correct_insider
    total_errors = false_positive + false_negative
    
    if total_correct + total_errors > 0:
        sizes = values
        labels = ['Correct Normal', 'Correct Insider', 'False Positive', 'False Negative']
        colors_pie = ['lightgreen', 'darkgreen', 'orange', 'red']
        explode = (0, 0, 0.1, 0.1)  # Sorot error
        
        ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
                startangle=90, explode=explode, textprops={'fontsize': 11})
        ax2.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Perfect\nPredictions!', ha='center', va='center', 
                 transform=ax2.transAxes, fontsize=20, fontweight='bold', color='green')
        ax2.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('result/visualizations/prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()