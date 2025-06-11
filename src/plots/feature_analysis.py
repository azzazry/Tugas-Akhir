import matplotlib.pyplot as plt
import numpy as np

def _plot_feature_importance_analysis(explanations):
    feature_names = [
        'Total Logon Events', 'Total File Events', 'Total Device Events', 'Total HTTP Events',
        'Encoded Role', 'Encoded Department', 'Logon Count', 'After Hours Logon', 'Weekend Logon',
        'File Open Count', 'File Write Count', 'File Copy Count', 'File Delete Count',
        'Device Connect Count', 'Device Disconnect Count', 'Visit Frequency', 'Unique Visit Days',
        'After Hours Browsing', 'Cloud Service Visits', 'Job Site Visits'
    ]

    feature_importance_sum = np.zeros(len(feature_names))
    feature_importance_counts = np.zeros(len(feature_names))

    for data in explanations.values():
        importance_scores = data['importance_scores']
        for i, score in enumerate(importance_scores):
            if i < len(feature_names):
                feature_importance_sum[i] += score
                feature_importance_counts[i] += 1

    # Filter hanya fitur dengan kontribusi > 0
    mask = feature_importance_counts > 0
    filtered_names = np.array(feature_names)[mask]
    filtered_avg_importance = np.divide(
        feature_importance_sum[mask], 
        feature_importance_counts[mask], 
        out=np.zeros_like(feature_importance_sum[mask]), 
        where=feature_importance_counts[mask] != 0
    )

    sorted_indices = np.argsort(filtered_avg_importance)[::-1]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(filtered_names)), filtered_avg_importance[sorted_indices], color='skyblue', alpha=0.8)
    plt.yticks(range(len(filtered_names)), filtered_names[sorted_indices])
    plt.xlabel('Average Feature Importance')
    plt.title('Feature Importance Analysis - GraphSVX Explanations', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    for i, v in enumerate(filtered_avg_importance[sorted_indices]):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('result/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()