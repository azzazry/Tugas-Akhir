import matplotlib.pyplot as plt
import numpy as np
import os
from core.explain import get_feature_names

def _plot_feature_importance_analysis(explanations, output_dir):
    
    if isinstance(output_dir, dict):
        output_dir = output_dir["visualization_dir"]

    os.makedirs(output_dir, exist_ok=True)
    
    feature_names = get_feature_names()
    feature_importance_sum = np.zeros(len(feature_names))
    feature_importance_counts = np.zeros(len(feature_names))

    for data in explanations.values():
        importance_scores = data['importance_scores']
        for i, score in enumerate(importance_scores):
            if i < len(feature_names):
                feature_importance_sum[i] += score
                feature_importance_counts[i] += 1

    # Filter hanya fitur yang pernah muncul
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
    plt.barh(range(len(filtered_names)), filtered_avg_importance[sorted_indices],
             color='skyblue', alpha=0.8)
    plt.yticks(range(len(filtered_names)), filtered_names[sorted_indices])
    plt.xlabel('Average Feature Importance')
    plt.title('Feature Importance Analysis - GraphSVX Explanations', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

    for i, v in enumerate(filtered_avg_importance[sorted_indices]):
        plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[✔] Feature importance plot")