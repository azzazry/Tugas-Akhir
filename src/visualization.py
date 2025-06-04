# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

def create_research_visualizations():
    # Load all results
    with open('evaluation_results.pkl', 'rb') as f:
        eval_results = pickle.load(f)
    with open('graphsvx_explanations.pkl', 'rb') as f:
        explanations = pickle.load(f)

    precision, recall, thresholds = eval_results['precision_recall_curve']
    test_preds = eval_results['test_predictions']
    test_labels = eval_results['test_labels']
    optimal_threshold = eval_results['optimal_threshold']

    # 1. Precision-Recall Curve
    plt.figure(figsize=(12, 8))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.scatter(recall[np.argmax(2 * precision * recall / (precision + recall + 1e-8))],
                precision[np.argmax(2 * precision * recall / (precision + recall + 1e-8))],
                color='red', label='Optimal threshold', zorder=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Distribution of prediction scores (Positive vs Negative)
    plt.figure(figsize=(12, 6))
    sns.histplot(test_preds[test_labels==0], color='blue', label='Negatif', kde=True, stat='density', bins=50)
    sns.histplot(test_preds[test_labels==1], color='red', label='Positif', kde=True, stat='density', bins=50)
    plt.axvline(optimal_threshold, color='green', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.3f})')
    plt.title('Distribusi Skor Prediksi pada Data Test')
    plt.xlabel('Skor Prediksi')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # 3. Shapley Values untuk beberapa high-risk user (maksimal 5)
    sample_users = list(explanations.keys())[:5]
    fig, axes = plt.subplots(len(sample_users), 1, figsize=(10, 4*len(sample_users)))
    if len(sample_users) == 1:
        axes = [axes]
    for i, user_idx in enumerate(sample_users):
        shap_values = explanations[user_idx]['shapley_values']
        feature_names = list(shap_values.keys())
        scores = np.array(list(shap_values.values()))
        
        axes[i].barh(feature_names, scores, color='teal')
        axes[i].set_title(f'Shapley Values untuk User Index {user_idx}')
        axes[i].set_xlabel('Importance')
        axes[i].grid(True, axis='x')
    plt.tight_layout()
    plt.show()