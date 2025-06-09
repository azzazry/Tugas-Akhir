import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def _plot_feature_importance_analysis():
    """Analisis feature importance dari GraphSVX"""
    explanation_path = 'result/logs/graphsvx_explanations.pkl'
    
    if os.path.exists(explanation_path):
        with open(explanation_path, 'rb') as f:
            explanations = pickle.load(f)
        
        # Feature names sesuai dengan yang ada di explain file
        feature_names = [
            'login_frequency',
            'after_hours_activity', 
            'data_access_volume',
            'failed_logins',
            'privileged_access',
            'behavioral_score'
        ]
        
        # Aggregate feature importance across all users
        feature_importance_sum = np.zeros(len(feature_names))
        feature_importance_counts = np.zeros(len(feature_names))
        
        for user_idx, data in explanations.items():
            importance_scores = data['importance_scores']
            for i, score in enumerate(importance_scores):
                if i < len(feature_names):  # Safety check
                    feature_importance_sum[i] += score
                    feature_importance_counts[i] += 1
        
        # Calculate average importance
        avg_importance = np.divide(feature_importance_sum, feature_importance_counts, 
                                 out=np.zeros_like(feature_importance_sum), 
                                 where=feature_importance_counts!=0)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sorted_indices = np.argsort(avg_importance)[::-1]
        
        plt.barh(range(len(feature_names)), avg_importance[sorted_indices], color='skyblue', alpha=0.8)
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_indices])
        plt.xlabel('Average Feature Importance')
        plt.title('Feature Importance Analysis - GraphSVX Explanations', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(avg_importance[sorted_indices]):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('result/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance analysis completed")
    else:
        print("GraphSVX explanations not found, skipping feature importance analysis")