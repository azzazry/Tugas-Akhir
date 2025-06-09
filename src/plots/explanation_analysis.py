import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def _plot_explanation_analysis():
    """Analisis hasil GraphSVX explanation jika tersedia"""
    explanation_path = 'result/logs/graphsvx_explanations.pkl'
    
    if os.path.exists(explanation_path):
        with open(explanation_path, 'rb') as f:
            explanations = pickle.load(f)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Risk level distribution
        risk_levels = []
        probabilities = []
        
        for user_idx, data in explanations.items():
            risk_levels.append(data['risk_classification'])
            probabilities.append(data['probability'])
        
        # Count risk levels
        risk_counts = {}
        for risk in risk_levels:
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        # Plot risk distribution
        ax1.bar(risk_counts.keys(), risk_counts.values(), 
                color=['red', 'orange', 'yellow'], alpha=0.7)
        ax1.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Users')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot probability distribution
        ax2.hist(probabilities, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_title('Insider Probability Distribution\n(Analyzed Users)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Insider Probability')
        ax2.set_ylabel('Number of Users')
        ax2.axvline(np.mean(probabilities), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(probabilities):.3f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('result/visualizations/explanation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Explanation analysis plotted for {len(explanations)} users")
    else:
        print("GraphSVX explanations not found, skipping explanation analysis")