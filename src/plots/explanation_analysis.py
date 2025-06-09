import matplotlib.pyplot as plt
import numpy as np

def _plot_explanation_analysis(explanations):
    """Analisis hasil GraphSVX explanation dari dictionary explanations"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    risk_levels = [data['risk_classification'] for data in explanations.values()]
    probabilities = [data['probability'] for data in explanations.values()]
    
    # Hitung distribusi risk levels
    risk_counts = {}
    for risk in risk_levels:
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    # Plot distribusi risk level
    colors_map = {'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'yellow'}
    colors = [colors_map.get(risk, 'gray') for risk in risk_counts.keys()]
    
    ax1.bar(risk_counts.keys(), risk_counts.values(), color=colors, alpha=0.7)
    ax1.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Users')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot distribusi probabilitas
    ax2.hist(probabilities, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_title('Insider Probability Distribution\n(Analyzed Users)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Insider Probability')
    ax2.set_ylabel('Number of Users')
    mean_prob = np.mean(probabilities)
    ax2.axvline(mean_prob, color='red', linestyle='--', label=f'Mean: {mean_prob:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('result/visualizations/explanation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()