import matplotlib.pyplot as plt
import numpy as np

def _plot_user_risk_distribution(eval_results):
    """Plot distribusi risiko user berdasarkan probabilitas"""
    if ('val_probabilities' in eval_results and 
        eval_results['val_probabilities'].shape[1] > 1):
        
        insider_probs = eval_results['val_probabilities'][:, 1]
        
        # Hanya plot jika ada variasi probabilitas
        if len(np.unique(insider_probs)) > 1:
            # Kategorisasi risiko
            high_risk = np.sum(insider_probs > 0.7)
            medium_risk = np.sum((insider_probs > 0.3) & (insider_probs <= 0.7))
            low_risk = np.sum(insider_probs <= 0.3)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Risk categories
            categories = ['Low Risk\n(≤0.3)', 'Medium Risk\n(0.3-0.7)', 'High Risk\n(>0.7)']
            counts = [low_risk, medium_risk, high_risk]
            colors = ['green', 'orange', 'red']
            
            bars = ax1.bar(categories, counts, color=colors, alpha=0.7)
            ax1.set_title('User Risk Distribution', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Number of Users')
            
            # Add count labels
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                            str(count), ha='center', va='bottom', fontweight='bold')
            
            # Top suspicious users
            top_n = min(10, len(insider_probs))
            top_indices = np.argsort(insider_probs)[-top_n:][::-1]
            
            ax2.barh(range(top_n), insider_probs[top_indices], color='red', alpha=0.7)
            ax2.set_yticks(range(top_n))
            ax2.set_yticklabels([f'User {i}' for i in top_indices])
            ax2.set_xlabel('Insider Probability')
            ax2.set_title(f'Top {top_n} Most Suspicious Users', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 1)
            
            plt.tight_layout()
            plt.savefig('result/visualizations/user_risk_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("All users have identical probabilities, skipping user risk distribution plot")
    else:
        print("No probability data available for user risk distribution")