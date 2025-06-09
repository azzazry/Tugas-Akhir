import matplotlib.pyplot as plt
import numpy as np

def _plot_user_risk_distribution(eval_results, threshold=0.1):
    """
    Plot distribusi risiko user berdasarkan probabilitas insider threat.
    Ditambah filter threshold probabilitas untuk highlight top suspicious users.
    """

    if ('val_probabilities' in eval_results
        and isinstance(eval_results['val_probabilities'], np.ndarray)
        and eval_results['val_probabilities'].shape[1] > 1):

        insider_probs = eval_results['val_probabilities'][:, 1]

        if len(np.unique(insider_probs)) > 1:
            # Hitung kategori risiko
            high_risk = np.sum(insider_probs > 0.7)
            medium_risk = np.sum((insider_probs > 0.3) & (insider_probs <= 0.7))
            low_risk = np.sum(insider_probs <= 0.3)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot histogram distribusi risiko
            categories = ['Low Risk\n(≤0.3)', 'Medium Risk\n(0.3-0.7)', 'High Risk\n(>0.7)']
            counts = [low_risk, medium_risk, high_risk]
            colors = ['green', 'orange', 'red']

            bars = ax1.bar(categories, counts, color=colors, alpha=0.7)
            ax1.set_title('User Risk Distribution', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Number of Users')

            for bar, count in zip(bars, counts):
                if count > 0:
                    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                             str(count), ha='center', va='bottom', fontweight='bold')

            # Filter user dengan probabilitas di atas threshold
            filtered_indices = np.where(insider_probs > threshold)[0]
            filtered_probs = insider_probs[filtered_indices]

            if len(filtered_probs) == 0:
                print("Tidak ada user dengan probabilitas > threshold. Menampilkan top 5 user dengan probabilitas tertinggi.")
                top_n = 5
                top_indices = np.argsort(insider_probs)[-top_n:][::-1]
                top_probs = insider_probs[top_indices]
                top_user_indices = top_indices
            else:
                top_n = min(5, len(filtered_probs))
                top_order = np.argsort(filtered_probs)[-top_n:][::-1]  # urut dari tinggi ke rendah
                top_user_indices = filtered_indices[top_order]
                top_probs = insider_probs[top_user_indices]

            # Ambil label user dari metadata jika ada
            user_meta = eval_results.get('user_metadata', None)
            if user_meta is not None:
                top_labels = [f"User {user_meta[i]['user_id']}" for i in top_user_indices]
            else:
                top_labels = [f'User {i}' for i in top_user_indices]

            # Plot bar horizontal top suspicious users
            ax2.barh(range(top_n), top_probs, color='red', alpha=0.7)
            ax2.set_yticks(range(top_n))
            ax2.set_yticklabels(top_labels)
            ax2.invert_yaxis()  # supaya yang terbesar di atas
            ax2.set_xlabel('Insider Probability')
            ax2.set_title(f'Top {top_n} Most Suspicious Users\n(Threshold > {threshold})', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 1)

            plt.tight_layout()
            plt.savefig('result/visualizations/user_risk_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("Semua user punya probabilitas yang sama, plot distribusi risiko dilewati.")
    else:
        print("Tidak ada data probabilitas yang valid untuk plot distribusi risiko.")