import matplotlib.pyplot as plt
import os
from core.explain import get_risk_classification

def _plot_user_risk_explanations(explanations, output_dir, top_n=5):
    
    if not explanations or len(explanations) == 0:
        print("[⚠] Tidak ada data di explanations untuk divisualisasikan.")
        return

    # Mendukung dict output_dir dari get_paths()
    if isinstance(output_dir, dict):
        output_dir = output_dir["visualization_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # === Inisialisasi ===
    risk_counts = {
        'Resiko Rendah (Top Candidate)': 0,
        'Resiko Sedang': 0,
        'Resiko Tinggi': 0
    }
    probs = []

    for uid, info in explanations.items():
        prob = info.get('probability', 0.0)
        classification = get_risk_classification(prob)
        risk_counts[classification] += 1
        label_id = info.get('user_id', f"User {uid}")
        probs.append((label_id, prob))

    # === Ambil Top-N Probabilitas Tertinggi ===
    top_probs = sorted(probs, key=lambda x: x[1], reverse=True)[:top_n]
    top_labels = [str(uid) for uid, _ in top_probs]
    top_values = [prob for _, prob in top_probs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # === Chart 1: Distribusi Risiko ===
    categories = list(risk_counts.keys())
    counts = [risk_counts[k] for k in categories]
    colors = ['green', 'orange', 'red']
    bars = ax1.bar(categories, counts, color=colors, alpha=0.7)
    ax1.set_title('Distribusi Risiko Pengguna', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Jumlah Pengguna')
    for bar, count in zip(bars, counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     str(count), ha='center', va='bottom', fontweight='bold')

    # === Chart 2: Top-N User Insider Probabilities ===
    bars2 = ax2.barh(range(len(top_labels)), top_values, color='crimson', alpha=0.8)
    ax2.set_yticks(range(len(top_labels)))
    ax2.set_yticklabels(top_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probabilitas Insider')
    ax2.set_xlim(0, 1)
    ax2.set_title(f'Top {top_n} Pengguna Paling Mencurigakan', fontsize=14, fontweight='bold')
    for i, bar in enumerate(bars2):
        prob = bar.get_width()
        ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, f"{prob:.3f}",
                 va='center', fontsize=10)

    # === Simpan Gambar ===
    plt.tight_layout()
    save_path = os.path.join(output_dir, "user_risk_explanation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[✔] Visualisasi user risk")