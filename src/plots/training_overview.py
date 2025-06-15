import matplotlib.pyplot as plt
import os

def _plot_training_overview(training_info, output_dir):
    if isinstance(output_dir, dict):
        output_dir = output_dir["visualization_dir"]

    os.makedirs(output_dir, exist_ok=True)

    # Set gaya untuk publikasi
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'figure.dpi': 300,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(len(training_info.get('train_losses', [])))

    # === Loss curve ===
    train_losses = training_info.get('train_losses', [])
    ax1.plot(epochs, train_losses, color='tab:red', label='Training Loss')
    ax1.set_title('Training Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_xlim(left=0)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # === Accuracy curve ===
    train_accs = training_info.get('train_accs', [])
    ax2.plot(epochs, train_accs, color='tab:blue', label='Training Accuracy')
    ax2.set_title('Training Accuracy Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.set_xlim(left=0)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_overview.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[✔] Training overview")