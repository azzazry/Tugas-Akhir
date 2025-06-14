import matplotlib.pyplot as plt
import os

def _plot_training_overview(training_info, output_dir):
    # Support dict output_dir
    if isinstance(output_dir, dict):
        output_dir = output_dir["visualization_dir"]
    
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    epochs = range(len(training_info.get('train_losses', [])))

    # === Loss curve ===
    train_losses = training_info.get('train_losses', [])
    ax1.plot(epochs, train_losses, color='red', linewidth=2)
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.grid(True, alpha=0.3)

    final_loss = training_info.get('final_loss')
    if final_loss is not None:
        ax1.text(0.7, 0.9, f'Final Loss: {final_loss:.4f}', 
                 transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))

    # === Accuracy curve ===
    train_accs = training_info.get('train_accs', [])
    ax2.plot(epochs, train_accs, color='blue', linewidth=2)
    ax2.set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    final_acc = training_info.get('final_acc')
    if final_acc is not None:
        ax2.text(0.7, 0.1, f'Final Acc: {final_acc:.4f}', 
                 transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))

    # === Save ===
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_overview.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[✔] Grafik training overview")