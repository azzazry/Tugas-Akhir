import matplotlib.pyplot as plt

def _plot_training_overview(training_info):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    epochs = range(len(training_info['train_losses']))

    # Loss curve
    ax1.plot(epochs, training_info['train_losses'], color='red', linewidth=2, alpha=0.8)
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.grid(True, alpha=0.3)
    final_loss = training_info.get('final_loss', None)
    if final_loss is not None:
        ax1.text(0.7, 0.9, f'Final Loss: {final_loss:.4f}', 
                 transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Accuracy curve
    ax2.plot(epochs, training_info['train_accs'], color='blue', linewidth=2)
    ax2.set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    final_acc = training_info.get('final_acc', None)
    if final_acc is not None:
        ax2.text(0.7, 0.1, f'Final Acc: {final_acc:.4f}', 
                 transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Optional: tampilkan class weights jika ada
    class_weights = training_info.get('class_weights', None)
    if class_weights:
        weight_text = f"Class weights: {class_weights}"
        fig.suptitle(weight_text, fontsize=10, y=1.02, color='gray')

    plt.tight_layout()
    plt.savefig('result/visualizations/training_overview.png', dpi=300, bbox_inches='tight')
    plt.close()