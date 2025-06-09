import matplotlib.pyplot as plt

def _plot_training_overview(training_info):
    """Plot training curves dengan data yang ada"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(len(training_info['train_losses']))
    
    # Loss curve
    ax1.plot(epochs, training_info['train_losses'], color='red', linewidth=2, alpha=0.8)
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.grid(True, alpha=0.3)
    
    # Add final loss annotation
    final_loss = training_info['final_loss']
    ax1.text(0.7, 0.9, f'Final Loss: {final_loss:.4f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Accuracy curve
    ax2.plot(epochs, training_info['train_accs'], color='blue', linewidth=2)
    ax2.set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Add final accuracy annotation
    final_acc = training_info['final_acc']
    ax2.text(0.7, 0.1, f'Final Acc: {final_acc:.4f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('result/visualizations/training_overview.png', dpi=300, bbox_inches='tight')
    plt.close()