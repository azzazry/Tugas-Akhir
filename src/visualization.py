import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
import networkx as nx
import pandas as pd

def create_research_visualizations():
    print("Creating comprehensive visualizations...")

    with open('result/logs/evaluation_results.pkl', 'rb') as f:
        eval_results = pickle.load(f)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Training Performance Overview
    _plot_training_overview(eval_results['training_info'])
    # 2. Model Performance Dashboard
    _plot_performance_dashboard(eval_results)
    # 3. ROC dan Precision-Recall Curves
    _plot_roc_pr_curves(eval_results)
    # 4. Confusion Matrix dengan detail
    _plot_detailed_confusion_matrix(eval_results)
    # 5. Model Prediction Analysis
    _plot_prediction_analysis(eval_results)
    # 6. Visualisasi struktur graph user-resource
    _plot_sample_graph_structure()
    # 7. Visualisasi top user yang paling sering diakses (berdasarkan interpretasi sederhana)
    _plot_top_suspicious_users(eval_results)
    
    print("All visualizations saved to result/visualizations/")

def _plot_training_overview(training_info):
    """Plot training curves dengan smoothing"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    epochs = range(len(training_info['train_losses']))
    ax1.plot(epochs, training_info['train_losses'], alpha=0.3, color='red', label='Raw Loss')
    window = max(1, len(training_info['train_losses']) // 20)
    if len(training_info['train_losses']) > window:
        smooth_loss = pd.Series(training_info['train_losses']).rolling(window=window).mean()
        ax1.plot(epochs, smooth_loss, color='red', linewidth=2, label='Smoothed Loss')
    
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy curve
    ax2.plot(epochs, training_info['train_accs'], color='blue', linewidth=2)
    ax2.set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('result/visualizations/training_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def _plot_performance_dashboard(eval_results):
    """Dashboard performa model"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Metrics comparison bar plot
    metrics = ['Accuracy', 'F1-Score', 'AUC']
    values = [eval_results['val_accuracy'], eval_results['val_f1_score'], eval_results['val_auc']]
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Class distribution
    true_labels = eval_results['val_true_labels']
    pred_labels = eval_results['val_predictions']
    
    class_names = ['Normal', 'Insider']
    true_counts = [np.sum(true_labels == 0), np.sum(true_labels == 1)]
    pred_counts = [np.sum(pred_labels == 0), np.sum(pred_labels == 1)]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax2.bar(x - width/2, true_counts, width, label='True Labels', alpha=0.8)
    ax2.bar(x + width/2, pred_counts, width, label='Predictions', alpha=0.8)
    ax2.set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    
    # 3. Prediction confidence distribution
    if eval_results['val_probabilities'].shape[1] > 1:
        insider_probs = eval_results['val_probabilities'][:, 1]
        ax3.hist(insider_probs[true_labels == 0], alpha=0.7, label='Normal Users', bins=20, color='blue')
        ax3.hist(insider_probs[true_labels == 1], alpha=0.7, label='Insider Users', bins=20, color='red')
        ax3.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Insider Probability')
        ax3.set_ylabel('Frequency')
        ax3.legend()
    
    # 4. Model complexity info
    ax4.axis('off')
    info_text = f"""Model Architecture Summary:
    
    Parameters: {eval_results['model_parameters']:,}
    Hidden Dimension: 64
    Number of Layers: 2
    
    Training Summary:
    Epochs: {eval_results['training_info']['epochs']}
    Final Loss: {eval_results['training_info']['final_loss']:.4f}
    Final Train Acc: {eval_results['training_info']['final_acc']:.4f}
    
    Validation Results:
    Accuracy: {eval_results['val_accuracy']:.4f}
    F1-Score: {eval_results['val_f1_score']:.4f}
    AUC: {eval_results['val_auc']:.4f}"""
    
    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_title('Model Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('result/visualizations/performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def _plot_roc_pr_curves(eval_results):
    """Plot ROC dan Precision-Recall curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    true_labels = eval_results['val_true_labels']
    
    # ROC Curve
    if eval_results['val_probabilities'].shape[1] > 1 and len(np.unique(true_labels)) > 1:
        probs = eval_results['val_probabilities'][:, 1]
        fpr, tpr, _ = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    if 'precision' in eval_results and 'recall' in eval_results:
        precision = eval_results['precision']
        recall = eval_results['recall']
        
        ax2.plot(recall, precision, color='blue', lw=2, label='PR curve')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig('result/visualizations/roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def _plot_detailed_confusion_matrix(eval_results):
    """Plot confusion matrix yang detailed"""
    cm = eval_results['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Insider'],
                yticklabels=['Normal', 'Insider'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Insider Threat Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentage annotations
    total = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'({cm[i,j]/total*100:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig('result/visualizations/detailed_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def _plot_prediction_analysis(eval_results):
    """Analisis prediksi model"""
    true_labels = eval_results['val_true_labels']
    pred_labels = eval_results['val_predictions']
    
    # Create prediction analysis
    correct_normal = np.sum((true_labels == 0) & (pred_labels == 0))
    correct_insider = np.sum((true_labels == 1) & (pred_labels == 1))
    false_positive = np.sum((true_labels == 0) & (pred_labels == 1))  # Normal predicted as Insider
    false_negative = np.sum((true_labels == 1) & (pred_labels == 0))  # Insider predicted as Normal
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prediction outcomes
    categories = ['Correct\nNormal', 'Correct\nInsider', 'False\nPositive', 'False\nNegative']
    values = [correct_normal, correct_insider, false_positive, false_negative]
    colors = ['green', 'darkgreen', 'orange', 'red']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_title('Prediction Outcomes Analysis', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # Error analysis pie chart
    error_labels = ['Correct Predictions', 'False Positives', 'False Negatives']
    error_values = [correct_normal + correct_insider, false_positive, false_negative]
    error_colors = ['lightgreen', 'orange', 'red']
    
    # Only plot if there are errors
    if sum(error_values[1:]) > 0:
        ax2.pie(error_values, labels=error_labels, colors=error_colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Perfect\nPredictions!', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=20, fontweight='bold')
        ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('result/visualizations/prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def _plot_sample_graph_structure():
    G = nx.Graph()

    user_nodes = [f"user_{i}" for i in range(5)]
    resource_nodes = [f"res_{i}" for i in range(3)]

    G.add_nodes_from(user_nodes, node_type='user')
    G.add_nodes_from(resource_nodes, node_type='resource')

    G.add_edge("user_0", "res_0")
    G.add_edge("user_1", "res_1")
    G.add_edge("user_2", "res_2")
    G.add_edge("user_3", "res_0")

    for user in user_nodes:
        G.add_edge(user, user)

    pos = nx.spring_layout(G, seed=42)
    node_colors = ['skyblue' if G.nodes[n]['node_type'] == 'user' else 'lightgreen' for n in G.nodes]

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            node_size=700, font_size=9, edge_color='gray')
    plt.title("User-Resource Interaction Graph", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("result/visualizations/sample_graph_structure.png", dpi=300)
    plt.close()

def _plot_top_suspicious_users(eval_results):
    if 'val_probabilities' in eval_results and eval_results['val_probabilities'].shape[1] > 1:
        insider_probs = eval_results['val_probabilities'][:, 1]
        user_ids = np.arange(len(insider_probs))
        top_users_idx = np.argsort(insider_probs)[-10:][::-1]  # Top 10

        plt.figure(figsize=(10, 6))
        sns.barplot(x=insider_probs[top_users_idx], y=[f"User {i}" for i in top_users_idx], palette="Reds_r")
        plt.xlabel("Predicted Insider Probability")
        plt.ylabel("User ID")
        plt.title("Top 10 Suspicious Users Based on Prediction Probability", fontsize=14, fontweight='bold')
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig("result/visualizations/top_suspicious_users.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    create_research_visualizations()