import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

def create_research_visualizations():
    print("Creating comprehensive visualizations...")
    
    # Create output directory if not exists
    os.makedirs('result/visualizations', exist_ok=True)
    
    # Load evaluation results
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
    # 6. GraphSVX Explanation Analysis (jika ada)
    _plot_explanation_analysis()
    # 7. Feature Importance dari GraphSVX
    _plot_feature_importance_analysis()
    # 8. User Risk Distribution
    _plot_user_risk_distribution(eval_results)
    
    print("All visualizations saved to result/visualizations/")

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

def _plot_performance_dashboard(eval_results):
    """Dashboard performa model sesuai struktur data"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Metrics comparison
    metrics = ['Accuracy', 'F1-Score', 'AUC']
    auc_value = eval_results.get('val_auc', 0)
    if np.isnan(auc_value):
        auc_value = 0
    
    values = [eval_results['val_accuracy'], eval_results['val_f1_score'], auc_value]
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim([0, 1])
    
    for bar, value in zip(bars, values):
        label = f'{value:.3f}' if not np.isnan(value) else 'N/A'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                label, ha='center', va='bottom', fontweight='bold')
    
    # 2. Class distribution
    true_labels = eval_results['val_true_labels']
    pred_labels = eval_results['val_predictions']
    
    class_names = ['Normal', 'Insider']
    true_counts = [np.sum(true_labels == 0), np.sum(true_labels == 1)]
    pred_counts = [np.sum(pred_labels == 0), np.sum(pred_labels == 1)]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax2.bar(x - width/2, true_counts, width, label='True Labels', alpha=0.8, color='blue')
    ax2.bar(x + width/2, pred_counts, width, label='Predictions', alpha=0.8, color='orange')
    ax2.set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    
    # Add count labels
    for i, (true_c, pred_c) in enumerate(zip(true_counts, pred_counts)):
        ax2.text(i - width/2, true_c + 0.5, str(true_c), ha='center', va='bottom')
        ax2.text(i + width/2, pred_c + 0.5, str(pred_c), ha='center', va='bottom')
    
    # 3. Prediction confidence distribution
    if 'val_probabilities' in eval_results and eval_results['val_probabilities'].shape[1] > 1:
        insider_probs = eval_results['val_probabilities'][:, 1]
        ax3.hist(insider_probs[true_labels == 0], alpha=0.7, label='True Normal', bins=20, color='blue')
        if np.sum(true_labels == 1) > 0:
            ax3.hist(insider_probs[true_labels == 1], alpha=0.7, label='True Insider', bins=20, color='red')
        ax3.set_title('Insider Probability Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Insider Probability')
        ax3.set_ylabel('Frequency')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No probability\ndata available', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Insider Probability Distribution', fontsize=14, fontweight='bold')
    
    # 4. Model architecture info
    ax4.axis('off')
    model_params = eval_results.get('model_parameters', {})
    training_info = eval_results.get('training_info', {})
    
    info_text = f"""GraphSAGE Model Summary:

Architecture:
• Hidden Dimension: {model_params.get('hidden_dim', 64)}
• Output Dimension: {model_params.get('out_dim', 2)}
• Number of Layers: {model_params.get('num_layers', 2)}
• Edge Types Used: {len(model_params.get('used_edges', []))}

Training Configuration:
• Epochs: {training_info.get('epochs', 100)}
• Final Loss: {training_info.get('final_loss', 0):.4f}
• Final Train Acc: {training_info.get('final_acc', 0):.4f}

Validation Performance:
• Accuracy: {eval_results['val_accuracy']:.4f}
• F1-Score: {eval_results['val_f1_score']:.4f}
• AUC: {'N/A' if np.isnan(auc_value) else f'{auc_value:.4f}'}"""
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_title('Model Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('result/visualizations/performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def _plot_roc_pr_curves(eval_results):
    """Plot ROC dan Precision-Recall curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    true_labels = eval_results['val_true_labels']
    
    if ('val_probabilities' in eval_results and 
        eval_results['val_probabilities'].shape[1] > 1 and 
        len(np.unique(true_labels)) > 1):
        
        probs = eval_results['val_probabilities'][:, 1]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, probs)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve - Insider Threat Detection', fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(true_labels, probs)
        avg_precision = np.trapz(precision, recall)
        
        ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
    else:
        for ax, title in zip([ax1, ax2], ['ROC Curve', 'Precision-Recall Curve']):
            ax.text(0.5, 0.5, 'Curve not available\n(insufficient class diversity)', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('result/visualizations/roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def _plot_detailed_confusion_matrix(eval_results):
    """Plot confusion matrix yang detailed dengan style menarik"""
    cm = eval_results['confusion_matrix']
    
    # Set style yang lebih modern
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    
    # Custom colormap yang lebih menarik
    colors = ['#ffffff', '#e3f2fd', '#90caf9', '#1976d2', '#0d47a1']
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=256)
    
    # Create heatmap dengan style yang lebih menarik
    sns.heatmap(cm, annot=False, fmt='d', cmap=custom_cmap,
                xticklabels=['Normal', 'Insider'],
                yticklabels=['Normal', 'Insider'],
                cbar_kws={'label': 'Count', 'shrink': 0.8},
                linewidths=2, linecolor='white',
                square=True, cbar=True)
    
    # Title dengan styling yang lebih menarik
    plt.title('Confusion Matrix - GraphSAGE Insider Threat Detection', 
              fontsize=16, fontweight='bold', pad=20, color='#1a237e')
    plt.ylabel('Actual Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    
    # Add custom annotations dengan styling yang lebih baik
    total = np.sum(cm)
    if total > 0:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Count dengan font besar
                plt.text(j + 0.5, i + 0.4, f'{cm[i,j]}', 
                        ha='center', va='center', fontsize=18, 
                        fontweight='bold', color='white' if cm[i,j] > cm.max()/2 else '#1a237e')
                
                # Percentage dengan font sedang
                plt.text(j + 0.5, i + 0.65, f'({cm[i,j]/total*100:.1f}%)', 
                        ha='center', va='center', fontsize=12, 
                        fontweight='medium', color='white' if cm[i,j] > cm.max()/2 else '#424242')
    
    # Styling untuk ticks
    plt.xticks(fontsize=11, fontweight='medium')
    plt.yticks(fontsize=11, fontweight='medium', rotation=0)
    
    # Remove top and right spines untuk tampilan yang lebih clean
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('result/visualizations/confusion_matrix.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def _plot_prediction_analysis(eval_results):
    """Analisis prediksi model yang lebih detail"""
    true_labels = eval_results['val_true_labels']
    pred_labels = eval_results['val_predictions']
    
    # Calculate prediction outcomes
    correct_normal = np.sum((true_labels == 0) & (pred_labels == 0))
    correct_insider = np.sum((true_labels == 1) & (pred_labels == 1))
    false_positive = np.sum((true_labels == 0) & (pred_labels == 1))
    false_negative = np.sum((true_labels == 1) & (pred_labels == 0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prediction outcomes bar chart
    categories = ['True\nNormal', 'True\nInsider', 'False\nPositive', 'False\nNegative']
    values = [correct_normal, correct_insider, false_positive, false_negative]
    colors = ['green', 'darkgreen', 'orange', 'red']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_title('Prediction Outcomes Breakdown', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Users')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        if value > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom', fontweight='bold')
    
    # Model performance metrics pie chart
    total_correct = correct_normal + correct_insider
    total_errors = false_positive + false_negative
    
    if total_errors > 0:
        sizes = [correct_normal, correct_insider, false_positive, false_negative]
        labels = ['Correct Normal', 'Correct Insider', 'False Positive', 'False Negative']
        colors_pie = ['lightgreen', 'darkgreen', 'orange', 'red']
        explode = (0, 0, 0.1, 0.1)  # Highlight errors
        
        ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
                startangle=90, explode=explode)
        ax2.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Perfect\nPredictions!', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=20, fontweight='bold', color='green')
        ax2.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('result/visualizations/prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

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

def _plot_feature_importance_analysis():
    """Analisis feature importance dari GraphSVX"""
    explanation_path = 'result/logs/graphsvx_explanations.pkl'
    
    if os.path.exists(explanation_path):
        with open(explanation_path, 'rb') as f:
            explanations = pickle.load(f)
        
        # Feature names sesuai dengan yang ada di explain file
        feature_names = [
            'login_frequency',
            'after_hours_activity', 
            'data_access_volume',
            'failed_logins',
            'privileged_access',
            'behavioral_score'
        ]
        
        # Aggregate feature importance across all users
        feature_importance_sum = np.zeros(len(feature_names))
        feature_importance_counts = np.zeros(len(feature_names))
        
        for user_idx, data in explanations.items():
            importance_scores = data['importance_scores']
            for i, score in enumerate(importance_scores):
                if i < len(feature_names):  # Safety check
                    feature_importance_sum[i] += score
                    feature_importance_counts[i] += 1
        
        # Calculate average importance
        avg_importance = np.divide(feature_importance_sum, feature_importance_counts, 
                                 out=np.zeros_like(feature_importance_sum), 
                                 where=feature_importance_counts!=0)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sorted_indices = np.argsort(avg_importance)[::-1]
        
        plt.barh(range(len(feature_names)), avg_importance[sorted_indices], color='skyblue', alpha=0.8)
        plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_indices])
        plt.xlabel('Average Feature Importance')
        plt.title('Feature Importance Analysis - GraphSVX Explanations', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(avg_importance[sorted_indices]):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('result/visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance analysis completed")
    else:
        print("GraphSVX explanations not found, skipping feature importance analysis")

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

if __name__ == "__main__":
    create_research_visualizations()