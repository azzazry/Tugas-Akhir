import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import pickle
from models.graphsage import HeteroGraphSAGE

def evaluate_insider_threat_model():
    """Evaluasi performa model Insider Threat GraphSAGE"""
    print("Loading data and trained model...")
    
    # Load data
    data = torch.load('data/data_graph.pt', weights_only=False)
    
    # Fix untuk NoneType error - pastikan semua x_dict tidak None (sama seperti training)
    for node_type in ['user', 'pc', 'url']:
        if node_type not in data.x_dict or data.x_dict[node_type] is None:
            print(f"Warning: {node_type} features is None, creating dummy features")
            if node_type == 'user':
                num_nodes = data['user'].x.shape[0] if hasattr(data['user'], 'x') else data['user'].num_nodes
                data.x_dict[node_type] = torch.randn(num_nodes, 6)
            elif node_type == 'pc':
                num_nodes = data['pc'].x.shape[0] if hasattr(data['pc'], 'x') else data['pc'].num_nodes
                data.x_dict[node_type] = torch.randn(num_nodes, 4)
            elif node_type == 'url':
                num_nodes = data['url'].x.shape[0] if hasattr(data['url'], 'x') else data['url'].num_nodes
                data.x_dict[node_type] = torch.randn(num_nodes, 3)
    
    # Pastikan format x_dict konsisten
    if not hasattr(data, 'x_dict'):
        data.x_dict = {
            'user': data['user'].x,
            'pc': data['pc'].x,
            'url': data['url'].x
        }
    
    # Load model
    model = HeteroGraphSAGE(hidden_dim=64, out_dim=2, num_layers=2)
    model.load_state_dict(torch.load('result/logs/insider_threat_graphsage.pt'))
    
    # TAMBAHAN: Verifikasi model architecture
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model has user_context_agg layer: {hasattr(model, 'user_context_agg')}")
    
    # Load training info
    with open('result/logs/training_info.pkl', 'rb') as f:
        training_info = pickle.load(f)
    
    # Filter edge_index_dict untuk hanya gunakan unidirectional edges (sama seperti training)
    expected_edge_types = [('user', 'uses', 'pc'), ('user', 'visits', 'url')]
    filtered_edge_index_dict = {}
    for edge_type in expected_edge_types:
        if edge_type in data.edge_index_dict:
            filtered_edge_index_dict[edge_type] = data.edge_index_dict[edge_type]
    
    print("Using edge types for evaluation:", list(filtered_edge_index_dict.keys()))
    
    print("Evaluating model on validation set...")
    model.eval()
    with torch.no_grad():
        try:
            # Forward pass dengan filtered edges (konsisten dengan training)
            out = model(data.x_dict, filtered_edge_index_dict)
            
            # Get probabilities for ROC analysis
            val_mask = data['user'].val_mask
            val_out = out[val_mask]
            val_labels = data['user'].y[val_mask]
            val_probs = torch.softmax(val_out, dim=1)
            
            val_pred = val_out.argmax(dim=1)
            val_acc = accuracy_score(val_labels.cpu(), val_pred.cpu())
            val_f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=0)
            
            # Calculate additional metrics
            try:
                val_auc = roc_auc_score(val_labels.cpu(), val_probs[:, 1].cpu())
            except:
                val_auc = 0.0  # Handle case when only one class in validation
        
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("Debugging info:")
            print(f"x_dict keys: {list(data.x_dict.keys())}")
            print(f"x_dict shapes: {[(k, v.shape if v is not None else 'None') for k, v in data.x_dict.items()]}")
            print(f"edge_index_dict keys: {list(filtered_edge_index_dict.keys())}")
            print(f"Model architecture: {model}")
            
            # TAMBAHAN: Debug untuk model layers
            if hasattr(model, 'user_context_agg'):
                print(f"user_context_agg weight shape: {model.user_context_agg.weight.shape}")
            raise e
    
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation F1-Score: {val_f1:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(val_labels.cpu(), val_pred.cpu(), 
                              target_names=['Normal', 'Insider'], zero_division=0))
    
    # Calculate precision recall curve data
    if len(np.unique(val_labels.cpu())) > 1:
        precision, recall, _ = precision_recall_curve(val_labels.cpu(), val_probs[:, 1].cpu())
    else:
        precision, recall = np.array([1]), np.array([1])
    
    # Save comprehensive evaluation results
    eval_results = {
        'val_accuracy': val_acc,
        'val_f1_score': val_f1,
        'val_auc': val_auc,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'val_predictions': val_pred.cpu().numpy(),
        'val_probabilities': val_probs.cpu().numpy(),
        'val_true_labels': val_labels.cpu().numpy(),
        'precision': precision,
        'recall': recall,
        'confusion_matrix': confusion_matrix(val_labels.cpu(), val_pred.cpu()),
        'training_info': training_info,
        'edge_types_used': list(filtered_edge_index_dict.keys()),  # Tambahan info
        'model_has_context_agg': hasattr(model, 'user_context_agg')  # TAMBAHAN: Info model architecture
    }
    
    with open('result/logs/evaluation_results.pkl', 'wb') as f:
        pickle.dump(eval_results, f)
    
    # Save text summary
    with open('result/logs/results_summary.txt', 'w') as f:
        f.write("=== Insider Threat Detection Results ===\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Validation F1-Score: {val_f1:.4f}\n")
        f.write(f"Validation AUC: {val_auc:.4f}\n")
        f.write(f"Model Parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write(f"Training Epochs: {training_info['epochs']}\n")
        f.write(f"Final Training Loss: {training_info['final_loss']:.4f}\n")
        f.write(f"Final Training Accuracy: {training_info['final_acc']:.4f}\n")
        f.write(f"Edge Types Used: {list(filtered_edge_index_dict.keys())}\n")  # Tambahan info
        f.write(f"Model Has Context Aggregation: {hasattr(model, 'user_context_agg')}\n")  # TAMBAHAN
        f.write(f"\nValidation Set Distribution:\n")
        f.write(f"Normal Users: {(val_labels.cpu() == 0).sum().item()}\n")
        f.write(f"Insider Users: {(val_labels.cpu() == 1).sum().item()}\n")
        f.write(f"\nDetailed Classification Report:\n")
        f.write(classification_report(val_labels.cpu(), val_pred.cpu(), 
                                    target_names=['Normal', 'Insider'], zero_division=0))
    
    print("Evaluation completed")
    return eval_results

if __name__ == "__main__":
    evaluate_insider_threat_model()