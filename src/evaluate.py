import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import pickle
from models.graphsage import GraphSAGE

def evaluate_insider_threat_model():
    
    data = torch.load('data/data_graph.pt', weights_only=False)
    
    if not hasattr(data, 'x_dict'):
        data.x_dict = {
            'user': data['user'].x,
            'pc': data['pc'].x,
            'url': data['url'].x
        }
    
    model = GraphSAGE(hidden_dim=64, out_dim=2, num_layers=2)
    model.load_state_dict(torch.load('result/logs/insider_threat_graphsage.pt'))
    model.eval()
    
    # Filter edges sesuai training (unidirectional user->pc/url)
    expected_edges = [('user', 'uses', 'pc'), ('user', 'visits', 'url'), ('user', 'self_loop', 'user')]
    edge_index_dict = {etype: data.edge_index_dict[etype] for etype in expected_edges if etype in data.edge_index_dict}
    
    with torch.no_grad():
        out = model(data.x_dict, edge_index_dict)
        
        val_mask = data['user'].val_mask
        val_out = out[val_mask]
        val_labels = data['user'].y[val_mask]
        val_probs = torch.softmax(val_out, dim=1)
        
        val_pred = val_out.argmax(dim=1)
        val_acc = accuracy_score(val_labels.cpu(), val_pred.cpu())
        val_f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average='weighted', zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels.cpu(), val_probs[:, 1].cpu())
        except ValueError:
            val_auc = 0.0
    
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation F1-Score: {val_f1:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(val_labels.cpu(), val_pred.cpu(), target_names=['Normal', 'Insider'], zero_division=0))
    
    # Save evaluation results
    eval_results = {
        'val_accuracy': val_acc,
        'val_f1_score': val_f1,
        'val_auc': val_auc,
        'val_predictions': val_pred.cpu().numpy(),
        'val_probabilities': val_probs.cpu().numpy(),
        'val_true_labels': val_labels.cpu().numpy(),
        'confusion_matrix': confusion_matrix(val_labels.cpu(), val_pred.cpu()),
        'edge_types_used': list(edge_index_dict.keys()),
    }
    
    with open('result/logs/evaluation_results.pkl', 'wb') as f:
        pickle.dump(eval_results, f)
    
    print("Evaluasi selesai!")
    return eval_results

if __name__ == "__main__":
    evaluate_insider_threat_model()