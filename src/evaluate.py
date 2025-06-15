import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
import pickle
import os
import numpy as np
from models.graphsage import GraphSAGE
from src.utils.config import get_paths

def evaluate_insider_threat_model(users='1000'):
    paths = get_paths(users)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_lines = []

    # Load data graph
    data = torch.load(paths['data_graph_path'], weights_only=False)
    if not hasattr(data, 'x_dict'):
        data.x_dict = {
            'user': data['user'].x,
            'pc': data['pc'].x,
            'url': data['url'].x
        }

    # Load model
    model = GraphSAGE(hidden_dim=64, out_dim=2, num_layers=2).to(device)
    model.load_state_dict(torch.load(paths['model_path'], map_location=device))
    model.eval()

    expected_edges = [('user', 'uses', 'pc'), ('user', 'visits', 'url'), ('user', 'interacts', 'user')]
    edge_index_dict = {etype: data.edge_index_dict[etype] for etype in expected_edges if etype in data.edge_index_dict}

    for key in data.x_dict:
        data.x_dict[key] = data.x_dict[key].to(device)

    with torch.no_grad():
        out = model(data.x_dict, edge_index_dict)
        val_mask = getattr(data['user'], 'val_mask', torch.ones(out.size(0), dtype=torch.bool)).to(device)

        val_out = out[val_mask]
        val_labels = data['user'].y[val_mask].to(device)
        val_probs = torch.softmax(val_out, dim=1)

        # Cari threshold terbaik berdasarkan F1
        thresholds = np.arange(0, 1.05, 0.05)
        best_f1 = 0
        best_thresh = 0.5

        for t in thresholds:
            val_pred_t = (val_probs[:, 1] > t).long()
            f1 = f1_score(val_labels.cpu(), val_pred_t.cpu(), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        val_pred = (val_probs[:, 1] > best_thresh).long()
        val_acc = accuracy_score(val_labels.cpu(), val_pred.cpu())
        val_f1 = f1_score(val_labels.cpu(), val_pred.cpu(), zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels.cpu(), val_probs[:, 1].cpu())
        except ValueError:
            val_auc = float('nan')

    log_lines.append(f"Best Threshold: {best_thresh:.2f}")
    log_lines.append(f"AUC: {val_auc:.4f}")

    print("\n".join(log_lines))

    class_report = classification_report(
        val_labels.cpu(), val_pred.cpu(),
        labels=[0, 1],
        target_names=['Normal', 'Insider'],
        zero_division=0
    )
    log_lines.append("\nClassification Report:")
    log_lines.append(class_report)
    print("\nClassification Report:")
    print(class_report)

    # Load metadata user
    try:
        with open(paths['user_metadata_path'], 'rb') as f:
            user_meta = pickle.load(f)
    except Exception as e:
        err_msg = f"Gagal load user metadata: {e}"
        log_lines.append(err_msg)
        print(err_msg)
        user_meta = None

    training_info = {}
    if os.path.exists(paths['training_info_path']):
        with open(paths['training_info_path'], 'rb') as f:
            training_info = pickle.load(f)

    # Kumpulkan hasil evaluasi
    eval_results = {
        'val_accuracy': val_acc,
        'val_f1_score': val_f1,
        'val_auc': val_auc,
        'val_predictions': val_pred.cpu().numpy(),
        'val_probabilities': val_probs.cpu().numpy(),
        'val_true_labels': val_labels.cpu().numpy(),
        'confusion_matrix': confusion_matrix(val_labels.cpu(), val_pred.cpu()),
        'best_threshold': best_thresh,
        'edge_types_used': list(edge_index_dict.keys()),
        'training_info': training_info,
        'model_parameters': {
            'hidden_dim': 64,
            'out_dim': 2,
            'num_layers': 2,
            'used_edges': list(edge_index_dict.keys()),
        },
        'user_metadata': user_meta
    }

    # Simpan hasil evaluasi
    with open(paths['evaluation_path'], 'wb') as f:
        pickle.dump(eval_results, f)

    # Simpan log evaluasi
    with open(paths['eval_log_path'], 'w') as f:
        for line in log_lines:
            f.write(line + '\n')

    print("Evaluasi selesai! Log disimpan ke", paths['eval_log_path'])
    return eval_results

if __name__ == "__main__":
    evaluate_insider_threat_model(users=1000)