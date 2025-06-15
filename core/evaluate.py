import torch
import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.models.graphsage import GraphSAGE
from src.utils.paths import get_paths
from src.utils.logger import log_line, clear_log_lines, flush_logs
from src.utils.constants import THRESHOLDS, EXPECTED_EDGE_TYPES, GRAPHSAGE_PARAMS, CLASS_NAMES, CLASS_LABELS
from src.utils.metrics import get_best_threshold, safe_auc_score
from src.utils.helpers import format_eval_line

def evaluate_insider_threat_model(users):
    paths = get_paths(users)
    clear_log_lines()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.load(paths['data_graph_path'], weights_only=False)
    if not hasattr(data, 'x_dict'):
        data.x_dict = {
            'user': data['user'].x,
            'pc': data['pc'].x,
            'url': data['url'].x
        }

    model = GraphSAGE(**GRAPHSAGE_PARAMS).to(device)
    model.load_state_dict(torch.load(paths['model_path'], map_location=device))
    model.eval()

    expected_edges = EXPECTED_EDGE_TYPES
    edge_index_dict = {etype: data.edge_index_dict[etype] for etype in expected_edges if etype in data.edge_index_dict}

    for key in data.x_dict:
        data.x_dict[key] = data.x_dict[key].to(device)

    with torch.no_grad():
        out = model(data.x_dict, edge_index_dict)
        val_mask = getattr(data['user'], 'val_mask', torch.ones(out.size(0), dtype=torch.bool)).to(device)

        val_out = out[val_mask]
        val_labels = data['user'].y[val_mask].to(device)
        val_probs = torch.softmax(val_out, dim=1)

        best_thresh, best_f1 = get_best_threshold(val_probs, val_labels, THRESHOLDS)
        val_pred = (val_probs[:, 1] > best_thresh).long()

        val_acc = accuracy_score(val_labels.cpu(), val_pred.cpu())
        val_f1 = f1_score(val_labels.cpu(), val_pred.cpu(), zero_division=0)
        val_auc = safe_auc_score(val_labels, val_probs)

    log_line(format_eval_line("Best Threshold", best_thresh))
    log_line(format_eval_line("AUC", val_auc))
    log_line(format_eval_line("Validation Accuracy", val_acc))
    log_line(format_eval_line("Validation F1-Score", val_f1))
    log_line(format_eval_line("F1 Score", best_f1))

    class_report = classification_report(
        val_labels.cpu(), val_pred.cpu(),
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0
    )
    log_line("\n[Classification Report]")
    log_line(class_report)

    # Load user metadata
    try:
        with open(paths['user_metadata_path'], 'rb') as f:
            user_meta = pickle.load(f)
    except Exception as e:
        log_line(f"[Warning] Gagal load user metadata: {e}")
        user_meta = None

    # Load training info
    training_info = {}
    if os.path.exists(paths['training_info_path']):
        with open(paths['training_info_path'], 'rb') as f:
            training_info = pickle.load(f)

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
            'num_layers': 1,
            'used_edges': list(edge_index_dict.keys()),
        },
        'user_metadata': user_meta
    }

    with open(paths['evaluation_path'], 'wb') as f:
        pickle.dump(eval_results, f)

    flush_logs(paths['eval_log_path'])

    return eval_results

if __name__ == "__main__":
    try:
        from src.utils.argparse import get_arguments
        args = get_arguments()
        evaluate_insider_threat_model(users=args.users)
    except ImportError:
        evaluate_insider_threat_model(users='1000')