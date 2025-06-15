import torch
from sklearn.metrics import f1_score, roc_auc_score

def compute_class_weights(labels, num_classes: int = 2, scaling: float = 1.0):
    classes, counts = torch.unique(labels, return_counts=True)
    weights = torch.zeros(num_classes, device=labels.device)
    for cls, cnt in zip(classes, counts):
        weights[cls] = 1.0 / cnt.float()
    weights = weights / weights.sum() * scaling
    return weights

def get_best_threshold(probabilities, labels, thresholds):
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        preds = (probabilities[:, 1] > t).long()
        f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1

def safe_auc_score(y_true, y_probs):
    try:
        return roc_auc_score(y_true.cpu(), y_probs[:, 1].cpu())
    except ValueError:
        return float('0.0')