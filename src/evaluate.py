# evaluate.py

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from models.graphsage_model import InsiderThreatGraphSAGE

def evaluate_insider_threat_model():
    # Load trained model checkpoint
    checkpoint = torch.load('insider_threat_graphsage.pt')

    # Load preprocessed graph data
    with open('graph_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Inisialisasi model dan muat bobot terlatih
    model = InsiderThreatGraphSAGE(data.metadata())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Inference untuk data test
    with torch.no_grad():
        test_out = model(data.x_dict, data.edge_index_dict)
        test_pred = torch.sigmoid(test_out[data.test_mask])
        test_labels = data.y[data.test_mask]

    # Hitung metrik evaluasi
    auc_score = roc_auc_score(test_labels.cpu(), test_pred.cpu())
    ap_score = average_precision_score(test_labels.cpu(), test_pred.cpu())

    # Cari threshold optimal berdasarkan F1-score
    precision, recall, thresholds = precision_recall_curve(
        test_labels.cpu(),
        test_pred.cpu()
    )
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Simpan hasil evaluasi ke file
    evaluation_results = {
        'auc_score': auc_score,
        'average_precision': ap_score,
        'optimal_threshold': optimal_threshold,
        'test_predictions': test_pred.cpu().numpy(),
        'test_labels': test_labels.cpu().numpy(),
        'precision_recall_curve': (precision, recall, thresholds)
    }

    with open('evaluation_results.pkl', 'wb') as f:
        pickle.dump(evaluation_results, f)

    # Print hasil evaluasi
    print(f"AUC Score           : {auc_score:.4f}")
    print(f"Average Precision   : {ap_score:.4f}")
    print(f"Optimal Threshold   : {optimal_threshold:.4f}")
