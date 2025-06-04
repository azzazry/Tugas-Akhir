# graphsvx_analysis.py

import torch
import pickle
import numpy as np
from models.explainability import GraphSVX  # Menggunakan model explainability yang telah dibuat
from models.graphsage_model import InsiderThreatGraphSAGE  # pastikan path ini sesuai struktur proyek

def explain_insider_predictions():
    # Load trained model checkpoint
    checkpoint = torch.load('insider_threat_graphsage.pt')

    # Load preprocessed graph data
    with open('graph_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Load evaluation results (berisi prediksi dan threshold optimal)
    with open('evaluation_results.pkl', 'rb') as f:
        eval_results = pickle.load(f)

    # Inisialisasi model dan muat bobot
    model = InsiderThreatGraphSAGE(data.metadata())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Identifikasi user dengan risiko tinggi berdasarkan prediksi dan threshold
    test_predictions = eval_results['test_predictions']
    optimal_threshold = eval_results['optimal_threshold']
    high_risk_mask = test_predictions > optimal_threshold
    high_risk_users = np.where(high_risk_mask)[0]

    # Inisialisasi GraphSVX explainer
    explainer = GraphSVX(
        model=model,
        data=data,
        num_samples=1000,    # untuk sampling nilai Shapley
        hops=3               # menggantikan max_subgraph_size
    )

    # Lakukan interpretasi untuk setiap user berisiko tinggi
    explanations = {}
    for user_idx in high_risk_users:
        explanation = explainer.explain_node(
            node_idx=user_idx,
            target_class=1  # Asumsi kelas 1 = insider threat
        )

        explanations[user_idx] = {
            'shapley_values': explanation['node_importance'],
            'important_subgraph': explanation['subgraph_nodes'],
            'feature_importance': explanation['feature_importance'],
            'neighboring_pcs': [],  # Kosongkan dulu, bisa diisi manual jika diperlukan
            'accessed_urls': []     # Kosongkan dulu, bisa diisi manual jika diperlukan
        }

    # Simpan hasil interpretasi ke file
    with open('graphsvx_explanations.pkl', 'wb') as f:
        pickle.dump(explanations, f)

    return explanations