# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.metrics import roc_auc_score
from models.graphsage import InsiderThreatGraphSAGE

def train_insider_threat_model():
    # Load preprocessed graph data
    with open('graph_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Inisialisasi model dan optimizer
    model = InsiderThreatGraphSAGE(data.metadata())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Tangani class imbalance: hanya 2 user positif dari 1000 user
    pos_weight = torch.tensor([498.0])  # 998 / 2
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        out = model(data.x_dict, data.edge_index_dict)
        loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()

        # Validasi setiap 50 epoch
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x_dict, data.edge_index_dict)
                val_pred = torch.sigmoid(val_out[data.val_mask])
                val_auc = roc_auc_score(
                    data.y[data.val_mask].cpu(),
                    val_pred.cpu()
                )
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

    # Simpan model terlatih
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_auc': val_auc,
    }, 'insider_threat_graphsage.pt')
