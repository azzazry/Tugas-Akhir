# models.py

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv


class InsiderThreatGraphSAGE(nn.Module):
    def __init__(self, metadata, hidden_dim=128, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Heterogeneous convolutions untuk User-PC-URL
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {
                edge_type: SAGEConv(
                    hidden_dim if i > 0 else -1,
                    hidden_dim
                )
                for edge_type in metadata[1]  # edge types
            }
            self.convs.append(HeteroConv(conv_dict))

        # Classifier untuk insider threat detection
        self.classifier = nn.Linear(hidden_dim, 2)  # binary classification
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                key: self.dropout(torch.relu(x))
                for key, x in x_dict.items()
            }

        # Focus pada user nodes untuk classification
        user_embeddings = x_dict['user']
        return self.classifier(user_embeddings)
