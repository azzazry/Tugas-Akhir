import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class HeteroGraphSAGE(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=2, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projections untuk setiap node type
        self.user_proj = Linear(6, hidden_dim)  # 6 user features
        self.pc_proj = Linear(4, hidden_dim)    # 4 pc features  
        self.url_proj = Linear(3, hidden_dim)   # 3 url features
        
        # Heterogeneous convolution layers - HANYA UNIDIRECTIONAL
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'uses', 'pc'): SAGEConv(hidden_dim, hidden_dim),
                ('user', 'visits', 'url'): SAGEConv(hidden_dim, hidden_dim),
                # HAPUS reverse edges untuk menghindari bias
            }, aggr='sum')
            self.convs.append(conv)
        
        # Resource aggregation layers - untuk handle resource yang tidak ter-update
        self.pc_aggregator = nn.Linear(hidden_dim, hidden_dim)
        self.url_aggregator = nn.Linear(hidden_dim, hidden_dim)
            
        # Classifier untuk user nodes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
    def forward(self, x_dict, edge_index_dict):
        # Project input features
        x_dict = {
            'user': self.user_proj(x_dict['user']),
            'pc': self.pc_proj(x_dict['pc']), 
            'url': self.url_proj(x_dict['url'])
        }
        
        # Store initial embeddings
        initial_x_dict = {k: v.clone() for k, v in x_dict.items()}
        
        # Apply heterogeneous convolutions - HANYA original edges
        for i, conv in enumerate(self.convs):
            # Hanya gunakan edge yang ada di input (unidirectional)
            edge_index_dict_filtered = {}
            if ('user', 'uses', 'pc') in edge_index_dict:
                edge_index_dict_filtered[('user', 'uses', 'pc')] = edge_index_dict[('user', 'uses', 'pc')]
            if ('user', 'visits', 'url') in edge_index_dict:
                edge_index_dict_filtered[('user', 'visits', 'url')] = edge_index_dict[('user', 'visits', 'url')]
            
            # Apply convolution
            x_dict_new = conv(x_dict, edge_index_dict_filtered)
            
            # Update hanya user embeddings (yang mendapat message passing)
            if 'user' in x_dict_new:
                x_dict['user'] = F.relu(x_dict_new['user'])
            
            # Untuk PC dan URL yang tidak mendapat update dari conv, 
            # kita aggregate dengan embeddings sebelumnya
            if i == 0:  # Hanya di layer pertama
                x_dict['pc'] = F.relu(self.pc_aggregator(initial_x_dict['pc']))
                x_dict['url'] = F.relu(self.url_aggregator(initial_x_dict['url']))
            
        # Classify user nodes
        user_out = self.classifier(x_dict['user'])
        return user_out