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
        
        # Heterogeneous convolution layers - tambahkan reverse edges
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'uses', 'pc'): SAGEConv(hidden_dim, hidden_dim),
                ('pc', 'used_by', 'user'): SAGEConv(hidden_dim, hidden_dim),
                ('user', 'visits', 'url'): SAGEConv(hidden_dim, hidden_dim),
                ('url', 'visited_by', 'user'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='sum')
            self.convs.append(conv)
            
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
        
        # Create reverse edges untuk bidirectional message passing
        edge_index_dict_full = {}
        
        # Original edges
        if ('user', 'uses', 'pc') in edge_index_dict:
            edge_index_dict_full[('user', 'uses', 'pc')] = edge_index_dict[('user', 'uses', 'pc')]
            # Reverse edge: pc -> user
            original_edge = edge_index_dict[('user', 'uses', 'pc')]
            edge_index_dict_full[('pc', 'used_by', 'user')] = torch.stack([original_edge[1], original_edge[0]])
            
        if ('user', 'visits', 'url') in edge_index_dict:
            edge_index_dict_full[('user', 'visits', 'url')] = edge_index_dict[('user', 'visits', 'url')]
            # Reverse edge: url -> user
            original_edge = edge_index_dict[('user', 'visits', 'url')]
            edge_index_dict_full[('url', 'visited_by', 'user')] = torch.stack([original_edge[1], original_edge[0]])
        
        # Apply heterogeneous convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict_full)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            
        # Classify user nodes
        user_out = self.classifier(x_dict['user'])
        return user_out