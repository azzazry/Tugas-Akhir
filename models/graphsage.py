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
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'uses', 'pc'): SAGEConv(hidden_dim, hidden_dim),
                ('user', 'visits', 'url'): SAGEConv(hidden_dim, hidden_dim),
                # ('pc', 'accesses', 'url'): SAGEConv(hidden_dim, hidden_dim),  # TAMBAHAN INI SAJA - BELUM ADA DATA
            }, aggr='sum')
            self.convs.append(conv)
        
        # Resource aggregation layers
        self.pc_aggregator = nn.Linear(hidden_dim, hidden_dim)
        self.url_aggregator = nn.Linear(hidden_dim, hidden_dim)
        
        self.user_context_agg = nn.Linear(hidden_dim * 3, hidden_dim)
        
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
        # TAMBAHAN: Store original user embeddings untuk context aggregation
        original_user_emb = x_dict['user'].clone()
        
        # Apply heterogeneous convolutions - HANYA original edges
        for i, conv in enumerate(self.convs):
            # Hanya gunakan edge yang ada di input (unidirectional)
            edge_index_dict_filtered = {}
            if ('user', 'uses', 'pc') in edge_index_dict:
                edge_index_dict_filtered[('user', 'uses', 'pc')] = edge_index_dict[('user', 'uses', 'pc')]
            if ('user', 'visits', 'url') in edge_index_dict:
                edge_index_dict_filtered[('user', 'visits', 'url')] = edge_index_dict[('user', 'visits', 'url')]
            # TAMBAHAN: Include directed edge jika ada
            # if ('pc', 'accesses', 'url') in edge_index_dict:
            #     edge_index_dict_filtered[('pc', 'accesses', 'url')] = edge_index_dict[('pc', 'accesses', 'url')]
            
            # Apply convolution
            x_dict_new = conv(x_dict, edge_index_dict_filtered)
            
            for node_type in ['pc', 'url']:
                if node_type in x_dict_new:
                    if i == 0:
                        x_dict[node_type] = F.relu(x_dict_new[node_type])
                    else:
                        x_dict[node_type] = F.relu(x_dict_new[node_type] + x_dict[node_type])
            
            # Update hanya user embeddings (yang mendapat message passing)
            if 'user' in x_dict_new:
                x_dict['user'] = F.relu(x_dict_new['user'])
            
            # Untuk PC dan URL yang tidak mendapat update dari conv, 
            # kita aggregate dengan embeddings sebelumnya
            if i == 0:  # Hanya di layer pertama
                if 'pc' not in x_dict_new:
                    x_dict['pc'] = F.relu(self.pc_aggregator(initial_x_dict['pc']))
                if 'url' not in x_dict_new:
                    x_dict['url'] = F.relu(self.url_aggregator(initial_x_dict['url']))
        
        user_with_context = self.aggregate_resource_context(
            original_user_emb, x_dict['pc'], x_dict['url'], edge_index_dict
        )
        
        # Classify user nodes dengan enriched embeddings
        user_out = self.classifier(user_with_context)
        return user_out

    def aggregate_resource_context(self, user_emb, pc_emb, url_emb, edge_index_dict):
        """
        Aggregate context dari enriched PC dan URL kembali ke user
        """
        batch_size = user_emb.shape[0]
        
        # Initialize context tensors
        user_pc_context = torch.zeros_like(user_emb)
        user_url_context = torch.zeros_like(user_emb)
        
        # Aggregate PC context ke user
        if ('user', 'uses', 'pc') in edge_index_dict:
            user_pc_edges = edge_index_dict[('user', 'uses', 'pc')]
            user_indices = user_pc_edges[0]  # source (user)
            pc_indices = user_pc_edges[1]    # destination (pc)
            
            # Untuk setiap user, aggregate context dari PC yang digunakan
            for i in range(len(user_indices)):
                user_idx = user_indices[i]
                pc_idx = pc_indices[i]
                user_pc_context[user_idx] += pc_emb[pc_idx]
        
        # Aggregate URL context ke user
        if ('user', 'visits', 'url') in edge_index_dict:
            user_url_edges = edge_index_dict[('user', 'visits', 'url')]
            user_indices = user_url_edges[0]  # source (user)
            url_indices = user_url_edges[1]   # destination (url)
            
            # Untuk setiap user, aggregate context dari URL yang dikunjungi
            for i in range(len(user_indices)):
                user_idx = user_indices[i]
                url_idx = url_indices[i]
                user_url_context[user_idx] += url_emb[url_idx]
        
        # Combine user original embedding dengan resource context
        user_combined = torch.cat([user_emb, user_pc_context, user_url_context], dim=1)
        user_enriched = self.user_context_agg(user_combined)
        
        return user_enriched