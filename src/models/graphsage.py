import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class GraphSAGE(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=2, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        self.user = Linear(6, hidden_dim)
        self.pc = Linear(4, hidden_dim)
        self.url = Linear(3, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'interacts', 'user'): SAGEConv(hidden_dim, hidden_dim),
                ('user', 'uses', 'pc'): SAGEConv(hidden_dim, hidden_dim),
                ('user', 'visits', 'url'): SAGEConv(hidden_dim, hidden_dim)
            }, aggr='sum')
            self.convs.append(conv)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'user': self.user(x_dict['user']),
            'pc': self.pc(x_dict['pc']),
            'url': self.url(x_dict['url']),
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        user_embeddings = x_dict['user']
        out = self.classifier(user_embeddings)
        return out