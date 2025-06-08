import torch
import torch.nn.functional as F
import pickle
from models.graphsage import GraphSAGE
from torch_geometric.data import HeteroData

class GraphSVXExplainer:
    def __init__(self, model, node_type='user', num_samples=20):
        self.model = model
        self.node_type = node_type
        self.num_samples = num_samples

    def explain(self, x_dict, edge_index_dict, target_node_idx):
        base_pred = self.model(x_dict, edge_index_dict)[target_node_idx]
        base_pred = F.softmax(base_pred, dim=0)
        
        x_orig = x_dict[self.node_type][target_node_idx].clone()
        num_feats = x_orig.shape[0]
        
        importances = torch.zeros(num_feats)
        for i in range(self.num_samples):
            mask = torch.bernoulli(torch.full_like(x_orig, 0.5))
            x_masked = x_orig * mask
            x_dict_mod = x_dict.copy()
            x_dict_mod[self.node_type] = x_dict[self.node_type].clone()
            x_dict_mod[self.node_type][target_node_idx] = x_masked
            
            pred = self.model(x_dict_mod, edge_index_dict)[target_node_idx]
            pred = F.softmax(pred, dim=0)
            delta = base_pred - pred
            importances += torch.abs(delta[1]) * mask
        
        return (importances / self.num_samples).cpu().numpy()

def explain_user_anomalies():
    data: HeteroData = torch.load('data/data_graph.pt')
    model = GraphSAGE(hidden_dim=64, out_dim=2, num_layers=2)
    model.load_state_dict(torch.load('result/logs/insider_threat_graphsage.pt'))
    model.eval()

    x_dict = {
        'user': data['user'].x,
        'pc': data['pc'].x,
        'url': data['url'].x
    }
    edge_index_dict = {
        ('user', 'uses', 'pc'): data['user', 'uses', 'pc'].edge_index,
        ('user', 'visits', 'url'): data['user', 'visits', 'url'].edge_index,
        ('user', 'interacts', 'user'): data['user', 'interacts', 'user'].edge_index
    }

    print("Running inference to detect insider users...")
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        predictions = out.argmax(dim=1)
        insider_indices = (predictions == 1).nonzero(as_tuple=True)[0]

    print(f"Found {len(insider_indices)} users predicted as insiders.")

    # Ambil max 5 insiders untuk dijelaskan
    explain_indices = insider_indices[:5]

    explainer = GraphSVXExplainer(model=model, node_type='user', num_samples=30)
    explanations = {}

    for idx in explain_indices:
        importance = explainer.explain(x_dict, edge_index_dict, idx.item())
        explanations[idx.item()] = importance.tolist()
        print(f"User {idx.item()} → Feature importance: {importance.round(3)}")

    with open('result/logs/graphsvx_explanations.pkl', 'wb') as f:
        pickle.dump(explanations, f)

    print("GraphSVX explanations tersimpan!")

if __name__ == "__main__":
    explain_user_anomalies()