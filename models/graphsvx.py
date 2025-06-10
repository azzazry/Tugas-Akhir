import torch
import torch.nn.functional as F

class GraphSVXExplainer:
    def __init__(self, model, node_type='user', num_samples=20):
        self.model = model
        self.node_type = node_type
        self.num_samples = num_samples

    def explain(self, x_dict, edge_index_dict, target_node_idx):
        with torch.no_grad():
            base_pred = self.model(x_dict, edge_index_dict)[target_node_idx]
            base_pred = F.softmax(base_pred, dim=0)
            
            x_orig = x_dict[self.node_type][target_node_idx].clone()
            num_feats = x_orig.shape[0]
            
            importances = torch.zeros(num_feats)
            for i in range(self.num_samples):
                # Mask random features
                mask = torch.bernoulli(torch.full_like(x_orig, 0.5))
                x_masked = x_orig * mask
                
                # Copy x_dict dan ganti target node
                x_dict_mod = {k: v.clone() for k, v in x_dict.items()}
                x_dict_mod[self.node_type][target_node_idx] = x_masked
                
                # Prediksi dengan features yang dimasking
                pred = self.model(x_dict_mod, edge_index_dict)[target_node_idx]
                pred = F.softmax(pred, dim=0)
                
                # Hitung selisih prediksi (fokus pada class insider = 1)
                delta = base_pred - pred
                importances += torch.abs(delta[1]) * mask
            
            return (importances / self.num_samples).cpu().numpy()