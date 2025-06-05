import torch
import pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from collections import defaultdict
import numpy as np

from models.graphsvx import GraphSVX 
from models.graphsage import HeteroGraphSAGE

def explain_insider_predictions():
    """
    Menjelaskan prediksi insider threat menggunakan GraphSVX pada homogeneous graph
    yang dikonversi dari heterogeneous graph
    """
    
    # Load preprocessed hetero data
    data = torch.load('data/data_graph.pt', weights_only=False)

    # Validasi fitur user
    if data['user'].x is None:
        print("Warning: user features missing, creating dummy features")
        num_users = data['user'].num_nodes
        data['user'].x = torch.randn(num_users, 6)

    # Load trained model
    model = HeteroGraphSAGE(hidden_dim=64, out_dim=2, num_layers=2)
    model.load_state_dict(torch.load('result/logs/insider_threat_graphsage.pt'))
    model.eval()

    # Load evaluation results
    with open('result/logs/evaluation_results.pkl', 'rb') as f:
        eval_results = pickle.load(f)

    optimal_threshold = eval_results.get('optimal_threshold', 0.5)

    # Prepare edge dict for model input
    expected_edge_types = [('user', 'uses', 'pc'), ('user', 'visits', 'url')]
    filtered_edge_index_dict = {
        etype: data.edge_index_dict[etype]
        for etype in expected_edge_types if etype in data.edge_index_dict
    }

    # Get predictions
    with torch.no_grad():
        out = model(data.x_dict, filtered_edge_index_dict)
        val_mask = data['user'].val_mask
        val_out = out[val_mask]
        val_probs = torch.softmax(val_out, dim=1)

    insider_probs = val_probs[:, 1].cpu().numpy()
    val_indices = torch.where(val_mask)[0].cpu().numpy()

    # Filter high-risk users
    high_risk_mask = insider_probs > optimal_threshold
    high_risk_user_ids = val_indices[high_risk_mask]

    print(f"Found {len(high_risk_user_ids)} high-risk users for GraphSVX explanation")

    if len(high_risk_user_ids) == 0:
        print("No high-risk users found. Exiting explanation process.")
        return {}

    # --------------------------------------
    # Create homogeneous user-user graph
    # --------------------------------------
    user_x = data['user'].x
    num_users = user_x.size(0)
    
    # Create user-user connections based on shared resources
    resource_to_users = defaultdict(set)

    # Process User-PC edges
    if ('user', 'uses', 'pc') in data.edge_index_dict:
        edge_index_pc = data.edge_index_dict[('user', 'uses', 'pc')]
        for src, dst in edge_index_pc.t():
            resource_to_users[('pc', int(dst))].add(int(src))

    # Process User-URL edges  
    if ('user', 'visits', 'url') in data.edge_index_dict:
        edge_index_url = data.edge_index_dict[('user', 'visits', 'url')]
        for src, dst in edge_index_url.t():
            resource_to_users[('url', int(dst))].add(int(src))

    # Create user-user edges based on shared resources
    user_edges = set()
    for users in resource_to_users.values():
        users = list(users)
        if len(users) > 1:  # Only if multiple users share the resource
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    if users[i] < num_users and users[j] < num_users:  # Safety check
                        user_edges.add((users[i], users[j]))

    print(f"Created {len(user_edges)} user-user edges based on shared resources")

    # Convert to tensor and make undirected
    if user_edges:
        edge_index = torch.tensor(list(user_edges), dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)
    else:
        print("Warning: No user-user edges found, creating empty edge_index")
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create homogeneous graph data
    homo_data = Data(x=user_x, edge_index=edge_index)
    
    print(f"Homogeneous graph: {homo_data.num_nodes} nodes, {homo_data.num_edges} edges")

    # --------------------------------------
    # Wrapper model for GraphSVX
    # --------------------------------------
    class HomogeneousModelWrapper(torch.nn.Module):
        """
        Wrapper untuk model heterogeneous agar bisa digunakan dengan homogeneous graph
        """
        def __init__(self, hetero_model, filtered_edge_dict):
            super().__init__()
            self.hetero_model = hetero_model
            self.filtered_edge_dict = filtered_edge_dict
            
        def forward(self, x, edge_index=None):
            # x adalah node features dari homogeneous graph (user features)
            # Kita perlu rekonstruksi format heterogeneous untuk model asli
            x_dict = {'user': x}
            
            # Gunakan edge_index_dict asli untuk prediksi
            out = self.hetero_model(x_dict, self.filtered_edge_dict)
            return out

    # Create wrapper model
    wrapped_model = HomogeneousModelWrapper(model, filtered_edge_index_dict)
    wrapped_model.eval()

    # --------------------------------------
    # Initialize GraphSVX explainer
    # --------------------------------------
    explainer = GraphSVX(
        model=wrapped_model,
        data=homo_data,
        num_samples=1000,
        hops=2  # Reduced from 3 to 2 for efficiency
    )

    # Helper function untuk mendapatkan neighbors dengan aman
    def safe_get_neighbors(edge_tensor, source_idx):
        """Safely get neighbors from edge tensor"""
        if edge_tensor.dim() != 2 or edge_tensor.size(0) != 2:
            print(f"Warning: Invalid edge_tensor shape {edge_tensor.shape}")
            return []
        
        src_nodes, dst_nodes = edge_tensor
        mask = src_nodes == source_idx
        neighbors = dst_nodes[mask].cpu().numpy().tolist()
        return neighbors

    # Generate explanations
    explanations = {}
    max_explanations = min(10, len(high_risk_user_ids))  # Limit to 10 users
    
    for i, user_idx in enumerate(high_risk_user_ids[:max_explanations]):
        print(f"Explaining user {user_idx} ({i+1}/{max_explanations}) using GraphSVX...")
        
        try:
            # Ensure user_idx is within valid range
            if user_idx >= num_users:
                print(f"Warning: user_idx {user_idx} exceeds number of users {num_users}")
                continue
                
            explanation = explainer.explain_node(
                node_idx=int(user_idx),
                target_class=1  # Insider class
            )

            # Get resources accessed by this user
            neighboring_pcs = []
            accessed_urls = []

            if ('user', 'uses', 'pc') in filtered_edge_index_dict:
                uses_edges = filtered_edge_index_dict[('user', 'uses', 'pc')]
                neighboring_pcs = safe_get_neighbors(uses_edges, user_idx)

            if ('user', 'visits', 'url') in filtered_edge_index_dict:
                visits_edges = filtered_edge_index_dict[('user', 'visits', 'url')]
                accessed_urls = safe_get_neighbors(visits_edges, user_idx)

            # Store explanation results
            explanations[user_idx] = {
                'user_id': int(user_idx),
                'risk_probability': float(insider_probs[high_risk_mask][i]),
                'shapley_values': explanation.get('node_importance', []),
                'important_subgraph': explanation.get('subgraph_nodes', []),
                'feature_importance': explanation.get('feature_importance', []),
                'neighboring_pcs': neighboring_pcs,
                'accessed_urls': accessed_urls,
                'original_score': explanation.get('original_score', 0.0),
                'explanation_quality': {
                    'num_important_nodes': len(explanation.get('subgraph_nodes', [])),
                    'max_shapley_value': max(explanation.get('node_importance', [0])),
                    'min_shapley_value': min(explanation.get('node_importance', [0]))
                }
            }

            print(f"  - Risk probability: {insider_probs[high_risk_mask][i]:.4f}")
            print(f"  - Important subgraph size: {len(explanation.get('subgraph_nodes', []))}")
            print(f"  - Connected PCs: {len(neighboring_pcs)}")
            print(f"  - Visited URLs: {len(accessed_urls)}")

        except Exception as e:
            print(f"Error explaining user {user_idx}: {str(e)}")
            continue

    # Save explanation results
    output_file = 'result/logs/graphsvx_explanations.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(explanations, f)

    print(f"\nGraphSVX explanation completed for {len(explanations)} users")
    print(f"Results saved to: {output_file}")
    
    # Print summary statistics
    if explanations:
        avg_risk = np.mean([exp['risk_probability'] for exp in explanations.values()])
        avg_subgraph_size = np.mean([len(exp['important_subgraph']) for exp in explanations.values()])
        print(f"Average risk probability: {avg_risk:.4f}")
        print(f"Average important subgraph size: {avg_subgraph_size:.2f}")
    
    return explanations

if __name__ == "__main__":
    explain_insider_predictions()