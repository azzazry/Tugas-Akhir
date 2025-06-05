import torch
import pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from collections import defaultdict

from models.graphsvx import GraphSVX 
from models.graphsage import HeteroGraphSAGE

def explain_insider_predictions():

    # Load preprocessed hetero data
    data = torch.load('data/data_graph.pt', weights_only=False)

    # Validasi fitur user (bisa ditambah buat pc/url kalo perlu)
    if data['user'].x is None:
        print("Warning: user features missing, create dummy features")
        num_users = data['user'].num_nodes
        data['user'].x = torch.randn(num_users, 6)

    # Load model trained
    model = HeteroGraphSAGE(hidden_dim=64, out_dim=2, num_layers=2)
    model.load_state_dict(torch.load('result/logs/insider_threat_graphsage.pt'))
    model.eval()

    # Load hasil evaluasi (untuk threshold optimal)
    with open('result/logs/evaluation_results.pkl', 'rb') as f:
        eval_results = pickle.load(f)

    optimal_threshold = eval_results.get('optimal_threshold', 0.5)

    # Buat filtered edge dict buat model input (user-pc, user-url)
    expected_edge_types = [('user', 'uses', 'pc'), ('user', 'visits', 'url')]
    filtered_edge_index_dict = {
        etype: data.edge_index_dict[etype]
        for etype in expected_edge_types if etype in data.edge_index_dict
    }

    with torch.no_grad():
        out = model(data.x_dict, filtered_edge_index_dict)
        val_mask = data['user'].val_mask
        val_out = out[val_mask]
        val_probs = torch.softmax(val_out, dim=1)

    insider_probs = val_probs[:, 1].cpu().numpy()
    val_indices = torch.where(val_mask)[0].cpu().numpy()

    # Filter user high-risk berdasarkan threshold
    high_risk_mask = insider_probs > optimal_threshold
    high_risk_user_ids = val_indices[high_risk_mask]

    print(f"Found {len(high_risk_user_ids)} high-risk users for GraphSVX explanation")

    # --------------------------------------
    # Buat graph homogen user-user dari resource (pc/url)
    # --------------------------------------
    user_x = data['user'].x
    resource_to_users = defaultdict(set)

    # User-PC edges
    edge_index_pc = data['user', 'uses', 'pc'].edge_index
    for src, dst in edge_index_pc.t():
        resource_to_users[('pc', int(dst))].add(int(src))

    # User-URL edges
    edge_index_url = data['user', 'visits', 'url'].edge_index
    for src, dst in edge_index_url.t():
        resource_to_users[('url', int(dst))].add(int(src))

    user_edges = set()
    for users in resource_to_users.values():
        users = list(users)
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user_edges.add((users[i], users[j]))

    if user_edges:
        edge_index = torch.tensor(list(user_edges), dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    homo_data = Data(x=user_x, edge_index=edge_index)

    # --------------------------------------
    # Init GraphSVX explainer dengan graph homogen user-user
    # --------------------------------------
    explainer = GraphSVX(
        model=model,
        data=homo_data,
        num_samples=1000,
        hops=3
    )

    # Helper function buat ambil neighbors aman dari edge_index (source node)
    def safe_get_neighbors(edge_tensor, source_idx):
        if edge_tensor.dim() != 2 or edge_tensor.size(0) != 2:
            print(f"Warning: Invalid edge_tensor shape {edge_tensor.shape}")
            return []
        src_nodes, dst_nodes = edge_tensor
        mask = src_nodes == source_idx
        return dst_nodes[mask].cpu().numpy().tolist()

    explanations = {}
    for i, user_idx in enumerate(high_risk_user_ids):
        if i >= 10:  # batasi 10 user biar gak berat
            break

        print(f"Explaining user {user_idx} using GraphSVX...")
        try:
            explanation = explainer.explain_node(
                node_idx=int(user_idx),
                target_class=1
            )

            # Ambil PC & URL yang diakses user
            neighboring_pcs = []
            accessed_urls = []

            if ('user', 'uses', 'pc') in filtered_edge_index_dict:
                uses_edges = filtered_edge_index_dict[('user', 'uses', 'pc')]
                neighboring_pcs = safe_get_neighbors(uses_edges, user_idx)

            if ('user', 'visits', 'url') in filtered_edge_index_dict:
                visits_edges = filtered_edge_index_dict[('user', 'visits', 'url')]
                accessed_urls = safe_get_neighbors(visits_edges, user_idx)

            explanations[user_idx] = {
                'user_id': int(user_idx),
                'risk_probability': float(insider_probs[high_risk_mask][i]),
                'shapley_values': explanation['node_importance'],
                'important_subgraph': explanation['subgraph_nodes'],
                'feature_importance': explanation['feature_importance'],
                'neighboring_pcs': neighboring_pcs,
                'accessed_urls': accessed_urls,
                'original_score': explanation['original_score']
            }

        except Exception as e:
            print(f"Error explaining user {user_idx}: {e}")
            continue

    # Simpan hasil interpretasi ke file
    with open('result/logs/graphsvx_explanations.pkl', 'wb') as f:
        pickle.dump(explanations, f)

    print(f"GraphSVX explanation completed for {len(explanations)} users")
    return explanations
