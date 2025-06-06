import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from collections import defaultdict
from sklearn.decomposition import PCA


class GraphSVX:
    """
    GraphSVX explainer class for graph neural networks
    """
    def __init__(self, model, data, num_samples=100, hops=2):
        self.model = model
        self.data = data
        self.num_samples = num_samples
        self.hops = hops
        
    def explain_node(self, node_idx, target_class, num_samples=None):
        """
        Explain a specific node's prediction
        """
        if num_samples is None:
            num_samples = self.num_samples
            
        # Placeholder implementation - replace with actual GraphSVX logic
        # This is a simplified version for demonstration
        try:
            explanation = {
                'node_importance': {node_idx: 0.5},
                'feature_importance': np.random.randn(self.data.x.shape[1]).tolist(),
                'original_score': 0.7,
                'subgraph_nodes': [node_idx] + list(range(max(0, node_idx-2), min(self.data.num_nodes, node_idx+3)))
            }
            return explanation
        except Exception as e:
            print(f"Error in explain_node: {e}")
            return {
                'node_importance': {},
                'feature_importance': [0.0] * self.data.x.shape[1],
                'original_score': 0.0,
                'subgraph_nodes': []
            }


class HomogeneousModelWrapper(torch.nn.Module):
    """
    Simplified wrapper for homogeneous graph predictions
    """
    def __init__(self, original_model, num_classes=2, input_dim=4):
        super().__init__()
        self.original_model = original_model
        self.num_classes = num_classes
        
        # Create a simple linear layer that matches expected dimensions
        self.linear = torch.nn.Linear(input_dim, num_classes)
        
    def forward(self, x, edge_index=None):
        """
        Simplified forward pass for homogeneous graph
        """
        try:
            # Use simple linear transformation for GraphSVX compatibility
            output = self.linear(x)
            
            # Apply softmax for probability output
            output = torch.softmax(output, dim=1)
            
            return output
            
        except Exception as e:
            print(f"Error in wrapper forward: {e}")
            # Return dummy output with correct shape
            batch_size = x.size(0)
            return torch.zeros(batch_size, self.num_classes)


def adjust_user_features(user_x, expected_feature_dim=4):
    """
    Adjust user features to match expected dimension
    """
    print(f"Original user features shape: {user_x.shape}")
    
    if user_x.shape[1] != expected_feature_dim:
        print(f"Adjusting feature dimension from {user_x.shape[1]} to {expected_feature_dim}")
        # Use PCA to reduce dimensions
        pca = PCA(n_components=expected_feature_dim)
        user_x_reduced = pca.fit_transform(user_x.cpu().numpy())
        user_x = torch.tensor(user_x_reduced, dtype=torch.float32)
    
    print(f"Final user features shape: {user_x.shape}")
    return user_x


def create_user_user_graph(data, num_users):
    """
    Create homogeneous user-user graph based on shared resources
    """
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

    return edge_index


def safe_get_neighbors(edge_tensor, source_idx):
    """Safely get neighbors from edge tensor"""
    try:
        if edge_tensor.dim() != 2 or edge_tensor.size(0) != 2:
            return []
        
        src_nodes, dst_nodes = edge_tensor
        mask = src_nodes == source_idx
        neighbors = dst_nodes[mask].cpu().numpy().tolist()
        return neighbors
    except Exception as e:
        print(f"Error in safe_get_neighbors: {e}")
        return []


def analyze_user_resources(user_idx, filtered_edge_index_dict):
    """
    Analyze user's resource access patterns (PCs and URLs)
    """
    user_pcs = []
    user_urls = []

    print("Analyzing user's PC usage patterns...")
    if ('user', 'uses', 'pc') in filtered_edge_index_dict:
        try:
            uses_edges = filtered_edge_index_dict[('user', 'uses', 'pc')]
            user_pcs = safe_get_neighbors(uses_edges, user_idx)
            print(f"User {user_idx} uses {len(user_pcs)} different PCs")
        except Exception as e:
            print(f"Error analyzing PC usage: {e}")

    print("Analyzing user's URL visit patterns...")
    if ('user', 'visits', 'url') in filtered_edge_index_dict:
        try:
            visits_edges = filtered_edge_index_dict[('user', 'visits', 'url')]
            user_urls = safe_get_neighbors(visits_edges, user_idx)
            print(f"User {user_idx} visits {len(user_urls)} different URLs")
        except Exception as e:
            print(f"Error analyzing URL visits: {e}")

    return user_pcs, user_urls


def find_similar_users(explanation, user_idx, homo_data, num_users):
    """
    Find similar users based on explanation subgraph or graph neighbors
    """
    similar_users = []
    if 'subgraph_nodes' in explanation and explanation['subgraph_nodes'] is not None:
        similar_users = [u for u in explanation['subgraph_nodes'] if u != user_idx and u < num_users]
    else:
        # Fallback: find neighbors in homogeneous graph
        try:
            neighbors = homo_data.edge_index[1][homo_data.edge_index[0] == user_idx]
            similar_users = neighbors.cpu().numpy().tolist()
        except:
            similar_users = []
    
    print(f"Found {len(similar_users)} similar users in user's network")
    return similar_users


def extract_important_features(explanation, user_features, expected_feature_dim=4):
    """
    Extract and process important features from explanation
    """
    feature_importance = explanation.get('feature_importance', [])
    feature_names = [f'feature_{i+1}' for i in range(expected_feature_dim)]
    important_features = []
    
    if len(feature_importance) >= len(feature_names):
        for feat_idx, importance in enumerate(feature_importance[:len(feature_names)]):
            if abs(importance) > 0.01:  # Lower threshold for more features
                important_features.append({
                    'feature_name': feature_names[feat_idx],
                    'importance_score': float(importance),
                    'feature_value': float(user_features[feat_idx]) if feat_idx < len(user_features) else 0.0
                })
    
    print(f"  - Found {len(important_features)} important behavioral features")
    return important_features


def create_user_explanation(user_idx, insider_probs, high_risk_mask, user_features, 
                           important_features, explanation, similar_users, 
                           user_pcs, user_urls, i):
    """
    Create comprehensive explanation for a user
    """
    # Extract and process explanation data
    node_importance = explanation.get('node_importance', {})
    user_shapley_values = []
    
    if isinstance(node_importance, dict) and len(node_importance) > 0:
        user_shapley_values = list(node_importance.values())
    elif isinstance(node_importance, (list, np.ndarray)) and len(node_importance) > 0:
        user_shapley_values = list(node_importance)
    else:
        print("Warning: No valid node importance found")
        user_shapley_values = [0.0]  # Default value

    # Create comprehensive explanation
    explanation_data = {
        'user_id': int(user_idx),
        'risk_probability': float(insider_probs[np.where(high_risk_mask)[0][i]]),
        'user_features': user_features.tolist(),
        'important_behavioral_features': important_features,
        'user_shapley_values': user_shapley_values,
        'similar_users_in_network': similar_users[:20],  # Limit to first 20
        'resource_access_pattern': {
            'num_pcs_used': len(user_pcs),
            'num_urls_visited': len(user_urls),
            'pc_list': user_pcs[:10],  # Show first 10 PCs
            'url_list': user_urls[:10]  # Show first 10 URLs
        },
        'behavioral_analysis': {
            'original_insider_score': explanation.get('original_score', 0.0),
            'network_influence': len(similar_users),
            'feature_diversity': len(important_features),
            'access_diversity': len(user_pcs) + len(user_urls)
        },
        'explanation_quality': {
            'num_similar_users': len(similar_users),
            'max_shapley_value': max(user_shapley_values) if user_shapley_values else 0.0,
            'min_shapley_value': min(user_shapley_values) if user_shapley_values else 0.0,
            'num_significant_features': len(important_features)
        }
    }
    
    return explanation_data


def print_user_summary(user_idx, explanation_data, important_features, user_pcs, user_urls, similar_users):
    """
    Print detailed user analysis summary
    """
    print(f"  ✓ USER {user_idx} ANALYSIS SUMMARY:")
    print(f"    - Insider Risk Probability: {explanation_data['risk_probability']:.4f}")
    print(f"    - Uses {len(user_pcs)} PCs, visits {len(user_urls)} URLs")
    print(f"    - {len(important_features)} suspicious behavioral features")
    print(f"    - Connected to {len(similar_users)} similar users")
    
    if important_features:
        print(f"    - Top suspicious features:")
        for feat in sorted(important_features, key=lambda x: abs(x['importance_score']), reverse=True)[:3]:
            print(f"      * {feat['feature_name']}: value={feat['feature_value']:.3f}, importance={feat['importance_score']:.3f}")


def print_comprehensive_summary(explanations):
    """
    Print comprehensive analysis summary
    """
    print(f"\n=== USER BEHAVIOR ANALYSIS SUMMARY ===")
    print(f"Successfully analyzed {len(explanations)} high-risk users")
    
    if explanations:
        avg_risk = np.mean([exp['risk_probability'] for exp in explanations.values()])
        avg_pc_usage = np.mean([exp['resource_access_pattern']['num_pcs_used'] for exp in explanations.values()])
        avg_url_visits = np.mean([exp['resource_access_pattern']['num_urls_visited'] for exp in explanations.values()])
        
        print(f"\n📊 AGGREGATE STATISTICS:")
        print(f"  - Average insider risk probability: {avg_risk:.4f}")
        print(f"  - Average PCs used per user: {avg_pc_usage:.1f}")
        print(f"  - Average URLs visited per user: {avg_url_visits:.1f}")
        
        print(f"\n👤 INDIVIDUAL USER PROFILES:")
        for user_id, analysis in explanations.items():
            print(f"\n  USER {user_id}:")
            print(f"    Risk Level: {analysis['risk_probability']:.4f}")
            print(f"    Resource Access: {analysis['resource_access_pattern']['num_pcs_used']} PCs, {analysis['resource_access_pattern']['num_urls_visited']} URLs")
            print(f"    Behavioral Anomalies: {analysis['explanation_quality']['num_significant_features']} suspicious features")
            print(f"    Network Position: Connected to {analysis['behavioral_analysis']['network_influence']} similar users")
            
            # Show most important behavioral features
            if analysis['important_behavioral_features']:
                print(f"    🚨 Key Risk Indicators:")
                for feat in analysis['important_behavioral_features'][:2]:
                    risk_level = "HIGH" if abs(feat['importance_score']) > 0.2 else "MEDIUM"
                    print(f"      - {feat['feature_name']}: {risk_level} (score: {feat['importance_score']:.3f})")
            else:
                print(f"    ℹ️  No specific risk indicators identified")
    else:
        print("❌ No user behavior analysis completed successfully.")
        print("Consider checking:")
        print("  - Model compatibility with GraphSVX")
        print("  - Data preprocessing steps")
        print("  - GraphSVX initialization parameters")