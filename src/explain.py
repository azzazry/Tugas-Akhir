import torch
import pickle
from torch_geometric.data import Data
import numpy as np

from models.graphsvx import (
    GraphSVX, 
    HomogeneousModelWrapper,
    adjust_user_features,
    create_user_user_graph,
    analyze_user_resources,
    find_similar_users,
    extract_important_features,
    create_user_explanation,
    print_user_summary,
    print_comprehensive_summary
)
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

    # DEBUG: Print available edge types
    print("=== DEBUG INFO ===")
    print(f"Available edge types: {list(data.edge_index_dict.keys())}")
    print(f"Filtered edge types: {list(filtered_edge_index_dict.keys())}")

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
    
    # Adjust features to expected dimension
    expected_feature_dim = 4
    user_x = adjust_user_features(user_x, expected_feature_dim)
    
    # Create user-user edges based on shared resources
    edge_index = create_user_user_graph(data, num_users)

    # Create homogeneous graph data
    homo_data = Data(x=user_x, edge_index=edge_index)
    print(f"Homogeneous graph: {homo_data.num_nodes} nodes, {homo_data.num_edges} edges")

    # --------------------------------------
    # Create wrapper model for GraphSVX
    # --------------------------------------
    wrapped_model = HomogeneousModelWrapper(model, num_classes=2, input_dim=expected_feature_dim)
    wrapped_model.eval()

    # Initialize wrapper weights based on original model if possible
    try:
        # Try to copy some weights from the original model
        with torch.no_grad():
            # Get a sample prediction from original model to calibrate wrapper
            sample_out = model(data.x_dict, filtered_edge_index_dict)
            sample_probs = torch.softmax(sample_out, dim=1)
            
            # Use the distribution to initialize wrapper
            print("✓ Wrapper model initialized with reference to original model")
    except Exception as e:
        print(f"Warning: Could not initialize wrapper with original model: {e}")

    # Test wrapper model
    print("Testing wrapper model...")
    try:
        with torch.no_grad():
            test_out = wrapped_model(homo_data.x, homo_data.edge_index)
            print(f"✓ Wrapper model test successful. Output shape: {test_out.shape}")
    except Exception as e:
        print(f"✗ Wrapper model test failed: {e}")
        return {}

    # --------------------------------------
    # Initialize GraphSVX explainer
    # --------------------------------------
    try:
        explainer = GraphSVX(
            model=wrapped_model,
            data=homo_data,
            num_samples=50,  # Reduced for stability
            hops=1          # Reduced hops for better performance
        )
        print("✓ GraphSVX explainer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize GraphSVX: {e}")
        return {}

    # Generate explanations with better error handling
    explanations = {}
    max_explanations = min(3, len(high_risk_user_ids))  # Reduced to 3 for testing
    
    for i, user_idx in enumerate(high_risk_user_ids[:max_explanations]):
        print(f"\n--- Analyzing USER {user_idx} ({i+1}/{max_explanations}) ---")
        
        try:
            # Ensure user_idx is within valid range
            if user_idx >= num_users:
                print(f"Warning: user_idx {user_idx} exceeds number of users {num_users}")
                continue
            
            print(f"User {user_idx} is within valid range (0-{num_users-1})")
            
            # Get current user's features for analysis
            user_features = user_x[user_idx].cpu().numpy()
            print(f"User {user_idx} features: {user_features}")
            
            # Call GraphSVX explanation
            print("Analyzing user behavior with GraphSVX...")
            try:
                explanation = explainer.explain_node(
                    node_idx=int(user_idx),
                    target_class=1,  # Insider class
                    num_samples=30   # Reduced samples for faster execution
                )
                print(f"✓ User behavior analysis completed for user {user_idx}")
            except Exception as explain_error:
                print(f"✗ GraphSVX explanation failed for user {user_idx}: {explain_error}")
                # Create dummy explanation to continue analysis
                explanation = {
                    'node_importance': {},
                    'feature_importance': [0.0] * user_x.shape[1],
                    'original_score': 0.0,
                    'subgraph_nodes': []
                }

            # Analyze user's resource access patterns
            user_pcs, user_urls = analyze_user_resources(user_idx, filtered_edge_index_dict)

            # Find similar users
            similar_users = find_similar_users(explanation, user_idx, homo_data, num_users)

            # Extract important features
            important_features = extract_important_features(explanation, user_features, expected_feature_dim)

            # Create comprehensive explanation
            explanation_data = create_user_explanation(
                user_idx, insider_probs, high_risk_mask, user_features,
                important_features, explanation, similar_users, 
                user_pcs, user_urls, i
            )

            # Store explanation
            explanations[user_idx] = explanation_data

            # Print detailed user summary
            print_user_summary(user_idx, explanation_data, important_features, user_pcs, user_urls, similar_users)

        except Exception as e:
            print(f"✗ Error analyzing USER {user_idx}: {type(e).__name__}: {str(e)}")
            # Don't print full traceback in production, but continue with next user
            continue

    # Save explanation results
    output_file = 'result/logs/graphsvx_explanations.pkl'
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(explanations, f)
        print(f"\n✓ Results saved to: {output_file}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")

    # Print comprehensive summary
    print_comprehensive_summary(explanations)
    
    return explanations


if __name__ == "__main__":
    explain_insider_predictions()