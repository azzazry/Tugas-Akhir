import torch
import numpy as np
from sklearn.linear_model import LinearRegression
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
import warnings
warnings.filterwarnings('ignore')


class GraphSVX:
    """
    GraphSVX: Shapley Value Explanations for Graph Neural Networks
    
    A post-hoc local model-agnostic explanation method specifically designed for GNNs.
    GraphSVX captures the "fair" contribution of each feature and node towards 
    the explained prediction by constructing a surrogate model on a perturbed dataset.
    """
    
    def __init__(self, model, data, device='cpu', num_samples=100, hops=2):
        """
        Initialize GraphSVX explainer
        
        Args:
            model: Trained GNN model to explain
            data: Graph data (torch_geometric Data object)
            device: Device to run computations on
            num_samples: Number of samples for approximation
            hops: Number of hops for k-hop subgraph
        """
        self.model = model
        self.data = data
        self.device = device
        self.num_samples = num_samples
        self.hops = hops
        self.model.eval()
        
    def explain_node(self, node_idx, target_class=None):
        """
        Explain prediction for a specific node
        
        Args:
            node_idx: Index of node to explain
            target_class: Target class to explain (if None, uses predicted class)
            
        Returns:
            dict: Explanation containing node and feature importances
        """
        # Get k-hop subgraph around target node
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.hops, self.data.edge_index, 
            relabel_nodes=True, num_nodes=self.data.x.size(0)
        )
        
        # Extract subgraph data
        subgraph_x = self.data.x[subset]
        subgraph_data = self.data.__class__(
            x=subgraph_x,
            edge_index=edge_index,
            batch=torch.zeros(len(subset), dtype=torch.long)
        )
        
        # Get original prediction
        with torch.no_grad():
            original_pred = self.model(self.data.x, self.data.edge_index)[node_idx]
            if target_class is None:
                target_class = original_pred.argmax().item()
            original_score = F.softmax(original_pred, dim=0)[target_class].item()
        
        # Generate coalition samples
        coalitions, predictions = self._generate_coalitions_and_predictions(
            subgraph_data, mapping, node_idx, target_class
        )
        
        # Compute Shapley values
        node_shapley = self._compute_shapley_values(coalitions, predictions, len(subset))
        feature_shapley = self._compute_feature_shapley_values(
            subgraph_data, mapping, node_idx, target_class
        )
        
        return {
            'node_idx': node_idx,
            'target_class': target_class,
            'original_score': original_score,
            'node_importance': dict(zip(subset.tolist(), node_shapley)),
            'feature_importance': feature_shapley,
            'subgraph_nodes': subset.tolist()
        }
    
    def _generate_coalitions_and_predictions(self, subgraph_data, mapping, target_node, target_class):
        """Generate coalitions and compute predictions"""
        num_nodes = subgraph_data.x.size(0)
        coalitions = []
        predictions = []
        
        # Generate random coalitions
        for _ in range(self.num_samples):
            # Random subset of nodes (coalition)
            coalition_size = np.random.randint(0, num_nodes + 1)
            if coalition_size == 0:
                coalition = []
            else:
                coalition = np.random.choice(num_nodes, coalition_size, replace=False).tolist()
            
            coalitions.append(coalition)
            
            # Create perturbed graph with only coalition nodes
            pred_score = self._predict_with_coalition(
                subgraph_data, coalition, mapping, target_node, target_class
            )
            predictions.append(pred_score)
        
        # Add empty and full coalitions
        coalitions.append([])  # Empty coalition
        coalitions.append(list(range(num_nodes)))  # Full coalition
        
        predictions.append(self._predict_with_coalition(
            subgraph_data, [], mapping, target_node, target_class
        ))
        predictions.append(self._predict_with_coalition(
            subgraph_data, list(range(num_nodes)), mapping, target_node, target_class
        ))
        
        return coalitions, predictions
    
    def _predict_with_coalition(self, subgraph_data, coalition, mapping, target_node, target_class):
        """Make prediction with specific node coalition"""
        if len(coalition) == 0:
            # Empty coalition - use baseline (mean features)
            perturbed_x = torch.zeros_like(subgraph_data.x)
            perturbed_x[:] = self.data.x.mean(dim=0)
        else:
            # Keep original features for coalition nodes, zero out others
            perturbed_x = torch.zeros_like(subgraph_data.x)
            perturbed_x[coalition] = subgraph_data.x[coalition]
            
            # For non-coalition nodes, use mean features
            non_coalition = [i for i in range(subgraph_data.x.size(0)) if i not in coalition]
            if non_coalition:
                perturbed_x[non_coalition] = self.data.x.mean(dim=0)
        
        # Create perturbed data
        perturbed_data = subgraph_data.__class__(
            x=perturbed_x,
            edge_index=subgraph_data.edge_index,
            batch=subgraph_data.batch if hasattr(subgraph_data, 'batch') else None
        )
        
        with torch.no_grad():
            try:
                pred = self.model(perturbed_data.x, perturbed_data.edge_index)
                if pred.dim() > 1:
                    if isinstance(mapping, torch.Tensor) and mapping.numel() == 1:
                        pred = pred[mapping.item()]
                    else:
                        pred = pred[mapping]
                pred_prob = F.softmax(pred, dim=0)[target_class]
                return pred_prob.item()
            except Exception as e:
                print(f"Prediction error: {e}")
                return 0.0
    
    def _compute_shapley_values(self, coalitions, predictions, num_players):
        """Compute Shapley values using regression approach"""
        # Create feature matrix for regression
        X = np.zeros((len(coalitions), num_players))
        for i, coalition in enumerate(coalitions):
            X[i, coalition] = 1
        
        y = np.array(predictions)
        
        # Fit linear regression
        try:
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X, y)
            shapley_values = reg.coef_
        except Exception as e:
            print(f"Regression error: {e}")
            # Fallback to marginal contributions
            shapley_values = np.zeros(num_players)
            for i in range(num_players):
                with_i = [j for j, coal in enumerate(coalitions) if i in coal]
                without_i = [j for j, coal in enumerate(coalitions) if i not in coal]
                
                if with_i and without_i:
                    shapley_values[i] = np.mean([predictions[j] for j in with_i]) - \
                                      np.mean([predictions[j] for j in without_i])
        
        return shapley_values
    
    def _compute_feature_shapley_values(self, subgraph_data, mapping, target_node, target_class):
        """Compute Shapley values for features"""
        num_features = subgraph_data.x.size(1)
        feature_coalitions = []
        feature_predictions = []
        
        # Generate feature coalitions
        for _ in range(min(self.num_samples, 2**num_features)):
            # Random subset of features
            coalition_size = np.random.randint(0, num_features + 1)
            if coalition_size == 0:
                coalition = []
            else:
                coalition = np.random.choice(num_features, coalition_size, replace=False).tolist()
            
            feature_coalitions.append(coalition)
            
            # Create perturbed features
            perturbed_x = subgraph_data.x.clone()
            mask = torch.ones(num_features, dtype=torch.bool)
            mask[coalition] = False
            perturbed_x[:, mask] = self.data.x.mean(dim=0)[mask]
            
            # Predict with perturbed features
            with torch.no_grad():
                try:
                    pred = self.model(perturbed_x, subgraph_data.edge_index)
                    if pred.dim() > 1:
                        if isinstance(mapping, torch.Tensor) and mapping.numel() == 1:
                            pred = pred[mapping.item()]
                        else:
                            pred = pred[mapping]
                    pred_prob = F.softmax(pred, dim=0)[target_class]
                    feature_predictions.append(pred_prob.item())
                except Exception as e:
                    print(f"Feature prediction error: {e}")
                    feature_predictions.append(0.0)
        
        # Compute feature Shapley values
        shapley_values = self._compute_shapley_values(
            feature_coalitions, feature_predictions, num_features
        )
        
        return shapley_values.tolist()
    
    def explain_graph(self, target_class=None):
        """
        Explain graph-level prediction
        
        Args:
            target_class: Target class to explain
            
        Returns:
            dict: Graph-level explanation
        """
        # Get original prediction
        with torch.no_grad():
            if hasattr(self.model, 'graph_forward'):
                original_pred = self.model.graph_forward(self.data.x, self.data.edge_index)
            else:
                # Assume graph-level prediction is mean/sum of node embeddings
                node_embeddings = self.model(self.data.x, self.data.edge_index)
                original_pred = node_embeddings.mean(dim=0)
            
            if target_class is None:
                target_class = original_pred.argmax().item()
            original_score = F.softmax(original_pred, dim=0)[target_class].item()
        
        # Generate node coalitions for graph explanation
        num_nodes = self.data.x.size(0)
        coalitions = []
        predictions = []
        
        for _ in range(self.num_samples):
            coalition_size = np.random.randint(0, num_nodes + 1)
            if coalition_size == 0:
                coalition = []
            else:
                coalition = np.random.choice(num_nodes, coalition_size, replace=False).tolist()
            
            coalitions.append(coalition)
            pred_score = self._predict_graph_with_coalition(coalition, target_class)
            predictions.append(pred_score)
        
        # Compute node importance for graph prediction
        node_shapley = self._compute_shapley_values(coalitions, predictions, num_nodes)
        
        return {
            'target_class': target_class,
            'original_score': original_score,
            'node_importance': node_shapley.tolist(),
            'num_nodes': num_nodes
        }
    
    def _predict_graph_with_coalition(self, coalition, target_class):
        """Make graph prediction with node coalition"""
        if len(coalition) == 0:
            return 0.0
        
        # Create subgraph with coalition nodes
        perturbed_x = self.data.x.clone()
        non_coalition = [i for i in range(self.data.x.size(0)) if i not in coalition]
        
        if non_coalition:
            perturbed_x[non_coalition] = 0  # Zero out non-coalition nodes
        
        with torch.no_grad():
            try:
                if hasattr(self.model, 'graph_forward'):
                    pred = self.model.graph_forward(perturbed_x, self.data.edge_index)
                else:
                    node_embeddings = self.model(perturbed_x, self.data.edge_index)
                    pred = node_embeddings.mean(dim=0)
                
                pred_prob = F.softmax(pred, dim=0)[target_class]
                return pred_prob.item()
            except Exception as e:
                print(f"Graph prediction error: {e}")
                return 0.0


class GraphSVXConfig:
    """Configuration class for GraphSVX parameters"""
    
    def __init__(self):
        self.num_samples = 100
        self.hops = 2
        self.feat = 'Expectation'  # Feature handling method
        self.coal = 'SmarterSeparate'  # Coalition sampling method
        self.regu = 0  # Regularization parameter
        self.S = 1  # Sampling parameter
        self.hv = 'compute_pred'  # How to compute values
        self.fullempty = True  # Include full and empty coalitions