import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import pickle
from models.graphsage import HeteroGraphSAGE

def train_insider_threat_model():
    print("Loading preprocessed data...")
    data = torch.load('data/data_graph.pt', weights_only=False)
    
    # Fix untuk NoneType error - pastikan semua x_dict tidak None
    for node_type in ['user', 'pc', 'url']:
        if node_type not in data.x_dict or data.x_dict[node_type] is None:
            print(f"Warning: {node_type} features is None, creating dummy features")
            if node_type == 'user':
                num_nodes = data['user'].x.shape[0] if hasattr(data['user'], 'x') else data['user'].num_nodes
                data.x_dict[node_type] = torch.randn(num_nodes, 6)
            elif node_type == 'pc':
                num_nodes = data['pc'].x.shape[0] if hasattr(data['pc'], 'x') else data['pc'].num_nodes
                data.x_dict[node_type] = torch.randn(num_nodes, 4)
            elif node_type == 'url':
                num_nodes = data['url'].x.shape[0] if hasattr(data['url'], 'x') else data['url'].num_nodes
                data.x_dict[node_type] = torch.randn(num_nodes, 3)
    
    # Pastikan format x_dict konsisten
    if not hasattr(data, 'x_dict'):
        data.x_dict = {
            'user': data['user'].x,
            'pc': data['pc'].x,
            'url': data['url'].x
        }
    
    print(f"Graph loaded: {data.x_dict['user'].shape[0]} users, "
          f"{data.x_dict['pc'].shape[0]} PCs, {data.x_dict['url'].shape[0]} URLs")
    
    # Debug: Print feature shapes
    for node_type, features in data.x_dict.items():
        print(f"{node_type} features shape: {features.shape}")
    
    # Initialize model
    model = HeteroGraphSAGE(hidden_dim=64, out_dim=2, num_layers=2)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Debug: Print edge types yang tersedia
    print("Available edge types:", list(data.edge_index_dict.keys()))
    
    # Validasi edge types - pastikan hanya ada unidirectional edges
    expected_edge_types = [('user', 'uses', 'pc'), ('user', 'visits', 'url')]
    available_edge_types = list(data.edge_index_dict.keys())
    
    print("Expected edge types:", expected_edge_types)
    for edge_type in expected_edge_types:
        if edge_type not in available_edge_types:
            print(f"Warning: {edge_type} not found in data")
    
    # Filter edge_index_dict untuk hanya gunakan yang diperlukan
    filtered_edge_index_dict = {}
    for edge_type in expected_edge_types:
        if edge_type in data.edge_index_dict:
            filtered_edge_index_dict[edge_type] = data.edge_index_dict[edge_type]
    
    print("Using edge types:", list(filtered_edge_index_dict.keys()))
    
    model.train()
    train_losses = []
    train_accs = []
    
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        try:
            # Forward pass dengan filtered edges
            out = model(data.x_dict, filtered_edge_index_dict)
            
            # Handle oversampling
            oversampling_info = data.oversampling_info
            resampled_indices = oversampling_info['resampled_indices']
            resampled_labels = torch.tensor(oversampling_info['resampled_labels'], dtype=torch.long)
            
            # Calculate loss
            train_out = out[resampled_indices]
            loss = F.cross_entropy(train_out, resampled_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                pred = train_out.argmax(dim=1)
                train_acc = accuracy_score(resampled_labels.cpu(), pred.cpu())
                
            train_losses.append(loss.item())
            train_accs.append(train_acc)
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}')
                
        except Exception as e:
            print(f"Error at epoch {epoch}: {e}")
            print("Debugging info:")
            print(f"x_dict keys: {list(data.x_dict.keys())}")
            print(f"x_dict shapes: {[(k, v.shape if v is not None else 'None') for k, v in data.x_dict.items()]}")
            print(f"edge_index_dict keys: {list(filtered_edge_index_dict.keys())}")
            raise e
    
    # Save model dan training info
    torch.save(model.state_dict(), 'result/logs/insider_threat_graphsage.pt')
    
    training_info = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'epochs': epochs,
        'final_loss': train_losses[-1],
        'final_acc': train_accs[-1]
    }
    
    with open('result/logs/training_info.pkl', 'wb') as f:
        pickle.dump(training_info, f)
    
    print(f"Training completed. Final loss: {train_losses[-1]:.4f}, Final acc: {train_accs[-1]:.4f}")
    return model, training_info