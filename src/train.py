import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pickle
from models.graphsage import HeteroGraphSAGE

def train_insider_threat_model():
    """Training model Insider Threat GraphSAGE"""
    print("Loading preprocessed data...")
    data = torch.load('data/data_graph.pt', weights_only=False)
    
    print(f"Graph loaded: {data['user'].x.shape[0]} users, "
          f"{data['pc'].x.shape[0]} PCs, {data['url'].x.shape[0]} URLs")
    
    # Initialize model
    model = HeteroGraphSAGE(hidden_dim=64, out_dim=2, num_layers=2)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Debug: Print edge types yang tersedia
    print("Available edge types:", list(data.edge_index_dict.keys()))
    
    model.train()
    train_losses = []
    train_accs = []
    
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x_dict, data.edge_index_dict)
        
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