import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import pickle
from models.graphsage import GraphSAGE

def train_insider_threat_model():
    # Load data graph, anggap sudah lengkap
    data = torch.load('data/data_graph.pt', weights_only=False)
    
    # Inisialisasi model & optimizer sederhana
    model = GraphSAGE(hidden_dim=64, out_dim=2)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    train_losses = []
    train_accs = []
    epochs = 100
    
    edge_index_dict = data.edge_index_dict
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x_dict, edge_index_dict)
        labels = data['user'].y
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            pred = out.argmax(dim=1)
            acc = accuracy_score(labels.cpu(), pred.cpu())
        
        train_losses.append(loss.item())
        train_accs.append(acc)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
    
    # Save model dan training info
    torch.save(model.state_dict(), 'result/logs/insider_threat_graphsage.pt')
    training_info = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'epochs': epochs,
        'final_loss': train_losses[-1],
        'final_acc': train_accs[-1],
    }
    with open('result/logs/training_info.pkl', 'wb') as f:
        pickle.dump(training_info, f)
    
    print(f"Training selesai. Final loss: {train_losses[-1]:.4f}, Final acc: {train_accs[-1]:.4f}")
    return model, training_info

if __name__ == "__main__":
    train_insider_threat_model()