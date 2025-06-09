import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
import pickle
from models.graphsage import GraphSAGE

def train_insider_threat_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load('data/data_graph.pt', weights_only=False)
    if not hasattr(data, 'x_dict'):
        data.x_dict = {
            'user': data['user'].x,
            'pc': data['pc'].x,
            'url': data['url'].x
        }
    
    model = GraphSAGE(hidden_dim=64, out_dim=2, num_layers=2).to(device)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Hitung class weights manual, sesuaikan dengan distribusi kelas
    labels = data['user'].y.to(device)
    classes, counts = torch.unique(labels, return_counts=True)
    class_weights = torch.zeros(2).to(device)
    for cls, cnt in zip(classes, counts):
        class_weights[cls] = 1.0 / cnt.float()
    class_weights = class_weights / class_weights.sum() * 2  # Normalisasi agar total 2
    criterion = CrossEntropyLoss(weight=class_weights)

    model.train()
    train_losses = []
    train_accs = []
    epochs = 100

    edge_index_dict = data.edge_index_dict

    # pindahin data ke device
    for key in data.x_dict:
        data.x_dict[key] = data.x_dict[key].to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x_dict, edge_index_dict)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = out.argmax(dim=1)
            acc = accuracy_score(labels.cpu(), pred.cpu())

        train_losses.append(loss.item())
        train_accs.append(acc)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    torch.save(model.state_dict(), 'result/logs/insider_threat_graphsage.pt')
    training_info = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'epochs': epochs,
        'final_loss': train_losses[-1],
        'final_acc': train_accs[-1],
        'class_weights': class_weights.cpu().tolist(),
    }
    with open('result/logs/training_info.pkl', 'wb') as f:
        pickle.dump(training_info, f)

    print(f"Training selesai. Final loss: {train_losses[-1]:.4f}, Final acc: {train_accs[-1]:.4f}")
    print(f"Class weights used: {training_info['class_weights']}")
    return model, training_info