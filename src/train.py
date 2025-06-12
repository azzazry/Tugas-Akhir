import torch
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

    labels = data['user'].y.to(device)
    classes, counts = torch.unique(labels, return_counts=True)
    class_weights = torch.zeros(2).to(device)
    for cls, cnt in zip(classes, counts):
        class_weights[cls] = 1.0 / cnt.float()
    class_weights = class_weights / class_weights.sum() * 2
    criterion = CrossEntropyLoss(weight=class_weights)

    model.train()
    train_losses = []
    train_accs = []
    epochs = 100
    edge_index_dict = data.edge_index_dict

    for key in data.x_dict:
        data.x_dict[key] = data.x_dict[key].to(device)

    log_lines = []
    log_lines.append("+--------+----------+----------+")
    log_lines.append("| Epoch  |  Loss    | Accuracy |")
    log_lines.append("+--------+----------+----------+")
    print(log_lines[-3])
    print(log_lines[-2])
    print(log_lines[-1])

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

        if epoch % 10 == 0:
            log_line = f"|  {epoch:03d}   |  {loss.item():<8.4f}|  {acc:<8.4f}|"
            print(log_line)
            log_lines.append(log_line)

    log_lines.append("+--------+----------+----------+")
    print(log_lines[-1])

    torch.save(model.state_dict(), 'result/data/insider_threat_graphsage.pt')
    training_info = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'epochs': epochs,
        'final_loss': train_losses[-1],
        'final_acc': train_accs[-1],
        'class_weights': class_weights.cpu().tolist(),
    }
    with open('result/data/training_info.pkl', 'wb') as f:
        pickle.dump(training_info, f)

    log_lines.append(f"Final loss: {train_losses[-1]:.4f}, Final acc: {train_accs[-1]:.4f}")
    log_lines.append(f"Class weights used: {training_info['class_weights']}")

    with open("result/logs/training_log.log", "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    print(log_lines[-2])
    print(log_lines[-1])

    return model, training_info

if __name__ == '__main__':
    train_insider_threat_model()