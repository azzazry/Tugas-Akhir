import torch
import pickle
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from src.models.graphsage import GraphSAGE
from src.utils.paths import get_paths
from src.utils.metrics import compute_class_weights
from src.utils.constants import CLASS_WEIGHT_SCALING, EPOCHS, WEIGHT_DECAY, GRAPHSAGE_PARAMS, LR
from src.utils.logger import log_line, clear_log_lines, flush_logs

def train_insider_threat_model(users):
    clear_log_lines()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = get_paths(users)

    data = torch.load(paths["data_graph_path"], weights_only=False)
    if not hasattr(data, 'x_dict'):
        data.x_dict = {
            'user': data['user'].x,
            'pc': data['pc'].x,
            'url': data['url'].x
        }

    model = GraphSAGE(**GRAPHSAGE_PARAMS).to(device)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    labels = data['user'].y.to(device)
    class_weights = compute_class_weights(labels, num_classes=2, scaling=CLASS_WEIGHT_SCALING)
    criterion = CrossEntropyLoss(weight=class_weights)

    model.train()
    train_losses = []
    train_accs = []
    epochs = EPOCHS
    edge_index_dict = data.edge_index_dict

    for key in data.x_dict:
        data.x_dict[key] = data.x_dict[key].to(device)

    log_line("+--------+----------+----------+")
    log_line("| Epoch  |  Loss    | Accuracy |")
    log_line("+--------+----------+----------+")

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
            log_line(f"|  {epoch:03d}   |  {loss.item():<8.4f}|  {acc:<8.4f}|")

    log_line("+--------+----------+----------+")
    log_line(f"Final loss: {train_losses[-1]:.4f}, Final acc: {train_accs[-1]:.4f}")
    log_line(f"Class weights used: {class_weights.cpu().tolist()}")

    torch.save(model.state_dict(), paths["model_path"])
    training_info = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'epochs': epochs,
        'final_loss': train_losses[-1],
        'final_acc': train_accs[-1],
        'class_weights': class_weights.cpu().tolist(),
    }
    with open(paths["training_info_path"], 'wb') as f:
        pickle.dump(training_info, f)

    flush_logs(paths['training_log_path'])

    return model, training_info

if __name__ == "__main__":
    try:
        from src.utils.argparse import get_arguments
        args = get_arguments()
        train_insider_threat_model(users=args.users)
    except ImportError:
        train_insider_threat_model(users='1000')