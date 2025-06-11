import torch
import torch.nn.functional as F
import pickle
from models.graphsage import GraphSAGE
from models.graphsvx import GraphSVXExplainer

# ANSI warna untuk terminal
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

log_lines = []

def log_line(line):
    print(line, flush=True)
    log_lines.append(line)

def explain_insider_predictions():
    data = torch.load('data/data_graph.pt', weights_only=False)
    model = GraphSAGE(hidden_dim=64, out_dim=2, num_layers=2)
    model.load_state_dict(torch.load('result/data/insider_threat_graphsage.pt'))
    model.eval()

    x_dict = getattr(data, 'x_dict', {
        'user': data['user'].x,
        'pc': data['pc'].x,
        'url': data['url'].x
    })

    expected_edges = [('user', 'uses', 'pc'), ('user', 'visits', 'url'), ('user', 'interacts', 'user')]
    edge_index_dict = {etype: data.edge_index_dict[etype] for etype in expected_edges if etype in data.edge_index_dict}

    try:
        with open('result/data/evaluation_results.pkl', 'rb') as f:
            eval_results = pickle.load(f)
            best_threshold = eval_results.get('best_threshold', 0.15)
    except FileNotFoundError:
        best_threshold = 0.15

    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        probs = F.softmax(out, dim=1)
        insider_candidates = (probs[:, 1] > best_threshold).nonzero(as_tuple=True)[0]

        log_line(f"Using threshold {best_threshold:.2f}, users with insider probability > threshold: {len(insider_candidates)}")
        if len(insider_candidates) == 0:
            top_probs, top_indices = torch.topk(probs[:, 1], k=min(2, len(probs)))
            insider_candidates = top_indices
            log_line("No users meet the threshold. Taking top 2 highest probabilities...")
            log_line(f"Top 2 insider probabilities: {[round(p.item(), 3) for p in top_probs]}")


        insider_probs = probs[insider_candidates, 1]

    log_line(f"Analyzing {len(insider_candidates)} users with highest insider risk...")

    sorted_indices = torch.argsort(insider_probs, descending=True)
    explain_indices = insider_candidates[sorted_indices[:5]]

    explainer = GraphSVXExplainer(model=model, node_type='user', num_samples=30)
    explanations = {}

    feature_names = [
        'Total Logon Events', 'Total File Events', 'Total Device Events', 'Total HTTP Events',
        'Encoded Role', 'Encoded Department', 'Logon Count', 'After Hours Logon', 'Weekend Logon',
        'File Open Count', 'File Write Count', 'File Copy Count', 'File Delete Count',
        'Device Connect Count', 'Device Disconnect Count', 'Visit Frequency', 'Unique Visit Days',
        'After Hours Browsing', 'Cloud Service Visits', 'Job Site Visits'
    ]

    with open('data/user_metadata.pkl', 'rb') as f:
        user_meta = pickle.load(f)

    for idx in explain_indices:
        user_idx = idx.item()
        prob = probs[user_idx, 1].item()
        user_features = x_dict['user'][user_idx]

        uid = user_meta[user_idx]['user_id']
        role = user_meta[user_idx]['role']
        dept = user_meta[user_idx]['department']

        log_line(f"\nUser {uid} - Insider Probability: {prob:.3f}")
        log_line(f"Role: {role}, Department: {dept}")

        if prob > 0.7:
            risk_class = "Resiko Tinggi"
        elif prob > 0.4:
            risk_class = "Resiko Sedang"
        else:
            risk_class = "Resiko Rendah (Top Candidate)"

        log_line(f"Risk Classification: {risk_class}")

        importance = explainer.explain(x_dict, edge_index_dict, user_idx)
        top_indices = importance.argsort()[-5:][::-1]

        printed = set()
        top_feat_names = [(feature_names[i] if i < len(feature_names) else f"feature_{i}") for i in top_indices]
        max_feat_len = max(len(name) for name in top_feat_names)
        max_bar_len = 20
        max_contrib_str_len = max_bar_len + len(" +0.00")

        line = f"+{'-' * (max_feat_len + 2)}+{'-' * (max_contrib_str_len + 2)}+"
        header = f"| {'Feature':<{max_feat_len}} | {'Contribution':<{max_contrib_str_len}} |"

        log_line(line)
        log_line(header)
        log_line(line)

        for feat_idx in top_indices:
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
            if feat_name in printed:
                continue
            feat_importance = importance[feat_idx]
            printed.add(feat_name)

            bar_len = int(abs(feat_importance) * max_bar_len)
            bar = "▓" * bar_len if bar_len > 0 else "-"
            sign = "+" if feat_importance >= 0 else "-"
            color = RED if abs(feat_importance) >= 0.5 else YELLOW if abs(feat_importance) >= 0.2 else GREEN
            contrib_str = f"{bar} {sign}{abs(feat_importance):.2f}"

            log_line(f"| {feat_name:<{max_feat_len}} | {color}{contrib_str:<{max_contrib_str_len}}{RESET} |")

        log_line(line)
        log_line(f"Recommendation: {get_recommendation(prob)}")

        explanations[user_idx] = {
            'user_id': uid,
            'role': role,
            'department': dept,
            'importance_scores': importance.tolist(),
            'importance_scores': importance.tolist(),
            'probability': prob,
            'risk_classification': risk_class,
            'top_features': [(feature_names[i] if i < len(feature_names) else f"feature_{i}", 
                            user_features[i].item(), importance[i]) for i in top_indices]
        }

    with open('result/data/graphsvx_explanations.pkl', 'wb') as f:
        pickle.dump(explanations, f)

    log_line(f"\nKesimpulan:")
    log_line(f"- Total users analyzed: {len(x_dict['user'])}")
    log_line(f"- Users meeting threshold (>{best_threshold}): {len(insider_candidates)}")
    log_line(f"- Max insider probability: {probs[:, 1].max():.3f}")
    log_line(f"- Average insider probability: {probs[:, 1].mean():.3f}")

    with open("result/logs/explanation_log.log", "w", encoding="utf-8") as f:
        for line in log_lines:
            clean_line = line.replace(RED, "").replace(YELLOW, "").replace(GREEN, "").replace(RESET, "")
            f.write(clean_line + "\n")

def get_recommendation(prob):
    if prob > 0.7:
        return "Segera lakukan investigasi mendalam dan monitoring aktif."
    elif prob > 0.4:
        return "Perketat pengawasan dan audit aktivitas pengguna."
    else:
        return "Monitor secara rutin dan lakukan edukasi keamanan."

if __name__ == "__main__":
    explain_insider_predictions()