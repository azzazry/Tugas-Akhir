import torch
import torch.nn.functional as F
import pickle
from src.models.graphsage import GraphSAGE
from src.models.graphsvx import GraphSVXExplainer
from src.utils.paths import get_paths

RED, YELLOW, GREEN, RESET = "\033[91m", "\033[93m", "\033[92m", "\033[0m"
log_lines = []

def log_line(line):
    print(line, flush=True)
    log_lines.append(line)

def get_feature_names():
    return [
        'Total Logon Events', 'Total File Events', 'Total Device Events', 'Total HTTP Events',
        'Logon Count', 'After Hours Logon', 'Weekend Logon',
        'File Open Count', 'File Write Count', 'File Copy Count', 'File Delete Count',
        'Device Connect Count', 'Device Disconnect Count',
        'Visit Frequency', 'Unique Visit Days',
        'After Hours Browsing', 'Cloud Service Visits', 'Job Site Visits'
    ]

def get_risk_classification(prob):
    if prob > 0.7:
        return "Resiko Tinggi"
    elif prob > 0.4:
        return "Resiko Sedang"
    else:
        return "Resiko Rendah (Top Candidate)"

def get_recommendation(prob):
    status = get_risk_classification(prob)
    if status == "Resiko Tinggi":
        return "Segera lakukan investigasi mendalam dan monitoring aktif."
    elif status == "Resiko Sedang":
        return "Perketat pengawasan dan audit aktivitas pengguna."
    else:
        return "Monitor secara rutin dan lakukan edukasi keamanan."

def explain_insider_predictions(users):
    paths = get_paths(users)

    # Load data dan model
    data = torch.load(paths['data_graph_path'], weights_only=False)
    model = GraphSAGE(hidden_dim=64, out_dim=2, num_layers=2)
    model.load_state_dict(torch.load(paths['model_path']))
    model.eval()

    x_dict = getattr(data, 'x_dict', {
        'user': data['user'].x,
        'pc': data['pc'].x,
        'url': data['url'].x
    })

    expected_edges = [('user', 'uses', 'pc'), ('user', 'visits', 'url'), ('user', 'interacts', 'user')]
    edge_index_dict = {etype: data.edge_index_dict[etype] for etype in expected_edges if etype in data.edge_index_dict}

    # Ambil threshold
    try:
        with open(paths['evaluation_path'], 'rb') as f:
            eval_results = pickle.load(f)
            best_threshold = eval_results.get('best_threshold', 0.15)
    except FileNotFoundError:
        best_threshold = 0.15

    # Hitung probabilitas
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        probs = F.softmax(out, dim=1)
        insider_candidates = (probs[:, 1] > best_threshold).nonzero(as_tuple=True)[0]

        if len(insider_candidates) == 0:
            top_probs, top_indices = torch.topk(probs[:, 1], k=min(2, len(probs)))
            insider_candidates = top_indices
            log_line("No users meet the threshold. Taking top 2 highest probabilities...")
            log_line(f"Top 2 insider probabilities: {[round(p.item(), 3) for p in top_probs]}")

        insider_probs = probs[insider_candidates, 1]

    sorted_indices = torch.argsort(insider_probs, descending=True)
    explain_indices = insider_candidates[sorted_indices[:5]]
    explainer = GraphSVXExplainer(model=model, node_type='user', num_samples=30)

    feature_names = get_feature_names()

    with open(paths['user_metadata_path'], 'rb') as f:
        user_meta = pickle.load(f)

    explanations = {}
    for idx in explain_indices:
        user_idx = idx.item()
        prob = probs[user_idx, 1].item()
        user_features = x_dict['user'][user_idx]
        risk_class = get_risk_classification(prob)

        uid = user_meta[user_idx]['user_id']
        role = user_meta[user_idx]['role']
        dept = user_meta[user_idx]['department']

        log_line(f"\nUser {uid} - Insider Probability: {prob:.3f}")
        log_line(f"Role: {role}, Department: {dept}")
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
            'probability': prob,
            'risk_classification': risk_class,
            'top_features': [
                (feature_names[i] if i < len(feature_names) else f"feature_{i}",
                 user_features[i].item(), importance[i])
                for i in top_indices
            ]
        }

    # Simpan hasil explanations
    with open(paths['explanation_path'], 'wb') as f:
        pickle.dump(explanations, f)

    # Summary log
    log_line(f"\nKesimpulan:")
    log_line(f"- Total users analyzed: {len(x_dict['user'])}")
    log_line(f"- Using threshold: {best_threshold:.2f}")
    log_line(f"- Users meeting threshold (>{best_threshold}): {len(insider_candidates)}")
    log_line(f"- Max insider probability: {probs[:, 1].max():.3f}")
    log_line(f"- Average insider probability: {probs[:, 1].mean():.3f}")

    with open(paths['explanation_log_path'], "w", encoding="utf-8") as f:
        for line in log_lines:
            clean = line.replace(RED, "").replace(YELLOW, "").replace(GREEN, "").replace(RESET, "")
            f.write(clean + "\n")

if __name__ == "__main__":
    explain_insider_predictions(users=1000)