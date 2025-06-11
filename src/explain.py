import torch
import torch.nn.functional as F
import pickle
from models.graphsage import GraphSAGE
from models.graphsvx import GraphSVXExplainer

def explain_insider_predictions():
    log_lines = []
    
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

        log_lines.append(f"Using threshold {best_threshold:.2f}, users with insider probability > threshold: {len(insider_candidates)}")
        if len(insider_candidates) == 0:
            top_probs, top_indices = torch.topk(probs[:, 1], k=min(5, len(probs)))
            insider_candidates = top_indices
            log_lines.append("No users meet the threshold. Taking top 5 highest probabilities...")
            log_lines.append(f"Top 5 insider probabilities: {top_probs.tolist()}")

        insider_probs = probs[insider_candidates, 1]

    log_lines.append(f"Analyzing {len(insider_candidates)} users with highest insider risk...")

    sorted_indices = torch.argsort(insider_probs, descending=True)
    explain_indices = insider_candidates[sorted_indices[:5]]

    explainer = GraphSVXExplainer(model=model, node_type='user', num_samples=30)
    explanations = {}

    feature_names = [
        'login_frequency',
        'after_hours_activity',
        'data_access_volume',
        'failed_logins',
        'privileged_access',
        'behavioral_score'
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

        log_lines.append(f"\nUser {uid} - Insider Probability: {prob:.3f}")
        log_lines.append(f"Role: {role}, Department: {dept}")

        if prob > 0.5:
            risk_class = "Resiko Tinggi"
        elif prob > 0.3:
            risk_class = "Resiko Sedang"
        else:
            risk_class = "Resiko Rendah (Top Candidate)"

        log_lines.append(f"Risk Classification: {risk_class}")
        log_lines.append("-" * 40)

        importance = explainer.explain(x_dict, edge_index_dict, user_idx)
        top_indices = importance.argsort()[-10:][::-1]

        log_lines.append("Key risk indicators:")
        for i, feat_idx in enumerate(top_indices):
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
            feat_value = user_features[feat_idx].item()
            feat_importance = importance[feat_idx]

            if feat_importance > 0.05:
                interpretation = interpret_feature(feat_name, feat_value, feat_importance)
                log_lines.append(f"  {i+1}. {feat_name}: {interpretation}")

        log_lines.append(f"Recommendation: {get_recommendation(prob)}")

        explanations[user_idx] = {
            'importance_scores': importance.tolist(),
            'probability': prob,
            'risk_classification': risk_class,
            'top_features': [(feature_names[i] if i < len(feature_names) else f"feature_{i}", 
                            user_features[i].item(), importance[i]) for i in top_indices]
        }

    with open('result/data/graphsvx_explanations.pkl', 'wb') as f:
        pickle.dump(explanations, f)

    log_lines.append(f"\nKesimpulan:")
    log_lines.append(f"- Total users analyzed: {len(x_dict['user'])}")
    log_lines.append(f"- Users meeting threshold (>{best_threshold}): {len(insider_candidates)}")
    log_lines.append(f"- Max insider probability: {probs[:, 1].max():.3f}")
    log_lines.append(f"- Average insider probability: {probs[:, 1].mean():.3f}")

    # Save log to file
    with open("result/logs/explanation_log.txt", "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    print("Penjelasan selesai! Log disimpan ke result/logs/explanation_log.txt")

def interpret_feature(feat_name, value, importance):
    interpretations = {
        'login_frequency': f"Login pattern: {value:.1f}/day (risk impact: {importance:.3f})",
        'after_hours_activity': f"Off-hours access: {value:.1f}% (risk impact: {importance:.3f})",
        'data_access_volume': f"Data access: {value:.1f} GB (risk impact: {importance:.3f})",
        'failed_logins': f"Failed login attempts: {value:.0f} (risk impact: {importance:.3f})",
        'privileged_access': f"Privileged access level: {value:.1f} (risk impact: {importance:.3f})",
        'behavioral_score': f"Behavioral anomaly score: {value:.2f} (risk impact: {importance:.3f})",
    }
    return interpretations.get(feat_name, f"{feat_name}: {value} (risk impact: {importance:.3f})")

def get_recommendation(prob):
    if prob > 0.7:
        return "Segera lakukan investigasi mendalam dan monitoring aktif."
    elif prob > 0.4:
        return "Perketat pengawasan dan audit aktivitas pengguna."
    else:
        return "Monitor secara rutin dan lakukan edukasi keamanan."

if __name__ == "__main__":
    explain_insider_predictions()