import torch
import torch.nn.functional as F
import pickle
from models.graphsage import GraphSAGE
from models.graphsvx import GraphSVXExplainer

def explain_insider_predictions():
    # Load data dan model
    data = torch.load('data/data_graph.pt', weights_only=False)
    model = GraphSAGE(hidden_dim=64, out_dim=2, num_layers=2)
    model.load_state_dict(torch.load('result/logs/insider_threat_graphsage.pt'))
    model.eval()

    # Sesuaikan dengan struktur data Anda
    if hasattr(data, 'x_dict'):
        x_dict = data.x_dict
    else:
        x_dict = {
            'user': data['user'].x,
            'pc': data['pc'].x,
            'url': data['url'].x
        }

    # Edge index sesuai dengan model Anda
    expected_edges = [('user', 'uses', 'pc'), ('user', 'visits', 'url'), ('user', 'interacts', 'user')]
    edge_index_dict = {etype: data.edge_index_dict[etype] for etype in expected_edges if etype in data.edge_index_dict}

    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
        probs = F.softmax(out, dim=1)
        predictions = out.argmax(dim=1)
        
        # PERBAIKAN: Gunakan threshold probability alih-alih argmax
        insider_threshold = 0.3  # Threshold lebih rendah untuk data imbalanced
        insider_candidates = (probs[:, 1] > insider_threshold).nonzero(as_tuple=True)[0]
        
        print(f"Users with insider probability > {insider_threshold}: {len(insider_candidates)}")
        
        # Jika tidak ada yang memenuhi threshold, ambil top 5 berdasarkan probabilitas
        if len(insider_candidates) == 0:
            print("No users meet the threshold. Taking top 5 highest probabilities...")
            top_probs, top_indices = torch.topk(probs[:, 1], k=min(5, len(probs)))
            insider_candidates = top_indices
            print(f"Top 5 insider probabilities: {top_probs.tolist()}")
        
        insider_probs = probs[insider_candidates, 1]

    print(f"Analyzing {len(insider_candidates)} users with highest insider risk...")

    # Ambil top 5 untuk dijelaskan
    sorted_indices = torch.argsort(insider_probs, descending=True)
    explain_indices = insider_candidates[sorted_indices[:5]]

    explainer = GraphSVXExplainer(model=model, node_type='user', num_samples=30)
    explanations = {}

    # Feature names untuk user
    feature_names = [
        'login_frequency',      # Feature 0: Seberapa sering login
        'after_hours_activity', # Feature 1: Aktivitas di luar jam kerja  
        'data_access_volume',   # Feature 2: Volume data yang diakses
        'failed_logins',        # Feature 3: Jumlah gagal login
        'privileged_access',    # Feature 4: Akses ke sistem sensitif
        'behavioral_score'      # Feature 5: Skor anomali perilaku
    ]
    
    with open('data/user_metadata.pkl', 'rb') as f:
        user_meta = pickle.load(f)

    for idx in explain_indices:
        user_idx = idx.item()
        prob = probs[idx, 1].item()
        user_features = x_dict['user'][idx]
        
        uid = user_meta[user_idx]['user_id']
        role = user_meta[user_idx]['role']
        dept = user_meta[user_idx]['department']

        print(f"\nUser {uid} - Insider Probability: {prob:.3f}")
        print(f"Role: {role}, Department: {dept}")
        
        # Klasifikasi risiko berdasarkan probabilitas
        if prob > 0.5:
            risk_class = "HIGH RISK"
        elif prob > 0.3:
            risk_class = "MEDIUM RISK"
        else:
            risk_class = "LOW RISK (Top Candidate)"
        
        print(f"Risk Classification: {risk_class}")
        print("-" * 40)
        
        # Dapatkan feature importance
        importance = explainer.explain(x_dict, edge_index_dict, user_idx)
        
        # Tampilkan top 3 features yang paling berpengaruh
        top_indices = importance.argsort()[-10:][::-1]
        
        print("Key risk indicators:")
        for i, feat_idx in enumerate(top_indices):
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
            feat_value = user_features[feat_idx].item()
            feat_importance = importance[feat_idx]
            
            # Interpretasi berdasarkan nilai dan importance
            if feat_importance > 0.05:
                interpretation = interpret_feature(feat_name, feat_value, feat_importance)
                print(f"  {i+1}. {feat_name}: {interpretation}")
        
        # Rekomendasi tindakan
        print(f"Recommendation: {get_recommendation(prob)}")
        
        # Simpan hasil
        explanations[user_idx] = {
            'importance_scores': importance.tolist(),
            'probability': prob,
            'risk_classification': risk_class,
            'top_features': [(feature_names[i] if i < len(feature_names) else f"feature_{i}", 
                            user_features[i].item(), importance[i]) for i in top_indices]
        }

    # Simpan hasil
    with open('result/logs/graphsvx_explanations.pkl', 'wb') as f:
        pickle.dump(explanations, f)
    
    # Summary statistics
    print(f"\nKesimpulan:")
    print(f"- Total users analyzed: {len(x_dict['user'])}")
    print(f"- Users meeting threshold (>{insider_threshold}): {len(insider_candidates)}")
    print(f"- Max insider probability: {probs[:, 1].max():.3f}")
    print(f"- Average insider probability: {probs[:, 1].mean():.3f}")

def interpret_feature(feat_name, value, importance):
    interpretations = {
        'login_frequency': f"Login pattern: {value:.1f}/day (risk impact: {importance:.3f})",
        'after_hours_activity': f"Off-hours access: {value:.1f}% (risk impact: {importance:.3f})",
        'data_access_volume': f"Data access: {value:.1f}GB (risk impact: {importance:.3f})",
        'failed_logins': f"Failed attempts: {value:.0f} (risk impact: {importance:.3f})",
        'privileged_access': f"Admin access: {'Yes' if value > 0.5 else 'No'} (risk impact: {importance:.3f})",
        'behavioral_score': f"Anomaly score: {value:.3f}/1.0 (risk impact: {importance:.3f})"
    }
    return interpretations.get(feat_name, f"Value: {value:.3f} (impact: {importance:.3f})")

def get_recommendation(prob):
    """Rekomendasi berdasarkan probabilitas"""
    if prob > 0.5:
        return "Enhanced monitoring and access review required"
    elif prob > 0.3:
        return "Regular monitoring recommended"
    else:
        return "Standard monitoring sufficient"

if __name__ == "__main__":
    explain_insider_predictions()