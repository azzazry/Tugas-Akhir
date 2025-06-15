import pickle
import os
from src.utils.paths import get_paths

from src.plots.training_overview import _plot_training_overview
from src.plots.performance_dashboard import _plot_performance_dashboard
from src.plots.roc_pr_curves import _plot_roc_pr_curves
from src.plots.confusion_matrix import _plot_detailed_confusion_matrix
from src.plots.prediction_analysis import _plot_prediction_analysis
from src.plots.explanation_analysis import _plot_explanation_analysis
from src.plots.feature_analysis import _plot_feature_importance_analysis
from src.plots.user_risk_explanation import _plot_user_risk_explanations

def create_research_visualizations(users):
    paths = get_paths(users)
    vis_dir = paths["visualization_dir"]
    os.makedirs(vis_dir, exist_ok=True)

    try:
        with open(paths["evaluation_path"], 'rb') as f:
            eval_results = pickle.load(f)
        with open(paths["explanation_path"], 'rb') as f:
            explanations = pickle.load(f)
    except FileNotFoundError as e:
        print(f"File tidak ditemukan: {e}")
        return
    except Exception as e:
        print(f"Gagal load data: {e}")
        return
    print("")

    # 1. Training Performance Overview
    _plot_training_overview(eval_results['training_info'], output_dir=paths)

    # 2. Model Performance Dashboard
    _plot_performance_dashboard(eval_results, output_dir=paths)

    # 3. ROC dan Precision-Recall Curves
    _plot_roc_pr_curves(eval_results, output_dir=paths)

    # 4. Confusion Matrix
    _plot_detailed_confusion_matrix(eval_results, output_dir=paths)

    # 5. Prediction Outcome Analysis
    _plot_prediction_analysis(eval_results, output_dir=paths)

    # 6. Explanation Summary (GraphSVX)
    _plot_explanation_analysis(explanations, output_dir=paths)

    # 7. Feature Importance (GraphSVX)
    _plot_feature_importance_analysis(explanations, output_dir=paths)

    # 8. User Risk Classification & Top List
    _plot_user_risk_explanations(explanations, output_dir=paths, top_n=5)
    
if __name__ == "__main__":
    create_research_visualizations(users=1000)