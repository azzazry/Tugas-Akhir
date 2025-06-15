import pickle
import os
from src.utils.paths import get_paths

from src.plots.training_overview import _plot_training_overview
from src.plots.performance_dashboard import _plot_performance_dashboard
from src.plots.roc_pr_curves import _plot_roc_pr_curves
from src.plots.confusion_matrix import _plot_confusion_matrix
from src.plots.prediction_analysis import _plot_prediction_analysis
from src.plots.explanation_analysis import _plot_explanation_analysis
from src.plots.feature_analysis import _plot_feature_importance_analysis
from src.plots.user_risk_explanation import _plot_user_risk_explanations

def create_research_visualizations(users, top_n):
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

    _plot_training_overview(eval_results['training_info'], output_dir=paths)
    _plot_performance_dashboard(eval_results, output_dir=paths)
    _plot_roc_pr_curves(eval_results, output_dir=paths)
    _plot_confusion_matrix(eval_results, output_dir=paths)
    _plot_prediction_analysis(eval_results, output_dir=paths)
    _plot_explanation_analysis(explanations, output_dir=paths)
    _plot_feature_importance_analysis(explanations, output_dir=paths)
    _plot_user_risk_explanations(explanations, output_dir=paths, top_n=top_n)

if __name__ == "__main__":
    try:
        from src.utils.argparse import get_arguments
        args = get_arguments()
        create_research_visualizations(users=args.users, top_n=args.top_n)
    except ImportError:
        create_research_visualizations(users='1000', top_n='5')