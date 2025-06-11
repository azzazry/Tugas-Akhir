import pickle
import os

from src.plots.training_overview import _plot_training_overview
from src.plots.performance_dashboard import _plot_performance_dashboard
from src.plots.roc_pr_curves import _plot_roc_pr_curves
from src.plots.confusion_matrix import _plot_detailed_confusion_matrix
from src.plots.prediction_analysis import _plot_prediction_analysis
from src.plots.explanation_analysis import _plot_explanation_analysis
from src.plots.feature_analysis import _plot_feature_importance_analysis
from src.plots.user_risk_explanation import _plot_user_risk_explanations

def create_research_visualizations():

    os.makedirs('result/visualizations', exist_ok=True)
    with open('result/data/evaluation_results.pkl', 'rb') as f:
        eval_results = pickle.load(f)
    with open('result/data/graphsvx_explanations.pkl', 'rb') as f:
        explanations = pickle.load(f)
    
    # 1. Training Performance Overview
    _plot_training_overview(eval_results['training_info'])
    print('1. training overview')
    
    # 2. Model Performance Dashboard
    _plot_performance_dashboard(eval_results)
    print('2. performance dashboard')
    
    # 3. ROC dan Precision-Recall Curves
    _plot_roc_pr_curves(eval_results)
    print('3. roc pr curves')
    
    # 4. Confusion Matrix dengan detail
    _plot_detailed_confusion_matrix(eval_results)
    print('4. confusion matrix')
    
    # 5. Model Prediction Analysis
    _plot_prediction_analysis(eval_results)
    print('5. prediction analysis')
    
    # 6. GraphSVX Explanation Analysis
    _plot_explanation_analysis(explanations)
    print('6. explanation analysis')
    
    # 7. Feature Importance dari GraphSVX
    _plot_feature_importance_analysis(explanations)
    print('7. feature importance analysis')
    
    # 8. User Risk Distribution
    _plot_user_risk_explanations(explanations)
    print('8. user risk distribution')
    
    print("Semua visualisasi tersimpan di result/visualizations/")

if __name__ == "__main__":
    create_research_visualizations()