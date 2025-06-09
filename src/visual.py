import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from src.plots.training_overview import _plot_training_overview
from src.plots.performance_dashboard import _plot_performance_dashboard
from src.plots.roc_pr_curves import _plot_roc_pr_curves
from src.plots.confusion_matrix import _plot_detailed_confusion_matrix
from src.plots.prediction_analysis import _plot_prediction_analysis
from src.plots.explanation_analysis import _plot_explanation_analysis
from src.plots.feature_analysis import _plot_feature_importance_analysis
from src.plots.user_risk_distribution import _plot_user_risk_distribution

def create_research_visualizations():
    
    # Create output directory if not exists
    os.makedirs('result/visualizations', exist_ok=True)
    # Load evaluation results
    with open('result/logs/evaluation_results.pkl', 'rb') as f:
        eval_results = pickle.load(f)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Training Performance Overview
    _plot_training_overview(eval_results['training_info'])
    
    # 2. Model Performance Dashboard
    _plot_performance_dashboard(eval_results)
    
    # 3. ROC dan Precision-Recall Curves
    _plot_roc_pr_curves(eval_results)
    
    # 4. Confusion Matrix dengan detail
    _plot_detailed_confusion_matrix(eval_results)
    
    # 5. Model Prediction Analysis
    _plot_prediction_analysis(eval_results)
    
    # 6. GraphSVX Explanation Analysis
    _plot_explanation_analysis()
    
    # 7. Feature Importance dari GraphSVX
    _plot_feature_importance_analysis()
    
    # 8. User Risk Distribution
    _plot_user_risk_distribution(eval_results)
    
    print("Semua visualisasi tersimpan di result/visualizations/")

if __name__ == "__main__":
    create_research_visualizations()