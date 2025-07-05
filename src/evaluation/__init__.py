"""
Comprehensive evaluation module for Trump Tweets Classification project.

This module provides:
- Detailed classification reports with sklearn metrics
- Confusion matrix visualization and analysis
- ROC curves and AUC calculation
- Cross-validation with statistical significance testing
- Performance comparison and visualization tools
- Error analysis and model interpretation

Usage:
    from src.evaluation import generate_classification_report, plot_confusion_matrix
    from src.evaluation.cross_validation import perform_detailed_cross_validation
    from src.evaluation.visualizations import plot_roc_curves_comparison
"""

from .metrics import (
    generate_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    create_performance_comparison_table,
    evaluate_model_comprehensive,
    cross_validate_with_detailed_metrics
)

from .cross_validation import (
    perform_detailed_cross_validation,
    create_cv_comparison_plot,
    plot_learning_curves,
    generate_cv_summary_table,
    analyze_cv_statistical_significance,
    select_best_model_from_cv
)

from .visualizations import (
    plot_confusion_matrices_comparison,
    plot_roc_curves_comparison,
    plot_precision_recall_curves,
    plot_performance_comparison_bar,
    plot_feature_importance_comparison,
    create_performance_heatmap,
    plot_error_analysis,
    save_all_performance_plots
)

__all__ = [
    # Metrics
    'generate_classification_report',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'create_performance_comparison_table',
    'evaluate_model_comprehensive',
    'cross_validate_with_detailed_metrics',
    
    # Cross-validation
    'perform_detailed_cross_validation',
    'create_cv_comparison_plot',
    'plot_learning_curves',
    'generate_cv_summary_table',
    'analyze_cv_statistical_significance',
    'select_best_model_from_cv',
    
    # Visualizations
    'plot_confusion_matrices_comparison',
    'plot_roc_curves_comparison',
    'plot_precision_recall_curves',
    'plot_performance_comparison_bar',
    'plot_feature_importance_comparison',
    'create_performance_heatmap',
    'plot_error_analysis',
    'save_all_performance_plots'
]