"""
Evaluation and Visualization Package.
"""
from src.evaluation.metrics import evaluate_predictions, print_classification_summary
from src.evaluation.visualization import (
    plot_comparison_bar_chart,
    plot_correlation_matrix,
    plot_cross_validation_curve,
    plot_feature_importance,
    plot_scatter_with_regression,
)

__all__ = [
    "evaluate_predictions",
    "print_classification_summary",
    "plot_correlation_matrix",
    "plot_scatter_with_regression",
    "plot_feature_importance",
    "plot_cross_validation_curve",
    "plot_comparison_bar_chart",
]
