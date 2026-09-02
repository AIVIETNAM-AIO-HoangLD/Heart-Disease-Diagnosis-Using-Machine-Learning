"""
Visualization & Plotting Utilities Module.

This module provides functions for:
1. Correlation heatmaps (Seaborn heatmap)
2. Pairwise scatter plots with linear regression overlays (np.polyfit)
3. Cross-Validation parameter tuning curves
4. Feature importance horizontal bar charts (Mutual Information & Decision Tree)
5. Multi-dataset validation vs test performance comparison bar charts
"""

from typing import List, Optional, Sequence, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Sequence[str],
    title: str = "Correlation Matrix (Numeric Features)",
    figsize=(7, 6),
):
    """
    Plot correlation matrix using Seaborn heatmap.

    HINT:
        1. corr = df[columns].corr()
        2. sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    """
    # TODO [USER IMPLEMENTATION]:
    # Plot and display heatmap
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={'size':9})
    plt.title(title)
    plt.show()


def plot_scatter_with_regression(
    xs: pd.Series,
    ys: pd.Series,
    target: pd.Series,
    xlab: str,
    ylab: str,
    title: str,
):
    """
    Plot scatter diagram between two continuous features with 1st-degree polynomial regression line.

    HINT:
        1. mask = xs.notna() & ys.notna()
        2. colors = np.where(target == 0, 'tab:blue', 'tab:red')
        3. plt.scatter(xs, ys, c=colors, alpha=0.7)
        4. k, b = np.polyfit(xs[mask], ys[mask], 1)
        5. xline = np.linspace(xs[mask].min(), xs[mask].max(), 100)
        6. plt.plot(xline, k * xline + b, color='black', linestyle='--')
    """
    # TODO [USER IMPLEMENTATION]:
    mask = xs.notna() & ys.notna()
    colors = np.where(target == 0, 'tab:blue', 'tab:red')
    plt.scatter(xs, ys, c=colors, alpha=0.7)
    k, b = np.polyfit(xs[mask], ys[mask], 1)
    xline = np.linspace(xs[mask].min(), xs[mask].max(), 100)
    plt.plot(xline, k * xline + b, color='black', linestyle='--')
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    raise NotImplementedError("Implement `plot_scatter_with_regression`.")


def plot_cross_validation_curve(
    param_values: Sequence[Union[int, float]],
    cv_scores: Sequence[float],
    param_name: str = "Hyperparameter",
    title: str = "Cross-Validation Accuracy vs Hyperparameter",
):
    """
    Plot hyperparameter search curve showing Mean CV Accuracy.

    HINT:
        1. plt.figure(figsize=(10, 6))
        2. plt.plot(param_values, cv_scores, 'bo-', linewidth=2)
        3. Annotate optimal point
        4. plt.grid(True)
    """
    # TODO [USER IMPLEMENTATION]:
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, cv_scores, 'bo-', linewidth=2)
    plt.grid(True)
    plt.show()
    raise NotImplementedError("Implement `plot_cross_validation_curve`.")


def plot_feature_importance(
    importance_series: pd.Series,
    title: str = "Top Feature Importance Scores",
    top_n: int = 20,
):
    """
    Plot horizontal bar chart of feature importance (MI or Decision Tree MDI).

    HINT:
        1. top = importance_series.head(top_n).iloc[::-1]
        2. plt.barh(top.index, top.values, color='tab:blue')
        3. plt.xlabel("Score")
        4. plt.ylabel("Feature")
    """
    # TODO [USER IMPLEMENTATION]:
    top = importance_series.head(top_n).iloc[::-1]
    plt.barh(top.index, top.values, color='tab:blue')
    plt.xlabel("Score")
    plt.ylabel("Feature")
    plt.title(title)
    plt.show()
    raise NotImplementedError("Implement `plot_feature_importance`.")


def plot_comparison_bar_chart(
    dataset_labels: List[str],
    val_accuracies: List[float],
    test_accuracies: List[float],
    model_name: str = "Model Performance",
    figsize=(7, 6),
):
    """
    Plot grouped bar chart comparing Validation vs Test accuracy across datasets.

    HINT:
        1. x = np.arange(len(dataset_labels))
        2. width = 0.35
        3. ax.bar(x - width/2, val_accuracies, width, label='Validation Accuracy', color='tab:blue')
        4. ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='tab:red')
        5. ax.set_ylim(0.5, 1.05)
        6. Annotate values on top of bars
    """
    # TODO [USER IMPLEMENTATION]:
    x = np.arange(len(dataset_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, val_accuracies, width, label='Validation Accuracy', color='tab:blue')
    ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='tab:red')
    ax.set_ylim(0.5, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_title(model_name)
    ax.legend()
    plt.show()
    raise NotImplementedError("Implement `plot_comparison_bar_chart`.")
