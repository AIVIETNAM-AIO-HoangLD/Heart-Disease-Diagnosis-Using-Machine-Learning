"""
Feature Selection Algorithms & Ranking Modules.

This module provides two primary feature selection strategies:
1. Mutual Information Selection (`mutual_info_classif`):
   Captures any non-linear statistical dependency between input features and target.
2. Decision Tree Feature Importance Selection (`DecisionTreeClassifier.feature_importances_`):
   Ranks features based on total reduction in Gini Impurity (or Information Gain).

=============================================================================
MATHEMATICAL BACKGROUND:
-----------------------------------------------------------------------------
1. Mutual Information (MI):
   I(X; Y) = ∬ p(x, y) log( p(x, y) / (p(x) * p(y)) ) dx dy
   - I(X; Y) = 0 if and only if X and Y are independent.
   - Non-parametric, capable of capturing arbitrary non-linear associations.

2. Gini Impurity Importance (MDI - Mean Decrease in Impurity):
   Impurity reduction at node t splitting on feature j:
       ΔGini(t, j) = Gini(t) - (N_left/N_t * Gini(t_left) + N_right/N_t * Gini(t_right))
   Total importance of feature j is the normalized sum of ΔGini across all nodes.

[IMPROVEMENT HINTS]:
- Low Variance Filter: `VarianceThreshold(threshold=0.01)` to remove static features.
- Collinearity Elimination: Calculate Pearson correlation matrix and remove features with |r| > 0.85.
- Recursive Feature Elimination: `RFE(estimator=RandomForestClassifier(), n_features_to_select=10)`
=============================================================================
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier

from src.config import DEFAULT_TOP_K_FEATURES, SEED


def select_top_k_mutual_info(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int = DEFAULT_TOP_K_FEATURES,
    categorical_feature_names: Optional[List[str]] = None,
    random_state: int = SEED,
) -> Tuple[List[str], pd.Series]:
    """
    Rank and select Top-K features using Mutual Information score.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        k: Number of top features to select.
        categorical_feature_names: List of column names that are discrete/one-hot encoded.
        random_state: Random state for MI estimation.

    Returns:
        Tuple: (selected_top_k_columns, sorted_mi_series)

    HINT:
        1. Construct boolean mask `is_discrete`:
           `is_discrete = [col in categorical_feature_names for col in X_train.columns]`
        2. Compute MI scores:
           `mi = mutual_info_classif(X_train.values, y_train.values, discrete_features=is_discrete, random_state=random_state)`
        3. Create pd.Series `mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)`
        4. Select top K: `top_cols = list(mi_series.head(k).index)`
        5. Return `(top_cols, mi_series)`
    """
    is_discrete = [col in categorical_feature_names for col in X_train.columns]
    mi = mutual_info_classif(
        X_train.values, y_train.values, discrete_features=is_discrete, random_state=random_state
    )
    mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    top_cols = list(mi_series.head(k).index)
    return top_cols, mi_series


def select_top_k_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int = DEFAULT_TOP_K_FEATURES,
    random_state: int = SEED,
) -> Tuple[List[str], pd.Series]:
    """
    Rank and select Top-K features using Decision Tree Gini feature importance.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        k: Number of top features to select.
        random_state: Random state for Decision Tree classifier.

    Returns:
        Tuple: (selected_top_k_columns, sorted_importance_series)

    HINT:
        1. Instantiate `dt = DecisionTreeClassifier(random_state=random_state)`
        2. Fit `dt.fit(X_train, y_train)`
        3. Create pd.Series `importance_series = pd.Series(dt.feature_importances_, index=X_train.columns).sort_values(ascending=False)`
        4. Select top K: `top_cols = list(importance_series.head(k).index)`
        5. Return `(top_cols, importance_series)`
    """
    dt = DecisionTreeClassifier(random_state=random_state)
    dt.fit(X_train, y_train)
    importance_series = pd.Series(dt.feature_importances_, index=X_train.columns)
    importance_series = importance_series.sort_values(ascending=False)
    top_cols = list(importance_series.head(k).index)
    return top_cols, importance_series


def filter_low_variance_features(
    X_train: pd.DataFrame,
    threshold: float = 0.01,
) -> List[str]:
    """
    [IMPROVEMENT HINT]
    Filter out quasi-constant features with variance below threshold.

    HINT:
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_train)
        return list(X_train.columns[selector.get_support()])
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)
    return list(X_train.columns[selector.get_support()])