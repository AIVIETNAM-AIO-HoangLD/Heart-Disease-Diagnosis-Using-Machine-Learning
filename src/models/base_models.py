"""
Baseline Machine Learning Models Module (Part 1).

This module implements:
1. Naive Bayes Classifier (`GaussianNB`)
2. K-Nearest Neighbors Classifier with Stratified K-Fold CV Tuning
3. Decision Tree Classifier with Depth Tuning & Feature Importance Extraction
4. K-Means Unsupervised Classifier with Cluster Majority-Voting Label Assignment
5. Stacking Ensemble Classifier (Combining KNN, Decision Tree, Naive Bayes)

=============================================================================
MATHEMATICAL FOUNDATIONS:
-----------------------------------------------------------------------------
1. Gaussian Naive Bayes:
   P(y|x) ∝ P(y) * ∏ P(x_i | y)
   where continuous feature likelihood is modeled as a Gaussian distribution:
   P(x_i | y) = 1 / sqrt(2π * σ_y^2) * exp( - (x_i - μ_y)^2 / (2 * σ_y^2) )

2. K-Nearest Neighbors (KNN):
   Given a query x, find K nearest neighbors N_K(x) by Euclidean distance d(x, z).
   Decision rule (Majority Voting):
   y_hat = argmax_c ∑_{z ∈ N_K(x)} I(y_z = c)

3. K-Means Unsupervised Classification via Majority Mapping:
   K-Means minimizes within-cluster sum of squares (WCSS):
   J = ∑_{k=1}^K ∑_{x ∈ C_k} ||x - μ_k||^2
   After clustering, each cluster k is mapped to class label c:
   Label(C_k) = mode( { y_i | x_i ∈ C_k and x_i ∈ Train } )
=============================================================================
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import CV_SPLITS, DT_DEPTH_RANGE, KNN_K_RANGE, SEED


class GaussianNBModel:
    """Wrapper for Gaussian Naive Bayes Classifier."""

    def __init__(self):
        self.model = GaussianNB()

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]):
        """Fit Gaussian Naive Bayes on training data."""
        # TODO [USER IMPLEMENTATION]:
        # Fit self.model on X_train, y_train and return self
        raise NotImplementedError("Implement `fit` for GaussianNBModel.")

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels."""
        # TODO [USER IMPLEMENTATION]:
        # Call and return self.model.predict(X)
        raise NotImplementedError("Implement `predict` for GaussianNBModel.")

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        # TODO [USER IMPLEMENTATION]:
        # Call and return self.model.predict_proba(X)
        raise NotImplementedError("Implement `predict_proba` for GaussianNBModel.")


class KNNModel:
    """K-Nearest Neighbors Classifier with Stratified K-Fold CV Tuning."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    @staticmethod
    def find_optimal_k(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        k_range=KNN_K_RANGE,
        cv_splits: int = CV_SPLITS,
        random_state: int = SEED,
    ) -> Tuple[int, List[float]]:
        """
        Find optimal K using Stratified K-Fold Cross-Validation.

        HINT:
            1. Setup `cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)`
            2. Iterate through each `k in k_range`:
               - Create `knn = KNeighborsClassifier(n_neighbors=k)`
               - Compute `scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)`
               - Record `mean_score = scores.mean()`
            3. Find `best_k = list(k_range)[np.argmax(scores_list)]`
            4. Return `(best_k, scores_list)`
        """
        # TODO [USER IMPLEMENTATION]:
        # Implement cross-validation loop to find best K
        raise NotImplementedError("Implement `find_optimal_k` for KNNModel.")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, optimal_k: Optional[int] = None):
        """Fit KNN with specified or optimal K."""
        # TODO [USER IMPLEMENTATION]:
        # 1. Update self.n_neighbors if optimal_k is provided
        # 2. Instantiate and fit self.model
        raise NotImplementedError("Implement `fit` for KNNModel.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class DecisionTreeModel:
    """Decision Tree Classifier with Max-Depth CV Tuning."""

    def __init__(self, max_depth: int = 3, random_state: int = SEED):
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)

    @staticmethod
    def find_optimal_depth(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        depth_range=DT_DEPTH_RANGE,
        cv_splits: int = CV_SPLITS,
        random_state: int = SEED,
    ) -> Tuple[int, List[float]]:
        """
        Find optimal tree depth using Stratified K-Fold Cross Validation.

        HINT:
            1. Create StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            2. For each depth in depth_range, fit DecisionTreeClassifier(max_depth=depth)
            3. Compute cross_val_score(..., scoring='accuracy')
            4. Pick best_depth corresponding to argmax of average scores
            5. Return (best_depth, scores_list)
        """
        # TODO [USER IMPLEMENTATION]:
        # Implement cross-validation loop to find optimal depth
        raise NotImplementedError("Implement `find_optimal_depth` for DecisionTreeModel.")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, max_depth: Optional[int] = None):
        """Fit Decision Tree model."""
        # TODO [USER IMPLEMENTATION]:
        # Update self.max_depth and fit self.model on training set
        raise NotImplementedError("Implement `fit` for DecisionTreeModel.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importances(self, feature_names: List[str]) -> pd.Series:
        """Return sorted Series of Gini feature importances."""
        # TODO [USER IMPLEMENTATION]:
        # Return pd.Series(self.model.feature_importances_, index=feature_names).sort_values(ascending=False)
        raise NotImplementedError("Implement `get_feature_importances`.")


class KMeansClassifierModel:
    """
    Unsupervised KMeans Clustering converted to a Classifier via Majority Class Mapping.
    """

    def __init__(self, n_clusters: int = 2, random_state: int = SEED):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, init="random")
        self.cluster_to_class_map: Dict[int, int] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit KMeans on X_train and compute cluster -> label mapping using majority voting.

        HINT:
            1. Fit `self.model.fit(X_train)`
            2. Get train cluster assignments: `train_clusters = self.model.labels_`
            3. For each cluster_id in unique(train_clusters):
               Find the mode of true labels:
               `majority_class = Counter(y_train[train_clusters == cluster_id]).most_common(1)[0][0]`
               `self.cluster_to_class_map[cluster_id] = majority_class`
        """
        # TODO [USER IMPLEMENTATION]:
        # 1. Fit KMeans on X_train
        # 2. Build self.cluster_to_class_map
        raise NotImplementedError("Implement `fit` for KMeansClassifierModel.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster for each sample and map it to binary class label.

        HINT:
            1. `clusters = self.model.predict(X)`
            2. Map clusters: `np.array([self.cluster_to_class_map[c] for c in clusters])`
        """
        # TODO [USER IMPLEMENTATION]:
        # Predict clusters and map to class labels
        raise NotImplementedError("Implement `predict` for KMeansClassifierModel.")


class StackingEnsembleModel:
    """
    Stacking Classifier combining KNN, Decision Tree, and Naive Bayes
    with a KNN final meta-estimator.
    """

    def __init__(self, random_state: int = SEED):
        self.random_state = random_state
        self.estimators = [
            ("knn", KNeighborsClassifier()),
            ("dt", DecisionTreeClassifier(random_state=self.random_state)),
            ("nb", GaussianNB()),
        ]
        self.final_estimator = KNeighborsClassifier()
        self.model = StackingClassifier(
            estimators=self.estimators,
            final_estimator=self.final_estimator,
            stack_method="predict_proba",
            passthrough=False,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit StackingClassifier on training split."""
        # TODO [USER IMPLEMENTATION]:
        # Fit self.model on X_train, y_train
        raise NotImplementedError("Implement `fit` for StackingEnsembleModel.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
