"""
Advanced Improvement Machine Learning Models Module.

This module provides implementations for candidate improvements suggested in the paper:
1. Regularized Logistic Regression (Interpretable linear baseline)
2. Support Vector Machines (Linear & RBF Kernels with soft-margin parameter C)
3. Multi-Layer Perceptron / Neural Network (MLP baseline on tabular features)
4. Distance-Weighted K-Nearest Neighbors (Weighted KNN)

=============================================================================
THEORETICAL INSIGHTS FOR IMPROVEMENTS:
-----------------------------------------------------------------------------
1. Support Vector Machine (SVM):
   Maximizes the margin between separating hyperplane and support vectors:
       min_{w, b, ξ} 1/2 ||w||^2 + C ∑ ξ_i
   RBF Kernel K(x, z) = exp( - γ ||x - z||^2 ) projects inputs into infinite-
   dimensional Hilbert space to discover non-linear medical boundaries.

2. Multi-Layer Perceptron (MLP) on Small Tabular Data:
   - With small sample size (N=303), Deep Learning MLPs easily overfit.
   - Mitigation techniques: Heavy L2 regularization (alpha), small hidden layers
     (e.g., (32, 16)), early stopping with validation fraction, and Dropout.

3. Weighted KNN vs Standard KNN:
   Standard KNN weights all K neighbors equally: w_i = 1.
   Distance-weighted KNN weights each neighbor inversely by distance:
       w_i = 1 / (d(x, x_i) + ε)
   This prevents distant noisy neighbors from drowning out very close matches.
=============================================================================
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.config import SEED


class LogisticRegressionModel:
    """Regularized Logistic Regression Baseline."""

    def __init__(self, C: float = 1.0, penalty: str = "l2", solver: str = "lbfgs", random_state: int = SEED):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.random_state = random_state
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            random_state=self.random_state,
            max_iter=1000,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit Logistic Regression model."""
        # TODO [USER IMPLEMENTATION]:
        raise NotImplementedError("Implement `fit` for LogisticRegressionModel.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class SupportVectorMachineModel:
    """Support Vector Machine (Linear or RBF Kernel)."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        probability: bool = True,
        random_state: int = SEED,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=self.probability,
            random_state=self.random_state,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit Support Vector Machine."""
        # TODO [USER IMPLEMENTATION]:
        raise NotImplementedError("Implement `fit` for SupportVectorMachineModel.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class MultiLayerPerceptronModel:
    """Tabular Deep Learning Baseline (MLP Classifier)."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (32, 16),
        activation: str = "relu",
        alpha: float = 0.01,
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        early_stopping: bool = True,
        random_state: int = SEED,
    ):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit MLP Classifier."""
        # TODO [USER IMPLEMENTATION]:
        raise NotImplementedError("Implement `fit` for MultiLayerPerceptronModel.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class WeightedKNNModel:
    """Distance-Weighted K-Nearest Neighbors."""

    def __init__(self, n_neighbors: int = 5, p: int = 2):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", p=p)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit Weighted KNN model."""
        # TODO [USER IMPLEMENTATION]:
        raise NotImplementedError("Implement `fit` for WeightedKNNModel.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
