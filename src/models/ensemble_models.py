"""
Advanced Ensemble Machine Learning Models Module (Part 2).

This module implements:
1. Random Forest (Bagging + Feature Randomness to reduce Variance)
2. AdaBoost (Sequential Error-Weight Boosting with Decision Stumps)
3. Gradient Boosting (Sequential Gradient/Residual Fitting to reduce Bias)
4. XGBoost (Extreme Gradient Boosting with L1/L2 Regularization & Histogram Splitting)

=============================================================================
MATHEMATICAL & THEORETICAL COMPARISON:
-----------------------------------------------------------------------------
1. Bagging vs. Boosting:
   - Bagging (Random Forest): Trains parallel independent deep trees on bootstrap
     samples. Averages predictions to drastically reduce VARIANCE without increasing BIAS.
   - Boosting (AdaBoost / GB / XGB): Trains sequential shallow trees. Each tree learns
     to correct the errors of the preceding ensemble, reducing BIAS.

2. AdaBoost Sample Weight Updating:
   - Initial sample weights: w_i = 1 / N
   - Error of stump m: ε_m = ∑_{y_i ≠ h_m(x_i)} w_i / ∑ w_i
   - Estimator weight: α_m = 1/2 * ln( (1 - ε_m) / ε_m )
   - Weight update: w_i ← w_i * exp( - α_m * y_i * h_m(x_i) ) (Incorrect samples get heavier)

3. Gradient Boosting Residual Descent:
   - Loss function L(y, F(x)) = logloss
   - Pseudo-residuals: r_{i, m} = - [ ∂L(y_i, F(x_i)) / ∂F(x_i) ]_{F = F_{m-1}}
   - Fit weak tree h_m(x) on residuals r_{i, m}
   - Update ensemble: F_m(x) = F_{m-1}(x) + η * h_m(x) (where η is learning_rate)

4. XGBoost Regularized Objective:
   - Obj^(t) = ∑_{i=1}^N [ g_i * f_t(x_i) + 1/2 * h_i * f_t^2(x_i) ] + γ * T + 1/2 * λ * ∑ w_j^2 + α * ∑ |w_j|
   - Uses 1st (g_i) and 2nd order (h_i) Taylor expansion.
=============================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    ADA_N_ESTIMATORS_RANGE,
    CV_SPLITS_ENSEMBLE,
    GB_N_ESTIMATORS_RANGE,
    RF_N_ESTIMATORS_RANGE,
    SEED,
    XGB_N_ESTIMATORS_RANGE,
)


class RandomForestModel:
    """Random Forest Classifier with Out-of-Fold Cross Validation tuning."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        random_state: int = SEED,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1,
        )

    @staticmethod
    def find_optimal_n_estimators(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_range=RF_N_ESTIMATORS_RANGE,
        cv_splits: int = CV_SPLITS_ENSEMBLE,
        max_depth: int = 5,
        random_state: int = SEED,
    ) -> Tuple[int, List[float]]:
        """
        Find optimal n_estimators for Random Forest using Stratified K-Fold CV.

        HINT:
            1. cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            2. For n in n_range:
               rf = RandomForestClassifier(n_estimators=n, max_depth=max_depth, random_state=random_state, n_jobs=-1)
               scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
            3. Find best n = list(n_range)[np.argmax(scores_list)]
            4. Return (best_n, scores_list)
        """
        # TODO [USER IMPLEMENTATION]:
        raise NotImplementedError("Implement `find_optimal_n_estimators` for RandomForestModel.")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, n_estimators: Optional[int] = None):
        """Fit Random Forest model."""
        if n_estimators is not None:
            self.n_estimators = n_estimators
            self.model.set_params(n_estimators=self.n_estimators)
        # TODO [USER IMPLEMENTATION]:
        # Fit self.model and return self
        raise NotImplementedError("Implement `fit` for RandomForestModel.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class AdaBoostModel:
    """AdaBoost Classifier with Decision Stump Weak Learners."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        base_max_depth: int = 1,
        algorithm: str = "SAMME",
        random_state: int = SEED,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_max_depth = base_max_depth
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=self.base_max_depth, random_state=self.random_state),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state,
        )

    @staticmethod
    def find_optimal_n_estimators(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_range=ADA_N_ESTIMATORS_RANGE,
        cv_splits: int = CV_SPLITS_ENSEMBLE,
        learning_rate: float = 0.1,
        random_state: int = SEED,
    ) -> Tuple[int, List[float]]:
        """
        Find optimal n_estimators for AdaBoost.

        HINT:
            1. StratifiedKFold CV
            2. For each n in n_range, evaluate AdaBoostClassifier(n_estimators=n, learning_rate=learning_rate)
            3. Return (best_n, scores_list)
        """
        # TODO [USER IMPLEMENTATION]:
        raise NotImplementedError("Implement `find_optimal_n_estimators` for AdaBoostModel.")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, n_estimators: Optional[int] = None):
        """Fit AdaBoost model."""
        if n_estimators is not None:
            self.n_estimators = n_estimators
            self.model.set_params(n_estimators=self.n_estimators)
        # TODO [USER IMPLEMENTATION]:
        raise NotImplementedError("Implement `fit` for AdaBoostModel.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class GradientBoostingModel:
    """Gradient Boosting Classifier (Residual descent trees)."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        subsample: float = 1.0,
        random_state: int = SEED,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            random_state=self.random_state,
        )

    @staticmethod
    def find_optimal_n_estimators(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_range=GB_N_ESTIMATORS_RANGE,
        cv_splits: int = CV_SPLITS_ENSEMBLE,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = SEED,
    ) -> Tuple[int, List[float]]:
        """Find optimal n_estimators for GradientBoosting."""
        # TODO [USER IMPLEMENTATION]:
        raise NotImplementedError("Implement `find_optimal_n_estimators` for GradientBoostingModel.")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, n_estimators: Optional[int] = None):
        """Fit Gradient Boosting model."""
        if n_estimators is not None:
            self.n_estimators = n_estimators
            self.model.set_params(n_estimators=self.n_estimators)
        # TODO [USER IMPLEMENTATION]:
        raise NotImplementedError("Implement `fit` for GradientBoostingModel.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class XGBoostModel:
    """Extreme Gradient Boosting (XGBoost) Classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        subsample: float = 1.0,
        tree_method: str = "hist",
        random_state: int = SEED,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.tree_method = tree_method
        self.random_state = random_state
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=self.tree_method,
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1,
        )

    @staticmethod
    def find_optimal_n_estimators(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_range=XGB_N_ESTIMATORS_RANGE,
        cv_splits: int = CV_SPLITS_ENSEMBLE,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        use_gpu: bool = False,
        random_state: int = SEED,
    ) -> Tuple[int, List[float]]:
        """
        Find optimal n_estimators for XGBoost.

        HINT:
            1. tree_method = 'gpu_hist' if use_gpu else 'hist'
            2. For each n in n_range:
               xgb = XGBClassifier(n_estimators=n, learning_rate=learning_rate, max_depth=max_depth,
                                   objective='binary:logistic', eval_metric='logloss',
                                   tree_method=tree_method, random_state=random_state)
               scores = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='accuracy')
            3. Return (best_n, scores_list)
        """
        # TODO [USER IMPLEMENTATION]:
        raise NotImplementedError("Implement `find_optimal_n_estimators` for XGBoostModel.")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, n_estimators: Optional[int] = None):
        """Fit XGBoost model."""
        if n_estimators is not None:
            self.n_estimators = n_estimators
            self.model.set_params(n_estimators=self.n_estimators)
        # TODO [USER IMPLEMENTATION]:
        raise NotImplementedError("Implement `fit` for XGBoostModel.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
