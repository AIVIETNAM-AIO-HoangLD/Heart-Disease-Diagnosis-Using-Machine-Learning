"""
Automated Hyperparameter Optimization Module (Optuna & GridSearchCV Hints).

This module provides templates for Bayesian Hyperparameter Tuning with Optuna:
- Replaces coarse 1D manual grid searches with Tree-structured Parzen Estimators (TPE).
- Jointly searches `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and regularizers.

=============================================================================
OPTUNA BAYESIAN OPTIMIZATION ADVANTAGES:
-----------------------------------------------------------------------------
- Grid Search checks all combinations uniformly (computationally expensive O(N^D)).
- Random Search checks arbitrary points without memory of past performance.
- Optuna TPE builds a probabilistic model of past trial performances:
      p(x | y) where y is the validation score
  and samples next candidates that maximize Expected Improvement (EI).
=============================================================================
"""

from typing import Any, Dict, Optional
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import SEED

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None


def optimize_random_forest_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv_splits: int = 5,
    random_state: int = SEED,
) -> Dict[str, Any]:
    """
    [IMPROVEMENT HINT]
    Bayesian Hyperparameter Search for Random Forest using Optuna.

    HINT:
        1. Define objective(trial):
           n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
           max_depth = trial.suggest_int('max_depth', 2, 15)
           min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
           min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
           max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

           model = RandomForestClassifier(...)
           cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
           score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
           return score

        2. study = optuna.create_study(direction='maximize')
        3. study.optimize(objective, n_trials=n_trials)
        4. return study.best_params
    """
    if optuna is None:
        raise ImportError("Optuna is not installed. Install via `pip install optuna`.")

    # TODO [USER IMPLEMENTATION - IMPROVEMENT]:
    # Build Optuna objective function and optimize
    raise NotImplementedError("Implement `optimize_random_forest_optuna`.")


def optimize_xgboost_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv_splits: int = 5,
    random_state: int = SEED,
) -> Dict[str, Any]:
    """
    [IMPROVEMENT HINT]
    Bayesian Hyperparameter Search for XGBoost using Optuna.

    HINT:
        1. Define objective(trial):
           params = {
               'n_estimators': trial.suggest_int('n_estimators', 50, 400, step=25),
               'max_depth': trial.suggest_int('max_depth', 2, 8),
               'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
               'subsample': trial.suggest_float('subsample', 0.6, 1.0),
               'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
               'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
               'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
           }
           xgb = XGBClassifier(**params, objective='binary:logistic', eval_metric='logloss', random_state=random_state)
           score = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='accuracy').mean()
           return score
    """
    if optuna is None:
        raise ImportError("Optuna is not installed. Install via `pip install optuna`.")

    # TODO [USER IMPLEMENTATION - IMPROVEMENT]:
    # Build Optuna objective for XGBoost and optimize
    raise NotImplementedError("Implement `optimize_xgboost_optuna`.")
