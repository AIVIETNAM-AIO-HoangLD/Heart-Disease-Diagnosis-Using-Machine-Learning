"""
Feature Engineering Transformers & Functions.

This module implements domain-specific medical feature transformations:
1. `chol_per_age` = chol / age (Cholesterol normalized by age)
2. `bps_per_age` = trestbps / age (Blood pressure normalized by age)
3. `hr_ratio` = thalach / age (Max heart rate to age ratio)
4. `age_bin` = pd.cut(age, bins=5, labels=False) (Age discretized into 5 bins)

=============================================================================
CLINICAL RATIONALE:
-----------------------------------------------------------------------------
- Cardiovascular risk factors are age-dependent. A high resting blood pressure
  or cholesterol level in a 35-year-old patient carries a different clinical risk
  profile than the same raw value in a 75-year-old patient.
- Creating interaction ratios and age bins helps linear and tree-based models
  capture non-linear risk interactions without needing deep trees.

[IMPROVEMENT HINTS]:
- Polynomial Features: `PolynomialFeatures(degree=2, interaction_only=True)`
- Log Transformations: `np.log1p(df['chol'])` to reduce heavy right tails.
- Framingham Risk proxy features: interaction between `sex`, `age`, and `trestbps`.
=============================================================================
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Generated feature names
GENERATED_NUMERIC = ["chol_per_age", "bps_per_age", "hr_ratio"]
GENERATED_CATEGORICAL = ["age_bin"]


def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate new interaction and discretized features from raw columns.

    Args:
        df: Input DataFrame containing at least 'age', 'chol', 'trestbps', 'thalach'.

    Returns:
        pd.DataFrame: DataFrame with new columns added.

    HINT:
        1. Make a copy: `df = df.copy()`
        2. If 'chol' and 'age' in df.columns:
               df['chol_per_age'] = df['chol'] / df['age']
        3. If 'trestbps' and 'age' in df.columns:
               df['bps_per_age'] = df['trestbps'] / df['age']
        4. If 'thalach' and 'age' in df.columns:
               df['hr_ratio'] = df['thalach'] / df['age']
        5. If 'age' in df.columns:
               df['age_bin'] = pd.cut(df['age'], bins=5, labels=False).astype('category')
        6. Return df
    """
    # TODO [USER IMPLEMENTATION]:
    # 1. Compute domain-specific age ratios
    # 2. Bin age into 5 discrete categories
    # 3. Return transformed DataFrame
    df = df.copy()
    if "chol" in df.columns and "age" in df.columns:
        df["chol_per_age"] = df["chol"] / df["age"]
    if "trestbps" in df.columns and "age" in df.columns:
        df["bps_per_age"] = df["trestbps"] / df["age"]
    if "thalach" in df.columns and "age" in df.columns:
        df["hr_ratio"] = df["thalach"] / df["age"]
    if "age" in df.columns:
        df["age_bin"] = pd.cut(df["age"], bins=5, labels=False).astype("category")
    return df


class AddNewFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible Transformer wrapper for `add_new_features`.
    Enables seamless integration inside a Scikit-Learn Pipeline.
    """

    def __init__(self):
        self.columns_ = None
        self.new_features_ = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit method to record input columns and expected new features."""
        # TODO [USER IMPLEMENTATION]:
        # 1. Record `self.columns_ = X.columns`
        # 2. Check for existence of 'chol', 'trestbps', 'thalach', 'age' to populate `self.new_features_`
        # 3. Return `self`
        self.columns = X.columns
        self.columns_ = []
        if "chol" in X.columns and "age" in X.columns:
            self.new_features_.append("chol_per_age")
        if "trestbps" in X.columns and "age" in X.columns:
            self.new_features_.append("bps_per_age")
        if "thalach" in X.columns and "age" in X.columns:
            self.new_features_.append("hr_ratio")
        if "age" in X.columns:
            self.new_features_.append("age_bin")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply `add_new_features` transformation to input X."""
        # TODO [USER IMPLEMENTATION]
        # 1. Return `add_new_features(X)`
        return add_new_features(X)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return list of all output column names after transformation."""
        if self.columns_ is not None:
            return list(self.columns_) + self.new_features_
        return list(input_features) if input_features is not None else []
