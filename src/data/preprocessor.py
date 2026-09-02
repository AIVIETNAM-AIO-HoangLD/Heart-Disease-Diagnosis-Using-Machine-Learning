"""
Data Preprocessing Pipeline Module.

This module constructs Scikit-Learn pipelines to handle:
1. Missing value imputation (Median for numeric, Most Frequent / Mode for categorical).
2. Numerical feature scaling (StandardScaler, MinMaxScaler, RobustScaler).
3. Categorical encoding (MinMaxScaler or OneHotEncoder).
4. Integration via ColumnTransformer to avoid Data Leakage.

=============================================================================
DATA LEAKAGE PREVENTION PRINCIPLE:
-----------------------------------------------------------------------------
Data leakage occurs when information from outside the training dataset (such as
Validation or Test distributions, means, medians, or variance) is used to create
the model.

RULE:
- ALWAYS run `pipeline.fit_transform(X_train, y_train)` ONLY on the training split.
- ALWAYS run `pipeline.transform(X_val)` and `pipeline.transform(X_test)` using
  the learned parameters (mean, variance, mode) from the training set.
=============================================================================
"""

from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from src.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_numeric_pipeline(
    impute_strategy: str = "median",
    scaler_type: str = "standard",
) -> Pipeline:
    """
    Construct a numerical preprocessing Pipeline.

    Args:
        impute_strategy: Strategy for SimpleImputer ('median', 'mean').
        scaler_type: Scaling method ('standard', 'minmax', 'robust').

    Returns:
        Pipeline: Scikit-learn Pipeline for numeric columns.

    HINT:
        1. Create `imputer = SimpleImputer(strategy=impute_strategy)`
        2. Create scaler based on `scaler_type`:
           - 'standard': `StandardScaler()`
           - 'minmax': `MinMaxScaler()`
           - 'robust': `RobustScaler()` (Useful for outliers in chol/trestbps)
        3. Return `Pipeline(steps=[('imputer', imputer), ('scaler', scaler)])`
    """
    # TODO [USER IMPLEMENTATION]:
    # 1. Select appropriate scaler
    # 2. Build and return Pipeline([('imputer', ...), ('scaler', ...)])
    # raise NotImplementedError("Implement `build_numeric_pipeline` following the hints.")
    num_proc = Pipeline([("imputer", SimpleImputer(strategy=impute_strategy)),("scaler", StandardScaler(scaler_type))])
    return num_proc

def build_categorical_pipeline(
    impute_strategy: str = "most_frequent",
    encoder_type: str = "minmax",
) -> Pipeline:
    """
    Construct a categorical preprocessing Pipeline. 

    Args:
        impute_strategy: Strategy for SimpleImputer ('most_frequent').
        encoder_type: Encoding method:
            - 'minmax': MinMaxScaler() (Used in Paper Part 1 Raw pipeline)
            - 'onehot': OneHotEncoder(handle_unknown='ignore', sparse_output=False) (Used in FE pipeline)

    Returns:
        Pipeline: Scikit-learn Pipeline for categorical columns.

    HINT:
        1. Create `imputer = SimpleImputer(strategy=impute_strategy)`
        2. Create encoder based on `encoder_type`:
           - If 'minmax': `MinMaxScaler()`
           - If 'onehot': `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
        3. Return `Pipeline(steps=[('imputer', imputer), ('encoder', encoder)])`
    """
    # TODO [USER IMPLEMENTATION]:
    # 1. Choose encoding / scaling strategy
    # 2. Return Pipeline([('imputer', ...), ('encoder', ...)])
    raise NotImplementedError("Implement `build_categorical_pipeline` following the hints.")
    if encoder_type == "onehot":
        cat_proc = Pipeline([("imputer", SimpleImputer(strategy=impute_strategy)),("encoder", OneHotEncoder())])
    elif encoder_type == "minmax":
        cat_proc = Pipeline([("imputer", SimpleImputer(strategy=impute_strategy)),("encoder", MinMaxScaler())])
    return cat_proc


def build_preprocessor(
    numeric_cols: List[str] = NUMERIC_COLUMNS,
    categorical_cols: List[str] = CATEGORICAL_COLUMNS,
    num_scaler: str = "standard",
    cat_encoder: str = "minmax",
) -> ColumnTransformer:
    """
    Combine numeric and categorical pipelines into a ColumnTransformer.

    Args:
        numeric_cols: List of numeric column names.
        categorical_cols: List of categorical column names.
        num_scaler: 'standard', 'minmax', or 'robust'.
        cat_encoder: 'minmax' or 'onehot'.

    Returns:
        ColumnTransformer: Preprocessing transformer.

    HINT:
        1. Build `num_pipe = build_numeric_pipeline(scaler_type=num_scaler)`
        2. Build `cat_pipe = build_categorical_pipeline(encoder_type=cat_encoder)`
        3. Return `ColumnTransformer(transformers=[
               ('num', num_pipe, numeric_cols),
               ('cat', cat_pipe, categorical_cols)
           ], verbose_feature_names_out=False).set_output(transform='pandas')`
    """
    # TODO [USER IMPLEMENTATION]:
    # 1. Instantiate numeric and categorical sub-pipelines
    # 2. Bundle them in ColumnTransformer
    # 3. Configure `.set_output(transform='pandas')` so transforms return DataFrames
    raise NotImplementedError("Implement `build_preprocessor` following the hints.")
    num_pipe = build_numeric_pipeline(scaler_type=num_scaler)
    cat_pipe = build_categorical_pipeline(encoder_type=cat_encoder)
    return ColumnTransformer(transformers=[("num",num_pipe, numeric_cols),("cat",cat_pipe,categorical_cols)],verbose_feature_names_out=False).set_output(transform="pandas")

