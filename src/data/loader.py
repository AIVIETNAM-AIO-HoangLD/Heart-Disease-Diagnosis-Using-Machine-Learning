"""
Data Loader & Dataset Splitting Module.

This module is responsible for:
1. Ingesting raw CSV data (which comes without headers from UCI).
2. Assigning standard medical column names.
3. Coercing numerical columns with missing string markers ('?') to float.
4. Binarizing the multi-class target (0 = no disease, >0 = disease).
5. Performing stratified splitting (Train:Val:Test = 80:10:10).

=============================================================================
MATHEMATICAL & CONCEPTUAL BACKGROUND:
-----------------------------------------------------------------------------
1. Target Binarization:
   The original Cleveland dataset records angiographic disease status:
   - 0: < 50% diameter narrowing (absence of significant disease)
   - 1, 2, 3, 4: > 50% diameter narrowing (increasing severity of CAD)
   For binary diagnosis, we apply the mapping:
       y_binary = 1 if y > 0 else 0

2. Stratified Splitting:
   In medical datasets with limited samples (N=303), a standard random split
   might distort the ratio of positive/negative disease cases across sets.
   Stratified sampling guarantees:
       P(y = 1 | Train) ≈ P(y = 1 | Val) ≈ P(y = 1 | Test) ≈ P(y = 1 | All)
=============================================================================
"""

from re import X
from pandas._typing import T
from pathlib import Path
from typing import Dict, Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ALL_COLUMNS,
    COERCE_NUMERIC_COLUMNS,
    SEED,
    TARGET,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


def load_raw_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load raw Cleveland dataset, set column names, handle types and binarize target.

    Args:
        data_path: Path to the raw CSV file.

    Returns:
        pd.DataFrame: Cleaned raw DataFrame with 14 columns.

    HINT:
        1. Use `pd.read_csv(data_path, header=None)`
        2. Set `df.columns = ALL_COLUMNS`
        3. Loop over `COERCE_NUMERIC_COLUMNS` and apply `pd.to_numeric(df[col], errors='coerce')`
           to convert any string '?' into `np.nan`.
        4. Binarize target: `df[TARGET] = (df[TARGET] > 0).astype(int)`
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    # TODO [USER IMPLEMENTATION]:
    # 1. Read raw CSV using pandas without header
    # 2. Assign standard column names from ALL_COLUMNS
    # 3. Convert numeric columns with possible '?' to numeric float (errors='coerce')
    # 4. Convert target column to binary (0 and 1)
    # 5. Return the prepared DataFrame
    # raise NotImplementedError("Implement `load_raw_data` following the hints in docstring.")
    data_frame = pd.read_csv(data_path, header=None)
    data_frame.columns = ALL_COLUMNS
    for col in COERCE_NUMERIC_COLUMNS:
        data_frame[col] = pd.to_numeric(data_frame[col], errors="coerce")
    data_frame[TARGET] = (data_frame[TARGET] > 0).astype(int)
    return data_frame

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Perform two-stage Stratified Train/Val/Test split (e.g., 80% Train, 10% Val, 10% Test).

    Args:
        df: Cleaned raw DataFrame containing features and target.
        train_ratio: Proportion for training set (default: 0.8).
        val_ratio: Proportion for validation set (default: 0.1).
        test_ratio: Proportion for test set (default: 0.1).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)

    HINT:
        1. Separate X = df.drop(columns=[TARGET]), y = df[TARGET]
        2. Stage 1: Split into Train (80%) and Temp (20%) using `train_test_split`:
           - `test_size = val_ratio + test_ratio` (e.g., 0.2)
           - `stratify = y`
        3. Stage 2: Split Temp into Val (50% of Temp = 10% total) and Test (50% of Temp = 10% total):
           - `test_size = 0.5`
           - `stratify = y_temp`
        4. Return X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state, 
        stratify=y
    )
    # Stage 2: Split 20% Temp equally (50% each) into 10% Val and 10% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, 
        random_state=random_state, 
        stratify=y_temp
    )


    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # TODO [USER IMPLEMENTATION]:
    # 1. Separate features X and target y
    # 2. Stage 1 split: train vs (val + test)
    # 3. Stage 2 split: val vs test
    # 4. Return (X_train, X_val, X_test, y_train, y_val, y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(splits_dict: Dict[str, pd.DataFrame], output_dir: Union[str, Path]) -> None:
    """
    Save train, val, and test DataFrames to CSV files in output_dir.

    Args:
        splits_dict: Dictionary of {split_name: DataFrame} (e.g., {'raw_train': df, ...})
        output_dir: Destination folder path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, df in splits_dict.items():
        file_path = output_path / f"{name}.csv"
        df.to_csv(file_path, index=False)
        print(f"[✓] Saved split {name} ({df.shape[0]} rows, {df.shape[1]} cols) -> {file_path}")
