"""
Model Evaluation Metrics & Diagnostic Protocols.

This module provides standardized metric calculation functions:
1. Accuracy: (TP + TN) / (TP + TN + FP + FN)
2. Precision: TP / (TP + FP)  -> Crucial to avoid false alarms
3. Recall (Sensitivity): TP / (TP + FN)  -> CRITICAL in healthcare to avoid missing diseased patients
4. F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
5. ROC-AUC: Area Under the Receiver Operating Characteristic Curve
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    y_proba: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive binary classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted class labels.
        y_proba: Predicted probabilities for positive class (disease).

    Returns:
        Dict: Dictionary containing accuracy, classification report, confusion matrix, and optional roc_auc.

    HINT:
        1. acc = accuracy_score(y_true, y_pred)
        2. report = classification_report(y_true, y_pred, output_dict=True)
        3. cm = confusion_matrix(y_true, y_pred)
        4. auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    """
    # TODO [USER IMPLEMENTATION]:
    # Calculate and package evaluation metrics
    raise NotImplementedError("Implement `evaluate_predictions` following the hints.")


def print_classification_summary(
    split_name: str,
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> float:
    """Print standard classification accuracy and text report."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[+] Performance on {split_name} Split:")
    print(f"    Accuracy: {acc:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred))
    return acc
