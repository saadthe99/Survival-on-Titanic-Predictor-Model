"""
evaluate.py
------------
Model evaluation: accuracy, precision, recall, F1 score, confusion
matrix, and the full scikit-learn classification report - all
computed on the held-out validation split.
"""

from typing import Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute and print the standard binary classification metrics.

    Args:
        y_true: Ground-truth 'Survived' labels from the validation split.
        y_pred: Model-predicted labels for the same samples.

    Returns:
        A dictionary with keys 'accuracy', 'precision', 'recall', and
        'f1_score' so the results can also be used programmatically
        (e.g. logging results or comparing models later).
    """
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)

    precision = precision_score(y_true, y_pred)
    print("Precision:", precision)

    recall = recall_score(y_true, y_pred)
    print("Recall:", recall)

    f1 = f1_score(y_true, y_pred)
    print("F1 Score:", f1)

    print(classification_report(y_true, y_pred))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
