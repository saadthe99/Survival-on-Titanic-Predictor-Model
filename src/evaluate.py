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
