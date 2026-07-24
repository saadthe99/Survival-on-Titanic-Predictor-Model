"""
model.py
---------
Creation, training, saving, and loading of the Logistic Regression
model used to predict Titanic survival.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


def create_model(max_iter: int = 10000) -> LogisticRegression:
    """Create a Logistic Regression classifier.

    Args:
        max_iter: Maximum number of solver iterations. 10000 matches
            the original notebook's setting and gives the solver
            plenty of room to converge on this dataset.

    Returns:
        An untrained LogisticRegression instance.
    """
    return LogisticRegression(max_iter=max_iter)


def train_model(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """Fit the model on the training features and target.

    Args:
        model: The (untrained) LogisticRegression instance.
        X_train: Training feature matrix.
        y_train: Training target vector ('Survived').

    Returns:
        The same model instance, now fitted to the training data.
    """
    model.fit(X_train, y_train)
    return model


def save_model(model: LogisticRegression, file_path: Path) -> None:
    """Persist a trained model to disk using joblib.

    Args:
        model: The trained model to save.
        file_path: Destination file path (e.g. outputs/model.joblib).
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path: Path) -> LogisticRegression:
    """Load a previously trained model from disk.

    Args:
        file_path: Path to a .joblib file created by save_model().

    Returns:
        The loaded, ready-to-use model.

    Raises:
        FileNotFoundError: If no model file exists at the given path.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"No saved model found at '{file_path}'. Run main.py first to train and save one."
        )
    return joblib.load(file_path)
