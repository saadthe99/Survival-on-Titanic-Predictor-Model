from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


def create_model(max_iter: int = 10000) -> LogisticRegression:
    
    return LogisticRegression(max_iter=max_iter)


def train_model(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    
    model.fit(X_train, y_train)
    return model


def save_model(model: LogisticRegression, file_path: Path) -> None:
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path: Path) -> LogisticRegression:
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"No saved model found at '{file_path}'. Run main.py first to train and save one."
        )
    return joblib.load(file_path)
