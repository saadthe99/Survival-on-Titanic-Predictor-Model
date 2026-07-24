"""
predict.py
-----------
Generates predictions on Kaggle's test.csv and writes them out in the
submission.csv format Kaggle expects: two columns, 'PassengerId' and
'Survived'.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


def predict_and_generate_submission(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Predict survival on the Kaggle test set and save a submission file.

    Args:
        model: A trained classifier.
        X_test: Fully preprocessed and encoded Kaggle test features.
            Must include a 'PassengerId' column (used for the submission,
            not as a feature the model needs to make sense of).
        output_path: Where to write the resulting submission.csv.

    Returns:
        The submission DataFrame that was written to disk.

    Raises:
        KeyError: If 'PassengerId' is missing from X_test.
    """
    if "PassengerId" not in X_test.columns:
        raise KeyError(
            "X_test must contain a 'PassengerId' column to build a Kaggle submission file."
        )

    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": X_test["PassengerId"],
        "Survived": predictions,
    })
    print(submission.head())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return submission
