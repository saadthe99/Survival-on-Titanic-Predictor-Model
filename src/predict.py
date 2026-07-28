from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


def predict_and_generate_submission(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    
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
