"""
data_loader.py
--------------
Responsible ONLY for reading the raw Titanic CSV files from disk and
returning them as pandas DataFrames.

It intentionally does NOT perform any cleaning, transformation, or
feature engineering - that responsibility belongs to preprocessing.py
and feature_engineering.py, keeping each module focused on a single job.
"""

from pathlib import Path

import pandas as pd


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a single CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file to load.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If no file exists at the given path.
        pd.errors.EmptyDataError: If the file exists but contains no data.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find data file at '{file_path}'. "
            "Make sure the Titanic dataset CSVs (train.csv, test.csv, "
            "gender_submission.csv) are placed inside the 'data/' folder. "
            "They can be downloaded from https://www.kaggle.com/c/titanic/data"
        )

    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError as error:
        raise pd.errors.EmptyDataError(f"The file '{file_path}' exists but is empty.") from error


def load_training_data(data_dir: Path) -> pd.DataFrame:
    """Load the Kaggle Titanic training dataset (train.csv).

    Args:
        data_dir: Directory containing the Titanic CSV files.

    Returns:
        The raw training DataFrame, including the 'Survived' target column.
    """
    training_data = load_csv(data_dir / "train.csv")
    print(training_data.head())
    print(training_data.columns)
    return training_data


def load_testing_data(data_dir: Path) -> pd.DataFrame:
    """Load the Kaggle Titanic test dataset (test.csv).

    Args:
        data_dir: Directory containing the Titanic CSV files.

    Returns:
        The raw test DataFrame (no 'Survived' column - this is the set
        Kaggle uses for scoring submissions).
    """
    testing_data = load_csv(data_dir / "test.csv")
    print(testing_data.head())
    print(testing_data.columns)
    return testing_data
