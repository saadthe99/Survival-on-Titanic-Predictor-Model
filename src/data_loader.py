from pathlib import Path

import pandas as pd


def load_csv(file_path: Path) -> pd.DataFrame:
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
    
    training_data = load_csv(data_dir / "train.csv")
    print(training_data.head())
    print(training_data.columns)
    return training_data


def load_testing_data(data_dir: Path) -> pd.DataFrame:
    
    testing_data = load_csv(data_dir / "test.csv")
    print(testing_data.head())
    print(testing_data.columns)
    return testing_data
