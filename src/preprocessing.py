from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  
import pandas as pd


def drop_columns(dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    
    existing_columns = [col for col in columns if col in dataframe.columns]
    return dataframe.drop(columns=existing_columns)


def fill_missing_embarked(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    dataframe["Embarked"] = dataframe["Embarked"].fillna(dataframe["Embarked"].mode()[0])
    return dataframe


def explore_age_distribution(dataframe: pd.DataFrame, figures_dir: Optional[Path] = None) -> None:

    print(dataframe["Age"].describe())
    print(dataframe["Age"].skew())

    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        dataframe["Age"].hist(bins=30, ax=ax)
        ax.set_title("Age Distribution")
        output_path = figures_dir / "age_distribution.png"
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Saved Age distribution histogram to {output_path}")


def fill_missing_age(dataframe: pd.DataFrame, reference_median: Optional[float] = None) -> pd.DataFrame:
    median_value = reference_median if reference_median is not None else dataframe["Age"].median()
    dataframe["Age"] = dataframe["Age"].fillna(median_value)
    return dataframe


def fill_missing_fare(dataframe: pd.DataFrame, reference_median: float) -> pd.DataFrame:
    dataframe["Fare"] = dataframe["Fare"].fillna(reference_median)
    return dataframe
