"""
preprocessing.py
-----------------
Cleaning and preparation of the raw Titanic data:
- Dropping columns that are not useful for modelling
- Handling missing values ('Embarked', 'Age', 'Fare')
- An optional exploratory histogram of the 'Age' distribution

The order and logic of every operation mirror the original notebook
exactly - only the structure (turning notebook cells into reusable
functions) has changed.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib

# Use a non-interactive backend so figures can be saved to disk instead
# of popping up a blocking window - important when this runs as a
# plain script (main.py) rather than inside a Jupyter notebook.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import pandas as pd


def drop_columns(dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drop one or more columns from a DataFrame.

    Only columns that actually exist in the DataFrame are dropped, so
    this function is safe to call even if a column was already removed.

    Args:
        dataframe: The DataFrame to modify.
        columns: List of column names to drop.

    Returns:
        A new DataFrame with the specified columns removed.
    """
    existing_columns = [col for col in columns if col in dataframe.columns]
    return dataframe.drop(columns=existing_columns)


def fill_missing_embarked(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill missing 'Embarked' values with the most frequent value (mode).

    Args:
        dataframe: DataFrame containing an 'Embarked' column.

    Returns:
        The DataFrame with missing 'Embarked' values filled.
    """
    dataframe["Embarked"] = dataframe["Embarked"].fillna(dataframe["Embarked"].mode()[0])
    return dataframe


def explore_age_distribution(dataframe: pd.DataFrame, figures_dir: Optional[Path] = None) -> None:
    """Print 'Age' summary statistics/skew and save a histogram figure.

    This mirrors the exploratory-data-analysis step from the original
    notebook (``trainingdata["Age"].describe()``, ``.skew()``, and
    ``.hist()``). Instead of calling ``plt.show()`` (which blocks
    execution when run as a script), the figure is written to
    ``outputs/figures/`` so it can be reviewed afterwards.

    Args:
        dataframe: DataFrame containing an 'Age' column.
        figures_dir: Folder to save the histogram image into. If None,
            the plot is skipped (statistics are still printed).
    """
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
    """Fill missing 'Age' values with a median.

    Args:
        dataframe: DataFrame containing an 'Age' column.
        reference_median: Median value to fill with. If None, the
            median is computed from this same DataFrame (this is what
            the original notebook does for the training set). For the
            Kaggle test set, pass the TRAINING set's median instead so
            information doesn't leak from the test set into the fill value.

    Returns:
        The DataFrame with missing 'Age' values filled.
    """
    median_value = reference_median if reference_median is not None else dataframe["Age"].median()
    dataframe["Age"] = dataframe["Age"].fillna(median_value)
    return dataframe


def fill_missing_fare(dataframe: pd.DataFrame, reference_median: float) -> pd.DataFrame:
    """Fill missing 'Fare' values with a given reference median.

    Args:
        dataframe: DataFrame containing a 'Fare' column.
        reference_median: Median 'Fare' value (from the training set)
            to fill missing entries with.

    Returns:
        The DataFrame with missing 'Fare' values filled.
    """
    dataframe["Fare"] = dataframe["Fare"].fillna(reference_median)
    return dataframe
