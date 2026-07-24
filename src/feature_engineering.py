"""
feature_engineering.py
-----------------------
Placeholder module for engineered features (e.g. FamilySize, IsAlone,
Title extracted from Name, Age/Fare binning, etc).

The original notebook does NOT create any additional engineered
features - it trains only on the cleaned/raw columns. This file is
kept (as requested) as an extension point so new features can be
added here in future without having to restructure the rest of the
project. main.py does not currently call any function in this file.
"""

import pandas as pd


def add_family_size(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Example future feature: total family members travelling together.

    NOT used in the current pipeline - provided as a ready-to-use
    template for a future improvement (see README.md).

    Args:
        dataframe: DataFrame containing 'SibSp' and 'Parch' columns.

    Returns:
        DataFrame with an added 'FamilySize' column
        (siblings/spouses + parents/children + the passenger themself).
    """
    dataframe["FamilySize"] = dataframe["SibSp"] + dataframe["Parch"] + 1
    return dataframe


def add_is_alone(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Example future feature: whether a passenger was travelling alone.

    NOT used in the current pipeline - requires 'FamilySize' to already
    exist (see add_family_size above).

    Args:
        dataframe: DataFrame containing a 'FamilySize' column.

    Returns:
        DataFrame with an added 'IsAlone' column (1 if travelling
        alone, 0 otherwise).
    """
    dataframe["IsAlone"] = (dataframe["FamilySize"] == 1).astype(int)
    return dataframe
