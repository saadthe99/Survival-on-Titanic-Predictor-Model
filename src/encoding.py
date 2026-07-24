"""
encoding.py
------------
Categorical encoding for the Titanic dataset:
- Label encoding for 'Sex' (binary category: male/female)
- One-hot encoding for 'Embarked' (three categories: C/Q/S)

Encoders are fit ONCE on the training data and then reused (via
``.transform``, not ``.fit_transform``) on the Kaggle test data -
exactly as in the original notebook. This avoids data leakage (the
test set never influences how categories are encoded) and guarantees
train/test end up with the same encoded column structure.
"""

from typing import Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_sex(
    dataframe: pd.DataFrame,
    encoder: Optional[LabelEncoder] = None,
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Label-encode the 'Sex' column ('male'/'female' -> 0/1).

    Args:
        dataframe: DataFrame containing a 'Sex' column.
        encoder: An already-fitted LabelEncoder to reuse (pass this in
            for the test set). If None, a new encoder is created and
            fit on this DataFrame (used for the training set).

    Returns:
        A tuple of (DataFrame with 'Sex' encoded, the LabelEncoder used).
    """
    if encoder is None:
        encoder = LabelEncoder()
        dataframe["Sex"] = encoder.fit_transform(dataframe["Sex"])
    else:
        dataframe["Sex"] = encoder.transform(dataframe["Sex"])
    return dataframe, encoder


def encode_embarked(
    dataframe: pd.DataFrame,
    encoder: Optional[OneHotEncoder] = None,
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """One-hot encode the 'Embarked' column.

    Args:
        dataframe: DataFrame containing an 'Embarked' column.
        encoder: An already-fitted OneHotEncoder to reuse (pass this in
            for the test set). If None, a new encoder is created and
            fit on this DataFrame (used for the training set).

    Returns:
        A tuple of (DataFrame with 'Embarked' replaced by its one-hot
        columns, the OneHotEncoder used).
    """
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_values = encoder.fit_transform(dataframe[["Embarked"]])
    else:
        encoded_values = encoder.transform(dataframe[["Embarked"]])

    encoded_df = pd.DataFrame(
        encoded_values,
        columns=encoder.get_feature_names_out(["Embarked"]),
        index=dataframe.index,
    )

    dataframe = dataframe.drop(columns=["Embarked"])
    dataframe = pd.concat([dataframe, encoded_df], axis=1)
    return dataframe, encoder
