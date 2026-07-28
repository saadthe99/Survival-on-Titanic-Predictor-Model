from typing import Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_sex(
    dataframe: pd.DataFrame,
    encoder: Optional[LabelEncoder] = None,
) -> Tuple[pd.DataFrame, LabelEncoder]:

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
