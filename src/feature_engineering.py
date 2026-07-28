import pandas as pd


def add_family_size(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    dataframe["FamilySize"] = dataframe["SibSp"] + dataframe["Parch"] + 1
    return dataframe


def add_is_alone(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    dataframe["IsAlone"] = (dataframe["FamilySize"] == 1).astype(int)
    return dataframe
