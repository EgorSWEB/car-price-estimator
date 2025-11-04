"""
Module for creating a preprocessor and feature engineering.
"""
from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_preprocessor(
        numeric_features: List[str],
        categorical_features: List[str]
    ) -> ColumnTransformer:
    """
    Create preprocessor.

    Args:
        numeric_features (List[str]): list of num features
        categorical_features (List[str]): list of cat features

    Returns:
        result (ColumnTransformer): result preprocessor
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

def add_car_age(df: pd.DataFrame, current_year: int = 2025) -> pd.DataFrame:
    """
    Add the car age to the dataset.

    Args:
        df (pd.DataFrame): input dataframe
        current_year (int, optional): current year. Defaults to 2025.

    Returns:
        result_df (pd.DataFrame): dataframe with car_age column 
    """
    df = df.copy()
    df["car_age"] = current_year - df["year"]
    return df
