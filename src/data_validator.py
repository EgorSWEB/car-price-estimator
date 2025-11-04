"""
Module for validate the car price dataset.
"""
import pandas as pd

def validate_columns(df: pd.DataFrame, required_columns: list):
    """Validate necessary columns exists.

    Args:
        df (pd.DataFrame): input data
        required_columns (list): required columns

    Raises:
        ValueError: missing required columns
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def validate_types(df: pd.DataFrame, schema: dict):
    """Validate necessary types of the columns.

    Args:
        df (pd.DataFrame): input data
        schema (dict): required types

    Raises:
        TypeError: Column {col} must be numeric
        ValueError: Column {col} must be non-negative
    """
    for col, expected_type in schema.items():
        if col not in df.columns:
            continue
        if (not pd.api.types.is_numeric_dtype(df[col]) and
            (expected_type in {"numeric", "non_negative"})):
            raise TypeError(f"Column {col} must be numeric")
        if expected_type == "non_negative" and (df[col] < 0).any():
            raise ValueError(f"Column {col} must be non-negative")

def validate_target(df: pd.DataFrame, target_col: str):
    """Validate NaN and non-negative in the target column.

    Args:
        df (pd.DataFrame): input data
        target_col (str): target column

    Raises:
        ValueError: Target column {target_col} contains NaN
        ValueError: Target {target_col} must be non-negative
    """
    if df[target_col].isnull().any():
        raise ValueError(f"Target column {target_col} contains NaN")
    if (df[target_col] < 0).any():
        raise ValueError(f"Target {target_col} must be non-negative")
