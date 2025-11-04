"""
Module for loading the car price dataset.
"""
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """Load data from csv to DataFrame.

    Args:
        path (str): input path

    Returns:
        df (pd.DataFrame): result DataFrame
    """
    return pd.read_csv(path)