"""Generating basic statistics based on data."""
import json
import pandas as pd
from pathlib import Path


def profile_data(df: pd.DataFrame, output_path: str):
    """
    Generates basic statistics based on numerical and categorical criteria.

    Args:
        df (pd.DataFrame): the source dataset
        output_path (str): the path for saving the report (JSON)
    """
    report = {}

    # Числовые признаки
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        report["numeric"] = df[numeric_cols].describe().to_dict()

    # Категориальные признаки
    categorical_cols = df.select_dtypes(exclude=["number"]).columns
    report["categorical"] = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts().to_dict()
        report["categorical"][col] = {
            "n_unique": df[col].nunique(),
            "most_common": df[col].mode().iloc[0] if not df[col].mode().empty else None,
            "value_counts": value_counts
        }

    # Общая информация
    report["general"] = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict()
    }

    # Сохранение
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
