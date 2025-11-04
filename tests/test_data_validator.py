"""Tests for data validation."""
import pandas as pd
import pytest
from src.data_validator import (
    validate_columns,
    validate_types,
    validate_target
)


def test_validate_columns_success(sample_car_data):
    """Test: All required columns are present."""
    required = ["year", "selling_price"]
    validate_columns(sample_car_data, required)


def test_validate_columns_missing():
    """Test: The required column is missing."""
    df = pd.DataFrame({"year": [2020]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(df, ["year", "selling_price"])


def test_validate_target_nan():
    """Test: The target variable contains NaN."""
    df = pd.DataFrame({"selling_price": [1.0, None]})
    with pytest.raises(ValueError, match="contains NaN"):
        validate_target(df, "selling_price")


def test_validate_target_negative():
    """Test: The target variable is negative."""
    df = pd.DataFrame({"selling_price": [-1.0]})
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_target(df, "selling_price")


def test_validate_types_non_negative():
    """Test: checking the non-negativity of a numeric column."""
    df = pd.DataFrame({"km_driven": [-100]})
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_types(df, {"km_driven": "non_negative"})


def test_validate_types_numeric_ok():
    """Test: The numeric column is correct."""
    df = pd.DataFrame({"year": [2020, 2021]})
    validate_types(df, {"year": "numeric"})


def test_validate_types_non_numeric():
    """Test: A non-numeric column is marked as numeric."""
    df = pd.DataFrame({"year": ["2020", "2021"]})
    with pytest.raises(TypeError, match="must be numeric"):
        validate_types(df, {"year": "numeric"})