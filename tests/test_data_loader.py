"""Tests for loading data."""
import os
import tempfile
import pandas as pd
import pytest
from src.data_loader import load_csv


def test_load_csv_valid_file(sample_car_data):
    """Test: Correct loading of a CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_car_data.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        df = load_csv(temp_path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == sample_car_data.shape
        assert list(df.columns) == list(sample_car_data.columns)
    finally:
        os.unlink(temp_path)


def test_load_csv_empty_file():
    """Test: loading an empty file causes an exception."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("")
        temp_path = f.name

    try:
        with pytest.raises(pd.errors.EmptyDataError):
            load_csv(temp_path)
    finally:
        os.unlink(temp_path)