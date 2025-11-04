"""Tests for validating a real data file from the repository."""
import os
import pytest
from src.data_loader import load_csv
from src.data_validator import validate_columns, validate_target, validate_types
from src.config_loader import load_config, get_required_columns, get_data_schema


CONFIG_PATH = "config/train_config.yaml"
DATA_PATH = "data/car_data.csv"


@pytest.fixture(scope="session")
def config():
    """Loads the configuration once for all tests."""
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def required_columns(config):
    """Gets the required columns from the config."""
    return get_required_columns(config)


@pytest.fixture(scope="session")
def data_schema(config):
    """Gets the data schema from the config."""
    return get_data_schema(config)


def test_car_data_file_exists():
    """Verification: The data file exists."""
    assert os.path.isfile(DATA_PATH), f"Data file not found: {DATA_PATH}"


def test_car_data_has_required_columns(required_columns):
    """Verification: the file contains all the columns from the config."""
    df = load_csv(DATA_PATH)
    validate_columns(df, required_columns)


def test_car_data_target_valid(config):
    """Verification: the target variable is correct."""
    df = load_csv(DATA_PATH)
    validate_target(df, config["data"]["target_column"])


def test_car_data_types_and_ranges(data_schema):
    """Checking: types and ranges from the config."""
    df = load_csv(DATA_PATH)
    validate_types(df, data_schema)


def test_car_data_no_empty_strings(config):
    """Проверка: нет пустых строк в категориальных колонках."""
    df = load_csv(DATA_PATH)
    categorical_cols = config["data"]["features"]["categorical"]
    for col in categorical_cols:
        if col in df.columns:
            assert not (df[col] == "").any(), f"Empty strings found in {col}"
            assert not df[col].isnull().any(), f"NaN values found in {col}"