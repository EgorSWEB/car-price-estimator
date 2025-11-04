"""Utilities for working with the configuration."""
import yaml


def load_config(config_path: str) -> dict:
    """
    Loads the configuration from a YAML file.

    Args:
        config_path (str): path to config

    Returns:
        result (dict): result dict
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_required_columns(config: dict) -> list:
    """
    Returns a list of required columns from the config.

    Args:
        config (dict): input config

    Returns:
        columns_list (list): necessary columns list
    """
    numeric_features = config["data"]["features"]["numeric"]
    categorical_features = config["data"]["features"]["categorical"]
    target_column = config["data"]["target_column"]

    return list(set(numeric_features + categorical_features + [target_column]))


def get_data_schema(config: dict) -> dict:
    """
    Returns the data validation scheme.

    Args:
        config (dict): input config

    Returns:
        schema (dict): non_negative schema
    """
    numeric_features = config["data"]["features"]["numeric"]
    target = config["data"]["target_column"]
    schema = {}
    for col in numeric_features + [target]:
        schema[col] = "non_negative"
    return schema
