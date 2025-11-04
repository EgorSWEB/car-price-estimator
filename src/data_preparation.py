"""
Module for loading and preprocessing the car price dataset.
"""
import logging

import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import load_csv
from data_validator import (
    validate_columns,
    validate_types,
    validate_target
)
from preprocessor import create_preprocessor, add_car_age

def load_and_preprocess_data(config):
    """
    Loads and preprocesses the car dataset.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: Processed train/test splits and the fitted preprocessor.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading data from %s", config["data"]["path"])

    df = load_csv(config["data"]["path"])

    check_schema = {feature: "numeric" for feature in config["data"]["features"]["numeric"]}
    check_schema.update(
        {feature: "categorical" for feature in config["data"]["features"]["categorical"]}
    )
    check_schema[config["data"]["target_column"]] = "non-negative"

    validate_columns(df, list(check_schema.keys()))
    validate_types(df, check_schema)
    validate_target(df, config["data"]["target_column"])

    logger.info("Dataset shape: %s", df.shape)

    # Feature engineering
    df = add_car_age(df, 2025)
    numeric_features = config["data"]["features"]["numeric"] + ["car_age"]
    categorical_features = config["data"]["features"]["categorical"]

    # Separate target
    target = df[config["data"]["target_column"]].values.astype(np.float32)
    X = df[numeric_features + categorical_features].copy()

    # Preprocessing pipeline
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    X_processed = preprocessor.fit_transform(X)
    logger.info("Processed feature matrix shape: %s", X_processed.shape)

    # train-deffered split
    deferred_size = config["training"]["test_size"] + config["training"]["val_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        target,
        test_size=deferred_size,
        random_state=config["training"]["random_seed"],
    )
    # val-test split
    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=config["training"]["test_size"]/deferred_size,
        random_state=config["training"]["random_seed"],
    )

    logger.info(
        "Train shape: %s, Val shape: %s, Test shape: %s",
        X_train.shape,
        X_val.shape,
        X_test.shape
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
