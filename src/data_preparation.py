"""
Module for loading and preprocessing the car price dataset.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

    df = pd.read_csv(config["data"]["path"])
    logger.info("Dataset shape: %s", df.shape)

    # Feature engineering
    df = df.copy()
    current_year = 2025
    df["car_age"] = current_year - df["year"]
    numeric_features = config["data"]["features"]["numeric"] + ["car_age"]
    categorical_features = config["data"]["features"]["categorical"]

    # Separate target
    target = df[config["data"]["target_column"]].values.astype(np.float32)
    X = df[numeric_features + categorical_features].copy()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

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
