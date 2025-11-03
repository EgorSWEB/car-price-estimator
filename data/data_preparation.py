# src/data_preparation.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
import os

def load_and_preprocess_data(config):
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {config['data']['path']}")
    
    df = pd.read_csv(config["data"]["path"])
    logger.info(f"Dataset shape: {df.shape}")

    # Feature engineering
    df = df.copy()
    current_year = 2025
    df["Car_Age"] = current_year - df["Year"]
    numeric_features = config["data"]["features"]["numeric"] + ["Car_Age"]
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
    logger.info(f"Processed feature matrix shape: {X_processed.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, target,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_seed"]
    )

    return X_train, X_test, y_train, y_test, preprocessor