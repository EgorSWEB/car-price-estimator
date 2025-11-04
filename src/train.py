"""
Training script for the car price prediction model.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
import joblib

from src.data_preparation import load_and_preprocess_data
from src.model import CarPriceMLP, CarPriceMLPConfig


def setup_logging(log_dir: str, level: str = "INFO"):
    """
    Sets up the logging configuration.

    Args:
        log_dir (str): Directory to save log files.
        level (str): Logging level.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(Path(log_dir) / "train.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def train(config_path: str, verbose=False):
    """
    Main training function.

    Args:
        config_path (str): Path to the YAML configuration file.
        verbose (bool): Whether to log more details.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    setup_logging(config["logging"]["log_dir"], config["logging"]["level"])
    logger = logging.getLogger(__name__)
    logger.info("Starting training pipeline")

    # --- Reproducibility ---
    torch.manual_seed(config["training"]["random_seed"])
    np.random.seed(config["training"]["random_seed"])

    # --- Data Loading ---
    # pylint: disable=invalid-name
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = load_and_preprocess_data(config)
    input_dim = X_train.shape[1]

    # --- Convert to Tensors ---
    device = torch.device(config["training"]["device"])
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    # --- Model, Optimizer, Loss ---
    model_config = CarPriceMLPConfig(
        input_dim=input_dim,
        hidden_sizes=config["model"]["hidden_sizes"],
        output_dim=config["model"]["output_dim"],
        dropout=config["model"]["dropout"]
    )

    model = CarPriceMLP(model_config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.MSELoss()

    # --- Training Loop ---
    logger.info("Starting training...")
    for epoch in range(config["training"]["epochs"]):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)['logits']
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t)['logits']
                val_mse = criterion(val_preds, y_val_t)
                val_rmse = torch.sqrt(val_mse).item()

            logger.info(
                "Epoch [%d/%d], Train Loss: %.4f, Train RMSE: %.4f, Val RMSE: %.4f,",
                epoch + 1,
                config["training"]["epochs"],
                loss.item(),
                loss.item() ** 0.5,
                val_rmse
            )

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)['logits']
        test_mse = criterion(test_preds, y_test_t)
        test_rmse = torch.sqrt(test_mse).item()
        mae = torch.mean(torch.abs(test_preds - y_test_t)).item()
    logger.info("Test RMSE: %.4f, Test MAE: %.4f", test_rmse, mae)

    # --- Save Artifacts ---
    model_path = Path(config["output"]["model_dir"])
    model_path.mkdir(parents=True, exist_ok=True)

    # Save model state dict and config
    model.save_pretrained(config["output"]["model_dir"])
    with open(model_path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    # Save preprocessor
    joblib.dump(preprocessor, model_path / "preprocessor.pkl")

    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    parser.add_argument("--verbose", action="store_true", help="Print more logs")
    args = parser.parse_args()
    train(args.config, verbose=args.verbose)
