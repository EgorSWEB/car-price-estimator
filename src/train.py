# src/train.py
import os
import sys
import yaml
import argparse
import logging
import torch
import numpy as np
from pathlib import Path

from data_preparation import load_and_preprocess_data
from model import CarPriceMLP

def setup_logging(log_dir, level="INFO"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def train(config_path, verbose=False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    setup_logging(config["logging"]["log_dir"], config["logging"]["level"])
    logger = logging.getLogger(__name__)
    logger.info("Starting training pipeline")

    # Set seed
    torch.manual_seed(config["training"]["random_seed"])
    np.random.seed(config["training"]["random_seed"])

    # Load data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(config)
    input_dim = X_train.shape[1]

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Create model
    model = CarPriceMLP(
        input_dim=input_dim,
        hidden_sizes=config["model"]["hidden_sizes"],
        output_dim=config["model"]["output_dim"],
        dropout=config["model"]["dropout"]
    )

    device = torch.device(config["training"]["device"])
    model.to(device)
    X_train_t = X_train_t.to(device)
    y_train_t = y_train_t.to(device)
    X_test_t = X_test_t.to(device)
    y_test_t = y_test_t.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.MSELoss()

    # Training loop
    logger.info("Starting training...")
    for epoch in range(config["training"]["epochs"]):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_mse = criterion(test_preds, y_test_t)
        test_rmse = torch.sqrt(test_mse).item()
        mae = torch.mean(torch.abs(test_preds - y_test_t)).item()
    logger.info(f"Test RMSE: {test_rmse:.4f}, MAE: {mae:.4f}")

    # Save model in Hugging Face compatible format
    model_path = Path(config["output"]["model_dir"])
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict and config
    torch.save(model.state_dict(), model_path / "pytorch_model.bin")
    with open(model_path / "config.yaml", "w") as f:
        yaml.dump(config, f)
    # Save preprocessor if needed (not standard HF, but useful)
    import joblib
    joblib.dump(preprocessor, model_path / "preprocessor.pkl")

    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Print more logs")
    args = parser.parse_args()
    train(args.config, verbose=args.verbose)