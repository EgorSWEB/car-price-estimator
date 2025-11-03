# src/model.py
import torch
import torch.nn as nn
import logging

class CarPriceMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim=1, dropout=0.0):
        super().__init__()
        logger = logging.getLogger(__name__)
        logger.info(f"Creating model with input_dim={input_dim}, hidden_sizes={hidden_sizes}")

        layers = []
        prev_size = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden
        layers.append(nn.Linear(prev_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)