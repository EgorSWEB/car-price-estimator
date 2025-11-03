"""
Module defining the neural network model for car price prediction.
"""
import logging

from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class CarPriceMLPConfig(PretrainedConfig):
    """
    A simple config for Multi-Layer Perceptron (MLP) for regression.
    """
    model_type = "car_price_mlp"

    def __init__(
        self,
        input_dim=10,
        hidden_sizes=None,
        output_dim=1,
        dropout=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        logger = logging.getLogger(__name__)
        logger.info(
            "Creating model with input_dim=%d, hidden_sizes=%s",
            input_dim,
            hidden_sizes,
        )
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes or [128, 64]
        self.output_dim = output_dim
        self.dropout = dropout

class CarPriceMLP(PreTrainedModel):
    """
    A simple Multi-Layer Perceptron (MLP) for regression.
    """
    config_class = CarPriceMLPConfig

    def __init__(self, config):
        super().__init__(config)
        layers = []
        prev_size = config.input_dim
        for hidden in config.hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            prev_size = hidden
        layers.append(nn.Linear(prev_size, config.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x, labels=None):
        outputs = self.network(x).squeeze(-1)
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(outputs, labels)
        return {"loss": loss, "logits": outputs}  # стандартный формат HF
