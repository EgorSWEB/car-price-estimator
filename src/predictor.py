"""
Module for obtaining the final car price prediction based on the initial data.
"""
import torch
import numpy as np

def postprocess_predictions(raw_outputs: torch.Tensor) -> np.ndarray:
    """Converts the raw outputs of the model into final predictions.
    For regression, we simply discard negative values (price ≥ 0).

    Args:
        raw_outputs (torch.Tensor): output model tensor

    Returns:
        preds (np.ndarray): processed response
    """
    preds = raw_outputs.detach().cpu().numpy()
    preds = np.maximum(preds, 0.0)  # цена не может быть отрицательной
    return preds
