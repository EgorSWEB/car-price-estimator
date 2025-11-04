"""Tests for post-processing of predictions."""
import torch
import numpy as np
from src.predictor import postprocess_predictions


def test_postprocess_predictions():
    """Test: Negative values are replaced by 0."""
    raw_output = torch.tensor([-2.0, 0.0, 3.5])
    result = postprocess_predictions(raw_output)
    expected = np.array([0.0, 0.0, 3.5])
    np.testing.assert_array_equal(result, expected)


def test_postprocess_predictions_empty():
    """Test: processing an empty tensor."""
    raw_output = torch.tensor([])
    result = postprocess_predictions(raw_output)
    assert result.shape == (0,)