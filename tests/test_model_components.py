"""Tests for the model and configuration."""
import torch
import tempfile
from pathlib import Path
from src.model import CarPriceMLP, CarPriceMLPConfig


def test_car_price_mlp_forward():
    """Test: the model makes a forward pass without errors."""
    config = CarPriceMLPConfig(input_dim=5, hidden_sizes=[10])
    model = CarPriceMLP(config)
    inputs = torch.randn(3, 5)
    output = model(inputs)
    assert "logits" in output
    assert output["logits"].shape == (3,)


def test_save_and_load_model():
    """Test: the model is saved and loaded correctly."""
    config = CarPriceMLPConfig(input_dim=4, hidden_sizes=[8])
    model_orig = CarPriceMLP(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model"
        model_orig.save_pretrained(model_path)

        # Загружаем
        model_loaded = CarPriceMLP.from_pretrained(model_path)

        # Сравниваем веса
        for p1, p2 in zip(model_orig.parameters(), model_loaded.parameters()):
            assert torch.equal(p1, p2)

        # Проверяем, что конфиг совпадает
        assert model_loaded.config.input_dim == 4
        assert model_loaded.config.hidden_sizes == [8]