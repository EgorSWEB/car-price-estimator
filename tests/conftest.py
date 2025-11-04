"""Fixtures for tests."""
import pytest
import pandas as pd


@pytest.fixture
def sample_car_data():
    """An example of a valid car dataset."""
    return pd.DataFrame({
        "name": ["Maruti 800 AC", "Hyundai Verna 1.6 SX"],
        "year": [2007, 2012],
        "selling_price": [60000, 600000],
        "km_driven": [70000, 100000],
        "fuel": ["Petrol", "Diesel"],
        "seller_type": ["Dealer", "Individual"],
        "transmission": ["Manual", "Automatic"],
        "owner": ["First", "Second"]
    })


@pytest.fixture
def minimal_car_data():
    """The minimum valid dataset."""
    return pd.DataFrame({
        "year": [2020],
        "selling_price": [60000]
    })