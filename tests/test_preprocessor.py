"""Preprocessing tests."""
import pandas as pd
from src.preprocessor import add_car_age, create_preprocessor


def test_add_car_age():
    """Test: correctly adding the age of the car."""
    df = pd.DataFrame({"year": [2020, 2018]})
    result = add_car_age(df, current_year=2025)
    expected_age = [5, 7]
    pd.testing.assert_series_equal(result["car_age"], pd.Series(expected_age, name="car_age"))


def test_create_preprocessor():
    """Test: creating a preprocessor without errors."""
    numeric = ["km_driven"]
    categorical = ["fuel"]
    preprocessor = create_preprocessor(numeric, categorical)
    assert preprocessor is not None
    assert hasattr(preprocessor, "fit_transform")