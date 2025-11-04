"""Tests for data profiling."""
import json
import tempfile
from pathlib import Path
from src.data_profiler import profile_data


def test_profile_data(sample_car_data):
    """Test: generating a valid JSON report."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "profile.json"
        profile_data(sample_car_data, str(output_path))

        assert output_path.exists()

        with open(output_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert "general" in report
        assert "numeric" in report
        assert "categorical" in report
        assert report["general"]["n_rows"] == len(sample_car_data)
