"""Tests for split loader node."""

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.split_loader import load_splits


@pytest.mark.asyncio
class TestLoadSplits:
    """Test data split loading and validation."""

    @pytest.fixture
    def valid_splits(self):
        """Create valid split data."""
        return {
            "train_data": {
                "X": np.random.rand(600, 10),
                "y": np.random.randint(0, 2, 600),
                "row_count": 600,
            },
            "validation_data": {
                "X": np.random.rand(200, 10),
                "y": np.random.randint(0, 2, 200),
                "row_count": 200,
            },
            "test_data": {
                "X": np.random.rand(150, 10),
                "y": np.random.randint(0, 2, 150),
                "row_count": 150,
            },
            "holdout_data": {
                "X": np.random.rand(50, 10),
                "y": np.random.randint(0, 2, 50),
                "row_count": 50,
            },
        }

    async def test_loads_splits_from_state(self, valid_splits):
        """Should load splits when already in state."""
        state = {**valid_splits}

        result = await load_splits(state)

        assert "error" not in result
        assert result["train_data"] == valid_splits["train_data"]
        assert result["validation_data"] == valid_splits["validation_data"]
        assert result["test_data"] == valid_splits["test_data"]
        assert result["holdout_data"] == valid_splits["holdout_data"]

    async def test_calculates_sample_counts(self, valid_splits):
        """Should calculate sample counts for all splits."""
        state = {**valid_splits}

        result = await load_splits(state)

        assert result["train_samples"] == 600
        assert result["validation_samples"] == 200
        assert result["test_samples"] == 150
        assert result["holdout_samples"] == 50
        assert result["total_samples"] == 1000

    async def test_calculates_split_ratios(self, valid_splits):
        """Should calculate actual split ratios."""
        state = {**valid_splits}

        result = await load_splits(state)

        # Expected: 60/20/15/5
        assert result["train_ratio"] == 0.60
        assert result["validation_ratio"] == 0.20
        assert result["test_ratio"] == 0.15
        assert result["holdout_ratio"] == 0.05

    async def test_error_when_experiment_id_without_feast(self):
        """Should return error when experiment_id present but Feast not implemented."""
        state = {
            "experiment_id": "exp_123",
        }

        result = await load_splits(state)

        assert "error" in result
        assert "not yet implemented" in result["error"].lower()
        assert result["error_type"] == "split_fetch_not_implemented"

    async def test_error_when_no_splits_and_no_experiment_id(self):
        """Should return error when neither splits nor experiment_id provided."""
        state = {}

        result = await load_splits(state)

        assert "error" in result
        assert "cannot load splits" in result["error"].lower()
        assert result["error_type"] == "missing_splits_error"

    async def test_validates_split_structure_is_dict(self, valid_splits):
        """Should validate each split is a dictionary."""
        state = {**valid_splits, "train_data": "not a dict"}

        result = await load_splits(state)

        assert "error" in result
        assert "not a dictionary" in result["error"].lower()
        assert result["error_type"] == "invalid_split_format"

    async def test_validates_split_has_required_keys(self, valid_splits):
        """Should validate each split has X, y, row_count."""
        state = {
            **valid_splits,
            "train_data": {"X": np.random.rand(100, 10)},  # Missing y and row_count
        }

        result = await load_splits(state)

        assert "error" in result
        assert "missing required key" in result["error"].lower()
        assert result["error_type"] == "invalid_split_format"

    async def test_error_when_all_splits_empty(self):
        """Should return error when all splits have 0 samples."""
        state = {
            "train_data": {"X": np.array([]), "y": np.array([]), "row_count": 0},
            "validation_data": {"X": np.array([]), "y": np.array([]), "row_count": 0},
            "test_data": {"X": np.array([]), "y": np.array([]), "row_count": 0},
            "holdout_data": {"X": np.array([]), "y": np.array([]), "row_count": 0},
        }

        result = await load_splits(state)

        assert "error" in result
        assert "empty" in result["error"].lower()
        assert result["error_type"] == "empty_splits_error"

    async def test_handles_partial_splits(self, valid_splits):
        """Should validate all 4 splits are present."""
        # Missing holdout_data
        state = {
            "train_data": valid_splits["train_data"],
            "validation_data": valid_splits["validation_data"],
            "test_data": valid_splits["test_data"],
        }

        result = await load_splits(state)

        # Should either error or try to fetch from experiment_id
        assert "error" in result or "experiment_id" in result

    async def test_different_split_ratios(self):
        """Should calculate ratios correctly for different distributions."""
        state = {
            "train_data": {"X": np.random.rand(700, 5), "y": np.random.rand(700), "row_count": 700},
            "validation_data": {
                "X": np.random.rand(150, 5),
                "y": np.random.rand(150),
                "row_count": 150,
            },
            "test_data": {"X": np.random.rand(100, 5), "y": np.random.rand(100), "row_count": 100},
            "holdout_data": {"X": np.random.rand(50, 5), "y": np.random.rand(50), "row_count": 50},
        }

        result = await load_splits(state)

        assert abs(result["train_ratio"] - 0.70) < 0.01
        assert abs(result["validation_ratio"] - 0.15) < 0.01
        assert abs(result["test_ratio"] - 0.10) < 0.01
        assert abs(result["holdout_ratio"] - 0.05) < 0.01

    async def test_handles_missing_validation_key(self, valid_splits):
        """Should validate validation_data key is required."""
        state = {
            "train_data": valid_splits["train_data"],
            "test_data": valid_splits["test_data"],
            "holdout_data": valid_splits["holdout_data"],
        }

        result = await load_splits(state)

        # Should error because not all 4 splits present
        assert "error" in result or "experiment_id" in result

    async def test_handles_large_dataset(self):
        """Should handle large datasets efficiently."""
        state = {
            "train_data": {
                "X": np.random.rand(60000, 50),
                "y": np.random.rand(60000),
                "row_count": 60000,
            },
            "validation_data": {
                "X": np.random.rand(20000, 50),
                "y": np.random.rand(20000),
                "row_count": 20000,
            },
            "test_data": {
                "X": np.random.rand(15000, 50),
                "y": np.random.rand(15000),
                "row_count": 15000,
            },
            "holdout_data": {
                "X": np.random.rand(5000, 50),
                "y": np.random.rand(5000),
                "row_count": 5000,
            },
        }

        result = await load_splits(state)

        assert result["total_samples"] == 100000
        assert result["train_samples"] == 60000

    async def test_preserves_original_split_data(self, valid_splits):
        """Should preserve original split data without modification."""
        state = {**valid_splits}
        original_X_train_shape = valid_splits["train_data"]["X"].shape

        result = await load_splits(state)

        # Shape should be preserved
        assert result["train_data"]["X"].shape == original_X_train_shape
