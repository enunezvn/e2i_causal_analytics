"""Tests for split loader node including Feast integration."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.model_trainer.nodes.split_loader import (
    _fetch_splits_from_feast,
    load_splits,
)


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

    async def test_error_when_experiment_id_without_feast_or_db(self):
        """Should return error when experiment_id present but Feast and DB unavailable."""
        state = {
            "experiment_id": "exp_123",
        }

        # Mock both Feast and DB as unavailable
        with (
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_feature_analyzer_adapter",
                return_value=None,
            ),
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_ml_data_loader",
                return_value=None,
            ),
        ):
            result = await load_splits(state)

        assert "error" in result
        assert "unavailable" in result["error"].lower()
        assert result["error_type"] == "split_loader_unavailable"

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


@pytest.mark.asyncio
class TestFeastSplitLoading:
    """Tests for Feast feature store integration in split loading."""

    @pytest.fixture
    def mock_split_metadata(self):
        """Create mock split metadata for Feast retrieval."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return {
            "train": {
                "entity_ids": ["hcp_001", "hcp_002", "hcp_003"],
                "event_timestamps": [base_ts, base_ts, base_ts],
                "targets": [0, 1, 0],
                "target_column": "target",
            },
            "validation": {
                "entity_ids": ["hcp_004", "hcp_005"],
                "event_timestamps": [base_ts, base_ts],
                "targets": [1, 0],
                "target_column": "target",
            },
            "test": {
                "entity_ids": ["hcp_006", "hcp_007"],
                "event_timestamps": [base_ts, base_ts],
                "targets": [0, 1],
                "target_column": "target",
            },
            "holdout": {
                "entity_ids": ["hcp_008"],
                "event_timestamps": [base_ts],
                "targets": [1],
                "target_column": "target",
            },
        }

    @pytest.fixture
    def mock_feast_features(self):
        """Create mock feature DataFrame from Feast."""
        return pd.DataFrame(
            {
                "hcp_id": ["hcp_001", "hcp_002", "hcp_003"],
                "event_timestamp": pd.to_datetime(["2024-01-01"] * 3),
                "feature_view__feature1": [0.1, 0.2, 0.3],
                "feature_view__feature2": [1.0, 2.0, 3.0],
                "target": [0, 1, 0],
            }
        )

    async def test_fetch_splits_from_feast_success(self, mock_split_metadata, mock_feast_features):
        """Should successfully fetch splits from Feast."""
        mock_adapter = MagicMock()
        mock_adapter.get_training_features = AsyncMock(return_value=mock_feast_features)

        mock_loader = MagicMock()
        mock_loader.get_split_metadata = AsyncMock(return_value=mock_split_metadata)

        with (
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_feature_analyzer_adapter",
                return_value=mock_adapter,
            ),
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_ml_data_loader",
                return_value=mock_loader,
            ),
        ):
            result = await _fetch_splits_from_feast(
                experiment_id="exp_test_123",
                feature_refs=["feature_view:feature1", "feature_view:feature2"],
            )

        assert result is not None
        assert result["feast_source"] is True
        assert "train_data" in result
        assert "validation_data" in result
        assert "test_data" in result
        assert "holdout_data" in result
        assert result["train_data"]["feast_retrieved"] is True

    async def test_fetch_splits_returns_none_when_adapter_unavailable(self):
        """Should return None when FeatureAnalyzerAdapter is unavailable."""
        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_feature_analyzer_adapter",
            return_value=None,
        ):
            result = await _fetch_splits_from_feast(experiment_id="exp_test_123")

        assert result is None

    async def test_fetch_splits_returns_none_when_ml_loader_unavailable(self):
        """Should return None when MLDataLoader is unavailable."""
        mock_adapter = MagicMock()

        with (
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_feature_analyzer_adapter",
                return_value=mock_adapter,
            ),
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_ml_data_loader",
                return_value=None,
            ),
        ):
            result = await _fetch_splits_from_feast(experiment_id="exp_test_123")

        assert result is None

    async def test_fetch_splits_returns_none_when_no_metadata(self):
        """Should return None when no split metadata found."""
        mock_adapter = MagicMock()
        mock_loader = MagicMock()
        mock_loader.get_split_metadata = AsyncMock(return_value=None)

        with (
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_feature_analyzer_adapter",
                return_value=mock_adapter,
            ),
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_ml_data_loader",
                return_value=mock_loader,
            ),
        ):
            result = await _fetch_splits_from_feast(experiment_id="exp_test_123")

        assert result is None

    async def test_load_splits_uses_feast_when_experiment_id_provided(
        self, mock_split_metadata, mock_feast_features
    ):
        """Should use Feast when experiment_id is provided."""
        mock_adapter = MagicMock()
        mock_adapter.get_training_features = AsyncMock(return_value=mock_feast_features)

        mock_loader = MagicMock()
        mock_loader.get_split_metadata = AsyncMock(return_value=mock_split_metadata)

        state = {"experiment_id": "exp_feast_test"}

        with (
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_feature_analyzer_adapter",
                return_value=mock_adapter,
            ),
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_ml_data_loader",
                return_value=mock_loader,
            ),
        ):
            result = await load_splits(state)

        # Should have successfully loaded splits
        assert "error" not in result
        assert "train_data" in result
        assert result["train_samples"] >= 0

    async def test_load_splits_falls_back_to_database(self, mock_split_metadata):
        """Should fall back to database when Feast fails."""
        mock_loader = MagicMock()
        # First call for Feast fails (returns None)
        mock_loader.get_split_metadata = AsyncMock(return_value=None)
        # Second call for DB fallback succeeds
        mock_loader.load_experiment_splits = AsyncMock(
            return_value={
                "train": {
                    "X": np.random.rand(100, 5),
                    "y": np.random.rand(100),
                    "row_count": 100,
                },
                "validation": {
                    "X": np.random.rand(30, 5),
                    "y": np.random.rand(30),
                    "row_count": 30,
                },
                "test": {
                    "X": np.random.rand(20, 5),
                    "y": np.random.rand(20),
                    "row_count": 20,
                },
                "holdout": {
                    "X": np.random.rand(10, 5),
                    "y": np.random.rand(10),
                    "row_count": 10,
                },
            }
        )

        state = {"experiment_id": "exp_db_fallback_test"}

        with (
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_feature_analyzer_adapter",
                return_value=None,  # Feast unavailable
            ),
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_ml_data_loader",
                return_value=mock_loader,
            ),
        ):
            result = await load_splits(state)

        # Should have successfully loaded splits from DB
        assert "error" not in result
        assert result["train_samples"] == 100

    async def test_fetch_splits_handles_exception(self, mock_split_metadata):
        """Should handle exceptions gracefully and return None."""
        mock_adapter = MagicMock()
        mock_adapter.get_training_features = AsyncMock(
            side_effect=Exception("Feast connection failed")
        )

        mock_loader = MagicMock()
        mock_loader.get_split_metadata = AsyncMock(return_value=mock_split_metadata)

        with (
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_feature_analyzer_adapter",
                return_value=mock_adapter,
            ),
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_ml_data_loader",
                return_value=mock_loader,
            ),
        ):
            result = await _fetch_splits_from_feast(experiment_id="exp_test_123")

        # Should return None on exception
        assert result is None

    async def test_fetch_splits_uses_custom_entity_key(
        self, mock_split_metadata, mock_feast_features
    ):
        """Should use custom entity key when provided."""
        mock_adapter = MagicMock()
        mock_adapter.get_training_features = AsyncMock(return_value=mock_feast_features)

        mock_loader = MagicMock()
        mock_loader.get_split_metadata = AsyncMock(return_value=mock_split_metadata)

        with (
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_feature_analyzer_adapter",
                return_value=mock_adapter,
            ),
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_ml_data_loader",
                return_value=mock_loader,
            ),
        ):
            result = await _fetch_splits_from_feast(
                experiment_id="exp_test_123",
                entity_key="custom_entity_id",
            )

        assert result is not None
        # Verify adapter was called (we can check the call count)
        assert mock_adapter.get_training_features.call_count > 0

    async def test_load_splits_passes_feature_refs_from_state(
        self, mock_split_metadata, mock_feast_features
    ):
        """Should pass feature_refs from state to Feast."""
        mock_adapter = MagicMock()
        mock_adapter.get_training_features = AsyncMock(return_value=mock_feast_features)

        mock_loader = MagicMock()
        mock_loader.get_split_metadata = AsyncMock(return_value=mock_split_metadata)

        state = {
            "experiment_id": "exp_feature_refs_test",
            "feature_refs": ["view1:feature1", "view2:feature2"],
            "entity_key": "custom_id",
        }

        with (
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_feature_analyzer_adapter",
                return_value=mock_adapter,
            ),
            patch(
                "src.agents.ml_foundation.model_trainer.nodes.split_loader._get_ml_data_loader",
                return_value=mock_loader,
            ),
        ):
            result = await load_splits(state)

        # Should have loaded successfully
        assert "error" not in result
        # Verify adapter was called with the right feature_refs
        call_kwargs = mock_adapter.get_training_features.call_args_list[0].kwargs
        assert call_kwargs["feature_refs"] == ["view1:feature1", "view2:feature2"]
