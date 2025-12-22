"""Tests for model checkpointer node.

Tests the save_checkpoint, load_checkpoint, and list_checkpoints functions.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.agents.ml_foundation.model_trainer.nodes.checkpointer import (
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
    delete_checkpoint,
    _compute_model_hash,
    _filter_serializable,
    _get_framework,
)


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def trained_model():
    """Create a trained sklearn model."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def checkpoint_state(trained_model, tmp_path):
    """Create state for checkpointing."""
    return {
        "trained_model": trained_model,
        "experiment_id": "exp_001",
        "algorithm_name": "RandomForest",
        "problem_type": "binary_classification",
        "framework": "sklearn",
        "best_hyperparameters": {"n_estimators": 5, "max_depth": None},
        "training_duration_seconds": 10.5,
        "hpo_completed": True,
        "hpo_best_value": 0.85,
        "hpo_trials_run": 20,
        "evaluation_metrics": {
            "train_metrics": {"accuracy": 0.95},
            "validation_metrics": {"accuracy": 0.88},
            "test_metrics": {"accuracy": 0.85, "roc_auc": 0.90},
        },
        "success_criteria_met": True,
        "enable_checkpointing": True,
        "checkpoint_dir": str(tmp_path),
    }


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    return tmp_path


# ============================================================================
# Test save_checkpoint function
# ============================================================================


@pytest.mark.asyncio
class TestSaveCheckpoint:
    """Test checkpoint saving."""

    async def test_saves_checkpoint_successfully(self, checkpoint_state):
        """Should save checkpoint successfully."""
        result = await save_checkpoint(checkpoint_state)

        assert result["checkpoint_status"] == "success"
        assert result["checkpoint_path"] is not None
        assert result["checkpoint_metadata_path"] is not None
        assert result["model_hash"] is not None

    async def test_creates_checkpoint_file(self, checkpoint_state):
        """Should create checkpoint file on disk."""
        result = await save_checkpoint(checkpoint_state)

        checkpoint_path = Path(result["checkpoint_path"])
        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == ".pkl"

    async def test_creates_metadata_file(self, checkpoint_state):
        """Should create metadata JSON file."""
        result = await save_checkpoint(checkpoint_state)

        metadata_path = Path(result["checkpoint_metadata_path"])
        assert metadata_path.exists()
        assert metadata_path.suffix == ".json"

        # Verify metadata content
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["algorithm_name"] == "RandomForest"
        assert metadata["experiment_id"] == "exp_001"
        assert metadata["model_hash"] == result["model_hash"]

    async def test_skips_when_disabled(self, checkpoint_state):
        """Should skip checkpointing when disabled."""
        checkpoint_state["enable_checkpointing"] = False

        result = await save_checkpoint(checkpoint_state)

        assert result["checkpoint_status"] == "disabled"
        assert result["checkpoint_path"] is None

    async def test_skips_when_no_model(self, checkpoint_state):
        """Should skip when no trained model."""
        checkpoint_state["trained_model"] = None

        result = await save_checkpoint(checkpoint_state)

        assert result["checkpoint_status"] == "skipped"
        assert "error" in result

    async def test_records_timestamp(self, checkpoint_state):
        """Should record checkpoint timestamp."""
        result = await save_checkpoint(checkpoint_state)

        assert "checkpoint_timestamp" in result
        assert result["checkpoint_timestamp"] is not None

    async def test_generates_checkpoint_name(self, checkpoint_state):
        """Should generate unique checkpoint name."""
        result = await save_checkpoint(checkpoint_state)

        assert "checkpoint_name" in result
        assert "randomforest" in result["checkpoint_name"].lower()
        assert "exp_001" in result["checkpoint_name"]

    async def test_saves_evaluation_metrics(self, checkpoint_state):
        """Should save evaluation metrics in metadata."""
        result = await save_checkpoint(checkpoint_state)

        metadata_path = Path(result["checkpoint_metadata_path"])
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "test_metrics" in metadata
        assert metadata["test_metrics"]["accuracy"] == 0.85


# ============================================================================
# Test load_checkpoint function
# ============================================================================


@pytest.mark.asyncio
class TestLoadCheckpoint:
    """Test checkpoint loading."""

    async def test_loads_checkpoint_by_path(self, checkpoint_state):
        """Should load checkpoint by path."""
        # First save
        save_result = await save_checkpoint(checkpoint_state)
        checkpoint_path = save_result["checkpoint_path"]

        # Then load
        load_state = {"checkpoint_path": checkpoint_path}
        load_result = await load_checkpoint(load_state)

        assert load_result["load_status"] == "success"
        assert load_result["loaded_model"] is not None

    async def test_loads_checkpoint_by_name(self, checkpoint_state):
        """Should load checkpoint by name."""
        # First save
        save_result = await save_checkpoint(checkpoint_state)
        checkpoint_name = save_result["checkpoint_name"]

        # Then load
        load_state = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_dir": checkpoint_state["checkpoint_dir"],
        }
        load_result = await load_checkpoint(load_state)

        assert load_result["load_status"] == "success"
        assert load_result["loaded_model"] is not None

    async def test_loads_metadata(self, checkpoint_state):
        """Should load metadata along with model."""
        save_result = await save_checkpoint(checkpoint_state)

        load_state = {"checkpoint_path": save_result["checkpoint_path"]}
        load_result = await load_checkpoint(load_state)

        assert "checkpoint_metadata" in load_result
        assert load_result["checkpoint_metadata"]["algorithm_name"] == "RandomForest"

    async def test_error_when_no_path_or_name(self):
        """Should return error when no path or name provided."""
        load_result = await load_checkpoint({})

        assert load_result["load_status"] == "failed"
        assert "error" in load_result

    async def test_error_when_file_not_found(self, tmp_checkpoint_dir):
        """Should return error when checkpoint not found."""
        load_state = {
            "checkpoint_path": str(tmp_checkpoint_dir / "nonexistent.pkl")
        }
        load_result = await load_checkpoint(load_state)

        assert load_result["load_status"] == "failed"
        assert "not found" in load_result["error"].lower()

    async def test_loaded_model_can_predict(self, checkpoint_state):
        """Should load a model that can make predictions."""
        save_result = await save_checkpoint(checkpoint_state)

        load_state = {"checkpoint_path": save_result["checkpoint_path"]}
        load_result = await load_checkpoint(load_state)

        model = load_result["loaded_model"]
        X_test = np.random.rand(10, 5)
        predictions = model.predict(X_test)

        assert len(predictions) == 10


# ============================================================================
# Test list_checkpoints function
# ============================================================================


class TestListCheckpoints:
    """Test checkpoint listing."""

    @pytest.mark.asyncio
    async def test_lists_checkpoints(self, checkpoint_state):
        """Should list available checkpoints."""
        # Save multiple checkpoints
        await save_checkpoint(checkpoint_state)
        checkpoint_state["experiment_id"] = "exp_002"
        await save_checkpoint(checkpoint_state)

        checkpoints = list_checkpoints(
            checkpoint_dir=checkpoint_state["checkpoint_dir"]
        )

        assert len(checkpoints) >= 2

    @pytest.mark.asyncio
    async def test_filters_by_experiment(self, checkpoint_state):
        """Should filter by experiment ID."""
        await save_checkpoint(checkpoint_state)
        checkpoint_state["experiment_id"] = "exp_002"
        await save_checkpoint(checkpoint_state)

        checkpoints = list_checkpoints(
            checkpoint_dir=checkpoint_state["checkpoint_dir"],
            experiment_id="exp_001",
        )

        assert len(checkpoints) == 1
        assert checkpoints[0]["experiment_id"] == "exp_001"

    @pytest.mark.asyncio
    async def test_filters_by_algorithm(self, checkpoint_state):
        """Should filter by algorithm name."""
        await save_checkpoint(checkpoint_state)

        checkpoints = list_checkpoints(
            checkpoint_dir=checkpoint_state["checkpoint_dir"],
            algorithm_name="RandomForest",
        )

        assert len(checkpoints) >= 1
        assert all(c["algorithm_name"] == "RandomForest" for c in checkpoints)

    def test_returns_empty_for_nonexistent_dir(self, tmp_path):
        """Should return empty list for nonexistent directory."""
        checkpoints = list_checkpoints(
            checkpoint_dir=str(tmp_path / "nonexistent")
        )

        assert checkpoints == []

    @pytest.mark.asyncio
    async def test_sorts_by_creation_time(self, checkpoint_state):
        """Should sort checkpoints by creation time (newest first)."""
        await save_checkpoint(checkpoint_state)
        checkpoint_state["experiment_id"] = "exp_002"
        await save_checkpoint(checkpoint_state)

        checkpoints = list_checkpoints(
            checkpoint_dir=checkpoint_state["checkpoint_dir"]
        )

        # Most recent should be first
        times = [c.get("created_at", "") for c in checkpoints]
        assert times == sorted(times, reverse=True)


# ============================================================================
# Test delete_checkpoint function
# ============================================================================


class TestDeleteCheckpoint:
    """Test checkpoint deletion."""

    @pytest.mark.asyncio
    async def test_deletes_by_name(self, checkpoint_state):
        """Should delete checkpoint by name."""
        save_result = await save_checkpoint(checkpoint_state)
        checkpoint_name = save_result["checkpoint_name"]

        deleted = delete_checkpoint(
            checkpoint_name=checkpoint_name,
            checkpoint_dir=checkpoint_state["checkpoint_dir"],
        )

        assert deleted is True
        assert not Path(save_result["checkpoint_path"]).exists()
        assert not Path(save_result["checkpoint_metadata_path"]).exists()

    @pytest.mark.asyncio
    async def test_deletes_by_path(self, checkpoint_state):
        """Should delete checkpoint by path."""
        save_result = await save_checkpoint(checkpoint_state)
        checkpoint_path = save_result["checkpoint_path"]

        deleted = delete_checkpoint(checkpoint_path=checkpoint_path)

        assert deleted is True
        assert not Path(checkpoint_path).exists()

    def test_returns_false_when_nothing_to_delete(self, tmp_path):
        """Should return False when nothing to delete."""
        deleted = delete_checkpoint(checkpoint_dir=str(tmp_path))

        assert deleted is False


# ============================================================================
# Test helper functions
# ============================================================================


class TestComputeModelHash:
    """Test model hash computation."""

    def test_computes_consistent_hash(self, trained_model):
        """Should compute consistent hash for same model."""
        hash1 = _compute_model_hash(trained_model)
        hash2 = _compute_model_hash(trained_model)

        assert hash1 == hash2

    def test_returns_16_char_hash(self, trained_model):
        """Should return 16 character hash."""
        model_hash = _compute_model_hash(trained_model)

        assert len(model_hash) == 16

    def test_handles_unpicklable_model(self):
        """Should return 'unknown' for unpicklable objects."""

        class UnpicklableModel:
            def __reduce__(self):
                raise TypeError("Cannot pickle")

        model_hash = _compute_model_hash(UnpicklableModel())

        assert model_hash == "unknown"


class TestFilterSerializable:
    """Test serializable filtering."""

    def test_keeps_basic_types(self):
        """Should keep basic JSON-serializable types."""
        data = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        }

        result = _filter_serializable(data)

        assert result == data

    def test_filters_numpy_arrays(self):
        """Should filter out numpy arrays."""
        data = {
            "array": np.array([1, 2, 3]),
            "string": "value",
        }

        result = _filter_serializable(data)

        assert "array" not in result
        assert result["string"] == "value"

    def test_handles_nested_dicts(self):
        """Should handle nested dictionaries."""
        data = {
            "outer": {
                "inner": "value",
                "array": np.array([1, 2, 3]),
            }
        }

        result = _filter_serializable(data)

        assert result["outer"]["inner"] == "value"
        assert "array" not in result["outer"]

    def test_keeps_serializable_lists(self):
        """Should keep JSON-serializable lists."""
        data = {"list": [1, 2, 3]}

        result = _filter_serializable(data)

        assert result["list"] == [1, 2, 3]


class TestGetFramework:
    """Test framework identification for checkpointer."""

    def test_identifies_sklearn_algorithms(self):
        """Should identify sklearn algorithms."""
        assert _get_framework("RandomForest") == "sklearn"
        assert _get_framework("LogisticRegression") == "sklearn"
        assert _get_framework("GradientBoosting") == "sklearn"

    def test_identifies_xgboost(self):
        """Should identify XGBoost."""
        assert _get_framework("XGBoost") == "xgboost"

    def test_identifies_lightgbm(self):
        """Should identify LightGBM."""
        assert _get_framework("LightGBM") == "lightgbm"

    def test_returns_sklearn_for_unknown(self):
        """Should default to sklearn."""
        assert _get_framework("UnknownAlgorithm") == "sklearn"
