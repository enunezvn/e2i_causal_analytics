"""Tests for preprocessor node."""

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.model_trainer.nodes.preprocessor import fit_preprocessing


@pytest.mark.asyncio
class TestFitPreprocessing:
    """Test preprocessing isolation (fit on train only)."""

    @pytest.fixture
    def valid_data(self):
        """Create valid split data."""
        return {
            "train_data": {
                "X": pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)]),
                "y": np.random.randint(0, 2, 100),
            },
            "validation_data": {
                "X": pd.DataFrame(np.random.rand(30, 5), columns=[f"f{i}" for i in range(5)]),
                "y": np.random.randint(0, 2, 30),
            },
            "test_data": {
                "X": pd.DataFrame(np.random.rand(20, 5), columns=[f"f{i}" for i in range(5)]),
                "y": np.random.randint(0, 2, 20),
            },
        }

    async def test_fits_preprocessor_on_train_only(self, valid_data):
        """Should fit preprocessor on training data only."""
        state = {**valid_data}

        result = await fit_preprocessing(state)

        assert "error" not in result
        assert "preprocessor" in result
        # Preprocessor should have been fit
        assert result["preprocessor"].train_statistics_ is not None

    async def test_transforms_all_splits(self, valid_data):
        """Should transform train, validation, and test splits."""
        state = {**valid_data}

        result = await fit_preprocessing(state)

        assert "X_train_preprocessed" in result
        assert "X_validation_preprocessed" in result
        assert "X_test_preprocessed" in result

    async def test_computes_statistics_from_train_only(self, valid_data):
        """Should compute preprocessing statistics from train set only."""
        state = {**valid_data}

        result = await fit_preprocessing(state)

        stats = result["preprocessing_statistics"]
        assert "train_statistics" in stats
        assert stats["train_statistics"] is not None

    async def test_preserves_feature_names(self, valid_data):
        """Should preserve feature names from training data."""
        state = {**valid_data}

        result = await fit_preprocessing(state)

        assert result["preprocessing_statistics"]["feature_names_out"] is not None
        # With sklearn ColumnTransformer, should match numeric feature names
        expected_names = [f"f{i}" for i in range(5)]
        assert result["preprocessing_statistics"]["feature_names_out"] == expected_names

    async def test_error_when_X_train_missing(self):
        """Should return error when X_train is None."""
        state = {
            "train_data": {"y": np.random.rand(100)},
            "validation_data": {"X": np.random.rand(30, 5), "y": np.random.rand(30)},
            "test_data": {"X": np.random.rand(20, 5), "y": np.random.rand(20)},
        }

        result = await fit_preprocessing(state)

        assert "error" in result
        assert result["error_type"] == "missing_training_data"

    async def test_error_when_X_validation_missing(self, valid_data):
        """Should return error when X_validation is None."""
        state = {
            "train_data": valid_data["train_data"],
            "validation_data": {"y": np.random.rand(30)},
            "test_data": valid_data["test_data"],
        }

        result = await fit_preprocessing(state)

        assert "error" in result
        assert result["error_type"] == "missing_validation_data"

    async def test_error_when_X_test_missing(self, valid_data):
        """Should return error when X_test is None."""
        state = {
            "train_data": valid_data["train_data"],
            "validation_data": valid_data["validation_data"],
            "test_data": {"y": np.random.rand(20)},
        }

        result = await fit_preprocessing(state)

        assert "error" in result
        assert result["error_type"] == "missing_test_data"

    async def test_handles_numpy_arrays(self):
        """Should handle numpy arrays (not just pandas)."""
        state = {
            "train_data": {"X": np.random.rand(100, 5), "y": np.random.rand(100)},
            "validation_data": {"X": np.random.rand(30, 5), "y": np.random.rand(30)},
            "test_data": {"X": np.random.rand(20, 5), "y": np.random.rand(20)},
        }

        result = await fit_preprocessing(state)

        assert "error" not in result
        assert "preprocessor" in result

    async def test_preprocessor_transforms_data(self, valid_data):
        """Preprocessor should transform data (scaling for numeric features)."""
        state = {**valid_data}
        original_X_train = state["train_data"]["X"].copy()

        result = await fit_preprocessing(state)

        # Preprocessed data should have same shape
        assert result["X_train_preprocessed"].shape == original_X_train.shape
        # With scaling, numeric data should be standardized (mean ~0, std ~1)
        preprocessed = result["X_train_preprocessed"]
        assert abs(preprocessed.mean()) < 0.1  # Mean close to 0
        assert 0.8 < preprocessed.std() < 1.2  # Std close to 1

    async def test_records_feature_count(self, valid_data):
        """Should record number of features."""
        state = {**valid_data}

        result = await fit_preprocessing(state)

        assert result["preprocessing_statistics"]["n_features_in"] == 5
        assert result["preprocessing_statistics"]["n_features_out"] == 5

    async def test_computes_mean_and_std(self, valid_data):
        """Should compute mean and std from training data."""
        state = {**valid_data}

        result = await fit_preprocessing(state)

        stats = result["preprocessing_statistics"]["train_statistics"]
        # Stats are nested under 'numeric' for numeric-only data
        assert "numeric" in stats
        assert "mean" in stats["numeric"]
        assert "std" in stats["numeric"]

    async def test_includes_preprocessing_type(self, valid_data):
        """Should include preprocessing type in statistics."""
        state = {**valid_data}

        result = await fit_preprocessing(state)

        assert "preprocessing_type" in result["preprocessing_statistics"]
