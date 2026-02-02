"""Tests for split enforcer node."""

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.model_trainer.nodes.split_enforcer import (
    _check_duplicate_indices,
    _check_feature_leakage,
    _check_temporal_ordering,
    _get_indices,
    enforce_splits,
)


@pytest.mark.asyncio
class TestEnforceSplits:
    """Test split ratio validation and leakage detection."""

    async def test_validates_perfect_split_ratios(self):
        """Should pass with perfect 60/20/15/5 ratios."""
        state = {
            "train_ratio": 0.60,
            "validation_ratio": 0.20,
            "test_ratio": 0.15,
            "holdout_ratio": 0.05,
            "train_samples": 600,
            "validation_samples": 200,
            "test_samples": 150,
            "holdout_samples": 50,
            "total_samples": 1000,
        }

        result = await enforce_splits(state)

        assert result["split_ratios_valid"] is True
        assert "validated" in result["split_validation_message"].lower()
        assert len(result["leakage_warnings"]) == 0

    async def test_allows_ratios_within_tolerance(self):
        """Should pass with ratios within ±2% tolerance."""
        state = {
            "train_ratio": 0.61,  # 60% + 1%
            "validation_ratio": 0.19,  # 20% - 1%
            "test_ratio": 0.15,  # 15% exact
            "holdout_ratio": 0.05,  # 5% exact
            "train_samples": 610,
            "validation_samples": 190,
            "test_samples": 150,
            "holdout_samples": 50,
            "total_samples": 1000,
        }

        result = await enforce_splits(state)

        assert result["split_ratios_valid"] is True

    async def test_fails_when_train_ratio_too_low(self):
        """Should fail when train ratio below 58% (60% - 2%)."""
        state = {
            "train_ratio": 0.57,  # Below threshold
            "validation_ratio": 0.20,
            "test_ratio": 0.18,
            "holdout_ratio": 0.05,
            "train_samples": 570,
            "validation_samples": 200,
            "test_samples": 180,
            "holdout_samples": 50,
            "total_samples": 1000,
        }

        result = await enforce_splits(state)

        assert result["split_ratios_valid"] is False
        assert any("train" in check.lower() for check in result["split_ratio_checks"])

    async def test_fails_when_validation_ratio_too_high(self):
        """Should fail when validation ratio above 22% (20% + 2%)."""
        state = {
            "train_ratio": 0.58,
            "validation_ratio": 0.23,  # Above threshold
            "test_ratio": 0.14,
            "holdout_ratio": 0.05,
            "train_samples": 580,
            "validation_samples": 230,
            "test_samples": 140,
            "holdout_samples": 50,
            "total_samples": 1000,
        }

        result = await enforce_splits(state)

        assert result["split_ratios_valid"] is False

    async def test_checks_minimum_sample_sizes(self):
        """Should fail if any split has < 10 samples."""
        state = {
            "train_ratio": 0.88,
            "validation_ratio": 0.08,
            "test_ratio": 0.03,
            "holdout_ratio": 0.01,
            "train_samples": 88,
            "validation_samples": 8,  # Below minimum (10)
            "test_samples": 3,  # Below minimum (10)
            "holdout_samples": 1,  # Below minimum (10)
            "total_samples": 100,
        }

        result = await enforce_splits(state)

        assert result["split_ratios_valid"] is False
        assert len(result["leakage_warnings"]) >= 1
        assert any("validation" in warning.lower() for warning in result["leakage_warnings"])

    async def test_detects_sum_not_equal_to_one(self):
        """Should detect if split ratios don't sum to 1.0."""
        state = {
            "train_ratio": 0.50,
            "validation_ratio": 0.20,
            "test_ratio": 0.15,
            "holdout_ratio": 0.10,  # Sum = 0.95, not 1.0
            "train_samples": 500,
            "validation_samples": 200,
            "test_samples": 150,
            "holdout_samples": 100,
            "total_samples": 950,
        }

        result = await enforce_splits(state)

        assert result["split_ratios_valid"] is False
        assert any("sum" in warning.lower() for warning in result["leakage_warnings"])

    async def test_handles_missing_ratio_fields(self):
        """Should handle missing ratio fields gracefully."""
        state = {
            "train_samples": 600,
            "validation_samples": 200,
            "test_samples": 150,
            "holdout_samples": 50,
            "total_samples": 1000,
        }

        result = await enforce_splits(state)

        # Should default to 0.0 for missing ratios
        assert result["split_ratios_valid"] is False

    async def test_boundary_case_exact_2_percent_tolerance(self):
        """Should pass at exact boundary of ±2% tolerance."""
        state = {
            "train_ratio": 0.62,  # 60% + 2% (exact boundary)
            "validation_ratio": 0.18,  # 20% - 2% (exact boundary)
            "test_ratio": 0.15,
            "holdout_ratio": 0.05,
            "train_samples": 620,
            "validation_samples": 180,
            "test_samples": 150,
            "holdout_samples": 50,
            "total_samples": 1000,
        }

        result = await enforce_splits(state)

        assert result["split_ratios_valid"] is True

    async def test_includes_all_ratio_checks_in_output(self):
        """Should include ratio checks for all 4 splits."""
        state = {
            "train_ratio": 0.60,
            "validation_ratio": 0.20,
            "test_ratio": 0.15,
            "holdout_ratio": 0.05,
            "train_samples": 600,
            "validation_samples": 200,
            "test_samples": 150,
            "holdout_samples": 50,
            "total_samples": 1000,
        }

        result = await enforce_splits(state)

        # Should have checks for all 4 splits
        checks = result["split_ratio_checks"]
        assert any("train" in check.lower() for check in checks)
        assert any("validation" in check.lower() for check in checks)
        assert any("test" in check.lower() for check in checks)
        assert any("holdout" in check.lower() for check in checks)

    async def test_validation_message_includes_actual_ratios(self):
        """Validation message should include actual split ratios."""
        state = {
            "train_ratio": 0.60,
            "validation_ratio": 0.20,
            "test_ratio": 0.15,
            "holdout_ratio": 0.05,
            "train_samples": 600,
            "validation_samples": 200,
            "test_samples": 150,
            "holdout_samples": 50,
            "total_samples": 1000,
        }

        result = await enforce_splits(state)

        message = result["split_validation_message"]
        # Should contain ratio information
        assert "60" in message or "0.6" in message
        assert "1,000" in message or "1000" in message

    async def test_failed_validation_message_includes_errors(self):
        """Failed validation message should mention errors."""
        state = {
            "train_ratio": 0.50,  # Too low
            "validation_ratio": 0.30,  # Too high
            "test_ratio": 0.15,
            "holdout_ratio": 0.05,
            "train_samples": 500,
            "validation_samples": 300,
            "test_samples": 150,
            "holdout_samples": 50,
            "total_samples": 1000,
        }

        result = await enforce_splits(state)

        assert result["split_ratios_valid"] is False
        assert "FAILED" in result["split_validation_message"]

    async def test_empty_leakage_warnings_when_valid(self):
        """Should have empty leakage warnings when all validations pass."""
        state = {
            "train_ratio": 0.60,
            "validation_ratio": 0.20,
            "test_ratio": 0.15,
            "holdout_ratio": 0.05,
            "train_samples": 600,
            "validation_samples": 200,
            "test_samples": 150,
            "holdout_samples": 50,
            "total_samples": 1000,
        }

        result = await enforce_splits(state)

        assert result["leakage_warnings"] == []

    async def test_multiple_violations_accumulate(self):
        """Should accumulate multiple validation failures."""
        state = {
            "train_ratio": 0.50,  # Too low (violation 1)
            "validation_ratio": 0.30,  # Too high (violation 2)
            "test_ratio": 0.10,  # Too low (violation 3)
            "holdout_ratio": 0.08,  # Too high (violation 4)
            "train_samples": 500,
            "validation_samples": 300,
            "test_samples": 100,
            "holdout_samples": 80,
            "total_samples": 980,  # Doesn't sum to 1.0 (violation 5)
        }

        result = await enforce_splits(state)

        assert result["split_ratios_valid"] is False
        # Should have multiple ratio check failures
        failed_checks = [c for c in result["split_ratio_checks"] if "outside" in c.lower()]
        assert len(failed_checks) >= 2

    async def test_zero_samples_in_split(self):
        """Should fail when a split has zero samples."""
        state = {
            "train_ratio": 0.70,
            "validation_ratio": 0.20,
            "test_ratio": 0.10,
            "holdout_ratio": 0.00,  # Zero holdout
            "train_samples": 700,
            "validation_samples": 200,
            "test_samples": 100,
            "holdout_samples": 0,  # Zero holdout samples
            "total_samples": 1000,
        }

        result = await enforce_splits(state)

        assert result["split_ratios_valid"] is False
        # Should warn about zero samples in holdout
        assert any("holdout" in warning.lower() for warning in result["leakage_warnings"])


class TestCheckDuplicateIndices:
    """Test duplicate index detection between splits."""

    def test_detects_train_validation_overlap(self):
        """Should detect duplicate indices between train and validation splits."""
        train_data = {"indices": [0, 1, 2, 3, 4]}
        validation_data = {"indices": [4, 5, 6]}  # Index 4 overlaps
        test_data = {"indices": [7, 8, 9]}
        holdout_data = {"indices": [10, 11]}

        warnings = _check_duplicate_indices(train_data, validation_data, test_data, holdout_data)

        assert len(warnings) == 1
        assert "CRITICAL" in warnings[0]
        assert "train and validation" in warnings[0].lower()
        assert "1" in warnings[0]  # 1 duplicate

    def test_detects_train_test_overlap(self):
        """Should detect duplicate indices between train and test splits."""
        train_data = {"indices": [0, 1, 2, 3, 4]}
        validation_data = {"indices": [5, 6, 7]}
        test_data = {"indices": [3, 4, 8, 9]}  # Indices 3, 4 overlap with train
        holdout_data = None

        warnings = _check_duplicate_indices(train_data, validation_data, test_data, holdout_data)

        assert len(warnings) == 1
        assert "CRITICAL" in warnings[0]
        assert "train and test" in warnings[0].lower()
        assert "2" in warnings[0]  # 2 duplicates

    def test_detects_validation_test_overlap(self):
        """Should detect duplicate indices between validation and test splits."""
        train_data = {"indices": [0, 1, 2]}
        validation_data = {"indices": [3, 4, 5]}
        test_data = {"indices": [5, 6, 7]}  # Index 5 overlaps with validation
        holdout_data = None

        warnings = _check_duplicate_indices(train_data, validation_data, test_data, holdout_data)

        assert len(warnings) == 1
        assert "CRITICAL" in warnings[0]
        assert "validation and test" in warnings[0].lower()

    def test_detects_train_holdout_overlap(self):
        """Should detect duplicate indices between train and holdout splits."""
        train_data = {"indices": [0, 1, 2, 3, 4]}
        validation_data = {"indices": [5, 6]}
        test_data = {"indices": [7, 8]}
        holdout_data = {"indices": [2, 9]}  # Index 2 overlaps with train

        warnings = _check_duplicate_indices(train_data, validation_data, test_data, holdout_data)

        assert len(warnings) == 1
        assert "CRITICAL" in warnings[0]
        assert "train and holdout" in warnings[0].lower()

    def test_no_warnings_when_splits_isolated(self):
        """Should return empty warnings when all splits are properly isolated."""
        train_data = {"indices": [0, 1, 2, 3]}
        validation_data = {"indices": [4, 5, 6]}
        test_data = {"indices": [7, 8, 9]}
        holdout_data = {"indices": [10, 11]}

        warnings = _check_duplicate_indices(train_data, validation_data, test_data, holdout_data)

        assert len(warnings) == 0

    def test_handles_empty_indices(self):
        """Should handle empty indices gracefully."""
        train_data = {"indices": []}
        validation_data = {"indices": []}
        test_data = {"indices": []}
        holdout_data = None

        warnings = _check_duplicate_indices(train_data, validation_data, test_data, holdout_data)

        assert len(warnings) == 0

    def test_detects_multiple_overlaps(self):
        """Should detect multiple overlaps across different split pairs."""
        train_data = {"indices": [0, 1, 2, 3]}
        validation_data = {"indices": [3, 4, 5]}  # Overlaps with train
        test_data = {"indices": [5, 6, 7]}  # Overlaps with validation
        holdout_data = {"indices": [0, 8]}  # Overlaps with train

        warnings = _check_duplicate_indices(train_data, validation_data, test_data, holdout_data)

        assert len(warnings) == 3  # train-val, val-test, train-holdout


class TestGetIndices:
    """Test index extraction from split data."""

    def test_extracts_explicit_indices(self):
        """Should extract explicitly provided indices."""
        split_data = {"indices": [0, 1, 2, 3]}

        indices = _get_indices(split_data)

        assert indices == {0, 1, 2, 3}

    def test_extracts_from_dataframe_index(self):
        """Should extract indices from DataFrame index."""
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])
        split_data = {"X": df}

        indices = _get_indices(split_data)

        assert indices == {10, 20, 30}

    def test_extracts_from_numpy_array(self):
        """Should use row numbers for numpy arrays."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        split_data = {"X": arr}

        indices = _get_indices(split_data)

        assert indices == {0, 1, 2}

    def test_returns_empty_set_for_none(self):
        """Should return empty set for None split data."""
        indices = _get_indices(None)

        assert indices == set()

    def test_returns_empty_set_for_empty_dict(self):
        """Should return empty set for empty split data."""
        indices = _get_indices({})

        assert indices == set()


class TestCheckFeatureLeakage:
    """Test feature leakage detection."""

    def test_detects_direct_target_leakage(self):
        """Should detect when target column is in features."""
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "target": [0, 1, 0],  # Target in features!
            }
        )

        warning = _check_feature_leakage(X, "target")

        assert warning is not None
        assert "CRITICAL" in warning
        assert "direct leakage" in warning.lower()

    def test_detects_similar_name_leakage(self):
        """Should warn about features with similar names to target."""
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "target_prediction": [0.5, 0.7, 0.3],  # Similar to target
            }
        )

        warning = _check_feature_leakage(X, "target")

        assert warning is not None
        assert "WARNING" in warning
        assert "may leak" in warning.lower()

    def test_no_warning_when_no_leakage(self):
        """Should return None when no leakage detected."""
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
            }
        )

        warning = _check_feature_leakage(X, "target")

        assert warning is None

    def test_handles_none_input(self):
        """Should handle None input gracefully."""
        warning = _check_feature_leakage(None, "target")

        assert warning is None

    def test_case_insensitive_matching(self):
        """Should match target regardless of case."""
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "TARGET": [0, 1, 0],  # Uppercase target
            }
        )

        warning = _check_feature_leakage(X, "target")

        assert warning is not None
        assert "CRITICAL" in warning


class TestCheckTemporalOrdering:
    """Test temporal ordering validation."""

    def test_detects_train_validation_temporal_leak(self):
        """Should detect when train data is after validation data."""
        train_data = {
            "X": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-15", "2024-01-20", "2024-01-25"]),
                    "value": [1, 2, 3],
                }
            )
        }
        validation_data = {
            "X": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-10", "2024-01-12"]),  # Before train!
                    "value": [4, 5],
                }
            )
        }
        test_data = {
            "X": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-30"]),
                    "value": [6],
                }
            )
        }

        warnings = _check_temporal_ordering(train_data, validation_data, test_data, "timestamp")

        assert len(warnings) >= 1
        assert any("temporal leakage" in w.lower() for w in warnings)

    def test_detects_train_test_temporal_leak(self):
        """Should detect when train data is after test data."""
        train_data = {
            "X": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-20", "2024-01-25"]),
                    "value": [1, 2],
                }
            )
        }
        validation_data = {
            "X": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-26", "2024-01-27"]),
                    "value": [3, 4],
                }
            )
        }
        test_data = {
            "X": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-10", "2024-01-15"]),  # Before train!
                    "value": [5, 6],
                }
            )
        }

        warnings = _check_temporal_ordering(train_data, validation_data, test_data, "timestamp")

        assert len(warnings) >= 1
        assert any("test" in w.lower() for w in warnings)

    def test_no_warning_when_properly_ordered(self):
        """Should return empty when data is properly ordered."""
        train_data = {
            "X": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-01", "2024-01-10"]),
                    "value": [1, 2],
                }
            )
        }
        validation_data = {
            "X": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-15", "2024-01-20"]),
                    "value": [3, 4],
                }
            )
        }
        test_data = {
            "X": pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-25", "2024-01-30"]),
                    "value": [5, 6],
                }
            )
        }

        warnings = _check_temporal_ordering(train_data, validation_data, test_data, "timestamp")

        assert len(warnings) == 0

    def test_handles_missing_time_column(self):
        """Should handle missing time column gracefully."""
        train_data = {"X": pd.DataFrame({"value": [1, 2, 3]})}
        validation_data = {"X": pd.DataFrame({"value": [4, 5]})}
        test_data = {"X": pd.DataFrame({"value": [6]})}

        warnings = _check_temporal_ordering(train_data, validation_data, test_data, "timestamp")

        assert len(warnings) == 0  # Should not crash, just return empty

    def test_handles_none_data(self):
        """Should handle None time values gracefully."""
        train_data = {"X": None}
        validation_data = {"X": None}
        test_data = {"X": None}

        warnings = _check_temporal_ordering(train_data, validation_data, test_data, "timestamp")

        assert len(warnings) == 0
