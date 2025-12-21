"""Tests for split enforcer node."""

import pytest

from src.agents.ml_foundation.model_trainer.nodes.split_enforcer import enforce_splits


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
