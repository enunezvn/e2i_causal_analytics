"""Unit tests for quality_checker node."""

import pytest
import pandas as pd
from datetime import datetime
from src.agents.ml_foundation.data_preparer.nodes.quality_checker import (
    run_quality_checks,
)


@pytest.fixture
def mock_state():
    """Create a mock state for testing."""
    # Create sample DataFrame
    train_df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": ["a", "b", "c", "d", "e"],
        "target": [0, 1, 0, 1, 0],
    })

    return {
        "experiment_id": "exp_test_123",
        "train_df": train_df,
        "scope_spec": {
            "experiment_id": "exp_test_123",
            "required_features": ["feature1", "feature2"],
            "prediction_target": "target",
        },
    }


@pytest.mark.asyncio
async def test_run_quality_checks_success(mock_state):
    """Test quality checks with passing data."""
    result = await run_quality_checks(mock_state)

    # Check that report ID was generated
    assert "report_id" in result
    assert result["report_id"].startswith("qc_exp_test_123_")

    # Check QC status
    assert result["qc_status"] in ["passed", "warning", "failed", "skipped"]

    # Check scores are in valid range
    assert 0.0 <= result["overall_score"] <= 1.0
    assert 0.0 <= result["completeness_score"] <= 1.0
    assert 0.0 <= result["validity_score"] <= 1.0
    assert 0.0 <= result["consistency_score"] <= 1.0
    assert 0.0 <= result["uniqueness_score"] <= 1.0
    assert 0.0 <= result["timeliness_score"] <= 1.0

    # Check row and column counts
    assert result["row_count"] == 5
    assert result["column_count"] == 3

    # Check timestamp format
    assert "validated_at" in result
    datetime.fromisoformat(result["validated_at"])  # Should not raise


@pytest.mark.asyncio
async def test_run_quality_checks_missing_train_df():
    """Test quality checks with missing train_df."""
    state = {
        "experiment_id": "exp_test_123",
        # Missing train_df
    }

    result = await run_quality_checks(state)

    # Should handle error gracefully
    assert "error" in result
    assert result["error_type"] == "quality_check_error"
    assert result["qc_status"] == "failed"
    assert len(result["blocking_issues"]) > 0


@pytest.mark.asyncio
async def test_quality_checks_low_score_blocks():
    """Test that low QC score results in blocking issues."""
    state = {
        "experiment_id": "exp_test_123",
        "train_df": pd.DataFrame({"col": [1, 2]}),  # Very small dataset
    }

    result = await run_quality_checks(state)

    # If overall score < 0.80, should have blocking issues
    if result["overall_score"] < 0.80:
        assert len(result["blocking_issues"]) > 0
        assert result["qc_status"] == "failed"


@pytest.mark.asyncio
async def test_quality_checks_duration_logged(mock_state):
    """Test that validation duration is logged."""
    result = await run_quality_checks(mock_state)

    assert "validation_duration_seconds" in result
    assert result["validation_duration_seconds"] >= 0.0
