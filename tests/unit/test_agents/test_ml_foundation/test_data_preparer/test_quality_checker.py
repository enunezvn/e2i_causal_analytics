"""Unit tests for quality_checker node."""

from datetime import datetime

import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.quality_checker import (
    _check_completeness,
    run_quality_checks,
)


@pytest.fixture
def mock_state():
    """Create a mock state for testing."""
    # Create sample DataFrame
    train_df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
        }
    )

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


class TestCheckCompletenessExcludedFeatures:
    """Backlog #13: ``_check_completeness`` honors ``excluded_features``.

    Always-null metadata cols (CSU's ``risk_score``, ``state``,
    ``data_lag_hours``, etc.) used to drag the global completeness
    score below 0.90 and block QC for reasons unrelated to the actual
    feature surface. The fix filters declared excluded cols out of
    both the table-level ratio and per-column null-percentage warnings.
    """

    def test_excluded_features_dropped_from_completeness(self) -> None:
        df = pd.DataFrame(
            {
                "patient_id": ["p1", "p2", "p3", "p4"],
                "feature_a": [1.0, 2.0, 3.0, 4.0],
                "metadata_x": [None, None, None, None],
                "metadata_y": [None, None, None, None],
            }
        )
        score_no_exclude, _ = _check_completeness(df, required_columns=["patient_id"])
        score_excluded, results = _check_completeness(
            df,
            required_columns=["patient_id"],
            excluded_features=["metadata_x", "metadata_y"],
        )
        # Without the filter half the cells are null → completeness 0.5.
        assert score_no_exclude == pytest.approx(0.5)
        # With the filter only the populated cols count → 1.0.
        assert score_excluded == pytest.approx(1.0)
        # The expectation result records the exclusion count.
        table_result = next(
            r for r in results if r["expectation_type"] == "expect_table_completeness"
        )
        assert table_result["result"]["excluded_features_count"] == 2

    def test_required_columns_always_checked_even_if_in_excluded(self) -> None:
        df = pd.DataFrame(
            {
                "id": [None, "p2", "p3"],
                "metadata": [None, None, None],
            }
        )
        # ``id`` is both required AND incorrectly listed as excluded —
        # the required-columns check must still run.
        _, results = _check_completeness(
            df,
            required_columns=["id"],
            excluded_features=["id", "metadata"],
        )
        per_col = [
            r
            for r in results
            if r["expectation_type"] == "expect_column_values_to_not_be_null"
            and r.get("column") == "id"
        ]
        assert per_col, "required-column check on 'id' was silently dropped"
        # ``success`` may be numpy bool; compare value, not identity.
        assert not per_col[0]["success"]
        assert per_col[0]["severity"] == "blocking"

    def test_excluded_features_skipped_in_per_col_warnings(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "metadata": [None, None, None],  # 100% null
            }
        )
        _, results_no_exclude = _check_completeness(df, required_columns=["id"])
        _, results_excluded = _check_completeness(
            df, required_columns=["id"], excluded_features=["metadata"]
        )
        warnings_no_exclude = [
            r
            for r in results_no_exclude
            if r["expectation_type"] == "expect_column_null_percentage"
        ]
        warnings_excluded = [
            r
            for r in results_excluded
            if r["expectation_type"] == "expect_column_null_percentage"
        ]
        # Without exclusion, ``metadata`` triggers a high-null warning.
        assert any(w.get("column") == "metadata" for w in warnings_no_exclude)
        # With exclusion, no warnings on excluded cols.
        assert not any(w.get("column") == "metadata" for w in warnings_excluded)

    def test_no_exclusions_preserves_legacy_behavior(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        score_default, _ = _check_completeness(df, required_columns=["a"])
        score_explicit, _ = _check_completeness(
            df, required_columns=["a"], excluded_features=[]
        )
        score_none, _ = _check_completeness(
            df, required_columns=["a"], excluded_features=None
        )
        assert score_default == pytest.approx(1.0)
        assert score_explicit == pytest.approx(1.0)
        assert score_none == pytest.approx(1.0)
