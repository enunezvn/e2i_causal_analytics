"""Tests for QC gate logic.

The QC gate is CRITICAL - it blocks downstream training if data quality fails.
These tests verify the gate logic matches the contract in tier0-contracts.md.
"""

import pytest
from src.agents.ml_foundation.data_preparer.graph import finalize_output


@pytest.fixture
def base_state():
    """Create a base state for QC gate testing."""
    return {
        "experiment_id": "exp_gate_test_123",
        "qc_status": "passed",
        "overall_score": 0.90,
        "completeness_score": 0.95,
        "validity_score": 0.92,
        "consistency_score": 0.89,
        "uniqueness_score": 0.96,
        "timeliness_score": 0.85,
        "expectation_results": [],
        "failed_expectations": [],
        "warnings": [],
        "remediation_steps": [],
        "blocking_issues": [],
        "report_id": "qc_test_123",
        "row_count": 1000,
        "column_count": 10,
        "validated_at": "2025-01-01T00:00:00",
        "scope_spec": {
            "required_features": ["feature1", "feature2"],
        },
    }


@pytest.mark.asyncio
async def test_qc_gate_passes_with_good_quality(base_state, mocker):
    """Test that QC gate passes with good data quality."""
    # Mock the train_df
    import pandas as pd

    mocker.patch.object(
        pd,
        "DataFrame",
        return_value=pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]}),
    )
    base_state["train_df"] = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})

    result = await finalize_output(base_state)

    # Gate should pass
    assert result["gate_passed"] is True
    assert result["qc_passed"] is True
    assert result["is_ready"] is True
    assert len(result["blockers"]) == 0


@pytest.mark.asyncio
async def test_qc_gate_blocks_on_failed_status(base_state, mocker):
    """Test that QC gate blocks when status is 'failed'.

    Per tier0-contracts.md:
    if qc_report.status == "failed":
        return False
    """
    import pandas as pd

    base_state["train_df"] = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    base_state["qc_status"] = "failed"

    result = await finalize_output(base_state)

    # Gate should be blocked
    assert result["gate_passed"] is False
    assert result["qc_passed"] is False
    assert result["is_ready"] is False


@pytest.mark.asyncio
async def test_qc_gate_blocks_on_blocking_issues(base_state, mocker):
    """Test that QC gate blocks when there are blocking issues.

    Per tier0-contracts.md:
    if qc_report.blocking_issues:
        return False
    """
    import pandas as pd

    base_state["train_df"] = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    base_state["blocking_issues"] = ["Critical data quality issue"]

    result = await finalize_output(base_state)

    # Gate should be blocked
    assert result["gate_passed"] is False
    assert result["qc_passed"] is False
    assert len(result["blockers"]) > 0


@pytest.mark.asyncio
async def test_qc_gate_blocks_on_low_score(base_state, mocker):
    """Test that QC gate blocks when overall score < 0.80.

    Per tier0-contracts.md:
    if qc_report.overall_score < 0.80:
        return False
    """
    import pandas as pd

    base_state["train_df"] = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    base_state["overall_score"] = 0.75  # Below threshold

    result = await finalize_output(base_state)

    # Gate should be blocked
    assert result["gate_passed"] is False
    assert result["qc_passed"] is False


@pytest.mark.asyncio
async def test_qc_gate_threshold_exactly_080(base_state, mocker):
    """Test QC gate behavior at exactly 0.80 threshold."""
    import pandas as pd

    base_state["train_df"] = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    base_state["overall_score"] = 0.80  # Exactly at threshold

    result = await finalize_output(base_state)

    # Should pass (threshold is <0.80, so 0.80 passes)
    assert result["gate_passed"] is True


@pytest.mark.asyncio
async def test_data_readiness_checks_missing_features(base_state, mocker):
    """Test that data readiness checks for missing required features."""
    import pandas as pd

    # DataFrame missing feature2
    base_state["train_df"] = pd.DataFrame({"feature1": [1, 2]})
    base_state["scope_spec"]["required_features"] = ["feature1", "feature2"]

    result = await finalize_output(base_state)

    # Should identify missing feature
    assert "feature2" in result["missing_required_features"]
    assert "feature2" not in result["available_features"]

    # Should add to blockers
    assert len(result["blockers"]) > 0

    # is_ready should be False (even if QC passed)
    assert result["is_ready"] is False


@pytest.mark.asyncio
async def test_data_readiness_sample_counts(mocker):
    """Test that data readiness correctly counts samples across splits."""
    import pandas as pd

    state = {
        "experiment_id": "exp_test_123",
        "train_df": pd.DataFrame({"col": range(100)}),
        "validation_df": pd.DataFrame({"col": range(20)}),
        "test_df": pd.DataFrame({"col": range(15)}),
        "holdout_df": pd.DataFrame({"col": range(5)}),
        "qc_status": "passed",
        "overall_score": 0.90,
        "blocking_issues": [],
        "scope_spec": {"required_features": ["col"]},
    }

    result = await finalize_output(state)

    # Check sample counts
    assert result["train_samples"] == 100
    assert result["validation_samples"] == 20
    assert result["test_samples"] == 15
    assert result["holdout_samples"] == 5
    assert result["total_samples"] == 140


@pytest.mark.asyncio
async def test_finalize_output_error_handling():
    """Test that finalize_output handles missing fields gracefully.

    The implementation uses defaults for missing fields:
    - qc_status defaults to "skipped"
    - overall_score defaults to 0.0 (below 0.80 threshold)
    - train_df defaults to None (handled gracefully)

    This results in gate_passed=False and is_ready=False.
    """
    # State with minimal fields
    state = {
        "experiment_id": "exp_test_123",
        # Missing many optional fields - uses defaults
    }

    result = await finalize_output(state)

    # Should handle gracefully with defaults
    assert result["gate_passed"] is False  # score 0.0 < 0.80
    assert result["qc_passed"] is False
    assert result["is_ready"] is False
    assert result["total_samples"] == 0
    assert result["available_features"] == []


@pytest.mark.asyncio
async def test_qc_gate_contract_compliance(base_state, mocker):
    """Integration test: Verify QC gate matches tier0-contracts.md specification.

    This test verifies the three blocking conditions from the contract:
    1. status == "failed"
    2. blocking_issues is non-empty
    3. overall_score < 0.80
    """
    import pandas as pd

    base_state["train_df"] = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})

    # Test 1: Status failed
    state1 = base_state.copy()
    state1["qc_status"] = "failed"
    result1 = await finalize_output(state1)
    assert result1["gate_passed"] is False, "Gate should block on status=failed"

    # Test 2: Blocking issues
    state2 = base_state.copy()
    state2["blocking_issues"] = ["Issue 1"]
    result2 = await finalize_output(state2)
    assert result2["gate_passed"] is False, "Gate should block on blocking_issues"

    # Test 3: Low score
    state3 = base_state.copy()
    state3["overall_score"] = 0.79
    result3 = await finalize_output(state3)
    assert result3["gate_passed"] is False, "Gate should block on score < 0.80"

    # Test 4: All conditions pass
    state4 = base_state.copy()
    state4["qc_status"] = "passed"
    state4["blocking_issues"] = []
    state4["overall_score"] = 0.90
    result4 = await finalize_output(state4)
    assert result4["gate_passed"] is True, "Gate should pass when all conditions met"
