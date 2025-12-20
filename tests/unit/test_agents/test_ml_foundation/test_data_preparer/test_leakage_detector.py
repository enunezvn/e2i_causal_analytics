"""Unit tests for leakage_detector node."""

import pytest
import pandas as pd
import numpy as np
from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import (
    detect_leakage,
    check_target_leakage,
    check_train_test_contamination,
)


@pytest.fixture
def mock_state_no_leakage():
    """Create mock state with no leakage."""
    np.random.seed(42)
    train_df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.binomial(1, 0.3, 100),
    })
    # Use non-overlapping indices to avoid train-test contamination
    validation_df = pd.DataFrame(
        {
            "feature1": np.random.randn(30),
            "feature2": np.random.randn(30),
            "target": np.random.binomial(1, 0.3, 30),
        },
        index=range(100, 130),  # Non-overlapping with train (0-99)
    )

    return {
        "experiment_id": "exp_test_123",
        "train_df": train_df,
        "validation_df": validation_df,
        "scope_spec": {
            "required_features": ["feature1", "feature2"],
            "prediction_target": "target",
        },
        "skip_leakage_check": False,
    }


@pytest.fixture
def mock_state_target_leakage():
    """Create mock state with target leakage."""
    np.random.seed(42)
    target = np.random.binomial(1, 0.3, 100)

    # Create a feature that's almost identical to target (leakage!)
    leaky_feature = target + np.random.randn(100) * 0.01

    train_df = pd.DataFrame({
        "feature1": np.random.randn(100),
        "leaky_feature": leaky_feature,
        "target": target,
    })

    return {
        "experiment_id": "exp_test_123",
        "train_df": train_df,
        "scope_spec": {
            "required_features": ["feature1", "leaky_feature"],
            "prediction_target": "target",
        },
        "skip_leakage_check": False,
    }


@pytest.fixture
def mock_state_train_test_contamination():
    """Create mock state with train-test contamination."""
    # Create overlapping indices
    train_df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "target": [0, 1, 0, 1, 0],
    })
    # Validation has overlapping indices 3 and 4!
    validation_df = pd.DataFrame(
        {
            "feature1": [33, 44],
            "target": [1, 0],
        },
        index=[3, 4],  # CONTAMINATION!
    )

    return {
        "experiment_id": "exp_test_123",
        "train_df": train_df,
        "validation_df": validation_df,
        "scope_spec": {
            "required_features": ["feature1"],
            "prediction_target": "target",
        },
        "skip_leakage_check": False,
    }


@pytest.mark.asyncio
async def test_detect_leakage_no_issues(mock_state_no_leakage):
    """Test leakage detection with clean data."""
    result = await detect_leakage(mock_state_no_leakage)

    assert "leakage_detected" in result
    assert result["leakage_detected"] is False
    assert "leakage_issues" in result
    assert len(result["leakage_issues"]) == 0


@pytest.mark.asyncio
async def test_detect_leakage_skip_check():
    """Test that leakage check can be skipped."""
    state = {
        "experiment_id": "exp_test_123",
        "skip_leakage_check": True,
    }

    result = await detect_leakage(state)

    assert result["leakage_detected"] is False
    assert len(result["leakage_issues"]) > 0
    assert "skipped" in result["leakage_issues"][0].lower()


@pytest.mark.asyncio
async def test_detect_target_leakage(mock_state_target_leakage):
    """Test detection of target leakage."""
    result = await detect_leakage(mock_state_target_leakage)

    # Should detect the leaky feature
    assert result["leakage_detected"] is True
    assert "leakage_issues" in result
    assert len(result["leakage_issues"]) > 0

    # Should mention the leaky feature
    issues_text = " ".join(result["leakage_issues"])
    assert "leaky_feature" in issues_text or "target leakage" in issues_text.lower()


@pytest.mark.asyncio
async def test_detect_train_test_contamination(mock_state_train_test_contamination):
    """Test detection of train-test contamination."""
    result = await detect_leakage(mock_state_train_test_contamination)

    # Should detect the contamination
    assert result["leakage_detected"] is True
    assert len(result["leakage_issues"]) > 0

    # Should mention contamination
    issues_text = " ".join(result["leakage_issues"])
    assert "contamination" in issues_text.lower() or "overlap" in issues_text.lower()


@pytest.mark.asyncio
async def test_leakage_adds_to_blocking_issues(mock_state_target_leakage):
    """Test that leakage detection adds to blocking_issues."""
    # Add existing blocking issue
    state = mock_state_target_leakage.copy()
    state["blocking_issues"] = ["Existing issue"]

    result = await detect_leakage(state)

    # Should have added leakage issues to blocking_issues
    if result["leakage_detected"]:
        assert "blocking_issues" in result
        blocking = result["blocking_issues"]
        assert len(blocking) > 1  # Existing + leakage issues


def test_check_target_leakage_direct():
    """Test check_target_leakage function directly."""
    # Create data with perfect correlation (leakage!)
    df = pd.DataFrame({
        "leaky": [1, 2, 3, 4, 5],
        "target": [1, 2, 3, 4, 5],  # Perfect correlation!
        "clean": [5, 4, 3, 2, 1],
    })

    issues = check_target_leakage(df, "target", ["leaky", "clean"])

    # Should detect leaky feature
    assert len(issues) > 0
    assert any("leaky" in issue for issue in issues)


def test_check_train_test_contamination_direct():
    """Test check_train_test_contamination function directly."""
    train_df = pd.DataFrame({"col": [1, 2, 3]}, index=[0, 1, 2])
    test_df = pd.DataFrame({"col": [4, 5]}, index=[1, 2])  # Overlap!

    issues = check_train_test_contamination(train_df, test_df=test_df)

    # Should detect contamination
    assert len(issues) > 0
    assert any("contamination" in issue.lower() for issue in issues)


@pytest.mark.asyncio
async def test_leakage_detector_missing_train_df():
    """Test leakage detection with missing train_df."""
    state = {
        "experiment_id": "exp_test_123",
        "skip_leakage_check": False,
        # Missing train_df
    }

    result = await detect_leakage(state)

    # Should handle error gracefully
    assert "error" in result
    assert result["error_type"] == "leakage_detection_error"
    assert result["leakage_detected"] is True  # Fail safe
