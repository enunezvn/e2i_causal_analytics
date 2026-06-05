"""Unit tests for leakage_detector node."""

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import (
    check_target_leakage,
    check_train_test_contamination,
    check_zero_variance_within_class,
    detect_leakage,
)


@pytest.fixture
def mock_state_no_leakage():
    """Create mock state with no leakage."""
    np.random.seed(42)
    train_df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.binomial(1, 0.3, 100),
        }
    )
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

    train_df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "leaky_feature": leaky_feature,
            "target": target,
        }
    )

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
    # Create data where validation rows are exact copies of training rows
    # (row-hash based detection requires identical data, not just index overlap)
    train_df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "target": [0, 1, 0, 1, 0],
        }
    )
    # Validation contains exact duplicates of train rows 3 and 4
    validation_df = pd.DataFrame(
        {
            "feature1": [4, 5],
            "target": [1, 0],
        },
        index=[3, 4],
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
    rng = np.random.RandomState(42)
    n = 20
    target = np.arange(n)
    df = pd.DataFrame(
        {
            "leaky": target.copy(),  # perfect correlation -> CRITICAL
            "target": target,
            "clean": rng.permutation(n),  # uncorrelated -> no finding
        }
    )

    issues, findings = check_target_leakage(df, "target", ["leaky", "clean"])

    # Should detect leaky feature in both legacy and structured outputs
    assert len(issues) > 0
    assert any("leaky" in issue for issue in issues)
    assert any(f.feature == "leaky" for f in findings)


def test_check_train_test_contamination_direct():
    """Test check_train_test_contamination function directly."""
    # Exact duplicate rows (hash-based detection requires identical data)
    train_df = pd.DataFrame({"col": [1, 2, 3]}, index=[0, 1, 2])
    test_df = pd.DataFrame({"col": [2, 3]}, index=[1, 2])  # Same data as train rows 1,2

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


class TestZeroVarianceRareEventGuard_RC1:
    """RC1 — check_zero_variance_within_class must NOT flag a cardinality-2 sparse
    pre-index predictor as leakage on a rare-event cohort. Constant-within-the-
    tiny-positive-class is small-sample degeneracy, not a leak."""

    @staticmethod
    def _rare_event_card2_cohort(n: int = 1000, n_pos: int = 30, n_flag: int = 40, seed: int = 0):
        rng = np.random.default_rng(seed)
        y = np.zeros(n, dtype=int)
        y[rng.choice(n, size=n_pos, replace=False)] = 1
        # All the 1s land in the NEGATIVE class -> std_1 == 0, mean_1 == 0,
        # std_0 > 0, mean_0 > 0. Pre-guard this triggered the HIGH branch; the
        # RC1 guard now skips it.
        flag = np.zeros(n, dtype=float)
        neg_idx = np.where(y == 0)[0]
        flag[rng.choice(neg_idx, size=n_flag, replace=False)] = 1.0
        return pd.DataFrame({"has_asthma": flag, "target": y}), "target", "has_asthma"

    def test_sparse_card2_flag_not_flagged_on_rare_event_cohort(self):
        df, target, feat = self._rare_event_card2_cohort()
        findings = check_zero_variance_within_class(df, target, [feat])
        assert findings == [], (
            "RC1 regression: zero_variance_within_class false-fired on a legitimate "
            f"cardinality-2 sparse predictor on a rare-event cohort: {findings}"
        )

    def test_guard_does_not_suppress_a_dense_separating_leak(self):
        rng = np.random.default_rng(1)
        n = 400
        y = rng.integers(0, 2, n)  # ~50% prevalence, dense
        feat = np.where(y == 1, 5.0, rng.normal(0.0, 0.0001, n))  # std_1==0, means differ
        df = pd.DataFrame({"leaky": feat, "target": y})
        findings = check_zero_variance_within_class(df, "target", ["leaky"])
        assert any(f.feature == "leaky" for f in findings), (
            "guard over-reached: a dense balanced within-class-constant separator "
            "must still be flagged"
        )

    def test_small_positive_class_above_5pct_still_guarded(self):
        """Exercises the len(class_1) < 30 arm: n_pos=20 in n=100 -> pos_rate=0.20
        (>5%, so the pos_rate arm is False) but the absolute positive count is < 30."""
        rng = np.random.default_rng(2)
        n, n_pos, n_flag = 100, 20, 8
        y = np.zeros(n, dtype=int)
        y[rng.choice(n, size=n_pos, replace=False)] = 1
        flag = np.zeros(n, dtype=float)
        neg_idx = np.where(y == 0)[0]
        flag[rng.choice(neg_idx, size=n_flag, replace=False)] = 1.0  # all 1s in negatives
        df = pd.DataFrame({"has_dx": flag, "target": y})
        findings = check_zero_variance_within_class(df, "target", ["has_dx"])
        assert findings == [], f"guard's len(class_1)<30 arm did not fire: {findings}"
