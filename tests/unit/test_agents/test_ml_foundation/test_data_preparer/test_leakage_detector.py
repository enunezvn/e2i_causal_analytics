"""Unit tests for leakage_detector node."""

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import (
    check_mutual_information,
    check_single_feature_auc,
    check_target_leakage,
    check_train_test_contamination,
    detect_leakage,
)


@pytest.fixture
def mock_state_no_leakage():
    """Create mock state with no leakage.

    Uses larger n and a target that's truly independent of features so the noise
    floor stays below the (Phase-1) MODERATE threshold of 0.55 effective AUC.
    """
    rng = np.random.default_rng(42)
    n_train = 500
    n_val = 150
    # Generate target FIRST, then independent features so AUC stays near 0.50.
    target_train = rng.binomial(1, 0.3, n_train)
    target_val = rng.binomial(1, 0.3, n_val)
    train_df = pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, n_train),
            "feature2": rng.normal(0, 1, n_train),
            "target": target_train,
        }
    )
    validation_df = pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, n_val),
            "feature2": rng.normal(0, 1, n_val),
            "target": target_val,
        },
        index=range(n_train, n_train + n_val),  # Non-overlapping with train
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


# =============================================================================
# Phase 1 (ml-leakage-holistic-fix) — Tightened threshold tests
#
# Background: csu_sub_gap_e2e_rerun_close_20260507.md flagged journey_duration_days
# (single-feature AUC=0.689) as escaping detection because the existing thresholds
# only flag features with single-feature AUC > 0.80. Phase 1 lowers the floors:
#   - single-feature AUC: 0.55 MODERATE / 0.65 HIGH / 0.80 CRITICAL
#   - target correlation: 0.50 (p<0.001) MODERATE / 0.70 HIGH / 0.85 CRITICAL
#   - mutual information: 0.30 MODERATE / 0.50 HIGH / 0.70 CRITICAL
# =============================================================================


def _make_feature_with_target_auc(
    target: np.ndarray, target_auc: float, seed: int = 42
) -> np.ndarray:
    """Construct a numeric feature whose single-feature AUC against `target` is ~target_auc.

    Mechanism: x = signal_weight * target + N(0, 1). signal_weight is empirically
    tuned (verified against `roc_auc_score`) to produce the requested AUC for
    binomial(0.30) targets, n≈2000. Returns the feature vector.
    """
    rng = np.random.default_rng(seed)
    signal_weights = {
        0.58: 0.30,
        0.65: 0.55,
        0.69: 0.70,
        0.75: 0.95,
        0.85: 1.55,
        0.95: 3.4,
    }
    weight = signal_weights.get(round(target_auc, 2), target_auc * 1.5)
    return weight * target.astype(float) + rng.normal(0.0, 1.0, len(target))


def test_single_feature_auc_flags_journey_duration_069():
    """A feature with single-feature AUC ~0.69 must be flagged HIGH or MODERATE.

    Reproduces the journey_duration_days mechanism from csu_sub_gap_e2e_rerun_close_20260507.md.
    Pre-Phase-1 thresholds (>0.80 HIGH, >0.90 CRITICAL) miss this entirely.
    """
    rng = np.random.default_rng(42)
    n = 2000
    target = rng.binomial(1, 0.30, n)
    leaky = _make_feature_with_target_auc(target, 0.69, seed=11)
    df = pd.DataFrame({"target": target, "leak_069": leaky})

    findings = check_single_feature_auc(df, "target", ["leak_069"])

    assert len(findings) >= 1, (
        f"Expected ≥1 finding for AUC≈0.69 feature; got {len(findings)}. "
        f"Pre-Phase-1 the 0.80 floor missed this."
    )
    assert findings[0].feature == "leak_069"
    assert findings[0].severity.value in ("moderate", "high", "critical")


def test_single_feature_auc_flags_moderate_058():
    """A feature with single-feature AUC ~0.58 must be flagged MODERATE.

    The MODERATE band is (0.55, 0.65]. We pick 0.58 to be safely above the noise
    floor (which sits ~0.54 for clean random fixtures) and well within MODERATE.
    """
    rng = np.random.default_rng(7)
    n = 2000
    target = rng.binomial(1, 0.30, n)
    leaky = _make_feature_with_target_auc(target, 0.58, seed=13)
    df = pd.DataFrame({"target": target, "leak_058": leaky})

    findings = check_single_feature_auc(df, "target", ["leak_058"])

    assert any(f.feature == "leak_058" for f in findings), (
        "Expected MODERATE finding for AUC≈0.58 feature"
    )


def test_single_feature_auc_does_not_flag_below_054():
    """Pure noise (AUC≈0.50) must NOT be flagged — avoid false positives."""
    rng = np.random.default_rng(5)
    n = 2000
    target = rng.binomial(1, 0.30, n)
    noise = rng.normal(0, 1, n)  # No relationship to target
    df = pd.DataFrame({"target": target, "noise": noise})

    findings = check_single_feature_auc(df, "target", ["noise"])

    assert all(f.feature != "noise" for f in findings), (
        f"Pure noise should NOT trigger a finding; got: {[f.to_dict() for f in findings]}"
    )


def test_single_feature_auc_critical_at_080():
    """Tightened: AUC > 0.80 should escalate to CRITICAL (was HIGH pre-Phase-1)."""
    rng = np.random.default_rng(3)
    n = 2000
    target = rng.binomial(1, 0.30, n)
    very_leaky = _make_feature_with_target_auc(target, 0.85, seed=17)
    df = pd.DataFrame({"target": target, "leak_085": very_leaky})

    findings = check_single_feature_auc(df, "target", ["leak_085"])

    assert any(f.feature == "leak_085" for f in findings)
    leaky_finding = next(f for f in findings if f.feature == "leak_085")
    assert leaky_finding.severity.value == "critical", (
        f"Expected CRITICAL for AUC>0.80; got {leaky_finding.severity.value}"
    )


def test_target_correlation_flags_moderate_060():
    """Pearson |r|≈0.60 with p<0.001 must be flagged MODERATE (was missed pre-Phase-1)."""
    rng = np.random.default_rng(23)
    n = 4000
    target = rng.binomial(1, 0.30, n).astype(float)
    feature = 0.60 * target + rng.normal(0.0, 1.0, n)  # |r|≈0.30 effective; tune
    # Increase signal until |r|≈0.60
    feature = 1.5 * target + rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"target": target, "moderate_060": feature})

    issues, findings = check_target_leakage(df, "target", ["moderate_060"])

    assert any(f.feature == "moderate_060" for f in findings), (
        "Expected target-correlation finding at |r|≈0.60 with p<0.001"
    )


def test_mi_flags_moderate_040():
    """Normalized MI ~0.40 must be flagged MODERATE (was missed pre-Phase-1)."""
    rng = np.random.default_rng(29)
    n = 4000
    target = rng.binomial(1, 0.30, n)
    # sigma=0.4 yields normalized MI ~0.40 — empirically verified
    feature = (target + rng.normal(0, 0.4, n) > 0.5).astype(int)
    df = pd.DataFrame({"target": target, "mi_moderate": feature})

    findings = check_mutual_information(df, "target", ["mi_moderate"])

    assert any(f.feature == "mi_moderate" for f in findings), (
        "Expected MI finding for moderate-strength relationship (MI_norm≈0.40)"
    )


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
