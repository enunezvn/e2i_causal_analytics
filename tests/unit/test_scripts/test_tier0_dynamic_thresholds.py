"""Unit tests for the dynamic statistical-threshold helpers in run_tier0_test.

Tier C of the dynamic-threshold pass: the AUC gate becomes CI-aware — it
surfaces the bootstrap confidence interval the evaluator already computes and
optionally requires the model to be *significantly* better than the 0.5
no-skill floor (CI lower bound > 0.5), instead of trusting a bare point
estimate against a hardcoded constant.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.run_tier0_test as tier0  # noqa: E402


def test_auc_ci_from_result_extracts_pair():
    assert tier0._auc_ci_from_result({"confidence_interval": {"auc": (0.61, 0.67)}}) == (0.61, 0.67)
    # list form is accepted too
    assert tier0._auc_ci_from_result({"confidence_interval": {"auc": [0.61, 0.67]}}) == (0.61, 0.67)
    # missing / malformed -> None (degrade gracefully)
    assert tier0._auc_ci_from_result({"confidence_interval": {}}) is None
    assert tier0._auc_ci_from_result({}) is None
    assert tier0._auc_ci_from_result({"confidence_interval": {"auc": (0.61,)}}) is None


def test_auc_gate_point_mode_uses_configured_floor():
    """Default (point) mode: pass iff the point AUC meets the configured floor."""
    passed, _ = tier0._auc_gate_verdict(0.638, (0.61, 0.67), 0.50, require_significance=False)
    assert passed is True
    passed, _ = tier0._auc_gate_verdict(0.48, None, 0.50, require_significance=False)
    assert passed is False


def test_auc_gate_significance_mode_requires_ci_above_half():
    """CI mode: also require the bootstrap CI lower bound to exceed 0.5
    (significantly better than chance)."""
    # point >= floor AND CI lower > 0.5 -> pass
    passed, detail = tier0._auc_gate_verdict(0.638, (0.55, 0.70), 0.50, require_significance=True)
    assert passed is True
    assert "significant" in detail.lower()
    # point >= floor but CI lower <= 0.5 -> NOT significant -> fail
    passed, _ = tier0._auc_gate_verdict(0.638, (0.49, 0.72), 0.50, require_significance=True)
    assert passed is False
    # significance gate ON but no CI available -> cannot establish significance -> fail
    passed, _ = tier0._auc_gate_verdict(0.638, None, 0.50, require_significance=True)
    assert passed is False


def test_auc_gate_no_auc_fails():
    passed, _ = tier0._auc_gate_verdict(0.0, None, 0.50, require_significance=False)
    assert passed is False


# --- Tier D: config-exposed discovery / tie-band / single-model knobs --------


def test_discover_model_feature_cols_configurable_caps():
    """The null-rate and categorical-cardinality caps are configurable (Tier D),
    defaulting to the historical 0.5 / 50; the nunique>1 constant-drop stays a
    correctness rule."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "good_num": list(range(10)),
            # 20% non-null numeric
            "sparse": [1.0, 2.0, None, None, None, None, None, None, None, None],
            "constant": [1] * 10,  # nunique==1 -> always dropped
            "hicard_cat": [f"c{i}" for i in range(10)],  # 10 unique categoricals
            "target": [0, 1] * 5,
        }
    )
    exclude = {"target"}
    cols = tier0._discover_model_feature_cols(df, exclude)
    assert "good_num" in cols
    assert "constant" not in cols  # correctness rule (nunique>1) always applies
    assert "sparse" not in cols  # 20% non-null < default 0.5
    assert "hicard_cat" in cols  # 10 unique <= default 50
    # relax the null cap -> sparse retained
    cols2 = tier0._discover_model_feature_cols(df, exclude, min_non_null_frac=0.1)
    assert "sparse" in cols2
    # tighten the cardinality cap -> high-cardinality categorical dropped
    cols3 = tier0._discover_model_feature_cols(df, exclude, max_categorical_cardinality=5)
    assert "hicard_cat" not in cols3


def test_select_champion_respects_configurable_tie_band():
    """The AUC tie band is CONFIG-driven (Tier D): within-band candidates are a
    discrimination tie decided by calibration; outside-band, higher AUC wins."""
    hist = [
        {"algorithm": "A", "auc_roc": 0.700, "calibration_slope_deviation": 0.50},
        {"algorithm": "B", "auc_roc": 0.692, "calibration_slope_deviation": 0.01},
    ]
    orig = tier0.CONFIG.auc_tie_band
    try:
        tier0.CONFIG.auc_tie_band = 0.01  # 0.008 gap is within band -> tie
        assert tier0._select_champion(hist)["algorithm"] == "B"  # best calibration
        tier0.CONFIG.auc_tie_band = 0.001  # 0.008 gap exceeds band -> not a tie
        assert tier0._select_champion(hist)["algorithm"] == "A"  # highest AUC
    finally:
        tier0.CONFIG.auc_tie_band = orig


def test_single_model_override_disables_alternatives():
    """--single-model maps to CONFIG.train_alternatives=False (Tier D memory
    lever: skip the champion-comparison alternative training)."""
    import scripts.run_optum_tier0_test as wrapper

    orig = tier0.CONFIG.train_alternatives
    try:
        wrapper.apply_overrides(
            "initiation_mart", wrapper.OptumTestConfig(train_alternatives=False)
        )
        assert tier0.CONFIG.train_alternatives is False
        wrapper.apply_overrides("initiation_mart", wrapper.OptumTestConfig(train_alternatives=True))
        assert tier0.CONFIG.train_alternatives is True
    finally:
        tier0.CONFIG.train_alternatives = orig
