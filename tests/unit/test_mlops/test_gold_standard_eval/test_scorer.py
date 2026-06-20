import numpy as np

from src.mlops.gold_standard_eval.scorer import (
    METRIC_NAMES,
    _calibration_slope,
    confusion,
    holdout_curve_records,
    roc_points,
    score,
)


def test_score_emits_page_aligned_names_plus_holdout_extras():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    out = score(y, s)
    # The 5 page-trend metrics are always present, each in [0, 1]...
    assert set(METRIC_NAMES) <= set(out)
    assert abs(out["auc_roc"] - 1.0) < 1e-9  # perfectly separable
    assert all(0.0 <= out[m] <= 1.0 for m in METRIC_NAMES)
    # ...plus the holdout scalar extras: PR-AUC + Brier (always) + calibration.
    assert "pr_auc" in out and 0.0 <= out["pr_auc"] <= 1.0
    assert abs(out["pr_auc"] - 1.0) < 1e-9  # perfectly separable -> AP 1.0
    assert "brier_score" in out and 0.0 <= out["brier_score"] <= 1.0
    assert "calibration_slope" in out and isinstance(out["calibration_slope"], float)


def test_calibration_slope_none_for_single_class():
    # A single-class holdout has no defined calibration slope -> None (omitted).
    assert _calibration_slope(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9])) is None


def test_calibration_slope_float_for_two_class():
    y = np.array([0, 1, 0, 1, 1, 0])
    s = np.array([0.3, 0.4, 0.6, 0.7, 0.5, 0.45])  # overlapping -> fit converges
    slope = _calibration_slope(y, s)
    assert isinstance(slope, float)


def test_score_omits_calibration_when_unfittable(monkeypatch):
    # When the slope helper can't fit (returns None), score() omits the key
    # rather than recording a null (the DECIMAL column is NOT NULL).
    import src.mlops.gold_standard_eval.scorer as scorer_mod

    monkeypatch.setattr(scorer_mod, "_calibration_slope", lambda *_a, **_k: None)
    out = score(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    assert "calibration_slope" not in out
    assert "pr_auc" in out and "brier_score" in out  # the always-present extras remain


# ---------------------------------------------------------------------------
# confusion() — exact 2x2 counts at a decision threshold
# ---------------------------------------------------------------------------


def test_confusion_perfectly_separable():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    cm = confusion(y, s, threshold=0.5)
    assert cm == {"tn": 2, "fp": 0, "fn": 0, "tp": 2, "threshold": 0.5}


def test_confusion_respects_threshold():
    y = np.array([0, 1, 1])
    s = np.array([0.2, 0.45, 0.9])
    # threshold 0.5 -> pred [0, 0, 1]: tn=1, fp=0, fn=1, tp=1
    cm = confusion(y, s, threshold=0.5)
    assert (cm["tn"], cm["fp"], cm["fn"], cm["tp"]) == (1, 0, 1, 1)
    # counts sum to n and reconstruct accuracy
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 3


def test_confusion_handles_single_predicted_class():
    # All scores below threshold -> everything predicted negative.
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    cm = confusion(y, s, threshold=0.5)
    assert (cm["tn"], cm["fp"], cm["fn"], cm["tp"]) == (2, 0, 2, 0)


# ---------------------------------------------------------------------------
# roc_points() — bounded, monotone curve points + auc
# ---------------------------------------------------------------------------


def test_roc_points_perfectly_separable_corners_and_auc():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    out = roc_points(y, s)
    assert abs(out["auc"] - 1.0) < 1e-9
    pts = out["points"]
    assert all(0.0 <= p["fpr"] <= 1.0 and 0.0 <= p["tpr"] <= 1.0 for p in pts)
    # thresholds are JSON-safe (no inf) and clamped to [0, 1]
    assert all(0.0 <= p["threshold"] <= 1.0 for p in pts)
    # fpr is non-decreasing and the curve spans the corners
    fprs = [p["fpr"] for p in pts]
    assert fprs == sorted(fprs)
    assert (pts[0]["fpr"], pts[0]["tpr"]) == (0.0, 0.0)
    assert (pts[-1]["fpr"], pts[-1]["tpr"]) == (1.0, 1.0)


def test_roc_points_downsamples_large_inputs():
    y = np.array([0] * 500 + [1] * 500)
    s = np.concatenate([np.linspace(0.0, 0.6, 500), np.linspace(0.4, 1.0, 500)])
    out = roc_points(y, s, max_points=50)
    assert len(out["points"]) <= 50
    # corners preserved even after downsampling
    assert out["points"][0]["fpr"] == 0.0
    assert out["points"][-1]["fpr"] == 1.0


def test_roc_points_single_class_returns_empty_not_crash():
    y = np.array([1, 1, 1])
    s = np.array([0.2, 0.5, 0.9])
    out = roc_points(y, s)
    assert out["points"] == []
    assert out["auc"] == 0.0


# ---------------------------------------------------------------------------
# holdout_curve_records() — packages confusion + ROC for the recorder
# ---------------------------------------------------------------------------


def test_holdout_curve_records_packages_confusion_and_roc():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    recs = holdout_curve_records(y, s)
    assert [r[0] for r in recs] == ["confusion_matrix", "roc_curve"]
    _, acc, cm = recs[0]
    assert acc == 1.0  # perfectly separable
    assert cm["tp"] == 2 and cm["tn"] == 2
    _, auc, roc = recs[1]
    assert abs(auc - 1.0) < 1e-9
    assert roc["points"][0]["fpr"] == 0.0


def test_holdout_curve_records_single_class_confusion_only():
    # ROC is undefined for a single-class holdout -> confusion matrix only.
    y = np.array([1, 1, 1])
    s = np.array([0.2, 0.6, 0.9])
    recs = holdout_curve_records(y, s)
    assert [r[0] for r in recs] == ["confusion_matrix"]
