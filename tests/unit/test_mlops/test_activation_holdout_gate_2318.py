"""#2318 Lane 5 — holdout harness + non-inferiority acceptance rule.

The rule (OD-2, owner-accepted 2026-09-30), on the SAME goldstd holdout rows:
  * DeLong paired AUC non-inferiority, margin 0.010, one-sided alpha 0.05;
  * Brier(candidate) - Brier(served) upper bound < 0.005 (paired bootstrap);
  * candidate calibration slope in [0.8, 1.25];
  * at least 100 rows per class.
It is a deterministic acceptance rule, not a confirmatory trial.

DeLong correctness is pinned three independent ways:
  1. a hand-computed 2x2 fixture;
  2. a brute-force O(m*n) pairwise structural-component implementation written
     here (no ranks, no shared code with the module);
  3. reference values from R pROC 1.19.1 on its own ``aSAH`` dataset (below).
"""

from __future__ import annotations

import asyncio
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from src.mlops.activation.holdout_gate import (
    GATE_KIND,
    GateConfig,
    delong_paired,
    evaluate_gate,
    load_holdout_snapshot,
    paired_bootstrap,
    require_same_raw_contract,
    rows_sha256,
    score_bundle,
)


def _scores(seed=0, n=2000, noise=0.0):
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, n)
    base = y + r.normal(0, 1.0, n)
    return y, 1 / (1 + np.exp(-base)), 1 / (1 + np.exp(-(base + r.normal(0, noise, n))))


# ---------------------------------------------------------------------------
# DeLong: three independent references
# ---------------------------------------------------------------------------


def test_delong_matches_sklearn_auc_and_a_hand_computed_variance():
    # Structural components by hand for y = [1,1,0,0], a = [.9,.4,.6,.1]:
    #   positives .9, .4 -> V10 = share of negatives each beats = [1, .5]
    #   negatives .6, .1 -> V01 = share of positives above each = [.5, 1]
    #   AUC = mean(V10) = 0.75
    #   var = var(V10)/m + var(V01)/n = 0.125/2 + 0.125/2 = 0.125 (ddof=1)
    y = np.array([1, 1, 0, 0])
    a = np.array([0.9, 0.4, 0.6, 0.1])
    aucs, cov = delong_paired(y, np.vstack([a, a]))
    assert aucs[0] == pytest.approx(0.75)
    assert cov[0, 0] == pytest.approx(0.125)
    y2, s1, s2 = _scores(noise=0.5)
    aucs, cov = delong_paired(y2, np.vstack([s1, s2]))
    assert aucs[0] == pytest.approx(roc_auc_score(y2, s1), abs=1e-12)
    assert aucs[1] == pytest.approx(roc_auc_score(y2, s2), abs=1e-12)


def _brute_force_delong(y, preds):
    """Independent O(m*n) DeLong: psi(x, z) = 1 if x > z, 1/2 if tied, else 0."""
    pos = preds[:, y == 1]
    neg = preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    v10 = np.empty((k, m))
    v01 = np.empty((k, n))
    for r in range(k):
        psi = (pos[r][:, None] > neg[r][None, :]) + 0.5 * (pos[r][:, None] == neg[r][None, :])
        v10[r] = psi.mean(axis=1)
        v01[r] = psi.mean(axis=0)
    aucs = v10.mean(axis=1)
    s10 = np.cov(v10)
    s01 = np.cov(v01)
    return aucs, s10 / m + s01 / n


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_delong_equals_brute_force_pairwise_components_with_heavy_ties(seed):
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, 157)
    # Rounded scores force many ties within and across classes.
    a = np.round(y * 0.6 + r.normal(0, 1, 157), 1)
    b = np.round(0.7 * a + r.normal(0, 0.8, 157), 1)
    c = np.round(r.normal(0, 1, 157), 0)
    preds = np.vstack([a, b, c])
    aucs, cov = delong_paired(y, preds)
    ref_aucs, ref_cov = _brute_force_delong(y, preds)
    np.testing.assert_allclose(aucs, ref_aucs, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cov, ref_cov, rtol=1e-10, atol=1e-15)


def test_ties_use_midranks():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    aucs, _ = delong_paired(y, np.vstack([s, s]))
    assert aucs[0] == pytest.approx(0.5)


# R pROC reference. Dataset: pROC::aSAH (Turck et al. 2010, 113 patients), shipped with
# the pROC package; y = outcome == "Poor" (41 positives). Values produced 2026-09-30 by
#   docker run rocker/r-ver:4.4.1 Rscript (pROC 1.19.1 from CRAN)
#   r1 <- roc(y, aSAH$s100b, levels=c(0,1), direction="<")
#   r2 <- roc(y, as.numeric(aSAH$wfns), levels=c(0,1), direction="<")
#   r3 <- roc(y, aSAH$ndka, levels=c(0,1), direction="<")
#   auc(), var(method="delong"), cov(method="delong"), roc.test(method="delong", paired=TRUE)
# printed with format(digits=17). wfns is a 5-level ordinal score, so it exercises ties.
# (The roc.test(s100b, wfns) result, Z = -2.209 / p = 0.02718, is also the example in
# pROC's own documentation.)
_ASAH_Y = [
    0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
    0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0,
    0,
]  # fmt: skip
_ASAH_S100B = [
    0.13, 0.14, 0.1, 0.04, 0.13, 0.1, 0.47, 0.16, 0.18, 0.1, 0.12, 0.1, 0.44, 0.71,
    0.04, 0.08, 0.49, 0.04, 0.07, 0.33, 0.09, 0.09, 0.07, 0.11, 0.07, 0.17, 0.07, 0.11,
    0.13, 0.19, 0.05, 0.16, 0.41, 0.14, 0.34, 0.35, 0.48, 0.09, 0.96, 0.25, 0.5, 0.46,
    0.16, 0.07, 0.43, 0.45, 0.11, 0.08, 0.09, 0.86, 0.52, 0.08, 0.06, 0.13, 2.07, 0.1,
    0.14, 0.15, 0.07, 0.06, 0.77, 0.05, 0.09, 0.3, 0.03, 0.09, 0.04, 0.23, 0.7, 0.09,
    0.27, 0.71, 0.08, 0.26, 0.08, 0.16, 0.09, 0.13, 0.1, 0.08, 0.11, 0.33, 0.11, 0.28,
    0.07, 0.1, 0.32, 0.22, 0.07, 0.05, 0.24, 0.38, 0.1, 0.15, 0.08, 0.14, 0.1, 0.07,
    0.04, 0.19, 0.56, 0.14, 0.58, 0.32, 0.82, 0.74, 0.15, 0.47, 0.17, 0.44, 0.15, 0.5,
    0.48,
]  # fmt: skip
_ASAH_WFNS = [
    1, 1, 1, 1, 3, 2, 5, 4, 1, 2, 5, 2, 5, 5, 1, 2, 5, 2, 2, 5, 2, 1, 1, 1, 1, 2, 1, 2,
    1, 1, 3, 4, 2, 1, 4, 5, 4, 1, 4, 2, 5, 5, 1, 1, 2, 4, 1, 1, 2, 5, 5, 2, 1, 2, 5, 1,
    2, 2, 2, 1, 5, 1, 1, 2, 5, 2, 1, 4, 5, 2, 4, 4, 1, 4, 1, 2, 1, 4, 2, 1, 2, 3, 2, 4,
    2, 1, 4, 5, 1, 1, 5, 4, 2, 2, 1, 1, 1, 3, 1, 2, 5, 2, 2, 5, 5, 5, 2, 4, 4, 5, 1, 1,
    1,
]  # fmt: skip
_ASAH_NDKA = [
    3.01, 8.54, 8.09, 10.42, 17.4, 12.75, 6.0, 13.2, 15.54, 6.01, 15.96, 17.86, 5.18,
    8.9, 13.41, 20.75, 11.6, 16.11, 32.37, 54.82, 32.41, 49.94, 40.34, 9.47, 6.29,
    12.53, 6.54, 6.3, 80.3, 12.8, 9.8, 9.81, 9.85, 18.21, 5.03, 14.04, 21.93, 8.02,
    7.42, 8.38, 6.59, 9.63, 13.12, 7.96, 14.34, 41.43, 7.63, 7.06, 12.59, 13.56, 3.87,
    9.44, 7.66, 12.98, 419.19, 27.19, 22.27, 9.95, 21.22, 11.73, 10.4, 58.83, 6.39,
    11.09, 12.22, 13.67, 17.21, 22.63, 12.9, 9.7, 21.57, 8.23, 72.57, 15.54, 10.51,
    10.6, 14.57, 5.19, 9.63, 4.61, 21.48, 17.3, 12.71, 9.44, 11.07, 19.46, 10.83, 5.37,
    11.97, 7.75, 9.83, 10.55, 28.49, 8.53, 12.57, 12.9, 46.83, 11.68, 12.67, 9.01, 9.57,
    34.06, 11.72, 14.26, 47.61, 11.67, 24.58, 10.33, 13.87, 15.89, 22.43, 6.79, 13.45,
]  # fmt: skip
_PROC_AUC = (0.73136856368563685, 0.82367886178861793, 0.61195799457994582)
_PROC_VAR = (0.0026686824571724378, 0.0014699147088236264, 0.0031908105493913021)
_PROC_COV_12 = 0.0011961556737675448
_PROC_COV_13 = -0.00075616493805657884
_PROC_Z_12, _PROC_P_12 = -2.2089835914409077, 0.02717578222918815
_PROC_Z_13, _PROC_P_13 = 1.3907700257355771, 0.16429517522305448


def test_delong_reproduces_r_proc_on_asah():
    y = np.array(_ASAH_Y)
    assert (len(y), int(y.sum())) == (113, 41)
    preds = np.vstack([_ASAH_S100B, _ASAH_WFNS, _ASAH_NDKA]).astype(float)
    aucs, cov = delong_paired(y, preds)
    np.testing.assert_allclose(aucs, _PROC_AUC, rtol=1e-12)
    np.testing.assert_allclose(np.diag(cov), _PROC_VAR, rtol=1e-10)
    assert cov[0, 1] == pytest.approx(_PROC_COV_12, rel=1e-10)
    assert cov[0, 2] == pytest.approx(_PROC_COV_13, rel=1e-10)
    for j, z_ref, p_ref in ((1, _PROC_Z_12, _PROC_P_12), (2, _PROC_Z_13, _PROC_P_13)):
        var_d = cov[0, 0] + cov[j, j] - 2 * cov[0, j]
        z = (aucs[0] - aucs[j]) / math.sqrt(var_d)
        assert z == pytest.approx(z_ref, rel=1e-10)
        assert 2 * norm.sf(abs(z)) == pytest.approx(p_ref, rel=1e-9)


def test_delong_refuses_bad_input():
    with pytest.raises(ValueError, match="both classes"):
        delong_paired(np.array([1, 1, 1]), np.vstack([[0.1, 0.2, 0.3]]))
    with pytest.raises(ValueError, match="0/1"):
        delong_paired(np.array([1, 2, 0, 0]), np.vstack([[0.1, 0.2, 0.3, 0.4]]))
    with pytest.raises(ValueError, match="shape"):
        delong_paired(np.array([1, 0, 1, 0]), np.vstack([[0.1, 0.2, 0.3]]))


# ---------------------------------------------------------------------------
# evaluate_gate — the acceptance rule
# ---------------------------------------------------------------------------


def test_identical_models_pass_with_zero_variance_handled():
    y, a, _ = _scores()
    rep = evaluate_gate(y, served=a, candidate=a.copy(), cfg=GateConfig())
    # var(delta) == 0 exactly: the bound is delta itself, never a division by zero
    assert rep["passed"] and rep["auc_delta"] == 0.0 and rep["se_delta"] == 0.0
    assert rep["failed_checks"] == []


def test_a_clearly_worse_candidate_fails_on_auc():
    y, a, _ = _scores()
    worse = 0.5 * a + 0.5 * np.random.default_rng(1).random(len(a))
    rep = evaluate_gate(y, served=a, candidate=worse, cfg=GateConfig())
    assert not rep["passed"] and "auc_noninferiority" in rep["failed_checks"]


def test_miscalibrated_candidate_fails_on_slope():
    y, a, _ = _scores()
    logit = np.log(a / (1 - a))
    sharp = 1 / (1 + np.exp(-3 * logit))  # same ranking, slope ~ 1/3
    rep = evaluate_gate(y, served=a, candidate=sharp, cfg=GateConfig())
    assert rep["auc_delta"] == pytest.approx(0.0, abs=1e-12)
    assert "calibration_slope" in rep["failed_checks"]
    assert not rep["passed"]


def test_brier_worse_candidate_fails_on_brier_only():
    # Same ranking (AUC identical), in-band slope, but a constant +0.12 probability
    # offset: calibration-in-the-large is off, so the Brier score degrades.
    y, a, _ = _scores()
    shifted = np.clip(a + 0.12, 0.0, 1.0)
    rep = evaluate_gate(y, served=a, candidate=shifted, cfg=GateConfig())
    assert rep["brier_delta_upper"] >= GateConfig().brier_margin
    assert "brier_noninferiority" in rep["failed_checks"]
    assert "auc_noninferiority" not in rep["failed_checks"]


@pytest.mark.parametrize("n_pos,n_neg", [(50, 500), (500, 50), (0, 500)])
def test_refuses_a_thin_or_one_class_holdout(n_pos, n_neg):
    y = np.array([1] * n_pos + [0] * n_neg)
    s = np.linspace(0, 1, len(y))
    with pytest.raises(ValueError, match="at least 100"):
        evaluate_gate(y, served=s, candidate=s, cfg=GateConfig())


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1])
def test_refuses_non_finite_or_out_of_range_scores_before_scoring(bad):
    y, a, _ = _scores()
    b = a.copy()
    b[7] = bad
    with pytest.raises(ValueError, match="candidate"):
        evaluate_gate(y, served=a, candidate=b, cfg=GateConfig())
    with pytest.raises(ValueError, match="served"):
        evaluate_gate(y, served=b, candidate=a, cfg=GateConfig())


def test_refuses_misaligned_inputs():
    y, a, _ = _scores()
    with pytest.raises(ValueError, match="same rows"):
        evaluate_gate(y, served=a, candidate=a[:-1], cfg=GateConfig())


def test_exact_zero_one_scores_give_a_finite_json_report():
    y, a, _ = _scores()
    hard = (a > 0.5).astype(float)  # exact 0/1 probabilities
    rep = evaluate_gate(y, served=a, candidate=hard, cfg=GateConfig())
    json.dumps(rep, allow_nan=False)  # jsonb-safe: no NaN / inf anywhere
    assert rep["calibration_slope"] is not None and math.isfinite(rep["calibration_slope"])
    assert not rep["passed"]


def test_unfittable_slope_fails_the_rule(monkeypatch):
    import src.mlops.activation.holdout_gate as hg

    monkeypatch.setattr(hg, "calibration_slope", lambda y, s: None)
    y, a, _ = _scores()
    rep = hg.evaluate_gate(y, served=a, candidate=a.copy(), cfg=GateConfig())
    assert rep["calibration_slope"] is None
    assert "calibration_slope_unfittable" in rep["failed_checks"]
    assert not rep["passed"]


def test_report_is_json_and_deterministic():
    y, a, b = _scores(noise=0.3)
    r1 = evaluate_gate(y, served=a, candidate=b, cfg=GateConfig())
    r2 = evaluate_gate(y, served=a, candidate=b, cfg=GateConfig())
    assert json.dumps(r1, sort_keys=True, allow_nan=False) == json.dumps(
        r2, sort_keys=True, allow_nan=False
    )


def test_report_carries_every_field_the_sql_predicate_reads():
    """Lane 4's ``activation_gate_passes`` treats any missing field as a failure."""
    y, a, b = _scores(noise=0.3)
    snap = {"splits": ["test", "holdout"], "n": len(y), "n_pos": int(y.sum())}
    rep = evaluate_gate(
        y,
        served=a,
        candidate=b,
        cfg=GateConfig(),
        served_bundle_sha256="a" * 64,
        candidate_bundle_sha256="b" * 64,
        snapshot=snap,
        model_name="initiation_kisqali_goldstd_lr_v1",
    )
    for key in (
        "kind",
        "n",
        "n_pos",
        "auc_served",
        "auc_candidate",
        "auc_delta",
        "se_delta",
        "auc_lower_bound",
        "brier_served",
        "brier_candidate",
        "brier_delta_upper",
        "pr_auc_served",
        "pr_auc_candidate",
        "calibration_slope",
        "calibration_intercept",
        "prevalence",
        "candidate_bundle_sha256",
        "served_bundle_sha256",
        "snapshot",
        "bootstrap_auc_delta_p05",
        "failed_checks",
        "passed",
        "config",
    ):
        assert key in rep, key
    assert rep["kind"] == GATE_KIND == "deterministic_acceptance_rule"
    assert rep["config"] == {
        "auc_margin": 0.010,
        "alpha": 0.05,
        "brier_margin": 0.005,
        "slope_band": [0.8, 1.25],
        "bootstrap_b": 2000,
        "seed": 0,
        "min_class_n": 100,
        "min_usable_bootstrap_frac": 0.9,
    }
    assert rep["snapshot"] == snap
    assert rep["candidate_bundle_sha256"] == "b" * 64
    # Non-hcp names carry no pathology fields.
    assert "hcp_pathology_passed" not in rep
    lb = rep["auc_delta"] - norm.ppf(0.95) * rep["se_delta"]
    assert rep["auc_lower_bound"] == pytest.approx(lb, abs=1e-15)


def test_gate_config_constants_are_the_owner_decision():
    cfg = GateConfig()
    assert (cfg.auc_margin, cfg.alpha, cfg.brier_margin) == (0.010, 0.05, 0.005)
    assert cfg.slope_band == (0.8, 1.25) and cfg.min_class_n == 100
    with pytest.raises(Exception):
        cfg.auc_margin = 0.05  # type: ignore[misc]  # frozen


# ---------------------------------------------------------------------------
# hcp_adoption: the #1354 pathology gate rides along for production names
# ---------------------------------------------------------------------------


def test_hcp_name_fails_on_brier_pathology_when_the_generic_rule_passes():
    # A weak model with the right slope but a +1.5 logit offset: Brier is above the
    # base-rate baseline p(1-p). Served and candidate are the same scores, so the generic
    # non-inferiority rule passes; the #1354 pathology gate must still refuse.
    r = np.random.default_rng(3)
    n = 3000
    x = r.normal(0, 1, n)
    y = (r.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    s = 1 / (1 + np.exp(-(x + 1.5)))
    generic = evaluate_gate(y, served=s, candidate=s.copy(), cfg=GateConfig())
    assert generic["failed_checks"] == [] and generic["passed"]
    rep = evaluate_gate(
        y,
        served=s,
        candidate=s.copy(),
        cfg=GateConfig(),
        model_name="hcp_adoption_kisqali_goldstd_lr_v1",
    )
    assert rep["brier_candidate"] >= rep["prevalence"] * (1 - rep["prevalence"])
    assert rep["hcp_pathology_passed"] is False
    assert rep["hcp_pathology_brier_ok"] is False and rep["hcp_pathology_slope_ok"] is True
    assert rep["hcp_pathology_reasons"]
    assert "hcp_pathology" in rep["failed_checks"] and not rep["passed"]


def test_hcp_slope_pathology_predicate_is_the_1354_band():
    """The pathology slope band [0.5, 2.0] strictly contains the generic [0.8, 1.25],
    so it can never fail while the generic slope check passes; pin it on the predicate."""
    from src.mlops.activation.holdout_gate import HCP_PATHOLOGY_SLOPE_RANGE, pathology_gate

    assert HCP_PATHOLOGY_SLOPE_RANGE == (0.5, 2.0)
    ok, reasons = pathology_gate({"calibration_slope": 2.5, "brier_score": 0.1}, 0.4)
    assert not ok and "calibration_slope" in reasons[0]
    ok, _ = pathology_gate({"calibration_slope": 1.0, "brier_score": 0.1}, 0.4)
    assert ok


def test_hcp_name_healthy_candidate_passes_with_pathology_fields():
    y, a, b = _scores(noise=0.1)
    rep = evaluate_gate(
        y, served=a, candidate=b, cfg=GateConfig(), model_name="hcp_adoption_fabhalta_goldstd_lr_v1"
    )
    assert rep["hcp_pathology_passed"] is True
    assert rep["hcp_pathology_slope_ok"] is True and rep["hcp_pathology_brier_ok"] is True
    assert rep["passed"]


def test_promote_script_reuses_the_moved_helpers():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import scripts.promote_hcp_adoption_champions as promo
    import src.mlops.activation.holdout_gate as hg

    assert promo.positive_class_scores is hg.positive_class_scores
    assert promo.calibration_intercept is hg.calibration_intercept
    assert promo.pathology_gate is hg.pathology_gate
    assert promo.SLOPE_RANGE == hg.HCP_PATHOLOGY_SLOPE_RANGE


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_auc_delta_matches_a_per_resample_sklearn_loop():
    """The vectorised weighted-AUC bootstrap equals roc_auc_score per resample."""
    y, a, b = _scores(n=300, noise=0.4)
    cfg = GateConfig(bootstrap_b=40)
    out = paired_bootstrap(y, a, b, cfg)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.integers(0, len(y), size=(cfg.bootstrap_b, len(y)))
    ref_d = []
    ref_b = []
    for row in idx:
        yy = y[row]
        ref_d.append(roc_auc_score(yy, b[row]) - roc_auc_score(yy, a[row]))
        ref_b.append(np.mean((b[row] - yy) ** 2) - np.mean((a[row] - yy) ** 2))
    np.testing.assert_allclose(out["auc_deltas"], ref_d, atol=1e-12)
    np.testing.assert_allclose(out["brier_deltas"], ref_b, atol=1e-12)
    assert out["usable"] == cfg.bootstrap_b and out["skipped_single_class"] == 0


def test_bootstrap_single_class_resamples_are_skipped_and_counted():
    # 2 positives in 12 rows: some of 400 resamples draw no positive at all.
    y = np.array([1, 1] + [0] * 10)
    s = np.linspace(0.05, 0.95, 12)
    out = paired_bootstrap(y, s, s[::-1].copy(), GateConfig(bootstrap_b=400))
    assert out["skipped_single_class"] > 0
    assert out["usable"] + out["skipped_single_class"] == 400
    assert len(out["auc_deltas"]) == out["usable"] == len(out["brier_deltas"])


def test_too_few_usable_bootstrap_resamples_fails_the_rule(monkeypatch):
    import src.mlops.activation.holdout_gate as hg

    real = hg.paired_bootstrap

    def thin(y, served, candidate, cfg):
        out = real(y, served, candidate, cfg)
        keep = int(0.5 * cfg.bootstrap_b)
        return {
            **out,
            "auc_deltas": out["auc_deltas"][:keep],
            "brier_deltas": out["brier_deltas"][:keep],
            "usable": keep,
            "skipped_single_class": cfg.bootstrap_b - keep,
        }

    monkeypatch.setattr(hg, "paired_bootstrap", thin)
    y, a, _ = _scores()
    rep = hg.evaluate_gate(y, served=a, candidate=a.copy(), cfg=GateConfig())
    assert "bootstrap_unusable" in rep["failed_checks"] and not rep["passed"]


# ---------------------------------------------------------------------------
# Planted truth at production n (Kisqali harness: n = 1766, 609 positives)
# ---------------------------------------------------------------------------


# Binormal truth: the latent signal z is N(+MU, SIG) for positives and N(-MU, SIG) for
# negatives. Each model observes w = z + N(0, sd) and outputs its Bayes posterior
# P(y=1 | w), so every planted model is calibrated (slope ~1) and its true AUC is known
# in closed form: AUC(sd) = Phi(2*MU / sqrt(2*(SIG^2 + sd^2))). A candidate "worse by
# d" is the sd that solves AUC(sd) = AUC(sd_served) - d. (A 400k-row Monte Carlo
# reproduces the closed form to 5e-4; PR evidence.)
#
# Noise sd 0.10 is PRODUCTION-FAITHFUL: at n=1766 it gives DeLong r = 0.991 and
# SE(delta) = 0.00125, the least favourable pair measured on the real Kisqali harness
# (plan OD-2: r 0.991-0.999, SE 0.0004-0.0012). The false-refusal rate at equal
# performance depends on SE(delta): it is <= alpha only while
# SE <= margin / (2 * z_0.95) = 0.0030. At sd 0.35 (SE 0.0042) it is ~23%; PR evidence.
_MU, _SIG, _SD_SERVED = 0.9, 1.2, 0.10  # served true AUC 0.8547, like the Kisqali harness


def _true_auc(sd):
    return float(norm.cdf(2 * _MU / math.sqrt(2 * (_SIG**2 + sd**2))))


def _sd_for_auc(auc):
    return math.sqrt((2 * _MU / norm.ppf(auc)) ** 2 / 2 - _SIG**2)


def _planted(seed, sd_served, sd_cand, n=1766, n_pos=609):
    r = np.random.default_rng(seed)
    y = np.concatenate([np.ones(n_pos, int), np.zeros(n - n_pos, int)])
    z = np.concatenate([r.normal(_MU, _SIG, n_pos), r.normal(-_MU, _SIG, n - n_pos)])
    prior = math.log(n_pos / (n - n_pos))

    def model(sd):
        w = z + r.normal(0, sd, n)
        v = _SIG**2 + sd**2
        return 1 / (1 + np.exp(-(2 * _MU * w / v + prior)))

    return y, model(sd_served), model(sd_cand)


def test_planted_equal_model_passes_and_a_0_03_worse_model_fails():
    sd_worse = _sd_for_auc(_true_auc(_SD_SERVED) - 0.03)
    for seed in range(3):
        y, s, c = _planted(seed, _SD_SERVED, _SD_SERVED)
        rep = evaluate_gate(y, served=s, candidate=c, cfg=GateConfig())
        assert rep["passed"], rep["failed_checks"]
        y, s, c = _planted(seed, _SD_SERVED, sd_worse)
        rep = evaluate_gate(y, served=s, candidate=c, cfg=GateConfig())
        assert rep["auc_delta"] < -0.015
        assert "auc_noninferiority" in rep["failed_checks"] and not rep["passed"]


def test_delong_bound_has_nominal_size_at_the_margin():
    """At a true delta of exactly -margin the AUC check must pass ~alpha of the time.

    400 replicates: the pass rate's MC SE is ~0.011, so a correct one-sided 0.05 test
    lands in [0.02, 0.09] with >99% probability. (The PR evidence runs 5,000.)
    """
    cfg = GateConfig()
    sd_margin = _sd_for_auc(_true_auc(_SD_SERVED) - cfg.auc_margin)
    z = norm.ppf(1 - cfg.alpha)
    passes = 0
    reps = 400
    for seed in range(reps):
        y, s, c = _planted(10_000 + seed, _SD_SERVED, sd_margin)
        aucs, cov = delong_paired(y, np.vstack([s, c]))
        se = math.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0))
        passes += (aucs[1] - aucs[0]) - z * se > -cfg.auc_margin
    rate = passes / reps
    assert 0.02 <= rate <= 0.09, rate


# ---------------------------------------------------------------------------
# Task 5.2 — score each artifact through its OWN persisted preprocessor
# ---------------------------------------------------------------------------


def _raw_frame(n=240, seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i:04d}" for i in range(n)],
            "data_split": r.choice(["test", "holdout"], n),
            "disease_severity": r.normal(5, 2, n),
            "academic_hcp": r.integers(0, 2, n),
            "geographic_region": r.choice(["south", "west", "northeast"], n),
        }
    )
    df.loc[3, "disease_severity"] = np.nan
    logit = 0.4 * (df["disease_severity"].fillna(5) - 5) + 0.8 * df["academic_hcp"] - 0.3
    df["treatment_initiated"] = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


_KEEP = ("disease_severity", "academic_hcp", "geographic_region")


def _spec():
    from src.mlops.gold_standard_eval.cohort_spec import CohortSpec

    return CohortSpec(
        name="initiation_kisqali",
        target="initiation_kisqali",
        brand="Kisqali",
        label_column="treatment_initiated",
        grain="patient",
        base_covariates=_KEEP,
    )


def _feature_builder_bundle(df):
    from sklearn.linear_model import LogisticRegression

    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder

    fb = FeatureBuilder(_spec())
    x, y = fb.build_from_frame(df)
    model = LogisticRegression(max_iter=1000).fit(x, y)
    return {"model": model, "preprocessor": fb, "feature_columns": list(fb.feature_columns)}


def _ct_bundle(df):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.frozen import FrozenEstimator
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = ["disease_severity", "academic_hcp"]
    cat = ["geographic_region"]
    ct = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            ),
        ]
    )
    x = ct.fit_transform(df[list(_KEEP)])
    y = df["treatment_initiated"]
    lr = LogisticRegression().fit(x, y)
    model = CalibratedClassifierCV(FrozenEstimator(lr), method="sigmoid").fit(x, y)
    return {
        "bundle_format": "sklearn_ct_v1",
        "model": model,
        "preprocessor": ct,
        "keep_columns": list(_KEEP),
        "numeric_columns": num,
        "feature_columns": [n.split("__", 1)[1] for n in ct.get_feature_names_out()],
        "sklearn_version": "1.6.1",
    }


def test_score_feature_builder_bundle_matches_its_own_encoder_without_fitting():
    df = _raw_frame()
    bundle = _feature_builder_bundle(df)
    fb = bundle["preprocessor"]
    expected = bundle["model"].predict_proba(
        fb.transform(df)[bundle["feature_columns"]].to_numpy(dtype=float)
    )[:, 1]

    def boom(*a, **k):
        raise AssertionError("the harness must never fit a persisted preprocessor")

    fb.build_from_frame = boom  # type: ignore[method-assign]
    bundle["model"].fit = boom
    np.testing.assert_array_equal(score_bundle(bundle, df), expected)


def test_score_ct_bundle_matches_its_own_column_transformer_without_fitting():
    df = _raw_frame()
    bundle = _ct_bundle(df)
    expected = bundle["model"].predict_proba(bundle["preprocessor"].transform(df[list(_KEEP)]))[
        :, 1
    ]

    def boom(*a, **k):
        raise AssertionError("the harness must never fit a persisted preprocessor")

    bundle["preprocessor"].fit = boom
    bundle["preprocessor"].fit_transform = boom
    bundle["model"].fit = boom
    np.testing.assert_array_equal(score_bundle(bundle, df), expected)


def test_score_bundle_refuses_an_unknown_shape():
    with pytest.raises(ValueError, match="bundle"):
        score_bundle({"model": object()}, _raw_frame())


def test_keep_columns_must_match():
    df = _raw_frame()
    served = _feature_builder_bundle(df)
    cand = _ct_bundle(df)
    require_same_raw_contract(served, cand)  # same 3 raw columns: ok
    cand["keep_columns"] = ["disease_severity", "academic_hcp"]
    with pytest.raises(ValueError, match="raw contract differs"):
        require_same_raw_contract(served, cand)


# ---------------------------------------------------------------------------
# Task 5.2 — one snapshot of the holdout rows, hashed
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, rows, calls):
        self._rows = rows
        self._calls = calls
        self._offset = 0
        self._end = None

    def select(self, expr):
        self._calls.append(("select", expr))
        return self

    def eq(self, col, val):
        self._calls.append(("eq", col, val))
        return self

    def in_(self, col, vals):
        self._calls.append(("in_", col, list(vals)))
        return self

    def lt(self, *a):
        return self

    def order(self, col):
        self._calls.append(("order", col))
        return self

    def range(self, start, end):
        self._offset, self._end = start, end
        return self

    async def execute(self):
        class R:
            pass

        r = R()
        r.data = self._rows[self._offset : self._end + 1]
        return r


class _FakeDb:
    def __init__(self, rows):
        self.rows = rows
        self.calls: list = []

    def table(self, name):
        self.calls.append(("table", name))
        return _FakeQuery(self.rows, self.calls)


def _rows(df):
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()}
        for rec in df.to_dict("records")
    ]


def test_load_holdout_snapshot_hashes_one_ordered_snapshot():
    df = _raw_frame()
    df["journey_start_date"] = "2026-09-01"
    db = _FakeDb(_rows(df))
    frame, y, snap = asyncio.run(load_holdout_snapshot(db, _spec()))
    assert ("in_", "data_split", ["test", "holdout"]) in db.calls
    assert ("eq", "is_synthetic", True) in db.calls
    assert list(frame["patient_id"]) == sorted(df["patient_id"])
    np.testing.assert_array_equal(y, frame["treatment_initiated"].astype(int).to_numpy())
    assert snap["splits"] == ["test", "holdout"]
    assert snap["n"] == len(df) and snap["n_pos"] == int(df["treatment_initiated"].sum())
    assert snap["key"] == "patient_id"
    assert len(snap["rows_sha256"]) == 64 and snap["loaded_at"]

    # Same rows again, and the same rows shuffled: same hash.
    _, _, snap2 = asyncio.run(load_holdout_snapshot(_FakeDb(_rows(df)), _spec()))
    shuffled = df.sample(frac=1.0, random_state=7)
    _, _, snap3 = asyncio.run(load_holdout_snapshot(_FakeDb(_rows(shuffled)), _spec()))
    assert snap["rows_sha256"] == snap2["rows_sha256"] == snap3["rows_sha256"]


def test_rows_sha256_changes_with_one_label_or_one_covariate():
    df = _raw_frame()
    base = rows_sha256(df, key="patient_id", label="treatment_initiated", keep_columns=_KEEP)
    flipped = df.copy()
    flipped.loc[10, "treatment_initiated"] = 1 - flipped.loc[10, "treatment_initiated"]
    assert rows_sha256(flipped, "patient_id", "treatment_initiated", _KEEP) != base
    moved = df.copy()
    moved.loc[10, "disease_severity"] += 1e-9
    assert rows_sha256(moved, "patient_id", "treatment_initiated", _KEEP) != base
    region = df.copy()
    region.loc[10, "geographic_region"] = "midwest"
    assert rows_sha256(region, "patient_id", "treatment_initiated", _KEEP) != base
    # Columns outside the contract do not move the hash.
    extra = df.copy()
    extra["journey_start_date"] = "2026-01-01"
    assert rows_sha256(extra, "patient_id", "treatment_initiated", _KEEP) == base


def test_rows_sha256_is_stable_for_categoricals_and_datetimes():
    df = _raw_frame()
    df["when"] = pd.Timestamp("2026-09-01", tz="UTC")
    a = rows_sha256(df, "patient_id", "treatment_initiated", (*_KEEP, "when"))
    cat = df.copy()
    cat["geographic_region"] = cat["geographic_region"].astype("category")
    cat["when"] = cat["when"].dt.tz_convert("America/New_York")  # same instant
    assert rows_sha256(cat, "patient_id", "treatment_initiated", (*_KEEP, "when")) == a


def test_snapshot_refuses_a_duplicated_key():
    df = _raw_frame()
    df.loc[5, "patient_id"] = df.loc[4, "patient_id"]
    with pytest.raises(ValueError, match="unique"):
        asyncio.run(load_holdout_snapshot(_FakeDb(_rows(df)), _spec()))


def test_snapshot_refuses_an_empty_or_unlabeled_holdout():
    with pytest.raises(ValueError, match="no rows"):
        asyncio.run(load_holdout_snapshot(_FakeDb([]), _spec()))
    df = _raw_frame()
    df["treatment_initiated"] = df["treatment_initiated"].astype(float)
    df.loc[2, "treatment_initiated"] = np.nan
    with pytest.raises(ValueError, match="label"):
        asyncio.run(load_holdout_snapshot(_FakeDb(_rows(df)), _spec()))


def test_oos_splits_are_in_lockstep_with_the_eval_that_sets_the_registry_auc():
    from src.mlops.activation.holdout_gate import OOS_EVAL_SPLITS
    from src.mlops.gold_standard_eval import run_initiation_eval, run_persistence_eval

    assert OOS_EVAL_SPLITS == run_persistence_eval._OOS_EVAL_SPLITS
    assert OOS_EVAL_SPLITS == run_initiation_eval._OOS_EVAL_SPLITS
