"""End-to-end integration tests for ADAPTIVE_CRITERIA (v3 — Option C).

These tests run ``scripts/run_tier0_test.py`` against the synthetic
generator with the runner's deterministic ``seed=42`` configuration. Each
test takes ~3-5 minutes; the ``@pytest.mark.integration`` decorator
keeps them out of fast unit lanes.

The runner emits a structured JSON artifact when ``TIER0_E2E_JSON_OUT``
is set (see end of ``run_pipeline``). These tests parse that artifact;
production CLI invocations are unaffected.

Per ``.claude/plans/adaptive_success_criteria/06-tests-integration.md``
v3 patch: precision/F1 assertions are dropped (those gates are removed
in Option C), MCC / NB / calibration / ``_adaptive_p_t`` assertions are
added.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "run_tier0_test.py"
# Use the active interpreter so the subprocess inherits whatever venv /
# pip layout the test runner is using. Hardcoding ``.venv/bin/python``
# breaks CI where the environment may use a global pip install.
PYTHON = sys.executable


def _run_tier0(env_overrides: Dict[str, str], regime: str) -> Dict[str, Any]:
    """Run ``scripts/run_tier0_test.py`` with a clean cache and return the
    parsed JSON artifact. Fails the test on a non-zero runner exit code.
    """
    cache = REPO / "scripts" / "tier0_output_cache" / "latest.pkl"
    if cache.exists():
        cache.unlink()
    artifact_dir = Path("/tmp/tier0_e2e")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{regime}.json"
    if artifact_path.exists():
        artifact_path.unlink()

    env = os.environ.copy()
    env.update(env_overrides)
    # #594/#556: this e2e has NO live Feast store. Post #556 the freshness check
    # FAILS CLOSED when Feast is unavailable → all features read stale → the
    # registrar QC gate hard-blocks training → empty validation_metrics →
    # "roc_auc missing". ALLOW_STALE_FEAST=1 is the #556 escape hatch for these
    # intentional no-Feast environments. (#594 set this in the 3 sibling synthetic
    # e2e helpers but missed this one — #617.)
    env["ALLOW_STALE_FEAST"] = "1"
    env["TIER0_E2E_JSON_OUT"] = str(artifact_path)

    completed = subprocess.run(
        [PYTHON, str(RUNNER), "--regime", regime, "--no-save"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"runner failed with exit code {completed.returncode}\n"
            f"stderr (tail):\n{completed.stderr[-2000:]}"
        )
    if not artifact_path.exists():
        pytest.fail(
            f"runner did not write the JSON artifact at {artifact_path}\n"
            f"stdout (tail):\n{completed.stdout[-2000:]}"
        )
    return json.loads(artifact_path.read_text())


# ---------------------------------------------------------------------------
# Apr-26 baseline reproduction (flag OFF) — non-negotiable snapshot
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_flag_off_reproduces_apr26_baseline_within_tolerance() -> None:
    """``ADAPTIVE_CRITERIA=false`` ⇒ default-regime run reproduces the
    post-PR-#29 baseline at ``docs/results/tier0_remediation_baseline_20260426.md``
    (see "Post-PR-#29 rebaseline (2026-05-01)" section) within
    deterministic tolerance.

    Rebaselined 2026-05-01: PR #29's generator/evaluator changes shifted
    default-regime val_auc from 0.6942 → 0.5585 (deterministic at
    seed=42; confirmed across two seeded runs). Doc + assertions updated
    atomically per the original docstring contract below.

    Rebaselined 2026-06-02 (#617): #594/#604 disabled the synthetic-fixture
    Layer-3 FDR over-drop, so the default-regime model now RETAINS
    days_on_therapy / prior_treatments etc. → val_auc 0.5585 → 0.6467
    (deterministic at seed=42; roc_auc reproduced exactly across multiple
    seeded CI runs). This was masked for weeks: the #556 Feast fail-closed
    gate halted the pipeline before the evaluator (KeyError 'roc_auc') until
    PR #625 added ALLOW_STALE_FEAST here. Doc + assertions updated atomically.

    Rebaselined 2026-06-06: the Tier-0 leakage over-drop remediation
    (RC1 732059a2 + Fix4 5f7c589e in leakage_detector.py; RC3a 8f0eb68b in
    leakage_remediation.py — PRs #729/#731/#733) stops the leakage detector
    from FALSE-dropping legitimate sparse pre-index clinical predictors on
    the rare-event synthetic cohort. RC1 adds a rare-event guard to
    check_zero_variance_within_class (skip binary/near-binary features that
    are constant in the tiny positive class); Fix4 demotes the cardinality>2
    rare-event branch HIGH→MODERATE (review, not auto-drop); RC3a honors
    declared-safe manifest immunity in the remediation re-check. Net effect
    on the flag-OFF default regime: days_on_therapy / prior_treatments /
    hcp_visits are now RETAINED (they are the top-3 importance features) →
    val roc_auc 0.6467 → 0.65325 (deterministic at seed=42, BIT-IDENTICAL
    across two seeded runs verified locally). These commits were ABSENT at
    the last GREEN nightly (0377c208, 2026-06-04) and PRESENT at the first
    RED nightly (faf3c346, 2026-06-06); the green baseline encoded the
    over-drop bug. This is NOT a flag-off-inertness violation: leakage
    remediation applies to ALL runs; the ADAPTIVE_CRITERIA flag governs
    SUCCESS-CRITERIA selection (the gates), not leakage detection. Doc +
    assertions updated atomically per the contract below; the secondary
    metrics (pr_auc/accuracy/recall/test_*) shifted in lockstep and are
    re-pinned to the same two-run measured values.

    Rebaselined 2026-06-12 (#773 W1): two stacked, deliberate shifts.
    (1) PR #761 (67be1cbf) + PR #760 (5a9e3e5b), merged 2026-06-06,
    changed the training path: LR solver l2/None saga→lbfgs, severe/extreme
    non_tree resampling SMOTE→class_weight (the default regime's ~15%
    positive share is in the severe band, so its SMOTE step disappeared),
    and param-less QC remediation drops now apply. First red nightly
    2026-06-07 (run 27087062518) observed val roc_auc 0.6444 vs the 0.6532
    pin. (2) That 0.6444 was itself measured in a DEGRADED environment:
    the harness one-hot names (age_group_<50 / age_group_>65) crash
    XGBoost at fit time (#773 W2), so every Step-5b XGBoost alternative
    had been silently dead and the LR primary won by default. With the W2
    fix family in place (PR #913 data_preparer + the harness sanitize in
    this PR), the XGBoost alternative trains and WINS the Step-5b champion
    comparison, changing the whole operating profile (e.g. recall 0.80 →
    0.378 at a 0.602 validation-frozen threshold, val roc_auc → 0.6366).
    Values below are pinned from the faithful slow-tests dispatch run
    27443822891 (2026-06-12, branch fix/issue-773-pr4-baseline-repins);
    the follow-up green dispatch on the same branch is the second seeded
    run confirming determinism. Tolerance widths unchanged.

    Tolerances (S3 fix):
      - AUC and PR-AUC: ±0.005 (deterministic at seed=42 modulo
        sklearn-version drift).
      - Precision / recall / F1: ±0.02 (model-direct; v1's ±0.05 was
        over-permissive and would mask real regressions).
      - ``success_criteria_met``: exact (False per Apr-26 baseline).

    If a future sklearn upgrade pushes a metric outside its tolerance,
    do NOT widen the tolerance silently — confirm the new value
    reproduces deterministically across two seeded runs, then update
    the doc + the assertion in the same commit.
    """
    out = _run_tier0({"ADAPTIVE_CRITERIA": "false"}, regime="default")

    assert out["regime"] == "default"
    assert out["criteria_source"] == "fixed"

    # Validation metrics — rebaselined 2026-06-06: Tier-0 leakage over-drop
    # remediation (RC1 732059a2 / Fix4 5f7c589e / RC3a 8f0eb68b, PRs
    # #729/#731/#733) now RETAINS the genuine sparse pre-index predictors
    # (days_on_therapy / prior_treatments / hcp_visits) on the rare-event
    # synthetic cohort. See the "Rebaselined 2026-06-06" docstring section.
    # Values are the bit-identical two-seeded-run measurement (seed=42).
    val = out["validation_metrics"]
    test = out["test_metrics"]
    # #773: emit the FULL realized metric surface in every assertion message
    # (same rationale as the #633 _diag in the clean-v3 test below) so one
    # failing pin reveals every faithful CI value in a single nightly run
    # instead of one value per run.
    _diag = (
        f"FAITHFUL flag-off default metrics — "
        f"validation_metrics={val} | test_metrics={test} | "
        f"model_usefulness={out.get('model_usefulness')}"
    )
    # Re-pinned 2026-06-12 (#773 W1): #761/#760 training-path changes + the
    # W2 fix restoring Step-5b XGBoost alternates (which now WIN the champion
    # comparison). See the "Rebaselined 2026-06-12" docstring section.
    # Values measured on slow-tests dispatch run 27443822891.
    assert val["roc_auc"] == pytest.approx(0.6366, abs=0.005), _diag
    assert val["pr_auc"] == pytest.approx(0.2735, abs=0.005), _diag
    assert val["accuracy"] == pytest.approx(0.8000, abs=0.02), _diag
    assert val["precision"] == pytest.approx(0.3469, abs=0.02), _diag
    assert val["recall"] == pytest.approx(0.3778, abs=0.02), _diag
    assert val["f1_score"] == pytest.approx(0.3617, abs=0.02), _diag

    # Test metrics — re-pinned 2026-06-12 (#773 W1, same dispatch run).
    assert test["roc_auc"] == pytest.approx(0.7044, abs=0.005), _diag
    assert test["accuracy"] == pytest.approx(0.7733, abs=0.02), _diag
    assert test["precision"] == pytest.approx(0.3043, abs=0.02), _diag
    assert test["recall"] == pytest.approx(0.4242, abs=0.02), _diag
    assert test["f1_score"] == pytest.approx(0.3544, abs=0.02), _diag

    # Apr-26 verdict line 18: Step 7 BLOCKED, success_criteria_met False.
    assert out["success_criteria_met"] is False

    # Adaptive-only keys absent under fixed mode (snapshot guarantee).
    sc_keys = set(out["success_criteria"].keys())
    assert "maximum_calibration_error" not in sc_keys
    assert "maximum_train_val_delta" not in sc_keys
    assert "minimum_mcc" not in sc_keys
    assert "minimum_net_benefit_at_p_t" not in sc_keys
    assert "maximum_calibration_slope_deviation" not in sc_keys
    assert "maximum_calibration_intercept_magnitude" not in sc_keys
    # Legacy fixed gates ARE present (Apr-26 reproducibility).
    assert out["success_criteria"]["minimum_auc"] == 0.75
    assert out["success_criteria"]["minimum_precision"] == 0.70
    assert out["success_criteria"]["minimum_recall"] == 0.65
    assert out["success_criteria"]["minimum_f1"] == 0.70


# ---------------------------------------------------------------------------
# Three-regime adaptive E2E (flag ON) — v3 (Option C) assertions
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_clean_regime_with_adaptive_flag_on_v3() -> None:
    """Clean regime under v3 (Option C): MCC / NB / calibration gates fire,
    precision/F1 are dropped entirely, and the deployer-success contract
    is unblocked at path-D values.

    The plan's headline acceptance criterion: v3 should produce
    ``success_criteria_met == True`` at the path-D clean regime where
    v2 (precision/F1 floor) blocked.
    """
    out = _run_tier0({"ADAPTIVE_CRITERIA": "true"}, regime="clean")

    assert out["criteria_source"] == "adaptive"
    sc = out["success_criteria"]
    # v3 worked-example table row 1 (clean, N=900, prev=0.50).
    assert sc["minimum_auc"] == pytest.approx(0.75, abs=0.01)
    assert sc["minimum_recall"] == pytest.approx(0.65, abs=0.01)
    assert sc["minimum_lift_over_baseline"] == pytest.approx(0.10, abs=1e-6)
    assert sc["minimum_net_benefit_at_p_t"] == pytest.approx(0.0, abs=1e-6)
    assert sc["minimum_mcc"] == pytest.approx(0.45, abs=1e-6)
    # Issue #866: the overfit/calibration caps are scaled at evaluation time
    # to the sampling-noise floor of the materialized splits (clean regime:
    # N=4000 → test split ≈ 600 → slope cap ≈ 0.15·sqrt(1000/600) ≈ 0.19).
    # The v3 floors are hard lower bounds; the upper bounds below allow the
    # noise-scaled value for these split sizes with headroom for eligibility
    # filtering. Exact-value pins live in the unit suites where split sizes
    # are controlled (test_adaptive_criteria.py::TestEvalSplitScaledCaps).
    assert 0.15 <= sc["maximum_calibration_slope_deviation"] <= 0.25
    assert 0.30 <= sc["maximum_calibration_intercept_magnitude"] <= 0.50
    # ECE: floor 0.05 at N >= 1000; the 10-bin noise floor at n_test ≈ 600
    # raises it to ≈ 0.076 (#866).
    assert 0.05 <= sc["maximum_calibration_error"] <= 0.10
    # train_val_delta: floor stays the strict 0.03 fpr tier; the SE term at
    # n_val ≈ 800 raises it to ≈ 0.042 (#866).
    assert 0.03 <= sc["maximum_train_val_delta"] <= 0.06
    # v3: precision and F1 are DROPPED entirely.
    assert "minimum_precision" not in sc
    assert "minimum_f1" not in sc
    # No skipped criteria for clean.
    # (The artifact filters underscore-prefix audit fields out of
    # ``success_criteria`` for serialization — read the audit list from
    # ``success_criteria_results`` instead, where skipped names appear
    # with ``met=None``.)

    # Per-criterion outcomes at path-D values: deployer-success contract.
    res = out["success_criteria_results"]
    # #633: emit the FAITHFUL (AVX512 CI) per-gate measured values so a
    # failing run still reports the realized metrics in the assertion
    # message (local AVX2 lies about these; only Job B is decisive).
    tm = out.get("test_metrics", {})
    trm = out.get("train_metrics", {})
    vm = out.get("validation_metrics", {})
    _diag = (
        "FAITHFUL clean-v3 metrics — "
        f"results={res} | "
        f"train_val_auc_delta={tm.get('train_val_auc_delta')} "
        f"(train_roc_auc={trm.get('roc_auc')}, val_roc_auc={vm.get('roc_auc')}) | "
        f"calibration_slope={tm.get('calibration_slope')} "
        f"slope_deviation={tm.get('calibration_slope_deviation')} "
        f"intercept_magnitude={tm.get('calibration_intercept_magnitude')} | "
        f"calibrated_ece={tm.get('calibrated_ece')} | "
        f"mcc={tm.get('mcc')} | recall={tm.get('recall')} | "
        f"deployed_model_is_calibrated={tm.get('deployed_model_is_calibrated')} | "
        f"test_roc_auc={tm.get('roc_auc')} | n_samples={out.get('n_samples')}"
    )
    assert res["minimum_auc"] is True, _diag
    assert res["minimum_recall"] is True, _diag
    assert res["minimum_mcc"] is True, _diag  # path-D MCC ≥ 0.45
    # NB at p_t=0.30 should be > 0 if model is calibrated and
    # discriminates better than treat-all.
    assert res["minimum_net_benefit_at_p_t"] in (True, None), _diag
    assert res["maximum_calibration_error"] is True, _diag
    assert res["maximum_train_val_delta"] is True, _diag
    # v3 (Option C) acceptance criterion: clean unblocks deployer.
    assert out["success_criteria_met"] is True, (
        "v3 (Option C) unblocks clean's deployer-success contract: AUC + "
        "recall + lift + NB > 0 at p_t=0.30 + MCC ≥ 0.45 + calibration in "
        "[-0.30, 0.30] / [0.85, 1.15] all pass at path-D values."
    )


@pytest.mark.integration
def test_default_regime_with_adaptive_flag_on_skips_auc() -> None:
    """Default regime under v3: ``minimum_auc`` skipped (Codex 2026-04-30
    correction); other v3 gates fire normally. Pipeline does NOT halt on
    the skip.
    """
    out = _run_tier0({"ADAPTIVE_CRITERIA": "true"}, regime="default")

    assert out["criteria_source"] == "adaptive"
    sc = out["success_criteria"]
    # v3: AUC is absent from the dict entirely.
    assert "minimum_auc" not in sc
    # Other v3 gates fire (default p_t=0.20, MCC=0.35).
    assert sc["minimum_recall"] == pytest.approx(0.65, abs=0.01)
    assert sc["minimum_mcc"] == pytest.approx(0.35, abs=1e-6)
    assert sc["minimum_net_benefit_at_p_t"] == pytest.approx(0.0, abs=1e-6)
    # Issue #866: caps are noise-scaled to the materialized splits (default
    # regime N=900 → test split ≈ 135 → slope cap ≈ 0.15·sqrt(1000/135) ≈
    # 0.41). Floors are hard lower bounds; see TestEvalSplitScaledCaps for
    # the exact-value pins.
    assert 0.15 <= sc["maximum_calibration_slope_deviation"] <= 0.55
    assert 0.30 <= sc["maximum_calibration_intercept_magnitude"] <= 1.10
    # v3 drops precision/F1.
    assert "minimum_precision" not in sc
    assert "minimum_f1" not in sc
    # Min lift fires at N=900, p=0.30 (v2 S1 fix).
    assert sc["minimum_lift_over_baseline"] == pytest.approx(0.10, abs=1e-6)

    # Per-criterion: AUC recorded as met=None via post-loop pass.
    res = out["success_criteria_results"]
    assert res["minimum_auc"] is None
    # MCC / NB / calibration must be evaluated (True / False / None).
    assert "minimum_mcc" in res
    assert "minimum_net_benefit_at_p_t" in res


@pytest.mark.integration
def test_adverse_regime_with_adaptive_flag_on_v3() -> None:
    """Adverse regime under v3: only lift is skipped (precision/F1 are
    dropped, not skipped). Calibration gates may record met=None due to
    NaN guard at low n_pos. Pipeline must NOT halt on the multiple skips.
    """
    out = _run_tier0({"ADAPTIVE_CRITERIA": "true"}, regime="adverse")

    assert out["criteria_source"] == "adaptive"
    sc = out["success_criteria"]
    assert sc["minimum_auc"] == pytest.approx(0.70, abs=0.01)
    assert sc["minimum_recall"] == pytest.approx(0.50, abs=0.01)
    # v3 adverse-regime gates fire.
    assert sc["minimum_net_benefit_at_p_t"] == pytest.approx(0.0, abs=1e-6)
    assert sc["minimum_mcc"] == pytest.approx(0.20, abs=1e-6)
    # Issue #866: caps are noise-scaled to the materialized splits (adverse
    # regime N=1500 → test split ≈ 225 → slope cap ≈ 0.15·sqrt(1000/225) ≈
    # 0.32). Floors are hard lower bounds; see TestEvalSplitScaledCaps for
    # the exact-value pins.
    assert 0.15 <= sc["maximum_calibration_slope_deviation"] <= 0.45
    assert 0.30 <= sc["maximum_calibration_intercept_magnitude"] <= 0.90
    # v3 drops precision/F1 — neither in success_criteria nor in skipped.
    assert "minimum_precision" not in sc
    assert "minimum_f1" not in sc
    # Lift skipped at n_pos=18 (2*SE ≈ 0.236 > 0.10).
    assert "minimum_lift_over_baseline" not in sc
    # ECE always fires; floor 0.05 at N=1500, raised by the 10-bin noise
    # floor at n_test ≈ 225 to ≈ 0.127 (#866).
    assert 0.05 <= sc["maximum_calibration_error"] <= 0.16

    # Pipeline did not halt — skipped names recorded as met=None.
    res = out["success_criteria_results"]
    # Lift soft-skipped via _adaptive_skipped post-loop pass.
    assert res["minimum_lift_over_baseline"] is None
    # Calibration deviations may be None (NaN at n_pos=18) or True/False.
    assert res["maximum_calibration_slope_deviation"] in (True, False, None)
    assert res["maximum_calibration_intercept_magnitude"] in (True, False, None)
    # AUC criterion was evaluated.
    assert res["minimum_auc"] in (True, False)


@pytest.mark.integration
@pytest.mark.parametrize("regime", ["default", "clean", "adverse"])
def test_flag_off_does_not_emit_adaptive_only_keys_for_any_regime(regime: str) -> None:
    """Triple-check: flag OFF + any regime ⇒ no adaptive-only keys appear.
    The Apr-26 snapshot must reproduce regardless of upstream regime label.
    """
    out = _run_tier0({"ADAPTIVE_CRITERIA": "false"}, regime=regime)
    assert out["criteria_source"] == "fixed"
    sc_keys = set(out["success_criteria"].keys())
    for v3_only_key in (
        "minimum_mcc",
        "minimum_net_benefit_at_p_t",
        "maximum_calibration_slope_deviation",
        "maximum_calibration_intercept_magnitude",
        "maximum_calibration_error",
        "maximum_train_val_delta",
    ):
        assert v3_only_key not in sc_keys, f"{regime}: leaked {v3_only_key}"
    # Fixed thresholds match the as-shipped values.
    sc = out["success_criteria"]
    assert sc["minimum_auc"] == 0.75
    assert sc["minimum_precision"] == 0.70
    assert sc["minimum_recall"] == 0.65
    assert sc["minimum_f1"] == 0.70
    assert sc["minimum_lift_over_baseline"] == 0.10
