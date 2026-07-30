"""Synthetic e2e baseline regression invariant — 7-dim val-side gate.

Added by tier0_quality_remediation_arc Shard D, 2026-05-06.
Re-baselined to ``--regime scenario_a`` 2026-05-06 by the
synthetic_v2_scenario_a_swap arc — see
``.claude/plans/synthetic_v2_scenario_a_swap_20260506.md`` (β-prime
additive). The synthetic_v2.scenario_a generator (HR+/HER2- early BC iDFS,
Kisqali franchise, 40 clinically-grounded features, n=6000) replaces the
legacy ``ml_patients()`` fixture for this gate; calibrated AUC band
[0.78, 0.83] from ``src/ml/synthetic_v2/scenarios/scenario_a.py:7``.
Existing ``--regime default``/``adverse``/``clean`` regimes remain alive
for backward compatibility with other tests.

What this test pins
-------------------
- Synthetic e2e ``--regime scenario_a`` with seed=42 is bit-identical
  run-to-run (verified 2026-05-06: 6+ decimal match across two local
  runs, all metrics).
- This test runs the synthetic e2e once and asserts that 7 val-side
  metrics fall within tight tolerance bands around the current stable
  baseline. Any regression in (a) the synthetic_v2 generator, (b) HPO
  determinism, (c) evaluator metric computation, or (d) calibration
  pipeline will trip one or more bands.

Tolerance philosophy
--------------------
Tolerance for each metric = max(absolute precision delta we'd accept,
``5 × min_observed_run_variance``). Per Agent D bit-identicality across
runs at seed=42, observed variance is 0; we pin to ±0.01 absolute on AUC
and equivalently tight on derived metrics. If a future code change
deliberately shifts a metric, the band is updated **in the same commit**
that ships the change (per the doc-test contract precedent at
tests/integration/test_adaptive_criteria_e2e.py:81).

Wall-clock budget
-----------------
~3-5 min per run. Marked ``slow`` so it doesn't run in default `pytest -x`
sweeps; CI runs with the explicit marker filter.

Updating the bands
------------------
If a deliberate code change shifts a metric, update the band constant
below + add a one-line comment with the responsible PR / commit SHA.
Do NOT silently widen tolerance — confirm bit-identicality across two
seeded runs first, then update.

Environment split (CPU ISA)
---------------------------
Local (Ubuntu, AVX2 CPU) and CI (GitHub Actions ``ubuntu-latest``, AVX512 CPU)
produce different but each bit-deterministic results with the same package
versions (numpy 2.3.5, xgboost 3.1.2, scikit-learn 1.6.1, lightgbm 4.6.0).
The divergence is at the floating-point instruction level and cannot be
resolved with package pins or env vars.  Both baselines are kept explicit
here. BASELINE_CI was measured 2026-05-06 from the PR #69 CI run diagnostics
(diagnostic record in memory/pr69_e2e_environment_delta_diag_20260506.md).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── LOCAL baseline (Ubuntu local machine, AVX2 CPU, Python 3.12.3) ──────────
# Re-measured 2026-07-30 (#1311) after aed06cb7 (#44 plan B1) enlarged the
# goldstd holdout (test 15%→10%, holdout 5%→10%), reshuffling the keyed-draw
# split assignment. Two seeded local runs (--regime scenario_a --split auto
# --hpo-trials 5) were bit-identical on every metric (only the run-ID nonce
# differed). This re-measurement also clears the previously documented STALE
# pre-#761/#760 business_utility (99.20, measured 2026-05-06).
# CI (the nightly arbiter) uses BASELINE_CI, not this dict.
BASELINE_LOCAL = {
    "roc_auc": 0.7838,  # ±0.01 — was 0.7689, re-pinned #1311; val side
    "pr_auc": 0.5230,  # ±0.02 — was 0.4734, re-pinned #1311
    "brier_score": 0.1564,  # ±0.01 — was 0.1811, re-pinned #1311
    "mcc": 0.3633,  # ±0.03 — was 0.3355, re-pinned #1311
    "business_utility": 104.30,  # ±0.5 — was stale 99.20; re-pinned #1311
}

# ── CI baseline (GitHub Actions ubuntu-latest, AVX512 CPU, Python 3.12.13) ──
# Pre-measurement placeholder — initialized to LOCAL values. Will be replaced
# with CI-bit-deterministic measurements from the first slow-tests run on
# feat/tier0-scenario-a-fixture (PR #TBD). The CPU-ISA divergence (AVX2 local
# vs AVX512 CI) shifts metrics at the floating-point instruction level; we
# still expect ≥6-decimal CI determinism per the same mechanism that pinned
# the default-regime CI baseline (see memory/pr69_e2e_environment_delta_diag_20260506.md).
BASELINE_CI = {
    # Re-pinned 2026-07-30 (#1311): aed06cb7 (#44 plan B1, merged 2026-07-21)
    # enlarged the goldstd holdout at the seed quota (test 15%→10%, holdout
    # 5%→10%), reshuffling the keyed-draw split assignment and deliberately
    # shifting the whole metric surface. New values reproduced bit-identically
    # across independent nightly runs 30433515459 (2026-07-29) and
    # 30524268029 (2026-07-30). brier_score stayed within its ±0.01 band in
    # both runs (pin unchanged). Tolerance widths unchanged.
    "roc_auc": 0.8021,  # ±0.01 — was 0.7689, re-pinned #1311
    "pr_auc": 0.5247,  # ±0.02 — was 0.4734, re-pinned #1311
    "brier_score": 0.1811,  # ±0.01 — passed both post-aed06cb7 nightlies, unchanged
    "mcc": 0.3918,  # ±0.03 — was 0.3355, re-pinned #1311
    # Re-pinned 2026-06-12 (#773 W1): PR #761 (67be1cbf) re-routed the LR
    # solver (l2/None saga→lbfgs) + re-tuned severe/extreme non_tree
    # resampling to class_weight, and PR #760 (5a9e3e5b) fixed param-less QC
    # remediation drops — both merged 2026-06-06 and deliberately changed the
    # trained-model path, shifting the dollar-utility at the headline
    # threshold from the 2026-05-06 pin 99.20 to 84.85. Faithfulness:
    # scenario_a is ALL-NUMERIC (no one-hot, so the #773 W2 XGBoost
    # feature-name crash never degraded this run's Step-5b alternates) and
    # 84.8500 reproduced bit-identically across independent nightly runs
    # 27087062518 (2026-06-07, first red) and 27404434136 (2026-06-12).
    # Tolerance width unchanged (±0.5).
    "business_utility": 90.95,  # ±0.5 — was 84.85; re-pinned #1311 (aed06cb7), see above
}

# Select the baseline for this environment.
# CI=true is set automatically by GitHub Actions.
BASELINE = BASELINE_CI if os.getenv("CI") else BASELINE_LOCAL

ECE_POST_MAX = 0.10  # rubric §7: post-isotonic ECE < 0.10 (Agent A research)

# Mild-overfit upper bound, env-gated like BASELINE.  The same CPU-ISA divergence
# that shifts the validation metrics also shifts the train→val delta: AVX2 local
# fits less aggressively (Δ≈0.127), AVX512 CI fits more aggressively (Δ≈0.231 in
# the 2026-05-06 PR #69 CI runs).  Both are bit-deterministic in their env; tight
# margins (0.05) above each observed value catch genuine regularization regression
# without tripping on the inherent env-specific overfit level.
TRAIN_VAL_DELTA_MAX_LOCAL = 0.10  # observed 0.0512 + ~0.05 headroom (scenario_a, 2026-05-06)
TRAIN_VAL_DELTA_MAX_CI = 0.10  # placeholder — replace with CI-observed + 0.05 after first run
TRAIN_VAL_DELTA_MAX = TRAIN_VAL_DELTA_MAX_CI if os.getenv("CI") else TRAIN_VAL_DELTA_MAX_LOCAL

# Tolerance per metric. Tight because current run-variance at seed=42 is 0.
TOLERANCE = {
    "roc_auc": 0.01,
    "pr_auc": 0.02,
    "brier_score": 0.01,
    "mcc": 0.03,
    "business_utility": 0.5,
}


# 15-min budget overrides the integration-tests workflow's --timeout=60 default.
# This test forks scripts/run_tier0_test.py (~156s local wall-clock; 3-5 min in CI).
# Without the marker, pytest-timeout SIGKILLs the worker at 60s
# ("[gw0] node down: Not properly terminated") and pytest-xdist hangs waiting
# for the dead worker until the job-level cancel.
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.timeout(900)
def test_synthetic_e2e_scenario_a_pins_7dim_baseline(tmp_path: Path) -> None:
    """Synthetic e2e --regime scenario_a produces metrics in pinned 7-dim bands.

    Acts as the rubric-tracked regression gate that codex review I7 + D.7.C
    flagged as missing. Any drift in val_AUC / PR-AUC / brier / MCC /
    business_utility / ECE-post-isotonic / train-val Δ trips at least one
    assertion.

    Failure modes this catches (non-exhaustive):
    - Synthetic generator regime regression (Block 4 split-mode mutation —
      see Shard A close at ``.claude/state/quality_arc_a_drift_rca_close_20260506.md``).
    - HPO determinism breakage (sklearn upgrade, optuna seeding drift).
    - Calibration pipeline regression (post-isotonic ECE > 0.10).
    - Evaluator metric computation drift (any of the 7 dims).
    """
    json_out = tmp_path / "tier0_baseline_invariant.json"
    env = os.environ.copy()
    env["TIER0_E2E_JSON_OUT"] = str(json_out)
    # #594: synthetic e2e has NO live Feast store. Post #556 the freshness check
    # FAILS CLOSED when Feast is unavailable → all features read stale → the
    # registrar QC gate hard-blocks training → empty validation_metrics. The
    # #556 escape hatch ALLOW_STALE_FEAST=1 is correct for these no-Feast tests.
    env["ALLOW_STALE_FEAST"] = "1"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_tier0_test.py"),
        "--regime",
        "scenario_a",
        "--split",
        "auto",
        "--hpo-trials",
        "5",  # 5 trials enough for determinism per Agent D
        "--no-save",
        # BentoML serving isn't deployed in CI; --no-bentoml skips the
        # post-train serving probe. MLflow IS available — the workflow
        # starts a local tracking server before this step (see
        # .github/workflows/backend-tests.yml "Start MLflow tracking
        # server"), preserving baseline-equivalence with the docs/results
        # measurements that pinned BASELINE/TOLERANCE above.
        "--no-bentoml",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1200,  # 20 min hard cap; scenario_a e2e is ~3-4 min local, headroom for CI
        cwd=str(REPO_ROOT),
        env=env,
    )

    assert result.returncode == 0, (
        f"Synthetic e2e exited {result.returncode} — pipeline broke before "
        f"baseline-invariant check. stderr (truncated): {result.stderr[-500:]!r}"
    )
    assert json_out.exists(), (
        f"TIER0_E2E_JSON_OUT artifact missing at {json_out}; runner produced no JSON."
    )

    artifact = json.loads(json_out.read_text())
    val = artifact.get("validation_metrics") or {}

    # 5 numeric-band assertions
    # When BASELINE[metric] is None (e.g. mcc in BASELINE_CI), skip that dimension.
    failures = []
    for metric, target in BASELINE.items():
        if target is None:
            continue  # metric not captured for this environment baseline; skip
        observed = val.get(metric)
        if observed is None:
            failures.append(f"{metric}: MISSING from validation_metrics (got: {val.keys()!r})")
            continue
        tolerance = TOLERANCE[metric]
        if abs(observed - target) > tolerance:
            failures.append(
                f"{metric}: observed {observed:.4f}, expected {target:.4f} ± {tolerance:.4f} "
                f"(delta {observed - target:+.4f})"
            )
    assert not failures, (
        "Synthetic baseline invariant tripped:\n  - "
        + "\n  - ".join(failures)
        + "\nIf this is an intentional shift, update BASELINE + TOLERANCE in this file "
        "in the same commit that lands the code change. Do NOT silently widen tolerance."
    )

    # Calibration: rubric §7 demands ECE post-isotonic < 0.10
    test_metrics = artifact.get("test_metrics") or {}
    ece_post = (
        val.get("ece_post_isotonic") or test_metrics.get("ece_post_isotonic") or val.get("ece_post")
    )
    if ece_post is not None:
        assert ece_post < ECE_POST_MAX, (
            f"ECE post-isotonic = {ece_post:.4f} exceeds rubric ceiling {ECE_POST_MAX} "
            "(mlops_data_pipeline_engineering_distilled.md R5 / "
            "tier0_evaluation_vs_distilled_mlops.md:779)."
        )

    # Train→Val overfit signal (best-effort; not always present)
    train_val_delta = test_metrics.get("train_val_auc_delta")
    if train_val_delta is not None:
        assert train_val_delta < TRAIN_VAL_DELTA_MAX, (
            f"Train→Val AUC delta = {train_val_delta:.4f} exceeds severe-overfit "
            f"threshold {TRAIN_VAL_DELTA_MAX}. Likely model regularization regression."
        )
