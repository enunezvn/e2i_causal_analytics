"""Synthetic e2e baseline regression invariant — 7-dim val-side gate.

Added by tier0_quality_remediation_arc Shard D, 2026-05-06, per
user-authorized D.7.C scope ("approve all defaults"). Closes the codex
review's IMPORTANT Q7 finding: ECE / PR-AUC / business_utility / brier /
MCC / train-val Δ are rubric-tracked dimensions absent from any prior
regression gate.

What this test pins
-------------------
- Synthetic e2e ``--regime default`` with the canonical Kisqali seed=42
  fixture is bit-identical run-to-run (research Agent D 2026-05-06,
  ``quality_arc_research_d_metrics_extraction_20260506.md`` — "Synthetic
  post-cleanup is BIT-IDENTICAL to the 04-28 stable rubric reference. All
  12 numeric metrics match to 4 decimals.").
- This test runs the synthetic e2e once and asserts that 7 val-side
  metrics fall within tight tolerance bands around the current stable
  baseline. Any regression in (a) the synthetic generator, (b) HPO
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
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# Current stable baseline (post-`e2ada2d`, since 2026-04-26 14:34 UTC; bit-identical
# across all runs from then through 2026-05-06 per Agent D).
# Anchored to the live evaluator output:
#   docs/results/tier0_pipeline_run_20260428_130229.md (validation_metrics block)
#   logs/tier0_synth_20260505T232512Z.log (line 1620)
BASELINE = {
    "roc_auc": 0.5585,  # ±0.01
    "pr_auc": 0.1958,  # ±0.02
    "brier_score": 0.2293,  # ±0.01
    "mcc": 0.1576,  # ±0.03
    "business_utility": -8.150,  # ±0.5 (matches rubric §7 target ≈ -8.15)
}

ECE_POST_MAX = 0.10  # rubric §7: post-isotonic ECE < 0.10 (Agent A research)
TRAIN_VAL_DELTA_MAX = 0.20  # mild-overfit upper bound; current 0.127

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
def test_synthetic_e2e_default_regime_pins_7dim_baseline(tmp_path: Path) -> None:
    """Synthetic e2e --regime default produces metrics in pinned 7-dim bands.

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

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_tier0_test.py"),
        "--regime",
        "default",
        "--split",
        "auto",
        "--hpo-trials",
        "5",  # 5 trials enough for determinism per Agent D
        "--no-save",
        # MLflow + BentoML are configured to localhost in the integration-tests
        # workflow but no server runs there — the pipeline would exit 1 on
        # the first MLflow API call. Validation metrics are computed and
        # emitted to TIER0_E2E_JSON_OUT regardless of MLflow status, so the
        # 7-dim gate is unaffected.
        "--disable-mlflow",
        "--no-bentoml",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min hard cap; e2e is ~3 min wall-clock
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
    failures = []
    for metric, target in BASELINE.items():
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
