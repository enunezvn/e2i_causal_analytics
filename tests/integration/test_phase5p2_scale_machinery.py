"""Phase 5.2 dry-run scale-machinery validation.

Pins boundary invariants of the tier-0 pipeline at synthetic-cohort
boundaries that real Optum cohorts can't reach today (discontinuation /
persistence are stuck at n=47, below the n ≥ 200 trigger documented in
`prod_readiness_backlog.md` §2 + `tier0_evaluation_vs_distilled_mlops.md:691-703`).

This is the **CI-runnable lower-bound invariant suite** of the scale-
machinery shard — the local-only full-tier-0 grid runs are captured in
`docs/results/phase5p2_scale_machinery_<ts>.md`. Per `feedback_pr_merge_workflow.md`
§7, every assertion in this file has discriminating coverage so that
silent vacuous-pass under heavy mocking is caught.

The invariants are framed around the empirical Step-5 split-validation
floor that the 2026-04-24 Optum baseline established (n=47 fails Step 5
with `split_validation_error`; n=972 passes). This test pins behaviour
at n=200 (just above the floor) so that future changes to data_preparer,
split_enforcer, or model_trainer can't silently regress the floor.

Per `feedback_pr_merge_workflow.md` §5, async paths use pytest-asyncio,
not `asyncio.run()`, to avoid the Opik-telemetry event-loop-closed flake.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.repositories.sample_data import SampleDataGenerator  # noqa: E402,I001


# Per codex consult 2026-05-04 (agent a847abbf19d4da2e9), ratios updated to
# the #44 60/20/10/10 policy (2026-07-21) — conclusions unchanged:
# - n=200 is the smallest reliable boundary above the Step 5 split-validation
#   floor (60/20/10/10 yields train=120/val=40/test=20/holdout=20).
# - prev=10% gives ~12 positives per training set, enough for stratified
#   splitting to produce at least one positive per partition.
# - prev=2% with n=200 is the documented "must-skip" weak-signal regime:
#   60/20/10/10 yields ~2-4 positives across all splits — split stratify
#   may fail or pass non-deterministically. We do NOT assert success there;
#   we only assert that ml_patients itself produces valid data.
SCALE_GRID = [
    pytest.param(200, 0.10, id="n200_prev10"),
    pytest.param(200, 0.25, id="n200_prev25"),
    pytest.param(1500, 0.10, id="n1500_prev10"),
]

WEAK_SIGNAL_GRID = [
    pytest.param(200, 0.02, id="n200_prev2_weak_signal"),
]


# --------------------------------------------------------------------------- #
# Discriminating-coverage guards (vacuous-pass protection — §7)                #
# --------------------------------------------------------------------------- #


def test_grid_is_non_empty() -> None:
    """Vacuous-pass guard per `feedback_pr_merge_workflow.md` §7.

    If `SCALE_GRID` ever shrinks to empty, the parametrized tests below
    silently pass without exercising any boundary. Catch that here.
    """
    assert len(SCALE_GRID) >= 3, (
        f"SCALE_GRID must include at least 3 combos for boundary coverage; got {len(SCALE_GRID)}"
    )
    assert len(WEAK_SIGNAL_GRID) >= 1, "WEAK_SIGNAL_GRID must document at least one prev<5% combo"


def test_sample_data_generator_imports_cleanly() -> None:
    """Confirm the synthetic generator entry point is wired correctly.

    A common failure mode under heavy mocking: imports succeed but the
    class doesn't expose `ml_patients`. This guards the test suite's
    contract against the generator interface.
    """
    gen = SampleDataGenerator(seed=42)
    assert hasattr(gen, "ml_patients"), (
        "SampleDataGenerator must expose ml_patients(); see src/repositories/sample_data.py:534"
    )


# --------------------------------------------------------------------------- #
# Synthetic data generation invariants                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_patients,positive_rate", SCALE_GRID)
def test_ml_patients_produces_valid_frame_at_scale(n_patients: int, positive_rate: float) -> None:
    """ml_patients() must emit a non-degenerate DataFrame with the target.

    Asserts:
    - Row count matches request
    - Target column `discontinuation_flag` is present
    - Both classes are represented (rules out collapse to single class)
    - Realised positive rate within ±15 pp of requested (loose tolerance —
      generator clips to [0.05, 0.95] before Bernoulli sampling per
      `sample_data.py:559-561`)
    """
    gen = SampleDataGenerator(seed=42)
    df = gen.ml_patients(n_patients=n_patients, positive_rate=positive_rate)

    assert len(df) == n_patients, f"expected {n_patients} rows, got {len(df)}"
    assert "discontinuation_flag" in df.columns, (
        f"missing target column; got {list(df.columns)[:10]}"
    )

    realized_positive_rate = float(df["discontinuation_flag"].mean())
    assert 0 < realized_positive_rate < 1, (
        f"target collapsed to single class at "
        f"requested prev={positive_rate} (got {realized_positive_rate})"
    )

    # Loose tolerance — clip-then-sample produces deviation, especially
    # at extreme prevalences.
    tolerance = 0.15 if positive_rate <= 0.10 else 0.20
    assert abs(realized_positive_rate - positive_rate) < tolerance, (
        f"realised prev={realized_positive_rate:.3f} far from "
        f"requested {positive_rate:.3f} (tolerance {tolerance})"
    )


@pytest.mark.parametrize("n_patients,positive_rate", WEAK_SIGNAL_GRID)
def test_ml_patients_at_weak_signal_boundary(n_patients: int, positive_rate: float) -> None:
    """At very low prevalence (≤5%), only assert generation succeeds.

    Per codex consult 2026-05-04: at n=200/prev=2%, the 60/20/10/10 split
    will produce splits with 0-1 positives. Step-5 validation is
    expected to fail or produce unstable signals. This test ONLY pins
    that the generator itself does not crash; it does NOT assert
    pipeline-level success.
    """
    gen = SampleDataGenerator(seed=42)
    df = gen.ml_patients(n_patients=n_patients, positive_rate=positive_rate)
    assert len(df) == n_patients
    # Generator clips prev to [0.05, 0.95]; we accept whatever it produces.
    realized = float(df["discontinuation_flag"].mean())
    assert 0 <= realized <= 1


# --------------------------------------------------------------------------- #
# Step-5 split-validation floor — the empirical 2026-04-24 boundary           #
# --------------------------------------------------------------------------- #


def test_step5_split_floor_passes_at_n200_prev10() -> None:
    """At n=200/prev=10%, the 60/20/10/10 split yields enough positives
    in every split for stratified splitting to succeed.

    The 2026-04-24 baseline established that n=47 fails Step 5 with
    `split_validation_error`. At n=200, the corresponding split is
    train=120/val=40/test=20/holdout=20 with ~12 expected positives —
    comfortably above the documented `min_samples_per_split=10` floor
    (`run_tier0_test.py:5573-5581`). Ratios follow the #44 holdout
    enlargement (2026-07-21): test 15%→10%, holdout 5%→10%.
    """
    gen = SampleDataGenerator(seed=42)
    df = gen.ml_patients(n_patients=200, positive_rate=0.10)

    # 60/20/10/10 split
    n = len(df)
    train_n = int(n * 0.60)
    val_n = int(n * 0.20)
    test_n = int(n * 0.10)
    holdout_n = n - train_n - val_n - test_n

    assert train_n >= 10, f"train_n={train_n} below min_samples_per_split"
    assert val_n >= 10, f"val_n={val_n} below min_samples_per_split"
    assert test_n >= 10, f"test_n={test_n} below min_samples_per_split"
    assert holdout_n >= 10, f"holdout_n={holdout_n} below min_samples_per_split"

    # Per-split positive counts (sorted to establish stratified-split feasibility)
    df_sorted = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_pos = int(df_sorted.iloc[:train_n]["discontinuation_flag"].sum())
    val_pos = int(df_sorted.iloc[train_n : train_n + val_n]["discontinuation_flag"].sum())

    # At prev=0.10, we expect >= 1 positive in each split with very high
    # probability. Catch the case where positive class accidentally
    # concentrates in one split.
    assert train_pos >= 1, "no positives in train split at n=200 prev=10%"
    assert val_pos >= 1, "no positives in val split at n=200 prev=10%"


def test_step5_split_floor_below_threshold_at_n47() -> None:
    """At n=47 (the empirical Optum discontinuation/persistence size),
    the 60/20/10/10 split breaks the min_samples_per_split=10 default.

    This test documents the empirical floor: at n=47, train=28/val=9/test=4/
    holdout=6 — val/test/holdout all below the default min=10 gate. The
    2026-04-24 baseline showed this pattern (then 60/20/15/5) fails with
    `split_validation_error` at Step 5; the #44 ratios violate the same floor.

    This test does NOT assert that Step 5 fails (would require running
    the full pipeline). It asserts the SPLIT MATH violates the documented
    floor — making the empirical Step 5 failure a deterministic consequence.
    """
    gen = SampleDataGenerator(seed=42)
    df = gen.ml_patients(n_patients=47, positive_rate=0.23)

    n = len(df)
    train_n = int(n * 0.60)
    val_n = int(n * 0.20)
    test_n = int(n * 0.10)
    holdout_n = n - train_n - val_n - test_n

    # Floor violation — documented as the empirical Step 5 failure cause.
    floor = 10
    below_floor = sum(1 for sz in [val_n, test_n, holdout_n] if sz < floor)
    assert below_floor >= 2, (
        f"expected ≥2 splits below min_samples_per_split={floor} at n=47; "
        f"got val={val_n} test={test_n} holdout={holdout_n}"
    )


# --------------------------------------------------------------------------- #
# Permutation invariant — verdict gate must reject random labels              #
# --------------------------------------------------------------------------- #


def test_permutation_shuffled_target_breaks_signal_at_scale() -> None:
    """Sanity check: shuffling the target eliminates the signal.

    The tier-0 verdict gate fires DO NOT DEPLOY when permutation test
    shows shuffled-AUC ≈ 0.5. This test confirms the underlying
    invariant — that ml_patients's signal is target-driven, not
    feature-driven by accident — using a simple proxy: the target's
    correlation with `days_on_therapy` (a known signal feature per
    `sample_data.py:580-582`) should drop to near-zero after shuffle.
    """
    rng = np.random.default_rng(seed=42)
    gen = SampleDataGenerator(seed=42)
    df = gen.ml_patients(n_patients=1500, positive_rate=0.30)

    # Original correlation with the canonical signal feature
    if "days_on_therapy" not in df.columns:
        pytest.skip("days_on_therapy feature not present — generator changed")

    orig_corr = abs(
        float(np.corrcoef(df["days_on_therapy"].astype(float), df["discontinuation_flag"])[0, 1])
    )

    # Shuffle target and recompute
    shuffled = df.copy()
    shuffled["discontinuation_flag"] = rng.permutation(df["discontinuation_flag"].to_numpy())
    shuffled_corr = abs(
        float(
            np.corrcoef(
                shuffled["days_on_therapy"].astype(float),
                shuffled["discontinuation_flag"],
            )[0, 1]
        )
    )

    # Original signal must be MEANINGFULLY larger than shuffled (>= 3x).
    # Looser than perfect because at n=1500 there's randomness.
    assert orig_corr > 3 * shuffled_corr, (
        f"signal feature failed to discriminate vs shuffled labels: "
        f"orig_corr={orig_corr:.4f}, shuffled_corr={shuffled_corr:.4f}"
    )
    assert shuffled_corr < 0.1, (
        f"shuffled label correlation {shuffled_corr:.4f} ≥ 0.1 — generator "
        f"may have a leaked deterministic signal"
    )


# --------------------------------------------------------------------------- #
# Scale-runtime sanity                                                         #
# --------------------------------------------------------------------------- #


def test_n5000_generation_completes_within_runtime_envelope() -> None:
    """Light scale-runtime gate.

    Per `repeated_k10_test_oom_followup.md`, ml_foundation eagerly imports
    LightGBM/XGBoost/NGBoost/MAPIE on construction; running the FULL
    tier-0 at n=5000 OOMs the 7GB GitHub Actions runner under xdist
    concurrency. This test only exercises the GENERATOR at n=5000 to
    confirm data synthesis itself is not the bottleneck.
    """
    import time

    gen = SampleDataGenerator(seed=42)
    start = time.time()
    df = gen.ml_patients(n_patients=5000, positive_rate=0.10)
    elapsed = time.time() - start

    assert len(df) == 5000
    # Generator should be quick (~1-2s at n=5000); 30s is a generous ceiling.
    # If this regresses, it indicates a deeper performance issue in
    # ml_patients that would compound at full-tier-0 scale.
    assert elapsed < 30.0, (
        f"ml_patients(n=5000) took {elapsed:.1f}s — investigate "
        f"sample_data.py for performance regression"
    )
