"""Optum initiation-arm CI regression guard.

Phase 5.2 closed the Optum initiation cohort via PR #42 (real Optum n=972,
manual ``scripts/run_tier0_test.py --target initiated_biologic_180d`` run
+ evidence doc in ``docs/results/``). The closure was a manual recipe,
NOT a CI-enforced test. If the runner's ``--target`` wiring or the
synthetic-data schema drifts, no test today catches the regression — the
documented recipe in ``phase5p2_state.md`` workflow note 3 would silently
break.

This file is the CI-friendly regression guard. Real Optum data is
gitignored and unavailable in CI; we use ``SampleDataGenerator`` + a
post-hoc rename of ``discontinuation_flag`` -> ``initiated_biologic_180d``
to exercise the initiation-arm code path without real data.

Shape selection (codex consult 2026-05-04, agent ``a117f54a623c1f2fc``):
shape (ii) — synthetic-data integration test with column-rename surgery.
Codex flagged scope-creep into Shard #6 (``test_agent_output_contracts.py``)
as NO since this guard tests the runner's ``--target`` override + the
generator's column contract, not any agent's output schema.

Discriminating-coverage protection per ``feedback_pr_merge_workflow.md`` §7:
the renamed frame is built from the live generator (not pre-injected by
the test fixture itself), so a generator regression — column drop, target
collapse, all-None — fires the assertions rather than silently passing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.repositories.sample_data import SampleDataGenerator  # noqa: E402,I001

# Initiation-arm canonical target column. Documented in phase5p2_state.md
# workflow note 3 + scripts/run_tier0_test.py:5623-5626 (--target arg help).
INITIATION_TARGET = "initiated_biologic_180d"

# SampleDataGenerator's hardcoded target column (sample_data.py:693). The
# generator does not natively support initiation targets; we rename
# post-hoc to mirror what the runner sees on real Optum data.
GENERATOR_TARGET = "discontinuation_flag"

# n=200 is the smallest reliable boundary above the Step-5 split-validation
# floor — see test_phase5p2_scale_machinery.py:38-46 for the codex-validated
# rationale (60/20/10/10 per #44 yields train=120/val=40/test=20/holdout=20,
# all above CONFIG.min_samples_per_split=10).
INITIATION_N = 200

# 30% positive rate keeps both classes comfortably represented in every
# 60/20/10/10 partition at n=200 (~12+ positives per split). Higher than
# the 10% used in scale-machinery's primary boundary test because real
# Optum initiation prevalence is closer to 30%.
INITIATION_POSITIVE_RATE = 0.30


@pytest.fixture(scope="module")
def renamed_initiation_frame() -> object:
    """Build a synthetic initiation-arm frame: ml_patients() + column rename.

    The runner's ``--target initiated_biologic_180d`` path expects the
    cohort DataFrame to expose that column. The synthetic generator emits
    ``discontinuation_flag``; rename in-place to mirror what the runner
    sees on real Optum data.
    """
    gen = SampleDataGenerator(seed=42)
    df = gen.ml_patients(
        n_patients=INITIATION_N,
        positive_rate=INITIATION_POSITIVE_RATE,
    )
    return df.rename(columns={GENERATOR_TARGET: INITIATION_TARGET})


# --------------------------------------------------------------------------- #
# Vacuous-pass guard (§7) — generator is alive and emits the expected shape   #
# --------------------------------------------------------------------------- #


def test_renamed_frame_has_expected_row_count(renamed_initiation_frame: object) -> None:
    """Sanity check: the synthetic-data path is alive at n=200.

    Without this guard, a generator that silently returns an empty frame
    would pass every other assertion below (vacuous truth). Mirrors
    test_phase5p2_scale_machinery.py:108 in spirit.
    """
    df = renamed_initiation_frame
    assert len(df) == INITIATION_N, (  # type: ignore[arg-type]
        f"Expected {INITIATION_N} rows from ml_patients(); got {len(df)}. "  # type: ignore[arg-type]
        "Generator regressed or fixture mis-built."
    )


# --------------------------------------------------------------------------- #
# Schema invariants — initiation arm column contract                           #
# --------------------------------------------------------------------------- #


def test_renamed_frame_exposes_initiation_target(renamed_initiation_frame: object) -> None:
    """Rename succeeded: the initiation target is present, the old one is gone.

    Catches the documented PR #42 blocker (phase5p2_state.md workflow note
    3): when ``CONFIG.target_outcome = 'initiated_biologic_180d'`` but the
    DataFrame still has ``discontinuation_flag``, the runner raises
    ZeroDivisionError at ``load_rwd_data`` (run_tier0_test.py:1296).
    """
    df = renamed_initiation_frame
    columns = list(df.columns)  # type: ignore[attr-defined]
    assert INITIATION_TARGET in columns, (
        f"Post-hoc rename failed: '{INITIATION_TARGET}' missing from frame "
        f"(first 10 columns: {columns[:10]!r}). The generator's target "
        f"column name may have changed in src/repositories/sample_data.py."
    )
    assert GENERATOR_TARGET not in columns, (
        f"Old target column '{GENERATOR_TARGET}' leaked through rename — "
        f"the runner would read the wrong column on the initiation arm."
    )


def test_renamed_frame_target_is_non_degenerate(renamed_initiation_frame: object) -> None:
    """Target must have both classes after rename (not all-None, not single-class).

    Catches the documented PR #42 ZeroDivisionError pattern in
    phase5p2_state.md workflow note 3 (Optum initiation cohort had
    all-None for ``discontinuation_flag`` until the ``--target`` override
    landed).
    """
    df = renamed_initiation_frame
    target = df[INITIATION_TARGET]  # type: ignore[index]
    assert target.notna().all(), (
        f"'{INITIATION_TARGET}' has {target.isna().sum()} None values — "
        "all-None crash pattern from phase5p2_state.md note 3 not guarded."
    )
    assert target.nunique() == 2, (
        f"'{INITIATION_TARGET}' collapsed to single class; unique values: {target.unique()!r}"
    )
    realised = float(target.mean())
    assert 0 < realised < 1, (
        f"'{INITIATION_TARGET}' degenerate (rate={realised:.3f}); "
        "split stratify would fail on this frame."
    )


# --------------------------------------------------------------------------- #
# Split-math floor — initiation-arm boundary at n=200                          #
# --------------------------------------------------------------------------- #


def test_initiation_split_math_passes_floor_at_n200(renamed_initiation_frame: object) -> None:
    """At n=200, 60/20/10/10 splits (#44 policy) all exceed min_samples_per_split=10.

    Mirrors test_step5_split_floor_passes_at_n200_prev10 in
    test_phase5p2_scale_machinery.py but pins the floor for the
    initiation arm specifically. If a future change to the runner's
    split logic regresses for the initiation cohort, this fires.

    CONFIG.min_samples_per_split=10 is the documented floor at
    scripts/run_tier0_test.py:5573-5581.
    """
    df = renamed_initiation_frame
    n = len(df)  # type: ignore[arg-type]
    train_n = int(n * 0.60)
    val_n = int(n * 0.20)
    test_n = int(n * 0.10)
    holdout_n = n - train_n - val_n - test_n
    floor = 10
    for label, sz in [
        ("train", train_n),
        ("val", val_n),
        ("test", test_n),
        ("holdout", holdout_n),
    ]:
        assert sz >= floor, (
            f"{label}_n={sz} below min_samples_per_split={floor} at "
            f"n={INITIATION_N} on initiation arm — Step 5 would fail."
        )


# --------------------------------------------------------------------------- #
# Runner argparser regression guard — --target wiring                          #
# --------------------------------------------------------------------------- #


def test_runner_target_arg_is_wired() -> None:
    """``--target`` argparser arg must remain accepted by the runner.

    Lightweight subprocess probe of ``run_tier0_test.py --help``. Does
    NOT execute any pipeline — just argparser. If the ``--target`` arg
    is removed or renamed, the documented PR #42 manual recipe breaks
    and this fires.

    Also implicitly guards the PR #44 ``%%`` format-string fix: a broken
    ``--help`` formatter raises ValueError before reaching ``--target``,
    and this test would surface the regression as a non-zero exit.
    """
    runner = REPO_ROOT / "scripts" / "run_tier0_test.py"
    assert runner.exists(), f"Runner not found at {runner}"
    result = subprocess.run(
        [sys.executable, str(runner), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"`run_tier0_test.py --help` exited {result.returncode}; "
        f"argparser or format-string broken? "
        f"stderr (truncated): {result.stderr[:500]!r}"
    )
    help_text = result.stdout
    assert "--target" in help_text, (
        "argparser missing --target; PR #42 initiation recipe would "
        f"break. Help excerpt (first 500 chars): {help_text[:500]!r}"
    )
    # The --target help text references CONFIG.target_outcome; if the
    # ``Override CONFIG.target_outcome`` doc string drifts, the manual
    # recipe in phase5p2_state.md becomes ambiguous.
    assert "target_outcome" in help_text or "treatment_initiated" in help_text, (
        "--target help text missing CONFIG.target_outcome reference. "
        f"Help excerpt (first 500 chars): {help_text[:500]!r}"
    )
