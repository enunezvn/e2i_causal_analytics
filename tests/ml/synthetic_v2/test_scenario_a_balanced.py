"""Tests for ``ScenarioName.A_DIAGNOSTIC_BC_IDFS_BALANCED`` prevalence convergence.

Per ``.claude/plans/synthetic_cohort_growth_plan_20260509.md`` Phase 3 Task 3.1b:
preflight test that the intercept solver in ``api.py:204`` reaches the shifted
``target_prevalence=0.50`` cleanly across multiple n_total + seed combinations.

Why preflight matters: scenario_a's locked DGP coefficients calibrate against
prevalence=0.20. Shifting target_prevalence to 0.50 in the subclass relies on
the bisection solver finding an intercept that drives the realised cohort
prevalence to 0.50 — but if scenario_a's linear-predictor range under the
data distribution is too narrow to reach the 0.50 fixed point, the solver
will land on a numerically-stable intercept that yields some other prevalence
(typically 0.45 or 0.55), silently breaking the balanced-cohort experiment.

This module catches that failure at unit-test time so Phase 3.2 (runner
wiring) and Phase 3.3 (3-condition empirical contrast) don't run blind.
"""

from __future__ import annotations

import pytest

from src.ml.synthetic_v2.api import generate_scenario
from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY, ScenarioName
from src.ml.synthetic_v2.scenarios.scenario_a import ScenarioABuilder
from src.ml.synthetic_v2.scenarios.scenario_a_balanced import (
    ScenarioABalancedBuilder,
)


class TestScenarioABalancedBuilderShape:
    def test_registers_in_scenario_registry(self) -> None:
        assert ScenarioName.A_DIAGNOSTIC_BC_IDFS_BALANCED in SCENARIO_REGISTRY

    def test_registry_factory_returns_balanced_builder(self) -> None:
        builder = SCENARIO_REGISTRY[ScenarioName.A_DIAGNOSTIC_BC_IDFS_BALANCED]()
        assert isinstance(builder, ScenarioABalancedBuilder)

    def test_inherits_from_scenario_a(self) -> None:
        builder = ScenarioABalancedBuilder()
        assert isinstance(builder, ScenarioABuilder)

    def test_target_prevalence_shifted_to_half(self) -> None:
        builder = ScenarioABalancedBuilder()
        assert builder.target_prevalence == 0.50

    def test_inherits_scenario_a_n_features(self) -> None:
        builder = ScenarioABalancedBuilder()
        assert builder.n_features == ScenarioABuilder().n_features

    def test_inherits_scenario_a_feature_manifest_identity(self) -> None:
        # Exact same manifest object — no DGP perturbation, only prevalence shifts.
        balanced = ScenarioABalancedBuilder()
        baseline = ScenarioABuilder()
        assert balanced.feature_manifest == baseline.feature_manifest

    def test_inherits_scenario_a_correlation_blocks(self) -> None:
        balanced = ScenarioABalancedBuilder()
        baseline = ScenarioABuilder()
        assert balanced.correlation_blocks == baseline.correlation_blocks

    def test_name_is_balanced_variant(self) -> None:
        builder = ScenarioABalancedBuilder()
        assert builder.name == ScenarioName.A_DIAGNOSTIC_BC_IDFS_BALANCED


# Plan Task 3.1b: preflight prevalence convergence
# Bernoulli SD on n=20000 = √(0.5·0.5/20000) ≈ 0.00354 → ±2σ ≈ ±0.007.
# We use a slightly looser tolerance (0.02) to absorb both the sampling
# noise and any minor solver imprecision (bisection ε is ~1e-6 on logits;
# at p=0.50 the sigmoid slope is 0.25 so ε on prevalence is ~2.5e-7,
# negligible compared to sampling).
PREVALENCE_TOLERANCE = 0.02


@pytest.mark.parametrize("n_total", [500, 2000, 6000, 20000])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_scenario_a_balanced_prevalence_converges_at_multiple_n(
    n_total: int, seed: int
) -> None:
    """Realised prevalence is within ±0.02 of target=0.50 for every (n, seed) pair.

    If this test fails for a given n, it means the intercept solver could
    not reach 0.50 with scenario_a's locked DGP coefficients at that cohort
    size. Phase 3.2 (runner wiring) and Phase 3.3 (empirical contrast)
    depend on this convergence; failure halts the cohort-growth plan and
    sends it to one of the four fallback paths in plan §Task 3.1b.
    """
    ds = generate_scenario(
        ScenarioName.A_DIAGNOSTIC_BC_IDFS_BALANCED, seed=seed, n_total=n_total
    )
    # Use the realised prevalence from metadata — that's the full-cohort
    # prevalence pre-split, which is what the solver targets.
    realised = ds.metadata.realized_prevalence
    delta = abs(realised - 0.50)
    assert delta <= PREVALENCE_TOLERANCE, (
        f"scenario_a_balanced realised prevalence {realised:.4f} "
        f"deviates from target 0.50 by {delta:.4f} > "
        f"tolerance {PREVALENCE_TOLERANCE} (n_total={n_total}, seed={seed}). "
        "Intercept solver may not reach 0.50 with scenario_a's locked DGP "
        "coefficients at this cohort size; see plan §Task 3.1b for fallback paths."
    )


def test_target_prevalence_is_in_metadata() -> None:
    """``ScenarioMetadata.target_prevalence`` reflects the subclass override."""
    ds = generate_scenario(
        ScenarioName.A_DIAGNOSTIC_BC_IDFS_BALANCED, seed=42, n_total=2000
    )
    assert ds.metadata.target_prevalence == 0.50


def test_baseline_scenario_a_unaffected_by_balanced_addition() -> None:
    """Adding scenario_a_balanced does NOT change scenario_a's own prevalence.

    Defends against an accidental shared-state mutation — e.g., if the
    balanced subclass mutated ``ScenarioABuilder.target_prevalence`` in
    place, this test would fail at 0.50 instead of 0.20.
    """
    ds = generate_scenario(ScenarioName.A_DIAGNOSTIC_BC_IDFS, seed=42, n_total=2000)
    # scenario_a target is 0.20; sampling tolerance ±0.02 (Bernoulli SD ≈ 0.009)
    assert abs(ds.metadata.realized_prevalence - 0.20) <= 0.02
    assert ds.metadata.target_prevalence == 0.20
