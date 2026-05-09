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
def test_scenario_a_balanced_prevalence_converges_at_multiple_n(n_total: int, seed: int) -> None:
    """Realised prevalence is within ±0.02 of target=0.50 for every (n, seed) pair.

    If this test fails for a given n, it means the intercept solver could
    not reach 0.50 with scenario_a's locked DGP coefficients at that cohort
    size. Phase 3.2 (runner wiring) and Phase 3.3 (empirical contrast)
    depend on this convergence; failure halts the cohort-growth plan and
    sends it to one of the four fallback paths in plan §Task 3.1b.
    """
    ds = generate_scenario(ScenarioName.A_DIAGNOSTIC_BC_IDFS_BALANCED, seed=seed, n_total=n_total)
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
    ds = generate_scenario(ScenarioName.A_DIAGNOSTIC_BC_IDFS_BALANCED, seed=42, n_total=2000)
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


# Codex-rescue H2 (2026-05-09): runner skips the LLM-assisted leakage
# check for ALL 4 synthetic_v2 regimes (was scenario_a only). Since the
# skip is the load-bearing fix that prevents the LLM remediator from
# hallucinating a 5-feature replacement set, we add a manifest-side
# sanity check here so future scenario authors get fail-loud feedback
# if their manifest has a single-feature-dominance pattern that would
# look like leakage to a trained discriminator.
@pytest.mark.parametrize(
    "scenario",
    [
        ScenarioName.A_DIAGNOSTIC_BC_IDFS,
        ScenarioName.A_DIAGNOSTIC_BC_IDFS_BALANCED,
        ScenarioName.B_SCREENING_IGAN_ESKD,
        ScenarioName.C_TREATMENT_CSU_RESPONSE,
    ],
)
def test_no_single_feature_dominance(scenario: ScenarioName) -> None:
    """No manifest feature carries > 50% of the total |coefficient| L1 mass.

    Codex-rescue pass-2 M1 fix (2026-05-09): prior version was a global-
    magnitude cap (max |coef × slope| ≤ 3.0) that did NOT actually proxy
    for dominance — a manifest with one coef=2.99 and 5 tiny coefs of 0.001
    would pass while being 99.8% single-feature-dominated.
    ``slope_multiplier`` is a per-scenario scalar that scales all features
    equally, so it cancels out of the relative-share calculation and adds
    no discrimination between features.

    The relative-share check ``max(|c|) / sum(|c|) ≤ 0.5`` is the actual
    dominance contract: no single feature explains more than half of the
    aggregate signal. Existing 4 manifests easily clear this bar
    (scenario_a's largest coef is ≈ 0.45 vs sum of |c| ≈ 12, giving max
    share ~0.04).

    Pairs with the runner-side ``skip_leakage_check=True`` decision: the
    skip is safe iff each scenario's manifest is non-leaky by construction;
    this test makes the dominance side of that contract explicit.
    """
    builder = SCENARIO_REGISTRY[scenario]()
    abs_coefs = [abs(m.coefficient) for m in builder.feature_manifest]
    total = sum(abs_coefs)
    assert total > 0, (
        f"{scenario.name}: no non-zero coefficients in manifest — "
        "scenario has no signal at all (separate from dominance)."
    )
    max_share = max(abs_coefs) / total
    DOMINANCE_THRESHOLD = 0.50
    assert max_share <= DOMINANCE_THRESHOLD, (
        f"{scenario.name}: max coefficient share = {max_share:.4f} "
        f"(largest |coef| {max(abs_coefs):.4f} / sum |coef| {total:.4f}) "
        f"exceeds dominance threshold {DOMINANCE_THRESHOLD}. "
        "A single-feature-dominance manifest would produce AUC ~ 1.0 and "
        "look like leakage to a trained discriminator — breaking the "
        "skip_leakage_check assumption in scripts/run_tier0_test.py:4515."
    )


@pytest.mark.parametrize(
    "scenario",
    [
        ScenarioName.A_DIAGNOSTIC_BC_IDFS,
        ScenarioName.A_DIAGNOSTIC_BC_IDFS_BALANCED,
        ScenarioName.B_SCREENING_IGAN_ESKD,
        ScenarioName.C_TREATMENT_CSU_RESPONSE,
    ],
)
def test_signal_is_distributed_across_multiple_features(scenario: ScenarioName) -> None:
    """At least 5 manifest features have non-zero coefficient.

    Distributed signal is the dual of "no single-feature dominance" — a
    healthy biology-grounded manifest has multiple features each
    contributing modestly. ≥5 non-zero coefficients is a low bar that the
    existing 4 manifests easily clear (scenario_a has 35 of 40 non-zero).
    """
    builder = SCENARIO_REGISTRY[scenario]()
    nonzero = sum(1 for m in builder.feature_manifest if abs(m.coefficient) > 1e-9)
    assert nonzero >= 5, (
        f"{scenario.name}: only {nonzero} features with non-zero coefficient. "
        "Distributed signal (≥5 non-zero) is required for the "
        "skip_leakage_check assumption to hold."
    )
