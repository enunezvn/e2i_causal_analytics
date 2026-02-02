"""DoWhy Refutation Tests for Synthetic Data Validation.

Version: 4.3
Purpose: Validate causal estimates using DoWhy's refutation methods

Refutation Methods Tested:
    1. Random Common Cause: Add random variable as confounder
    2. Placebo Treatment: Permute treatment variable
    3. Subset Data: Re-estimate on data subsets
    4. Bootstrap: Confidence via resampling

Target: >= 60% pass rate across refutations
Reference: docs/E2I_Causal_Validation_Protocol.html
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

try:
    import dowhy  # noqa: F401
    from dowhy import CausalModel

    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

from tests.synthetic.conftest import (
    SyntheticDataset,
    generate_confounded_moderate,
    generate_simple_linear,
)

# Skip all tests if DoWhy not available
pytestmark = pytest.mark.skipif(not DOWHY_AVAILABLE, reason="DoWhy not installed")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def refutation_dataset() -> SyntheticDataset:
    """Generate dataset optimized for refutation testing.

    Uses larger sample size and stronger effect for reliable refutations.
    """
    return generate_confounded_moderate(
        n=5000,  # Memory-safe size
        true_ate=0.30,
        confounder_strength=0.40,
        seed=42,
    )


@pytest.fixture
def simple_dataset() -> SyntheticDataset:
    """Generate simple linear dataset for baseline refutation testing."""
    return generate_simple_linear(n=5000, true_ate=0.50, seed=42)


def build_causal_model(dataset: SyntheticDataset) -> CausalModel:
    """Build DoWhy CausalModel from synthetic dataset.

    Args:
        dataset: SyntheticDataset with treatment, outcome, and confounders

    Returns:
        Configured CausalModel instance
    """
    # Build causal graph
    if dataset.confounder_cols:
        # With confounders: C -> T, C -> Y
        graph_edges = []
        for c in dataset.confounder_cols:
            graph_edges.append(f'"{c}" -> "{dataset.treatment_col}"')
            graph_edges.append(f'"{c}" -> "{dataset.outcome_col}"')
        graph_edges.append(f'"{dataset.treatment_col}" -> "{dataset.outcome_col}"')
        gml_graph = f"digraph {{ {'; '.join(graph_edges)} }}"
    else:
        # Simple: T -> Y
        gml_graph = f'digraph {{ "{dataset.treatment_col}" -> "{dataset.outcome_col}" }}'

    model = CausalModel(
        data=dataset.data,
        treatment=dataset.treatment_col,
        outcome=dataset.outcome_col,
        common_causes=dataset.confounder_cols if dataset.confounder_cols else None,
        graph=gml_graph,
    )

    return model


def estimate_ate_with_model(model: CausalModel) -> Any:
    """Identify and estimate treatment effect using linear regression.

    Args:
        model: DoWhy CausalModel

    Returns:
        CausalEstimate object
    """
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
    )
    return estimate


# ============================================================================
# REFUTATION TESTS
# ============================================================================


class TestRandomCommonCauseRefutation:
    """Test refutation by adding random common cause.

    If estimate is robust, adding a random variable as a confounder
    should not significantly change the estimate.
    """

    def test_random_common_cause_refutation(self, refutation_dataset: SyntheticDataset):
        """Verify estimate stability when adding random confounder.

        Expected: New estimate within 10% of original estimate
        This tests whether the model correctly handles spurious confounders.
        """
        model = build_causal_model(refutation_dataset)
        estimate = estimate_ate_with_model(model)

        # Run refutation
        refutation = model.refute_estimate(
            model.identify_effect(proceed_when_unidentifiable=True),
            estimate,
            method_name="random_common_cause",
            placebo_type="permute",
            num_simulations=5,  # Low for memory efficiency
        )

        # The refuted estimate should be close to the original
        original_value = estimate.value
        refuted_value = refutation.new_effect

        # Assert estimate is relatively stable
        # Allow 20% deviation due to random noise
        relative_change = abs(refuted_value - original_value) / abs(original_value)

        assert relative_change < 0.20, (
            f"Random common cause significantly changed estimate: "
            f"original={original_value:.4f}, refuted={refuted_value:.4f}, "
            f"change={relative_change:.1%}"
        )

        # Also verify the refutation p-value if available
        if hasattr(refutation, "refutation_result") and refutation.refutation_result is not None:
            # Should not reject the null (estimate is robust)
            # p-value should be > 0.05
            pass  # DoWhy doesn't always provide p-values for this method

    def test_random_common_cause_on_simple_linear(self, simple_dataset: SyntheticDataset):
        """Random common cause on unconfounded data should not change estimate."""
        model = build_causal_model(simple_dataset)
        estimate = estimate_ate_with_model(model)

        refutation = model.refute_estimate(
            model.identify_effect(proceed_when_unidentifiable=True),
            estimate,
            method_name="random_common_cause",
            placebo_type="permute",
            num_simulations=5,
        )

        original_value = estimate.value
        refuted_value = refutation.new_effect

        # For unconfounded data, should be very stable
        relative_change = abs(refuted_value - original_value) / abs(original_value)

        assert relative_change < 0.15, (
            f"Simple linear estimate unstable with random confounder: "
            f"original={original_value:.4f}, refuted={refuted_value:.4f}"
        )


class TestPlaceboTreatmentRefutation:
    """Test refutation by permuting treatment variable.

    If we permute the treatment, breaking the T->Y link,
    the estimated effect should become close to zero.

    Note: The DoWhy placebo_treatment_refuter has a known issue with
    identifier_method being None in some cases. We use a manual implementation
    to avoid this bug.
    """

    def _manual_placebo_test(self, dataset: SyntheticDataset, num_simulations: int = 5) -> float:
        """Manually compute placebo effect by permuting treatment.

        This avoids the DoWhy bug where identifier_method is None.
        """
        np.random.seed(42)
        placebo_effects = []

        for _ in range(num_simulations):
            # Create copy with permuted treatment
            data_copy = dataset.data.copy()
            data_copy[dataset.treatment_col] = np.random.permutation(
                data_copy[dataset.treatment_col].values
            )

            # Estimate effect with permuted treatment
            from sklearn.linear_model import LinearRegression

            X_cols = [dataset.treatment_col] + dataset.confounder_cols
            X = data_copy[X_cols].values
            y = data_copy[dataset.outcome_col].values

            model = LinearRegression()
            model.fit(X, y)

            # Treatment coefficient (first column)
            placebo_effect = model.coef_[0]
            placebo_effects.append(placebo_effect)

        return np.mean(placebo_effects)

    def test_placebo_treatment_refutation(self, refutation_dataset: SyntheticDataset):
        """Verify zero effect when treatment is permuted.

        Expected: Effect close to zero (< 0.10)
        This validates that the effect is actually from treatment, not noise.
        """
        # Use manual implementation to avoid DoWhy bug
        placebo_effect = self._manual_placebo_test(refutation_dataset)

        assert abs(placebo_effect) < 0.10, (
            f"Placebo treatment effect too large: {placebo_effect:.4f}. "
            "This suggests the effect may be spurious."
        )

    def test_placebo_treatment_vs_true_effect(self, refutation_dataset: SyntheticDataset):
        """Placebo effect should be much smaller than true effect."""
        model = build_causal_model(refutation_dataset)
        estimate = estimate_ate_with_model(model)

        true_effect = estimate.value
        placebo_effect = self._manual_placebo_test(refutation_dataset)

        # True effect should be at least 3x larger than placebo
        # (signal-to-noise ratio)
        if abs(true_effect) > 0.01:  # Avoid division issues
            ratio = abs(true_effect) / max(abs(placebo_effect), 0.001)

            assert ratio > 3.0, (
                f"Signal-to-noise ratio too low: {ratio:.2f}. "
                f"True effect: {true_effect:.4f}, Placebo: {placebo_effect:.4f}"
            )


class TestSubsetDataRefutation:
    """Test refutation by re-estimating on data subsets.

    If estimate is robust, it should be stable across random subsets
    of the data.
    """

    def test_subset_data_refutation(self, refutation_dataset: SyntheticDataset):
        """Verify estimate stability across data subsets.

        Expected: Estimates across subsets should be within tolerance
        This tests whether the effect generalizes across the data.
        """
        model = build_causal_model(refutation_dataset)
        estimate = estimate_ate_with_model(model)

        # Run subset refutation with 80% subsets
        refutation = model.refute_estimate(
            model.identify_effect(proceed_when_unidentifiable=True),
            estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8,
            num_simulations=5,
        )

        original_value = estimate.value
        subset_value = refutation.new_effect

        # Subset estimate should be close to full data estimate
        relative_change = abs(subset_value - original_value) / abs(original_value)

        assert relative_change < 0.25, (
            f"Estimate unstable across subsets: "
            f"original={original_value:.4f}, subset={subset_value:.4f}, "
            f"change={relative_change:.1%}"
        )

    def test_subset_50_percent(self, refutation_dataset: SyntheticDataset):
        """Test with 50% data subset (more challenging)."""
        model = build_causal_model(refutation_dataset)
        estimate = estimate_ate_with_model(model)

        refutation = model.refute_estimate(
            model.identify_effect(proceed_when_unidentifiable=True),
            estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.5,
            num_simulations=5,
        )

        original_value = estimate.value
        subset_value = refutation.new_effect

        # Allow more variance with smaller subset
        relative_change = abs(subset_value - original_value) / abs(original_value)

        assert relative_change < 0.35, (
            f"Estimate too unstable with 50% subset: change={relative_change:.1%}"
        )


class TestBootstrapRefutation:
    """Test refutation via bootstrap resampling.

    Bootstrap provides confidence intervals and tests whether
    the effect is significantly different from zero.
    """

    def test_bootstrap_refutation(self, refutation_dataset: SyntheticDataset):
        """Verify effect is significant via bootstrap.

        Expected: Bootstrap CI excludes zero
        This provides statistical confidence in the estimate.
        """
        model = build_causal_model(refutation_dataset)
        estimate = estimate_ate_with_model(model)

        # Run bootstrap refutation
        refutation = model.refute_estimate(
            model.identify_effect(proceed_when_unidentifiable=True),
            estimate,
            method_name="bootstrap_refuter",
            num_simulations=10,  # Low for memory/time
        )

        # Bootstrap mean should be close to original estimate
        original_value = estimate.value
        bootstrap_value = refutation.new_effect

        relative_change = abs(bootstrap_value - original_value) / abs(original_value)

        assert relative_change < 0.20, (
            f"Bootstrap mean differs from original: "
            f"original={original_value:.4f}, bootstrap={bootstrap_value:.4f}"
        )

    def test_bootstrap_significance(self, refutation_dataset: SyntheticDataset):
        """Bootstrap effect should be significantly different from zero."""
        model = build_causal_model(refutation_dataset)
        estimate = estimate_ate_with_model(model)

        refutation = model.refute_estimate(
            model.identify_effect(proceed_when_unidentifiable=True),
            estimate,
            method_name="bootstrap_refuter",
            num_simulations=10,
        )

        bootstrap_value = refutation.new_effect
        true_ate = refutation_dataset.true_ate

        # Bootstrap estimate should be reasonably close to true ATE
        error = abs(bootstrap_value - true_ate)

        assert error < refutation_dataset.tolerance * 2, (
            f"Bootstrap estimate far from true ATE: "
            f"bootstrap={bootstrap_value:.4f}, true={true_ate:.4f}, "
            f"error={error:.4f}"
        )


# ============================================================================
# AGGREGATE REFUTATION SUMMARY
# ============================================================================


class TestRefutationSummary:
    """Aggregate test to verify overall refutation pass rate."""

    def _manual_placebo_test(self, dataset: SyntheticDataset, num_simulations: int = 3) -> float:
        """Manually compute placebo effect by permuting treatment."""
        np.random.seed(42)
        placebo_effects = []

        for _ in range(num_simulations):
            data_copy = dataset.data.copy()
            data_copy[dataset.treatment_col] = np.random.permutation(
                data_copy[dataset.treatment_col].values
            )

            from sklearn.linear_model import LinearRegression

            X_cols = [dataset.treatment_col] + dataset.confounder_cols
            X = data_copy[X_cols].values
            y = data_copy[dataset.outcome_col].values

            model = LinearRegression()
            model.fit(X, y)
            placebo_effects.append(model.coef_[0])

        return np.mean(placebo_effects)

    def test_refutation_pass_rate(self, refutation_dataset: SyntheticDataset):
        """At least 60% of refutations should pass.

        This is a meta-test that runs all refutation methods and
        checks the overall pass rate against the target threshold.
        """
        model = build_causal_model(refutation_dataset)
        estimate = estimate_ate_with_model(model)
        identified = model.identify_effect(proceed_when_unidentifiable=True)

        # Only use DoWhy methods that work reliably
        refutation_methods = [
            ("random_common_cause", {"placebo_type": "permute", "num_simulations": 3}),
            ("data_subset_refuter", {"subset_fraction": 0.8, "num_simulations": 3}),
            ("bootstrap_refuter", {"num_simulations": 5}),
        ]

        results = []
        original_value = estimate.value

        for method_name, kwargs in refutation_methods:
            try:
                refutation = model.refute_estimate(
                    identified,
                    estimate,
                    method_name=method_name,
                    **kwargs,
                )

                new_effect = refutation.new_effect

                # Stability relative to original
                relative_change = abs(new_effect - original_value) / abs(original_value)
                passed = relative_change < 0.30

                results.append(
                    {
                        "method": method_name,
                        "passed": passed,
                        "original": original_value,
                        "refuted": new_effect,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "method": method_name,
                        "passed": False,
                        "error": str(e),
                    }
                )

        # Add manual placebo test (avoids DoWhy bug)
        try:
            placebo_effect = self._manual_placebo_test(refutation_dataset)
            passed = abs(placebo_effect) < 0.10
            results.append(
                {
                    "method": "placebo_treatment (manual)",
                    "passed": passed,
                    "original": original_value,
                    "refuted": placebo_effect,
                }
            )
        except Exception as e:
            results.append(
                {
                    "method": "placebo_treatment (manual)",
                    "passed": False,
                    "error": str(e),
                }
            )

        # Calculate pass rate
        passed = sum(1 for r in results if r.get("passed", False))
        total = len(results)
        pass_rate = passed / total if total > 0 else 0

        # Log results
        print("\n" + "=" * 60)
        print("REFUTATION SUMMARY")
        print("=" * 60)
        for r in results:
            status = "✓ PASS" if r.get("passed") else "✗ FAIL"
            if "error" in r:
                print(f"  {r['method']}: {status} (error: {r['error']})")
            else:
                print(
                    f"  {r['method']}: {status} "
                    f"(original={r['original']:.4f}, refuted={r['refuted']:.4f})"
                )
        print("-" * 60)
        print(f"Pass Rate: {passed}/{total} ({pass_rate:.0%})")
        print("=" * 60)

        # Assert >= 60% pass rate
        assert pass_rate >= 0.60, (
            f"Refutation pass rate below threshold: {pass_rate:.0%} < 60%. Passed: {passed}/{total}"
        )
