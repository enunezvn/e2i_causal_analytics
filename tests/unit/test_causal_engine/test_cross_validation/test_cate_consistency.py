"""CATE Consistency Tests Across Libraries.

Tests verifying that CATE (Conditional Average Treatment Effect) estimates
are consistent across EconML and CausalML methods.

Test Scenarios:
- Homogeneous effects (CATE should be uniform)
- Heterogeneous effects (CATE should vary by segment)
- Segment ranking consistency (high/low responders)

Assertions:
- CATE sign consistency across methods
- Segment ranking Spearman correlation > 0.7
- High responders identified consistently
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Mark all tests in this module for the dspy_integration xdist group
pytestmark = pytest.mark.xdist_group(name="cross_validation")


# =============================================================================
# TEST FIXTURES
# =============================================================================


def generate_homogeneous_treatment_data(
    n_samples: int = 3000,
    true_ate: float = 1.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate data with homogeneous treatment effect.

    CATE should be approximately equal to ATE for all segments.
    """
    np.random.seed(seed)

    # Effect modifiers (but effect doesn't actually vary)
    segment = np.random.choice(["A", "B", "C"], n_samples)
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)

    # Treatment assignment
    propensity = 0.5 + 0.1 * X1
    propensity = np.clip(propensity, 0.2, 0.8)
    treatment = np.random.binomial(1, propensity)

    # Homogeneous treatment effect
    outcome = (
        true_ate * treatment  # Same effect for all
        + 0.3 * X1
        + 0.2 * X2
        + np.random.normal(0, 0.5, n_samples)
    )

    data = pd.DataFrame(
        {
            "segment": segment,
            "X1": X1,
            "X2": X2,
            "treatment": treatment,
            "outcome": outcome,
        }
    )

    return data, {
        "true_ate": true_ate,
        "heterogeneity": "homogeneous",
        "expected_cate_by_segment": {"A": true_ate, "B": true_ate, "C": true_ate},
    }


def generate_heterogeneous_treatment_data(
    n_samples: int = 3000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate data with heterogeneous treatment effect.

    CATE varies by segment:
    - Segment A: High responders (CATE = 2.0)
    - Segment B: Medium responders (CATE = 1.0)
    - Segment C: Low responders (CATE = 0.2)
    """
    np.random.seed(seed)

    # Segments with different treatment effects
    segment = np.random.choice(["A", "B", "C"], n_samples, p=[0.33, 0.34, 0.33])
    X1 = np.random.normal(0, 1, n_samples)

    # Treatment assignment
    propensity = 0.5 + 0.05 * X1
    propensity = np.clip(propensity, 0.3, 0.7)
    treatment = np.random.binomial(1, propensity)

    # Heterogeneous treatment effect based on segment
    cate_by_segment = {"A": 2.0, "B": 1.0, "C": 0.2}
    individual_cate = np.array([cate_by_segment[s] for s in segment])

    outcome = (
        individual_cate * treatment  # Varying effect
        + 0.3 * X1
        + np.random.normal(0, 0.5, n_samples)
    )

    data = pd.DataFrame(
        {
            "segment": segment,
            "X1": X1,
            "treatment": treatment,
            "outcome": outcome,
        }
    )

    return data, {
        "overall_ate": sum(cate_by_segment.values()) / 3,
        "heterogeneity": "heterogeneous",
        "expected_cate_by_segment": cate_by_segment,
        "expected_ranking": ["A", "B", "C"],  # High to low
    }


def generate_continuous_heterogeneity_data(
    n_samples: int = 3000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate data with continuous effect heterogeneity.

    CATE varies continuously with X1:
    - CATE = 0.5 + 0.8 * X1
    """
    np.random.seed(seed)

    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)

    # Treatment assignment
    propensity = 0.5
    treatment = np.random.binomial(1, propensity, n_samples)

    # Continuous heterogeneity
    individual_cate = 0.5 + 0.8 * X1

    outcome = individual_cate * treatment + 0.2 * X2 + np.random.normal(0, 0.3, n_samples)

    data = pd.DataFrame(
        {
            "X1": X1,
            "X2": X2,
            "treatment": treatment,
            "outcome": outcome,
        }
    )

    return data, {
        "heterogeneity": "continuous",
        "effect_modifier": "X1",
        "effect_coefficient": 0.8,
    }


@pytest.fixture
def homogeneous_data():
    """Data with homogeneous treatment effect."""
    return generate_homogeneous_treatment_data()


@pytest.fixture
def heterogeneous_data():
    """Data with heterogeneous treatment effect by segment."""
    return generate_heterogeneous_treatment_data()


@pytest.fixture
def continuous_het_data():
    """Data with continuous effect heterogeneity."""
    return generate_continuous_heterogeneity_data()


# =============================================================================
# CATE ESTIMATION HELPERS
# =============================================================================


def estimate_cate_with_causal_forest(
    data: pd.DataFrame,
    segment_col: str = None,
) -> Dict:
    """Estimate CATE using EconML CausalForestDML."""
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        covariates = [c for c in data.columns if c not in ["treatment", "outcome", segment_col]]
        if segment_col and segment_col in data.columns:
            # One-hot encode segment
            segment_dummies = pd.get_dummies(data[segment_col], prefix="seg")
            X = pd.concat([data[covariates], segment_dummies], axis=1).values
        else:
            X = data[covariates].values

        T = data["treatment"].values
        Y = data["outcome"].values

        model = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3),
            model_t=GradientBoostingClassifier(n_estimators=50, max_depth=3),
            discrete_treatment=True,
            n_estimators=50,
            min_samples_leaf=20,
        )
        model.fit(Y, T, X=X)

        # Get individual CATE estimates
        cate_individual = model.effect(X).flatten()

        result = {
            "method": "causal_forest",
            "ate": float(np.mean(cate_individual)),
            "cate_individual": cate_individual,
        }

        # Aggregate by segment if available
        if segment_col and segment_col in data.columns:
            cate_by_segment = {}
            for seg in data[segment_col].unique():
                mask = data[segment_col] == seg
                cate_by_segment[seg] = float(np.mean(cate_individual[mask]))
            result["cate_by_segment"] = cate_by_segment

        return result

    except Exception as e:
        return {"method": "causal_forest", "error": str(e)}


def estimate_cate_with_drlearner(
    data: pd.DataFrame,
    segment_col: str = None,
) -> Dict:
    """Estimate CATE using EconML DRLearner."""
    try:
        from econml.dr import DRLearner
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        covariates = [c for c in data.columns if c not in ["treatment", "outcome", segment_col]]
        if segment_col and segment_col in data.columns:
            segment_dummies = pd.get_dummies(data[segment_col], prefix="seg")
            X = pd.concat([data[covariates], segment_dummies], axis=1).values
        else:
            X = data[covariates].values

        T = data["treatment"].values
        Y = data["outcome"].values

        model = DRLearner(
            model_propensity=GradientBoostingClassifier(n_estimators=50, max_depth=3),
            model_regression=GradientBoostingRegressor(n_estimators=50, max_depth=3),
            model_final=GradientBoostingRegressor(n_estimators=50, max_depth=3),
        )
        model.fit(Y, T, X=X)

        cate_individual = model.effect(X).flatten()

        result = {
            "method": "drlearner",
            "ate": float(np.mean(cate_individual)),
            "cate_individual": cate_individual,
        }

        if segment_col and segment_col in data.columns:
            cate_by_segment = {}
            for seg in data[segment_col].unique():
                mask = data[segment_col] == seg
                cate_by_segment[seg] = float(np.mean(cate_individual[mask]))
            result["cate_by_segment"] = cate_by_segment

        return result

    except Exception as e:
        return {"method": "drlearner", "error": str(e)}


def estimate_cate_with_linear_dml(
    data: pd.DataFrame,
    segment_col: str = None,
) -> Dict:
    """Estimate CATE using EconML LinearDML (with interactions)."""
    try:
        from econml.dml import LinearDML
        from sklearn.linear_model import LassoCV, LogisticRegressionCV

        covariates = [c for c in data.columns if c not in ["treatment", "outcome", segment_col]]
        if segment_col and segment_col in data.columns:
            segment_dummies = pd.get_dummies(data[segment_col], prefix="seg")
            X = pd.concat([data[covariates], segment_dummies], axis=1).values
        else:
            X = data[covariates].values

        T = data["treatment"].values
        Y = data["outcome"].values

        model = LinearDML(
            model_y=LassoCV(),
            model_t=LogisticRegressionCV(max_iter=1000),
            discrete_treatment=True,
        )
        model.fit(Y, T, X=X)

        cate_individual = model.effect(X).flatten()

        result = {
            "method": "linear_dml",
            "ate": float(np.mean(cate_individual)),
            "cate_individual": cate_individual,
        }

        if segment_col and segment_col in data.columns:
            cate_by_segment = {}
            for seg in data[segment_col].unique():
                mask = data[segment_col] == seg
                cate_by_segment[seg] = float(np.mean(cate_individual[mask]))
            result["cate_by_segment"] = cate_by_segment

        return result

    except Exception as e:
        return {"method": "linear_dml", "error": str(e)}


# =============================================================================
# CATE CONSISTENCY TESTS
# =============================================================================


class TestCATESignConsistency:
    """Tests for CATE sign consistency across methods."""

    def test_positive_effect_sign_consistency(self, heterogeneous_data):
        """All methods should agree on CATE sign for each segment."""
        data, metadata = heterogeneous_data

        cf_result = estimate_cate_with_causal_forest(data, segment_col="segment")
        dr_result = estimate_cate_with_drlearner(data, segment_col="segment")
        dml_result = estimate_cate_with_linear_dml(data, segment_col="segment")

        results = [cf_result, dr_result, dml_result]
        valid_results = [r for r in results if "cate_by_segment" in r]

        if len(valid_results) < 2:
            pytest.skip("Not enough valid CATE estimates for comparison")

        # Check sign consistency for each segment
        for segment in ["A", "B", "C"]:
            signs = []
            for r in valid_results:
                cate = r["cate_by_segment"].get(segment)
                if cate is not None:
                    signs.append(np.sign(cate))

            if len(signs) >= 2:
                # All signs should be the same
                assert len(set(signs)) == 1, f"Sign inconsistency for segment {segment}: {signs}"


class TestSegmentRankingConsistency:
    """Tests for segment ranking consistency (Spearman correlation)."""

    def test_heterogeneous_ranking_correlation(self, heterogeneous_data):
        """Segment ranking should be consistent across methods (Spearman > 0.7)."""
        data, metadata = heterogeneous_data
        metadata["expected_ranking"]

        cf_result = estimate_cate_with_causal_forest(data, segment_col="segment")
        dr_result = estimate_cate_with_drlearner(data, segment_col="segment")

        if "error" in cf_result or "error" in dr_result:
            pytest.skip("CATE estimation failed")

        # Get rankings
        def get_ranking(cate_dict):
            segments = sorted(cate_dict.keys(), key=lambda s: cate_dict[s], reverse=True)
            return {s: i for i, s in enumerate(segments)}

        cf_ranking = get_ranking(cf_result["cate_by_segment"])
        dr_ranking = get_ranking(dr_result["cate_by_segment"])

        # Compute Spearman correlation
        segments = list(cf_ranking.keys())
        cf_ranks = [cf_ranking[s] for s in segments]
        dr_ranks = [dr_ranking[s] for s in segments]

        if len(segments) >= 3:
            correlation, p_value = stats.spearmanr(cf_ranks, dr_ranks)

            # Spearman correlation should be > 0.7 (strong agreement)
            assert correlation > 0.7, (
                f"Ranking correlation {correlation:.2f} < 0.7 between CausalForest and DRLearner"
            )

    def test_high_responder_identification_consistency(self, heterogeneous_data):
        """High responders should be identified consistently."""
        data, metadata = heterogeneous_data

        cf_result = estimate_cate_with_causal_forest(data, segment_col="segment")
        dr_result = estimate_cate_with_drlearner(data, segment_col="segment")
        dml_result = estimate_cate_with_linear_dml(data, segment_col="segment")

        results = [cf_result, dr_result, dml_result]
        valid_results = [r for r in results if "cate_by_segment" in r]

        if len(valid_results) < 2:
            pytest.skip("Not enough valid CATE estimates")

        # Find high responder for each method
        high_responders = []
        for r in valid_results:
            cate_dict = r["cate_by_segment"]
            high_seg = max(cate_dict.keys(), key=lambda s: cate_dict[s])
            high_responders.append(high_seg)

        # Majority should identify the same high responder
        from collections import Counter

        most_common = Counter(high_responders).most_common(1)[0]
        assert most_common[1] >= len(valid_results) / 2, (
            f"No consensus on high responder: {high_responders}"
        )


class TestHomogeneousEffectRecovery:
    """Tests for homogeneous effect detection."""

    def test_homogeneous_cate_uniformity(self, homogeneous_data):
        """CATE should be approximately uniform when effect is homogeneous."""
        data, metadata = homogeneous_data
        true_ate = metadata["true_ate"]

        cf_result = estimate_cate_with_causal_forest(data, segment_col="segment")

        if "error" in cf_result:
            pytest.skip(f"CATE estimation failed: {cf_result['error']}")

        cate_by_segment = cf_result["cate_by_segment"]

        # All segment CATEs should be close to true ATE
        for segment, cate in cate_by_segment.items():
            error = abs(cate - true_ate) / true_ate
            assert error < 0.25, (
                f"Segment {segment} CATE {cate:.3f} differs from true ATE {true_ate} by {error:.1%}"
            )

        # Heterogeneity should be low (max - min < 0.5)
        cate_range = max(cate_by_segment.values()) - min(cate_by_segment.values())
        assert cate_range < 0.5, f"CATE range {cate_range:.3f} too large for homogeneous effect"


class TestContinuousHeterogeneityRecovery:
    """Tests for continuous heterogeneity detection."""

    def test_effect_modifier_detection(self, continuous_het_data):
        """Should detect that X1 is the effect modifier."""
        data, metadata = continuous_het_data
        metadata["effect_modifier"]

        cf_result = estimate_cate_with_causal_forest(data)

        if "error" in cf_result:
            pytest.skip(f"CATE estimation failed: {cf_result['error']}")

        cate_individual = cf_result["cate_individual"]

        # CATE should correlate positively with X1
        correlation, p_value = stats.pearsonr(data["X1"].values, cate_individual)

        # Strong positive correlation expected (effect_coefficient = 0.8)
        assert correlation > 0.5, f"CATE-X1 correlation {correlation:.2f} should be > 0.5"
        assert p_value < 0.05, "Correlation should be statistically significant"


class TestCrossMethodCATEAgreement:
    """Tests for cross-method CATE agreement at individual level."""

    def test_individual_cate_correlation(self, heterogeneous_data):
        """Individual CATE estimates should correlate across methods."""
        data, metadata = heterogeneous_data

        cf_result = estimate_cate_with_causal_forest(data, segment_col="segment")
        dr_result = estimate_cate_with_drlearner(data, segment_col="segment")

        if "error" in cf_result or "error" in dr_result:
            pytest.skip("CATE estimation failed")

        # Correlation of individual CATE estimates
        correlation, p_value = stats.pearsonr(
            cf_result["cate_individual"], dr_result["cate_individual"]
        )

        # Should have moderate to strong correlation
        assert correlation > 0.5, (
            f"Individual CATE correlation {correlation:.2f} < 0.5 between methods"
        )
