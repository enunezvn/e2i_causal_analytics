"""Unit tests for src/utils/sufficiency_resolver.py.

Covers the three-tier resolution hierarchy: user_override > computed_from_data >
literature_default. Every resolver is tested at each tier so the priority order
is locked in.
"""

from __future__ import annotations

from src.utils.sufficiency_defaults import (
    ABSOLUTE_FLOORS,
    DEFAULT_ALPHA,
    DEFAULT_MDE_CONTINUOUS_COHENS_D,
    DEFAULT_OBSERVATIONAL_INFLATION,
    DEFAULT_POWER,
    EPV_FLOORS,
    STRICTNESS_MULTIPLIERS,
)
from src.utils.sufficiency_resolver import (
    resolve_absolute_floor,
    resolve_alpha,
    resolve_epv_floor,
    resolve_observational_inflation,
    resolve_power,
    resolve_regression_ratio,
    resolve_target_mde,
    resolve_timeseries_min_n,
)


class TestResolveEpvFloor:
    def test_user_override_wins(self):
        result = resolve_epv_floor(user_config={"epv_floor": 42}, algorithm_family="linear")
        assert result.value == 42
        assert result.source == "user_override"

    def test_computed_from_algorithm_family(self):
        result = resolve_epv_floor(user_config=None, algorithm_family="tree_based")
        assert result.value == EPV_FLOORS["tree_based"]
        assert result.source == "computed_from_data"

    def test_literature_default_when_unknown_algorithm(self):
        result = resolve_epv_floor(user_config=None, algorithm_family="unknown")
        assert result.value == EPV_FLOORS["unknown"]
        assert result.source == "literature_default"

    def test_strictness_conservative_halves(self):
        moderate = resolve_epv_floor(
            user_config={"strictness_preset": "moderate"}, algorithm_family="tree_based"
        )
        conservative = resolve_epv_floor(
            user_config={"strictness_preset": "conservative"},
            algorithm_family="tree_based",
        )
        assert conservative.value == max(
            1, int(round(moderate.value * STRICTNESS_MULTIPLIERS["conservative"]))
        )

    def test_strictness_strict_doubles(self):
        moderate = resolve_epv_floor(user_config=None, algorithm_family="tree_based")
        strict = resolve_epv_floor(
            user_config={"strictness_preset": "strict"}, algorithm_family="tree_based"
        )
        assert strict.value == int(round(moderate.value * STRICTNESS_MULTIPLIERS["strict"]))

    def test_unknown_family_falls_back(self):
        result = resolve_epv_floor(user_config=None, algorithm_family="bogus_family")
        assert result.value == EPV_FLOORS["unknown"]


class TestResolveRegressionRatio:
    def test_linear_default(self):
        result = resolve_regression_ratio(user_config=None, algorithm_family="linear")
        assert result.value == 5

    def test_tree_default(self):
        result = resolve_regression_ratio(user_config=None, algorithm_family="tree_based")
        assert result.value == 10

    def test_user_override_wins(self):
        result = resolve_regression_ratio(user_config={"epv_floor": 8}, algorithm_family="linear")
        assert result.value == 8
        assert result.source == "user_override"


class TestResolveAbsoluteFloor:
    def test_user_override_wins(self):
        result = resolve_absolute_floor(
            user_config={"absolute_floor": 500},
            problem_type="binary_classification",
        )
        assert result.value == 500
        assert result.source == "user_override"

    def test_computed_when_features_and_prevalence_provided(self):
        # 30 features × 2 EPV / 0.05 prevalence = 1200; literature floor is 100
        result = resolve_absolute_floor(
            user_config=None,
            problem_type="binary_classification",
            n_features=30,
            minority_prevalence=0.05,
        )
        assert result.value == 1200
        assert result.source == "computed_from_data"

    def test_computed_falls_back_to_literature_when_higher(self):
        # 5 features × 2 / 0.5 = 20; literature floor is 100 → use 100
        result = resolve_absolute_floor(
            user_config=None,
            problem_type="binary_classification",
            n_features=5,
            minority_prevalence=0.5,
        )
        assert result.value == ABSOLUTE_FLOORS["binary_classification"]

    def test_literature_default_when_no_data(self):
        result = resolve_absolute_floor(user_config=None, problem_type="regression")
        assert result.value == ABSOLUTE_FLOORS["regression"]
        assert result.source == "literature_default"

    def test_unknown_problem_type_uses_safe_default(self):
        result = resolve_absolute_floor(user_config=None, problem_type="not_a_real_type")
        assert result.value == 100  # safe fallback in code


class TestResolveObservationalInflation:
    def test_user_override_wins(self):
        result = resolve_observational_inflation(user_config={"observational_inflation": 3.5})
        assert result.value == 3.5
        assert result.source == "user_override"

    def test_computed_from_observed_overlap(self):
        result = resolve_observational_inflation(user_config=None, observed_overlap=0.5)
        assert result.value == 2.0  # 1 / 0.5
        assert result.source == "computed_from_data"

    def test_literature_default_when_no_overlap(self):
        result = resolve_observational_inflation(user_config=None)
        assert result.value == DEFAULT_OBSERVATIONAL_INFLATION
        assert result.source == "literature_default"

    def test_invalid_overlap_falls_back(self):
        result = resolve_observational_inflation(user_config=None, observed_overlap=1.5)
        assert result.source == "literature_default"


class TestResolveTargetMde:
    def test_user_override_wins(self):
        result = resolve_target_mde(
            user_config={"target_mde": 0.15},
            outcome_type="binary",
            baseline_rate=0.30,
        )
        assert result.value == 0.15
        assert result.source == "user_override"

    def test_continuous_data_driven(self):
        result = resolve_target_mde(user_config=None, outcome_type="continuous", sigma_outcome=2.0)
        assert result.value == 1.0  # 0.5 * 2.0
        assert result.source == "computed_from_data"

    def test_continuous_literature_default_when_no_sigma(self):
        result = resolve_target_mde(user_config=None, outcome_type="continuous")
        assert result.value == DEFAULT_MDE_CONTINUOUS_COHENS_D
        assert result.source == "literature_default"

    def test_binary_floor_when_baseline_too_low(self):
        # baseline=0.10 → 0.20 * 0.10 = 0.02, below 0.05 floor → 0.05
        result = resolve_target_mde(user_config=None, outcome_type="binary", baseline_rate=0.10)
        assert result.value == 0.05
        assert result.source == "computed_from_data"

    def test_binary_relative_when_baseline_high(self):
        # baseline=0.40 → 0.20 * 0.40 = 0.08, above 0.05 floor → 0.08
        result = resolve_target_mde(user_config=None, outcome_type="binary", baseline_rate=0.40)
        assert abs(result.value - 0.08) < 1e-9
        assert result.source == "computed_from_data"

    def test_time_to_event_uses_literature(self):
        result = resolve_target_mde(user_config=None, outcome_type="time_to_event")
        assert result.source == "literature_default"

    # F5 (PR #462 hotfix): invalid target_mde overrides must fall back to
    # the data-driven / literature default with a WARN, not silently pass
    # through. SufficiencyConfig also rejects them at construction time
    # (see test_sufficiency_schemas.py::TestF5TargetMdeValidation), but
    # the resolver also runs on raw dicts that may not have gone through
    # schema validation.
    import math as _math

    import pytest

    @pytest.mark.parametrize(
        "bad_override",
        [-0.5, -0.01, 0.0, 1.0, 1.5, 5.0, _math.nan, _math.inf, -_math.inf, "not_a_number"],
    )
    def test_invalid_override_falls_back_to_default(self, bad_override):
        result = resolve_target_mde(
            user_config={"target_mde": bad_override},
            outcome_type="binary",
            baseline_rate=0.30,
        )
        assert result.source != "user_override", (
            f"resolver accepted invalid override {bad_override!r} as user_override; "
            f"expected fallback to data-driven / literature default"
        )

    def test_valid_in_bounds_override_accepted(self):
        # 0.05 is in-bounds → user_override wins.
        result = resolve_target_mde(
            user_config={"target_mde": 0.05},
            outcome_type="binary",
            baseline_rate=0.30,
        )
        assert result.value == 0.05
        assert result.source == "user_override"


class TestResolveAlphaAndPower:
    def test_alpha_user_override(self):
        result = resolve_alpha(user_config={"alpha": 0.01})
        assert result.value == 0.01
        assert result.source == "user_override"

    def test_alpha_default(self):
        result = resolve_alpha(user_config=None)
        assert result.value == DEFAULT_ALPHA
        assert result.source == "literature_default"

    def test_power_user_override(self):
        result = resolve_power(user_config={"power_target": 0.90})
        assert result.value == 0.90
        assert result.source == "user_override"

    def test_power_default(self):
        result = resolve_power(user_config=None)
        assert result.value == DEFAULT_POWER
        assert result.source == "literature_default"


class TestResolveTimeseriesMinN:
    def test_user_absolute_floor_override(self):
        result = resolve_timeseries_min_n(user_config={"absolute_floor": 500}, seasonal_period=12)
        assert result.value == 500
        assert result.source == "user_override"

    def test_hyndman_formula(self):
        # 2 cycles × 12 (monthly) × 1.0 noise + 0 features + 1 = 25
        result = resolve_timeseries_min_n(user_config=None, seasonal_period=12, n_features=0)
        # Should be max(literature_floor=100, 25) = 100
        assert result.value == max(100, 25)

    def test_high_seasonal_period_exceeds_floor(self):
        # 2 cycles × 52 (weekly) × 1.0 + 5 features + 1 = 110
        result = resolve_timeseries_min_n(user_config=None, seasonal_period=52, n_features=5)
        assert result.value == 110

    def test_noise_inflates_floor(self):
        clean = resolve_timeseries_min_n(
            user_config=None, seasonal_period=52, n_features=5, cv_outcome=0.0
        )
        noisy = resolve_timeseries_min_n(
            user_config=None, seasonal_period=52, n_features=5, cv_outcome=1.0
        )
        assert noisy.value > clean.value

    def test_unknown_period_falls_back_to_floor(self):
        result = resolve_timeseries_min_n(user_config=None, seasonal_period=None)
        assert result.value == ABSOLUTE_FLOORS["time_series"]
        assert result.source == "literature_default"


class TestAuditFields:
    def test_resolution_includes_citation(self):
        result = resolve_epv_floor(user_config=None, algorithm_family="linear")
        assert result.citation
        assert len(result.citation) > 0

    def test_resolution_includes_inputs(self):
        result = resolve_observational_inflation(user_config=None, observed_overlap=0.5)
        assert "observed_overlap" in result.inputs
        assert result.inputs["observed_overlap"] == 0.5

    def test_user_override_citation_points_to_scope_spec(self):
        result = resolve_alpha(user_config={"alpha": 0.01})
        assert "scope_spec.sufficiency" in result.citation
