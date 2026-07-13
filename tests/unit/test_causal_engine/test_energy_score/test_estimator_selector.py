"""
Tests for src/causal_engine/energy_score/estimator_selector.py

Covers:
- EstimatorType enum
- SelectionStrategy enum
- EstimatorResult dataclass
- SelectionResult dataclass
- EstimatorConfig dataclass
- EstimatorSelectorConfig dataclass
- BaseEstimatorWrapper and wrappers
- EstimatorSelector class
- select_best_estimator convenience function
"""

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.energy_score.estimator_selector import (
    ESTIMATOR_WRAPPERS,
    CausalForestWrapper,
    DRLearnerWrapper,
    EstimatorConfig,
    EstimatorResult,
    EstimatorSelector,
    EstimatorSelectorConfig,
    EstimatorType,
    LinearDMLWrapper,
    OLSWrapper,
    SelectionResult,
    SelectionStrategy,
    select_best_estimator,
)
from src.causal_engine.energy_score.score_calculator import (
    EnergyScoreConfig,
    EnergyScoreResult,
)

# =============================================================================
# EstimatorType Enum Tests
# =============================================================================


class TestEstimatorType:
    """Tests for EstimatorType enum."""

    def test_all_estimator_types_exist(self):
        """Test all estimator types are defined."""
        expected = [
            "CAUSAL_FOREST",
            "LINEAR_DML",
            "DML_LEARNER",
            "DRLEARNER",
            "ORTHO_FOREST",
            "S_LEARNER",
            "T_LEARNER",
            "X_LEARNER",
            "OLS",
        ]
        actual = [t.name for t in EstimatorType]
        assert sorted(expected) == sorted(actual)

    def test_estimator_type_values_are_strings(self):
        """Test estimator type values are strings."""
        for est_type in EstimatorType:
            assert isinstance(est_type.value, str)

    def test_causal_forest_type(self):
        """Test CAUSAL_FOREST type."""
        assert EstimatorType.CAUSAL_FOREST.value == "causal_forest"

    def test_ols_type(self):
        """Test OLS type."""
        assert EstimatorType.OLS.value == "ols"

    def test_meta_learner_types(self):
        """Test meta-learner types."""
        assert EstimatorType.S_LEARNER.value == "s_learner"
        assert EstimatorType.T_LEARNER.value == "t_learner"
        assert EstimatorType.X_LEARNER.value == "x_learner"

    def test_estimator_type_is_str_enum(self):
        """Test EstimatorType inherits from str."""
        assert issubclass(EstimatorType, str)


# =============================================================================
# SelectionStrategy Enum Tests
# =============================================================================


class TestSelectionStrategy:
    """Tests for SelectionStrategy enum."""

    def test_all_strategies_exist(self):
        """Test all selection strategies are defined."""
        expected = ["FIRST_SUCCESS", "BEST_ENERGY_SCORE", "ENSEMBLE"]
        actual = [s.name for s in SelectionStrategy]
        assert sorted(expected) == sorted(actual)

    def test_first_success_strategy(self):
        """Test FIRST_SUCCESS strategy."""
        assert SelectionStrategy.FIRST_SUCCESS.value == "first_success"

    def test_best_energy_score_strategy(self):
        """Test BEST_ENERGY_SCORE strategy."""
        assert SelectionStrategy.BEST_ENERGY_SCORE.value == "best_energy"

    def test_ensemble_strategy(self):
        """Test ENSEMBLE strategy."""
        assert SelectionStrategy.ENSEMBLE.value == "ensemble"

    def test_selection_strategy_is_str_enum(self):
        """Test SelectionStrategy inherits from str."""
        assert issubclass(SelectionStrategy, str)


# =============================================================================
# EstimatorResult Dataclass Tests
# =============================================================================


class TestEstimatorResult:
    """Tests for EstimatorResult dataclass."""

    def test_create_successful_result(self):
        """Test creating a successful EstimatorResult."""
        result = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=True,
            ate=2.5,
            cate=np.array([2.5, 2.5, 2.5]),
            ate_std=0.1,
            ate_ci_lower=2.3,
            ate_ci_upper=2.7,
            estimation_time_ms=50.0,
        )

        assert result.success is True
        assert result.ate == 2.5
        assert result.estimator_type == EstimatorType.OLS

    def test_create_failed_result(self):
        """Test creating a failed EstimatorResult."""
        result = EstimatorResult(
            estimator_type=EstimatorType.CAUSAL_FOREST,
            success=False,
            error_message="Insufficient samples",
            error_type="ValueError",
            estimation_time_ms=10.0,
        )

        assert result.success is False
        assert result.error_message == "Insufficient samples"
        assert result.ate is None

    def test_energy_score_property_with_result(self):
        """Test energy_score property when energy_score_result is set."""
        energy_result = EnergyScoreResult(
            estimator_name="OLS",
            energy_score=0.35,
            treatment_balance_score=0.30,
            outcome_fit_score=0.40,
            propensity_calibration=0.25,
            n_samples=100,
            n_treated=50,
            n_control=50,
            computation_time_ms=10.0,
        )

        result = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=True,
            ate=2.0,
            energy_score_result=energy_result,
        )

        assert result.energy_score == 0.35

    def test_energy_score_property_without_result(self):
        """Test energy_score property returns infinity when not set."""
        result = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=True,
            ate=2.0,
        )

        assert result.energy_score == float("inf")

    def test_to_dict(self):
        """Test to_dict method."""
        result = EstimatorResult(
            estimator_type=EstimatorType.LINEAR_DML,
            success=True,
            ate=1.5,
            ate_std=0.2,
            ate_ci_lower=1.1,
            ate_ci_upper=1.9,
            estimation_time_ms=100.0,
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["estimator_type"] == "linear_dml"
        assert d["success"] is True
        assert d["ate"] == 1.5

    def test_to_dict_with_error(self):
        """Test to_dict with failed result."""
        result = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=False,
            error_message="Test error",
        )

        d = result.to_dict()

        assert d["success"] is False
        assert d["error_message"] == "Test error"
        assert d["energy_score"] is None


# =============================================================================
# SelectionResult Dataclass Tests
# =============================================================================


class TestSelectionResult:
    """Tests for SelectionResult dataclass."""

    def test_create_selection_result(self):
        """Test creating a SelectionResult."""
        selected = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=True,
            ate=2.0,
        )

        result = SelectionResult(
            selected=selected,
            selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
            selection_reason="Lowest energy score",
            total_time_ms=500.0,
            energy_scores={"ols": 0.3, "linear_dml": 0.4},
            energy_score_gap=0.1,
        )

        assert result.selected.estimator_type == EstimatorType.OLS
        assert result.selection_strategy == SelectionStrategy.BEST_ENERGY_SCORE
        assert result.energy_score_gap == 0.1

    def test_selection_result_with_all_results(self):
        """Test SelectionResult with all evaluated results."""
        selected = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=True,
            ate=2.0,
        )

        other = EstimatorResult(
            estimator_type=EstimatorType.LINEAR_DML,
            success=False,
            error_message="Failed",
        )

        result = SelectionResult(
            selected=selected,
            selection_strategy=SelectionStrategy.FIRST_SUCCESS,
            all_results=[selected, other],
        )

        assert len(result.all_results) == 2

    def test_to_dict(self):
        """Test to_dict method."""
        selected = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=True,
            ate=2.0,
        )

        result = SelectionResult(
            selected=selected,
            selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
            all_results=[selected],
            total_time_ms=100.0,
            energy_scores={"ols": 0.3},
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["selected_estimator"] == "ols"
        assert d["selection_strategy"] == "best_energy"
        assert d["n_estimators_evaluated"] == 1
        assert d["n_estimators_succeeded"] == 1


# =============================================================================
# EstimatorConfig Dataclass Tests
# =============================================================================


class TestEstimatorConfig:
    """Tests for EstimatorConfig dataclass."""

    def test_default_config(self):
        """Test default EstimatorConfig."""
        config = EstimatorConfig(estimator_type=EstimatorType.OLS)

        assert config.enabled is True
        assert config.priority == 1
        assert config.timeout_seconds == 30.0
        assert config.params == {}

    def test_custom_config(self):
        """Test custom EstimatorConfig."""
        config = EstimatorConfig(
            estimator_type=EstimatorType.CAUSAL_FOREST,
            enabled=True,
            priority=2,
            params={"n_estimators": 200},
            timeout_seconds=60.0,
        )

        assert config.estimator_type == EstimatorType.CAUSAL_FOREST
        assert config.priority == 2
        assert config.params["n_estimators"] == 200

    def test_disabled_config(self):
        """Test disabled estimator config."""
        config = EstimatorConfig(
            estimator_type=EstimatorType.LINEAR_DML,
            enabled=False,
        )

        assert config.enabled is False


# =============================================================================
# EstimatorSelectorConfig Dataclass Tests
# =============================================================================


class TestEstimatorSelectorConfig:
    """Tests for EstimatorSelectorConfig dataclass."""

    def test_default_config(self):
        """Test default selector configuration."""
        config = EstimatorSelectorConfig()

        assert config.strategy == SelectionStrategy.BEST_ENERGY_SCORE
        assert len(config.estimators) > 0
        assert config.fallback_on_all_fail is True
        assert config.fallback_estimator == EstimatorType.OLS

    def test_default_estimator_chain(self):
        """Test default estimator chain order."""
        config = EstimatorSelectorConfig()

        types = [e.estimator_type for e in config.estimators]
        assert EstimatorType.CAUSAL_FOREST in types
        assert EstimatorType.OLS in types

    def test_custom_strategy(self):
        """Test custom selection strategy."""
        config = EstimatorSelectorConfig(
            strategy=SelectionStrategy.FIRST_SUCCESS,
        )

        assert config.strategy == SelectionStrategy.FIRST_SUCCESS

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = EstimatorSelectorConfig(
            min_energy_score_gap=0.10,
            max_acceptable_energy_score=0.7,
        )

        assert config.min_energy_score_gap == 0.10
        assert config.max_acceptable_energy_score == 0.7


# =============================================================================
# ESTIMATOR_WRAPPERS Factory Tests
# =============================================================================


class TestEstimatorWrappersFactory:
    """Tests for ESTIMATOR_WRAPPERS factory dictionary."""

    def test_factory_contains_ols(self):
        """Test factory contains OLS wrapper."""
        assert EstimatorType.OLS in ESTIMATOR_WRAPPERS
        assert ESTIMATOR_WRAPPERS[EstimatorType.OLS] == OLSWrapper

    def test_factory_contains_meta_learners(self):
        """Test factory contains meta-learner wrappers."""
        assert EstimatorType.S_LEARNER in ESTIMATOR_WRAPPERS
        assert EstimatorType.T_LEARNER in ESTIMATOR_WRAPPERS
        assert EstimatorType.X_LEARNER in ESTIMATOR_WRAPPERS

    def test_factory_wrapper_count(self):
        """Test factory contains expected number of wrappers."""
        # At least 8 wrappers should be registered
        assert len(ESTIMATOR_WRAPPERS) >= 8


# =============================================================================
# OLSWrapper Tests
# =============================================================================


class TestOLSWrapper:
    """Tests for OLSWrapper class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for OLS testing."""
        np.random.seed(42)
        n = 150

        covariates = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n),
                "x2": np.random.normal(0, 1, n),
            }
        )

        treatment = np.random.binomial(1, 0.5, n)
        true_effect = 2.0
        outcome = (
            1.0 + 0.5 * covariates["x1"] + true_effect * treatment + np.random.normal(0, 0.5, n)
        )

        return {
            "treatment": treatment,
            "outcome": outcome,
            "covariates": covariates,
            "true_effect": true_effect,
        }

    def test_wrapper_estimator_type(self):
        """Test OLSWrapper returns correct estimator type."""
        config = EstimatorConfig(estimator_type=EstimatorType.OLS)
        wrapper = OLSWrapper(config)

        assert wrapper.estimator_type == EstimatorType.OLS

    def test_wrapper_fit_returns_result(self, sample_data):
        """Test OLSWrapper fit returns EstimatorResult."""
        config = EstimatorConfig(estimator_type=EstimatorType.OLS)
        wrapper = OLSWrapper(config)

        result = wrapper.fit(
            treatment=sample_data["treatment"],
            outcome=sample_data["outcome"],
            covariates=sample_data["covariates"],
        )

        assert isinstance(result, EstimatorResult)
        assert result.estimator_type == EstimatorType.OLS

    def test_wrapper_fit_success(self, sample_data):
        """Test OLSWrapper fit succeeds."""
        config = EstimatorConfig(estimator_type=EstimatorType.OLS)
        wrapper = OLSWrapper(config)

        result = wrapper.fit(
            treatment=sample_data["treatment"],
            outcome=sample_data["outcome"],
            covariates=sample_data["covariates"],
        )

        assert result.success is True
        assert result.ate is not None

    def test_wrapper_fit_estimates_ate(self, sample_data):
        """Test OLSWrapper estimates reasonable ATE."""
        config = EstimatorConfig(estimator_type=EstimatorType.OLS)
        wrapper = OLSWrapper(config)

        result = wrapper.fit(
            treatment=sample_data["treatment"],
            outcome=sample_data["outcome"],
            covariates=sample_data["covariates"],
        )

        # OLS should estimate close to true effect
        assert 1.0 < result.ate < 3.0  # True effect is 2.0

    def test_wrapper_fit_returns_cate(self, sample_data):
        """Test OLSWrapper returns CATE array (constant for OLS)."""
        config = EstimatorConfig(estimator_type=EstimatorType.OLS)
        wrapper = OLSWrapper(config)

        result = wrapper.fit(
            treatment=sample_data["treatment"],
            outcome=sample_data["outcome"],
            covariates=sample_data["covariates"],
        )

        assert result.cate is not None
        assert len(result.cate) == len(sample_data["treatment"])
        # OLS CATE should be constant (all equal to ATE)
        assert np.allclose(result.cate, result.ate)

    def test_wrapper_fit_returns_confidence_interval(self, sample_data):
        """Test OLSWrapper returns confidence interval."""
        config = EstimatorConfig(estimator_type=EstimatorType.OLS)
        wrapper = OLSWrapper(config)

        result = wrapper.fit(
            treatment=sample_data["treatment"],
            outcome=sample_data["outcome"],
            covariates=sample_data["covariates"],
        )

        assert result.ate_ci_lower is not None
        assert result.ate_ci_upper is not None
        assert result.ate_ci_lower < result.ate < result.ate_ci_upper

    def test_wrapper_fit_returns_propensity_scores(self, sample_data):
        """Test OLSWrapper returns propensity scores."""
        config = EstimatorConfig(estimator_type=EstimatorType.OLS)
        wrapper = OLSWrapper(config)

        result = wrapper.fit(
            treatment=sample_data["treatment"],
            outcome=sample_data["outcome"],
            covariates=sample_data["covariates"],
        )

        assert result.propensity_scores is not None
        assert len(result.propensity_scores) == len(sample_data["treatment"])
        assert all(0 < p < 1 for p in result.propensity_scores)

    def test_wrapper_fit_records_time(self, sample_data):
        """Test OLSWrapper records estimation time."""
        config = EstimatorConfig(estimator_type=EstimatorType.OLS)
        wrapper = OLSWrapper(config)

        result = wrapper.fit(
            treatment=sample_data["treatment"],
            outcome=sample_data["outcome"],
            covariates=sample_data["covariates"],
        )

        assert result.estimation_time_ms > 0


# =============================================================================
# EstimatorSelector Tests
# =============================================================================


class TestEstimatorSelector:
    """Tests for EstimatorSelector class."""

    @pytest.fixture
    def simple_data(self):
        """Create simple data for selector testing."""
        np.random.seed(42)
        n = 100

        covariates = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n),
                "x2": np.random.normal(0, 1, n),
            }
        )

        treatment = np.random.binomial(1, 0.5, n)
        outcome = 1.0 + 2.0 * treatment + np.random.normal(0, 0.5, n)

        return {
            "treatment": treatment,
            "outcome": outcome,
            "covariates": covariates,
        }

    def test_selector_init_default_config(self):
        """Test selector initializes with default config."""
        selector = EstimatorSelector()

        assert selector.config is not None
        assert selector.config.strategy == SelectionStrategy.BEST_ENERGY_SCORE

    def test_selector_init_custom_config(self):
        """Test selector with custom config."""
        config = EstimatorSelectorConfig(
            strategy=SelectionStrategy.FIRST_SUCCESS,
        )
        selector = EstimatorSelector(config)

        assert selector.config.strategy == SelectionStrategy.FIRST_SUCCESS

    def test_selector_builds_estimator_chain(self):
        """Test selector builds estimator chain from config."""
        config = EstimatorSelectorConfig(
            estimators=[
                EstimatorConfig(EstimatorType.OLS, priority=1),
            ]
        )
        selector = EstimatorSelector(config)

        assert len(selector.estimators) == 1
        assert selector.estimators[0].estimator_type == EstimatorType.OLS

    def test_selector_filters_disabled_estimators(self):
        """Test selector filters out disabled estimators."""
        config = EstimatorSelectorConfig(
            estimators=[
                EstimatorConfig(EstimatorType.OLS, enabled=True, priority=1),
                EstimatorConfig(EstimatorType.LINEAR_DML, enabled=False, priority=2),
            ]
        )
        selector = EstimatorSelector(config)

        assert len(selector.estimators) == 1
        assert selector.estimators[0].estimator_type == EstimatorType.OLS

    def test_selector_select_returns_result(self, simple_data):
        """Test select returns SelectionResult."""
        config = EstimatorSelectorConfig(
            estimators=[
                EstimatorConfig(EstimatorType.OLS, priority=1),
            ],
            energy_score_config=EnergyScoreConfig(enable_bootstrap=False),
        )
        selector = EstimatorSelector(config)

        result = selector.select(
            treatment=simple_data["treatment"],
            outcome=simple_data["outcome"],
            covariates=simple_data["covariates"],
        )

        assert isinstance(result, SelectionResult)

    def test_selector_select_has_selected_estimator(self, simple_data):
        """Test select returns a selected estimator."""
        config = EstimatorSelectorConfig(
            estimators=[
                EstimatorConfig(EstimatorType.OLS, priority=1),
            ],
            energy_score_config=EnergyScoreConfig(enable_bootstrap=False),
        )
        selector = EstimatorSelector(config)

        result = selector.select(
            treatment=simple_data["treatment"],
            outcome=simple_data["outcome"],
            covariates=simple_data["covariates"],
        )

        assert result.selected is not None
        assert result.selected.estimator_type == EstimatorType.OLS

    def test_selector_records_all_results(self, simple_data):
        """Test select records all evaluated results."""
        config = EstimatorSelectorConfig(
            estimators=[
                EstimatorConfig(EstimatorType.OLS, priority=1),
            ],
            energy_score_config=EnergyScoreConfig(enable_bootstrap=False),
        )
        selector = EstimatorSelector(config)

        result = selector.select(
            treatment=simple_data["treatment"],
            outcome=simple_data["outcome"],
            covariates=simple_data["covariates"],
        )

        assert len(result.all_results) == 1

    def test_selector_computes_energy_scores(self, simple_data):
        """Test select computes energy scores."""
        config = EstimatorSelectorConfig(
            estimators=[
                EstimatorConfig(EstimatorType.OLS, priority=1),
            ],
            energy_score_config=EnergyScoreConfig(enable_bootstrap=False),
        )
        selector = EstimatorSelector(config)

        result = selector.select(
            treatment=simple_data["treatment"],
            outcome=simple_data["outcome"],
            covariates=simple_data["covariates"],
        )

        assert "ols" in result.energy_scores

    def test_selector_records_total_time(self, simple_data):
        """Test select records total computation time."""
        config = EstimatorSelectorConfig(
            estimators=[
                EstimatorConfig(EstimatorType.OLS, priority=1),
            ],
            energy_score_config=EnergyScoreConfig(enable_bootstrap=False),
        )
        selector = EstimatorSelector(config)

        result = selector.select(
            treatment=simple_data["treatment"],
            outcome=simple_data["outcome"],
            covariates=simple_data["covariates"],
        )

        assert result.total_time_ms > 0

    def test_selector_first_success_strategy(self, simple_data):
        """Test first success strategy selects first working estimator."""
        config = EstimatorSelectorConfig(
            strategy=SelectionStrategy.FIRST_SUCCESS,
            estimators=[
                EstimatorConfig(EstimatorType.OLS, priority=1),
            ],
            energy_score_config=EnergyScoreConfig(enable_bootstrap=False),
        )
        selector = EstimatorSelector(config)

        result = selector.select(
            treatment=simple_data["treatment"],
            outcome=simple_data["outcome"],
            covariates=simple_data["covariates"],
        )

        assert result.selection_strategy == SelectionStrategy.FIRST_SUCCESS


# =============================================================================
# #622 Fast-estimator tiebreak Tests
# =============================================================================


def _energy(score: float) -> EnergyScoreResult:
    """Build a minimal EnergyScoreResult carrying a specific energy_score."""
    return EnergyScoreResult(
        estimator_name="x",
        energy_score=score,
        treatment_balance_score=0.0,
        outcome_fit_score=0.0,
        propensity_calibration=0.0,
        n_samples=100,
        n_treated=50,
        n_control=50,
        computation_time_ms=1.0,
    )


def _result(est_type: EstimatorType, score: float) -> EstimatorResult:
    """Build a successful EstimatorResult with a fixed energy score."""
    return EstimatorResult(
        estimator_type=est_type,
        success=True,
        ate=1.0,
        cate=np.array([1.0, 1.0]),
        energy_score_result=_energy(score),
    )


class TestFastEstimatorTiebreak:
    """Energy-score tiebreak = (confounding_blind?, speed, energy).

    #622 originally broke ties by raw SPEED to avoid the slow CausalForestDML
    refutation suite (~0.05s/re-estimation for OLS vs ~3.1s for CausalForest,
    MEASURED → ~30s vs ~35-60 min). But the energy score measures fit, not
    causal validity: raw-speed ties always handed the run to naive OLS, which on
    the patient_journeys gold standard fits as well as the DML/forest family yet
    fails the refutation gate (OLS gate=BLOCK vs DML/forest gate=PROCEED at equal
    energy). The tiebreak now prefers a CONFOUNDING-ROBUST estimator first, and
    keeps #622's fast-among-robust preference second — so on a tie we still avoid
    the slow forest when a faster robust estimator (e.g. linear_dml) is present,
    but we never select confounding-blind OLS over a robust estimator.
    """

    def _selector(self, gap: float = 0.05) -> EstimatorSelector:
        # Empty estimator chain — we call _select_best_energy directly with
        # crafted results, so we don't fit anything (PR-lane fast, no slow refit).
        config = EstimatorSelectorConfig(
            estimators=[EstimatorConfig(EstimatorType.OLS, priority=1)],
            min_energy_score_gap=gap,
        )
        return EstimatorSelector(config)

    def test_exact_tie_prefers_fastest_robust_not_ols_or_forest(self):
        """On an exact energy-score tie the selector prefers the fastest
        CONFOUNDING-ROBUST estimator: not naive OLS (confounding-blind, even
        though fastest) and not the slow chain-priority head causal_forest.
        linear_dml is the fastest robust estimator here."""
        selector = self._selector()
        # Chain-priority order: causal_forest first (the slow trap).
        results = [
            _result(EstimatorType.CAUSAL_FOREST, 0.5382),
            _result(EstimatorType.LINEAR_DML, 0.5382),
            _result(EstimatorType.DRLEARNER, 0.5382),
            _result(EstimatorType.OLS, 0.5382),
        ]
        selected = selector._select_best_energy(results)
        assert selected.estimator_type == EstimatorType.LINEAR_DML, (
            "On an exact energy-score tie the selector must prefer the fastest "
            "confounding-robust estimator (linear_dml) — never naive OLS, and "
            "not the slow causal_forest."
        )

    def test_within_gap_tie_excludes_naive_ols_even_when_faster(self):
        """A confounding-robust estimator wins the tie band over naive OLS even
        though OLS is faster: energy ties do not justify a confounding-blind
        estimator. Here causal_forest is the only robust candidate in the band."""
        selector = self._selector(gap=0.05)
        results = [
            _result(EstimatorType.CAUSAL_FOREST, 0.500),
            _result(EstimatorType.OLS, 0.530),  # +0.03 within gap 0.05, faster
        ]
        selected = selector._select_best_energy(results)
        assert selected.estimator_type == EstimatorType.CAUSAL_FOREST

    def test_meaningfully_better_score_wins_over_speed(self):
        """Speed must NOT override a genuinely better (lower) energy score."""
        selector = self._selector(gap=0.05)
        results = [
            _result(EstimatorType.CAUSAL_FOREST, 0.300),  # clearly best
            _result(EstimatorType.OLS, 0.500),  # 0.2 worse, outside gap
        ]
        selected = selector._select_best_energy(results)
        assert selected.estimator_type == EstimatorType.CAUSAL_FOREST

    def test_tie_band_picks_fastest_among_several(self):
        """Within a wide tie band, the single fastest estimator is chosen."""
        selector = self._selector(gap=0.10)
        results = [
            _result(EstimatorType.ORTHO_FOREST, 0.500),
            _result(EstimatorType.DRLEARNER, 0.505),
            _result(EstimatorType.LINEAR_DML, 0.510),  # rank 2
            _result(EstimatorType.S_LEARNER, 0.515),  # rank 1 (faster than linear_dml)
        ]
        selected = selector._select_best_energy(results)
        assert selected.estimator_type == EstimatorType.S_LEARNER

    def test_single_estimator_unaffected(self):
        """A single successful estimator is returned regardless of tiebreak."""
        selector = self._selector()
        results = [_result(EstimatorType.CAUSAL_FOREST, 0.42)]
        selected = selector._select_best_energy(results)
        assert selected.estimator_type == EstimatorType.CAUSAL_FOREST

    def test_all_energy_uncomputed_prefers_fastest_robust(self):
        """When energy scores are all inf (uncomputed), prefer the fastest
        confounding-ROBUST estimator (linear_dml) — not naive OLS (even though
        fastest) and not the slow chain-priority causal_forest head."""
        selector = self._selector()
        cf = EstimatorResult(estimator_type=EstimatorType.CAUSAL_FOREST, success=True, ate=1.0)
        ldml = EstimatorResult(estimator_type=EstimatorType.LINEAR_DML, success=True, ate=1.0)
        ols = EstimatorResult(estimator_type=EstimatorType.OLS, success=True, ate=1.0)
        assert cf.energy_score == float("inf")
        assert ols.energy_score == float("inf")
        selected = selector._select_best_energy([cf, ldml, ols])
        assert selected.estimator_type == EstimatorType.LINEAR_DML

    def test_tie_band_prefers_robust_over_naive_ols_patient_journeys(self):
        """Mirrors the patient_journeys gold standard: all four estimators tie on
        energy (~0.36, causal_forest marginally lowest) but OLS is the fastest.
        The fix must pick a confounding-robust estimator (fastest robust =
        linear_dml), NOT naive OLS — which is what the raw-speed #622 tiebreak
        wrongly did, sending every default 'Auto' run to a gate=BLOCK result."""
        selector = self._selector(gap=0.05)
        results = [
            _result(EstimatorType.CAUSAL_FOREST, 0.35989),
            _result(EstimatorType.LINEAR_DML, 0.36034),
            _result(EstimatorType.DRLEARNER, 0.36160),
            _result(EstimatorType.OLS, 0.36032),
        ]
        selected = selector._select_best_energy(results)
        assert selected.estimator_type != EstimatorType.OLS
        assert selected.estimator_type == EstimatorType.LINEAR_DML


# =============================================================================
# select_best_estimator Convenience Function Tests
# =============================================================================


class TestSelectBestEstimatorFunction:
    """Tests for select_best_estimator convenience function."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        np.random.seed(123)
        n = 100

        covariates = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n),
                "x2": np.random.normal(0, 1, n),
            }
        )

        treatment = np.random.binomial(1, 0.5, n)
        outcome = 1.0 + 2.0 * treatment + np.random.normal(0, 0.3, n)

        return {
            "treatment": treatment,
            "outcome": outcome,
            "covariates": covariates,
        }

    def test_function_returns_selection_result(self, simple_data):
        """Test convenience function returns SelectionResult."""
        config = EstimatorSelectorConfig(
            estimators=[
                EstimatorConfig(EstimatorType.OLS, priority=1),
            ],
            energy_score_config=EnergyScoreConfig(enable_bootstrap=False),
        )

        result = select_best_estimator(
            treatment=simple_data["treatment"],
            outcome=simple_data["outcome"],
            covariates=simple_data["covariates"],
            config=config,
        )

        assert isinstance(result, SelectionResult)

    def test_function_with_default_config(self, simple_data):
        """Test convenience function with default config."""
        # Using a minimal config to avoid heavy ML dependencies
        config = EstimatorSelectorConfig(
            estimators=[
                EstimatorConfig(EstimatorType.OLS, priority=1),
            ],
            energy_score_config=EnergyScoreConfig(enable_bootstrap=False),
        )

        result = select_best_estimator(
            treatment=simple_data["treatment"],
            outcome=simple_data["outcome"],
            covariates=simple_data["covariates"],
            config=config,
        )

        assert result.selected.success is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestEstimatorSelectorIntegration:
    """Integration tests for estimator selector."""

    def test_full_workflow_with_ols(self):
        """Test full workflow with OLS estimator."""
        np.random.seed(42)
        n = 200

        covariates = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n),
                "x2": np.random.normal(0, 1, n),
            }
        )

        propensity = 1 / (1 + np.exp(-0.5 * covariates["x1"]))
        treatment = (np.random.random(n) < propensity).astype(int)

        true_effect = 2.5
        outcome = (
            1.0 + 0.3 * covariates["x1"] + true_effect * treatment + np.random.normal(0, 0.5, n)
        )

        # Use only OLS for fast testing
        config = EstimatorSelectorConfig(
            estimators=[
                EstimatorConfig(EstimatorType.OLS, priority=1),
            ],
            energy_score_config=EnergyScoreConfig(enable_bootstrap=False),
        )

        result = select_best_estimator(
            treatment=treatment,
            outcome=outcome,
            covariates=covariates,
            config=config,
        )

        # Verify result structure
        assert result.selected.success is True
        assert result.selected.ate is not None
        assert result.selected.cate is not None
        assert result.total_time_ms > 0
        assert len(result.energy_scores) > 0

        # ATE should be somewhat close to true effect
        assert 1.0 < result.selected.ate < 4.0

    def test_selection_result_serialization(self):
        """Test that SelectionResult can be serialized to dict."""
        np.random.seed(42)
        n = 100

        covariates = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n),
            }
        )

        treatment = np.random.binomial(1, 0.5, n)
        outcome = 1.0 + 2.0 * treatment + np.random.normal(0, 0.5, n)

        config = EstimatorSelectorConfig(
            estimators=[
                EstimatorConfig(EstimatorType.OLS, priority=1),
            ],
            energy_score_config=EnergyScoreConfig(enable_bootstrap=False),
        )

        result = select_best_estimator(
            treatment=treatment,
            outcome=outcome,
            covariates=covariates,
            config=config,
        )

        # Should serialize without errors
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "selected_estimator" in d
        assert "selection_strategy" in d
        assert "energy_scores" in d


class TestSelectionResultRequiresReview:
    """M-est3: a best energy score above ``max_acceptable_energy_score`` must
    flag the SelectionResult as requiring review instead of only logging a
    warning while the unreliable ATE flows downstream as a clean result.
    """

    def _selector(self, max_acceptable: float) -> EstimatorSelector:
        # Single-OLS chain; we call select()'s internals via crafted results,
        # so nothing slow is fit (matches TestFastEstimatorTiebreak pattern).
        config = EstimatorSelectorConfig(
            estimators=[EstimatorConfig(EstimatorType.OLS, priority=1)],
            max_acceptable_energy_score=max_acceptable,
        )
        return EstimatorSelector(config)

    def test_requires_review_default_false(self):
        selected = EstimatorResult(estimator_type=EstimatorType.OLS, success=True, ate=2.0)
        result = SelectionResult(
            selected=selected,
            selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
        )
        assert result.requires_review is False
        assert result.exceeded_max_energy_score is False

    def test_to_dict_includes_review_flags(self):
        selected = EstimatorResult(estimator_type=EstimatorType.OLS, success=True, ate=2.0)
        result = SelectionResult(
            selected=selected,
            selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
            all_results=[selected],
            requires_review=True,
            exceeded_max_energy_score=True,
        )
        d = result.to_dict()
        assert d["requires_review"] is True
        assert d["exceeded_max_energy_score"] is True

    def test_select_flags_review_when_best_exceeds_max(self):
        # max_acceptable low so the single result (energy 0.5) breaches it.
        selector = self._selector(max_acceptable=0.3)
        results = [_result(EstimatorType.OLS, 0.5)]
        sel = selector._select_best_energy(results)
        breached = sel.energy_score > selector.config.max_acceptable_energy_score
        assert breached is True  # precondition for the flag
        # Now assert the full SelectionResult produced carries it.
        sr = selector._build_selection_result(
            selection=sel,
            results=results,
            total_time_ms=0.0,
        )
        assert sr.requires_review is True
        assert sr.exceeded_max_energy_score is True

    def test_select_no_review_when_best_within_max(self):
        selector = self._selector(max_acceptable=0.8)
        results = [_result(EstimatorType.OLS, 0.5)]
        sel = selector._select_best_energy(results)
        sr = selector._build_selection_result(
            selection=sel,
            results=results,
            total_time_ms=0.0,
        )
        assert sr.requires_review is False
        assert sr.exceeded_max_energy_score is False


# =============================================================================
# Empty-backdoor (0-covariate) unadjusted estimation
#
# An empty validated backdoor is the CORRECT adjustment set for a randomized
# (RCT) or exogenous treatment — there is nothing to confound, so P(T|X)
# reduces to the constant marginal P(T). The energy-score CATE chain
# (CausalForest/LinearDML/DRLearner) genuinely cannot fit a 0-feature X, but
# the unadjusted ATE (OLS coefficient on the treatment alone == difference in
# means for a binary treatment) is well-defined and must be produced rather
# than fail-closing the whole question. See estimation.py empty-backdoor path.
# =============================================================================


class TestEmptyBackdoorUnadjusted:
    """OLS + selector must produce an unadjusted ATE when there are 0 covariates."""

    @staticmethod
    def _rct_frame(n: int = 1500, p_treat: float = 0.4, lift: float = 0.3):
        rng = np.random.RandomState(0)
        treatment = (rng.rand(n) < p_treat).astype(int)
        outcome = (rng.rand(n) < (0.3 + lift * treatment)).astype(float)
        covariates = pd.DataFrame(index=range(n))  # ZERO columns
        diff = float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())
        return treatment, outcome, covariates, diff

    def test_ols_wrapper_succeeds_with_zero_covariates(self):
        """OLSWrapper.fit on a 0-column covariate frame returns the unadjusted ATE
        (== difference in means) instead of raising on the propensity fit."""
        treatment, outcome, covariates, diff = self._rct_frame()
        wrapper = OLSWrapper(EstimatorConfig(EstimatorType.OLS))

        result = wrapper.fit(treatment, outcome, covariates)

        assert result.success is True, result.error_message
        assert result.ate is not None
        assert result.ate == pytest.approx(diff, abs=1e-9)
        assert result.ate_std is not None and result.ate_std > 0.0
        assert result.ate_ci_lower is not None and result.ate_ci_upper is not None
        assert result.ate_ci_lower < result.ate < result.ate_ci_upper
        # Propensity for an empty backdoor is the constant marginal P(T)=mean(T).
        assert result.propensity_scores is not None
        assert np.allclose(result.propensity_scores, float(np.mean(treatment)))

    def test_selector_selects_ols_with_zero_covariates(self):
        """The selector returns a successful unadjusted estimate (OLS) when the
        CATE estimators all fail on 0 features; the energy score stays finite and
        below the review threshold (a clean RCT estimate is NOT 'unreliable')."""
        treatment, outcome, covariates, diff = self._rct_frame()
        selector = EstimatorSelector()

        sel = selector.select(treatment=treatment, outcome=outcome, covariates=covariates)

        assert sel.selected.success is True, sel.selected.error_message
        assert sel.selected.estimator_type == EstimatorType.OLS
        assert sel.selected.ate == pytest.approx(diff, abs=1e-9)
        # The CATE estimators genuinely cannot fit 0-feature X — recorded as failures.
        failed = {r.estimator_type for r in sel.all_results if not r.success}
        assert EstimatorType.CAUSAL_FOREST in failed
        assert EstimatorType.LINEAR_DML in failed
        # Finite, good-tier energy score -> not requires_review.
        assert np.isfinite(sel.selected.energy_score)
        assert sel.requires_review is False

    def test_ols_wrapper_non_01_encoding_normalized_to_diff_in_means(self):
        """A non-0/1 two-arm encoding (e.g. {1,2}) is accepted and normalized so
        the ATE is the difference-in-means, NOT a per-unit slope (codex r2)."""
        rng = np.random.RandomState(3)
        n = 1500
        base = (rng.rand(n) < 0.4).astype(int)
        treatment = base + 1  # arms encoded as {1, 2}
        outcome = (rng.rand(n) < (0.3 + 0.3 * base)).astype(float)
        covariates = pd.DataFrame(index=range(n))
        diff = float(outcome[base == 1].mean() - outcome[base == 0].mean())

        result = OLSWrapper(EstimatorConfig(EstimatorType.OLS)).fit(treatment, outcome, covariates)

        assert result.success is True, result.error_message
        assert result.ate == pytest.approx(diff, abs=1e-9)

    def test_ols_wrapper_one_arm_fails_closed(self):
        """A one-arm (single distinct value) empty-backdoor sample fails-closed
        rather than emitting a degenerate ate / constant 0-or-1 propensity."""
        n = 1500
        treatment = np.ones(n, dtype=int)  # only one arm present
        outcome = np.random.RandomState(4).rand(n)
        covariates = pd.DataFrame(index=range(n))

        result = OLSWrapper(EstimatorConfig(EstimatorType.OLS)).fit(treatment, outcome, covariates)

        assert result.success is False
        assert "two treatment arms" in (result.error_message or "")


# =============================================================================
# Empty-backdoor (randomized / zero-covariate) skip behavior
# =============================================================================


class TestEmptyBackdoorSkip:
    """A ZERO-covariate design matrix is the correct adjustment set for a randomized
    / exogenous treatment (e.g. the nba_triggers RCT). The covariate-requiring
    estimators (causal_forest, linear_dml, drlearner) cannot fit a 0-feature X, so
    they must be SKIPPED as not-applicable — not surfaced as a raw sklearn 'Found
    array with 0 feature(s)' failure. Only OLS (the unadjusted contrast) applies.
    """

    def _empty_backdoor_frame(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        treatment = rng.integers(0, 2, n)
        covariates = pd.DataFrame(index=range(n))  # 0 columns
        outcome = (0.4 * treatment + rng.normal(0, 0.3, n)).astype(float)
        return treatment, outcome, covariates

    def test_covariate_estimators_skipped_not_failed(self):
        treatment, outcome, covariates = self._empty_backdoor_frame()
        result = EstimatorSelector().select(treatment, outcome, covariates)

        by_type = {r.estimator_type: r for r in result.all_results}
        for et in (
            EstimatorType.CAUSAL_FOREST,
            EstimatorType.LINEAR_DML,
            EstimatorType.DRLEARNER,
        ):
            r = by_type[et]
            assert r.skipped is True, f"{et.value} should be skipped on an empty backdoor"
            assert r.success is False
            # Honest not-applicable reason, NOT the cryptic sklearn traceback.
            assert "not applicable" in (r.error_message or "").lower()
            assert "0 feature" not in (r.error_message or "")

    def test_ols_still_fits_and_is_selected(self):
        treatment, outcome, covariates = self._empty_backdoor_frame()
        result = EstimatorSelector().select(treatment, outcome, covariates)

        ols = next(r for r in result.all_results if r.estimator_type == EstimatorType.OLS)
        assert ols.skipped is False
        assert ols.success is True
        assert result.selected.estimator_type == EstimatorType.OLS
        # The unadjusted contrast recovers the planted ~0.4 effect.
        assert result.selected.ate == pytest.approx(0.4, abs=0.1)

    def test_selection_reason_explains_empty_backdoor(self):
        treatment, outcome, covariates = self._empty_backdoor_frame()
        result = EstimatorSelector().select(treatment, outcome, covariates)
        reason = result.selection_reason.lower()
        assert "randomized" in reason or "empty-backdoor" in reason or "no covariates" in reason

    def test_with_covariates_nothing_is_skipped(self):
        """Regression guard: the skip fires ONLY on a truly empty backdoor. With >=1
        covariate every estimator is attempted as before."""
        rng = np.random.default_rng(1)
        n = 400
        treatment = rng.integers(0, 2, n)
        covariates = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
        outcome = (
            0.3 * treatment + 0.5 * covariates["x1"].to_numpy() + rng.normal(0, 0.3, n)
        ).astype(float)
        result = EstimatorSelector().select(treatment, outcome, covariates)
        assert all(r.skipped is False for r in result.all_results)


# =============================================================================
# #1188: RCT efficiency controls (ANCOVA-style variance reduction)
# =============================================================================


def _prognostic_rct_frame(n=3000, seed=0, tau=0.3):
    """Randomized T, STRONGLY prognostic baseline x1 (pure outcome signal,
    no confounding): the substrate where baseline adjustment buys precision."""
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, n)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)  # pure-noise stratum
    outcome = (tau * treatment + 1.0 * x1 + rng.normal(0, 0.3, n)).astype(float)
    baselines = pd.DataFrame({"disease_severity": x1, "age_at_diagnosis": x2})
    diff = float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())
    return treatment, outcome, baselines, diff


def _fast_dml_ols_config():
    """LinearDML + OLS only — keeps the statistical tests fast."""
    return EstimatorSelectorConfig(
        estimators=[
            EstimatorConfig(EstimatorType.LINEAR_DML, priority=1),
            EstimatorConfig(EstimatorType.OLS, priority=2),
        ]
    )


class TestEfficiencyControls:
    """#1188: on a randomized (empty-backdoor) design, curated pre-treatment
    baselines may be supplied as EFFICIENCY CONTROLS. Covariate estimators then
    become applicable FOR PRECISION (variance reduction); OLS stays the
    UNADJUSTED diff-in-means anchor; labeling must say efficiency, never
    confounding."""

    def test_efficiency_controls_make_covariate_estimators_applicable(self):
        treatment, outcome, baselines, _ = _prognostic_rct_frame()
        empty = pd.DataFrame(index=range(len(treatment)))

        result = EstimatorSelector(_fast_dml_ols_config()).select(
            treatment, outcome, empty, efficiency_controls=baselines
        )

        dml = next(r for r in result.all_results if r.estimator_type == EstimatorType.LINEAR_DML)
        assert dml.skipped is False, "efficiency controls must un-skip covariate estimators"
        assert dml.success is True, dml.error_message

    def test_ols_anchor_stays_unadjusted_diff_in_means(self):
        """OLS must NOT absorb the efficiency controls — it anchors the
        comparison at the raw randomized contrast."""
        treatment, outcome, baselines, diff = _prognostic_rct_frame()
        empty = pd.DataFrame(index=range(len(treatment)))

        result = EstimatorSelector(_fast_dml_ols_config()).select(
            treatment, outcome, empty, efficiency_controls=baselines
        )

        ols = next(r for r in result.all_results if r.estimator_type == EstimatorType.OLS)
        assert ols.success is True, ols.error_message
        assert ols.ate == pytest.approx(diff, abs=1e-9)

    def test_adjustment_type_efficiency_and_reason(self):
        treatment, outcome, baselines, _ = _prognostic_rct_frame()
        empty = pd.DataFrame(index=range(len(treatment)))

        result = EstimatorSelector(_fast_dml_ols_config()).select(
            treatment, outcome, empty, efficiency_controls=baselines
        )

        assert result.adjustment_type == "efficiency"
        reason = result.selection_reason.lower()
        assert "variance reduction" in reason or "precision" in reason
        assert "confound" not in reason, (
            "efficiency adjustment must never be framed as de-confounding"
        )

    def test_adjustment_type_none_on_plain_empty_backdoor(self):
        """No efficiency controls => the existing skip behavior and labeling
        are untouched."""
        rng = np.random.default_rng(3)
        n = 400
        treatment = rng.integers(0, 2, n)
        outcome = (0.4 * treatment + rng.normal(0, 0.3, n)).astype(float)
        empty = pd.DataFrame(index=range(n))

        result = EstimatorSelector().select(treatment, outcome, empty)

        assert result.adjustment_type == "none"
        skipped = {r.estimator_type for r in result.all_results if r.skipped}
        assert EstimatorType.LINEAR_DML in skipped

    def test_adjustment_type_confounding_with_real_covariates(self):
        """A non-empty de-confounding covariate set keeps today's semantics and
        labels the run as confounding adjustment; efficiency controls are a
        strictly empty-backdoor feature."""
        rng = np.random.default_rng(4)
        n = 600
        x1 = rng.normal(0, 1, n)
        treatment = (rng.random(n) < 1 / (1 + np.exp(-x1))).astype(int)
        outcome = (0.3 * treatment + 0.8 * x1 + rng.normal(0, 0.3, n)).astype(float)
        covariates = pd.DataFrame({"x1": x1})

        result = EstimatorSelector(_fast_dml_ols_config()).select(
            treatment, outcome, covariates
        )

        assert result.adjustment_type == "confounding"

    def test_adjusted_ci_narrower_than_unadjusted_anchor(self):
        """THE #1188 point: with a strongly prognostic baseline, the adjusted
        estimator's HONEST CI must be strictly narrower than the unadjusted OLS
        anchor CI, while the point estimates stay statistically
        indistinguishable (randomization keeps both unbiased)."""
        treatment, outcome, baselines, diff = _prognostic_rct_frame(n=4000, seed=5)
        empty = pd.DataFrame(index=range(len(treatment)))

        result = EstimatorSelector(_fast_dml_ols_config()).select(
            treatment, outcome, empty, efficiency_controls=baselines
        )

        ols = next(r for r in result.all_results if r.estimator_type == EstimatorType.OLS)
        dml = next(r for r in result.all_results if r.estimator_type == EstimatorType.LINEAR_DML)
        assert ols.success and dml.success
        ols_width = ols.ate_ci_upper - ols.ate_ci_lower
        dml_width = dml.ate_ci_upper - dml.ate_ci_lower
        assert dml_width < ols_width, (
            f"adjusted CI must be narrower: dml={dml_width:.4f} ols={ols_width:.4f}"
        )
        # Unbiasedness: the adjusted point recovers the PLANTED tau=0.3 (it may
        # differ from the raw contrast by the chance-imbalance correction — the
        # very thing baseline adjustment is for on a strongly prognostic x1).
        assert dml.ate == pytest.approx(0.3, abs=0.06)


class TestHonestAteCi:
    """#1188 prerequisite: the wrapper ATE CIs must be SAMPLING intervals, not
    the fake-precise std(cate)/sqrt(n) heterogeneity spread the (silently
    failing) legacy inference path fell back to. Analytic truth for
    y = tau*T + 1.0*x1 + N(0,0.3): adjusted SE ~= 0.3*sqrt(1/n1+1/n0)."""

    def _frame(self, n=3000, seed=8):
        treatment, outcome, baselines, diff = _prognostic_rct_frame(n=n, seed=seed)
        n1 = int(treatment.sum())
        n0 = n - n1
        analytic_width = 2 * 1.96 * 0.3 * np.sqrt(1.0 / n1 + 1.0 / n0)
        return treatment, outcome, baselines, analytic_width

    def test_lineardml_ate_ci_is_honest_sampling_interval(self):
        treatment, outcome, baselines, analytic = self._frame()
        r = LinearDMLWrapper(EstimatorConfig(EstimatorType.LINEAR_DML)).fit(
            treatment, outcome, baselines
        )
        assert r.success, r.error_message
        width = r.ate_ci_upper - r.ate_ci_lower
        assert 0.5 * analytic <= width <= 2.5 * analytic, (
            f"LinearDML CI width {width:.5f} vs analytic {analytic:.5f} — "
            "fake-precise (or absurdly wide) ATE interval"
        )

    def test_drlearner_ate_ci_is_honest_sampling_interval(self):
        treatment, outcome, baselines, analytic = self._frame()
        r = DRLearnerWrapper(EstimatorConfig(EstimatorType.DRLEARNER)).fit(
            treatment, outcome, baselines
        )
        assert r.success, r.error_message
        width = r.ate_ci_upper - r.ate_ci_lower
        assert width >= 0.5 * analytic, (
            f"DRLearner CI width {width:.5f} vs analytic {analytic:.5f} — fake-precise"
        )

    def test_causalforest_ate_ci_not_fake_precise(self):
        """CausalForestDML's population interval is a conservative upper bound —
        wide is acceptable, fake-narrow is not."""
        treatment, outcome, baselines, analytic = self._frame()
        r = CausalForestWrapper(EstimatorConfig(EstimatorType.CAUSAL_FOREST)).fit(
            treatment, outcome, baselines
        )
        assert r.success, r.error_message
        width = r.ate_ci_upper - r.ate_ci_lower
        assert width >= 0.5 * analytic, (
            f"CausalForestDML CI width {width:.5f} vs analytic {analytic:.5f} — fake-precise"
        )
