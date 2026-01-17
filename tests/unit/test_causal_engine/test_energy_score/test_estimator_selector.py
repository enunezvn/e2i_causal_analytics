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
from unittest.mock import MagicMock, patch

from src.causal_engine.energy_score.estimator_selector import (
    EstimatorType,
    SelectionStrategy,
    EstimatorResult,
    SelectionResult,
    EstimatorConfig,
    EstimatorSelectorConfig,
    BaseEstimatorWrapper,
    OLSWrapper,
    SLearnerWrapper,
    TLearnerWrapper,
    ESTIMATOR_WRAPPERS,
    EstimatorSelector,
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

        assert result.energy_score == float('inf')

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

        covariates = pd.DataFrame({
            "x1": np.random.normal(0, 1, n),
            "x2": np.random.normal(0, 1, n),
        })

        treatment = np.random.binomial(1, 0.5, n)
        true_effect = 2.0
        outcome = (
            1.0 + 0.5 * covariates["x1"] +
            true_effect * treatment + np.random.normal(0, 0.5, n)
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

        covariates = pd.DataFrame({
            "x1": np.random.normal(0, 1, n),
            "x2": np.random.normal(0, 1, n),
        })

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
# select_best_estimator Convenience Function Tests
# =============================================================================


class TestSelectBestEstimatorFunction:
    """Tests for select_best_estimator convenience function."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        np.random.seed(123)
        n = 100

        covariates = pd.DataFrame({
            "x1": np.random.normal(0, 1, n),
            "x2": np.random.normal(0, 1, n),
        })

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

        covariates = pd.DataFrame({
            "x1": np.random.normal(0, 1, n),
            "x2": np.random.normal(0, 1, n),
        })

        propensity = 1 / (1 + np.exp(-0.5 * covariates["x1"]))
        treatment = (np.random.random(n) < propensity).astype(int)

        true_effect = 2.5
        outcome = (
            1.0 + 0.3 * covariates["x1"] +
            true_effect * treatment + np.random.normal(0, 0.5, n)
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

        covariates = pd.DataFrame({
            "x1": np.random.normal(0, 1, n),
        })

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
