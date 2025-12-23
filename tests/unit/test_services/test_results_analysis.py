"""
Unit tests for Results Analysis Service.

Phase 15: A/B Testing Infrastructure

Tests for:
- Intent-to-treat (ITT) analysis
- Per-protocol analysis
- Heterogeneous treatment effects (HTE)
- Sample Ratio Mismatch (SRM) detection
- Digital Twin fidelity tracking
- Statistical power calculation
- Result persistence
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import numpy as np
import pytest
from scipy import stats

from src.services.results_analysis import (
    AnalysisMethod,
    AnalysisType,
    ExperimentResults,
    FidelityComparison,
    ResultsAnalysisConfig,
    ResultsAnalysisService,
    SRMCheckResult,
    SRMSeverity,
    get_results_analysis_service,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def default_config() -> ResultsAnalysisConfig:
    """Default results analysis configuration."""
    return ResultsAnalysisConfig(
        alpha=0.05,
        power_threshold=0.8,
        srm_p_threshold=0.001,
        srm_warning_threshold=0.01,
        fidelity_excellent_threshold=0.1,
        fidelity_good_threshold=0.2,
        fidelity_acceptable_threshold=0.3,
    )


@pytest.fixture
def service(default_config: ResultsAnalysisConfig) -> ResultsAnalysisService:
    """Default results analysis service."""
    return ResultsAnalysisService(config=default_config)


@pytest.fixture
def experiment_id() -> UUID:
    """Sample experiment ID."""
    return uuid4()


@pytest.fixture
def twin_simulation_id() -> UUID:
    """Sample twin simulation ID."""
    return uuid4()


@pytest.fixture
def control_data_baseline() -> np.ndarray:
    """Control group data with baseline conversion."""
    np.random.seed(42)
    return np.random.normal(0.10, 0.03, 500)


@pytest.fixture
def treatment_data_positive_effect() -> np.ndarray:
    """Treatment group data with positive effect."""
    np.random.seed(43)
    return np.random.normal(0.12, 0.03, 500)  # +20% lift


@pytest.fixture
def treatment_data_no_effect() -> np.ndarray:
    """Treatment group data with no effect."""
    np.random.seed(44)
    return np.random.normal(0.10, 0.03, 500)


@pytest.fixture
def treatment_data_negative_effect() -> np.ndarray:
    """Treatment group data with negative effect."""
    np.random.seed(45)
    return np.random.normal(0.08, 0.03, 500)  # -20% lift


@pytest.fixture
def secondary_metrics_data(
    control_data_baseline: np.ndarray,
    treatment_data_positive_effect: np.ndarray,
) -> dict:
    """Secondary metrics data."""
    np.random.seed(46)
    return {
        "clicks": (
            np.random.normal(5.0, 1.0, len(control_data_baseline)),
            np.random.normal(5.5, 1.0, len(treatment_data_positive_effect)),
        ),
        "time_on_page": (
            np.random.normal(60.0, 15.0, len(control_data_baseline)),
            np.random.normal(62.0, 15.0, len(treatment_data_positive_effect)),
        ),
    }


# =============================================================================
# TEST ENUMS
# =============================================================================


class TestAnalysisMethod:
    """Tests for AnalysisMethod enum."""

    def test_analysis_method_values(self):
        """Test all analysis method values exist."""
        assert AnalysisMethod.ITT.value == "itt"
        assert AnalysisMethod.PER_PROTOCOL.value == "per_protocol"
        assert AnalysisMethod.AS_TREATED.value == "as_treated"

    def test_analysis_method_is_str_enum(self):
        """Test analysis method is string enum."""
        assert isinstance(AnalysisMethod.ITT, str)
        assert AnalysisMethod.ITT == "itt"


class TestAnalysisType:
    """Tests for AnalysisType enum."""

    def test_analysis_type_values(self):
        """Test all analysis type values exist."""
        assert AnalysisType.INTERIM.value == "interim"
        assert AnalysisType.FINAL.value == "final"
        assert AnalysisType.AD_HOC.value == "ad_hoc"


class TestSRMSeverity:
    """Tests for SRMSeverity enum."""

    def test_srm_severity_values(self):
        """Test all SRM severity values exist."""
        assert SRMSeverity.NONE.value == "none"
        assert SRMSeverity.WARNING.value == "warning"
        assert SRMSeverity.SEVERE.value == "severe"
        assert SRMSeverity.CRITICAL.value == "critical"


# =============================================================================
# TEST DATACLASSES
# =============================================================================


class TestExperimentResults:
    """Tests for ExperimentResults dataclass."""

    def test_experiment_results_creation(self):
        """Test experiment results creation."""
        exp_id = uuid4()
        now = datetime.now(timezone.utc)

        results = ExperimentResults(
            experiment_id=exp_id,
            analysis_type=AnalysisType.FINAL,
            analysis_method=AnalysisMethod.ITT,
            computed_at=now,
            primary_metric="conversion_rate",
            control_mean=0.10,
            treatment_mean=0.12,
            effect_estimate=0.02,
            effect_ci_lower=0.01,
            effect_ci_upper=0.03,
            relative_lift=20.0,
            relative_lift_ci_lower=10.0,
            relative_lift_ci_upper=30.0,
            p_value=0.001,
            is_significant=True,
            sample_size_control=500,
            sample_size_treatment=500,
            statistical_power=0.95,
        )

        assert results.experiment_id == exp_id
        assert results.analysis_method == AnalysisMethod.ITT
        assert results.is_significant is True
        assert results.secondary_metrics == []
        assert results.segment_results is None


class TestSRMCheckResult:
    """Tests for SRMCheckResult dataclass."""

    def test_srm_check_result_creation(self):
        """Test SRM check result creation."""
        exp_id = uuid4()
        now = datetime.now(timezone.utc)

        result = SRMCheckResult(
            experiment_id=exp_id,
            checked_at=now,
            expected_ratio={"control": 0.5, "treatment": 0.5},
            actual_counts={"control": 490, "treatment": 510},
            actual_ratio={"control": 0.49, "treatment": 0.51},
            chi_squared_statistic=0.4,
            p_value=0.527,
            is_srm_detected=False,
            severity=SRMSeverity.NONE,
        )

        assert result.is_srm_detected is False
        assert result.severity == SRMSeverity.NONE


class TestFidelityComparison:
    """Tests for FidelityComparison dataclass."""

    def test_fidelity_comparison_creation(self):
        """Test fidelity comparison creation."""
        exp_id = uuid4()
        twin_id = uuid4()
        now = datetime.now(timezone.utc)

        comparison = FidelityComparison(
            experiment_id=exp_id,
            twin_simulation_id=twin_id,
            comparison_timestamp=now,
            predicted_effect=0.025,
            actual_effect=0.020,
            prediction_error=-0.005,
            prediction_error_percent=20.0,
            predicted_ci_lower=0.01,
            predicted_ci_upper=0.04,
            ci_coverage=True,
            fidelity_score=0.8,
            fidelity_grade="B",
        )

        assert comparison.ci_coverage is True
        assert comparison.fidelity_grade == "B"
        assert comparison.calibration_adjustment == {}


class TestResultsAnalysisConfig:
    """Tests for ResultsAnalysisConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = ResultsAnalysisConfig()
        assert config.alpha == 0.05
        assert config.power_threshold == 0.8
        assert config.srm_p_threshold == 0.001
        assert config.srm_warning_threshold == 0.01
        assert config.fidelity_excellent_threshold == 0.1
        assert config.fidelity_good_threshold == 0.2
        assert config.fidelity_acceptable_threshold == 0.3

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = ResultsAnalysisConfig(
            alpha=0.10,
            power_threshold=0.9,
            srm_p_threshold=0.0001,
        )
        assert config.alpha == 0.10
        assert config.power_threshold == 0.9
        assert config.srm_p_threshold == 0.0001


# =============================================================================
# TEST COMPUTE ITT RESULTS
# =============================================================================


class TestComputeITTResults:
    """Tests for compute_itt_results method."""

    @pytest.mark.asyncio
    async def test_itt_with_significant_positive_effect(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        control_data_baseline: np.ndarray,
        treatment_data_positive_effect: np.ndarray,
    ):
        """Test ITT analysis detects significant positive effect."""
        with patch.object(service, "_persist_results", new_callable=AsyncMock):
            results = await service.compute_itt_results(
                experiment_id=experiment_id,
                primary_metric="conversion_rate",
                control_data=control_data_baseline,
                treatment_data=treatment_data_positive_effect,
            )

            assert results.experiment_id == experiment_id
            assert results.analysis_method == AnalysisMethod.ITT
            assert results.analysis_type == AnalysisType.FINAL
            assert results.effect_estimate > 0
            assert results.is_significant == True  # Use == for numpy bool compatibility
            assert results.p_value < 0.05

    @pytest.mark.asyncio
    async def test_itt_with_no_effect(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        control_data_baseline: np.ndarray,
        treatment_data_no_effect: np.ndarray,
    ):
        """Test ITT analysis with no effect is not significant."""
        with patch.object(service, "_persist_results", new_callable=AsyncMock):
            results = await service.compute_itt_results(
                experiment_id=experiment_id,
                primary_metric="conversion_rate",
                control_data=control_data_baseline,
                treatment_data=treatment_data_no_effect,
            )

            # Effect should be small
            assert abs(results.effect_estimate) < 0.01
            # May or may not be significant depending on random seed

    @pytest.mark.asyncio
    async def test_itt_with_negative_effect(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        control_data_baseline: np.ndarray,
        treatment_data_negative_effect: np.ndarray,
    ):
        """Test ITT analysis detects negative effect."""
        with patch.object(service, "_persist_results", new_callable=AsyncMock):
            results = await service.compute_itt_results(
                experiment_id=experiment_id,
                primary_metric="conversion_rate",
                control_data=control_data_baseline,
                treatment_data=treatment_data_negative_effect,
            )

            assert results.effect_estimate < 0
            assert results.relative_lift < 0

    @pytest.mark.asyncio
    async def test_itt_with_secondary_metrics(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        control_data_baseline: np.ndarray,
        treatment_data_positive_effect: np.ndarray,
        secondary_metrics_data: dict,
    ):
        """Test ITT analysis includes secondary metrics."""
        with patch.object(service, "_persist_results", new_callable=AsyncMock):
            results = await service.compute_itt_results(
                experiment_id=experiment_id,
                primary_metric="conversion_rate",
                control_data=control_data_baseline,
                treatment_data=treatment_data_positive_effect,
                secondary_metrics=secondary_metrics_data,
            )

            assert len(results.secondary_metrics) == 2
            metric_names = [m["name"] for m in results.secondary_metrics]
            assert "clicks" in metric_names
            assert "time_on_page" in metric_names

    @pytest.mark.asyncio
    async def test_itt_interim_analysis_type(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        control_data_baseline: np.ndarray,
        treatment_data_positive_effect: np.ndarray,
    ):
        """Test ITT with interim analysis type."""
        with patch.object(service, "_persist_results", new_callable=AsyncMock):
            results = await service.compute_itt_results(
                experiment_id=experiment_id,
                primary_metric="conversion_rate",
                control_data=control_data_baseline,
                treatment_data=treatment_data_positive_effect,
                analysis_type=AnalysisType.INTERIM,
            )

            assert results.analysis_type == AnalysisType.INTERIM

    @pytest.mark.asyncio
    async def test_itt_calculates_confidence_intervals(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        control_data_baseline: np.ndarray,
        treatment_data_positive_effect: np.ndarray,
    ):
        """Test ITT calculates proper confidence intervals."""
        with patch.object(service, "_persist_results", new_callable=AsyncMock):
            results = await service.compute_itt_results(
                experiment_id=experiment_id,
                primary_metric="conversion_rate",
                control_data=control_data_baseline,
                treatment_data=treatment_data_positive_effect,
            )

            # CI should contain the effect estimate
            assert results.effect_ci_lower < results.effect_estimate < results.effect_ci_upper
            # Relative lift CI should contain the relative lift
            assert results.relative_lift_ci_lower < results.relative_lift < results.relative_lift_ci_upper


# =============================================================================
# TEST COMPUTE PER-PROTOCOL RESULTS
# =============================================================================


class TestComputePerProtocolResults:
    """Tests for compute_per_protocol_results method."""

    @pytest.mark.asyncio
    async def test_per_protocol_filters_non_compliant(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        control_data_baseline: np.ndarray,
        treatment_data_positive_effect: np.ndarray,
    ):
        """Test per-protocol analysis filters non-compliant units."""
        # 80% compliance rate
        np.random.seed(42)
        control_mask = np.random.random(len(control_data_baseline)) < 0.8
        treatment_mask = np.random.random(len(treatment_data_positive_effect)) < 0.8

        with patch.object(service, "_persist_results", new_callable=AsyncMock):
            results = await service.compute_per_protocol_results(
                experiment_id=experiment_id,
                primary_metric="conversion_rate",
                control_data=control_data_baseline,
                treatment_data=treatment_data_positive_effect,
                control_compliant_mask=control_mask,
                treatment_compliant_mask=treatment_mask,
            )

            assert results.analysis_method == AnalysisMethod.PER_PROTOCOL
            # Sample sizes should be smaller due to filtering
            assert results.sample_size_control < len(control_data_baseline)
            assert results.sample_size_treatment < len(treatment_data_positive_effect)

    @pytest.mark.asyncio
    async def test_per_protocol_with_secondary_metrics(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        control_data_baseline: np.ndarray,
        treatment_data_positive_effect: np.ndarray,
        secondary_metrics_data: dict,
    ):
        """Test per-protocol analysis filters secondary metrics too."""
        np.random.seed(42)
        control_mask = np.random.random(len(control_data_baseline)) < 0.9
        treatment_mask = np.random.random(len(treatment_data_positive_effect)) < 0.9

        with patch.object(service, "_persist_results", new_callable=AsyncMock):
            results = await service.compute_per_protocol_results(
                experiment_id=experiment_id,
                primary_metric="conversion_rate",
                control_data=control_data_baseline,
                treatment_data=treatment_data_positive_effect,
                control_compliant_mask=control_mask,
                treatment_compliant_mask=treatment_mask,
                secondary_metrics=secondary_metrics_data,
            )

            # Secondary metrics should have same reduced sample size
            for metric in results.secondary_metrics:
                assert metric["sample_size_control"] <= len(control_data_baseline)

    @pytest.mark.asyncio
    async def test_per_protocol_full_compliance(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        control_data_baseline: np.ndarray,
        treatment_data_positive_effect: np.ndarray,
    ):
        """Test per-protocol with 100% compliance equals ITT."""
        control_mask = np.ones(len(control_data_baseline), dtype=bool)
        treatment_mask = np.ones(len(treatment_data_positive_effect), dtype=bool)

        with patch.object(service, "_persist_results", new_callable=AsyncMock):
            pp_results = await service.compute_per_protocol_results(
                experiment_id=experiment_id,
                primary_metric="conversion_rate",
                control_data=control_data_baseline,
                treatment_data=treatment_data_positive_effect,
                control_compliant_mask=control_mask,
                treatment_compliant_mask=treatment_mask,
            )

            itt_results = await service.compute_itt_results(
                experiment_id=experiment_id,
                primary_metric="conversion_rate",
                control_data=control_data_baseline,
                treatment_data=treatment_data_positive_effect,
            )

            # Results should be very similar
            assert abs(pp_results.effect_estimate - itt_results.effect_estimate) < 0.001


# =============================================================================
# TEST ANALYZE METRIC
# =============================================================================


class TestAnalyzeMetric:
    """Tests for _analyze_metric method."""

    def test_analyze_metric_basic(self, service: ResultsAnalysisService):
        """Test basic metric analysis."""
        np.random.seed(42)
        control = np.random.normal(10.0, 2.0, 100)
        treatment = np.random.normal(12.0, 2.0, 100)

        result = service._analyze_metric("test_metric", control, treatment)

        assert result["name"] == "test_metric"
        assert result["control_mean"] < result["treatment_mean"]
        assert result["effect"] > 0
        assert result["relative_lift"] > 0
        assert "p_value" in result
        assert "is_significant" in result

    def test_analyze_metric_no_effect(self, service: ResultsAnalysisService):
        """Test metric analysis with no effect."""
        np.random.seed(42)
        control = np.random.normal(10.0, 2.0, 100)
        treatment = np.random.normal(10.0, 2.0, 100)

        result = service._analyze_metric("test_metric", control, treatment)

        # Effect should be small
        assert abs(result["effect"]) < 1.0

    def test_analyze_metric_zero_control_mean(self, service: ResultsAnalysisService):
        """Test metric analysis handles zero control mean."""
        control = np.zeros(100)
        treatment = np.ones(100)

        result = service._analyze_metric("test_metric", control, treatment)

        assert result["relative_lift"] == 0  # Avoid division by zero


# =============================================================================
# TEST CALCULATE POWER
# =============================================================================


class TestCalculatePower:
    """Tests for _calculate_power method."""

    def test_power_high_effect(self, service: ResultsAnalysisService):
        """Test power calculation with large effect."""
        power = service._calculate_power(
            effect=0.5,
            se=0.1,
            n_control=100,
            n_treatment=100,
        )
        assert power > 0.9

    def test_power_small_effect(self, service: ResultsAnalysisService):
        """Test power calculation with small effect."""
        power = service._calculate_power(
            effect=0.01,
            se=0.1,
            n_control=100,
            n_treatment=100,
        )
        assert power < 0.5

    def test_power_zero_se(self, service: ResultsAnalysisService):
        """Test power calculation with zero SE."""
        power = service._calculate_power(
            effect=0.5,
            se=0.0,
            n_control=100,
            n_treatment=100,
        )
        assert power == 0.0

    def test_power_bounded(self, service: ResultsAnalysisService):
        """Test power is bounded between 0 and 1."""
        power = service._calculate_power(
            effect=10.0,  # Very large effect
            se=0.1,
            n_control=100,
            n_treatment=100,
        )
        assert 0.0 <= power <= 1.0


# =============================================================================
# TEST COMPUTE HETEROGENEOUS EFFECTS
# =============================================================================


class TestComputeHeterogeneousEffects:
    """Tests for compute_heterogeneous_effects method."""

    @pytest.mark.asyncio
    async def test_hte_multiple_segments(self, service: ResultsAnalysisService):
        """Test HTE analysis with multiple segments."""
        np.random.seed(42)
        segment_data = {
            "high_value": {
                "control": np.random.normal(0.15, 0.03, 200),
                "treatment": np.random.normal(0.20, 0.03, 200),
            },
            "low_value": {
                "control": np.random.normal(0.08, 0.02, 200),
                "treatment": np.random.normal(0.09, 0.02, 200),
            },
        }

        results = await service.compute_heterogeneous_effects(
            experiment_id=uuid4(),
            primary_metric="conversion_rate",
            segment_data=segment_data,
        )

        assert len(results) == 2
        assert "high_value" in results
        assert "low_value" in results
        # High value segment should have larger effect
        assert results["high_value"]["effect"] > results["low_value"]["effect"]

    @pytest.mark.asyncio
    async def test_hte_skips_small_segments(self, service: ResultsAnalysisService):
        """Test HTE skips segments with insufficient data."""
        np.random.seed(42)
        segment_data = {
            "large_segment": {
                "control": np.random.normal(0.10, 0.03, 100),
                "treatment": np.random.normal(0.12, 0.03, 100),
            },
            "tiny_segment": {
                "control": np.random.normal(0.10, 0.03, 5),  # Too small
                "treatment": np.random.normal(0.12, 0.03, 5),
            },
        }

        results = await service.compute_heterogeneous_effects(
            experiment_id=uuid4(),
            primary_metric="conversion_rate",
            segment_data=segment_data,
        )

        assert len(results) == 1
        assert "large_segment" in results
        assert "tiny_segment" not in results

    @pytest.mark.asyncio
    async def test_hte_result_structure(self, service: ResultsAnalysisService):
        """Test HTE result structure."""
        np.random.seed(42)
        segment_data = {
            "test_segment": {
                "control": np.random.normal(0.10, 0.03, 100),
                "treatment": np.random.normal(0.12, 0.03, 100),
            },
        }

        results = await service.compute_heterogeneous_effects(
            experiment_id=uuid4(),
            primary_metric="conversion_rate",
            segment_data=segment_data,
        )

        segment_result = results["test_segment"]
        assert "segment" in segment_result
        assert segment_result["segment"] == "test_segment"
        assert "n_total" in segment_result
        assert segment_result["n_total"] == 200


# =============================================================================
# TEST CHECK SAMPLE RATIO MISMATCH
# =============================================================================


class TestCheckSampleRatioMismatch:
    """Tests for check_sample_ratio_mismatch method."""

    @pytest.mark.asyncio
    async def test_no_srm_balanced_allocation(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
    ):
        """Test no SRM detected with balanced allocation."""
        with patch.object(service, "_persist_srm_check", new_callable=AsyncMock):
            result = await service.check_sample_ratio_mismatch(
                experiment_id=experiment_id,
                expected_ratio={"control": 0.5, "treatment": 0.5},
                actual_counts={"control": 498, "treatment": 502},  # Very close to 50/50
            )

            assert result.is_srm_detected is False
            assert result.severity == SRMSeverity.NONE

    @pytest.mark.asyncio
    async def test_srm_detected_critical(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
    ):
        """Test critical SRM detected with large imbalance."""
        with patch.object(service, "_persist_srm_check", new_callable=AsyncMock):
            result = await service.check_sample_ratio_mismatch(
                experiment_id=experiment_id,
                expected_ratio={"control": 0.5, "treatment": 0.5},
                actual_counts={"control": 400, "treatment": 600},  # 40/60 split
            )

            assert result.is_srm_detected is True
            assert result.severity in [SRMSeverity.WARNING, SRMSeverity.CRITICAL]

    @pytest.mark.asyncio
    async def test_srm_warning_threshold(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
    ):
        """Test SRM warning threshold detection."""
        with patch.object(service, "_persist_srm_check", new_callable=AsyncMock):
            # Moderate imbalance
            result = await service.check_sample_ratio_mismatch(
                experiment_id=experiment_id,
                expected_ratio={"control": 0.5, "treatment": 0.5},
                actual_counts={"control": 470, "treatment": 530},
            )

            # Should be warning or none depending on p-value
            assert result.severity in [SRMSeverity.NONE, SRMSeverity.WARNING]

    @pytest.mark.asyncio
    async def test_srm_multi_arm(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
    ):
        """Test SRM check with multi-arm experiment."""
        with patch.object(service, "_persist_srm_check", new_callable=AsyncMock):
            result = await service.check_sample_ratio_mismatch(
                experiment_id=experiment_id,
                expected_ratio={
                    "control": 0.33,
                    "treatment_a": 0.33,
                    "treatment_b": 0.34,
                },
                actual_counts={
                    "control": 330,
                    "treatment_a": 335,
                    "treatment_b": 335,
                },
            )

            assert result.is_srm_detected is False

    @pytest.mark.asyncio
    async def test_srm_calculates_actual_ratio(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
    ):
        """Test SRM check calculates actual ratios."""
        with patch.object(service, "_persist_srm_check", new_callable=AsyncMock):
            result = await service.check_sample_ratio_mismatch(
                experiment_id=experiment_id,
                expected_ratio={"control": 0.5, "treatment": 0.5},
                actual_counts={"control": 400, "treatment": 600},
            )

            assert result.actual_ratio["control"] == 0.4
            assert result.actual_ratio["treatment"] == 0.6

    @pytest.mark.asyncio
    async def test_srm_zero_total(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
    ):
        """Test SRM check handles zero total."""
        with patch.object(service, "_persist_srm_check", new_callable=AsyncMock):
            result = await service.check_sample_ratio_mismatch(
                experiment_id=experiment_id,
                expected_ratio={"control": 0.5, "treatment": 0.5},
                actual_counts={"control": 0, "treatment": 0},
            )

            assert result.chi_squared_statistic == 0.0
            assert result.p_value == 1.0


# =============================================================================
# TEST COMPARE WITH TWIN PREDICTION
# =============================================================================


class TestCompareWithTwinPrediction:
    """Tests for compare_with_twin_prediction method."""

    @pytest.mark.asyncio
    async def test_fidelity_excellent(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        twin_simulation_id: UUID,
    ):
        """Test excellent fidelity comparison."""
        actual_results = ExperimentResults(
            experiment_id=experiment_id,
            analysis_type=AnalysisType.FINAL,
            analysis_method=AnalysisMethod.ITT,
            computed_at=datetime.now(timezone.utc),
            primary_metric="conversion_rate",
            control_mean=0.10,
            treatment_mean=0.12,
            effect_estimate=0.02,  # Close to predicted
            effect_ci_lower=0.01,
            effect_ci_upper=0.03,
            relative_lift=20.0,
            relative_lift_ci_lower=10.0,
            relative_lift_ci_upper=30.0,
            p_value=0.001,
            is_significant=True,
            sample_size_control=500,
            sample_size_treatment=500,
            statistical_power=0.95,
        )

        with patch.object(service, "_persist_fidelity_comparison", new_callable=AsyncMock):
            result = await service.compare_with_twin_prediction(
                experiment_id=experiment_id,
                twin_simulation_id=twin_simulation_id,
                actual_results=actual_results,
                predicted_effect=0.021,  # Very close prediction
                predicted_ci_lower=0.01,
                predicted_ci_upper=0.03,
            )

            assert result.prediction_error_percent < 10
            assert result.fidelity_grade in ["A", "B"]
            assert result.ci_coverage is True

    @pytest.mark.asyncio
    async def test_fidelity_poor(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        twin_simulation_id: UUID,
    ):
        """Test poor fidelity comparison."""
        actual_results = ExperimentResults(
            experiment_id=experiment_id,
            analysis_type=AnalysisType.FINAL,
            analysis_method=AnalysisMethod.ITT,
            computed_at=datetime.now(timezone.utc),
            primary_metric="conversion_rate",
            control_mean=0.10,
            treatment_mean=0.12,
            effect_estimate=0.02,
            effect_ci_lower=0.01,
            effect_ci_upper=0.03,
            relative_lift=20.0,
            relative_lift_ci_lower=10.0,
            relative_lift_ci_upper=30.0,
            p_value=0.001,
            is_significant=True,
            sample_size_control=500,
            sample_size_treatment=500,
            statistical_power=0.95,
        )

        with patch.object(service, "_persist_fidelity_comparison", new_callable=AsyncMock):
            result = await service.compare_with_twin_prediction(
                experiment_id=experiment_id,
                twin_simulation_id=twin_simulation_id,
                actual_results=actual_results,
                predicted_effect=0.05,  # Very different prediction
                predicted_ci_lower=0.04,
                predicted_ci_upper=0.06,
            )

            assert result.prediction_error_percent > 50
            assert result.fidelity_grade in ["D", "F"]
            assert result.ci_coverage is False

    @pytest.mark.asyncio
    async def test_fidelity_ci_coverage(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        twin_simulation_id: UUID,
    ):
        """Test CI coverage calculation."""
        actual_results = ExperimentResults(
            experiment_id=experiment_id,
            analysis_type=AnalysisType.FINAL,
            analysis_method=AnalysisMethod.ITT,
            computed_at=datetime.now(timezone.utc),
            primary_metric="conversion_rate",
            control_mean=0.10,
            treatment_mean=0.12,
            effect_estimate=0.025,  # Actual effect
            effect_ci_lower=0.01,
            effect_ci_upper=0.04,
            relative_lift=25.0,
            relative_lift_ci_lower=10.0,
            relative_lift_ci_upper=40.0,
            p_value=0.001,
            is_significant=True,
            sample_size_control=500,
            sample_size_treatment=500,
            statistical_power=0.95,
        )

        with patch.object(service, "_persist_fidelity_comparison", new_callable=AsyncMock):
            # Wide CI should contain actual
            result_wide = await service.compare_with_twin_prediction(
                experiment_id=experiment_id,
                twin_simulation_id=twin_simulation_id,
                actual_results=actual_results,
                predicted_effect=0.03,
                predicted_ci_lower=0.01,  # Wide CI includes 0.025
                predicted_ci_upper=0.05,
            )
            assert result_wide.ci_coverage is True

            # Narrow CI should not contain actual
            result_narrow = await service.compare_with_twin_prediction(
                experiment_id=experiment_id,
                twin_simulation_id=twin_simulation_id,
                actual_results=actual_results,
                predicted_effect=0.04,
                predicted_ci_lower=0.035,  # Narrow CI excludes 0.025
                predicted_ci_upper=0.045,
            )
            assert result_narrow.ci_coverage is False

    @pytest.mark.asyncio
    async def test_fidelity_zero_predicted_effect(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
        twin_simulation_id: UUID,
    ):
        """Test fidelity with zero predicted effect."""
        actual_results = ExperimentResults(
            experiment_id=experiment_id,
            analysis_type=AnalysisType.FINAL,
            analysis_method=AnalysisMethod.ITT,
            computed_at=datetime.now(timezone.utc),
            primary_metric="conversion_rate",
            control_mean=0.10,
            treatment_mean=0.11,
            effect_estimate=0.01,
            effect_ci_lower=0.005,
            effect_ci_upper=0.015,
            relative_lift=10.0,
            relative_lift_ci_lower=5.0,
            relative_lift_ci_upper=15.0,
            p_value=0.01,
            is_significant=True,
            sample_size_control=500,
            sample_size_treatment=500,
            statistical_power=0.9,
        )

        with patch.object(service, "_persist_fidelity_comparison", new_callable=AsyncMock):
            result = await service.compare_with_twin_prediction(
                experiment_id=experiment_id,
                twin_simulation_id=twin_simulation_id,
                actual_results=actual_results,
                predicted_effect=0.0,  # Zero prediction
                predicted_ci_lower=-0.01,
                predicted_ci_upper=0.01,
            )

            assert result.prediction_error_percent == 0.0  # Avoid division by zero


# =============================================================================
# TEST FIDELITY SCORE CALCULATION
# =============================================================================


class TestFidelityScoreCalculation:
    """Tests for _calculate_fidelity_score method."""

    def test_excellent_fidelity(self, service: ResultsAnalysisService):
        """Test excellent fidelity score (<10% error)."""
        score = service._calculate_fidelity_score(
            error_percent=5.0,
            ci_coverage=True,
        )
        assert score >= 0.9

    def test_good_fidelity(self, service: ResultsAnalysisService):
        """Test good fidelity score (10-20% error)."""
        score = service._calculate_fidelity_score(
            error_percent=15.0,
            ci_coverage=True,
        )
        assert 0.8 <= score < 1.0

    def test_acceptable_fidelity(self, service: ResultsAnalysisService):
        """Test acceptable fidelity score (20-30% error)."""
        score = service._calculate_fidelity_score(
            error_percent=25.0,
            ci_coverage=True,
        )
        assert 0.6 <= score < 0.8

    def test_poor_fidelity(self, service: ResultsAnalysisService):
        """Test poor fidelity score (>30% error)."""
        score = service._calculate_fidelity_score(
            error_percent=50.0,
            ci_coverage=False,
        )
        assert score < 0.6

    def test_ci_coverage_bonus(self, service: ResultsAnalysisService):
        """Test CI coverage adds bonus to score."""
        score_without = service._calculate_fidelity_score(
            error_percent=15.0,
            ci_coverage=False,
        )
        score_with = service._calculate_fidelity_score(
            error_percent=15.0,
            ci_coverage=True,
        )
        assert score_with > score_without

    def test_score_bounded(self, service: ResultsAnalysisService):
        """Test fidelity score is bounded 0-1."""
        score_high = service._calculate_fidelity_score(
            error_percent=0.0,
            ci_coverage=True,
        )
        score_low = service._calculate_fidelity_score(
            error_percent=100.0,
            ci_coverage=False,
        )
        assert 0.0 <= score_high <= 1.0
        assert 0.0 <= score_low <= 1.0


# =============================================================================
# TEST FIDELITY GRADE ASSIGNMENT
# =============================================================================


class TestFidelityGradeAssignment:
    """Tests for _assign_fidelity_grade method."""

    def test_grade_a(self, service: ResultsAnalysisService):
        """Test grade A assignment."""
        assert service._assign_fidelity_grade(0.95) == "A"
        assert service._assign_fidelity_grade(0.90) == "A"

    def test_grade_b(self, service: ResultsAnalysisService):
        """Test grade B assignment."""
        assert service._assign_fidelity_grade(0.89) == "B"
        assert service._assign_fidelity_grade(0.80) == "B"

    def test_grade_c(self, service: ResultsAnalysisService):
        """Test grade C assignment."""
        assert service._assign_fidelity_grade(0.79) == "C"
        assert service._assign_fidelity_grade(0.70) == "C"

    def test_grade_d(self, service: ResultsAnalysisService):
        """Test grade D assignment."""
        assert service._assign_fidelity_grade(0.69) == "D"
        assert service._assign_fidelity_grade(0.60) == "D"

    def test_grade_f(self, service: ResultsAnalysisService):
        """Test grade F assignment."""
        assert service._assign_fidelity_grade(0.59) == "F"
        assert service._assign_fidelity_grade(0.0) == "F"


# =============================================================================
# TEST CALIBRATION RECOMMENDATIONS
# =============================================================================


class TestCalibrationRecommendations:
    """Tests for _generate_calibration_recommendations method."""

    def test_recommendations_overestimated(self, service: ResultsAnalysisService):
        """Test recommendations when twin overestimated."""
        recommendations = service._generate_calibration_recommendations(
            prediction_error=-0.02,  # Actual < predicted
            prediction_error_percent=25.0,
            ci_coverage=True,
        )

        assert recommendations["direction"] == "overestimated"
        assert recommendations["magnitude_adjustment"] == 0.02

    def test_recommendations_underestimated(self, service: ResultsAnalysisService):
        """Test recommendations when twin underestimated."""
        recommendations = service._generate_calibration_recommendations(
            prediction_error=0.02,  # Actual > predicted
            prediction_error_percent=25.0,
            ci_coverage=True,
        )

        assert recommendations["direction"] == "underestimated"
        assert recommendations["magnitude_adjustment"] == -0.02

    def test_recommendations_needs_calibration(self, service: ResultsAnalysisService):
        """Test needs_calibration flag."""
        recommendations_yes = service._generate_calibration_recommendations(
            prediction_error=0.05,
            prediction_error_percent=25.0,  # > 20%
            ci_coverage=True,
        )
        assert recommendations_yes["needs_calibration"] is True

        recommendations_no = service._generate_calibration_recommendations(
            prediction_error=0.01,
            prediction_error_percent=10.0,  # < 20%
            ci_coverage=True,
        )
        assert recommendations_no["needs_calibration"] is False

    def test_recommendations_suggestions_large_error(self, service: ResultsAnalysisService):
        """Test suggestions for large prediction error."""
        recommendations = service._generate_calibration_recommendations(
            prediction_error=0.1,
            prediction_error_percent=35.0,
            ci_coverage=True,
        )

        assert len(recommendations["suggestions"]) > 0
        assert any("parameter" in s.lower() for s in recommendations["suggestions"])

    def test_recommendations_suggestions_ci_miss(self, service: ResultsAnalysisService):
        """Test suggestions when CI doesn't cover actual."""
        recommendations = service._generate_calibration_recommendations(
            prediction_error=0.05,
            prediction_error_percent=25.0,
            ci_coverage=False,
        )

        assert any("uncertainty" in s.lower() or "widening" in s.lower()
                   for s in recommendations["suggestions"])


# =============================================================================
# TEST PERSISTENCE METHODS
# =============================================================================


class TestPersistenceMethods:
    """Tests for persistence methods."""

    @pytest.mark.asyncio
    async def test_persist_results_calls_repository(
        self,
        service: ResultsAnalysisService,
    ):
        """Test _persist_results calls repository."""
        results = ExperimentResults(
            experiment_id=uuid4(),
            analysis_type=AnalysisType.FINAL,
            analysis_method=AnalysisMethod.ITT,
            computed_at=datetime.now(timezone.utc),
            primary_metric="conversion_rate",
            control_mean=0.10,
            treatment_mean=0.12,
            effect_estimate=0.02,
            effect_ci_lower=0.01,
            effect_ci_upper=0.03,
            relative_lift=20.0,
            relative_lift_ci_lower=10.0,
            relative_lift_ci_upper=30.0,
            p_value=0.001,
            is_significant=True,
            sample_size_control=500,
            sample_size_treatment=500,
            statistical_power=0.95,
        )

        with patch("src.repositories.ab_results.ABResultsRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo

            await service._persist_results(results)

            mock_repo.save_results.assert_called_once_with(results)

    @pytest.mark.asyncio
    async def test_persist_srm_check_calls_repository(
        self,
        service: ResultsAnalysisService,
    ):
        """Test _persist_srm_check calls repository."""
        result = SRMCheckResult(
            experiment_id=uuid4(),
            checked_at=datetime.now(timezone.utc),
            expected_ratio={"control": 0.5, "treatment": 0.5},
            actual_counts={"control": 500, "treatment": 500},
            actual_ratio={"control": 0.5, "treatment": 0.5},
            chi_squared_statistic=0.0,
            p_value=1.0,
            is_srm_detected=False,
            severity=SRMSeverity.NONE,
        )

        with patch("src.repositories.ab_results.ABResultsRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo

            await service._persist_srm_check(result)

            mock_repo.save_srm_check.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_persist_fidelity_comparison_calls_repository(
        self,
        service: ResultsAnalysisService,
    ):
        """Test _persist_fidelity_comparison calls repository."""
        result = FidelityComparison(
            experiment_id=uuid4(),
            twin_simulation_id=uuid4(),
            comparison_timestamp=datetime.now(timezone.utc),
            predicted_effect=0.02,
            actual_effect=0.02,
            prediction_error=0.0,
            prediction_error_percent=0.0,
            predicted_ci_lower=0.01,
            predicted_ci_upper=0.03,
            ci_coverage=True,
            fidelity_score=1.0,
            fidelity_grade="A",
        )

        with patch("src.repositories.ab_results.ABResultsRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo

            await service._persist_fidelity_comparison(result)

            mock_repo.save_fidelity_comparison.assert_called_once_with(result)


# =============================================================================
# TEST FACTORY FUNCTION
# =============================================================================


class TestFactoryFunction:
    """Tests for get_results_analysis_service factory."""

    def test_factory_default_config(self):
        """Test factory with default config."""
        service = get_results_analysis_service()
        assert isinstance(service, ResultsAnalysisService)
        assert service.config.alpha == 0.05

    def test_factory_custom_config(self):
        """Test factory with custom config."""
        config = ResultsAnalysisConfig(
            alpha=0.10,
            power_threshold=0.9,
            srm_p_threshold=0.0001,
        )
        service = get_results_analysis_service(config)

        assert service.config.alpha == 0.10
        assert service.config.power_threshold == 0.9
        assert service.config.srm_p_threshold == 0.0001


# =============================================================================
# TEST EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_init_without_config(self):
        """Test service initialization without config."""
        service = ResultsAnalysisService()
        assert service.config is not None
        assert service.config.alpha == 0.05

    @pytest.mark.asyncio
    async def test_itt_with_small_samples(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
    ):
        """Test ITT with very small sample sizes."""
        np.random.seed(42)
        control = np.random.normal(0.10, 0.03, 5)
        treatment = np.random.normal(0.12, 0.03, 5)

        with patch.object(service, "_persist_results", new_callable=AsyncMock):
            results = await service.compute_itt_results(
                experiment_id=experiment_id,
                primary_metric="test",
                control_data=control,
                treatment_data=treatment,
            )

            # Should still produce valid results
            assert results.sample_size_control == 5
            assert results.sample_size_treatment == 5
            assert 0 <= results.p_value <= 1

    @pytest.mark.asyncio
    async def test_itt_identical_data(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
    ):
        """Test ITT when control and treatment are identical."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with patch.object(service, "_persist_results", new_callable=AsyncMock):
            results = await service.compute_itt_results(
                experiment_id=experiment_id,
                primary_metric="test",
                control_data=data,
                treatment_data=data.copy(),
            )

            assert results.effect_estimate == 0.0
            assert results.p_value == 1.0
            assert results.is_significant == False  # Use == for numpy bool compatibility

    @pytest.mark.asyncio
    async def test_hte_empty_segment_data(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
    ):
        """Test HTE with empty segment data."""
        results = await service.compute_heterogeneous_effects(
            experiment_id=experiment_id,
            primary_metric="test",
            segment_data={},
        )

        assert results == {}

    @pytest.mark.asyncio
    async def test_hte_segment_missing_keys(
        self,
        service: ResultsAnalysisService,
        experiment_id: UUID,
    ):
        """Test HTE handles missing keys in segment data."""
        segment_data = {
            "incomplete": {
                "control": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                # Missing "treatment" key
            },
        }

        results = await service.compute_heterogeneous_effects(
            experiment_id=experiment_id,
            primary_metric="test",
            segment_data=segment_data,
        )

        # Should skip incomplete segment
        assert "incomplete" not in results
