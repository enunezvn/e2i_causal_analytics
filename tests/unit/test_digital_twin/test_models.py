"""
Unit Tests for Digital Twin Pydantic Models.

Tests cover:
- Twin models: TwinType, Brand, Region enums
- Feature models: HCPTwinFeatures, PatientTwinFeatures, TerritoryTwinFeatures
- Core models: DigitalTwin, TwinPopulation
- Config models: TwinModelConfig, TwinModelMetrics
- Simulation models: InterventionConfig, PopulationFilter, SimulationResult
- Fidelity models: FidelityRecord, FidelityGrade
"""

from datetime import datetime
from typing import Any, Dict
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from src.digital_twin.models.twin_models import (
    Brand,
    DigitalTwin,
    HCPTwinFeatures,
    PatientTwinFeatures,
    Region,
    TerritoryTwinFeatures,
    TwinModelConfig,
    TwinModelMetrics,
    TwinPopulation,
    TwinType,
)
from src.digital_twin.models.simulation_models import (
    EffectHeterogeneity,
    FidelityGrade,
    FidelityRecord,
    InterventionConfig,
    PopulationFilter,
    SimulationRecommendation,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_hcp_features() -> Dict[str, Any]:
    """Create sample HCP twin features."""
    return {
        "specialty": "rheumatology",
        "sub_specialty": "autoimmune",
        "years_experience": 15,
        "practice_type": "academic",
        "practice_size": "large",
        "region": Region.NORTHEAST,
        "state": "NY",
        "urban_rural": "urban",
        "decile": 8,
        "priority_tier": 1,
        "total_patient_volume": 5000,
        "target_patient_volume": 500,
        "prescribing_volume": 200,
        "digital_engagement_score": 0.75,
        "preferred_channel": "email",
        "last_interaction_days": 14,
        "interaction_frequency": 2.5,
        "adoption_stage": "early_adopter",
        "peer_influence_score": 0.85,
    }


@pytest.fixture
def sample_patient_features() -> Dict[str, Any]:
    """Create sample patient twin features."""
    return {
        "age_group": "45-54",
        "gender": "female",
        "geographic_region": Region.SOUTH,
        "socioeconomic_index": 0.65,
        "primary_diagnosis_code": "L40.0",
        "comorbidity_count": 2,
        "risk_score": 0.45,
        "journey_complexity_score": 0.60,
        "insurance_type": "commercial",
        "insurance_coverage_flag": True,
        "prior_auth_required": True,
        "journey_stage": "treatment",
        "journey_duration_days": 180,
        "treatment_line": 2,
    }


@pytest.fixture
def sample_territory_features() -> Dict[str, Any]:
    """Create sample territory twin features."""
    return {
        "region": Region.MIDWEST,
        "state_count": 5,
        "zip_count": 150,
        "total_hcps": 500,
        "covered_hcps": 400,
        "coverage_rate": 0.80,
        "total_patient_volume": 25000,
        "market_share": 0.35,
        "growth_rate": 0.08,
        "competitor_presence": 0.55,
    }


@pytest.fixture
def sample_intervention_config() -> InterventionConfig:
    """Create sample intervention configuration."""
    return InterventionConfig(
        intervention_type="email_campaign",
        channel="email",
        frequency="weekly",
        duration_weeks=8,
        content_type="clinical_data",
        personalization_level="high",
        target_segment="high_value",
        target_deciles=[1, 2, 3],
        target_specialties=["rheumatology", "dermatology"],
        target_regions=["northeast"],
        intensity_multiplier=1.5,
    )


@pytest.fixture
def sample_model_id() -> UUID:
    """Create sample model ID."""
    return uuid4()


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestTwinEnums:
    """Tests for Digital Twin enums."""

    def test_twin_type_values(self):
        """Test TwinType enum has expected values."""
        assert TwinType.HCP.value == "hcp"
        assert TwinType.PATIENT.value == "patient"
        assert TwinType.TERRITORY.value == "territory"
        assert len(TwinType) == 3

    def test_brand_values(self):
        """Test Brand enum has expected values."""
        assert Brand.REMIBRUTINIB.value == "Remibrutinib"
        assert Brand.FABHALTA.value == "Fabhalta"
        assert Brand.KISQALI.value == "Kisqali"
        assert len(Brand) == 3

    def test_region_values(self):
        """Test Region enum has expected values."""
        assert Region.NORTHEAST.value == "northeast"
        assert Region.SOUTH.value == "south"
        assert Region.MIDWEST.value == "midwest"
        assert Region.WEST.value == "west"
        assert len(Region) == 4


class TestSimulationEnums:
    """Tests for Simulation-related enums."""

    def test_simulation_status_values(self):
        """Test SimulationStatus enum has expected values."""
        assert SimulationStatus.PENDING.value == "pending"
        assert SimulationStatus.RUNNING.value == "running"
        assert SimulationStatus.COMPLETED.value == "completed"
        assert SimulationStatus.FAILED.value == "failed"
        assert len(SimulationStatus) == 4

    def test_simulation_recommendation_values(self):
        """Test SimulationRecommendation enum has expected values."""
        assert SimulationRecommendation.DEPLOY.value == "deploy"
        assert SimulationRecommendation.SKIP.value == "skip"
        assert SimulationRecommendation.REFINE.value == "refine"
        assert len(SimulationRecommendation) == 3

    def test_fidelity_grade_values(self):
        """Test FidelityGrade enum has expected values."""
        assert FidelityGrade.EXCELLENT.value == "excellent"
        assert FidelityGrade.GOOD.value == "good"
        assert FidelityGrade.FAIR.value == "fair"
        assert FidelityGrade.POOR.value == "poor"
        assert FidelityGrade.UNVALIDATED.value == "unvalidated"
        assert len(FidelityGrade) == 5


# =============================================================================
# FEATURE MODEL TESTS
# =============================================================================


class TestHCPTwinFeatures:
    """Tests for HCPTwinFeatures model."""

    def test_valid_hcp_features(self, sample_hcp_features):
        """Test creating valid HCP features."""
        features = HCPTwinFeatures(**sample_hcp_features)
        assert features.specialty == "rheumatology"
        assert features.years_experience == 15
        assert features.decile == 8
        assert features.region == Region.NORTHEAST

    def test_decile_validation_min(self, sample_hcp_features):
        """Test decile minimum validation."""
        sample_hcp_features["decile"] = 0
        with pytest.raises(ValidationError):
            HCPTwinFeatures(**sample_hcp_features)

    def test_decile_validation_max(self, sample_hcp_features):
        """Test decile maximum validation."""
        sample_hcp_features["decile"] = 11
        with pytest.raises(ValidationError):
            HCPTwinFeatures(**sample_hcp_features)

    def test_digital_engagement_score_bounds(self, sample_hcp_features):
        """Test digital engagement score bounds."""
        sample_hcp_features["digital_engagement_score"] = 1.5
        with pytest.raises(ValidationError):
            HCPTwinFeatures(**sample_hcp_features)

    def test_years_experience_bounds(self, sample_hcp_features):
        """Test years experience bounds."""
        sample_hcp_features["years_experience"] = -1
        with pytest.raises(ValidationError):
            HCPTwinFeatures(**sample_hcp_features)


class TestPatientTwinFeatures:
    """Tests for PatientTwinFeatures model."""

    def test_valid_patient_features(self, sample_patient_features):
        """Test creating valid patient features."""
        features = PatientTwinFeatures(**sample_patient_features)
        assert features.age_group == "45-54"
        assert features.risk_score == 0.45
        assert features.treatment_line == 2

    def test_risk_score_bounds(self, sample_patient_features):
        """Test risk score bounds."""
        sample_patient_features["risk_score"] = 1.5
        with pytest.raises(ValidationError):
            PatientTwinFeatures(**sample_patient_features)

    def test_treatment_line_minimum(self, sample_patient_features):
        """Test treatment line minimum."""
        sample_patient_features["treatment_line"] = 0
        with pytest.raises(ValidationError):
            PatientTwinFeatures(**sample_patient_features)


class TestTerritoryTwinFeatures:
    """Tests for TerritoryTwinFeatures model."""

    def test_valid_territory_features(self, sample_territory_features):
        """Test creating valid territory features."""
        features = TerritoryTwinFeatures(**sample_territory_features)
        assert features.region == Region.MIDWEST
        assert features.coverage_rate == 0.80
        assert features.market_share == 0.35

    def test_coverage_rate_bounds(self, sample_territory_features):
        """Test coverage rate bounds."""
        sample_territory_features["coverage_rate"] = 1.5
        with pytest.raises(ValidationError):
            TerritoryTwinFeatures(**sample_territory_features)


# =============================================================================
# DIGITAL TWIN MODEL TESTS
# =============================================================================


class TestDigitalTwin:
    """Tests for DigitalTwin model."""

    def test_create_hcp_twin(self, sample_hcp_features):
        """Test creating HCP digital twin."""
        twin = DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            features=sample_hcp_features,
            baseline_outcome=0.15,
            baseline_propensity=0.65,
        )
        assert twin.twin_type == TwinType.HCP
        assert twin.brand == Brand.REMIBRUTINIB
        assert twin.baseline_outcome == 0.15
        assert isinstance(twin.twin_id, UUID)

    def test_features_cannot_be_empty(self):
        """Test that features dictionary cannot be empty."""
        with pytest.raises(ValidationError):
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={},
                baseline_outcome=0.15,
                baseline_propensity=0.65,
            )

    def test_propensity_bounds(self, sample_hcp_features):
        """Test baseline propensity bounds."""
        with pytest.raises(ValidationError):
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features=sample_hcp_features,
                baseline_outcome=0.15,
                baseline_propensity=1.5,
            )

    def test_twin_has_generation_timestamp(self, sample_hcp_features):
        """Test that twin has generation timestamp."""
        twin = DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            features=sample_hcp_features,
            baseline_outcome=0.15,
            baseline_propensity=0.65,
        )
        assert isinstance(twin.generation_timestamp, datetime)


class TestTwinPopulation:
    """Tests for TwinPopulation model."""

    def test_create_population(self, sample_hcp_features):
        """Test creating a twin population."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={**sample_hcp_features, "decile": i % 10 + 1},
                baseline_outcome=0.1 + i * 0.01,
                baseline_propensity=0.5,
            )
            for i in range(10)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=10,
        )

        assert len(population) == 10
        assert population.twin_type == TwinType.HCP

    def test_population_filter(self, sample_hcp_features):
        """Test filtering a twin population."""
        twins = [
            DigitalTwin(
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                features={**sample_hcp_features, "specialty": "rheumatology" if i % 2 == 0 else "dermatology"},
                baseline_outcome=0.15,
                baseline_propensity=0.5,
            )
            for i in range(10)
        ]

        population = TwinPopulation(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            twins=twins,
            size=10,
        )

        filtered = population.filter(specialty="rheumatology")
        assert len(filtered) == 5


# =============================================================================
# MODEL CONFIG TESTS
# =============================================================================


class TestTwinModelConfig:
    """Tests for TwinModelConfig model."""

    def test_valid_config(self):
        """Test creating valid model configuration."""
        config = TwinModelConfig(
            model_name="hcp_remibrutinib_v1",
            model_description="HCP twin model for Remibrutinib",
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
            feature_columns=["specialty", "decile", "region"],
            target_column="conversion_rate",
        )
        assert config.model_name == "hcp_remibrutinib_v1"
        assert config.algorithm == "random_forest"  # default
        assert config.n_estimators == 100  # default

    def test_estimators_bounds(self):
        """Test n_estimators bounds."""
        with pytest.raises(ValidationError):
            TwinModelConfig(
                model_name="test",
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                feature_columns=["a"],
                target_column="b",
                n_estimators=5,  # below minimum
            )

    def test_validation_split_bounds(self):
        """Test validation split bounds."""
        with pytest.raises(ValidationError):
            TwinModelConfig(
                model_name="test",
                twin_type=TwinType.HCP,
                brand=Brand.REMIBRUTINIB,
                feature_columns=["a"],
                target_column="b",
                validation_split=0.6,  # above maximum
            )


class TestTwinModelMetrics:
    """Tests for TwinModelMetrics model."""

    def test_valid_metrics(self, sample_model_id):
        """Test creating valid model metrics."""
        metrics = TwinModelMetrics(
            model_id=sample_model_id,
            r2_score=0.85,
            rmse=0.05,
            mae=0.03,
            cv_scores=[0.83, 0.85, 0.84, 0.86, 0.84],
            cv_mean=0.844,
            cv_std=0.01,
            feature_importances={"decile": 0.35, "specialty": 0.25, "region": 0.15},
            top_features=["decile", "specialty", "region"],
            training_samples=50000,
            training_duration_seconds=120.5,
        )
        assert metrics.r2_score == 0.85
        assert len(metrics.cv_scores) == 5

    def test_metrics_to_dict(self, sample_model_id):
        """Test metrics to_dict method."""
        metrics = TwinModelMetrics(
            model_id=sample_model_id,
            r2_score=0.85,
            rmse=0.05,
            mae=0.03,
            training_samples=50000,
            training_duration_seconds=120.5,
        )
        result = metrics.to_dict()
        assert "r2_score" in result
        assert "rmse" in result
        assert result["training_samples"] == 50000


# =============================================================================
# INTERVENTION CONFIG TESTS
# =============================================================================


class TestInterventionConfig:
    """Tests for InterventionConfig model."""

    def test_valid_intervention(self, sample_intervention_config):
        """Test creating valid intervention configuration."""
        assert sample_intervention_config.intervention_type == "email_campaign"
        assert sample_intervention_config.duration_weeks == 8
        assert sample_intervention_config.intensity_multiplier == 1.5

    def test_decile_validation(self):
        """Test decile validation in target_deciles."""
        with pytest.raises(ValidationError):
            InterventionConfig(
                intervention_type="email_campaign",
                target_deciles=[0, 11],  # Invalid deciles
            )

    def test_duration_bounds(self):
        """Test duration weeks bounds."""
        with pytest.raises(ValidationError):
            InterventionConfig(
                intervention_type="email_campaign",
                duration_weeks=60,  # Above max 52
            )

    def test_intensity_bounds(self):
        """Test intensity multiplier bounds."""
        with pytest.raises(ValidationError):
            InterventionConfig(
                intervention_type="email_campaign",
                intensity_multiplier=15.0,  # Above max 10
            )


class TestPopulationFilter:
    """Tests for PopulationFilter model."""

    def test_valid_filter(self):
        """Test creating valid population filter."""
        filter_ = PopulationFilter(
            specialties=["rheumatology", "dermatology"],
            deciles=[1, 2, 3],
            regions=["northeast"],
            min_baseline_outcome=0.05,
            max_baseline_outcome=0.25,
        )
        assert len(filter_.specialties) == 2
        assert filter_.min_baseline_outcome == 0.05

    def test_to_dict(self):
        """Test filter to_dict method."""
        filter_ = PopulationFilter(
            specialties=["rheumatology"],
            deciles=[1, 2],
        )
        result = filter_.to_dict()
        assert result["specialties"] == ["rheumatology"]
        assert result["deciles"] == [1, 2]


# =============================================================================
# SIMULATION RESULT TESTS
# =============================================================================


class TestSimulationResult:
    """Tests for SimulationResult model."""

    def test_valid_simulation_result(self, sample_intervention_config, sample_model_id):
        """Test creating valid simulation result."""
        result = SimulationResult(
            model_id=sample_model_id,
            intervention_config=sample_intervention_config,
            twin_count=10000,
            simulated_ate=0.08,
            simulated_ci_lower=0.05,
            simulated_ci_upper=0.11,
            simulated_std_error=0.015,
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Effect is significant and positive",
            simulation_confidence=0.92,
            execution_time_ms=1500,
        )
        assert result.simulated_ate == 0.08
        assert result.recommendation == SimulationRecommendation.DEPLOY
        assert result.is_significant()

    def test_ci_bounds_validation(self, sample_intervention_config, sample_model_id):
        """Test CI bounds validation (lower must be <= upper)."""
        with pytest.raises(ValidationError):
            SimulationResult(
                model_id=sample_model_id,
                intervention_config=sample_intervention_config,
                twin_count=10000,
                simulated_ate=0.08,
                simulated_ci_lower=0.15,  # Higher than upper
                simulated_ci_upper=0.05,  # Lower than lower
                simulated_std_error=0.015,
                recommendation=SimulationRecommendation.DEPLOY,
                recommendation_rationale="Test",
                simulation_confidence=0.92,
                execution_time_ms=1500,
            )

    def test_is_significant_positive(self, sample_intervention_config, sample_model_id):
        """Test significance detection for positive effect."""
        result = SimulationResult(
            model_id=sample_model_id,
            intervention_config=sample_intervention_config,
            twin_count=10000,
            simulated_ate=0.08,
            simulated_ci_lower=0.02,  # Both positive
            simulated_ci_upper=0.14,
            simulated_std_error=0.03,
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Test",
            simulation_confidence=0.90,
            execution_time_ms=1000,
        )
        assert result.is_significant() is True

    def test_is_significant_crosses_zero(self, sample_intervention_config, sample_model_id):
        """Test significance detection when CI crosses zero."""
        result = SimulationResult(
            model_id=sample_model_id,
            intervention_config=sample_intervention_config,
            twin_count=10000,
            simulated_ate=0.02,
            simulated_ci_lower=-0.03,  # Crosses zero
            simulated_ci_upper=0.07,
            simulated_std_error=0.025,
            recommendation=SimulationRecommendation.REFINE,
            recommendation_rationale="Effect not significant",
            simulation_confidence=0.60,
            execution_time_ms=1000,
        )
        assert result.is_significant() is False

    def test_effect_direction(self, sample_intervention_config, sample_model_id):
        """Test effect direction detection."""
        positive_result = SimulationResult(
            model_id=sample_model_id,
            intervention_config=sample_intervention_config,
            twin_count=10000,
            simulated_ate=0.08,
            simulated_ci_lower=0.05,
            simulated_ci_upper=0.11,
            simulated_std_error=0.015,
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Test",
            simulation_confidence=0.92,
            execution_time_ms=1000,
        )
        assert positive_result.effect_direction() == "positive"

        negative_result = SimulationResult(
            model_id=sample_model_id,
            intervention_config=sample_intervention_config,
            twin_count=10000,
            simulated_ate=-0.05,
            simulated_ci_lower=-0.08,
            simulated_ci_upper=-0.02,
            simulated_std_error=0.015,
            recommendation=SimulationRecommendation.SKIP,
            recommendation_rationale="Negative effect",
            simulation_confidence=0.92,
            execution_time_ms=1000,
        )
        assert negative_result.effect_direction() == "negative"

    def test_to_summary_dict(self, sample_intervention_config, sample_model_id):
        """Test to_summary_dict method."""
        result = SimulationResult(
            model_id=sample_model_id,
            intervention_config=sample_intervention_config,
            twin_count=10000,
            simulated_ate=0.08,
            simulated_ci_lower=0.05,
            simulated_ci_upper=0.11,
            simulated_std_error=0.015,
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Effect is significant",
            simulation_confidence=0.92,
            execution_time_ms=1500,
        )
        summary = result.to_summary_dict()
        assert "simulation_id" in summary
        assert summary["simulated_ate"] == 0.08
        assert summary["recommendation"] == "deploy"
        assert summary["is_significant"] is True


class TestEffectHeterogeneity:
    """Tests for EffectHeterogeneity model."""

    def test_get_top_segments(self):
        """Test getting top segments by effect size."""
        heterogeneity = EffectHeterogeneity(
            by_specialty={
                "rheumatology": {"ate": 0.12, "n": 500},
                "dermatology": {"ate": 0.08, "n": 400},
            },
            by_decile={
                "1": {"ate": 0.15, "n": 200},
                "5": {"ate": 0.05, "n": 200},
            },
            by_region={
                "northeast": {"ate": 0.10, "n": 300},
            },
        )

        top = heterogeneity.get_top_segments(n=3)
        assert len(top) == 3
        assert top[0]["ate"] == 0.15  # Highest effect
        assert top[0]["dimension"] == "decile"
        assert top[0]["segment"] == "1"


# =============================================================================
# FIDELITY RECORD TESTS
# =============================================================================


class TestFidelityRecord:
    """Tests for FidelityRecord model."""

    def test_valid_fidelity_record(self):
        """Test creating valid fidelity record."""
        record = FidelityRecord(
            simulation_id=uuid4(),
            simulated_ate=0.08,
            simulated_ci_lower=0.05,
            simulated_ci_upper=0.11,
        )
        assert record.fidelity_grade == FidelityGrade.UNVALIDATED
        assert record.actual_ate is None

    def test_calculate_fidelity_excellent(self):
        """Test fidelity calculation for excellent grade."""
        record = FidelityRecord(
            simulation_id=uuid4(),
            simulated_ate=0.10,
            simulated_ci_lower=0.07,
            simulated_ci_upper=0.13,
            actual_ate=0.095,  # 5% error
        )
        record.calculate_fidelity()
        assert record.fidelity_grade == FidelityGrade.EXCELLENT
        assert record.ci_coverage is True

    def test_calculate_fidelity_good(self):
        """Test fidelity calculation for good grade."""
        record = FidelityRecord(
            simulation_id=uuid4(),
            simulated_ate=0.10,
            simulated_ci_lower=0.07,
            simulated_ci_upper=0.13,
            actual_ate=0.085,  # 15% error
        )
        record.calculate_fidelity()
        assert record.fidelity_grade == FidelityGrade.GOOD

    def test_calculate_fidelity_fair(self):
        """Test fidelity calculation for fair grade."""
        record = FidelityRecord(
            simulation_id=uuid4(),
            simulated_ate=0.10,
            simulated_ci_lower=0.07,
            simulated_ci_upper=0.13,
            actual_ate=0.075,  # 25% error
        )
        record.calculate_fidelity()
        assert record.fidelity_grade == FidelityGrade.FAIR

    def test_calculate_fidelity_poor(self):
        """Test fidelity calculation for poor grade."""
        record = FidelityRecord(
            simulation_id=uuid4(),
            simulated_ate=0.10,
            simulated_ci_lower=0.07,
            simulated_ci_upper=0.13,
            actual_ate=0.05,  # 50% error
        )
        record.calculate_fidelity()
        assert record.fidelity_grade == FidelityGrade.POOR

    def test_ci_coverage_true(self):
        """Test CI coverage when actual falls within predicted CI."""
        record = FidelityRecord(
            simulation_id=uuid4(),
            simulated_ate=0.10,
            simulated_ci_lower=0.07,
            simulated_ci_upper=0.13,
            actual_ate=0.11,  # Within CI
        )
        record.calculate_fidelity()
        assert record.ci_coverage is True

    def test_ci_coverage_false(self):
        """Test CI coverage when actual falls outside predicted CI."""
        record = FidelityRecord(
            simulation_id=uuid4(),
            simulated_ate=0.10,
            simulated_ci_lower=0.07,
            simulated_ci_upper=0.13,
            actual_ate=0.15,  # Outside CI
        )
        record.calculate_fidelity()
        assert record.ci_coverage is False


# =============================================================================
# SIMULATION REQUEST TESTS
# =============================================================================


class TestSimulationRequest:
    """Tests for SimulationRequest model."""

    def test_valid_request(self):
        """Test creating valid simulation request."""
        request = SimulationRequest(
            intervention_type="email_campaign",
            intervention_config={"channel": "email", "frequency": "weekly"},
            brand="Remibrutinib",
            target_population="hcp",
            twin_count=10000,
        )
        assert request.intervention_type == "email_campaign"
        assert request.twin_count == 10000

    def test_twin_count_bounds(self):
        """Test twin count bounds."""
        with pytest.raises(ValidationError):
            SimulationRequest(
                intervention_type="email_campaign",
                intervention_config={},
                brand="Remibrutinib",
                twin_count=50,  # Below minimum 100
            )

    def test_to_intervention_config(self):
        """Test conversion to InterventionConfig."""
        request = SimulationRequest(
            intervention_type="call_frequency",
            intervention_config={
                "channel": "phone",
                "frequency": "daily",
                "duration_weeks": 4,
            },
            brand="Remibrutinib",
        )
        config = request.to_intervention_config()
        assert isinstance(config, InterventionConfig)
        assert config.intervention_type == "call_frequency"
        assert config.channel == "phone"

    def test_to_population_filter(self):
        """Test conversion to PopulationFilter."""
        request = SimulationRequest(
            intervention_type="email_campaign",
            intervention_config={},
            brand="Remibrutinib",
            population_filters={
                "specialties": ["rheumatology"],
                "deciles": [1, 2, 3],
            },
        )
        filter_ = request.to_population_filter()
        assert isinstance(filter_, PopulationFilter)
        assert filter_.specialties == ["rheumatology"]
        assert filter_.deciles == [1, 2, 3]

    def test_confidence_level_bounds(self):
        """Test confidence level bounds."""
        with pytest.raises(ValidationError):
            SimulationRequest(
                intervention_type="email_campaign",
                intervention_config={},
                brand="Remibrutinib",
                confidence_level=0.5,  # Below minimum 0.8
            )
