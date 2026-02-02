"""
Unit tests for digital twin API routes.

Tests all endpoints in src/api/routes/digital_twin.py including:
- Digital Twin simulation
- Simulation listing and filtering
- Fidelity validation
- Twin model management
- Fidelity reporting
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4, UUID

from fastapi import HTTPException


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_twin_generator():
    """Mock TwinGenerator."""
    with patch("src.digital_twin.twin_generator.TwinGenerator") as mock_gen:
        instance = MagicMock()
        mock_gen.return_value = instance

        # Mock population
        mock_population = MagicMock()
        mock_population.get_size.return_value = 1000
        instance.generate.return_value = mock_population

        instance.model_id = uuid4()

        yield instance


@pytest.fixture
def mock_simulation_engine():
    """Mock SimulationEngine."""
    with patch("src.digital_twin.simulation_engine.SimulationEngine") as mock_engine:
        instance = MagicMock()
        mock_engine.return_value = instance

        # Mock simulation result
        mock_result = MagicMock()
        mock_result.simulation_id = uuid4()
        mock_result.model_id = uuid4()
        mock_result.twin_count = 1000
        mock_result.simulated_ate = 0.075
        mock_result.simulated_ci_lower = 0.050
        mock_result.simulated_ci_upper = 0.100
        mock_result.simulated_std_error = 0.012
        mock_result.effect_size_cohens_d = 0.35
        mock_result.statistical_power = 0.85
        mock_result.recommendation = MagicMock(value="deploy")
        mock_result.recommendation_rationale = "Strong positive effect"
        mock_result.recommended_sample_size = 500
        mock_result.recommended_duration_weeks = 8
        mock_result.simulation_confidence = 0.92
        mock_result.fidelity_warning = False
        mock_result.fidelity_warning_reason = None
        mock_result.model_fidelity_score = 0.88
        mock_result.status = MagicMock(value="completed")
        mock_result.error_message = None
        mock_result.execution_time_ms = 250
        mock_result.created_at = datetime.now(timezone.utc)
        mock_result.completed_at = datetime.now(timezone.utc)
        mock_result.population_filters = None
        mock_result.intervention_config = MagicMock()
        mock_result.intervention_config.intervention_type = "email_campaign"
        mock_result.intervention_config.extra_params = {"brand": "Remibrutinib", "twin_type": "hcp"}
        mock_result.intervention_config.model_dump.return_value = {"intervention_type": "email_campaign"}
        mock_result.effect_heterogeneity = MagicMock()
        mock_result.effect_heterogeneity.by_specialty = {}
        mock_result.effect_heterogeneity.by_decile = {}
        mock_result.effect_heterogeneity.by_region = {}
        mock_result.effect_heterogeneity.by_adoption_stage = {}
        mock_result.effect_heterogeneity.get_top_segments.return_value = []
        mock_result.is_significant.return_value = True
        mock_result.effect_direction.return_value = "positive"

        instance.simulate.return_value = mock_result

        yield instance


@pytest.fixture
def mock_twin_repository():
    """Mock TwinRepository."""
    with patch("src.digital_twin.twin_repository.TwinRepository") as mock_repo:
        instance = AsyncMock()
        mock_repo.return_value = instance

        # For save_simulation
        instance.save_simulation.return_value = None

        # For list_simulations
        mock_sim = {
            "simulation_id": str(uuid4()),
            "intervention_type": "email_campaign",
            "brand": "Remibrutinib",
            "twin_type": "hcp",
            "twin_count": 1000,
            "simulated_ate": 0.075,
            "recommendation": "deploy",
            "simulation_status": "completed",
            "created_at": datetime.now(timezone.utc),
        }
        instance.list_simulations.return_value = [mock_sim]

        # For get_simulation
        mock_result = MagicMock()
        mock_result.simulation_id = uuid4()
        mock_result.model_id = uuid4()
        mock_result.intervention_config = MagicMock()
        mock_result.intervention_config.intervention_type = "email_campaign"
        mock_result.intervention_config.extra_params = {"brand": "Remibrutinib", "twin_type": "hcp"}
        mock_result.intervention_config.model_dump.return_value = {"intervention_type": "email_campaign"}
        mock_result.twin_count = 1000
        mock_result.simulated_ate = 0.075
        mock_result.simulated_ci_lower = 0.050
        mock_result.simulated_ci_upper = 0.100
        mock_result.simulated_std_error = 0.012
        mock_result.effect_size_cohens_d = 0.35
        mock_result.statistical_power = 0.85
        mock_result.recommendation = MagicMock(value="deploy")
        mock_result.recommendation_rationale = "Strong effect"
        mock_result.recommended_sample_size = 500
        mock_result.recommended_duration_weeks = 8
        mock_result.simulation_confidence = 0.92
        mock_result.fidelity_warning = False
        mock_result.fidelity_warning_reason = None
        mock_result.model_fidelity_score = 0.88
        mock_result.status = MagicMock(value="completed")
        mock_result.error_message = None
        mock_result.execution_time_ms = 250
        mock_result.created_at = datetime.now(timezone.utc)
        mock_result.completed_at = datetime.now(timezone.utc)
        mock_result.population_filters = None
        mock_result.effect_heterogeneity = MagicMock()
        mock_result.effect_heterogeneity.by_specialty = {}
        mock_result.effect_heterogeneity.by_decile = {}
        mock_result.effect_heterogeneity.by_region = {}
        mock_result.effect_heterogeneity.by_adoption_stage = {}
        mock_result.effect_heterogeneity.get_top_segments.return_value = []
        mock_result.is_significant.return_value = True
        mock_result.effect_direction.return_value = "positive"

        instance.get_simulation.return_value = mock_result

        # For list_active_models
        mock_model = {
            "model_id": str(uuid4()),
            "model_name": "HCP Twin Model",
            "twin_type": "hcp",
            "brand": "Remibrutinib",
            "algorithm": "RandomForest",
            "r2_score": 0.85,
            "rmse": 0.12,
            "training_samples": 5000,
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
        }
        instance.list_active_models.return_value = [mock_model]

        # For get_model
        mock_model_detail = {
            **mock_model,
            "model_description": "Test model",
            "feature_columns": ["feature1", "feature2"],
            "target_column": "outcome",
            "cv_mean": 0.83,
            "cv_std": 0.02,
            "feature_importances": {"feature1": 0.6, "feature2": 0.4},
            "top_features": ["feature1", "feature2"],
            "training_duration_seconds": 120.5,
            "config": {},
        }
        instance.get_model.return_value = mock_model_detail

        # For get_model_fidelity_records
        mock_fidelity_record = MagicMock()
        mock_fidelity_record.tracking_id = uuid4()
        mock_fidelity_record.simulation_id = uuid4()
        mock_fidelity_record.actual_experiment_id = uuid4()
        mock_fidelity_record.simulated_ate = 0.075
        mock_fidelity_record.simulated_ci_lower = 0.050
        mock_fidelity_record.simulated_ci_upper = 0.100
        mock_fidelity_record.actual_ate = 0.072
        mock_fidelity_record.actual_ci_lower = 0.048
        mock_fidelity_record.actual_ci_upper = 0.096
        mock_fidelity_record.actual_sample_size = 1000
        mock_fidelity_record.prediction_error = 0.003
        mock_fidelity_record.absolute_error = 0.003
        mock_fidelity_record.ci_coverage = True
        mock_fidelity_record.fidelity_grade = MagicMock(value="excellent")
        mock_fidelity_record.validation_notes = None
        mock_fidelity_record.confounding_factors = []
        mock_fidelity_record.created_at = datetime.now(timezone.utc)
        mock_fidelity_record.validated_at = datetime.now(timezone.utc)
        mock_fidelity_record.validated_by = "test_user"

        instance.get_model_fidelity_records.return_value = [mock_fidelity_record]

        yield instance


@pytest.fixture
def mock_fidelity_tracker():
    """Mock FidelityTracker."""
    with patch("src.digital_twin.fidelity_tracker.FidelityTracker") as mock_tracker:
        instance = MagicMock()
        mock_tracker.return_value = instance

        # Mock fidelity record
        mock_record = MagicMock()
        mock_record.tracking_id = uuid4()
        mock_record.simulation_id = uuid4()
        mock_record.actual_experiment_id = uuid4()
        mock_record.simulated_ate = 0.075
        mock_record.simulated_ci_lower = 0.050
        mock_record.simulated_ci_upper = 0.100
        mock_record.actual_ate = 0.072
        mock_record.actual_ci_lower = 0.048
        mock_record.actual_ci_upper = 0.096
        mock_record.actual_sample_size = 1000
        mock_record.prediction_error = 0.003
        mock_record.absolute_error = 0.003
        mock_record.ci_coverage = True
        mock_record.fidelity_grade = MagicMock(value="excellent")
        mock_record.validation_notes = "Test validation"
        mock_record.confounding_factors = []
        mock_record.created_at = datetime.now(timezone.utc)
        mock_record.validated_at = datetime.now(timezone.utc)
        mock_record.validated_by = "test_user"

        instance.get_simulation_record.return_value = None  # No existing record
        instance.record_prediction.return_value = mock_record
        instance.validate.return_value = mock_record

        # Mock fidelity report
        mock_report = {
            "validation_count": 10,
            "fidelity_score": 0.88,
            "metrics": {"ci_coverage_rate": 0.9},
            "degradation_alert": False,
            "grade_distribution": {"excellent": 8, "good": 2},
            "computed_at": datetime.now(timezone.utc),
        }
        instance.get_model_fidelity_report.return_value = mock_report

        yield instance


# =============================================================================
# TESTS - Health Check
# =============================================================================


@pytest.mark.asyncio
async def test_digital_twin_health():
    """Test Digital Twin service health check."""
    from src.api.routes.digital_twin import digital_twin_health

    result = await digital_twin_health()

    assert result.status == "healthy"
    assert result.service == "digital-twin"
    assert result.models_available == 3
    assert result.simulations_pending == 0


# =============================================================================
# TESTS - Simulation
# =============================================================================


@pytest.mark.asyncio
async def test_run_simulation_success(mock_twin_generator, mock_simulation_engine, mock_twin_repository):
    """Test running a successful simulation."""
    from src.api.routes.digital_twin import run_simulation, SimulateRequest, InterventionConfigRequest, BrandEnum, TwinTypeEnum

    request = SimulateRequest(
        intervention=InterventionConfigRequest(
            intervention_type="email_campaign",
            channel="email",
            frequency="weekly",
            duration_weeks=8,
        ),
        brand=BrandEnum.REMIBRUTINIB,
        twin_type=TwinTypeEnum.HCP,
        twin_count=1000,
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await run_simulation(request, user)

    assert result.intervention_type == "email_campaign"
    assert result.brand == "Remibrutinib"
    assert result.twin_count == 1000
    assert result.simulated_ate == 0.075
    assert result.recommendation.value == "deploy"


@pytest.mark.asyncio
async def test_run_simulation_with_filters(mock_twin_generator, mock_simulation_engine, mock_twin_repository):
    """Test simulation with population filters."""
    from src.api.routes.digital_twin import run_simulation, SimulateRequest, InterventionConfigRequest, PopulationFilterRequest, BrandEnum

    request = SimulateRequest(
        intervention=InterventionConfigRequest(
            intervention_type="email_campaign",
            duration_weeks=8,
        ),
        brand=BrandEnum.REMIBRUTINIB,
        twin_count=1000,
        population_filters=PopulationFilterRequest(
            specialties=["oncology"],
            deciles=[1, 2, 3],
            regions=["northeast"],
        ),
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await run_simulation(request, user)

    assert result.simulated_ate > 0


@pytest.mark.asyncio
async def test_run_simulation_with_specific_model(mock_twin_generator, mock_simulation_engine, mock_twin_repository):
    """Test simulation with specific model ID."""
    from src.api.routes.digital_twin import run_simulation, SimulateRequest, InterventionConfigRequest, BrandEnum

    model_id = str(uuid4())
    request = SimulateRequest(
        intervention=InterventionConfigRequest(
            intervention_type="email_campaign",
            duration_weeks=8,
        ),
        brand=BrandEnum.REMIBRUTINIB,
        twin_count=1000,
        model_id=model_id,
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await run_simulation(request, user)

    assert result.model_id == str(mock_simulation_engine.simulate.return_value.model_id)


@pytest.mark.asyncio
async def test_run_simulation_validation_error(mock_twin_generator, mock_simulation_engine, mock_twin_repository):
    """Test simulation with validation error."""
    from src.api.routes.digital_twin import run_simulation, SimulateRequest, InterventionConfigRequest, BrandEnum

    mock_twin_generator.generate.side_effect = ValueError("Invalid parameters")

    request = SimulateRequest(
        intervention=InterventionConfigRequest(
            intervention_type="email_campaign",
            duration_weeks=8,
        ),
        brand=BrandEnum.REMIBRUTINIB,
        twin_count=1000,
    )
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await run_simulation(request, user)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_run_simulation_general_error(mock_twin_generator, mock_simulation_engine, mock_twin_repository):
    """Test simulation with general error."""
    from src.api.routes.digital_twin import run_simulation, SimulateRequest, InterventionConfigRequest, BrandEnum

    mock_simulation_engine.simulate.side_effect = Exception("Simulation failed")

    request = SimulateRequest(
        intervention=InterventionConfigRequest(
            intervention_type="email_campaign",
            duration_weeks=8,
        ),
        brand=BrandEnum.REMIBRUTINIB,
        twin_count=1000,
    )
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await run_simulation(request, user)

    assert exc_info.value.status_code == 500


# =============================================================================
# TESTS - Simulation Listing
# =============================================================================


@pytest.mark.asyncio
async def test_list_simulations_all(mock_twin_repository):
    """Test listing all simulations."""
    from src.api.routes.digital_twin import list_simulations

    result = await list_simulations(brand=None, model_id=None, status=None, page=1, page_size=20)

    assert result.total_count == 1
    assert len(result.simulations) == 1
    assert result.page == 1
    assert result.page_size == 20


@pytest.mark.asyncio
async def test_list_simulations_filtered_by_brand(mock_twin_repository):
    """Test listing simulations filtered by brand."""
    from src.api.routes.digital_twin import list_simulations, BrandEnum

    result = await list_simulations(brand=BrandEnum.REMIBRUTINIB, model_id=None, status=None, page=1, page_size=20)

    assert result.total_count >= 0


@pytest.mark.asyncio
async def test_list_simulations_filtered_by_model(mock_twin_repository):
    """Test listing simulations filtered by model ID."""
    from src.api.routes.digital_twin import list_simulations

    model_id = str(uuid4())
    result = await list_simulations(brand=None, model_id=model_id, status=None, page=1, page_size=20)

    assert result.total_count >= 0


@pytest.mark.asyncio
async def test_list_simulations_filtered_by_status(mock_twin_repository):
    """Test listing simulations filtered by status."""
    from src.api.routes.digital_twin import list_simulations, SimulationStatusEnum

    result = await list_simulations(brand=None, model_id=None, status=SimulationStatusEnum.COMPLETED, page=1, page_size=20)

    assert result.total_count >= 0


@pytest.mark.asyncio
async def test_list_simulations_pagination(mock_twin_repository):
    """Test simulation listing with pagination."""
    from src.api.routes.digital_twin import list_simulations

    # Create multiple mock simulations
    sims = []
    for i in range(5):
        sims.append({
            "simulation_id": str(uuid4()),
            "intervention_type": "email_campaign",
            "brand": "Remibrutinib",
            "twin_type": "hcp",
            "twin_count": 1000,
            "simulated_ate": 0.075,
            "recommendation": "deploy",
            "simulation_status": "completed",
            "created_at": datetime.now(timezone.utc),
        })
    mock_twin_repository.list_simulations.return_value = sims

    result = await list_simulations(brand=None, model_id=None, status=None, page=2, page_size=2)

    assert result.page == 2
    assert result.page_size == 2


# =============================================================================
# TESTS - Simulation Details
# =============================================================================


@pytest.mark.asyncio
async def test_get_simulation_success(mock_twin_repository):
    """Test getting simulation details."""
    from src.api.routes.digital_twin import get_simulation

    simulation_id = str(uuid4())

    result = await get_simulation(simulation_id)

    assert result.intervention_type == "email_campaign"
    assert result.twin_count == 1000
    assert "effect_heterogeneity" in result.model_dump()


@pytest.mark.asyncio
async def test_get_simulation_not_found(mock_twin_repository):
    """Test getting non-existent simulation."""
    from src.api.routes.digital_twin import get_simulation

    mock_twin_repository.get_simulation.return_value = None

    simulation_id = str(uuid4())

    with pytest.raises(HTTPException) as exc_info:
        await get_simulation(simulation_id)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_simulation_error(mock_twin_repository):
    """Test getting simulation with error."""
    from src.api.routes.digital_twin import get_simulation

    mock_twin_repository.get_simulation.side_effect = Exception("Database error")

    simulation_id = str(uuid4())

    with pytest.raises(HTTPException) as exc_info:
        await get_simulation(simulation_id)

    assert exc_info.value.status_code == 500


# =============================================================================
# TESTS - Fidelity Validation
# =============================================================================


@pytest.mark.asyncio
async def test_validate_simulation_success(mock_fidelity_tracker, mock_twin_repository):
    """Test validating simulation against actual results."""
    from src.api.routes.digital_twin import validate_simulation, ValidateFidelityRequest

    # Mock simulation exists
    mock_sim = {"model_id": str(uuid4()), "simulated_ate": 0.075}
    mock_twin_repository.get_simulation.return_value = mock_sim

    request = ValidateFidelityRequest(
        simulation_id=str(uuid4()),
        experiment_id=str(uuid4()),
        actual_ate=0.072,
        actual_ci_lower=0.048,
        actual_ci_upper=0.096,
        actual_sample_size=1000,
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await validate_simulation(request, user)

    assert result.simulated_ate == 0.075
    assert result.actual_ate == 0.072
    assert result.fidelity_grade.value == "excellent"


@pytest.mark.asyncio
async def test_validate_simulation_existing_record(mock_fidelity_tracker, mock_twin_repository):
    """Test validating simulation with existing fidelity record."""
    from src.api.routes.digital_twin import validate_simulation, ValidateFidelityRequest

    # Mock existing record
    existing_record = MagicMock()
    existing_record.tracking_id = uuid4()
    mock_fidelity_tracker.get_simulation_record.return_value = existing_record

    mock_sim = {"model_id": str(uuid4()), "simulated_ate": 0.075}
    mock_twin_repository.get_simulation.return_value = mock_sim

    request = ValidateFidelityRequest(
        simulation_id=str(uuid4()),
        experiment_id=str(uuid4()),
        actual_ate=0.072,
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await validate_simulation(request, user)

    assert result.fidelity_grade.value == "excellent"


@pytest.mark.asyncio
async def test_validate_simulation_not_found(mock_fidelity_tracker, mock_twin_repository):
    """Test validating non-existent simulation."""
    from src.api.routes.digital_twin import validate_simulation, ValidateFidelityRequest

    mock_twin_repository.get_simulation.return_value = None

    request = ValidateFidelityRequest(
        simulation_id=str(uuid4()),
        experiment_id=str(uuid4()),
        actual_ate=0.072,
    )
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await validate_simulation(request, user)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_validate_simulation_error(mock_fidelity_tracker, mock_twin_repository):
    """Test validation with error."""
    from src.api.routes.digital_twin import validate_simulation, ValidateFidelityRequest

    mock_sim = {"model_id": str(uuid4()), "simulated_ate": 0.075}
    mock_twin_repository.get_simulation.return_value = mock_sim

    mock_fidelity_tracker.validate.side_effect = Exception("Validation failed")

    request = ValidateFidelityRequest(
        simulation_id=str(uuid4()),
        experiment_id=str(uuid4()),
        actual_ate=0.072,
    )
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await validate_simulation(request, user)

    assert exc_info.value.status_code == 500


# =============================================================================
# TESTS - Model Management
# =============================================================================


@pytest.mark.asyncio
async def test_list_models_all(mock_twin_repository):
    """Test listing all active models."""
    from src.api.routes.digital_twin import list_models

    result = await list_models(brand=None, twin_type=None)

    assert result.total_count == 1
    assert len(result.models) == 1


@pytest.mark.asyncio
async def test_list_models_filtered_by_brand(mock_twin_repository):
    """Test listing models filtered by brand."""
    from src.api.routes.digital_twin import list_models, BrandEnum

    result = await list_models(brand=BrandEnum.REMIBRUTINIB, twin_type=None)

    assert result.total_count >= 0


@pytest.mark.asyncio
async def test_list_models_filtered_by_type(mock_twin_repository):
    """Test listing models filtered by twin type."""
    from src.api.routes.digital_twin import list_models, TwinTypeEnum

    result = await list_models(brand=None, twin_type=TwinTypeEnum.HCP)

    assert result.total_count >= 0


@pytest.mark.asyncio
async def test_get_model_success(mock_twin_repository):
    """Test getting model details."""
    from src.api.routes.digital_twin import get_model

    model_id = str(uuid4())

    result = await get_model(model_id)

    assert result.model_name == "HCP Twin Model"
    assert result.algorithm == "RandomForest"
    assert len(result.feature_columns) == 2


@pytest.mark.asyncio
async def test_get_model_not_found(mock_twin_repository):
    """Test getting non-existent model."""
    from src.api.routes.digital_twin import get_model

    mock_twin_repository.get_model.return_value = None

    model_id = str(uuid4())

    with pytest.raises(HTTPException) as exc_info:
        await get_model(model_id)

    assert exc_info.value.status_code == 404


# =============================================================================
# TESTS - Fidelity History
# =============================================================================


@pytest.mark.asyncio
async def test_get_model_fidelity_all(mock_twin_repository):
    """Test getting model fidelity history."""
    from src.api.routes.digital_twin import get_model_fidelity

    model_id = str(uuid4())

    result = await get_model_fidelity(model_id, limit=20, validated_only=False)

    assert result.model_id == model_id
    assert result.total_validations == 1
    assert result.average_fidelity_score > 0


@pytest.mark.asyncio
async def test_get_model_fidelity_validated_only(mock_twin_repository):
    """Test getting only validated fidelity records."""
    from src.api.routes.digital_twin import get_model_fidelity

    model_id = str(uuid4())

    result = await get_model_fidelity(model_id, validated_only=True)

    assert result.model_id == model_id


@pytest.mark.asyncio
async def test_get_model_fidelity_grade_distribution(mock_twin_repository):
    """Test fidelity grade distribution."""
    from src.api.routes.digital_twin import get_model_fidelity

    model_id = str(uuid4())

    result = await get_model_fidelity(model_id)

    assert "excellent" in result.grade_distribution
    assert "good" in result.grade_distribution


# =============================================================================
# TESTS - Fidelity Report
# =============================================================================


@pytest.mark.asyncio
async def test_get_fidelity_report_excellent(mock_fidelity_tracker, mock_twin_repository):
    """Test fidelity report with excellent performance."""
    from src.api.routes.digital_twin import get_fidelity_report

    model_id = str(uuid4())

    result = await get_fidelity_report(model_id, lookback_days=90)

    assert result.model_id == model_id
    assert result.total_validations == 10
    assert result.average_fidelity_score == 0.88
    assert result.trend == "excellent"
    assert result.is_degrading is False


@pytest.mark.asyncio
async def test_get_fidelity_report_degrading(mock_fidelity_tracker, mock_twin_repository):
    """Test fidelity report with degrading performance."""
    from src.api.routes.digital_twin import get_fidelity_report

    # Mock degrading report
    mock_report = {
        "validation_count": 10,
        "fidelity_score": 0.75,
        "metrics": {"ci_coverage_rate": 0.7},
        "degradation_alert": True,
        "grade_distribution": {"good": 5, "fair": 5},
        "computed_at": datetime.now(timezone.utc),
    }
    mock_fidelity_tracker.get_model_fidelity_report.return_value = mock_report

    model_id = str(uuid4())

    result = await get_fidelity_report(model_id)

    assert result.is_degrading is True
    assert result.trend == "degrading"
    assert "retraining" in result.recommendation.lower()


@pytest.mark.asyncio
async def test_get_fidelity_report_insufficient_data(mock_fidelity_tracker, mock_twin_repository):
    """Test fidelity report with insufficient data."""
    from src.api.routes.digital_twin import get_fidelity_report

    # Mock insufficient data
    mock_report = {
        "validation_count": 0,
        "fidelity_score": 0.0,
        "metrics": {},
        "degradation_alert": False,
        "grade_distribution": {},
        "computed_at": datetime.now(timezone.utc),
    }
    mock_fidelity_tracker.get_model_fidelity_report.return_value = mock_report

    model_id = str(uuid4())

    result = await get_fidelity_report(model_id)

    assert result.trend == "insufficient_data"
    assert "more validated" in result.recommendation.lower()


@pytest.mark.asyncio
async def test_get_fidelity_report_poor_performance(mock_fidelity_tracker, mock_twin_repository):
    """Test fidelity report with poor performance."""
    from src.api.routes.digital_twin import get_fidelity_report

    # Mock poor performance
    mock_report = {
        "validation_count": 10,
        "fidelity_score": 0.5,
        "metrics": {"ci_coverage_rate": 0.4},
        "degradation_alert": False,
        "grade_distribution": {"poor": 10},
        "computed_at": datetime.now(timezone.utc),
    }
    mock_fidelity_tracker.get_model_fidelity_report.return_value = mock_report

    model_id = str(uuid4())

    result = await get_fidelity_report(model_id)

    assert result.trend == "poor"
    assert "below threshold" in result.recommendation.lower()


# =============================================================================
# TESTS - Edge Cases
# =============================================================================


@pytest.mark.asyncio
async def test_simulation_with_all_intervention_params(mock_twin_generator, mock_simulation_engine, mock_twin_repository):
    """Test simulation with all intervention parameters."""
    from src.api.routes.digital_twin import run_simulation, SimulateRequest, InterventionConfigRequest, BrandEnum

    request = SimulateRequest(
        intervention=InterventionConfigRequest(
            intervention_type="email_campaign",
            channel="email",
            frequency="weekly",
            duration_weeks=12,
            content_type="clinical_data",
            personalization_level="high",
            target_segment="high_value_hcps",
            target_deciles=[1, 2, 3],
            target_specialties=["oncology"],
            target_regions=["northeast", "southwest"],
            intensity_multiplier=1.5,
            extra_params={"custom_field": "value"},
        ),
        brand=BrandEnum.REMIBRUTINIB,
        twin_count=5000,
        confidence_level=0.99,
        calculate_heterogeneity=True,
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await run_simulation(request, user)

    assert result.twin_count == 1000  # From mock


@pytest.mark.asyncio
async def test_list_simulations_empty(mock_twin_repository):
    """Test listing when no simulations exist."""
    from src.api.routes.digital_twin import list_simulations

    mock_twin_repository.list_simulations.return_value = []

    result = await list_simulations(brand=None, model_id=None, status=None, page=1, page_size=20)

    assert result.total_count == 0
    assert len(result.simulations) == 0


@pytest.mark.asyncio
async def test_list_models_empty(mock_twin_repository):
    """Test listing when no models exist."""
    from src.api.routes.digital_twin import list_models

    mock_twin_repository.list_active_models.return_value = []

    result = await list_models(brand=None, twin_type=None)

    assert result.total_count == 0
    assert len(result.models) == 0


@pytest.mark.asyncio
async def test_fidelity_history_no_records(mock_twin_repository):
    """Test fidelity history when no records exist."""
    from src.api.routes.digital_twin import get_model_fidelity

    mock_twin_repository.get_model_fidelity_records.return_value = []

    model_id = str(uuid4())

    result = await get_model_fidelity(model_id)

    assert result.total_validations == 0
    assert result.average_fidelity_score is None
