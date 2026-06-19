"""
Unit tests for digital twin API routes.

Tests all endpoints in src/api/routes/digital_twin.py including:
- Digital Twin simulation
- Simulation listing and filtering
- Fidelity validation
- Twin model management
- Fidelity reporting
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

# Admin user for direct handler calls — the read GETs now require a viewer-tier
# user (#705 H11) and direct calls bypass the Depends injection, so pass one.
# Admin / cross-brand so brand scoping is a no-op for these repo-mocked tests.
_ADMIN_USER = {"app_metadata": {"role": "admin"}}

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
        mock_result.data_provenance = "synthetic_uplift_v1"
        mock_result.error_message = None
        mock_result.execution_time_ms = 250
        mock_result.created_at = datetime.now(timezone.utc)
        mock_result.completed_at = datetime.now(timezone.utc)
        mock_result.population_filters = None
        mock_result.intervention_config = MagicMock()
        mock_result.intervention_config.intervention_type = "email_campaign"
        mock_result.intervention_config.extra_params = {"brand": "Remibrutinib", "twin_type": "hcp"}
        mock_result.intervention_config.model_dump.return_value = {
            "intervention_type": "email_campaign"
        }
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
    """Mock TwinRepository.

    Also patches ``get_async_supabase_client`` so the route's ``_get_twin_repo``
    helper (#705 H6) does not reach for a real Supabase client during unit tests.
    The patched ``TwinRepository(supabase_client=...)`` returns this AsyncMock
    instance regardless of the (mocked) client argument.
    """
    with (
        patch("src.digital_twin.twin_repository.TwinRepository") as mock_repo,
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=MagicMock()),
        ),
    ):
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
        instance.simulations.list_simulations.return_value = [mock_sim]

        # For get_simulation — repo.get_simulation returns the RAW twin_simulations
        # ROW (a dict), not an object (#705 H5b/H11). Mirror the real row shape.
        mock_result = {
            "simulation_id": str(uuid4()),
            "model_id": str(uuid4()),
            "intervention_type": "email_campaign",
            "intervention_config": {"channel": "email", "duration_weeks": 8},
            "brand": "Remibrutinib",
            "twin_count": 1000,
            "simulated_ate": 0.075,
            "simulated_ci_lower": 0.050,
            "simulated_ci_upper": 0.100,
            "simulated_std_error": 0.012,
            "recommendation": "deploy",
            "recommendation_rationale": "Strong effect",
            "recommended_sample_size": 500,
            "recommended_duration_weeks": 8,
            "simulation_confidence": 0.92,
            "fidelity_warning": False,
            "fidelity_warning_reason": None,
            "simulation_status": "completed",
            "data_provenance": "synthetic_uplift_v1",
            "error_message": None,
            "execution_time_ms": 250,
            "created_at": datetime.now(timezone.utc),
            "completed_at": datetime.now(timezone.utc),
            "population_filters": {},
            "effect_heterogeneity": {
                "by_specialty": {},
                "by_decile": {},
                "by_region": {},
                "by_adoption_stage": {},
                "top_segments": [],
            },
        }

        instance.get_simulation.return_value = mock_result

        # For list_active_models. This mirrors the REAL row shape save_model
        # writes (#705 H4): metrics nested under performance_metrics, tuning under
        # training_config, target_columns plural — NOT flat keys. (A flat fixture
        # would falsely pass while real prod rows render metric-less.) MLflow refs
        # are flat (as stored); hydration is patched via mock_twin_hydrate.
        mock_model = {
            "model_id": str(uuid4()),
            "model_name": "HCP Twin Model",
            "twin_type": "hcp",
            "brand": "Remibrutinib",
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
            "mlflow_model_uri": "models:/m-test",
            "mlflow_run_id": "run-test",
            "training_config": {"algorithm": "RandomForest", "training_samples": 5000},
            "performance_metrics": {"r2_score": 0.85, "rmse": 0.12, "training_samples": 5000},
        }
        instance.list_active_models.return_value = [mock_model]

        # For get_model
        mock_model_detail = {
            **mock_model,
            "model_description": "Test model",
            "feature_columns": ["feature1", "feature2"],
            "target_columns": ["outcome"],
            "performance_metrics": {
                "r2_score": 0.85,
                "rmse": 0.12,
                "cv_mean": 0.83,
                "cv_std": 0.02,
                "feature_importances": {"feature1": 0.6, "feature2": 0.4},
                "top_features": ["feature1", "feature2"],
                "training_samples": 5000,
                "training_duration_seconds": 120.5,
            },
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
def mock_twin_hydrate():
    """Patch the MLflow round-trip so route tests don't touch a real registry.

    The real hydration is covered by test_twin_persistence.py; here we only need
    the route's load-before-generate step (#705 H4) to succeed so the simulation
    flow proceeds to the (mocked) generator/engine.
    """
    with patch("src.digital_twin.twin_persistence.hydrate_generator", return_value=True) as m:
        yield m


@pytest.fixture
def mock_fidelity_tracker():
    """Mock FidelityTracker.

    The tracker's record_prediction / validate / get_simulation_record are async
    coroutines (#705 H7), so the instance must be an AsyncMock for ``await`` to
    work in the route.
    """
    with patch("src.digital_twin.fidelity_tracker.FidelityTracker") as mock_tracker:
        instance = AsyncMock()
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

        # These three tracker methods are async coroutines (#705 H7): on an
        # AsyncMock instance they auto-return awaitables, which is what the route
        # awaits.
        instance.get_simulation_record.return_value = None  # No existing record
        instance.record_prediction.return_value = mock_record
        instance.validate.return_value = mock_record

        # ``get_model_fidelity_report`` is still SYNC; force it to a plain
        # MagicMock so it returns the dict directly (not a coroutine) on the
        # AsyncMock instance.
        mock_report = {
            "validation_count": 10,
            "fidelity_score": 0.88,
            "metrics": {"ci_coverage_rate": 0.9},
            "degradation_alert": False,
            "grade_distribution": {"excellent": 8, "good": 2},
            "computed_at": datetime.now(timezone.utc),
        }
        instance.get_model_fidelity_report = MagicMock(return_value=mock_report)

        yield instance


# =============================================================================
# TESTS - Health Check
# =============================================================================


@pytest.mark.asyncio
async def test_digital_twin_health_reports_real_stats(mock_twin_repository):
    """Health must report REAL model/simulation counts from the repository,
    not hardcoded operational stats (was: models_available=3, pending=0)."""
    from src.api.routes.digital_twin import digital_twin_health

    # Repository fixture returns one active model and one (completed) simulation.
    result = await digital_twin_health()

    assert result.service == "digital-twin"
    assert result.models_available == 1  # from mock_twin_repository.list_active_models
    assert result.status == "healthy"
    # No pending simulations in the fixture (status == completed).
    assert result.simulations_pending == 0


@pytest.mark.asyncio
async def test_digital_twin_health_counts_pending(mock_twin_repository):
    """Pending simulation count must reflect repository data."""
    from src.api.routes.digital_twin import digital_twin_health

    mock_twin_repository.simulations.list_simulations.return_value = [
        {"simulation_id": str(uuid4()), "simulation_status": "pending"},
        {"simulation_id": str(uuid4()), "simulation_status": "running"},
        {"simulation_id": str(uuid4()), "simulation_status": "completed"},
    ]

    result = await digital_twin_health()

    # pending + running are both "in flight" / not yet complete
    assert result.simulations_pending == 2


@pytest.mark.asyncio
async def test_digital_twin_health_degraded_on_repo_failure(mock_twin_repository):
    """If the repository is unreachable, health must report degraded WITHOUT
    fabricating operational stats."""
    from src.api.routes.digital_twin import digital_twin_health

    mock_twin_repository.list_active_models.side_effect = Exception("db down")

    result = await digital_twin_health()

    assert result.status == "degraded"
    assert result.models_available == 0


# =============================================================================
# TESTS - Simulation History (contract: GET /simulations/history)
# =============================================================================


@pytest.mark.asyncio
async def test_get_simulation_history_returns_rows(mock_twin_repository):
    """GET /simulations/history must return the frontend-contracted shape
    (ate_estimate, recommendation_type, total/offset/limit)."""
    from src.api.routes.digital_twin import get_simulation_history

    result = await get_simulation_history(limit=10, offset=0, user=_ADMIN_USER)

    assert result.total >= 1
    assert result.limit == 10
    assert result.offset == 0
    assert len(result.simulations) >= 1
    row = result.simulations[0]
    assert hasattr(row, "ate_estimate")
    assert hasattr(row, "recommendation_type")
    assert hasattr(row, "simulation_id")


@pytest.mark.asyncio
async def test_get_simulation_history_brand_filter_passed_to_repo(mock_twin_repository):
    """An admin's brand filter is threaded into the repository read."""
    from src.api.routes.digital_twin import BrandEnum, get_simulation_history

    await get_simulation_history(brand=BrandEnum.KISQALI, limit=10, offset=0, user=_ADMIN_USER)

    _args, kwargs = mock_twin_repository.simulations.list_simulations.call_args
    assert kwargs.get("brand") == "Kisqali"


@pytest.mark.asyncio
async def test_get_simulation_history_all_brands_passes_no_brand(mock_twin_repository):
    """Omitting brand ('All brands') reads every brand the admin may see (brand=None)."""
    from src.api.routes.digital_twin import get_simulation_history

    await get_simulation_history(limit=10, offset=0, user=_ADMIN_USER)

    _args, kwargs = mock_twin_repository.simulations.list_simulations.call_args
    assert kwargs.get("brand") is None


@pytest.mark.asyncio
async def test_get_simulation_history_not_shadowed_by_dynamic_route():
    """The literal /simulations/history route MUST be registered BEFORE the
    dynamic /simulations/{simulation_id} route, otherwise 'history' is parsed
    as a UUID and 500s. Verify route registration order on the router."""
    from src.api.routes.digital_twin import router

    paths = [getattr(r, "path", "") for r in router.routes]
    history_path = "/digital-twin/simulations/history"
    dynamic_path = "/digital-twin/simulations/{simulation_id}"
    assert history_path in paths
    assert dynamic_path in paths
    assert paths.index(history_path) < paths.index(dynamic_path)


@pytest.mark.asyncio
async def test_get_simulation_history_repo_error_is_generic(mock_twin_repository):
    """Repository failure must NOT leak raw exception text in the 5xx detail."""
    from src.api.routes.digital_twin import get_simulation_history

    mock_twin_repository.simulations.list_simulations.side_effect = Exception("SECRET-DSN-LEAK")

    with pytest.raises(HTTPException) as exc_info:
        await get_simulation_history(limit=10, offset=0, user=_ADMIN_USER)

    assert exc_info.value.status_code == 500
    assert "SECRET-DSN-LEAK" not in str(exc_info.value.detail)


# =============================================================================
# TESTS - Intervention Types (contract: GET /intervention-types)
# =============================================================================


@pytest.mark.asyncio
async def test_list_intervention_types_is_canonical_source_of_truth(
    mock_twin_repository, monkeypatch
):
    """The endpoint must serve exactly the backend SUPPORTED_INTERVENTIONS — the
    single source of truth the FE dropdown mirrors (so they can never drift)."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        TwinTypeEnum,
        list_intervention_types,
    )
    from src.digital_twin.effect.provider import SUPPORTED_INTERVENTIONS

    mock_twin_repository.list_active_models = AsyncMock(return_value=[{"model_id": "m1"}])
    # No cohort -> every type stays on the uniform synthetic basis.
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.brand_has_cohort",
        AsyncMock(return_value=False),
    )

    result = await list_intervention_types(
        brand=BrandEnum.REMIBRUTINIB, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER
    )

    assert {i.value for i in result.interventions} == SUPPORTED_INTERVENTIONS
    assert len(result.interventions) == 6
    assert all(i.effect_basis == "synthetic" for i in result.interventions)
    # A trained twin model exists for the brand -> every type is available.
    assert all(i.available for i in result.interventions)
    assert result.brand == "Remibrutinib"


@pytest.mark.asyncio
async def test_list_intervention_types_cohort_estimable_flip_when_cohort_present(
    mock_twin_repository, monkeypatch
):
    """Phase 2: when the brand has a usable synthetic-gold cohort, the
    cohort-estimable interventions report effect_basis 'cohort_estimated' while
    the rest stay 'synthetic'. (Verified against the live DB: only
    digital_engagement + call_frequency_increase have a cohort treatment column.)"""
    from src.api.routes.digital_twin import (
        BrandEnum,
        TwinTypeEnum,
        list_intervention_types,
    )
    from src.digital_twin.effect.provider import COHORT_ESTIMABLE_INTERVENTIONS

    mock_twin_repository.list_active_models = AsyncMock(return_value=[{"model_id": "m1"}])
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.brand_has_cohort",
        AsyncMock(return_value=True),
    )

    result = await list_intervention_types(
        brand=BrandEnum.REMIBRUTINIB, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER
    )

    by_basis = {i.value: i.effect_basis for i in result.interventions}
    cohort_types = {v for v, b in by_basis.items() if b == "cohort_estimated"}
    assert cohort_types == set(COHORT_ESTIMABLE_INTERVENTIONS)
    assert {"digital_engagement", "call_frequency_increase"} == cohort_types
    # Everything else stays on the uniform synthetic basis.
    assert all(
        b == "synthetic" for v, b in by_basis.items() if v not in COHORT_ESTIMABLE_INTERVENTIONS
    )


@pytest.mark.asyncio
async def test_list_intervention_types_unavailable_without_trained_model(mock_twin_repository):
    """Brand-aware availability: no trained twin model for the brand -> the types
    are reported unavailable (honest — /simulate would 503), never fabricated."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        TwinTypeEnum,
        list_intervention_types,
    )

    mock_twin_repository.list_active_models = AsyncMock(return_value=[])

    result = await list_intervention_types(
        brand=BrandEnum.KISQALI, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER
    )

    assert len(result.interventions) == 6
    assert all(not i.available for i in result.interventions)


# =============================================================================
# TESTS - Scenario Comparison (contract: POST /simulations/compare)
# =============================================================================


@pytest.mark.asyncio
async def test_compare_scenarios_returns_result(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    """POST /simulations/compare must run the base + alternative scenarios and
    return a comparison with a best_scenario_index."""
    from src.api.routes.digital_twin import (
        ScenarioComparisonRequest,
        ScenarioSimulateRequest,
        compare_scenarios,
    )

    base = ScenarioSimulateRequest(
        intervention_type="email_campaign",
        brand="Remibrutinib",
    )
    alt = ScenarioSimulateRequest(
        intervention_type="call_frequency_increase",
        brand="Remibrutinib",
    )
    request = ScenarioComparisonRequest(
        base_scenario=base,
        alternative_scenarios=[alt],
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await compare_scenarios(request, user)

    assert result.base_result is not None
    assert len(result.alternative_results) == 1
    assert hasattr(result.comparison, "best_scenario_index")


@pytest.mark.asyncio
async def test_compare_scenarios_error_is_generic(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    """Compare failure must NOT leak raw exception text in the 5xx detail."""
    from src.api.routes.digital_twin import (
        ScenarioComparisonRequest,
        ScenarioSimulateRequest,
        compare_scenarios,
    )

    mock_simulation_engine.simulate.side_effect = Exception("SECRET-COMPARE-LEAK")

    request = ScenarioComparisonRequest(
        base_scenario=ScenarioSimulateRequest(
            intervention_type="email_campaign", brand="Remibrutinib"
        ),
        alternative_scenarios=[],
    )
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await compare_scenarios(request, user)

    assert exc_info.value.status_code == 500
    assert "SECRET-COMPARE-LEAK" not in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_compare_scenarios_503_when_no_active_model(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository
):
    """compare must fail closed with 503 (not collapse to 500) when a scenario has
    no trained model — mirroring /simulate, and never generating from an untrained
    generator (#705 H4)."""
    from src.api.routes.digital_twin import (
        ScenarioComparisonRequest,
        ScenarioSimulateRequest,
        compare_scenarios,
    )

    mock_twin_repository.list_active_models.return_value = []

    request = ScenarioComparisonRequest(
        base_scenario=ScenarioSimulateRequest(
            intervention_type="email_campaign", brand="Remibrutinib"
        ),
        alternative_scenarios=[],
    )
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await compare_scenarios(request, user)

    assert exc_info.value.status_code == 503
    mock_twin_generator.generate.assert_not_called()


# =============================================================================
# TESTS - Simulation
# =============================================================================


@pytest.mark.asyncio
async def test_run_simulation_success(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    """Test running a successful simulation."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
        run_simulation,
    )

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
async def test_run_simulation_with_filters(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    """Test simulation with population filters."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        PopulationFilterRequest,
        SimulateRequest,
        run_simulation,
    )

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
async def test_run_simulation_with_specific_model(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    """Test simulation with specific model ID."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        run_simulation,
    )

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
async def test_run_simulation_validation_error(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    """Test simulation with validation error."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        run_simulation,
    )

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
async def test_run_simulation_general_error(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    """Test simulation with general error."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        run_simulation,
    )

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


@pytest.mark.asyncio
async def test_run_simulation_503_when_no_active_model(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository
):
    """No trained model for the brand/twin_type → honest 503 (#705 H4).

    Before H4 a fresh untrained generator raised RuntimeError → an opaque 500.
    The endpoint must fail closed with 503 + Retry-After and NEVER fabricate a
    result, so generate() is never even reached.
    """
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
        run_simulation,
    )

    mock_twin_repository.list_active_models.return_value = []

    request = SimulateRequest(
        intervention=InterventionConfigRequest(
            intervention_type="email_campaign", duration_weeks=8
        ),
        brand=BrandEnum.REMIBRUTINIB,
        twin_type=TwinTypeEnum.HCP,
        twin_count=1000,
    )
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await run_simulation(request, user)

    assert exc_info.value.status_code == 503
    assert exc_info.value.headers and "Retry-After" in exc_info.value.headers
    mock_twin_generator.generate.assert_not_called()


@pytest.mark.asyncio
async def test_run_simulation_503_when_model_unloadable(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository
):
    """An active model row whose artifact can't be hydrated → 503, not 500/fake."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
        run_simulation,
    )

    mock_twin_repository.list_active_models.return_value = [
        {
            "model_id": str(uuid4()),
            "twin_type": "hcp",
            "brand": "Remibrutinib",
            "mlflow_model_uri": "models:/m-gone",
            "mlflow_run_id": "run-gone",
            "is_active": True,
        }
    ]

    request = SimulateRequest(
        intervention=InterventionConfigRequest(
            intervention_type="email_campaign", duration_weeks=8
        ),
        brand=BrandEnum.REMIBRUTINIB,
        twin_type=TwinTypeEnum.HCP,
        twin_count=1000,
    )
    user = {"user_id": "test_user", "role": "operator"}

    with patch("src.digital_twin.twin_persistence.hydrate_generator", return_value=False):
        with pytest.raises(HTTPException) as exc_info:
            await run_simulation(request, user)

    assert exc_info.value.status_code == 503
    mock_twin_generator.generate.assert_not_called()


# =============================================================================
# TESTS - Simulation Listing
# =============================================================================


@pytest.mark.asyncio
async def test_list_simulations_all(mock_twin_repository):
    """Test listing all simulations."""
    from src.api.routes.digital_twin import list_simulations

    result = await list_simulations(
        brand=None, model_id=None, status=None, page=1, page_size=20, user=_ADMIN_USER
    )

    assert result.total_count == 1
    assert len(result.simulations) == 1
    assert result.page == 1
    assert result.page_size == 20


@pytest.mark.asyncio
async def test_list_simulations_filtered_by_brand(mock_twin_repository):
    """Test listing simulations filtered by brand."""
    from src.api.routes.digital_twin import BrandEnum, list_simulations

    result = await list_simulations(
        brand=BrandEnum.REMIBRUTINIB,
        model_id=None,
        status=None,
        page=1,
        page_size=20,
        user=_ADMIN_USER,
    )

    assert result.total_count >= 0


@pytest.mark.asyncio
async def test_list_simulations_filtered_by_model(mock_twin_repository):
    """Test listing simulations filtered by model ID."""
    from src.api.routes.digital_twin import list_simulations

    model_id = str(uuid4())
    result = await list_simulations(
        brand=None, model_id=model_id, status=None, page=1, page_size=20, user=_ADMIN_USER
    )

    assert result.total_count >= 0


@pytest.mark.asyncio
async def test_list_simulations_filtered_by_status(mock_twin_repository):
    """Test listing simulations filtered by status."""
    from src.api.routes.digital_twin import SimulationStatusEnum, list_simulations

    result = await list_simulations(
        brand=None,
        model_id=None,
        status=SimulationStatusEnum.COMPLETED,
        page=1,
        page_size=20,
        user=_ADMIN_USER,
    )

    assert result.total_count >= 0


@pytest.mark.asyncio
async def test_list_simulations_pagination(mock_twin_repository):
    """Test simulation listing with pagination."""
    from src.api.routes.digital_twin import list_simulations

    # Create multiple mock simulations
    sims = []
    for _i in range(5):
        sims.append(
            {
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
        )
    mock_twin_repository.simulations.list_simulations.return_value = sims

    result = await list_simulations(
        brand=None, model_id=None, status=None, page=2, page_size=2, user=_ADMIN_USER
    )

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

    result = await get_simulation(simulation_id, user=_ADMIN_USER)

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
        await get_simulation(simulation_id, user=_ADMIN_USER)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_simulation_error(mock_twin_repository):
    """Test getting simulation with error returns a generic 500 (no raw leak)."""
    from src.api.routes.digital_twin import get_simulation

    mock_twin_repository.get_simulation.side_effect = Exception("SECRET-DB-LEAK")

    simulation_id = str(uuid4())

    with pytest.raises(HTTPException) as exc_info:
        await get_simulation(simulation_id, user=_ADMIN_USER)

    assert exc_info.value.status_code == 500
    # Info-disclosure fix: raw exception text must NOT reach the client.
    assert "SECRET-DB-LEAK" not in str(exc_info.value.detail)


# =============================================================================
# TESTS - Fidelity Validation
# =============================================================================


@pytest.mark.asyncio
async def test_validate_simulation_success(mock_fidelity_tracker, mock_twin_repository):
    """Test validating simulation against actual results."""
    from src.api.routes.digital_twin import ValidateFidelityRequest, validate_simulation

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
    from src.api.routes.digital_twin import ValidateFidelityRequest, validate_simulation

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
    from src.api.routes.digital_twin import ValidateFidelityRequest, validate_simulation

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
    from src.api.routes.digital_twin import ValidateFidelityRequest, validate_simulation

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

    result = await list_models(brand=None, twin_type=None, user=_ADMIN_USER)

    assert result.total_count == 1
    assert len(result.models) == 1


@pytest.mark.asyncio
async def test_list_models_filtered_by_brand(mock_twin_repository):
    """Test listing models filtered by brand."""
    from src.api.routes.digital_twin import BrandEnum, list_models

    result = await list_models(brand=BrandEnum.REMIBRUTINIB, twin_type=None, user=_ADMIN_USER)

    assert result.total_count >= 0


@pytest.mark.asyncio
async def test_list_models_filtered_by_type(mock_twin_repository):
    """Test listing models filtered by twin type."""
    from src.api.routes.digital_twin import TwinTypeEnum, list_models

    result = await list_models(brand=None, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER)

    assert result.total_count >= 0


@pytest.mark.asyncio
async def test_get_model_success(mock_twin_repository):
    """Test getting model details."""
    from src.api.routes.digital_twin import get_model

    model_id = str(uuid4())

    result = await get_model(model_id, user=_ADMIN_USER)

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
        await get_model(model_id, user=_ADMIN_USER)

    assert exc_info.value.status_code == 404


# =============================================================================
# TESTS - Fidelity History
# =============================================================================


@pytest.mark.asyncio
async def test_get_model_fidelity_all(mock_twin_repository):
    """Test getting model fidelity history."""
    from src.api.routes.digital_twin import get_model_fidelity

    model_id = str(uuid4())

    result = await get_model_fidelity(model_id, limit=20, validated_only=False, user=_ADMIN_USER)

    assert result.model_id == model_id
    assert result.total_validations == 1
    assert result.average_fidelity_score > 0


@pytest.mark.asyncio
async def test_get_model_fidelity_validated_only(mock_twin_repository):
    """Test getting only validated fidelity records."""
    from src.api.routes.digital_twin import get_model_fidelity

    model_id = str(uuid4())

    result = await get_model_fidelity(model_id, validated_only=True, user=_ADMIN_USER)

    assert result.model_id == model_id


@pytest.mark.asyncio
async def test_get_model_fidelity_grade_distribution(mock_twin_repository):
    """Test fidelity grade distribution."""
    from src.api.routes.digital_twin import get_model_fidelity

    model_id = str(uuid4())

    result = await get_model_fidelity(model_id, user=_ADMIN_USER)

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

    result = await get_fidelity_report(model_id, lookback_days=90, user=_ADMIN_USER)

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

    result = await get_fidelity_report(model_id, user=_ADMIN_USER)

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

    result = await get_fidelity_report(model_id, user=_ADMIN_USER)

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

    result = await get_fidelity_report(model_id, user=_ADMIN_USER)

    assert result.trend == "poor"
    assert "below threshold" in result.recommendation.lower()


# =============================================================================
# TESTS - Edge Cases
# =============================================================================


@pytest.mark.asyncio
async def test_simulation_with_all_intervention_params(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    """Test simulation with all intervention parameters."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        run_simulation,
    )

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

    mock_twin_repository.simulations.list_simulations.return_value = []

    result = await list_simulations(
        brand=None, model_id=None, status=None, page=1, page_size=20, user=_ADMIN_USER
    )

    assert result.total_count == 0
    assert len(result.simulations) == 0


@pytest.mark.asyncio
async def test_list_models_empty(mock_twin_repository):
    """Test listing when no models exist."""
    from src.api.routes.digital_twin import list_models

    mock_twin_repository.list_active_models.return_value = []

    result = await list_models(brand=None, twin_type=None, user=_ADMIN_USER)

    assert result.total_count == 0
    assert len(result.models) == 0


@pytest.mark.asyncio
async def test_fidelity_history_no_records(mock_twin_repository):
    """Test fidelity history when no records exist."""
    from src.api.routes.digital_twin import get_model_fidelity

    mock_twin_repository.get_model_fidelity_records.return_value = []

    model_id = str(uuid4())

    result = await get_model_fidelity(model_id, user=_ADMIN_USER)

    assert result.total_validations == 0
    assert result.average_fidelity_score is None


# =============================================================================
# TESTS - Priority 1 OOM bounding: heavy-compute slot
# =============================================================================


@pytest.fixture
def _heavy_compute_one_slot(monkeypatch):
    """Bound heavy compute to a single in-flight op and reset the limiter."""
    monkeypatch.setenv("HEAVY_COMPUTE_MAX_CONCURRENCY", "1")
    import src.api.dependencies.compute as compute_mod

    compute_mod._reset_limiter_cache_for_tests()
    yield compute_mod
    compute_mod._reset_limiter_cache_for_tests()


@pytest.mark.asyncio
async def test_run_simulation_rejects_when_heavy_compute_saturated(
    mock_twin_generator,
    mock_simulation_engine,
    mock_twin_repository,
    mock_twin_hydrate,
    _heavy_compute_one_slot,
):
    """When the per-worker heavy-compute slot is exhausted, /simulate must reject
    fast (HeavyComputeSaturated) instead of running another ~1.3 GiB simulation
    that could OOM-kill the container. Exercises the REAL limiter (not mocked)."""
    from src.api.dependencies.compute import HeavyComputeSaturated
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
        run_simulation,
    )

    # Occupy the single slot to simulate a concurrent in-flight heavy request.
    limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
    limiter.acquire()

    request = SimulateRequest(
        intervention=InterventionConfigRequest(
            intervention_type="email_campaign",
            duration_weeks=8,
        ),
        brand=BrandEnum.REMIBRUTINIB,
        twin_type=TwinTypeEnum.HCP,
        twin_count=1000,
    )
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HeavyComputeSaturated):
        await run_simulation(request, user)

    # The simulation must NOT have run while saturated.
    mock_simulation_engine.simulate.assert_not_called()


@pytest.mark.asyncio
async def test_run_simulation_succeeds_when_slot_available(
    mock_twin_generator,
    mock_simulation_engine,
    mock_twin_repository,
    mock_twin_hydrate,
    _heavy_compute_one_slot,
):
    """With a free slot, /simulate runs through the real slot + bounded executor
    and returns the unchanged success response shape."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
        run_simulation,
    )

    request = SimulateRequest(
        intervention=InterventionConfigRequest(
            intervention_type="email_campaign",
            duration_weeks=8,
        ),
        brand=BrandEnum.REMIBRUTINIB,
        twin_type=TwinTypeEnum.HCP,
        twin_count=1000,
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await run_simulation(request, user)

    assert result.twin_count == 1000
    assert result.simulated_ate == 0.075
    assert result.recommendation.value == "deploy"

    # The slot must be released after a successful run (in_flight back to 0).
    limiter = _heavy_compute_one_slot.get_heavy_compute_limiter()
    assert limiter.in_flight == 0


# =============================================================================
# TESTS - H6 Supabase client injection (#705 Lane 1)
# =============================================================================


def _fake_supabase_client():
    """A fake async Supabase client whose .table(..).<op>(..).execute() is awaitable.

    Mirrors the supabase-py async fluent API used by TwinRepository:
        await client.table(name).insert(row).execute()
        await client.table(name).update(updates).eq(...).select().execute()
        await client.table(name).select(..).eq(..).execute()
    """
    client = MagicMock()
    execute_result = MagicMock()
    execute_result.data = [{"tracking_id": str(uuid4()), "actual_ate": 0.072}]

    # Every fluent step returns the same chainable mock; only execute() is async.
    chain = MagicMock()
    for method in ("insert", "update", "select", "eq", "order", "limit", "range"):
        getattr(chain, method).return_value = chain
    chain.execute = AsyncMock(return_value=execute_result)

    client.table = MagicMock(return_value=chain)
    return client, chain


@pytest.mark.asyncio
async def test_get_twin_repo_injects_real_client():
    """H6: _get_twin_repo builds a repo whose sub-repos all have a non-None client."""
    from src.api.routes.digital_twin import _get_twin_repo

    fake_client, _ = _fake_supabase_client()
    with patch(
        "src.memory.services.factories.get_async_supabase_client",
        new=AsyncMock(return_value=fake_client),
    ):
        repo = await _get_twin_repo()

    assert repo.simulations.client is fake_client
    assert repo.fidelity.client is fake_client
    assert repo.models.client is fake_client


@pytest.mark.asyncio
async def test_validate_reaches_db_with_injected_client():
    """H6+H7: /validate update path reaches client.table('twin_fidelity_tracking').

    No TwinRepository / FidelityTracker patch — the REAL repository + tracker are
    used with the injected client so we prove the update actually hits the DB
    layer via the real ``update_fidelity_validation`` coroutine.
    """
    from src.api.routes.digital_twin import ValidateFidelityRequest, validate_simulation

    fake_client, chain = _fake_supabase_client()
    sim_id = str(uuid4())

    # Each awaited .execute() returns the next stubbed result in order:
    chain.execute = AsyncMock(
        side_effect=[
            # 1) repo.get_simulation(...) -> simulation row present
            MagicMock(data=[{"simulation_id": sim_id, "model_id": str(uuid4())}]),
            # 2) get_fidelity_by_simulation (cache-miss read) -> no existing record
            MagicMock(data=[]),
            # 3) save_fidelity_record insert (record_prediction)
            MagicMock(data=[{}]),
            # 4) update_fidelity_validation update
            MagicMock(data=[{"tracking_id": str(uuid4()), "actual_ate": 0.072}]),
        ]
    )

    with patch(
        "src.memory.services.factories.get_async_supabase_client",
        new=AsyncMock(return_value=fake_client),
    ):
        request = ValidateFidelityRequest(
            simulation_id=sim_id,
            experiment_id=str(uuid4()),
            actual_ate=0.072,
            actual_ci_lower=0.048,
            actual_ci_upper=0.096,
            actual_sample_size=1000,
        )
        user = {"user_id": "test_user", "role": "operator"}
        await validate_simulation(request, user)

    # The fidelity table must have been touched (insert + update).
    table_calls = [c.args[0] for c in fake_client.table.call_args_list if c.args]
    assert "twin_fidelity_tracking" in table_calls


@pytest.mark.asyncio
async def test_simulate_save_path_uses_injected_client(mock_twin_generator, mock_simulation_engine):
    """H6: /simulate save path persists to twin_simulations via injected client."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
        run_simulation,
    )

    fake_client, chain = _fake_supabase_client()

    # The REAL SimulationRepository.save_simulation serializes the result; give
    # the mocked engine result JSON-able population_filters / heterogeneity so
    # the save reaches the injected client instead of raising on None.to_dict().
    result = mock_simulation_engine.simulate.return_value
    result.population_filters = MagicMock()
    result.population_filters.to_dict.return_value = {}
    result.effect_heterogeneity.model_dump.return_value = {}
    result.memory_usage_mb = 0.0

    # This test targets the SAVE path; bypass model resolution/loading (covered by
    # test_twin_persistence.py + the 503 tests) so it reaches save_simulation.
    with (
        patch(
            "src.memory.services.factories.get_async_supabase_client",
            new=AsyncMock(return_value=fake_client),
        ),
        patch(
            "src.api.routes.digital_twin._resolve_active_model_row",
            new=AsyncMock(
                return_value={
                    "model_id": str(uuid4()),
                    "mlflow_model_uri": "models:/m-x",
                    "mlflow_run_id": "run-x",
                }
            ),
        ),
        patch("src.digital_twin.twin_persistence.hydrate_generator", return_value=True),
    ):
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
        await run_simulation(request, user)

    table_calls = [c.args[0] for c in fake_client.table.call_args_list if c.args]
    assert "twin_simulations" in table_calls


def test_no_bare_twin_repository_in_route_source():
    """Regression guard (#705 H6): no client-less TwinRepository() in the route."""
    import re
    from pathlib import Path

    import src.api.routes.digital_twin as route_mod

    source = Path(route_mod.__file__).read_text()
    # A client-less ``repo = TwinRepository()`` assignment must be gone from every
    # handler (the docstring may still *mention* ``TwinRepository()`` as history).
    assert re.search(r"=\s*TwinRepository\(\s*\)", source) is None
