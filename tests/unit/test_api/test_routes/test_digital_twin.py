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


@pytest.fixture(autouse=True)
def _default_identified_cohort(monkeypatch):
    """Direction-2 identification gate: by default make it PASS (a cohort provider is
    available) so the existing /simulate tests exercise their intended path (slot,
    filters, model resolution, engine — all mocked) regardless of intervention_type.

    The route honestly 422s when ``build_cohort_provider_or_none`` returns None
    (intervention not identified in the cohort); the test of THAT path overrides this
    fixture. Patching the source module covers the route's function-local import.
    """
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.build_cohort_provider_or_none",
        AsyncMock(return_value=MagicMock()),
    )


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
        # Real response-domain value: route serialization must be proven against the
        # production contract, not MagicMock's unconstrained attribute shape (#2162).
        from src.digital_twin.models.simulation_models import EffectHeterogeneity, FidelityStatus

        mock_result.effect_heterogeneity = EffectHeterogeneity()
        # A real domain value again (#2206): the route maps result.fidelity_status.value
        # into FidelityStatusEnum; the fixture's 0.88 score is a validated model.
        mock_result.fidelity_status = FidelityStatus.VALIDATED
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
async def test_digital_twin_health_reports_real_stats(mock_twin_repository, monkeypatch):
    """Health must report REAL model/simulation counts from the repository,
    not hardcoded operational stats (was: models_available=3, pending=0)."""
    from src.api.routes.digital_twin import digital_twin_health

    # Repository fixture returns one active model and one (completed) simulation.
    # "healthy" now has to MEAN simulable, so the brand's cohort must identify an effect.
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability",
        AsyncMock(return_value={"email_campaign": True, "digital_engagement": False}),
    )
    result = await digital_twin_health()
    assert result.brands_simulable == 1

    assert result.service == "digital-twin"
    assert result.models_available == 1  # from mock_twin_repository.list_active_models
    assert result.status == "healthy"
    # No pending simulations in the fixture (status == completed).
    assert result.simulations_pending == 0


@pytest.fixture(autouse=True)
def _fresh_simulable_cache():
    """The health readout remembers per-brand simulability briefly; tests must not share it."""
    from src.api.routes import digital_twin_capability as capability

    capability._simulable_cache.clear()
    yield
    capability._simulable_cache.clear()


@pytest.mark.asyncio
async def test_digital_twin_health_is_degraded_when_models_exist_but_nothing_is_simulable(
    mock_twin_repository, monkeypatch
):
    """The 2026-09-21 outage: an active, loadable model for every brand while the cohort's planted
    treatment channels were gone. Health said ``healthy, models_available=3`` — a count of models is
    a proxy; the capability is "can /simulate serve a brand"."""
    from src.api.routes.digital_twin import digital_twin_health

    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability",
        AsyncMock(return_value={"email_campaign": False, "digital_engagement": False}),
    )
    result = await digital_twin_health()

    assert result.models_available == 1
    assert result.brands_simulable == 0
    assert result.status == "degraded"


@pytest.mark.asyncio
async def test_digital_twin_health_with_no_models_is_not_a_cohort_problem(
    mock_twin_repository, monkeypatch
):
    """No trained model is the OTHER gate; it must not be reported as missing cohort data, and the
    cohort must not even be queried for brands that have no model."""
    from src.api.routes.digital_twin import digital_twin_health

    availability = AsyncMock(return_value={})
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability", availability
    )
    mock_twin_repository.list_active_models.return_value = []
    result = await digital_twin_health()

    assert (result.models_available, result.brands_simulable) == (0, 0)
    assert result.status == "healthy"
    availability.assert_not_awaited()


@pytest.mark.asyncio
async def test_digital_twin_health_remembers_simulability_between_polls(
    mock_twin_repository, monkeypatch
):
    """The page polls /health every 60 s per viewer and one answer costs eight exact counts."""
    from src.api.routes.digital_twin import digital_twin_health

    availability = AsyncMock(return_value={"email_campaign": True})
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability", availability
    )
    await digital_twin_health()
    await digital_twin_health()

    assert availability.await_count == 1


class _Unmeasured(dict):
    """What the loader returns when every column probe ERRORED: all False, n_probe_errors > 0."""

    n_probe_errors = 8


@pytest.mark.asyncio
async def test_a_failed_probe_is_not_remembered_and_does_not_recommend_a_replant(
    mock_twin_repository, monkeypatch, caplog
):
    """codex r1 MEDIUM: a transient DB error read as "no cohort data", was cached for five minutes,
    and logged "re-run backfill_segment_engagement.py --execute" — a production write recommended
    for a connection blip. Unknown is not empty: degrade, say so, ask again next poll."""
    import logging

    from src.api.routes.digital_twin import digital_twin_health

    availability = AsyncMock(return_value=_Unmeasured({"email_campaign": False}))
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability", availability
    )
    with caplog.at_level(logging.WARNING, logger="src.api.routes.digital_twin"):
        first = await digital_twin_health()
        await digital_twin_health()

    assert (first.status, first.brands_simulable) == ("degraded", 0)
    assert availability.await_count == 2  # not remembered
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "could not be measured" in text
    assert "backfill_segment_engagement" not in text


@pytest.mark.asyncio
async def test_intervention_types_payload_says_unmeasured_not_missing_when_the_probe_failed(
    mock_twin_repository, monkeypatch
):
    """codex r2 MEDIUM: fixing the log was not enough. The payload still read as a MEASURED empty
    cohort, so the page told the reader to restore the cohort data, and React Query kept that
    answer fresh. The payload now carries the measurement state."""
    from src.api.routes.digital_twin import BrandEnum, TwinTypeEnum, list_intervention_types

    mock_twin_repository.list_active_models = AsyncMock(return_value=[{"model_id": "m1"}])
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability",
        AsyncMock(return_value=_Unmeasured({"email_campaign": False})),
    )
    result = await list_intervention_types(
        brand=BrandEnum.KISQALI, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER
    )
    assert result.effect_availability_status == "unmeasured"
    assert all(not i.available_for_effect for i in result.interventions)


@pytest.mark.asyncio
async def test_intervention_types_says_the_model_could_not_be_resolved_when_the_repo_fails(
    mock_twin_repository,
):
    """codex r3 MEDIUM: a repository outage returned available=False under a 'measured' status, so
    the page reported an ESTABLISHED absence ("No trained twin model") for a transient blip and
    kept it fresh for five minutes."""
    from src.api.routes.digital_twin import BrandEnum, TwinTypeEnum, list_intervention_types

    mock_twin_repository.list_active_models = AsyncMock(side_effect=RuntimeError("db down"))
    result = await list_intervention_types(
        brand=BrandEnum.KISQALI, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER
    )
    assert result.model_resolution == "unavailable"
    assert result.effect_availability_status is None
    assert all(not i.available and not i.available_for_effect for i in result.interventions)


@pytest.mark.asyncio
async def test_intervention_types_without_a_brand_resolves_nothing(mock_twin_repository):
    from src.api.routes.digital_twin import TwinTypeEnum, list_intervention_types

    result = await list_intervention_types(brand=None, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER)
    assert result.model_resolution == "not_requested"
    assert result.effect_availability_status is None
    mock_twin_repository.list_active_models.assert_not_awaited()


@pytest.mark.asyncio
async def test_intervention_types_payload_says_measured_when_the_probes_ran(
    mock_twin_repository, monkeypatch
):
    from src.api.routes.digital_twin import BrandEnum, TwinTypeEnum, list_intervention_types

    mock_twin_repository.list_active_models = AsyncMock(return_value=[{"model_id": "m1"}])
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability",
        AsyncMock(return_value={"email_campaign": False}),
    )
    result = await list_intervention_types(
        brand=BrandEnum.KISQALI, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER
    )
    assert result.effect_availability_status == "measured"


@pytest.mark.asyncio
async def test_intervention_types_does_not_recommend_a_replant_when_the_probe_failed(
    mock_twin_repository, monkeypatch, caplog
):
    import logging

    from src.api.routes.digital_twin import BrandEnum, TwinTypeEnum, list_intervention_types

    mock_twin_repository.list_active_models = AsyncMock(return_value=[{"model_id": "m1"}])
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability",
        AsyncMock(return_value=_Unmeasured({"email_campaign": False})),
    )
    with caplog.at_level(logging.WARNING, logger="src.api.routes.digital_twin"):
        await list_intervention_types(
            brand=BrandEnum.KISQALI, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER
        )

    text = " ".join(r.getMessage() for r in caplog.records)
    assert "Kisqali" in text and "could not be measured" in text
    assert "backfill_segment_engagement" not in text


@pytest.mark.asyncio
async def test_intervention_types_warns_when_a_model_exists_but_the_cohort_has_no_effect_data(
    mock_twin_repository, monkeypatch, caplog
):
    """On 2026-09-21 this state produced NO log line at all: the check ran cleanly and found
    nothing. It must name the brand and the remedy."""
    import logging

    from src.api.routes.digital_twin import BrandEnum, TwinTypeEnum, list_intervention_types

    mock_twin_repository.list_active_models = AsyncMock(return_value=[{"model_id": "m1"}])
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability",
        AsyncMock(return_value={"email_campaign": False, "digital_engagement": False}),
    )
    with caplog.at_level(logging.WARNING, logger="src.api.routes.digital_twin"):
        await list_intervention_types(
            brand=BrandEnum.KISQALI, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER
        )

    text = " ".join(r.getMessage() for r in caplog.records)
    assert "Kisqali" in text and "backfill_segment_engagement" in text


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

    result = await get_simulation_history(brand=None, limit=10, offset=0, user=_ADMIN_USER)

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

    await get_simulation_history(brand=None, limit=10, offset=0, user=_ADMIN_USER)

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
        await get_simulation_history(brand=None, limit=10, offset=0, user=_ADMIN_USER)

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
    # No cohort -> no intervention's effect is identified (all unavailable).
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability",
        AsyncMock(return_value={}),
    )

    result = await list_intervention_types(
        brand=BrandEnum.REMIBRUTINIB, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER
    )

    assert {i.value for i in result.interventions} == SUPPORTED_INTERVENTIONS
    assert len(result.interventions) == 8
    # No cohort -> nothing is effect-identified -> honest unavailable (never fabricated).
    assert all(i.effect_basis == "unavailable" for i in result.interventions)
    assert all(not i.available_for_effect for i in result.interventions)
    # A trained twin model exists for the brand -> every type is model-available.
    assert all(i.available for i in result.interventions)
    assert result.brand == "Remibrutinib"


@pytest.mark.asyncio
async def test_list_intervention_types_identified_flip_when_cohort_present(
    mock_twin_repository, monkeypatch
):
    """Availability is PER-INTERVENTION: exactly the interventions whose planted
    treatment channel is usable report effect_basis 'cohort_causal' and
    available_for_effect=True; every other intervention is honestly
    'unavailable' (no fabricated effect). A partial map (e.g. pre-backfill, or
    future RWD with partial channel coverage) must NOT flip the whole catalog."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        TwinTypeEnum,
        list_intervention_types,
    )
    from src.digital_twin.effect.provider import COHORT_ESTIMABLE_INTERVENTIONS

    mock_twin_repository.list_active_models = AsyncMock(return_value=[{"model_id": "m1"}])
    partial = dict.fromkeys(COHORT_ESTIMABLE_INTERVENTIONS, False)
    partial["digital_engagement"] = True
    partial["email_campaign"] = True
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability",
        AsyncMock(return_value=partial),
    )

    result = await list_intervention_types(
        brand=BrandEnum.REMIBRUTINIB, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER
    )

    by_basis = {i.value: i.effect_basis for i in result.interventions}
    by_effect = {i.value: i.available_for_effect for i in result.interventions}
    cohort_types = {v for v, b in by_basis.items() if b == "cohort_causal"}
    assert cohort_types == {"digital_engagement", "email_campaign"}
    # available_for_effect is True exactly for the identified (cohort_causal) interventions.
    assert {v for v, e in by_effect.items() if e} == cohort_types
    # Everything else is honestly unavailable (no fabricated synthetic uplift).
    assert all(b == "unavailable" for v, b in by_basis.items() if v not in cohort_types)


@pytest.mark.asyncio
async def test_list_intervention_types_unavailable_without_trained_model(
    mock_twin_repository, monkeypatch
):
    """Brand-aware availability: no trained twin model for the brand -> the types
    are reported unavailable (honest — /simulate would 503), never fabricated."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        TwinTypeEnum,
        list_intervention_types,
    )

    mock_twin_repository.list_active_models = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.cohort_treatment_availability",
        AsyncMock(return_value={}),
    )

    result = await list_intervention_types(
        brand=BrandEnum.KISQALI, twin_type=TwinTypeEnum.HCP, user=_ADMIN_USER
    )

    assert len(result.interventions) == 8
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
async def test_run_simulation_unidentified_intervention_returns_422(
    mock_twin_repository, monkeypatch
):
    """Direction 2: an intervention NOT identified in the cohort returns an honest 422
    'no effect data' BEFORE any compute — never a fabricated synthetic uplift. (Gate runs
    after model resolution, before generator load / slot / engine.)"""
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
        run_simulation,
    )

    # Override the autouse gate-pass: this intervention is not identified -> provider None.
    monkeypatch.setattr(
        "src.digital_twin.effect.cohort_loader.build_cohort_provider_or_none",
        AsyncMock(return_value=None),
    )

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

    assert exc_info.value.status_code == 422
    assert "no effect data" in str(exc_info.value.detail).lower()


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
    # the mocked engine result JSON-able population_filters so the save reaches
    # the injected client instead of raising on None.to_dict(). The shared
    # fixture already supplies a real EffectHeterogeneity domain value.
    result = mock_simulation_engine.simulate.return_value
    result.population_filters = MagicMock()
    result.population_filters.to_dict.return_value = {}
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


# =============================================================================
# #2206 — honest surfacing: one shared synthetic fit, unvalidated fidelity
# =============================================================================

_ADMIN = {"user_id": "admin", "role": "admin"}


def _shared_fit_row(brand: str, *, duration: float, fidelity_score=None, sample_count=0):
    """A digital_twin_models row as prod holds it: the three brands are one seed-0
    synthetic fit — identical metrics/config/features, only the wall-clock differs."""
    return {
        "model_id": str(uuid4()),
        "model_name": f"hcp_twin_{brand}",
        "twin_type": "hcp",
        "brand": brand,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
        "mlflow_model_uri": "models:/m-test",
        "mlflow_run_id": "run-test",
        "feature_columns": ["specialty", "region", "decile"],
        "target_columns": ["outcome"],
        "training_config": {
            "algorithm": "random_forest",
            "data_provenance": "synthetic",
            "training_samples": 2000,
            "n_estimators": 100,
        },
        "performance_metrics": {
            "r2_score": 0.8104269688784731,
            "rmse": 0.12,
            "cv_scores": [0.8, 0.81, 0.82],
            "feature_importances": {"specialty": 0.5, "region": 0.3, "decile": 0.2},
            "training_samples": 2000,
            "training_duration_seconds": duration,
        },
        "fidelity_score": fidelity_score,
        "fidelity_sample_count": sample_count,
    }


@pytest.mark.asyncio
async def test_list_models_states_shared_fit_and_synthetic_r2_basis(mock_twin_repository):
    """The listing derives 'shared fit' from the rows' fit fingerprints (not a
    hardcoded sentence), labels R² as a synthetic-target score, says brand is not a
    feature, and reports a NULL fidelity as UNVALIDATED."""
    from src.api.routes.digital_twin import list_models

    rows = [
        _shared_fit_row("Remibrutinib", duration=7.1),
        _shared_fit_row("Fabhalta", duration=7.05),
        _shared_fit_row("Kisqali", duration=7.69),
    ]
    mock_twin_repository.list_active_models = AsyncMock(return_value=rows)

    result = await list_models(brand=None, twin_type=None, user=_ADMIN)

    assert result.total_count == 3
    fingerprints = {m.training_fingerprint for m in result.models}
    assert len(fingerprints) == 1, "one fit → one fingerprint (duration is not part of the fit)"
    for m in result.models:
        assert m.fidelity_status.value == "unvalidated"
        assert m.fidelity_score is None
        assert m.fidelity_sample_count == 0
        assert m.data_provenance == "synthetic"
        assert m.r2_score_basis.value == "synthetic_target"
        assert m.brand_is_feature is False
        assert m.shared_fit_model_count == 3
        assert set(m.shared_fit_with) == {"Remibrutinib", "Fabhalta", "Kisqali"} - {m.brand}


@pytest.mark.asyncio
async def test_list_models_distinct_fits_are_not_called_shared(mock_twin_repository):
    from src.api.routes.digital_twin import list_models

    a = _shared_fit_row("Remibrutinib", duration=7.1)
    b = _shared_fit_row("Kisqali", duration=7.1)
    b["performance_metrics"] = {**b["performance_metrics"], "r2_score": 0.61}
    b["training_config"] = {**b["training_config"], "data_provenance": "rwd_file"}
    b["fidelity_score"] = 0.9
    b["fidelity_sample_count"] = 4
    mock_twin_repository.list_active_models = AsyncMock(return_value=[a, b])

    result = await list_models(brand=None, twin_type=None, user=_ADMIN)

    by_brand = {m.brand: m for m in result.models}
    assert by_brand["Remibrutinib"].training_fingerprint != by_brand["Kisqali"].training_fingerprint
    for m in result.models:
        assert m.shared_fit_model_count == 1
        assert m.shared_fit_with == []
    assert by_brand["Kisqali"].r2_score_basis.value == "rwd_target"
    assert by_brand["Kisqali"].fidelity_status.value == "validated"
    assert by_brand["Kisqali"].fidelity_score == 0.9


@pytest.mark.asyncio
async def test_list_models_brand_scoped_caller_sees_the_shared_count_not_other_brands(
    mock_twin_repository,
):
    """H11 brand scoping: a Kisqali-only viewer gets Kisqali's row, and the fact
    that its fit is shared (count over ALL active models), but not the other
    brands' names."""
    from src.api.routes.digital_twin import list_models

    rows = [
        _shared_fit_row("Remibrutinib", duration=7.1),
        _shared_fit_row("Fabhalta", duration=7.05),
        _shared_fit_row("Kisqali", duration=7.69),
    ]
    mock_twin_repository.list_active_models = AsyncMock(return_value=rows)
    viewer = {"user_id": "v", "role": "viewer", "brands": ["Kisqali"]}

    result = await list_models(brand=None, twin_type=None, user=viewer)

    assert [m.brand for m in result.models] == ["Kisqali"]
    only = result.models[0]
    assert only.shared_fit_model_count == 3
    assert only.shared_fit_with == []
    # The census ran over every brand (brand=None), the listing was filtered after.
    assert mock_twin_repository.list_active_models.await_args.kwargs.get("brand") is None


@pytest.mark.asyncio
async def test_get_model_detail_carries_the_honesty_fields(mock_twin_repository):
    from src.api.routes.digital_twin import get_model

    rows = [_shared_fit_row("Remibrutinib", duration=7.1), _shared_fit_row("Kisqali", duration=7.7)]
    mock_twin_repository.list_active_models = AsyncMock(return_value=rows)
    mock_twin_repository.get_model = AsyncMock(return_value=rows[1])

    detail = await get_model(model_id=rows[1]["model_id"], user=_ADMIN)

    assert detail.fidelity_status.value == "unvalidated"
    assert detail.r2_score_basis.value == "synthetic_target"
    assert detail.brand_is_feature is False
    assert detail.shared_fit_model_count == 2
    assert detail.shared_fit_with == ["Remibrutinib"]


@pytest.mark.asyncio
async def test_run_simulation_hands_the_model_fidelity_to_the_engine_and_reports_status(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    """The route resolved the model row but never passed its fidelity_score to the
    engine — the gate could not fire even with a real score (#2206). It now does,
    and the response states the fidelity status explicitly."""
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
        run_simulation,
    )
    from src.digital_twin import simulation_engine as engine_mod
    from src.digital_twin.models.simulation_models import FidelityStatus

    row = mock_twin_repository.list_active_models.return_value[0]
    row["fidelity_score"] = 0.55
    row["fidelity_sample_count"] = 3
    mock_simulation_engine.simulate.return_value.fidelity_status = FidelityStatus.BELOW_THRESHOLD

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
    result = await run_simulation(request, {"user_id": "test_user", "role": "operator"})

    assert engine_mod.SimulationEngine.call_args.kwargs["model_fidelity_score"] == 0.55
    assert result.fidelity_status.value == "below_threshold"


@pytest.mark.asyncio
async def test_run_simulation_with_a_null_model_fidelity_reports_unvalidated(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
        run_simulation,
    )
    from src.digital_twin import simulation_engine as engine_mod
    from src.digital_twin.models.simulation_models import FidelityStatus

    row = mock_twin_repository.list_active_models.return_value[0]
    row["fidelity_score"] = None
    row["fidelity_sample_count"] = 0
    mock_simulation_engine.simulate.return_value.fidelity_status = FidelityStatus.UNVALIDATED

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
    result = await run_simulation(request, {"user_id": "test_user", "role": "operator"})

    assert "model_fidelity_score" in engine_mod.SimulationEngine.call_args.kwargs
    assert engine_mod.SimulationEngine.call_args.kwargs["model_fidelity_score"] is None
    assert result.fidelity_status.value == "unvalidated"


@pytest.mark.asyncio
async def test_get_simulation_stored_row_derives_fidelity_status_from_its_model(
    mock_twin_repository,
):
    """twin_simulations persists only the old gate's verdict (a NULL score passed as
    fidelity_warning=False). The stored read derives status AND warning from the
    model row it points at, as it stands now (NULL there → unvalidated + warning)."""
    from src.api.routes.digital_twin import get_simulation

    sim_row = mock_twin_repository.get_simulation.return_value
    sim_row["fidelity_warning"] = False  # what the silent gate recorded at run time
    sim_row["fidelity_warning_reason"] = None
    model_row = _shared_fit_row("Remibrutinib", duration=7.1)
    model_row["model_id"] = str(sim_row["model_id"])
    mock_twin_repository.get_model = AsyncMock(return_value=model_row)

    detail = await get_simulation(simulation_id=str(sim_row["simulation_id"]), user=_ADMIN)

    assert detail.fidelity_status.value == "unvalidated"
    assert detail.model_fidelity_score is None
    assert detail.fidelity_warning is True
    assert "unvalidated" in (detail.fidelity_warning_reason or "").lower()
    mock_twin_repository.get_model.assert_awaited_once()

    model_row["fidelity_score"] = 0.91
    model_row["fidelity_sample_count"] = 2
    detail2 = await get_simulation(simulation_id=str(sim_row["simulation_id"]), user=_ADMIN)
    assert detail2.fidelity_status.value == "validated"
    assert detail2.model_fidelity_score == 0.91
    assert detail2.fidelity_warning is False


@pytest.mark.asyncio
async def test_run_simulation_persists_the_experiment_link_it_was_given(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    """codex r1 #1: SimulateRequest accepted experiment_design_id and dropped it on
    save, so no twin_simulations row was ever linked and the fidelity producer
    (experiment-scoped resolution) could only skip. The link is now written."""
    from uuid import UUID

    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
        run_simulation,
    )

    exp_id = uuid4()
    saved_id = uuid4()
    mock_twin_repository.client = _experiment_lookup([{"id": str(exp_id), "brand": "Remibrutinib"}])
    mock_twin_repository.save_simulation = AsyncMock(return_value=saved_id)
    mock_twin_repository.simulations.link_experiment = AsyncMock(return_value=True)

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
        experiment_design_id=str(exp_id),
    )
    await run_simulation(request, {"user_id": "test_user", "role": "operator"})

    mock_twin_repository.simulations.link_experiment.assert_awaited_once_with(
        saved_id, UUID(str(exp_id))
    )


@pytest.mark.asyncio
async def test_run_simulation_rejects_a_malformed_experiment_link_before_simulating(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
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
        experiment_design_id="not-a-uuid",
    )
    with pytest.raises(HTTPException) as exc:
        await run_simulation(request, {"user_id": "test_user", "role": "operator"})
    assert exc.value.status_code == 422
    assert "experiment_design_id" in str(exc.value.detail)
    mock_simulation_engine.simulate.assert_not_called()
    mock_twin_repository.save_simulation.assert_not_called()


def _experiment_lookup(rows):
    """repo.client.table('ml_experiments').select(...).eq(...).limit(1).execute() (async)."""
    chain = MagicMock()
    chain.table.return_value.select.return_value.eq.return_value.limit.return_value.execute = (
        AsyncMock(return_value=MagicMock(data=rows))
    )
    return chain


def _linked_request(exp_id, brand=None):
    from src.api.routes.digital_twin import (
        BrandEnum,
        InterventionConfigRequest,
        SimulateRequest,
        TwinTypeEnum,
    )

    return SimulateRequest(
        intervention=InterventionConfigRequest(
            intervention_type="email_campaign",
            channel="email",
            frequency="weekly",
            duration_weeks=8,
        ),
        brand=brand or BrandEnum.REMIBRUTINIB,
        twin_type=TwinTypeEnum.HCP,
        twin_count=1000,
        experiment_design_id=str(exp_id),
    )


@pytest.mark.asyncio
async def test_run_simulation_verifies_the_experiment_exists_before_simulating(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    """codex r2 #1: a well-formed but nonexistent experiment id would have produced a
    permanently orphaned pre-screen. It is now a 404 before any heavy work."""
    from src.api.routes.digital_twin import run_simulation

    mock_twin_repository.client = _experiment_lookup([])
    with pytest.raises(HTTPException) as exc:
        await run_simulation(_linked_request(uuid4()), {"user_id": "u", "role": "operator"})
    assert exc.value.status_code == 404
    mock_simulation_engine.simulate.assert_not_called()


@pytest.mark.asyncio
async def test_run_simulation_refuses_to_link_an_experiment_of_another_brand(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    from src.api.routes.digital_twin import run_simulation

    exp_id = uuid4()
    mock_twin_repository.client = _experiment_lookup([{"id": str(exp_id), "brand": "Kisqali"}])
    with pytest.raises(HTTPException) as exc:
        await run_simulation(_linked_request(exp_id), {"user_id": "u", "role": "operator"})
    assert exc.value.status_code == 422
    assert "Kisqali" in str(exc.value.detail)
    mock_simulation_engine.simulate.assert_not_called()


@pytest.mark.asyncio
async def test_run_simulation_surfaces_a_failed_link_instead_of_returning_success(
    mock_twin_generator, mock_simulation_engine, mock_twin_repository, mock_twin_hydrate
):
    from src.api.routes.digital_twin import run_simulation

    exp_id = uuid4()
    saved_id = uuid4()
    mock_twin_repository.client = _experiment_lookup([{"id": str(exp_id), "brand": "Remibrutinib"}])
    mock_twin_repository.save_simulation = AsyncMock(return_value=saved_id)
    mock_twin_repository.simulations.link_experiment = AsyncMock(return_value=False)
    with pytest.raises(HTTPException) as exc:
        await run_simulation(_linked_request(exp_id), {"user_id": "u", "role": "operator"})
    assert exc.value.status_code == 500
    assert str(saved_id) in str(exc.value.detail)
    assert str(exp_id) in str(exc.value.detail)


def test_fit_fingerprint_is_stable_across_numeric_representation_and_key_order():
    """codex r3 #2: `1` vs `1.0` (JSONB round-trips) must not split one recorded fit."""
    from src.api.routes.digital_twin import _fit_fingerprint

    a = _shared_fit_row("Remibrutinib", duration=7.1)
    b = _shared_fit_row("Kisqali", duration=9.9)
    b["training_config"] = {
        "n_estimators": 100.0,
        "algorithm": "random_forest",
        "training_samples": 2000.0,
        "data_provenance": "synthetic",
    }
    b["performance_metrics"] = {**b["performance_metrics"], "cv_scores": [0.8, 0.81, 0.82]}
    assert _fit_fingerprint(a) == _fit_fingerprint(b)


def test_fit_fingerprint_includes_the_recorded_training_frame():
    """Two trainings that differ only in the recorded frame (seed) are different fits
    even when every reported metric happens to coincide."""
    from src.api.routes.digital_twin import _fit_fingerprint

    a = _shared_fit_row("Remibrutinib", duration=7.1)
    b = _shared_fit_row("Kisqali", duration=7.1)
    a["training_config"] = {
        **a["training_config"],
        "training_frame": {"source": "synthetic", "seed": 0, "n_rows": 2000},
    }
    b["training_config"] = {
        **b["training_config"],
        "training_frame": {"source": "synthetic", "seed": 1, "n_rows": 2000},
    }
    assert _fit_fingerprint(a) != _fit_fingerprint(b)


def test_fit_fingerprint_does_not_collapse_large_integers():
    """codex r4 #2: int→float normalisation collides above 2**53; ints stay exact."""
    from src.api.routes.digital_twin import _fit_fingerprint

    a = _shared_fit_row("Remibrutinib", duration=1.0)
    b = _shared_fit_row("Kisqali", duration=1.0)
    a["training_config"] = {**a["training_config"], "seed": 2**53 + 1}
    b["training_config"] = {**b["training_config"], "seed": 2**53 + 2}
    assert _fit_fingerprint(a) != _fit_fingerprint(b)
    # ... while an integral float still equals its int.
    c = _shared_fit_row("Fabhalta", duration=1.0)
    c["training_config"] = {**c["training_config"], "seed": 7.0}
    d = _shared_fit_row("Fabhalta", duration=1.0)
    d["training_config"] = {**d["training_config"], "seed": 7}
    assert _fit_fingerprint(c) == _fit_fingerprint(d)


@pytest.mark.asyncio
async def test_list_models_says_whether_the_training_frame_was_recorded(mock_twin_repository):
    """codex r4 #2: legacy rows carry no frame identity; the listing says so, so the
    UI can scope its 'same frame' claim to rows that recorded one."""
    from src.api.routes.digital_twin import list_models

    legacy = _shared_fit_row("Remibrutinib", duration=7.1)
    recorded = _shared_fit_row("Kisqali", duration=7.1)
    recorded["training_config"] = {
        **recorded["training_config"],
        "training_frame": {"source": "synthetic_training_frame", "seed": 0, "n_rows": 2000},
    }
    mock_twin_repository.list_active_models = AsyncMock(return_value=[legacy, recorded])

    result = await list_models(brand=None, twin_type=None, user=_ADMIN)

    by_brand = {m.brand: m for m in result.models}
    assert by_brand["Remibrutinib"].training_frame_recorded is False
    assert by_brand["Kisqali"].training_frame_recorded is True
    # A recorded frame is part of the fingerprint: these two are not one recorded fit.
    assert by_brand["Remibrutinib"].training_fingerprint != by_brand["Kisqali"].training_fingerprint
