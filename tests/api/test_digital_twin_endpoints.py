"""
Tests for Digital Twin API endpoints.

Phase 2A of API Audit - Digital Twin API
Tests organized by batch as per api-endpoints-audit-plan.md

Endpoints covered:
- Batch 2A.1: Simulation Core (POST /simulate, GET /simulations, GET /simulations/{id}, POST /validate)
- Batch 2A.2: Model Management (GET /models, GET /models/{id}, GET /models/{id}/fidelity, GET /models/{id}/fidelity/report)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _patch_async_supabase_client():
    """Stop the route's ``_get_twin_repo`` helper (#705 H6) from reaching for a
    real Supabase client during these HTTP-level tests. Each test still patches
    ``TwinRepository`` so the (mocked) client argument is irrelevant.
    """
    with patch(
        "src.memory.services.factories.get_async_supabase_client",
        new=AsyncMock(return_value=MagicMock()),
    ):
        yield


@pytest.fixture
def identified_cohort():
    """Direction-2 identification gate: make it PASS for the /simulate success tests.

    The route builds a cohort provider from ``repo.client`` and honestly 422s
    ("No effect data available ...") when that returns None — which a bare MagicMock
    client guarantees, because the loader awaits it. Patch the source module (the
    route imports the function locally) at the same seam as the unit route tests'
    ``_default_identified_cohort`` (theirs is autouse via monkeypatch; this one is
    opt-in), so these tests exercise the simulate path they were written for.
    """
    with patch(
        "src.digital_twin.effect.cohort_loader.build_cohort_provider_or_none",
        new=AsyncMock(return_value=MagicMock()),
    ):
        yield


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simulate_request():
    """Sample simulation request."""
    return {
        "intervention": {
            "intervention_type": "email_campaign",
            "channel": "email",
            "frequency": "weekly",
            "duration_weeks": 8,
            "personalization_level": "high",
            "target_deciles": [1, 2, 3],
        },
        "brand": "Remibrutinib",
        "twin_type": "hcp",
        "twin_count": 1000,
        "confidence_level": 0.95,
        "calculate_heterogeneity": True,
    }


@pytest.fixture
def validate_request():
    """Sample validation request."""
    return {
        "simulation_id": "550e8400-e29b-41d4-a716-446655440000",
        "experiment_id": "660e8400-e29b-41d4-a716-446655440000",
        "actual_ate": 0.072,
        "actual_ci_lower": 0.045,
        "actual_ci_upper": 0.099,
        "actual_sample_size": 5000,
        "validation_notes": "Post-campaign analysis",
    }


@pytest.fixture
def mock_simulation_result():
    """Mock simulation result object."""
    from src.digital_twin.models.simulation_models import EffectHeterogeneity

    result = MagicMock()
    result.simulation_id = UUID("550e8400-e29b-41d4-a716-446655440000")
    result.model_id = UUID("660e8400-e29b-41d4-a716-446655440000")
    result.twin_count = 10000
    result.simulated_ate = 0.085
    result.simulated_ci_lower = 0.065
    result.simulated_ci_upper = 0.105
    result.simulated_std_error = 0.010
    result.effect_size_cohens_d = 0.25
    result.statistical_power = 0.85
    result.recommendation = MagicMock(value="deploy")
    result.recommendation_rationale = "Effect size significant"
    result.recommended_sample_size = 5000
    result.recommended_duration_weeks = 8
    result.simulation_confidence = 0.92
    result.fidelity_warning = False
    result.fidelity_warning_reason = None
    result.model_fidelity_score = 0.88
    # A real domain value (#2206): the route maps result.fidelity_status.value.
    from src.digital_twin.models.simulation_models import FidelityStatus

    result.fidelity_status = FidelityStatus.VALIDATED
    result.status = MagicMock(value="completed")
    result.error_message = None
    result.execution_time_ms = 1500
    result.created_at = datetime.now(timezone.utc)
    result.is_significant = MagicMock(return_value=True)
    result.effect_direction = MagicMock(return_value="positive")
    # The response mapping reads these SimulationResult fields (#705 H5b, #2053); a
    # MagicMock attribute fails ``SimulationResponse`` validation (a ValueError → 400).
    result.data_provenance = "cohort_estimated_synthetic_gold_v1"
    result.target_regions = []
    result.cohort_ate = None
    result.cohort_ci_lower = None
    result.cohort_ci_upper = None
    # Keep the response boundary honest: production returns this domain model,
    # and the API intentionally rejects unconstrained MagicMock values.
    result.effect_heterogeneity = EffectHeterogeneity()
    return result


@pytest.fixture
def mock_simulation_data():
    """Mock simulation data from repository."""
    return {
        "simulation_id": "550e8400-e29b-41d4-a716-446655440000",
        "model_id": "660e8400-e29b-41d4-a716-446655440000",
        "intervention_type": "email_campaign",
        "brand": "Remibrutinib",
        "twin_type": "hcp",
        "twin_count": 10000,
        "simulated_ate": 0.085,
        "simulated_ci_lower": 0.065,
        "simulated_ci_upper": 0.105,
        "simulated_std_error": 0.010,
        "recommendation": "deploy",
        "simulation_status": "completed",
        "simulation_confidence": 0.92,
        "execution_time_ms": 1500,
        "created_at": datetime.now(timezone.utc),
    }


@pytest.fixture
def mock_model_data():
    """Mock model data from repository."""
    return {
        "model_id": "770e8400-e29b-41d4-a716-446655440000",
        "model_name": "hcp_email_twin_v2",
        "model_description": "HCP email campaign twin generator",
        "twin_type": "hcp",
        "brand": "Remibrutinib",
        "algorithm": "gradient_boosting",
        "feature_columns": ["specialty", "decile", "region", "baseline_trx"],
        "target_column": "trx_response",
        "r2_score": 0.78,
        "rmse": 0.045,
        "cv_mean": 0.76,
        "cv_std": 0.03,
        "feature_importances": {
            "specialty": 0.35,
            "decile": 0.30,
            "region": 0.20,
            "baseline_trx": 0.15,
        },
        "top_features": ["specialty", "decile", "region"],
        "training_samples": 50000,
        "training_duration_seconds": 125.5,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
        "config": {"max_depth": 6, "n_estimators": 100},
    }


@pytest.fixture
def mock_fidelity_record():
    """Mock fidelity record object."""
    record = MagicMock()
    record.tracking_id = UUID("880e8400-e29b-41d4-a716-446655440000")
    record.simulation_id = UUID("550e8400-e29b-41d4-a716-446655440000")
    record.actual_experiment_id = UUID("660e8400-e29b-41d4-a716-446655440000")
    record.simulated_ate = 0.085
    record.simulated_ci_lower = 0.065
    record.simulated_ci_upper = 0.105
    record.actual_ate = 0.072
    record.actual_ci_lower = 0.045
    record.actual_ci_upper = 0.099
    record.actual_sample_size = 5000
    record.prediction_error = 0.013
    record.absolute_error = 0.013
    record.ci_coverage = True
    record.fidelity_grade = MagicMock(value="good")
    record.validation_notes = "Post-campaign validation"
    record.confounding_factors = []
    record.created_at = datetime.now(timezone.utc)
    record.validated_at = datetime.now(timezone.utc)
    record.validated_by = "analyst_001"
    return record


# =============================================================================
# BATCH 2A.1 - SIMULATION CORE TESTS
# =============================================================================


class TestRunSimulation:
    """Tests for POST /digital-twin/simulate."""

    def test_run_simulation_success(
        self, simulate_request, mock_simulation_result, identified_cohort
    ):
        """Should run simulation and return results."""
        mock_generator = MagicMock()
        mock_generator.generate = MagicMock(return_value=[])
        mock_generator.model_id = UUID("660e8400-e29b-41d4-a716-446655440000")

        mock_engine = MagicMock()
        mock_engine.simulate = MagicMock(return_value=mock_simulation_result)

        mock_repo = MagicMock()
        mock_repo.save_simulation = AsyncMock()
        # #705 H4: /simulate resolves + loads an active model before generating.
        mock_repo.list_active_models = AsyncMock(
            return_value=[
                {
                    "model_id": "660e8400-e29b-41d4-a716-446655440000",
                    "twin_type": "hcp",
                    "brand": "Remibrutinib",
                    "is_active": True,
                    "mlflow_model_uri": "models:/m-x",
                    "mlflow_run_id": "run-x",
                }
            ]
        )

        with (
            patch("src.digital_twin.twin_generator.TwinGenerator", return_value=mock_generator),
            patch("src.digital_twin.simulation_engine.SimulationEngine", return_value=mock_engine),
            patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo),
            patch("src.digital_twin.twin_persistence.hydrate_generator", return_value=True),
        ):
            response = client.post("/api/digital-twin/simulate", json=simulate_request)

        assert response.status_code == 200
        data = response.json()
        assert "simulation_id" in data
        assert data["brand"] == "Remibrutinib"
        assert data["twin_type"] == "hcp"
        assert "simulated_ate" in data
        assert "recommendation" in data
        assert data["status"] == "completed"

    def test_run_simulation_with_population_filters(
        self, simulate_request, mock_simulation_result, identified_cohort
    ):
        """Should run simulation with population filters."""
        simulate_request["population_filters"] = {
            "specialties": ["oncology", "hematology"],
            "deciles": [1, 2, 3],
            "regions": ["northeast"],
        }

        mock_generator = MagicMock()
        mock_generator.generate = MagicMock(return_value=[])
        mock_generator.model_id = UUID("660e8400-e29b-41d4-a716-446655440000")

        mock_engine = MagicMock()
        mock_engine.simulate = MagicMock(return_value=mock_simulation_result)

        mock_repo = MagicMock()
        mock_repo.save_simulation = AsyncMock()
        # #705 H4: /simulate resolves + loads an active model before generating.
        mock_repo.list_active_models = AsyncMock(
            return_value=[
                {
                    "model_id": "660e8400-e29b-41d4-a716-446655440000",
                    "twin_type": "hcp",
                    "brand": "Remibrutinib",
                    "is_active": True,
                    "mlflow_model_uri": "models:/m-x",
                    "mlflow_run_id": "run-x",
                }
            ]
        )

        with (
            patch("src.digital_twin.twin_generator.TwinGenerator", return_value=mock_generator),
            patch("src.digital_twin.simulation_engine.SimulationEngine", return_value=mock_engine),
            patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo),
            patch("src.digital_twin.twin_persistence.hydrate_generator", return_value=True),
        ):
            response = client.post("/api/digital-twin/simulate", json=simulate_request)

        assert response.status_code == 200
        # The filter must reach the engine: the route hands it over as the
        # ``population_filter`` kwarg (a PopulationFilter), and the mocked engine
        # discards it, so status alone cannot see a regions regression.
        passed_filter = mock_engine.simulate.call_args.kwargs["population_filter"]
        assert passed_filter.regions == ["northeast"]

    def test_run_simulation_invalid_brand(self, simulate_request):
        """Should return 422 for invalid brand."""
        simulate_request["brand"] = "invalid_brand"

        response = client.post("/api/digital-twin/simulate", json=simulate_request)

        assert response.status_code == 422


class TestListSimulations:
    """Tests for GET /digital-twin/simulations."""

    def test_list_simulations_success(self, mock_simulation_data):
        """Should list simulations with pagination."""
        mock_repo = MagicMock()
        mock_repo.simulations.list_simulations = AsyncMock(return_value=[mock_simulation_data])

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get("/api/digital-twin/simulations")

        assert response.status_code == 200
        data = response.json()
        assert "total_count" in data
        assert "simulations" in data
        assert "page" in data
        assert "page_size" in data

    def test_list_simulations_filter_by_brand(self, mock_simulation_data):
        """Should filter simulations by brand."""
        mock_repo = MagicMock()
        mock_repo.simulations.list_simulations = AsyncMock(return_value=[mock_simulation_data])

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get(
                "/api/digital-twin/simulations",
                params={"brand": "Remibrutinib"},
            )

        assert response.status_code == 200

    def test_list_simulations_pagination(self, mock_simulation_data):
        """Should handle pagination parameters."""
        mock_repo = MagicMock()
        mock_repo.simulations.list_simulations = AsyncMock(return_value=[mock_simulation_data])

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get(
                "/api/digital-twin/simulations",
                params={"page": 2, "page_size": 10},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 10


class TestGetSimulation:
    """Tests for GET /digital-twin/simulations/{simulation_id}."""

    def test_get_simulation_success(self):
        """Should return simulation details."""
        # repo.get_simulation returns the RAW twin_simulations ROW (a dict), not a
        # SimulationResult (#705 H5b/H11); the route maps from it with ``.get``.
        # Mirror the real row shape (as the unit route tests do).
        mock_result = {
            "simulation_id": "550e8400-e29b-41d4-a716-446655440000",
            "model_id": "660e8400-e29b-41d4-a716-446655440000",
            "intervention_type": "email_campaign",
            "intervention_config": {"intervention_type": "email_campaign", "channel": "email"},
            "brand": "Remibrutinib",
            "twin_count": 10000,
            "simulated_ate": 0.085,
            "simulated_ci_lower": 0.065,
            "simulated_ci_upper": 0.105,
            "simulated_std_error": 0.010,
            "effect_size_cohens_d": 0.25,
            "statistical_power": 0.85,
            "recommendation": "deploy",
            "recommendation_rationale": "Significant effect",
            "recommended_sample_size": 5000,
            "recommended_duration_weeks": 8,
            "simulation_confidence": 0.92,
            "fidelity_warning": False,
            "fidelity_warning_reason": None,
            "model_fidelity_score": 0.88,
            "simulation_status": "completed",
            "data_provenance": "cohort_estimated_synthetic_gold_v1",
            "error_message": None,
            "execution_time_ms": 1500,
            "created_at": datetime.now(timezone.utc),
            "completed_at": datetime.now(timezone.utc),
            "population_filters": {"deciles": [1, 2, 3]},
            "effect_heterogeneity": {
                "by_specialty": {"oncology": {"ate": 0.12}},
                "by_decile": {"1": {"ate": 0.15}},
                "by_region": {"northeast": {"ate": 0.10}},
                "by_adoption_stage": {"early": {"ate": 0.14}},
                "top_segments": [{"segment": "oncology_d1", "ate": 0.18}],
            },
        }

        mock_repo = MagicMock()
        mock_repo.get_simulation = AsyncMock(return_value=mock_result)

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get(
                "/api/digital-twin/simulations/550e8400-e29b-41d4-a716-446655440000"
            )

        assert response.status_code == 200
        data = response.json()
        assert "simulation_id" in data
        assert "effect_heterogeneity" in data
        assert "intervention_config" in data

    def test_get_simulation_not_found(self):
        """Should return 404 for missing simulation."""
        mock_repo = MagicMock()
        mock_repo.get_simulation = AsyncMock(return_value=None)

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get(
                "/api/digital-twin/simulations/999e8400-e29b-41d4-a716-446655440000"
            )

        assert response.status_code == 404


class TestValidateSimulation:
    """Tests for POST /digital-twin/validate."""

    def test_validate_simulation_success(
        self, validate_request, mock_simulation_data, mock_fidelity_record
    ):
        """Should validate simulation and return fidelity record."""
        mock_repo = MagicMock()
        mock_repo.get_simulation = AsyncMock(return_value=mock_simulation_data)

        # get_simulation_record / validate are async coroutines (#705 H7); the
        # route awaits them, so they must be AsyncMocks.
        mock_tracker = MagicMock()
        mock_tracker.get_simulation_record = AsyncMock(return_value=mock_fidelity_record)
        mock_tracker.record_prediction = AsyncMock(return_value=mock_fidelity_record)
        mock_tracker.validate = AsyncMock(return_value=mock_fidelity_record)

        with (
            patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo),
            patch("src.digital_twin.fidelity_tracker.FidelityTracker", return_value=mock_tracker),
        ):
            response = client.post("/api/digital-twin/validate", json=validate_request)

        assert response.status_code == 200
        data = response.json()
        assert "tracking_id" in data
        assert "simulation_id" in data
        assert "fidelity_grade" in data
        assert data["actual_ate"] == 0.072

    def test_validate_simulation_not_found(self, validate_request):
        """Should return 404 for missing simulation."""
        mock_repo = MagicMock()
        mock_repo.get_simulation = AsyncMock(return_value=None)

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.post("/api/digital-twin/validate", json=validate_request)

        assert response.status_code == 404


# =============================================================================
# BATCH 2A.2 - MODEL MANAGEMENT TESTS
# =============================================================================


class TestListModels:
    """Tests for GET /digital-twin/models."""

    def test_list_models_success(self, mock_model_data):
        """Should list twin models."""
        mock_repo = MagicMock()
        mock_repo.list_active_models = AsyncMock(return_value=[mock_model_data])

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get("/api/digital-twin/models")

        assert response.status_code == 200
        data = response.json()
        assert "total_count" in data
        assert "models" in data
        assert len(data["models"]) == 1

    def test_list_models_filter_by_brand(self, mock_model_data):
        """Should filter models by brand."""
        mock_repo = MagicMock()
        mock_repo.list_active_models = AsyncMock(return_value=[mock_model_data])

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get("/api/digital-twin/models", params={"brand": "Remibrutinib"})

        assert response.status_code == 200

    def test_list_models_filter_by_twin_type(self, mock_model_data):
        """Should filter models by twin type."""
        mock_repo = MagicMock()
        mock_repo.list_active_models = AsyncMock(return_value=[mock_model_data])

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get("/api/digital-twin/models", params={"twin_type": "hcp"})

        assert response.status_code == 200


class TestGetModel:
    """Tests for GET /digital-twin/models/{model_id}."""

    def test_get_model_success(self, mock_model_data):
        """Should return model details."""
        mock_repo = MagicMock()
        mock_repo.get_model = AsyncMock(return_value=mock_model_data)
        # The detail runs the shared-fit census over the active models (#2206).
        mock_repo.list_active_models = AsyncMock(return_value=[mock_model_data])

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get("/api/digital-twin/models/770e8400-e29b-41d4-a716-446655440000")

        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "770e8400-e29b-41d4-a716-446655440000"
        assert "feature_columns" in data
        assert "feature_importances" in data
        assert "r2_score" in data

    def test_get_model_not_found(self):
        """Should return 404 for missing model."""
        mock_repo = MagicMock()
        mock_repo.get_model = AsyncMock(return_value=None)

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get("/api/digital-twin/models/999e8400-e29b-41d4-a716-446655440000")

        assert response.status_code == 404


class TestGetModelFidelity:
    """Tests for GET /digital-twin/models/{model_id}/fidelity."""

    def test_get_model_fidelity_success(self, mock_fidelity_record, mock_model_data):
        """Should return fidelity history."""
        mock_repo = MagicMock()
        mock_repo.get_model = AsyncMock(return_value=mock_model_data)
        mock_repo.get_model_fidelity_records = AsyncMock(return_value=[mock_fidelity_record])

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get(
                "/api/digital-twin/models/770e8400-e29b-41d4-a716-446655440000/fidelity"
            )

        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
        assert "total_validations" in data
        assert "grade_distribution" in data
        assert "records" in data

    def test_get_model_fidelity_validated_only(self, mock_fidelity_record, mock_model_data):
        """Should filter to validated records only."""
        mock_repo = MagicMock()
        mock_repo.get_model = AsyncMock(return_value=mock_model_data)
        mock_repo.get_model_fidelity_records = AsyncMock(return_value=[mock_fidelity_record])

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get(
                "/api/digital-twin/models/770e8400-e29b-41d4-a716-446655440000/fidelity",
                params={"validated_only": True},
            )

        assert response.status_code == 200


class TestGetFidelityReport:
    """Tests for GET /digital-twin/models/{model_id}/fidelity/report."""

    def test_get_fidelity_report_success(self, mock_model_data):
        """Should return fidelity report with trend analysis."""
        mock_repo = MagicMock()
        mock_repo.get_model = AsyncMock(return_value=mock_model_data)

        mock_tracker = MagicMock()
        mock_tracker.get_model_fidelity_report = MagicMock(
            return_value={
                "model_id": "770e8400-e29b-41d4-a716-446655440000",
                "validation_count": 15,
                "fidelity_score": 0.82,
                "degradation_alert": False,
                "metrics": {
                    "ci_coverage_rate": 0.87,
                    "mean_absolute_error": 0.025,
                },
                "grade_distribution": {
                    "excellent": 5,
                    "good": 8,
                    "fair": 2,
                    "poor": 0,
                    "unvalidated": 0,
                },
                "computed_at": datetime.now(timezone.utc),
            }
        )

        with (
            patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo),
            patch("src.digital_twin.fidelity_tracker.FidelityTracker", return_value=mock_tracker),
        ):
            response = client.get(
                "/api/digital-twin/models/770e8400-e29b-41d4-a716-446655440000/fidelity/report"
            )

        assert response.status_code == 200
        data = response.json()
        assert "model_id" in data
        assert "total_validations" in data
        assert "average_fidelity_score" in data
        assert "trend" in data
        assert "is_degrading" in data
        assert "recommendation" in data

    def test_get_fidelity_report_with_lookback(self, mock_model_data):
        """Should respect lookback_days parameter."""
        mock_repo = MagicMock()
        mock_repo.get_model = AsyncMock(return_value=mock_model_data)

        mock_tracker = MagicMock()
        mock_tracker.get_model_fidelity_report = MagicMock(
            return_value={
                "model_id": "770e8400-e29b-41d4-a716-446655440000",
                "validation_count": 5,
                "fidelity_score": 0.75,
                "degradation_alert": False,
                "metrics": {"ci_coverage_rate": 0.80},
                "grade_distribution": {
                    "excellent": 1,
                    "good": 3,
                    "fair": 1,
                    "poor": 0,
                    "unvalidated": 0,
                },
                "computed_at": datetime.now(timezone.utc),
            }
        )

        with (
            patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo),
            patch("src.digital_twin.fidelity_tracker.FidelityTracker", return_value=mock_tracker),
        ):
            response = client.get(
                "/api/digital-twin/models/770e8400-e29b-41d4-a716-446655440000/fidelity/report",
                params={"lookback_days": 30},
            )

        assert response.status_code == 200

    def test_get_fidelity_report_degrading(self, mock_model_data):
        """Should detect degrading model fidelity."""
        mock_repo = MagicMock()
        mock_repo.get_model = AsyncMock(return_value=mock_model_data)

        mock_tracker = MagicMock()
        mock_tracker.get_model_fidelity_report = MagicMock(
            return_value={
                "model_id": "770e8400-e29b-41d4-a716-446655440000",
                "validation_count": 20,
                "fidelity_score": 0.55,
                "degradation_alert": True,
                "metrics": {"ci_coverage_rate": 0.65},
                "grade_distribution": {
                    "excellent": 2,
                    "good": 5,
                    "fair": 8,
                    "poor": 5,
                    "unvalidated": 0,
                },
                "computed_at": datetime.now(timezone.utc),
            }
        )

        with (
            patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo),
            patch("src.digital_twin.fidelity_tracker.FidelityTracker", return_value=mock_tracker),
        ):
            response = client.get(
                "/api/digital-twin/models/770e8400-e29b-41d4-a716-446655440000/fidelity/report"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["is_degrading"] is True
        assert "retrain" in data["recommendation"].lower()
