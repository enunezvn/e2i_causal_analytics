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
    result.status = MagicMock(value="completed")
    result.error_message = None
    result.execution_time_ms = 1500
    result.created_at = datetime.now(timezone.utc)
    result.is_significant = MagicMock(return_value=True)
    result.effect_direction = MagicMock(return_value="positive")
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

    def test_run_simulation_success(self, simulate_request, mock_simulation_result):
        """Should run simulation and return results."""
        mock_generator = MagicMock()
        mock_generator.generate = MagicMock(return_value=[])
        mock_generator.model_id = UUID("660e8400-e29b-41d4-a716-446655440000")

        mock_engine = MagicMock()
        mock_engine.simulate = MagicMock(return_value=mock_simulation_result)

        mock_repo = MagicMock()
        mock_repo.save_simulation = AsyncMock()

        with (
            patch("src.digital_twin.twin_generator.TwinGenerator", return_value=mock_generator),
            patch("src.digital_twin.simulation_engine.SimulationEngine", return_value=mock_engine),
            patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo),
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

    def test_run_simulation_with_population_filters(self, simulate_request, mock_simulation_result):
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

        with (
            patch("src.digital_twin.twin_generator.TwinGenerator", return_value=mock_generator),
            patch("src.digital_twin.simulation_engine.SimulationEngine", return_value=mock_engine),
            patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo),
        ):
            response = client.post("/api/digital-twin/simulate", json=simulate_request)

        assert response.status_code == 200

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
        mock_repo.list_simulations = AsyncMock(return_value=[mock_simulation_data])

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
        mock_repo.list_simulations = AsyncMock(return_value=[mock_simulation_data])

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get(
                "/api/digital-twin/simulations",
                params={"brand": "Remibrutinib"},
            )

        assert response.status_code == 200

    def test_list_simulations_pagination(self, mock_simulation_data):
        """Should handle pagination parameters."""
        mock_repo = MagicMock()
        mock_repo.list_simulations = AsyncMock(return_value=[mock_simulation_data])

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
        mock_result = MagicMock()
        mock_result.simulation_id = UUID("550e8400-e29b-41d4-a716-446655440000")
        mock_result.model_id = UUID("660e8400-e29b-41d4-a716-446655440000")
        mock_result.twin_count = 10000
        mock_result.simulated_ate = 0.085
        mock_result.simulated_ci_lower = 0.065
        mock_result.simulated_ci_upper = 0.105
        mock_result.simulated_std_error = 0.010
        mock_result.effect_size_cohens_d = 0.25
        mock_result.statistical_power = 0.85
        mock_result.recommendation = MagicMock(value="deploy")
        mock_result.recommendation_rationale = "Significant effect"
        mock_result.recommended_sample_size = 5000
        mock_result.recommended_duration_weeks = 8
        mock_result.simulation_confidence = 0.92
        mock_result.fidelity_warning = False
        mock_result.fidelity_warning_reason = None
        mock_result.model_fidelity_score = 0.88
        mock_result.status = MagicMock(value="completed")
        mock_result.error_message = None
        mock_result.execution_time_ms = 1500
        mock_result.created_at = datetime.now(timezone.utc)
        mock_result.completed_at = datetime.now(timezone.utc)
        mock_result.is_significant = MagicMock(return_value=True)
        mock_result.effect_direction = MagicMock(return_value="positive")
        mock_result.population_filters = MagicMock()
        mock_result.population_filters.to_dict = MagicMock(return_value={"deciles": [1, 2, 3]})
        mock_result.intervention_config = MagicMock()
        mock_result.intervention_config.intervention_type = "email_campaign"
        mock_result.intervention_config.model_dump = MagicMock(
            return_value={"intervention_type": "email_campaign"}
        )
        mock_result.intervention_config.extra_params = {"brand": "Remibrutinib", "twin_type": "hcp"}

        mock_effect_het = MagicMock()
        mock_effect_het.by_specialty = {"oncology": {"ate": 0.12}}
        mock_effect_het.by_decile = {"1": {"ate": 0.15}}
        mock_effect_het.by_region = {"northeast": {"ate": 0.10}}
        mock_effect_het.by_adoption_stage = {"early": {"ate": 0.14}}
        mock_effect_het.get_top_segments = MagicMock(
            return_value=[{"segment": "oncology_d1", "ate": 0.18}]
        )
        mock_result.effect_heterogeneity = mock_effect_het

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

        mock_tracker = MagicMock()
        mock_tracker.get_simulation_record = MagicMock(return_value=mock_fidelity_record)
        mock_tracker.validate = MagicMock(return_value=mock_fidelity_record)

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

    def test_get_model_fidelity_success(self, mock_fidelity_record):
        """Should return fidelity history."""
        mock_repo = MagicMock()
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

    def test_get_model_fidelity_validated_only(self, mock_fidelity_record):
        """Should filter to validated records only."""
        mock_repo = MagicMock()
        mock_repo.get_model_fidelity_records = AsyncMock(return_value=[mock_fidelity_record])

        with patch("src.digital_twin.twin_repository.TwinRepository", return_value=mock_repo):
            response = client.get(
                "/api/digital-twin/models/770e8400-e29b-41d4-a716-446655440000/fidelity",
                params={"validated_only": True},
            )

        assert response.status_code == 200


class TestGetFidelityReport:
    """Tests for GET /digital-twin/models/{model_id}/fidelity/report."""

    def test_get_fidelity_report_success(self):
        """Should return fidelity report with trend analysis."""
        mock_repo = MagicMock()

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

    def test_get_fidelity_report_with_lookback(self):
        """Should respect lookback_days parameter."""
        mock_repo = MagicMock()

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

    def test_get_fidelity_report_degrading(self):
        """Should detect degrading model fidelity."""
        mock_repo = MagicMock()

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
