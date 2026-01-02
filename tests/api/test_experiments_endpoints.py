"""
Tests for Experiments API endpoints.

Phase 1B of API Audit - A/B Testing & Experiment Execution API
Tests organized by batch as per api-endpoints-audit-plan.md

Endpoints covered:
- Batch 1B.1: Randomization (POST /randomize, GET /assignments, POST /enroll)
- Batch 1B.2: Enrollment (GET /enrollments, DELETE /enrollments/{id})
- Batch 1B.3: Analysis (POST /interim-analysis, GET /results, GET /srm-checks, GET /fidelity)
- Batch 1B.4: Monitoring (POST /monitor, GET /health, GET /alerts)
- Additional: interim-analyses, results/segments, srm-check, fidelity update
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
def sample_assignment():
    """Sample assignment mock object."""
    assignment = MagicMock()
    assignment.id = UUID("00000000-0000-0000-0000-000000000001")
    assignment.experiment_id = UUID("11111111-1111-1111-1111-111111111111")
    assignment.unit_id = "hcp_001"
    assignment.unit_type = "hcp"
    assignment.variant = "treatment"
    assignment.assigned_at = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    assignment.randomization_method = "stratified"
    assignment.stratification_key = {"region": "northeast"}
    assignment.block_id = None
    return assignment


@pytest.fixture
def sample_enrollment():
    """Sample enrollment mock object."""
    enrollment = MagicMock()
    enrollment.id = UUID("22222222-2222-2222-2222-222222222222")
    enrollment.assignment_id = UUID("00000000-0000-0000-0000-000000000001")
    enrollment.experiment_id = UUID("11111111-1111-1111-1111-111111111111")
    enrollment.unit_id = "hcp_001"
    enrollment.variant = "treatment"
    enrollment.enrolled_at = datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
    enrollment.enrollment_status = "active"
    enrollment.consent_timestamp = datetime(2025, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
    return enrollment


@pytest.fixture
def sample_enrollment_stats():
    """Sample enrollment stats mock object."""
    stats = MagicMock()
    stats.total_enrolled = 100
    stats.active_count = 85
    stats.withdrawn_count = 10
    stats.completed_count = 5
    stats.enrollment_rate_per_day = 3.5
    stats.variant_breakdown = {"control": 50, "treatment": 50}
    stats.enrollment_trend = [
        {"date": "2025-01-01", "count": 10},
        {"date": "2025-01-02", "count": 15},
    ]
    return stats


@pytest.fixture
def sample_interim_analysis():
    """Sample interim analysis mock object."""
    analysis = MagicMock()
    analysis.id = UUID("33333333-3333-3333-3333-333333333333")
    analysis.experiment_id = UUID("11111111-1111-1111-1111-111111111111")
    analysis.analysis_number = 2
    analysis.performed_at = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    analysis.information_fraction = 0.65
    analysis.alpha_spent = 0.015
    analysis.adjusted_alpha = 0.025
    analysis.test_statistic = 2.1
    analysis.p_value = 0.036
    analysis.conditional_power = 0.82
    analysis.decision = "continue"
    analysis.metrics_snapshot = {"primary_metric": 0.12}
    return analysis


@pytest.fixture
def sample_experiment_result():
    """Sample experiment result mock object."""
    result = MagicMock()
    result.id = UUID("44444444-4444-4444-4444-444444444444")
    result.experiment_id = UUID("11111111-1111-1111-1111-111111111111")
    result.analysis_type = "final"
    result.analysis_method = "itt"
    result.computed_at = datetime(2025, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
    result.primary_metric = "conversion_rate"
    result.control_mean = 0.10
    result.treatment_mean = 0.15
    result.effect_estimate = 0.05
    result.effect_ci_lower = 0.02
    result.effect_ci_upper = 0.08
    result.p_value = 0.003
    result.sample_size_control = 500
    result.sample_size_treatment = 500
    result.statistical_power = 0.92
    result.is_significant = True
    result.secondary_metrics = {"visits": 0.03}
    result.segment_results = {"northeast": {"effect": 0.06}}
    return result


@pytest.fixture
def sample_srm_check():
    """Sample SRM check mock object."""
    check = MagicMock()
    check.id = UUID("55555555-5555-5555-5555-555555555555")
    check.experiment_id = UUID("11111111-1111-1111-1111-111111111111")
    check.checked_at = datetime(2025, 1, 10, 8, 0, 0, tzinfo=timezone.utc)
    check.expected_ratio = {"control": 0.5, "treatment": 0.5}
    check.actual_counts = {"control": 480, "treatment": 520}
    check.chi_squared_statistic = 1.6
    check.p_value = 0.206
    check.is_srm_detected = False
    check.investigation_notes = None
    return check


@pytest.fixture
def sample_fidelity_comparison():
    """Sample fidelity comparison mock object."""
    comparison = MagicMock()
    comparison.id = UUID("66666666-6666-6666-6666-666666666666")
    comparison.experiment_id = UUID("11111111-1111-1111-1111-111111111111")
    comparison.twin_simulation_id = UUID("77777777-7777-7777-7777-777777777777")
    comparison.comparison_timestamp = datetime(2025, 1, 20, 14, 0, 0, tzinfo=timezone.utc)
    comparison.predicted_effect = 0.045
    comparison.actual_effect = 0.050
    comparison.prediction_error = 0.005
    comparison.confidence_interval_coverage = True
    comparison.fidelity_score = 0.89
    comparison.calibration_adjustment = None
    return comparison


@pytest.fixture
def sample_celery_task():
    """Mock Celery task for async operations."""
    task = MagicMock()
    task.id = "celery_task_12345"
    task.delay = MagicMock(return_value=task)
    return task


@pytest.fixture
def sample_monitor_result():
    """Sample experiment monitor result."""
    result = MagicMock()
    result.experiments_checked = 3
    result.healthy_count = 2
    result.warning_count = 1
    result.critical_count = 0
    result.experiments = [
        {
            "experiment_id": "11111111-1111-1111-1111-111111111111",
            "name": "Q1 Campaign Test",
            "health_status": "healthy",
            "total_enrolled": 500,
            "enrollment_rate_per_day": 15.0,
            "current_information_fraction": 0.75,
            "has_srm": False,
            "active_alerts": 0,
        },
    ]
    result.alerts = [
        {
            "alert_id": "alert_001",
            "alert_type": "enrollment_slow",
            "severity": "warning",
            "experiment_id": "22222222-2222-2222-2222-222222222222",
            "experiment_name": "Regional Test",
            "message": "Enrollment rate below target",
            "details": {"current_rate": 2.5, "target_rate": 5.0},
            "recommended_action": "Review recruitment strategy",
            "timestamp": "2025-01-02T10:00:00Z",
        }
    ]
    result.monitor_summary = "2 healthy, 1 warning, 0 critical"
    result.recommended_actions = ["Review Regional Test enrollment"]
    result.check_latency_ms = 150
    return result


@pytest.fixture
def sample_alert():
    """Sample experiment alert mock object."""
    alert = MagicMock()
    alert.id = UUID("88888888-8888-8888-8888-888888888888")
    alert.experiment_id = UUID("11111111-1111-1111-1111-111111111111")
    alert.alert_type = "srm_detected"
    alert.severity = "critical"
    alert.message = "Sample Ratio Mismatch detected"
    alert.details = {"chi_squared": 12.5, "p_value": 0.0004}
    alert.timestamp = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
    return alert


# =============================================================================
# BATCH 1B.1 - RANDOMIZATION TESTS
# =============================================================================


class TestRandomizeUnits:
    """Tests for POST /experiments/{experiment_id}/randomize."""

    def test_stratified_randomization_success(self, sample_assignment):
        """Should randomize units with stratified method."""
        mock_service = MagicMock()
        mock_service.stratified_randomize = AsyncMock(return_value=[sample_assignment])

        with patch(
            "src.services.randomization.RandomizationService",
            return_value=mock_service,
        ):
            response = client.post(
                "/experiments/11111111-1111-1111-1111-111111111111/randomize",
                json={
                    "units": [
                        {"unit_id": "hcp_001", "unit_type": "hcp", "region": "northeast"},
                    ],
                    "method": "stratified",
                    "strata_columns": ["region"],
                    "allocation_ratio": {"control": 0.5, "treatment": 0.5},
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == "11111111-1111-1111-1111-111111111111"
        assert data["total_units"] == 1
        assert data["randomization_method"] == "stratified"
        assert "assignments" in data
        assert len(data["assignments"]) == 1

    def test_simple_randomization_success(self, sample_assignment):
        """Should randomize units with simple method."""
        sample_assignment.randomization_method = "simple"
        mock_service = MagicMock()
        mock_service.simple_randomize = AsyncMock(return_value=[sample_assignment])

        with patch(
            "src.services.randomization.RandomizationService",
            return_value=mock_service,
        ):
            response = client.post(
                "/experiments/11111111-1111-1111-1111-111111111111/randomize",
                json={
                    "units": [
                        {"unit_id": "hcp_001", "unit_type": "hcp"},
                    ],
                    "method": "simple",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["randomization_method"] == "simple"

    def test_block_randomization_success(self, sample_assignment):
        """Should randomize units with block method."""
        sample_assignment.randomization_method = "block"
        sample_assignment.block_id = "block_001"
        mock_service = MagicMock()
        mock_service.block_randomize = AsyncMock(return_value=[sample_assignment])

        with patch(
            "src.services.randomization.RandomizationService",
            return_value=mock_service,
        ):
            response = client.post(
                "/experiments/11111111-1111-1111-1111-111111111111/randomize",
                json={
                    "units": [
                        {"unit_id": "hcp_001", "unit_type": "hcp"},
                    ],
                    "method": "block",
                    "block_size": 4,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["randomization_method"] == "block"

    def test_randomization_service_error(self):
        """Should return 500 on service error."""
        mock_service = MagicMock()
        mock_service.stratified_randomize = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        with patch(
            "src.services.randomization.RandomizationService",
            return_value=mock_service,
        ):
            response = client.post(
                "/experiments/11111111-1111-1111-1111-111111111111/randomize",
                json={
                    "units": [{"unit_id": "hcp_001", "unit_type": "hcp"}],
                    "method": "stratified",
                },
            )

        assert response.status_code == 500


class TestGetAssignments:
    """Tests for GET /experiments/{experiment_id}/assignments."""

    def test_get_assignments_success(self, sample_assignment):
        """Should return experiment assignments."""
        mock_repo = MagicMock()
        mock_repo.get_assignments = AsyncMock(return_value=[sample_assignment])

        with patch(
            "src.repositories.ab_experiment.ABExperimentRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/assignments"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"] == "11111111-1111-1111-1111-111111111111"
        assert data["total_count"] == 1
        assert len(data["assignments"]) == 1
        assert data["assignments"][0]["unit_id"] == "hcp_001"

    def test_get_assignments_with_filters(self, sample_assignment):
        """Should filter assignments by variant and unit_type."""
        mock_repo = MagicMock()
        mock_repo.get_assignments = AsyncMock(return_value=[sample_assignment])

        with patch(
            "src.repositories.ab_experiment.ABExperimentRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/assignments",
                params={"variant": "treatment", "unit_type": "hcp", "limit": 50},
            )

        assert response.status_code == 200
        mock_repo.get_assignments.assert_called_once()

    def test_get_assignments_empty(self):
        """Should return empty list when no assignments exist."""
        mock_repo = MagicMock()
        mock_repo.get_assignments = AsyncMock(return_value=[])

        with patch(
            "src.repositories.ab_experiment.ABExperimentRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/assignments"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0


# =============================================================================
# BATCH 1B.2 - ENROLLMENT TESTS
# =============================================================================


class TestEnrollUnit:
    """Tests for POST /experiments/{experiment_id}/enroll."""

    def test_enroll_unit_success(self, sample_enrollment):
        """Should enroll a unit in the experiment."""
        mock_service = MagicMock()
        mock_service.enroll_unit = AsyncMock(return_value=sample_enrollment)

        with patch(
            "src.services.enrollment.EnrollmentService",
            return_value=mock_service,
        ):
            response = client.post(
                "/experiments/11111111-1111-1111-1111-111111111111/enroll",
                json={
                    "unit_id": "hcp_001",
                    "unit_type": "hcp",
                    "consent_timestamp": "2025-01-01T10:30:00Z",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["unit_id"] == "hcp_001"
        assert data["variant"] == "treatment"
        assert data["enrollment_status"] == "active"

    def test_enroll_unit_not_assigned(self):
        """Should return 400 when unit is not assigned to experiment."""
        mock_service = MagicMock()
        mock_service.enroll_unit = AsyncMock(
            side_effect=ValueError("Unit not assigned to experiment")
        )

        with patch(
            "src.services.enrollment.EnrollmentService",
            return_value=mock_service,
        ):
            response = client.post(
                "/experiments/11111111-1111-1111-1111-111111111111/enroll",
                json={
                    "unit_id": "hcp_999",
                    "unit_type": "hcp",
                },
            )

        assert response.status_code == 400

    def test_enroll_unit_service_error(self):
        """Should return 500 on service error."""
        mock_service = MagicMock()
        mock_service.enroll_unit = AsyncMock(side_effect=Exception("Database error"))

        with patch(
            "src.services.enrollment.EnrollmentService",
            return_value=mock_service,
        ):
            response = client.post(
                "/experiments/11111111-1111-1111-1111-111111111111/enroll",
                json={
                    "unit_id": "hcp_001",
                    "unit_type": "hcp",
                },
            )

        assert response.status_code == 500


class TestWithdrawUnit:
    """Tests for DELETE /experiments/{experiment_id}/enrollments/{enrollment_id}."""

    def test_withdraw_unit_success(self):
        """Should withdraw a unit from the experiment."""
        mock_service = MagicMock()
        mock_service.withdraw_unit = AsyncMock(return_value=None)

        with patch(
            "src.services.enrollment.EnrollmentService",
            return_value=mock_service,
        ):
            response = client.request(
                "DELETE",
                "/experiments/11111111-1111-1111-1111-111111111111/enrollments/22222222-2222-2222-2222-222222222222",
                json={"reason": "Subject requested withdrawal"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "withdrawn"
        assert data["reason"] == "Subject requested withdrawal"

    def test_withdraw_unit_not_found(self):
        """Should return 404 when enrollment not found."""
        mock_service = MagicMock()
        mock_service.withdraw_unit = AsyncMock(
            side_effect=ValueError("Enrollment not found")
        )

        with patch(
            "src.services.enrollment.EnrollmentService",
            return_value=mock_service,
        ):
            response = client.request(
                "DELETE",
                "/experiments/11111111-1111-1111-1111-111111111111/enrollments/99999999-9999-9999-9999-999999999999",
                json={"reason": "Test withdrawal"},
            )

        assert response.status_code == 404


class TestGetEnrollmentStats:
    """Tests for GET /experiments/{experiment_id}/enrollments."""

    def test_get_enrollment_stats_success(self, sample_enrollment_stats):
        """Should return enrollment statistics."""
        mock_service = MagicMock()
        mock_service.get_enrollment_stats = AsyncMock(return_value=sample_enrollment_stats)

        with patch(
            "src.services.enrollment.EnrollmentService",
            return_value=mock_service,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/enrollments"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_enrolled"] == 100
        assert data["active_count"] == 85
        assert data["enrollment_rate_per_day"] == 3.5
        assert "variant_breakdown" in data

    def test_get_enrollment_stats_error(self):
        """Should return 500 on service error."""
        mock_service = MagicMock()
        mock_service.get_enrollment_stats = AsyncMock(side_effect=Exception("Error"))

        with patch(
            "src.services.enrollment.EnrollmentService",
            return_value=mock_service,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/enrollments"
            )

        assert response.status_code == 500


# =============================================================================
# BATCH 1B.3 - ANALYSIS TESTS
# =============================================================================


class TestTriggerInterimAnalysis:
    """Tests for POST /experiments/{experiment_id}/interim-analysis."""

    def test_interim_analysis_sync_success(self, sample_interim_analysis):
        """Should perform interim analysis synchronously."""
        mock_service = MagicMock()
        mock_service.perform_interim_analysis = AsyncMock(
            return_value=sample_interim_analysis
        )

        with patch(
            "src.services.interim_analysis.InterimAnalysisService",
            return_value=mock_service,
        ):
            response = client.post(
                "/experiments/11111111-1111-1111-1111-111111111111/interim-analysis",
                json={"analysis_number": 2, "force": False},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["analysis_number"] == 2
        assert data["information_fraction"] == 0.65
        assert data["decision"] == "continue"

    def test_interim_analysis_async_mode(self, sample_celery_task):
        """Should queue interim analysis in async mode."""
        with patch(
            "src.tasks.ab_testing_tasks.scheduled_interim_analysis",
            sample_celery_task,
        ):
            response = client.post(
                "/experiments/11111111-1111-1111-1111-111111111111/interim-analysis",
                params={"async_mode": "true"},
                json={"force": False},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["metrics_snapshot"]["status"] == "queued"
        assert "task_id" in data["metrics_snapshot"]

    def test_interim_analysis_validation_error(self):
        """Should return 400 on validation error."""
        mock_service = MagicMock()
        mock_service.perform_interim_analysis = AsyncMock(
            side_effect=ValueError("Milestone not reached")
        )

        with patch(
            "src.services.interim_analysis.InterimAnalysisService",
            return_value=mock_service,
        ):
            response = client.post(
                "/experiments/11111111-1111-1111-1111-111111111111/interim-analysis",
                json={"analysis_number": 5, "force": False},
            )

        assert response.status_code == 400


class TestListInterimAnalyses:
    """Tests for GET /experiments/{experiment_id}/interim-analyses."""

    def test_list_interim_analyses_success(self, sample_interim_analysis):
        """Should list all interim analyses for experiment."""
        mock_repo = MagicMock()
        mock_repo.get_interim_analyses = AsyncMock(return_value=[sample_interim_analysis])

        with patch(
            "src.repositories.ab_experiment.ABExperimentRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/interim-analyses"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_analyses"] == 1
        assert len(data["analyses"]) == 1


class TestGetExperimentResults:
    """Tests for GET /experiments/{experiment_id}/results."""

    def test_get_results_cached(self, sample_experiment_result):
        """Should return cached experiment results."""
        mock_repo = MagicMock()
        mock_repo.get_results = AsyncMock(return_value=[sample_experiment_result])

        with patch(
            "src.repositories.ab_results.ABResultsRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/results"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["primary_metric"] == "conversion_rate"
        assert data["effect_estimate"] == 0.05
        assert data["is_significant"] is True

    def test_get_results_recompute(self, sample_experiment_result):
        """Should recompute results when requested."""
        mock_service = MagicMock()
        mock_service.compute_itt_results = AsyncMock(return_value=sample_experiment_result)

        with patch(
            "src.services.results_analysis.ResultsAnalysisService",
            return_value=mock_service,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/results",
                params={"recompute": "true"},
            )

        assert response.status_code == 200
        mock_service.compute_itt_results.assert_called_once()


class TestGetSegmentResults:
    """Tests for GET /experiments/{experiment_id}/results/segments."""

    def test_get_segment_results_success(self):
        """Should return heterogeneous treatment effects by segment."""
        hte_results = {
            "region": {"northeast": {"effect": 0.06}, "southwest": {"effect": 0.04}},
        }
        mock_service = MagicMock()
        mock_service.compute_heterogeneous_effects = AsyncMock(return_value=hte_results)

        with patch(
            "src.services.results_analysis.ResultsAnalysisService",
            return_value=mock_service,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/results/segments",
                params={"segments": ["region"]},
            )

        assert response.status_code == 200
        data = response.json()
        assert "segment_results" in data
        assert "region" in data["segment_results"]


class TestGetSRMChecks:
    """Tests for GET /experiments/{experiment_id}/srm-checks."""

    def test_get_srm_checks_success(self, sample_srm_check):
        """Should return SRM check history."""
        mock_repo = MagicMock()
        mock_repo.get_srm_history = AsyncMock(return_value=[sample_srm_check])

        with patch(
            "src.repositories.ab_results.ABResultsRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/srm-checks"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_checks"] == 1
        assert data["srm_detected_count"] == 0
        assert len(data["checks"]) == 1

    def test_get_srm_checks_with_detection(self, sample_srm_check):
        """Should count SRM detections correctly."""
        sample_srm_check.is_srm_detected = True
        mock_repo = MagicMock()
        mock_repo.get_srm_history = AsyncMock(return_value=[sample_srm_check])

        with patch(
            "src.repositories.ab_results.ABResultsRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/srm-checks"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["srm_detected_count"] == 1


class TestRunSRMCheck:
    """Tests for POST /experiments/{experiment_id}/srm-check."""

    def test_run_srm_check_success(self, sample_srm_check):
        """Should run SRM check and return result."""
        mock_service = MagicMock()
        mock_service.check_sample_ratio_mismatch = AsyncMock(return_value=sample_srm_check)

        with patch(
            "src.services.results_analysis.ResultsAnalysisService",
            return_value=mock_service,
        ):
            response = client.post(
                "/experiments/11111111-1111-1111-1111-111111111111/srm-check"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["is_srm_detected"] is False
        assert data["p_value"] == 0.206


class TestGetFidelityComparisons:
    """Tests for GET /experiments/{experiment_id}/fidelity."""

    def test_get_fidelity_comparisons_success(self, sample_fidelity_comparison):
        """Should return Digital Twin fidelity comparisons."""
        mock_repo = MagicMock()
        mock_repo.get_fidelity_comparisons = AsyncMock(
            return_value=[sample_fidelity_comparison]
        )

        with patch(
            "src.repositories.ab_results.ABResultsRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/fidelity"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_comparisons"] == 1
        assert data["average_fidelity_score"] == 0.89
        assert len(data["comparisons"]) == 1

    def test_get_fidelity_comparisons_empty(self):
        """Should return zero average when no comparisons exist."""
        mock_repo = MagicMock()
        mock_repo.get_fidelity_comparisons = AsyncMock(return_value=[])

        with patch(
            "src.repositories.ab_results.ABResultsRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/fidelity"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["average_fidelity_score"] == 0.0


class TestUpdateFidelityComparison:
    """Tests for POST /experiments/{experiment_id}/fidelity/{twin_simulation_id}."""

    def test_update_fidelity_comparison_success(self, sample_fidelity_comparison):
        """Should update fidelity comparison with latest results."""
        mock_service = MagicMock()
        mock_service.compare_with_twin_prediction = AsyncMock(
            return_value=sample_fidelity_comparison
        )

        with patch(
            "src.services.results_analysis.ResultsAnalysisService",
            return_value=mock_service,
        ):
            response = client.post(
                "/experiments/11111111-1111-1111-1111-111111111111/fidelity/77777777-7777-7777-7777-777777777777"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["fidelity_score"] == 0.89
        assert data["prediction_error"] == 0.005


# =============================================================================
# BATCH 1B.4 - MONITORING TESTS
# =============================================================================


class TestTriggerExperimentMonitoring:
    """Tests for POST /experiments/monitor."""

    def test_monitor_sync_success(self, sample_monitor_result):
        """Should run experiment monitoring synchronously."""
        mock_agent = MagicMock()
        mock_agent.run_async = AsyncMock(return_value=sample_monitor_result)

        with patch(
            "src.agents.experiment_monitor.ExperimentMonitorAgent",
            return_value=mock_agent,
        ):
            response = client.post(
                "/experiments/monitor",
                json={
                    "check_srm": True,
                    "check_enrollment": True,
                    "check_fidelity": True,
                    "srm_threshold": 0.001,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["experiments_checked"] == 3
        assert data["healthy_count"] == 2
        assert data["warning_count"] == 1
        assert len(data["alerts"]) == 1

    def test_monitor_async_mode(self, sample_celery_task):
        """Should queue monitoring in async mode."""
        with patch(
            "src.tasks.ab_testing_tasks.check_all_active_experiments",
            sample_celery_task,
        ):
            response = client.post(
                "/experiments/monitor",
                params={"async_mode": "true"},
                json={"check_srm": True},
            )

        assert response.status_code == 200
        data = response.json()
        assert "Monitoring task queued" in data["monitor_summary"]

    def test_monitor_specific_experiments(self, sample_monitor_result):
        """Should monitor specific experiments when IDs provided."""
        mock_agent = MagicMock()
        mock_agent.run_async = AsyncMock(return_value=sample_monitor_result)

        with patch(
            "src.agents.experiment_monitor.ExperimentMonitorAgent",
            return_value=mock_agent,
        ):
            response = client.post(
                "/experiments/monitor",
                json={
                    "experiment_ids": ["11111111-1111-1111-1111-111111111111"],
                    "check_srm": True,
                },
            )

        assert response.status_code == 200


class TestGetExperimentHealth:
    """Tests for GET /experiments/{experiment_id}/health."""

    def test_get_health_success(self, sample_monitor_result):
        """Should return experiment health status."""
        mock_agent = MagicMock()
        mock_agent.run_async = AsyncMock(return_value=sample_monitor_result)

        with patch(
            "src.agents.experiment_monitor.ExperimentMonitorAgent",
            return_value=mock_agent,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/health"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["health_status"] == "healthy"
        assert data["total_enrolled"] == 500

    def test_get_health_not_found(self):
        """Should return 404 when experiment not found."""
        mock_result = MagicMock()
        mock_result.experiments = []
        mock_result.alerts = []

        mock_agent = MagicMock()
        mock_agent.run_async = AsyncMock(return_value=mock_result)

        with patch(
            "src.agents.experiment_monitor.ExperimentMonitorAgent",
            return_value=mock_agent,
        ):
            response = client.get(
                "/experiments/99999999-9999-9999-9999-999999999999/health"
            )

        assert response.status_code == 404


class TestGetExperimentAlerts:
    """Tests for GET /experiments/{experiment_id}/alerts."""

    def test_get_alerts_success(self, sample_alert):
        """Should return experiment alerts."""
        mock_repo = MagicMock()
        mock_repo.get_experiment_alerts = AsyncMock(return_value=[sample_alert])

        with patch(
            "src.repositories.ab_results.ABResultsRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/alerts"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_alerts"] == 1
        assert data["critical_count"] == 1
        assert data["warning_count"] == 0

    def test_get_alerts_with_severity_filter(self, sample_alert):
        """Should filter alerts by severity."""
        mock_repo = MagicMock()
        mock_repo.get_experiment_alerts = AsyncMock(return_value=[sample_alert])

        with patch(
            "src.repositories.ab_results.ABResultsRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/alerts",
                params={"severity": "critical"},
            )

        assert response.status_code == 200
        mock_repo.get_experiment_alerts.assert_called_once()

    def test_get_alerts_empty(self):
        """Should return empty list when no alerts exist."""
        mock_repo = MagicMock()
        mock_repo.get_experiment_alerts = AsyncMock(return_value=[])

        with patch(
            "src.repositories.ab_results.ABResultsRepository",
            return_value=mock_repo,
        ):
            response = client.get(
                "/experiments/11111111-1111-1111-1111-111111111111/alerts"
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total_alerts"] == 0
        assert data["critical_count"] == 0
