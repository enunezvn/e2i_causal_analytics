"""
Unit tests for experiments API routes.

Tests all endpoints in src/api/routes/experiments.py including:
- Randomization (simple, stratified, block)
- Enrollment and withdrawal
- Interim analysis
- Results analysis (ITT, per-protocol, heterogeneous)
- SRM checks
- Fidelity tracking
- Experiment monitoring
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks, HTTPException

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_randomization_service():
    """Mock RandomizationService."""
    with patch("src.services.randomization.RandomizationService") as mock_svc:
        instance = AsyncMock()
        mock_svc.return_value = instance

        mock_assignment = MagicMock()
        mock_assignment.id = uuid4()
        mock_assignment.experiment_id = uuid4()
        mock_assignment.unit_id = "hcp_001"
        mock_assignment.unit_type = "hcp"
        mock_assignment.variant = "treatment"
        mock_assignment.assigned_at = datetime.now(timezone.utc)
        mock_assignment.randomization_method = "stratified"
        mock_assignment.stratification_key = {"region": "northeast"}
        mock_assignment.block_id = None

        instance.stratified_randomize.return_value = [mock_assignment]
        instance.block_randomize.return_value = [mock_assignment]
        instance.simple_randomize.return_value = [mock_assignment]

        yield instance


@pytest.fixture
def mock_enrollment_service():
    """Mock EnrollmentService."""
    with patch("src.services.enrollment.EnrollmentService") as mock_svc:
        instance = AsyncMock()
        mock_svc.return_value = instance

        mock_enrollment = MagicMock()
        mock_enrollment.id = uuid4()
        mock_enrollment.assignment_id = uuid4()
        mock_enrollment.variant = "treatment"
        mock_enrollment.enrolled_at = datetime.now(timezone.utc)
        mock_enrollment.enrollment_status = "active"
        mock_enrollment.consent_timestamp = datetime.now(timezone.utc)

        instance.enroll_unit.return_value = mock_enrollment

        mock_stats = MagicMock()
        mock_stats.total_enrolled = 100
        mock_stats.active_count = 90
        mock_stats.withdrawn_count = 5
        mock_stats.completed_count = 5
        mock_stats.enrollment_rate_per_day = 10.5
        mock_stats.variant_breakdown = {"control": 50, "treatment": 50}
        mock_stats.enrollment_trend = [{"date": "2024-01-01", "count": 10}]

        instance.get_enrollment_stats.return_value = mock_stats

        yield instance


@pytest.fixture
def mock_interim_analysis_service():
    """Mock InterimAnalysisService."""
    with patch("src.services.interim_analysis.InterimAnalysisService") as mock_svc:
        instance = AsyncMock()
        mock_svc.return_value = instance

        mock_result = MagicMock()
        mock_result.id = uuid4()
        mock_result.analysis_number = 1
        mock_result.performed_at = datetime.now(timezone.utc)
        mock_result.information_fraction = 0.5
        mock_result.alpha_spent = 0.001
        mock_result.adjusted_alpha = 0.024
        mock_result.test_statistic = 2.5
        mock_result.p_value = 0.012
        mock_result.conditional_power = 0.85
        mock_result.decision = "continue"
        mock_result.metrics_snapshot = {"primary_metric": 0.05}

        instance.perform_interim_analysis.return_value = mock_result

        yield instance


@pytest.fixture
def mock_results_analysis_service():
    """Mock ResultsAnalysisService."""
    with patch("src.services.results_analysis.ResultsAnalysisService") as mock_svc:
        instance = AsyncMock()
        mock_svc.return_value = instance

        mock_result = MagicMock()
        mock_result.id = uuid4()
        mock_result.analysis_type = "final"
        mock_result.analysis_method = "itt"
        mock_result.computed_at = datetime.now(timezone.utc)
        mock_result.primary_metric = "trx_lift"
        mock_result.control_mean = 10.5
        mock_result.treatment_mean = 12.8
        mock_result.effect_estimate = 2.3
        mock_result.effect_ci_lower = 1.2
        mock_result.effect_ci_upper = 3.4
        mock_result.p_value = 0.001
        mock_result.sample_size_control = 500
        mock_result.sample_size_treatment = 500
        mock_result.statistical_power = 0.85
        mock_result.is_significant = True
        mock_result.secondary_metrics = {}
        mock_result.segment_results = {}

        mock_per_protocol_result = MagicMock()
        mock_per_protocol_result.id = uuid4()
        mock_per_protocol_result.analysis_type = "final"
        mock_per_protocol_result.analysis_method = "per_protocol"
        mock_per_protocol_result.computed_at = datetime.now(timezone.utc)
        mock_per_protocol_result.primary_metric = "trx_lift"
        mock_per_protocol_result.control_mean = 10.5
        mock_per_protocol_result.treatment_mean = 12.8
        mock_per_protocol_result.effect_estimate = 2.3
        mock_per_protocol_result.effect_ci_lower = 1.2
        mock_per_protocol_result.effect_ci_upper = 3.4
        mock_per_protocol_result.p_value = 0.001
        mock_per_protocol_result.sample_size_control = 500
        mock_per_protocol_result.sample_size_treatment = 500
        mock_per_protocol_result.statistical_power = 0.85
        mock_per_protocol_result.is_significant = True
        mock_per_protocol_result.secondary_metrics = {}
        mock_per_protocol_result.segment_results = {}

        instance.compute_itt_results.return_value = mock_result
        instance.compute_per_protocol_results.return_value = mock_per_protocol_result
        instance.compute_heterogeneous_effects.return_value = {
            "region": {"northeast": {"ate": 2.5}}
        }

        mock_srm = MagicMock()
        mock_srm.id = uuid4()
        mock_srm.checked_at = datetime.now(timezone.utc)
        mock_srm.expected_ratio = {"control": 0.5, "treatment": 0.5}
        mock_srm.actual_counts = {"control": 495, "treatment": 505}
        mock_srm.chi_squared_statistic = 0.1
        mock_srm.p_value = 0.75
        mock_srm.is_srm_detected = False
        mock_srm.investigation_notes = None

        instance.check_sample_ratio_mismatch.return_value = mock_srm

        mock_fidelity = MagicMock()
        mock_fidelity.id = uuid4()
        mock_fidelity.comparison_timestamp = datetime.now(timezone.utc)
        mock_fidelity.predicted_effect = 2.2
        mock_fidelity.actual_effect = 2.3
        mock_fidelity.prediction_error = 0.1
        mock_fidelity.confidence_interval_coverage = True
        mock_fidelity.fidelity_score = 0.95
        mock_fidelity.calibration_adjustment = None

        instance.compare_with_twin_prediction.return_value = mock_fidelity

        yield instance


@pytest.fixture
def mock_ab_experiment_repository():
    """Mock ABExperimentRepository."""
    with patch("src.repositories.ab_experiment.ABExperimentRepository") as mock_repo:
        instance = AsyncMock()
        mock_repo.return_value = instance

        mock_assignment = MagicMock()
        mock_assignment.id = uuid4()
        mock_assignment.unit_id = "hcp_001"
        mock_assignment.unit_type = "hcp"
        mock_assignment.variant = "treatment"
        mock_assignment.assigned_at = datetime.now(timezone.utc)
        mock_assignment.randomization_method = "stratified"

        instance.get_assignments.return_value = [mock_assignment]

        mock_analysis = MagicMock()
        mock_analysis.id = uuid4()
        mock_analysis.analysis_number = 1
        mock_analysis.performed_at = datetime.now(timezone.utc)
        mock_analysis.information_fraction = 0.5
        mock_analysis.p_value = 0.01
        mock_analysis.decision = "continue"

        instance.get_interim_analyses.return_value = [mock_analysis]

        yield instance


@pytest.fixture
def mock_ab_results_repository():
    """Mock ABResultsRepository."""
    with patch("src.repositories.ab_results.ABResultsRepository") as mock_repo:
        instance = AsyncMock()
        mock_repo.return_value = instance

        mock_result = MagicMock()
        mock_result.id = uuid4()
        mock_result.analysis_type = "final"
        mock_result.analysis_method = "itt"
        mock_result.computed_at = datetime.now(timezone.utc)
        mock_result.primary_metric = "trx_lift"
        mock_result.control_mean = 10.5
        mock_result.treatment_mean = 12.8
        mock_result.effect_estimate = 2.3
        mock_result.effect_ci_lower = 1.2
        mock_result.effect_ci_upper = 3.4
        mock_result.p_value = 0.001
        mock_result.sample_size_control = 500
        mock_result.sample_size_treatment = 500
        mock_result.statistical_power = 0.85
        mock_result.is_significant = True
        mock_result.secondary_metrics = {}
        mock_result.segment_results = {}

        instance.get_results.return_value = [mock_result]

        mock_srm = MagicMock()
        mock_srm.id = uuid4()
        mock_srm.checked_at = datetime.now(timezone.utc)
        mock_srm.expected_ratio = {"control": 0.5, "treatment": 0.5}
        mock_srm.actual_counts = {"control": 495, "treatment": 505}
        mock_srm.chi_squared_statistic = 0.1
        mock_srm.p_value = 0.75
        mock_srm.is_srm_detected = False

        instance.get_srm_history.return_value = [mock_srm]

        mock_fidelity = MagicMock()
        mock_fidelity.id = uuid4()
        mock_fidelity.twin_simulation_id = uuid4()
        mock_fidelity.comparison_timestamp = datetime.now(timezone.utc)
        mock_fidelity.predicted_effect = 2.2
        mock_fidelity.actual_effect = 2.3
        mock_fidelity.prediction_error = 0.1
        mock_fidelity.fidelity_score = 0.95

        instance.get_fidelity_comparisons.return_value = [mock_fidelity]

        mock_alert = MagicMock()
        mock_alert.id = uuid4()
        mock_alert.alert_type = "srm_detected"
        mock_alert.severity = "warning"
        mock_alert.message = "SRM detected"
        mock_alert.details = {}
        mock_alert.timestamp = datetime.now(timezone.utc)

        instance.get_experiment_alerts.return_value = [mock_alert]

        yield instance


@pytest.fixture
def mock_experiment_monitor_agent():
    """Mock ExperimentMonitorAgent."""
    with patch("src.agents.experiment_monitor.ExperimentMonitorAgent") as mock_agent:
        instance = AsyncMock()
        mock_agent.return_value = instance

        mock_output = MagicMock()
        mock_output.experiments_checked = 5
        mock_output.healthy_count = 4
        mock_output.warning_count = 1
        mock_output.critical_count = 0
        mock_output.experiments = [
            {
                "experiment_id": str(uuid4()),
                "name": "Test Experiment",
                "health_status": "healthy",
                "total_enrolled": 100,
                "enrollment_rate_per_day": 10.0,
                "current_information_fraction": 0.5,
                "has_srm": False,
                "active_alerts": 0,
            }
        ]
        mock_output.alerts = []
        mock_output.monitor_summary = "All experiments healthy"
        mock_output.recommended_actions = []
        mock_output.check_latency_ms = 150

        instance.run_async.return_value = mock_output

        yield instance


# =============================================================================
# TESTS - Randomization
# =============================================================================


@pytest.mark.asyncio
async def test_randomize_units_stratified(mock_randomization_service):
    """Test stratified randomization."""
    from src.api.routes.experiments import RandomizationMethod, RandomizeRequest, randomize_units

    experiment_id = str(uuid4())
    request = RandomizeRequest(
        units=[{"unit_id": "hcp_001", "unit_type": "hcp", "region": "northeast"}],
        method=RandomizationMethod.STRATIFIED,
        strata_columns=["region"],
        allocation_ratio={"control": 0.5, "treatment": 0.5},
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await randomize_units(experiment_id, request, user)

    assert result.experiment_id == experiment_id
    assert result.total_units == 1
    assert result.randomization_method == "stratified"
    assert len(result.assignments) == 1


@pytest.mark.asyncio
async def test_randomize_units_block(mock_randomization_service):
    """Test block randomization."""
    from src.api.routes.experiments import RandomizationMethod, RandomizeRequest, randomize_units

    experiment_id = str(uuid4())
    request = RandomizeRequest(
        units=[{"unit_id": "hcp_001", "unit_type": "hcp"}],
        method=RandomizationMethod.BLOCK,
        block_size=4,
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await randomize_units(experiment_id, request, user)

    assert result.randomization_method == "block"


@pytest.mark.asyncio
async def test_randomize_units_simple(mock_randomization_service):
    """Test simple randomization."""
    from src.api.routes.experiments import RandomizationMethod, RandomizeRequest, randomize_units

    experiment_id = str(uuid4())
    request = RandomizeRequest(
        units=[{"unit_id": "hcp_001", "unit_type": "hcp"}],
        method=RandomizationMethod.SIMPLE,
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await randomize_units(experiment_id, request, user)

    assert result.randomization_method == "simple"


@pytest.mark.asyncio
async def test_randomize_units_error(mock_randomization_service):
    """Test randomization with error."""
    from src.api.routes.experiments import RandomizeRequest, randomize_units

    mock_randomization_service.stratified_randomize.side_effect = Exception("Randomization failed")

    experiment_id = str(uuid4())
    request = RandomizeRequest(
        units=[{"unit_id": "hcp_001", "unit_type": "hcp"}],
    )
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await randomize_units(experiment_id, request, user)

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_get_assignments(mock_ab_experiment_repository):
    """Test getting experiment assignments."""
    from src.api.routes.experiments import get_assignments

    experiment_id = str(uuid4())

    result = await get_assignments(experiment_id)

    assert result["experiment_id"] == experiment_id
    assert result["total_count"] == 1
    assert len(result["assignments"]) == 1


@pytest.mark.asyncio
async def test_get_assignments_filtered(mock_ab_experiment_repository):
    """Test getting assignments with filters."""
    from src.api.routes.experiments import get_assignments

    experiment_id = str(uuid4())

    result = await get_assignments(
        experiment_id,
        variant="treatment",
        unit_type="hcp",
        limit=50,
        offset=0,
    )

    assert result["experiment_id"] == experiment_id


# =============================================================================
# TESTS - Enrollment
# =============================================================================


@pytest.mark.asyncio
async def test_enroll_unit_success(mock_enrollment_service):
    """Test enrolling a unit."""
    from src.api.routes.experiments import EnrollUnitRequest, enroll_unit

    experiment_id = str(uuid4())
    request = EnrollUnitRequest(
        unit_id="hcp_001",
        unit_type="hcp",
        consent_timestamp=datetime.now(timezone.utc),
        eligibility_criteria_met={"specialty": True},
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await enroll_unit(experiment_id, request, user)

    assert result.unit_id == "hcp_001"
    assert result.enrollment_status.value == "active"


@pytest.mark.asyncio
async def test_enroll_unit_validation_error(mock_enrollment_service):
    """Test enrollment with validation error."""
    from src.api.routes.experiments import EnrollUnitRequest, enroll_unit

    mock_enrollment_service.enroll_unit.side_effect = ValueError("Unit not assigned")

    experiment_id = str(uuid4())
    request = EnrollUnitRequest(
        unit_id="hcp_001",
        unit_type="hcp",
    )
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await enroll_unit(experiment_id, request, user)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_withdraw_unit_success(mock_enrollment_service):
    """Test withdrawing a unit."""
    from src.api.routes.experiments import WithdrawRequest, withdraw_unit

    experiment_id = str(uuid4())
    enrollment_id = str(uuid4())
    request = WithdrawRequest(reason="Subject withdrew consent")
    user = {"user_id": "test_user", "role": "auth"}

    result = await withdraw_unit(experiment_id, enrollment_id, request, user)

    assert result["status"] == "withdrawn"
    assert result["enrollment_id"] == enrollment_id


@pytest.mark.asyncio
async def test_withdraw_unit_not_found(mock_enrollment_service):
    """Test withdrawing non-existent enrollment."""
    from src.api.routes.experiments import WithdrawRequest, withdraw_unit

    mock_enrollment_service.withdraw_unit.side_effect = ValueError("Not found")

    experiment_id = str(uuid4())
    enrollment_id = str(uuid4())
    request = WithdrawRequest(reason="Test")
    user = {"user_id": "test_user", "role": "auth"}

    with pytest.raises(HTTPException) as exc_info:
        await withdraw_unit(experiment_id, enrollment_id, request, user)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_enrollment_stats(mock_enrollment_service):
    """Test getting enrollment statistics."""
    from src.api.routes.experiments import get_enrollment_stats

    experiment_id = str(uuid4())

    result = await get_enrollment_stats(experiment_id)

    assert result.total_enrolled == 100
    assert result.active_count == 90
    assert result.enrollment_rate_per_day == 10.5


# =============================================================================
# TESTS - Interim Analysis
# =============================================================================


@pytest.mark.asyncio
async def test_trigger_interim_analysis_sync(mock_interim_analysis_service):
    """Test triggering interim analysis synchronously."""
    from fastapi import BackgroundTasks

    from src.api.routes.experiments import TriggerInterimAnalysisRequest, trigger_interim_analysis

    experiment_id = str(uuid4())
    request = TriggerInterimAnalysisRequest(analysis_number=1, force=False)
    background_tasks = BackgroundTasks()
    user = {"user_id": "test_user", "role": "operator"}

    result = await trigger_interim_analysis(
        experiment_id, request, background_tasks, async_mode=False, user=user
    )

    assert result.analysis_number == 1
    assert result.decision.value == "continue"


@pytest.mark.asyncio
async def test_trigger_interim_analysis_async():
    """Test triggering interim analysis asynchronously."""
    from fastapi import BackgroundTasks

    from src.api.routes.experiments import TriggerInterimAnalysisRequest, trigger_interim_analysis

    with patch("src.tasks.ab_testing_tasks.scheduled_interim_analysis") as mock_task:
        mock_celery_result = MagicMock()
        mock_celery_result.id = "task_123"
        mock_task.delay.return_value = mock_celery_result

        experiment_id = str(uuid4())
        request = TriggerInterimAnalysisRequest(analysis_number=1)
        background_tasks = BackgroundTasks()
        user = {"user_id": "test_user", "role": "operator"}

        result = await trigger_interim_analysis(
            experiment_id, request, background_tasks, async_mode=True, user=user
        )

        assert result.metrics_snapshot.get("task_id") == "task_123"


@pytest.mark.asyncio
async def test_trigger_interim_analysis_validation_error(mock_interim_analysis_service):
    """Test interim analysis with validation error."""
    from fastapi import BackgroundTasks

    from src.api.routes.experiments import TriggerInterimAnalysisRequest, trigger_interim_analysis

    mock_interim_analysis_service.perform_interim_analysis.side_effect = ValueError(
        "Milestone not reached"
    )

    experiment_id = str(uuid4())
    request = TriggerInterimAnalysisRequest(force=False)
    background_tasks = BackgroundTasks()
    user = {"user_id": "test_user", "role": "operator"}

    with pytest.raises(HTTPException) as exc_info:
        await trigger_interim_analysis(
            experiment_id, request, background_tasks, async_mode=False, user=user
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_list_interim_analyses(mock_ab_experiment_repository):
    """Test listing interim analyses."""
    from src.api.routes.experiments import list_interim_analyses

    experiment_id = str(uuid4())

    result = await list_interim_analyses(experiment_id)

    assert result["experiment_id"] == experiment_id
    assert result["total_analyses"] == 1


# =============================================================================
# TESTS - Results
# =============================================================================


@pytest.mark.asyncio
async def test_get_experiment_results_recompute(mock_results_analysis_service):
    """Test getting experiment results with recompute."""
    from src.api.routes.experiments import AnalysisMethod, AnalysisType, get_experiment_results

    experiment_id = str(uuid4())

    result = await get_experiment_results(
        experiment_id,
        background_tasks=BackgroundTasks(),
        analysis_type=AnalysisType.FINAL,
        analysis_method=AnalysisMethod.ITT,
        recompute=True,
    )

    assert result.experiment_id == experiment_id
    assert result.analysis_method == AnalysisMethod.ITT
    assert result.is_significant is True


@pytest.mark.asyncio
async def test_get_experiment_results_cached(
    mock_results_analysis_service, mock_ab_results_repository
):
    """Test getting cached experiment results."""
    from src.api.routes.experiments import AnalysisMethod, AnalysisType, get_experiment_results

    experiment_id = str(uuid4())

    result = await get_experiment_results(
        experiment_id,
        background_tasks=BackgroundTasks(),
        analysis_type=AnalysisType.FINAL,
        analysis_method=AnalysisMethod.ITT,
        recompute=False,
    )

    assert result.experiment_id == experiment_id


@pytest.mark.asyncio
async def test_get_experiment_results_per_protocol(mock_results_analysis_service):
    """Test getting per-protocol results."""
    from src.api.routes.experiments import AnalysisMethod, AnalysisType, get_experiment_results

    experiment_id = str(uuid4())

    result = await get_experiment_results(
        experiment_id,
        background_tasks=BackgroundTasks(),
        analysis_type=AnalysisType.FINAL,
        analysis_method=AnalysisMethod.PER_PROTOCOL,
        recompute=True,
    )

    assert result.analysis_method == AnalysisMethod.PER_PROTOCOL


@pytest.mark.asyncio
async def test_get_segment_results(mock_results_analysis_service):
    """Test getting heterogeneous treatment effects by segment."""
    from src.api.routes.experiments import get_segment_results

    experiment_id = str(uuid4())

    result = await get_segment_results(experiment_id, segments=["region", "specialty"])

    assert result["experiment_id"] == experiment_id
    assert "segment_results" in result


# =============================================================================
# TESTS - SRM Checks
# =============================================================================


@pytest.mark.asyncio
async def test_get_srm_checks(mock_ab_results_repository):
    """Test getting SRM check history."""
    from src.api.routes.experiments import get_srm_checks

    experiment_id = str(uuid4())

    result = await get_srm_checks(experiment_id, limit=10)

    assert result["experiment_id"] == experiment_id
    assert result["total_checks"] == 1
    assert result["srm_detected_count"] == 0


@pytest.mark.asyncio
async def test_run_srm_check(mock_results_analysis_service):
    """Test running an SRM check."""
    from src.api.routes.experiments import run_srm_check

    experiment_id = str(uuid4())
    user = {"user_id": "test_user", "role": "auth"}

    result = await run_srm_check(experiment_id, user)

    assert result.experiment_id == experiment_id
    assert result.is_srm_detected is False
    assert result.p_value > 0.05


# =============================================================================
# TESTS - Fidelity
# =============================================================================


@pytest.mark.asyncio
async def test_get_fidelity_comparisons(mock_ab_results_repository):
    """Test getting fidelity comparison history."""
    from src.api.routes.experiments import get_fidelity_comparisons

    experiment_id = str(uuid4())

    result = await get_fidelity_comparisons(experiment_id, limit=10)

    assert result["experiment_id"] == experiment_id
    assert result["total_comparisons"] == 1
    assert "average_fidelity_score" in result


@pytest.mark.asyncio
async def test_update_fidelity_comparison(mock_results_analysis_service):
    """Test updating fidelity comparison."""
    from src.api.routes.experiments import update_fidelity_comparison

    experiment_id = str(uuid4())
    twin_simulation_id = str(uuid4())
    user = {"user_id": "test_user", "role": "auth"}

    result = await update_fidelity_comparison(experiment_id, twin_simulation_id, user)

    assert result.experiment_id == experiment_id
    assert result.twin_simulation_id == twin_simulation_id
    assert result.fidelity_score > 0.9


# =============================================================================
# TESTS - Monitoring
# =============================================================================


@pytest.mark.asyncio
async def test_trigger_experiment_monitoring_sync(mock_experiment_monitor_agent):
    """Test triggering experiment monitoring synchronously."""
    from fastapi import BackgroundTasks

    from src.api.routes.experiments import TriggerMonitorRequest, trigger_experiment_monitoring

    request = TriggerMonitorRequest(
        experiment_ids=None,
        check_srm=True,
        check_enrollment=True,
        check_fidelity=True,
    )
    background_tasks = BackgroundTasks()
    user = {"user_id": "test_user", "role": "auth"}

    result = await trigger_experiment_monitoring(
        request, background_tasks, async_mode=False, user=user
    )

    assert result.experiments_checked == 5
    assert result.healthy_count == 4


@pytest.mark.asyncio
async def test_trigger_experiment_monitoring_async():
    """Test triggering monitoring asynchronously."""
    from fastapi import BackgroundTasks

    from src.api.routes.experiments import TriggerMonitorRequest, trigger_experiment_monitoring

    with patch("src.tasks.ab_testing_tasks.check_all_active_experiments") as mock_task:
        mock_celery_result = MagicMock()
        mock_celery_result.id = "task_123"
        mock_task.delay.return_value = mock_celery_result

        request = TriggerMonitorRequest()
        background_tasks = BackgroundTasks()
        user = {"user_id": "test_user", "role": "auth"}

        result = await trigger_experiment_monitoring(
            request, background_tasks, async_mode=True, user=user
        )

        assert (
            "task_id" in result.monitor_summary.lower()
            or "queued" in result.monitor_summary.lower()
        )


@pytest.mark.asyncio
async def test_trigger_experiment_monitoring_specific_experiments(mock_experiment_monitor_agent):
    """Test monitoring specific experiments."""
    from fastapi import BackgroundTasks

    from src.api.routes.experiments import TriggerMonitorRequest, trigger_experiment_monitoring

    exp_id = str(uuid4())
    request = TriggerMonitorRequest(experiment_ids=[exp_id])
    background_tasks = BackgroundTasks()
    user = {"user_id": "test_user", "role": "auth"}

    result = await trigger_experiment_monitoring(
        request, background_tasks, async_mode=False, user=user
    )

    assert result.experiments_checked > 0


@pytest.mark.asyncio
async def test_get_experiment_health(mock_experiment_monitor_agent):
    """Test getting health status for single experiment."""
    from src.api.routes.experiments import get_experiment_health

    experiment_id = str(uuid4())

    result = await get_experiment_health(experiment_id)

    # Mock returns a valid experiment_id (may differ from input)
    assert result.experiment_id is not None
    assert isinstance(result.experiment_id, str)
    assert result.health_status.value in ["healthy", "warning", "critical"]


@pytest.mark.asyncio
async def test_get_experiment_health_not_found(mock_experiment_monitor_agent):
    """Test getting health for non-existent experiment."""
    from src.api.routes.experiments import get_experiment_health

    mock_experiment_monitor_agent.run_async.return_value.experiments = []

    experiment_id = str(uuid4())

    with pytest.raises(HTTPException) as exc_info:
        await get_experiment_health(experiment_id)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_experiment_alerts(mock_ab_results_repository):
    """Test getting experiment alerts."""
    from src.api.routes.experiments import get_experiment_alerts

    experiment_id = str(uuid4())

    result = await get_experiment_alerts(experiment_id, severity=None, limit=50)

    assert result["experiment_id"] == experiment_id
    assert result["total_alerts"] == 1


@pytest.mark.asyncio
async def test_get_experiment_alerts_filtered(mock_ab_results_repository):
    """Test getting alerts with severity filter."""
    from src.api.routes.experiments import AlertSeverity, get_experiment_alerts

    experiment_id = str(uuid4())

    result = await get_experiment_alerts(experiment_id, severity=AlertSeverity.WARNING, limit=50)

    assert result["experiment_id"] == experiment_id


# =============================================================================
# TESTS - Edge Cases
# =============================================================================


@pytest.mark.asyncio
async def test_randomize_units_multiple_units(mock_randomization_service):
    """Test randomizing multiple units."""
    from src.api.routes.experiments import RandomizeRequest, randomize_units

    # Create multiple mock assignments
    assignments = []
    for i in range(5):
        mock_assignment = MagicMock()
        mock_assignment.id = uuid4()
        mock_assignment.experiment_id = uuid4()
        mock_assignment.unit_id = f"hcp_{i:03d}"
        mock_assignment.unit_type = "hcp"
        mock_assignment.variant = "treatment" if i % 2 == 0 else "control"
        mock_assignment.assigned_at = datetime.now(timezone.utc)
        mock_assignment.randomization_method = "stratified"
        mock_assignment.stratification_key = None
        mock_assignment.block_id = None
        assignments.append(mock_assignment)

    mock_randomization_service.stratified_randomize.return_value = assignments

    experiment_id = str(uuid4())
    request = RandomizeRequest(
        units=[{"unit_id": f"hcp_{i:03d}", "unit_type": "hcp"} for i in range(5)],
    )
    user = {"user_id": "test_user", "role": "operator"}

    result = await randomize_units(experiment_id, request, user)

    assert result.total_units == 5
    assert len(result.assignments) == 5


@pytest.mark.asyncio
async def test_get_assignments_pagination(mock_ab_experiment_repository):
    """Test assignments pagination."""
    from src.api.routes.experiments import get_assignments

    experiment_id = str(uuid4())

    result = await get_assignments(experiment_id, limit=10, offset=20)

    assert result["experiment_id"] == experiment_id


@pytest.mark.asyncio
async def test_enrollment_stats_empty_experiment(mock_enrollment_service):
    """Test enrollment stats for experiment with no enrollments."""
    from src.api.routes.experiments import get_enrollment_stats

    mock_stats = MagicMock()
    mock_stats.total_enrolled = 0
    mock_stats.active_count = 0
    mock_stats.withdrawn_count = 0
    mock_stats.completed_count = 0
    mock_stats.enrollment_rate_per_day = 0.0
    mock_stats.variant_breakdown = {}
    mock_stats.enrollment_trend = []

    mock_enrollment_service.get_enrollment_stats.return_value = mock_stats

    experiment_id = str(uuid4())

    result = await get_enrollment_stats(experiment_id)

    assert result.total_enrolled == 0


@pytest.mark.asyncio
async def test_results_no_cached_results(mock_results_analysis_service, mock_ab_results_repository):
    """Test getting results when no cached results exist."""
    from src.api.routes.experiments import AnalysisMethod, AnalysisType, get_experiment_results

    mock_ab_results_repository.get_results.return_value = []

    experiment_id = str(uuid4())

    result = await get_experiment_results(
        experiment_id,
        analysis_type=AnalysisType.FINAL,
        analysis_method=AnalysisMethod.ITT,
        recompute=False,
    )

    # Should compute fresh results
    assert result.experiment_id == experiment_id


@pytest.mark.asyncio
async def test_fidelity_comparisons_empty(mock_ab_results_repository):
    """Test fidelity comparisons when none exist."""
    from src.api.routes.experiments import get_fidelity_comparisons

    mock_ab_results_repository.get_fidelity_comparisons.return_value = []

    experiment_id = str(uuid4())

    result = await get_fidelity_comparisons(experiment_id)

    assert result["total_comparisons"] == 0
    assert result["average_fidelity_score"] == 0.0
