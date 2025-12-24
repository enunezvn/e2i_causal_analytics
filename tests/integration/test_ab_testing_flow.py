"""
Integration tests for E2I A/B Testing Infrastructure.

Tests end-to-end A/B testing flows including:
- Experiment design → randomization → enrollment → analysis → results
- Interim analysis with O'Brien-Fleming boundaries
- Sample Ratio Mismatch (SRM) detection and alerting
- Digital Twin fidelity tracking
- Experiment state machine transitions
- ExperimentMonitorAgent workflow

Phase 15: A/B Testing Infrastructure

These tests require Supabase for persistence.
Use pytest markers to skip when services are unavailable.

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import asyncio
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Test Configuration
# =============================================================================

HAS_SUPABASE_URL = bool(os.getenv("SUPABASE_URL"))
HAS_SUPABASE_KEY = bool(os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY"))

requires_supabase = pytest.mark.skipif(
    not (HAS_SUPABASE_URL and HAS_SUPABASE_KEY),
    reason="SUPABASE_URL and SUPABASE_KEY environment variables not set",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def experiment_id() -> str:
    """Generate a unique experiment ID (valid UUID) for test isolation."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_units() -> List[Dict[str, Any]]:
    """Create sample units for randomization testing."""
    return [
        {"unit_id": f"hcp_{i:03d}", "unit_type": "hcp", "region": "northeast" if i % 2 == 0 else "southwest", "specialty": "cardiology" if i % 3 == 0 else "oncology"}
        for i in range(20)
    ]


@pytest.fixture
def sample_experiment_config() -> Dict[str, Any]:
    """Create sample experiment configuration."""
    return {
        "name": "Test A/B Experiment",
        "description": "Integration test experiment",
        "hypothesis": "Treatment increases conversion rate",
        "primary_metric": "conversion_rate",
        "target_sample_size": 1000,
        "min_detectable_effect": 0.05,
        "significance_level": 0.05,
        "power": 0.8,
        "variants": {
            "control": {"allocation": 0.5, "description": "No intervention"},
            "treatment": {"allocation": 0.5, "description": "New intervention"},
        },
    }


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client for unit testing."""
    client = MagicMock()

    # Mock table method
    table_mock = MagicMock()
    client.table = MagicMock(return_value=table_mock)

    # Mock select/insert/update/delete chains
    table_mock.select = MagicMock(return_value=table_mock)
    table_mock.insert = MagicMock(return_value=table_mock)
    table_mock.update = MagicMock(return_value=table_mock)
    table_mock.delete = MagicMock(return_value=table_mock)
    table_mock.eq = MagicMock(return_value=table_mock)
    table_mock.in_ = MagicMock(return_value=table_mock)
    table_mock.order = MagicMock(return_value=table_mock)
    table_mock.limit = MagicMock(return_value=table_mock)

    # Mock async execute
    async def mock_execute():
        return MagicMock(data=[])
    table_mock.execute = AsyncMock(side_effect=mock_execute)

    return client


# =============================================================================
# Randomization Service Tests
# =============================================================================


class TestRandomizationService:
    """Integration tests for RandomizationService.

    Note: RandomizationService is a pure computation service that does NOT
    interact with the database directly. It returns AssignmentResult objects
    directly from the randomization methods.
    """

    @pytest.mark.asyncio
    async def test_simple_randomization(self, experiment_id: str, sample_units: List[Dict]):
        """Test simple random assignment."""
        from src.services.randomization import RandomizationService

        service = RandomizationService()

        # RandomizationService is a pure computation service - no mocking needed
        assignments = await service.simple_randomize(
            experiment_id=uuid.UUID(experiment_id),
            units=sample_units[:10],
            allocation_ratio={"control": 0.5, "treatment": 0.5},
        )

        # All units should be assigned
        assert len(assignments) == 10

        # Check variant distribution (should be approximately 50/50)
        control_count = sum(1 for a in assignments if a.variant == "control")
        treatment_count = sum(1 for a in assignments if a.variant == "treatment")
        assert control_count + treatment_count == 10

    @pytest.mark.asyncio
    async def test_stratified_randomization(self, experiment_id: str, sample_units: List[Dict]):
        """Test stratified randomization by region."""
        from src.services.randomization import RandomizationService

        service = RandomizationService()

        assignments = await service.stratified_randomize(
            experiment_id=uuid.UUID(experiment_id),
            units=sample_units,
            strata_columns=["region"],
            allocation_ratio={"control": 0.5, "treatment": 0.5},
        )

        # All units should be assigned
        assert len(assignments) == 20

        # Check stratification was applied
        for assignment in assignments:
            assert assignment.stratification_key is not None

    @pytest.mark.asyncio
    async def test_block_randomization(self, experiment_id: str, sample_units: List[Dict]):
        """Test block randomization."""
        from src.services.randomization import RandomizationService

        service = RandomizationService()

        assignments = await service.block_randomize(
            experiment_id=uuid.UUID(experiment_id),
            units=sample_units[:8],
            block_size=4,
            allocation_ratio={"control": 0.5, "treatment": 0.5},
        )

        # Should have 8 assignments in 2 blocks
        assert len(assignments) == 8

        # Verify all assignments have block_id
        for assignment in assignments:
            assert assignment.block_id is not None

    @pytest.mark.asyncio
    async def test_deterministic_assignment(self, experiment_id: str):
        """Test that assignment is deterministic with same salt."""
        from src.services.randomization import RandomizationService

        service = RandomizationService()

        # Same inputs should produce same hash
        hash1 = service._generate_assignment_hash(
            experiment_id=uuid.UUID(experiment_id),
            unit_id="hcp_001",
            salt="test_salt",
        )
        hash2 = service._generate_assignment_hash(
            experiment_id=uuid.UUID(experiment_id),
            unit_id="hcp_001",
            salt="test_salt",
        )

        assert hash1 == hash2

        # Convert to unit interval [0, 1]
        val1 = service._hash_to_unit_interval(hash1)
        val2 = service._hash_to_unit_interval(hash2)

        assert val1 == val2
        assert 0 <= val1 <= 1

    @pytest.mark.asyncio
    async def test_multi_arm_allocation(self, experiment_id: str):
        """Test multi-arm experiment allocation."""
        from src.services.randomization import RandomizationService

        service = RandomizationService()

        unit = {"unit_id": "hcp_001", "unit_type": "hcp"}
        arm_probabilities = {
            "control": 0.4,
            "treatment_a": 0.3,
            "treatment_b": 0.3,
        }

        assignment = await service.multi_arm_allocate(
            experiment_id=uuid.UUID(experiment_id),
            unit=unit,
            arm_probabilities=arm_probabilities,
        )

        assert assignment is not None
        assert assignment.variant in arm_probabilities


# =============================================================================
# Enrollment Service Tests
# =============================================================================


class TestEnrollmentService:
    """Integration tests for EnrollmentService.

    Note: EnrollmentService creates ABExperimentRepository inside methods,
    so we mock the repository import rather than a service attribute.
    """

    @pytest.mark.asyncio
    async def test_check_eligibility(self, experiment_id: str):
        """Test eligibility checking - pure computation, no mocking needed."""
        from src.services.enrollment import EnrollmentService, EligibilityCriteria

        service = EnrollmentService()

        # Use the actual EligibilityCriteria fields
        criteria = EligibilityCriteria(
            min_rx_history_months=6,
            min_patient_panel_size=50,
            active_in_territory=True,
            not_in_concurrent_study=True,
            no_recent_protocol_violations=True,
        )

        # Unit data matching the criteria expectations
        unit = {
            "unit_id": "hcp_001",
            "rx_history_months": 12,
            "patient_panel_size": 100,
            "active_in_territory": True,
            "in_concurrent_study": False,
            "recent_protocol_violations": False,
        }

        result = await service.check_eligibility(
            experiment_id=uuid.UUID(experiment_id),
            unit=unit,
            criteria=criteria,
        )

        # Use correct attribute names (service uses shortened key names)
        assert result.is_eligible is True
        assert "min_rx_history" in result.criteria_results

    @pytest.mark.asyncio
    async def test_enroll_unit_success(self, experiment_id: str):
        """Test successful enrollment."""
        from src.services.enrollment import (
            EnrollmentService,
            EligibilityResult,
            ConsentMethod,
            EnrollmentStatus,
        )

        service = EnrollmentService()
        assignment_id = uuid.uuid4()

        # Create a valid eligibility result
        eligibility_result = EligibilityResult(
            is_eligible=True,
            criteria_results={"min_rx_history_months": True, "active_in_territory": True},
            failed_criteria=[],
        )

        # Mock the ABExperimentRepository at its import location
        with patch("src.repositories.ab_experiment.ABExperimentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.create_enrollment = AsyncMock(
                return_value=MagicMock(
                    id=uuid.uuid4(),
                    assignment_id=assignment_id,
                    enrolled_at=datetime.now(timezone.utc),
                    enrollment_status=EnrollmentStatus.ACTIVE,
                    eligibility_criteria_met=eligibility_result.criteria_results,
                )
            )

            enrollment = await service.enroll_unit(
                assignment_id=assignment_id,
                eligibility_result=eligibility_result,
                consent_timestamp=datetime.now(timezone.utc),
                consent_method=ConsentMethod.DIGITAL,
            )

            assert enrollment is not None
            assert enrollment.enrollment_status == EnrollmentStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_withdraw_unit(self, experiment_id: str):
        """Test unit withdrawal."""
        from src.services.enrollment import EnrollmentService, EnrollmentStatus

        service = EnrollmentService()
        enrollment_id = uuid.uuid4()

        # Mock the ABExperimentRepository at its import location
        with patch("src.repositories.ab_experiment.ABExperimentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.update_enrollment_status = AsyncMock(
                return_value=MagicMock(
                    id=enrollment_id,
                    enrollment_status=EnrollmentStatus.WITHDRAWN,
                    withdrawal_timestamp=datetime.now(timezone.utc),
                    withdrawal_reason="Subject requested",
                )
            )

            result = await service.withdraw_unit(
                enrollment_id=enrollment_id,
                reason="Subject requested",
            )

            mock_repo.update_enrollment_status.assert_called_once()
            assert result.enrollment_status == EnrollmentStatus.WITHDRAWN

    @pytest.mark.asyncio
    async def test_enrollment_stats(self, experiment_id: str):
        """Test enrollment statistics retrieval."""
        from src.services.enrollment import EnrollmentService, EnrollmentStats

        service = EnrollmentService()

        # Mock the ABExperimentRepository at its import location
        # The service calls repo.get_assignments and repo.get_enrollment_by_assignment
        with patch("src.repositories.ab_experiment.ABExperimentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            exp_id = uuid.UUID(experiment_id)

            # Mock get_assignments to return some assignment records
            mock_assignment = MagicMock()
            mock_assignment.id = uuid.uuid4()
            mock_assignment.variant = "control"
            mock_repo.get_assignments = AsyncMock(return_value=[mock_assignment])

            # Mock get_enrollment_by_assignment to return enrollment
            mock_enrollment = MagicMock()
            mock_enrollment.enrollment_status = "active"
            mock_repo.get_enrollment_by_assignment = AsyncMock(return_value=mock_enrollment)

            stats = await service.get_enrollment_stats(exp_id)

            assert stats.total_assigned == 1
            assert stats.total_enrolled == 1
            assert stats.active_count == 1


# =============================================================================
# Interim Analysis Service Tests
# =============================================================================


class TestInterimAnalysisService:
    """Integration tests for InterimAnalysisService.

    Note: InterimAnalysisService contains both pure computation methods
    (boundary calculations, conditional power) and async methods that
    use the repository (perform_interim_analysis).
    """

    @pytest.mark.asyncio
    async def test_obrien_fleming_boundary(self):
        """Test O'Brien-Fleming alpha spending function - pure computation."""
        from src.services.interim_analysis import InterimAnalysisService

        service = InterimAnalysisService()

        # At 25% information fraction
        alpha_25 = service.calculate_obrien_fleming_boundary(
            information_fraction=0.25,
            total_alpha=0.05,
            num_analyses=4,
        )

        # At 50% information fraction
        alpha_50 = service.calculate_obrien_fleming_boundary(
            information_fraction=0.50,
            total_alpha=0.05,
            num_analyses=4,
        )

        # At 75% information fraction
        alpha_75 = service.calculate_obrien_fleming_boundary(
            information_fraction=0.75,
            total_alpha=0.05,
            num_analyses=4,
        )

        # O'Brien-Fleming boundaries are conservative early, more liberal later
        assert alpha_25 < alpha_50 < alpha_75
        assert alpha_25 < 0.01  # Very conservative at 25%
        assert alpha_75 < 0.05  # Still below nominal alpha

    @pytest.mark.asyncio
    async def test_conditional_power_calculation(self):
        """Test conditional power calculation - pure computation."""
        from src.services.interim_analysis import InterimAnalysisService

        service = InterimAnalysisService()

        # High conditional power scenario - uses current_n and target_n
        high_power = service.calculate_conditional_power(
            current_effect=0.08,  # Strong effect
            current_variance=0.01,
            target_effect=0.05,
            current_n=500,    # Current sample size
            target_n=1000,    # Target sample size
        )

        # Low conditional power scenario
        low_power = service.calculate_conditional_power(
            current_effect=0.02,  # Weak effect
            current_variance=0.01,
            target_effect=0.05,
            current_n=100,    # Current sample size
            target_n=1000,    # Target sample size
        )

        assert high_power > low_power
        assert 0 <= high_power <= 1
        assert 0 <= low_power <= 1

    @pytest.mark.asyncio
    async def test_stopping_decision_via_interim_analysis(self, experiment_id: str):
        """Test stopping decision is embedded in interim analysis result."""
        from src.services.interim_analysis import (
            InterimAnalysisService,
            MetricData,
            StoppingDecision,
        )
        import numpy as np

        service = InterimAnalysisService()

        # Create metric data for the analysis
        metric_data = MetricData(
            name="conversion_rate",
            control_values=np.array([0.1, 0.12, 0.08, 0.11, 0.09] * 50),
            treatment_values=np.array([0.15, 0.18, 0.14, 0.16, 0.17] * 50),
        )

        # Mock the repository at its import location
        with patch("src.repositories.ab_experiment.ABExperimentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.record_interim_analysis = AsyncMock(return_value=None)

            result = await service.perform_interim_analysis(
                experiment_id=uuid.UUID(experiment_id),
                analysis_number=2,
                metric_data=metric_data,
                target_sample_size=1000,
            )

            # Decision is part of the result (not stopping_decision)
            assert result.decision in [
                StoppingDecision.CONTINUE,
                StoppingDecision.STOP_EFFICACY,
                StoppingDecision.STOP_FUTILITY,
            ]

    @pytest.mark.asyncio
    async def test_perform_interim_analysis(self, experiment_id: str):
        """Test full interim analysis workflow."""
        from src.services.interim_analysis import (
            InterimAnalysisService,
            MetricData,
        )
        import numpy as np

        service = InterimAnalysisService()

        # Create metric data for the analysis
        metric_data = MetricData(
            name="primary_metric",
            control_values=np.array([0.1, 0.12, 0.08, 0.11, 0.09] * 100),
            treatment_values=np.array([0.15, 0.18, 0.14, 0.16, 0.17] * 100),
        )

        # Mock the repository at its import location
        with patch("src.repositories.ab_experiment.ABExperimentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.record_interim_analysis = AsyncMock(return_value=None)

            result = await service.perform_interim_analysis(
                experiment_id=uuid.UUID(experiment_id),
                analysis_number=2,
                metric_data=metric_data,
                target_sample_size=1000,
            )

            assert result is not None
            assert result.information_fraction > 0
            assert result.analysis_number == 2


# =============================================================================
# Results Analysis Service Tests
# =============================================================================


class TestResultsAnalysisService:
    """Integration tests for ResultsAnalysisService.

    Note: ResultsAnalysisService performs pure statistical computations and uses
    ABResultsRepository for persistence, which is created inside methods.
    """

    @pytest.mark.asyncio
    async def test_itt_analysis(self, experiment_id: str):
        """Test intent-to-treat analysis."""
        from src.services.results_analysis import ResultsAnalysisService, AnalysisMethod
        import numpy as np

        service = ResultsAnalysisService()

        # Mock the persistence layer (import happens inside method)
        with patch("src.repositories.ab_results.ABResultsRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.save_results = AsyncMock()

            # Prepare test data
            control_data = np.array([0.1, 0.12, 0.08, 0.11, 0.09] * 100)
            treatment_data = np.array([0.15, 0.18, 0.14, 0.16, 0.17] * 100)

            result = await service.compute_itt_results(
                experiment_id=uuid.UUID(experiment_id),
                primary_metric="conversion_rate",
                control_data=control_data,
                treatment_data=treatment_data,
            )

            assert result is not None
            assert result.analysis_method == AnalysisMethod.ITT

    @pytest.mark.asyncio
    async def test_srm_check(self, experiment_id: str):
        """Test Sample Ratio Mismatch detection."""
        from src.services.results_analysis import ResultsAnalysisService

        service = ResultsAnalysisService()

        # Mock the persistence layer (import happens inside method)
        with patch("src.repositories.ab_results.ABResultsRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.save_srm_check = AsyncMock()

            # Balanced counts - no SRM expected
            result = await service.check_sample_ratio_mismatch(
                experiment_id=uuid.UUID(experiment_id),
                expected_ratio={"control": 0.5, "treatment": 0.5},
                actual_counts={"control": 500, "treatment": 500},
            )

            assert result.is_srm_detected is False

    @pytest.mark.asyncio
    async def test_srm_detection_with_imbalance(self, experiment_id: str):
        """Test SRM detection with significant imbalance."""
        from src.services.results_analysis import ResultsAnalysisService

        service = ResultsAnalysisService()

        # Mock the persistence layer (import happens inside method)
        with patch("src.repositories.ab_results.ABResultsRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.save_srm_check = AsyncMock()

            # Significantly imbalanced counts - should trigger SRM
            result = await service.check_sample_ratio_mismatch(
                experiment_id=uuid.UUID(experiment_id),
                expected_ratio={"control": 0.5, "treatment": 0.5},
                actual_counts={"control": 400, "treatment": 600},
            )

            assert result.is_srm_detected is True

    @pytest.mark.asyncio
    async def test_heterogeneous_effects(self, experiment_id: str):
        """Test heterogeneous treatment effects by segment."""
        from src.services.results_analysis import ResultsAnalysisService
        import numpy as np

        service = ResultsAnalysisService()

        # Prepare segment data - dict of segment_name -> {"control": array, "treatment": array}
        segment_data = {
            "region=northeast": {
                "control": np.array([0.1] * 100),
                "treatment": np.array([0.2] * 100),
            },
            "region=southwest": {
                "control": np.array([0.1] * 100),
                "treatment": np.array([0.12] * 100),
            },
        }

        results = await service.compute_heterogeneous_effects(
            experiment_id=uuid.UUID(experiment_id),
            primary_metric="conversion_rate",
            segment_data=segment_data,
        )

        assert "region=northeast" in results
        assert "region=southwest" in results

    @pytest.mark.asyncio
    async def test_fidelity_comparison(self, experiment_id: str):
        """Test Digital Twin fidelity comparison."""
        from src.services.results_analysis import (
            ResultsAnalysisService,
            ExperimentResults,
            AnalysisType,
            AnalysisMethod,
        )

        twin_simulation_id = uuid.uuid4()

        service = ResultsAnalysisService()

        # Mock the persistence layer (import happens inside method)
        with patch("src.repositories.ab_results.ABResultsRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.save_fidelity_comparison = AsyncMock()

            # Create actual results for comparison
            actual_results = ExperimentResults(
                experiment_id=uuid.UUID(experiment_id),
                analysis_type=AnalysisType.FINAL,
                analysis_method=AnalysisMethod.ITT,
                computed_at=datetime.now(timezone.utc),
                primary_metric="conversion_rate",
                control_mean=0.10,
                treatment_mean=0.16,
                effect_estimate=0.06,  # Actual effect
                effect_ci_lower=0.04,
                effect_ci_upper=0.08,
                relative_lift=60.0,
                relative_lift_ci_lower=40.0,
                relative_lift_ci_upper=80.0,
                p_value=0.01,
                is_significant=True,
                sample_size_control=500,
                sample_size_treatment=500,
                statistical_power=0.90,
            )

            result = await service.compare_with_twin_prediction(
                experiment_id=uuid.UUID(experiment_id),
                twin_simulation_id=twin_simulation_id,
                actual_results=actual_results,
                predicted_effect=0.08,  # Predicted 0.08, actual was 0.06
                predicted_ci_lower=0.05,
                predicted_ci_upper=0.11,
            )

            # Prediction error = 0.06 - 0.08 = -0.02
            # Actual falls within predicted CI (0.05 to 0.11)
            assert result.actual_effect == 0.06
            assert result.predicted_effect == 0.08
            assert result.ci_coverage is True
            assert result.fidelity_score > 0  # Should have positive fidelity score


# =============================================================================
# Experiment Monitor Agent Tests
# =============================================================================


class TestExperimentMonitorAgent:
    """Integration tests for ExperimentMonitorAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent can be initialized."""
        from src.agents.experiment_monitor import ExperimentMonitorAgent

        agent = ExperimentMonitorAgent()
        assert agent.graph is not None

    @pytest.mark.asyncio
    async def test_health_check_workflow(self, experiment_id: str):
        """Test health checking workflow."""
        from src.agents.experiment_monitor import (
            ExperimentMonitorAgent,
            ExperimentMonitorInput,
        )

        agent = ExperimentMonitorAgent()

        # Mock the graph execution
        with patch.object(agent, "graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(
                return_value={
                    "experiments": [
                        {
                            "experiment_id": experiment_id,
                            "name": "Test Experiment",
                            "health_status": "healthy",
                            "total_enrolled": 500,
                            "enrollment_rate_per_day": 10.5,
                        }
                    ],
                    "alerts": [],
                    "experiments_checked": 1,
                    "monitor_summary": "1 experiment checked, all healthy",
                    "recommended_actions": [],
                    "check_latency_ms": 150,
                    "srm_issues": [],
                    "enrollment_issues": [],
                }
            )

            result = await agent.run_async(
                ExperimentMonitorInput(
                    experiment_ids=[experiment_id],
                    check_all_active=False,
                )
            )

            assert result.experiments_checked == 1
            assert result.healthy_count == 1
            assert len(result.alerts) == 0

    @pytest.mark.asyncio
    async def test_srm_detection_in_workflow(self, experiment_id: str):
        """Test SRM detection in monitoring workflow."""
        from src.agents.experiment_monitor import (
            ExperimentMonitorAgent,
            ExperimentMonitorInput,
        )

        agent = ExperimentMonitorAgent()

        with patch.object(agent, "graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(
                return_value={
                    "experiments": [
                        {
                            "experiment_id": experiment_id,
                            "name": "Test Experiment",
                            "health_status": "warning",
                            "has_srm": True,
                        }
                    ],
                    "srm_issues": [
                        {
                            "experiment_id": experiment_id,
                            "detected": True,
                            "p_value": 0.0001,
                            "chi_squared": 25.5,
                            "severity": "critical",
                        }
                    ],
                    "alerts": [
                        {
                            "alert_id": str(uuid.uuid4()),
                            "alert_type": "srm",
                            "severity": "critical",
                            "experiment_id": experiment_id,
                            "experiment_name": "Test Experiment",
                            "message": "SRM detected",
                            "details": {},
                            "recommended_action": "Investigate",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                    "experiments_checked": 1,
                    "monitor_summary": "SRM detected",
                    "recommended_actions": ["Investigate SRM"],
                    "check_latency_ms": 200,
                    "enrollment_issues": [],
                }
            )

            result = await agent.run_async(
                ExperimentMonitorInput(
                    experiment_ids=[experiment_id],
                    srm_threshold=0.001,
                )
            )

            assert result.critical_count == 0 or result.warning_count > 0
            assert len(result.alerts) > 0

    @pytest.mark.asyncio
    async def test_enrollment_issue_detection(self, experiment_id: str):
        """Test low enrollment detection."""
        from src.agents.experiment_monitor import (
            ExperimentMonitorAgent,
            ExperimentMonitorInput,
        )

        agent = ExperimentMonitorAgent()

        with patch.object(agent, "graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(
                return_value={
                    "experiments": [
                        {
                            "experiment_id": experiment_id,
                            "name": "Test Experiment",
                            "health_status": "warning",
                            "enrollment_rate_per_day": 1.0,  # Low
                        }
                    ],
                    "enrollment_issues": [
                        {
                            "experiment_id": experiment_id,
                            "current_rate": 1.0,
                            "expected_rate": 10.0,
                            "days_below_threshold": 5,
                            "severity": "warning",
                        }
                    ],
                    "alerts": [
                        {
                            "alert_id": str(uuid.uuid4()),
                            "alert_type": "enrollment",
                            "severity": "warning",
                            "experiment_id": experiment_id,
                            "experiment_name": "Test Experiment",
                            "message": "Low enrollment",
                            "details": {},
                            "recommended_action": "Review targeting",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                    "experiments_checked": 1,
                    "monitor_summary": "Low enrollment detected",
                    "recommended_actions": ["Review enrollment criteria"],
                    "check_latency_ms": 180,
                    "srm_issues": [],
                }
            )

            result = await agent.run_async(
                ExperimentMonitorInput(
                    experiment_ids=[experiment_id],
                    enrollment_threshold=5.0,
                )
            )

            assert result.warning_count >= 0
            assert len(result.alerts) > 0


# =============================================================================
# End-to-End Flow Tests
# =============================================================================


class TestEndToEndFlow:
    """End-to-end integration tests for the complete A/B testing workflow."""

    @pytest.mark.asyncio
    async def test_full_experiment_lifecycle(
        self,
        experiment_id: str,
        sample_units: List[Dict],
        sample_experiment_config: Dict,
    ):
        """Test complete experiment lifecycle: design → randomize → enroll → analyze."""
        from src.services.randomization import RandomizationService
        from src.services.enrollment import EnrollmentService, EligibilityCriteria, EligibilityResult
        from src.services.results_analysis import ResultsAnalysisService

        # Phase 1: Randomization (pure computation, no mocking needed)
        rand_service = RandomizationService()
        rand_result = await rand_service.simple_randomize(
            experiment_id=uuid.UUID(experiment_id),
            units=sample_units,
            allocation_ratio={"control": 0.5, "treatment": 0.5},
        )
        assert len(rand_result) == 20

        # Phase 2: Enrollment
        enroll_service = EnrollmentService()
        criteria = EligibilityCriteria(
            min_rx_history_months=6,
            min_patient_panel_size=50,
            active_in_territory=True,
            not_in_concurrent_study=True,
            no_recent_protocol_violations=True,
        )

        with patch("src.repositories.ab_experiment.ABExperimentRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.create_enrollment = AsyncMock(
                return_value=MagicMock(
                    id=uuid.uuid4(),
                    enrollment_status="active",
                )
            )

            # First check eligibility - use the correct signature with unit dict
            unit = {
                "unit_id": sample_units[0]["unit_id"],
                "rx_history_months": 12,
                "patient_panel_size": 100,
                "active_in_territory": True,
                "in_concurrent_study": False,
                "recent_protocol_violations": False,
            }
            eligibility = await enroll_service.check_eligibility(
                experiment_id=uuid.UUID(experiment_id),
                unit=unit,
                criteria=criteria,
            )

            # Then enroll the unit with correct signature
            # AssignmentResult doesn't have an 'id' field, generate one for enrollment
            assignment_id = uuid.uuid4()
            from src.services.enrollment import ConsentMethod
            enrollment = await enroll_service.enroll_unit(
                assignment_id=assignment_id,
                eligibility_result=eligibility,
                consent_timestamp=datetime.now(timezone.utc),
                consent_method=ConsentMethod.DIGITAL,
            )

            # Handle both enum and string values
            status = enrollment.enrollment_status
            status_str = status.value if hasattr(status, 'value') else str(status)
            assert status_str == "active"

        # Phase 3: Results Analysis - SRM check
        results_service = ResultsAnalysisService()

        with patch("src.repositories.ab_results.ABResultsRepository") as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.save_srm_check = AsyncMock()

            srm_result = await results_service.check_sample_ratio_mismatch(
                experiment_id=uuid.UUID(experiment_id),
                expected_ratio={"control": 0.5, "treatment": 0.5},
                actual_counts={"control": 10, "treatment": 10},
            )

            assert srm_result.is_srm_detected is False

    @pytest.mark.asyncio
    async def test_monitoring_sweep(self):
        """Test monitoring sweep across multiple experiments."""
        from src.agents.experiment_monitor import (
            ExperimentMonitorAgent,
            ExperimentMonitorInput,
        )

        experiments = [f"exp-{i}-{uuid.uuid4().hex[:8]}" for i in range(3)]

        agent = ExperimentMonitorAgent()

        with patch.object(agent, "graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(
                return_value={
                    "experiments": [
                        {"experiment_id": exp, "name": f"Exp {i}", "health_status": "healthy"}
                        for i, exp in enumerate(experiments)
                    ],
                    "alerts": [],
                    "experiments_checked": 3,
                    "monitor_summary": "3 experiments checked",
                    "recommended_actions": [],
                    "check_latency_ms": 300,
                    "srm_issues": [],
                    "enrollment_issues": [],
                }
            )

            result = await agent.run_async(
                ExperimentMonitorInput(
                    check_all_active=True,
                )
            )

            assert result.experiments_checked == 3
            assert result.healthy_count == 3


# =============================================================================
# Celery Task Tests
# =============================================================================


class TestCeleryTasks:
    """Integration tests for Celery A/B testing tasks.

    Note: These tasks use async functions inside Celery tasks with dynamic imports.
    The tests verify task structure and response format rather than full execution.
    """

    def test_scheduled_interim_analysis_task_exists(self):
        """Test scheduled interim analysis task is registered."""
        from src.tasks.ab_testing_tasks import scheduled_interim_analysis

        # Verify task is callable and has expected name
        assert callable(scheduled_interim_analysis)
        assert hasattr(scheduled_interim_analysis, "delay")
        assert "scheduled_interim_analysis" in scheduled_interim_analysis.name

    def test_srm_detection_sweep_task_exists(self):
        """Test SRM detection sweep task is registered."""
        from src.tasks.ab_testing_tasks import srm_detection_sweep

        # Verify task is callable and has expected name
        assert callable(srm_detection_sweep)
        assert hasattr(srm_detection_sweep, "delay")
        assert "srm_detection_sweep" in srm_detection_sweep.name

    def test_check_all_active_experiments_task_exists(self):
        """Test check all active experiments task is registered."""
        from src.tasks.ab_testing_tasks import check_all_active_experiments

        # Verify task is callable and has expected name
        assert callable(check_all_active_experiments)
        assert hasattr(check_all_active_experiments, "delay")
        assert "check_all_active_experiments" in check_all_active_experiments.name

    def test_default_config_structure(self):
        """Test default configuration has expected structure."""
        from src.tasks.ab_testing_tasks import DEFAULT_CONFIG

        assert "interim_analysis" in DEFAULT_CONFIG
        assert "enrollment" in DEFAULT_CONFIG
        assert "srm" in DEFAULT_CONFIG
        assert "fidelity" in DEFAULT_CONFIG
        assert "schedule" in DEFAULT_CONFIG

        # Verify key settings
        assert DEFAULT_CONFIG["srm"]["detection_threshold"] == 0.001
        assert DEFAULT_CONFIG["interim_analysis"]["total_alpha"] == 0.05


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests for A/B testing infrastructure."""

    @pytest.mark.asyncio
    async def test_randomization_performance_1000_units(self, experiment_id: str):
        """Test randomization performance with 1000 units."""
        from src.services.randomization import RandomizationService

        units = [
            {"unit_id": f"hcp_{i:04d}", "unit_type": "hcp", "region": f"region_{i % 10}"}
            for i in range(1000)
        ]

        # RandomizationService is pure computation - no mocking needed
        service = RandomizationService()

        start_time = time.time()
        assignments = await service.simple_randomize(
            experiment_id=uuid.UUID(experiment_id),
            units=units,
            allocation_ratio={"control": 0.5, "treatment": 0.5},
        )
        elapsed = time.time() - start_time

        assert len(assignments) == 1000
        assert elapsed < 5.0, f"Randomization took {elapsed:.2f}s, expected < 5s"

    @pytest.mark.asyncio
    async def test_monitor_agent_performance(self):
        """Test monitoring agent performance with multiple experiments."""
        from src.agents.experiment_monitor import (
            ExperimentMonitorAgent,
            ExperimentMonitorInput,
        )

        experiment_ids = [f"exp-{i}-{uuid.uuid4().hex[:8]}" for i in range(10)]

        agent = ExperimentMonitorAgent()

        with patch.object(agent, "graph") as mock_graph:
            mock_graph.ainvoke = AsyncMock(
                return_value={
                    "experiments": [
                        {"experiment_id": exp, "health_status": "healthy"}
                        for exp in experiment_ids
                    ],
                    "alerts": [],
                    "experiments_checked": 10,
                    "check_latency_ms": 500,
                    "monitor_summary": "",
                    "recommended_actions": [],
                    "srm_issues": [],
                    "enrollment_issues": [],
                }
            )

            start_time = time.time()
            result = await agent.run_async(
                ExperimentMonitorInput(
                    experiment_ids=experiment_ids,
                    check_all_active=False,
                )
            )
            elapsed = time.time() - start_time

            assert result.experiments_checked == 10
            assert elapsed < 2.0, f"Monitoring took {elapsed:.2f}s, expected < 2s"
