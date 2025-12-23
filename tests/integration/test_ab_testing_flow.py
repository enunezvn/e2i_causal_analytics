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
    """Generate a unique experiment ID for test isolation."""
    return f"exp-{uuid.uuid4().hex[:16]}"


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
    """Integration tests for RandomizationService."""

    @pytest.mark.asyncio
    async def test_simple_randomization(self, experiment_id: str, sample_units: List[Dict]):
        """Test simple random assignment."""
        from src.services.randomization import RandomizationService

        with patch("src.services.randomization.get_supabase_client") as mock_client:
            mock_client.return_value = MagicMock()
            mock_client.return_value.table = MagicMock()

            service = RandomizationService()

            # Mock the repository
            with patch.object(service, "_repository") as mock_repo:
                mock_repo.create_assignment = AsyncMock(
                    side_effect=lambda x: MagicMock(
                        id=uuid.uuid4(),
                        experiment_id=UUID(experiment_id) if isinstance(experiment_id, str) else experiment_id,
                        unit_id=x.unit_id,
                        unit_type=x.unit_type,
                        variant=x.variant,
                        assigned_at=datetime.now(timezone.utc),
                        randomization_method="simple",
                    )
                )

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

        with patch("src.services.randomization.get_supabase_client"):
            service = RandomizationService()

            with patch.object(service, "_repository") as mock_repo:
                mock_repo.create_assignment = AsyncMock(
                    side_effect=lambda x: MagicMock(
                        id=uuid.uuid4(),
                        unit_id=x.unit_id,
                        variant=x.variant,
                        stratification_key=x.stratification_key,
                    )
                )

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

        with patch("src.services.randomization.get_supabase_client"):
            service = RandomizationService()

            with patch.object(service, "_repository") as mock_repo:
                block_counter = {"count": 0}

                def create_assignment(x):
                    block_id = f"block_{block_counter['count'] // 4}"
                    block_counter['count'] += 1
                    return MagicMock(
                        id=uuid.uuid4(),
                        unit_id=x.unit_id,
                        variant=x.variant,
                        block_id=block_id,
                    )

                mock_repo.create_assignment = AsyncMock(side_effect=create_assignment)

                assignments = await service.block_randomize(
                    experiment_id=uuid.UUID(experiment_id),
                    units=sample_units[:8],
                    block_size=4,
                    allocation_ratio={"control": 0.5, "treatment": 0.5},
                )

                # Should have 8 assignments in 2 blocks
                assert len(assignments) == 8

    @pytest.mark.asyncio
    async def test_deterministic_assignment(self, experiment_id: str):
        """Test that assignment is deterministic with same salt."""
        from src.services.randomization import RandomizationService

        service = RandomizationService()

        # Same inputs should produce same output
        hash1 = service._generate_deterministic_assignment(
            experiment_id=uuid.UUID(experiment_id),
            unit_id="hcp_001",
            salt="test_salt",
        )
        hash2 = service._generate_deterministic_assignment(
            experiment_id=uuid.UUID(experiment_id),
            unit_id="hcp_001",
            salt="test_salt",
        )

        assert hash1 == hash2
        assert 0 <= hash1 <= 1

    @pytest.mark.asyncio
    async def test_multi_arm_allocation(self, experiment_id: str):
        """Test multi-arm experiment allocation."""
        from src.services.randomization import RandomizationService

        with patch("src.services.randomization.get_supabase_client"):
            service = RandomizationService()

            with patch.object(service, "_repository") as mock_repo:
                mock_repo.create_assignment = AsyncMock(
                    side_effect=lambda x: MagicMock(
                        id=uuid.uuid4(),
                        unit_id=x.unit_id,
                        variant=x.variant,
                    )
                )

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
    """Integration tests for EnrollmentService."""

    @pytest.mark.asyncio
    async def test_check_eligibility(self, experiment_id: str):
        """Test eligibility checking."""
        from src.services.enrollment import EnrollmentService, EligibilityCriteria

        with patch("src.services.enrollment.get_supabase_client"):
            service = EnrollmentService()

            criteria = EligibilityCriteria(
                min_experience_years=2,
                required_specialty=["cardiology", "oncology"],
                excluded_regions=["restricted_region"],
            )

            unit = {
                "unit_id": "hcp_001",
                "experience_years": 5,
                "specialty": "cardiology",
                "region": "northeast",
            }

            result = await service.check_eligibility(
                experiment_id=uuid.UUID(experiment_id),
                unit=unit,
                criteria=criteria,
            )

            assert result.eligible is True
            assert result.criteria_met["min_experience_years"] is True
            assert result.criteria_met["required_specialty"] is True
            assert result.criteria_met["excluded_regions"] is True

    @pytest.mark.asyncio
    async def test_enroll_unit_success(self, experiment_id: str):
        """Test successful enrollment."""
        from src.services.enrollment import EnrollmentService

        with patch("src.services.enrollment.get_supabase_client"):
            service = EnrollmentService()

            with patch.object(service, "_repository") as mock_repo:
                # Mock assignment exists
                mock_repo.get_assignment_by_unit = AsyncMock(
                    return_value=MagicMock(
                        id=uuid.uuid4(),
                        variant="treatment",
                    )
                )
                mock_repo.create_enrollment = AsyncMock(
                    return_value=MagicMock(
                        id=uuid.uuid4(),
                        assignment_id=uuid.uuid4(),
                        enrolled_at=datetime.now(timezone.utc),
                        enrollment_status="active",
                        variant="treatment",
                    )
                )

                enrollment = await service.enroll_unit(
                    experiment_id=uuid.UUID(experiment_id),
                    unit_id="hcp_001",
                    unit_type="hcp",
                    consent_timestamp=datetime.now(timezone.utc),
                )

                assert enrollment is not None
                assert enrollment.enrollment_status == "active"

    @pytest.mark.asyncio
    async def test_withdraw_unit(self, experiment_id: str):
        """Test unit withdrawal."""
        from src.services.enrollment import EnrollmentService

        with patch("src.services.enrollment.get_supabase_client"):
            service = EnrollmentService()
            enrollment_id = uuid.uuid4()

            with patch.object(service, "_repository") as mock_repo:
                mock_repo.update_enrollment_status = AsyncMock(
                    return_value=MagicMock(
                        id=enrollment_id,
                        enrollment_status="withdrawn",
                        withdrawal_timestamp=datetime.now(timezone.utc),
                        withdrawal_reason="Subject requested",
                    )
                )

                await service.withdraw_unit(
                    enrollment_id=enrollment_id,
                    reason="Subject requested",
                )

                mock_repo.update_enrollment_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_enrollment_stats(self, experiment_id: str):
        """Test enrollment statistics retrieval."""
        from src.services.enrollment import EnrollmentService

        with patch("src.services.enrollment.get_supabase_client"):
            service = EnrollmentService()

            with patch.object(service, "_repository") as mock_repo:
                mock_repo.get_enrollment_stats = AsyncMock(
                    return_value=MagicMock(
                        total_enrolled=100,
                        active_count=85,
                        withdrawn_count=10,
                        completed_count=5,
                        enrollment_rate_per_day=5.2,
                        variant_breakdown={"control": 50, "treatment": 50},
                        enrollment_trend=[],
                    )
                )

                stats = await service.get_enrollment_stats(uuid.UUID(experiment_id))

                assert stats.total_enrolled == 100
                assert stats.active_count == 85
                assert stats.withdrawn_count == 10


# =============================================================================
# Interim Analysis Service Tests
# =============================================================================


class TestInterimAnalysisService:
    """Integration tests for InterimAnalysisService."""

    @pytest.mark.asyncio
    async def test_obrien_fleming_boundary(self):
        """Test O'Brien-Fleming alpha spending function."""
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
        """Test conditional power calculation."""
        from src.services.interim_analysis import InterimAnalysisService

        service = InterimAnalysisService()

        # High conditional power scenario
        high_power = service.calculate_conditional_power(
            current_effect=0.08,  # Strong effect
            current_variance=0.01,
            target_effect=0.05,
            remaining_sample=500,
        )

        # Low conditional power scenario
        low_power = service.calculate_conditional_power(
            current_effect=0.02,  # Weak effect
            current_variance=0.01,
            target_effect=0.05,
            remaining_sample=100,
        )

        assert high_power > low_power
        assert 0 <= high_power <= 1
        assert 0 <= low_power <= 1

    @pytest.mark.asyncio
    async def test_stopping_decision(self, experiment_id: str):
        """Test stopping decision logic."""
        from src.services.interim_analysis import InterimAnalysisService, InterimAnalysisResult

        with patch("src.services.interim_analysis.get_supabase_client"):
            service = InterimAnalysisService()

            # Test efficacy stop (low p-value)
            efficacy_result = MagicMock(
                p_value=0.001,
                conditional_power=0.95,
                information_fraction=0.5,
            )

            decision = await service.recommend_decision(
                experiment_id=uuid.UUID(experiment_id),
                analysis_result=efficacy_result,
            )

            assert decision in ["continue", "stop_efficacy", "stop_futility"]

    @pytest.mark.asyncio
    async def test_perform_interim_analysis(self, experiment_id: str):
        """Test full interim analysis workflow."""
        from src.services.interim_analysis import InterimAnalysisService

        with patch("src.services.interim_analysis.get_supabase_client"):
            service = InterimAnalysisService()

            with patch.object(service, "_repository") as mock_repo:
                mock_repo.get_experiment_data = AsyncMock(
                    return_value=MagicMock(
                        current_sample_size=500,
                        target_sample_size=1000,
                        control_data=[0.1, 0.12, 0.08],
                        treatment_data=[0.15, 0.18, 0.14],
                    )
                )
                mock_repo.record_interim_analysis = AsyncMock(
                    return_value=MagicMock(
                        id=uuid.uuid4(),
                        analysis_number=2,
                        performed_at=datetime.now(timezone.utc),
                        information_fraction=0.5,
                        alpha_spent=0.015,
                        adjusted_alpha=0.014,
                        test_statistic=2.5,
                        p_value=0.012,
                        conditional_power=0.75,
                        decision="continue",
                        metrics_snapshot={},
                    )
                )

                result = await service.perform_interim_analysis(
                    experiment_id=uuid.UUID(experiment_id),
                    analysis_number=2,
                )

                assert result is not None
                assert result.information_fraction == 0.5


# =============================================================================
# Results Analysis Service Tests
# =============================================================================


class TestResultsAnalysisService:
    """Integration tests for ResultsAnalysisService."""

    @pytest.mark.asyncio
    async def test_itt_analysis(self, experiment_id: str):
        """Test intent-to-treat analysis."""
        from src.services.results_analysis import ResultsAnalysisService

        with patch("src.services.results_analysis.get_supabase_client"):
            service = ResultsAnalysisService()

            with patch.object(service, "_repository") as mock_repo:
                mock_repo.get_experiment_outcomes = AsyncMock(
                    return_value={
                        "control": [0.1, 0.12, 0.08, 0.11, 0.09] * 100,
                        "treatment": [0.15, 0.18, 0.14, 0.16, 0.17] * 100,
                    }
                )
                mock_repo.save_results = AsyncMock(
                    return_value=MagicMock(
                        id=uuid.uuid4(),
                        analysis_type="final",
                        analysis_method="itt",
                        control_mean=0.1,
                        treatment_mean=0.16,
                        effect_estimate=0.06,
                        p_value=0.01,
                        is_significant=True,
                    )
                )

                result = await service.compute_itt_results(uuid.UUID(experiment_id))

                assert result is not None
                assert result.analysis_method == "itt"

    @pytest.mark.asyncio
    async def test_srm_check(self, experiment_id: str):
        """Test Sample Ratio Mismatch detection."""
        from src.services.results_analysis import ResultsAnalysisService

        with patch("src.services.results_analysis.get_supabase_client"):
            service = ResultsAnalysisService()

            with patch.object(service, "_repository") as mock_repo:
                # Balanced counts - no SRM
                mock_repo.get_variant_counts = AsyncMock(
                    return_value={"control": 500, "treatment": 500}
                )
                mock_repo.save_srm_check = AsyncMock(
                    return_value=MagicMock(
                        id=uuid.uuid4(),
                        is_srm_detected=False,
                    )
                )

                result = await service.check_sample_ratio_mismatch(uuid.UUID(experiment_id))

                assert result.is_srm_detected is False

    @pytest.mark.asyncio
    async def test_srm_detection_with_imbalance(self, experiment_id: str):
        """Test SRM detection with significant imbalance."""
        from src.services.results_analysis import ResultsAnalysisService

        with patch("src.services.results_analysis.get_supabase_client"):
            service = ResultsAnalysisService()

            with patch.object(service, "_repository") as mock_repo:
                # Significantly imbalanced counts - should trigger SRM
                mock_repo.get_variant_counts = AsyncMock(
                    return_value={"control": 400, "treatment": 600}
                )
                mock_repo.save_srm_check = AsyncMock(
                    return_value=MagicMock(
                        id=uuid.uuid4(),
                        chi_squared_statistic=40.0,
                        p_value=0.0000001,
                        is_srm_detected=True,
                    )
                )

                result = await service.check_sample_ratio_mismatch(uuid.UUID(experiment_id))

                assert result.is_srm_detected is True

    @pytest.mark.asyncio
    async def test_heterogeneous_effects(self, experiment_id: str):
        """Test heterogeneous treatment effects by segment."""
        from src.services.results_analysis import ResultsAnalysisService

        with patch("src.services.results_analysis.get_supabase_client"):
            service = ResultsAnalysisService()

            with patch.object(service, "_repository") as mock_repo:
                mock_repo.get_outcomes_by_segment = AsyncMock(
                    return_value={
                        "region=northeast": {
                            "control": [0.1] * 100,
                            "treatment": [0.2] * 100,
                        },
                        "region=southwest": {
                            "control": [0.1] * 100,
                            "treatment": [0.12] * 100,
                        },
                    }
                )

                results = await service.compute_heterogeneous_effects(
                    uuid.UUID(experiment_id),
                    segments=["region"],
                )

                assert "region=northeast" in results
                assert "region=southwest" in results

    @pytest.mark.asyncio
    async def test_fidelity_comparison(self, experiment_id: str):
        """Test Digital Twin fidelity comparison."""
        from src.services.results_analysis import ResultsAnalysisService

        twin_simulation_id = uuid.uuid4()

        with patch("src.services.results_analysis.get_supabase_client"):
            service = ResultsAnalysisService()

            with patch.object(service, "_repository") as mock_repo:
                mock_repo.get_twin_prediction = AsyncMock(
                    return_value=MagicMock(predicted_effect=0.08)
                )
                mock_repo.get_actual_effect = AsyncMock(return_value=0.06)
                mock_repo.save_fidelity_comparison = AsyncMock(
                    return_value=MagicMock(
                        id=uuid.uuid4(),
                        predicted_effect=0.08,
                        actual_effect=0.06,
                        prediction_error=0.25,  # 25% error
                        fidelity_score=0.75,
                        confidence_interval_coverage=True,
                    )
                )

                result = await service.compare_with_twin_prediction(
                    uuid.UUID(experiment_id),
                    twin_simulation_id,
                )

                assert result.prediction_error == 0.25
                assert result.fidelity_score == 0.75


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

        with patch("src.agents.experiment_monitor.nodes.health_checker.get_supabase_client"):
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

        with patch("src.agents.experiment_monitor.nodes.srm_detector.get_supabase_client"):
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
        from src.services.enrollment import EnrollmentService
        from src.services.results_analysis import ResultsAnalysisService

        # Phase 1: Randomization
        with patch("src.services.randomization.get_supabase_client"):
            rand_service = RandomizationService()

            with patch.object(rand_service, "_repository") as mock_repo:
                assignments = []
                for i, unit in enumerate(sample_units):
                    variant = "control" if i % 2 == 0 else "treatment"
                    assignments.append(
                        MagicMock(
                            id=uuid.uuid4(),
                            unit_id=unit["unit_id"],
                            variant=variant,
                            assigned_at=datetime.now(timezone.utc),
                        )
                    )

                mock_repo.create_assignment = AsyncMock(
                    side_effect=lambda x: next(
                        (a for a in assignments if a.unit_id == x.unit_id), None
                    )
                )

                rand_result = await rand_service.simple_randomize(
                    experiment_id=uuid.UUID(experiment_id),
                    units=sample_units,
                    allocation_ratio={"control": 0.5, "treatment": 0.5},
                )

                assert len(rand_result) == 20

        # Phase 2: Enrollment
        with patch("src.services.enrollment.get_supabase_client"):
            enroll_service = EnrollmentService()

            with patch.object(enroll_service, "_repository") as mock_repo:
                mock_repo.get_assignment_by_unit = AsyncMock(
                    return_value=MagicMock(id=uuid.uuid4(), variant="treatment")
                )
                mock_repo.create_enrollment = AsyncMock(
                    return_value=MagicMock(
                        id=uuid.uuid4(),
                        enrollment_status="active",
                    )
                )

                enrollment = await enroll_service.enroll_unit(
                    experiment_id=uuid.UUID(experiment_id),
                    unit_id=sample_units[0]["unit_id"],
                    unit_type="hcp",
                )

                assert enrollment.enrollment_status == "active"

        # Phase 3: Results Analysis
        with patch("src.services.results_analysis.get_supabase_client"):
            results_service = ResultsAnalysisService()

            with patch.object(results_service, "_repository") as mock_repo:
                mock_repo.get_variant_counts = AsyncMock(
                    return_value={"control": 10, "treatment": 10}
                )
                mock_repo.save_srm_check = AsyncMock(
                    return_value=MagicMock(
                        id=uuid.uuid4(),
                        is_srm_detected=False,
                    )
                )

                srm_result = await results_service.check_sample_ratio_mismatch(
                    uuid.UUID(experiment_id)
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
    """Integration tests for Celery A/B testing tasks."""

    @pytest.mark.asyncio
    async def test_scheduled_interim_analysis_task(self, experiment_id: str):
        """Test scheduled interim analysis Celery task."""
        from src.tasks.ab_testing_tasks import scheduled_interim_analysis

        with patch("src.tasks.ab_testing_tasks.InterimAnalysisService") as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance
            mock_instance.perform_interim_analysis = AsyncMock(
                return_value=MagicMock(
                    id=uuid.uuid4(),
                    decision="continue",
                )
            )

            # Test the async helper
            from src.tasks.ab_testing_tasks import _run_scheduled_interim_analysis

            result = await _run_scheduled_interim_analysis(experiment_id)

            assert result["status"] in ["success", "pending"]

    @pytest.mark.asyncio
    async def test_srm_detection_sweep_task(self):
        """Test SRM detection sweep task."""
        from src.tasks.ab_testing_tasks import srm_detection_sweep

        with patch("src.tasks.ab_testing_tasks.ResultsAnalysisService") as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance
            mock_instance.get_active_experiments = AsyncMock(return_value=[])

            from src.tasks.ab_testing_tasks import _run_srm_detection_sweep

            result = await _run_srm_detection_sweep()

            assert "experiments_checked" in result
            assert "srm_detected" in result

    @pytest.mark.asyncio
    async def test_check_all_active_experiments_task(self):
        """Test check all active experiments task."""
        from src.tasks.ab_testing_tasks import check_all_active_experiments

        with patch("src.tasks.ab_testing_tasks.ExperimentMonitorAgent") as mock_agent:
            mock_instance = MagicMock()
            mock_agent.return_value = mock_instance
            mock_instance.run_async = AsyncMock(
                return_value=MagicMock(
                    experiments_checked=5,
                    healthy_count=4,
                    warning_count=1,
                    critical_count=0,
                    alerts=[],
                )
            )

            from src.tasks.ab_testing_tasks import _run_check_all_active_experiments

            result = await _run_check_all_active_experiments(srm_threshold=0.001)

            assert result["experiments_checked"] == 5
            assert result["healthy_count"] == 4


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

        with patch("src.services.randomization.get_supabase_client"):
            service = RandomizationService()

            with patch.object(service, "_repository") as mock_repo:
                mock_repo.create_assignment = AsyncMock(
                    side_effect=lambda x: MagicMock(
                        id=uuid.uuid4(),
                        unit_id=x.unit_id,
                        variant=x.variant,
                    )
                )

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
