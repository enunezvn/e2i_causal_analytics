"""
Unit Tests for Enrollment Service (Phase 15).

Tests cover:
- Eligibility checking
- Unit enrollment
- Withdrawal handling
- Protocol deviation recording
- Enrollment statistics
- Batch enrollment
"""

from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.services.enrollment import (
    ConsentMethod,
    DeviationSeverity,
    EligibilityCriteria,
    EligibilityResult,
    EnrollmentConfig,
    EnrollmentRecord,
    EnrollmentService,
    EnrollmentStats,
    EnrollmentStatus,
    ProtocolDeviation,
    WithdrawalInitiator,
    get_enrollment_service,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def default_config() -> EnrollmentConfig:
    """Create default enrollment configuration."""
    return EnrollmentConfig(
        require_explicit_consent=True,
        default_consent_version="1.0",
        max_minor_deviations=3,
        auto_exclude_on_major_deviation=True,
    )


@pytest.fixture
def lenient_config() -> EnrollmentConfig:
    """Create lenient enrollment configuration."""
    return EnrollmentConfig(
        require_explicit_consent=False,
        max_minor_deviations=10,
        auto_exclude_on_major_deviation=False,
    )


@pytest.fixture
def service(default_config: EnrollmentConfig) -> EnrollmentService:
    """Create enrollment service instance."""
    return EnrollmentService(config=default_config)


@pytest.fixture
def lenient_service(lenient_config: EnrollmentConfig) -> EnrollmentService:
    """Create lenient enrollment service instance."""
    return EnrollmentService(config=lenient_config)


@pytest.fixture
def experiment_id() -> UUID:
    """Create test experiment ID."""
    return uuid4()


@pytest.fixture
def assignment_id() -> UUID:
    """Create test assignment ID."""
    return uuid4()


@pytest.fixture
def eligible_unit() -> Dict[str, Any]:
    """Create an eligible unit."""
    return {
        "id": "unit_001",
        "rx_history_months": 12,
        "patient_panel_size": 100,
        "active_in_territory": True,
        "in_concurrent_study": False,
        "recent_protocol_violations": False,
    }


@pytest.fixture
def ineligible_unit() -> Dict[str, Any]:
    """Create an ineligible unit."""
    return {
        "id": "unit_002",
        "rx_history_months": 3,  # Too short
        "patient_panel_size": 5,  # Too small
        "active_in_territory": False,
        "in_concurrent_study": True,
        "recent_protocol_violations": True,
    }


@pytest.fixture
def strict_criteria() -> EligibilityCriteria:
    """Create strict eligibility criteria."""
    return EligibilityCriteria(
        min_rx_history_months=6,
        min_patient_panel_size=50,
        active_in_territory=True,
        not_in_concurrent_study=True,
        no_recent_protocol_violations=True,
    )


@pytest.fixture
def lenient_criteria() -> EligibilityCriteria:
    """Create lenient eligibility criteria."""
    return EligibilityCriteria(
        min_rx_history_months=0,
        min_patient_panel_size=0,
        active_in_territory=False,
        not_in_concurrent_study=False,
        no_recent_protocol_violations=False,
    )


@pytest.fixture
def mock_repo():
    """Create mock ABExperimentRepository."""
    mock = MagicMock()
    mock.create_enrollment = AsyncMock()
    mock.update_enrollment_status = AsyncMock()
    mock.get_enrollment = AsyncMock()
    mock.get_enrollment_by_assignment = AsyncMock()
    mock.get_assignments = AsyncMock()
    mock.update_protocol_deviations = AsyncMock()
    return mock


# =============================================================================
# ENROLLMENT STATUS TESTS
# =============================================================================


class TestEnrollmentStatus:
    """Tests for EnrollmentStatus enum."""

    def test_active_status_exists(self):
        """Test ACTIVE status is defined."""
        assert EnrollmentStatus.ACTIVE == "active"

    def test_withdrawn_status_exists(self):
        """Test WITHDRAWN status is defined."""
        assert EnrollmentStatus.WITHDRAWN == "withdrawn"

    def test_excluded_status_exists(self):
        """Test EXCLUDED status is defined."""
        assert EnrollmentStatus.EXCLUDED == "excluded"

    def test_completed_status_exists(self):
        """Test COMPLETED status is defined."""
        assert EnrollmentStatus.COMPLETED == "completed"

    def test_lost_to_followup_status_exists(self):
        """Test LOST_TO_FOLLOWUP status is defined."""
        assert EnrollmentStatus.LOST_TO_FOLLOWUP == "lost_to_followup"


# =============================================================================
# WITHDRAWAL INITIATOR TESTS
# =============================================================================


class TestWithdrawalInitiator:
    """Tests for WithdrawalInitiator enum."""

    def test_subject_initiator_exists(self):
        """Test SUBJECT initiator is defined."""
        assert WithdrawalInitiator.SUBJECT == "subject"

    def test_investigator_initiator_exists(self):
        """Test INVESTIGATOR initiator is defined."""
        assert WithdrawalInitiator.INVESTIGATOR == "investigator"

    def test_sponsor_initiator_exists(self):
        """Test SPONSOR initiator is defined."""
        assert WithdrawalInitiator.SPONSOR == "sponsor"

    def test_system_initiator_exists(self):
        """Test SYSTEM initiator is defined."""
        assert WithdrawalInitiator.SYSTEM == "system"


# =============================================================================
# CONSENT METHOD TESTS
# =============================================================================


class TestConsentMethod:
    """Tests for ConsentMethod enum."""

    def test_email_method_exists(self):
        """Test EMAIL consent method is defined."""
        assert ConsentMethod.EMAIL == "email"

    def test_phone_method_exists(self):
        """Test PHONE consent method is defined."""
        assert ConsentMethod.PHONE == "phone"

    def test_in_person_method_exists(self):
        """Test IN_PERSON consent method is defined."""
        assert ConsentMethod.IN_PERSON == "in_person"

    def test_implied_method_exists(self):
        """Test IMPLIED consent method is defined."""
        assert ConsentMethod.IMPLIED == "implied"

    def test_digital_method_exists(self):
        """Test DIGITAL consent method is defined."""
        assert ConsentMethod.DIGITAL == "digital"


# =============================================================================
# DEVIATION SEVERITY TESTS
# =============================================================================


class TestDeviationSeverity:
    """Tests for DeviationSeverity enum."""

    def test_minor_severity_exists(self):
        """Test MINOR severity is defined."""
        assert DeviationSeverity.MINOR == "minor"

    def test_major_severity_exists(self):
        """Test MAJOR severity is defined."""
        assert DeviationSeverity.MAJOR == "major"

    def test_critical_severity_exists(self):
        """Test CRITICAL severity is defined."""
        assert DeviationSeverity.CRITICAL == "critical"


# =============================================================================
# ELIGIBILITY CRITERIA TESTS
# =============================================================================


class TestEligibilityCriteria:
    """Tests for EligibilityCriteria dataclass."""

    def test_default_criteria(self):
        """Test default eligibility criteria values."""
        criteria = EligibilityCriteria()

        assert criteria.min_rx_history_months == 0
        assert criteria.min_patient_panel_size == 0
        assert criteria.active_in_territory is True
        assert criteria.not_in_concurrent_study is True
        assert criteria.no_recent_protocol_violations is True

    def test_custom_criteria(self):
        """Test custom eligibility criteria."""
        criteria = EligibilityCriteria(
            min_rx_history_months=12,
            min_patient_panel_size=100,
            custom_criteria={"is_key_opinion_leader": True},
        )

        assert criteria.min_rx_history_months == 12
        assert criteria.custom_criteria["is_key_opinion_leader"] is True


# =============================================================================
# ENROLLMENT CONFIG TESTS
# =============================================================================


class TestEnrollmentConfig:
    """Tests for EnrollmentConfig dataclass."""

    def test_default_config(self):
        """Test default enrollment configuration."""
        config = EnrollmentConfig()

        assert config.require_explicit_consent is True
        assert config.default_consent_version == "1.0"
        assert config.max_minor_deviations == 3
        assert config.auto_exclude_on_major_deviation is True

    def test_custom_config(self):
        """Test custom enrollment configuration."""
        config = EnrollmentConfig(
            require_explicit_consent=False,
            max_minor_deviations=5,
        )

        assert config.require_explicit_consent is False
        assert config.max_minor_deviations == 5


# =============================================================================
# ELIGIBILITY CHECK TESTS
# =============================================================================


class TestEligibilityCheck:
    """Tests for eligibility checking."""

    @pytest.mark.asyncio
    async def test_eligible_unit_passes(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
        eligible_unit: Dict,
    ):
        """Test eligible unit passes eligibility check."""
        result = await service.check_eligibility(
            experiment_id=experiment_id,
            unit=eligible_unit,
        )

        assert isinstance(result, EligibilityResult)
        assert result.is_eligible is True
        assert len(result.failed_criteria) == 0

    @pytest.mark.asyncio
    async def test_ineligible_unit_fails(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
        ineligible_unit: Dict,
        strict_criteria: EligibilityCriteria,
    ):
        """Test ineligible unit fails eligibility check."""
        result = await service.check_eligibility(
            experiment_id=experiment_id,
            unit=ineligible_unit,
            criteria=strict_criteria,
        )

        assert result.is_eligible is False
        assert len(result.failed_criteria) > 0

    @pytest.mark.asyncio
    async def test_eligibility_check_returns_criteria_results(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
        eligible_unit: Dict,
    ):
        """Test eligibility check returns detailed criteria results."""
        result = await service.check_eligibility(
            experiment_id=experiment_id,
            unit=eligible_unit,
        )

        assert "min_rx_history" in result.criteria_results
        assert "min_patient_panel" in result.criteria_results
        assert "active_in_territory" in result.criteria_results

    @pytest.mark.asyncio
    async def test_eligibility_check_min_rx_history(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
    ):
        """Test eligibility check for minimum Rx history."""
        criteria = EligibilityCriteria(min_rx_history_months=12)
        unit = {"id": "test", "rx_history_months": 6}

        result = await service.check_eligibility(
            experiment_id=experiment_id,
            unit=unit,
            criteria=criteria,
        )

        assert result.is_eligible is False
        assert any("rx history" in c.lower() for c in result.failed_criteria)

    @pytest.mark.asyncio
    async def test_eligibility_check_min_panel_size(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
    ):
        """Test eligibility check for minimum patient panel size."""
        criteria = EligibilityCriteria(min_patient_panel_size=50)
        unit = {"id": "test", "patient_panel_size": 25}

        result = await service.check_eligibility(
            experiment_id=experiment_id,
            unit=unit,
            criteria=criteria,
        )

        assert result.is_eligible is False
        assert any("panel" in c.lower() for c in result.failed_criteria)

    @pytest.mark.asyncio
    async def test_eligibility_check_concurrent_study(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
    ):
        """Test eligibility check for concurrent study exclusion."""
        criteria = EligibilityCriteria(not_in_concurrent_study=True)
        unit = {"id": "test", "in_concurrent_study": True}

        result = await service.check_eligibility(
            experiment_id=experiment_id,
            unit=unit,
            criteria=criteria,
        )

        assert result.is_eligible is False
        assert any("concurrent" in c.lower() for c in result.failed_criteria)

    @pytest.mark.asyncio
    async def test_eligibility_check_custom_criteria(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
    ):
        """Test eligibility check with custom criteria."""
        criteria = EligibilityCriteria(
            custom_criteria={"is_contracted": True}
        )
        unit = {"id": "test", "is_contracted": False}

        result = await service.check_eligibility(
            experiment_id=experiment_id,
            unit=unit,
            criteria=criteria,
        )

        assert result.is_eligible is False
        assert any("custom criterion" in c.lower() for c in result.failed_criteria)

    @pytest.mark.asyncio
    async def test_eligibility_check_sets_timestamp(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
        eligible_unit: Dict,
    ):
        """Test eligibility check sets checked_at timestamp."""
        before = datetime.now(timezone.utc)
        result = await service.check_eligibility(
            experiment_id=experiment_id,
            unit=eligible_unit,
        )
        after = datetime.now(timezone.utc)

        assert before <= result.checked_at <= after


# =============================================================================
# ENROLLMENT TESTS
# =============================================================================


class TestEnrollment:
    """Tests for unit enrollment."""

    @pytest.mark.asyncio
    async def test_enroll_eligible_unit(
        self,
        service: EnrollmentService,
        assignment_id: UUID,
        mock_repo,
    ):
        """Test enrolling an eligible unit."""
        eligibility = EligibilityResult(
            is_eligible=True,
            criteria_results={"all": True},
            failed_criteria=[],
        )

        mock_enrollment = MagicMock()
        mock_enrollment.id = uuid4()
        mock_repo.create_enrollment.return_value = mock_enrollment

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            result = await service.enroll_unit(
                assignment_id=assignment_id,
                eligibility_result=eligibility,
                consent_timestamp=datetime.now(timezone.utc),
                consent_method=ConsentMethod.DIGITAL,
            )

            mock_repo.create_enrollment.assert_called_once()

    @pytest.mark.asyncio
    async def test_enroll_ineligible_unit_raises_error(
        self,
        service: EnrollmentService,
        assignment_id: UUID,
    ):
        """Test enrolling an ineligible unit raises ValueError."""
        eligibility = EligibilityResult(
            is_eligible=False,
            criteria_results={"all": False},
            failed_criteria=["Criterion X not met"],
        )

        with pytest.raises(ValueError, match="Cannot enroll ineligible"):
            await service.enroll_unit(
                assignment_id=assignment_id,
                eligibility_result=eligibility,
            )

    @pytest.mark.asyncio
    async def test_enroll_with_implied_consent(
        self,
        service: EnrollmentService,
        assignment_id: UUID,
        mock_repo,
    ):
        """Test enrollment applies implied consent when explicit consent required but not provided."""
        eligibility = EligibilityResult(
            is_eligible=True,
            criteria_results={"all": True},
            failed_criteria=[],
        )

        mock_enrollment = MagicMock()
        mock_repo.create_enrollment.return_value = mock_enrollment

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            await service.enroll_unit(
                assignment_id=assignment_id,
                eligibility_result=eligibility,
            )

            # Check that consent was set
            call_args = mock_repo.create_enrollment.call_args
            assert call_args.kwargs.get("consent_timestamp") is not None
            assert call_args.kwargs.get("consent_method") == ConsentMethod.IMPLIED.value


# =============================================================================
# WITHDRAWAL TESTS
# =============================================================================


class TestWithdrawal:
    """Tests for unit withdrawal."""

    @pytest.mark.asyncio
    async def test_withdraw_unit(
        self,
        service: EnrollmentService,
        mock_repo,
    ):
        """Test withdrawing a unit from experiment."""
        enrollment_id = uuid4()
        mock_enrollment = MagicMock()
        mock_repo.update_enrollment_status.return_value = mock_enrollment

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            result = await service.withdraw_unit(
                enrollment_id=enrollment_id,
                reason="Subject requested withdrawal",
            )

            mock_repo.update_enrollment_status.assert_called_once()
            call_args = mock_repo.update_enrollment_status.call_args
            assert call_args.kwargs["status"] == EnrollmentStatus.WITHDRAWN.value

    @pytest.mark.asyncio
    async def test_withdraw_unit_with_initiator(
        self,
        service: EnrollmentService,
        mock_repo,
    ):
        """Test withdrawal with specific initiator."""
        enrollment_id = uuid4()
        mock_enrollment = MagicMock()
        mock_repo.update_enrollment_status.return_value = mock_enrollment

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            await service.withdraw_unit(
                enrollment_id=enrollment_id,
                reason="Safety concern",
                initiated_by=WithdrawalInitiator.INVESTIGATOR,
            )

            call_args = mock_repo.update_enrollment_status.call_args
            assert call_args.kwargs["withdrawal_initiated_by"] == WithdrawalInitiator.INVESTIGATOR.value


# =============================================================================
# STATUS UPDATE TESTS
# =============================================================================


class TestStatusUpdates:
    """Tests for enrollment status updates."""

    @pytest.mark.asyncio
    async def test_mark_completed(
        self,
        service: EnrollmentService,
        mock_repo,
    ):
        """Test marking enrollment as completed."""
        enrollment_id = uuid4()
        mock_enrollment = MagicMock()
        mock_repo.update_enrollment_status.return_value = mock_enrollment

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            await service.mark_completed(enrollment_id=enrollment_id)

            call_args = mock_repo.update_enrollment_status.call_args
            assert call_args.kwargs["status"] == EnrollmentStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_mark_excluded(
        self,
        service: EnrollmentService,
        mock_repo,
    ):
        """Test marking enrollment as excluded."""
        enrollment_id = uuid4()
        mock_enrollment = MagicMock()
        mock_repo.update_enrollment_status.return_value = mock_enrollment

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            await service.mark_excluded(
                enrollment_id=enrollment_id,
                reason="Protocol violation",
            )

            call_args = mock_repo.update_enrollment_status.call_args
            assert call_args.kwargs["status"] == EnrollmentStatus.EXCLUDED.value

    @pytest.mark.asyncio
    async def test_mark_lost_to_followup(
        self,
        service: EnrollmentService,
        mock_repo,
    ):
        """Test marking enrollment as lost to follow-up."""
        enrollment_id = uuid4()
        mock_enrollment = MagicMock()
        mock_repo.update_enrollment_status.return_value = mock_enrollment

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            await service.mark_lost_to_followup(enrollment_id=enrollment_id)

            call_args = mock_repo.update_enrollment_status.call_args
            assert call_args.kwargs["status"] == EnrollmentStatus.LOST_TO_FOLLOWUP.value


# =============================================================================
# PROTOCOL DEVIATION TESTS
# =============================================================================


class TestProtocolDeviations:
    """Tests for protocol deviation handling."""

    @pytest.mark.asyncio
    async def test_record_minor_deviation(
        self,
        lenient_service: EnrollmentService,
        mock_repo,
    ):
        """Test recording a minor protocol deviation."""
        enrollment_id = uuid4()
        deviation = ProtocolDeviation(
            date=datetime.now(timezone.utc),
            deviation_type="missed_visit",
            severity=DeviationSeverity.MINOR,
            description="Subject missed scheduled follow-up visit",
        )

        mock_enrollment = MagicMock()
        mock_enrollment.protocol_deviations = []
        mock_repo.get_enrollment.return_value = mock_enrollment
        mock_repo.update_protocol_deviations.return_value = mock_enrollment

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            await lenient_service.record_protocol_deviation(
                enrollment_id=enrollment_id,
                deviation=deviation,
            )

            mock_repo.update_protocol_deviations.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_major_deviation_auto_excludes(
        self,
        service: EnrollmentService,
        mock_repo,
    ):
        """Test recording a major deviation auto-excludes when configured."""
        enrollment_id = uuid4()
        deviation = ProtocolDeviation(
            date=datetime.now(timezone.utc),
            deviation_type="unauthorized_treatment",
            severity=DeviationSeverity.MAJOR,
            description="Subject received unauthorized intervention",
        )

        mock_enrollment = MagicMock()
        mock_enrollment.protocol_deviations = []
        mock_repo.get_enrollment.return_value = mock_enrollment
        mock_repo.update_protocol_deviations.return_value = mock_enrollment
        mock_repo.update_enrollment_status.return_value = mock_enrollment

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            await service.record_protocol_deviation(
                enrollment_id=enrollment_id,
                deviation=deviation,
            )

            # Should have called update_enrollment_status to exclude
            assert mock_repo.update_enrollment_status.called

    @pytest.mark.asyncio
    async def test_record_deviation_enrollment_not_found(
        self,
        service: EnrollmentService,
        mock_repo,
    ):
        """Test recording deviation for non-existent enrollment raises error."""
        enrollment_id = uuid4()
        deviation = ProtocolDeviation(
            date=datetime.now(timezone.utc),
            deviation_type="minor_issue",
            severity=DeviationSeverity.MINOR,
            description="Minor protocol deviation",
        )

        mock_repo.get_enrollment.return_value = None

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            with pytest.raises(ValueError, match="not found"):
                await service.record_protocol_deviation(
                    enrollment_id=enrollment_id,
                    deviation=deviation,
                )


# =============================================================================
# ENROLLMENT STATS TESTS
# =============================================================================


class TestEnrollmentStats:
    """Tests for enrollment statistics."""

    @pytest.mark.asyncio
    async def test_get_enrollment_stats(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
        mock_repo,
    ):
        """Test getting enrollment statistics."""
        # Create mock assignments
        mock_assignments = []
        for i in range(10):
            assignment = MagicMock()
            assignment.id = uuid4()
            assignment.variant = "control" if i < 5 else "treatment"
            mock_assignments.append(assignment)

        mock_repo.get_assignments.return_value = mock_assignments

        # Create mock enrollments
        def get_enrollment_by_assignment(assignment_id):
            enrollment = MagicMock()
            enrollment.enrollment_status = EnrollmentStatus.ACTIVE.value
            return enrollment

        mock_repo.get_enrollment_by_assignment = AsyncMock(side_effect=get_enrollment_by_assignment)

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            stats = await service.get_enrollment_stats(experiment_id=experiment_id)

            assert isinstance(stats, EnrollmentStats)
            assert stats.total_assigned == 10
            assert stats.experiment_id == experiment_id

    @pytest.mark.asyncio
    async def test_enrollment_stats_calculates_rates(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
        mock_repo,
    ):
        """Test enrollment stats calculates enrollment and withdrawal rates."""
        # Create 10 assignments
        mock_assignments = [MagicMock() for _ in range(10)]
        for i, a in enumerate(mock_assignments):
            a.id = uuid4()
            a.variant = "control"

        mock_repo.get_assignments.return_value = mock_assignments

        # 8 enrolled, 2 withdrawn
        enrollment_count = [0]

        def get_enrollment(assignment_id):
            enrollment_count[0] += 1
            enrollment = MagicMock()
            if enrollment_count[0] <= 8:
                enrollment.enrollment_status = EnrollmentStatus.ACTIVE.value
            else:
                return None
            return enrollment

        mock_repo.get_enrollment_by_assignment = AsyncMock(side_effect=get_enrollment)

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            stats = await service.get_enrollment_stats(experiment_id=experiment_id)

            assert stats.total_assigned == 10
            assert stats.enrollment_rate == 0.8  # 8/10


# =============================================================================
# BATCH ENROLLMENT TESTS
# =============================================================================


class TestBatchEnrollment:
    """Tests for batch enrollment."""

    @pytest.mark.asyncio
    async def test_batch_enroll_all_eligible(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
        mock_repo,
    ):
        """Test batch enrollment with all eligible units."""
        assignments = [
            {
                "assignment_id": str(uuid4()),
                "experiment_id": experiment_id,
                "unit": {
                    "id": f"unit_{i}",
                    "rx_history_months": 12,
                    "patient_panel_size": 100,
                },
            }
            for i in range(5)
        ]

        mock_enrollment = MagicMock()
        mock_repo.create_enrollment.return_value = mock_enrollment

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            result = await service.batch_enroll(
                assignments=assignments,
                auto_consent=True,
            )

            assert result["enrolled"] == 5
            assert result["ineligible"] == 0
            assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_batch_enroll_with_ineligible(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
        mock_repo,
    ):
        """Test batch enrollment with some ineligible units."""
        strict_criteria = EligibilityCriteria(min_rx_history_months=12)

        assignments = [
            {
                "assignment_id": str(uuid4()),
                "experiment_id": experiment_id,
                "unit": {
                    "id": f"unit_{i}",
                    "rx_history_months": 6 if i < 2 else 12,  # First 2 ineligible
                },
            }
            for i in range(5)
        ]

        mock_enrollment = MagicMock()
        mock_repo.create_enrollment.return_value = mock_enrollment

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            result = await service.batch_enroll(
                assignments=assignments,
                eligibility_criteria=strict_criteria,
                auto_consent=True,
            )

            assert result["enrolled"] == 3
            assert result["ineligible"] == 2

    @pytest.mark.asyncio
    async def test_batch_enroll_returns_details(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
        mock_repo,
    ):
        """Test batch enrollment returns detailed results."""
        assignments = [
            {
                "assignment_id": str(uuid4()),
                "experiment_id": experiment_id,
                "unit": {"id": "unit_1"},
            }
        ]

        mock_enrollment = MagicMock()
        mock_repo.create_enrollment.return_value = mock_enrollment

        with patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_repo):
            result = await service.batch_enroll(
                assignments=assignments,
                auto_consent=True,
            )

            assert "details" in result
            assert len(result["details"]) == 1
            assert result["details"][0]["status"] == "enrolled"


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunction:
    """Tests for factory function."""

    def test_get_enrollment_service_default(self):
        """Test factory creates service with default config."""
        service = get_enrollment_service()

        assert isinstance(service, EnrollmentService)
        assert service.config is not None

    def test_get_enrollment_service_custom_config(self):
        """Test factory creates service with custom config."""
        config = EnrollmentConfig(max_minor_deviations=10)

        service = get_enrollment_service(config=config)

        assert service.config.max_minor_deviations == 10


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_eligibility_check_missing_fields(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
    ):
        """Test eligibility check handles missing fields gracefully."""
        unit = {"id": "unit_with_missing_fields"}  # No other fields

        result = await service.check_eligibility(
            experiment_id=experiment_id,
            unit=unit,
        )

        # Should still return a result (with defaults)
        assert isinstance(result, EligibilityResult)

    @pytest.mark.asyncio
    async def test_eligibility_check_with_none_values(
        self,
        service: EnrollmentService,
        experiment_id: UUID,
    ):
        """Test eligibility check handles None values."""
        unit = {
            "id": "unit_with_nones",
            "rx_history_months": None,
            "patient_panel_size": None,
        }

        result = await service.check_eligibility(
            experiment_id=experiment_id,
            unit=unit,
        )

        # Should handle None gracefully
        assert isinstance(result, EligibilityResult)

    def test_protocol_deviation_dataclass(self):
        """Test ProtocolDeviation dataclass creation."""
        deviation = ProtocolDeviation(
            date=datetime.now(timezone.utc),
            deviation_type="test_deviation",
            severity=DeviationSeverity.MINOR,
            description="Test description",
            corrective_action="Test action",
        )

        assert deviation.deviation_type == "test_deviation"
        assert deviation.severity == DeviationSeverity.MINOR
        assert deviation.corrective_action == "Test action"
