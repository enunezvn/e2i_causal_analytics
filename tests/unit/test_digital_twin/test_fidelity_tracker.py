"""
Unit tests for FidelityTracker.

Tests fidelity tracking, validation, grading, and degradation detection.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.digital_twin.fidelity_tracker import FidelityTracker
from src.digital_twin.models.simulation_models import (
    EffectHeterogeneity,
    FidelityGrade,
    FidelityRecord,
    InterventionConfig,
    PopulationFilter,
    SimulationRecommendation,
    SimulationResult,
    SimulationStatus,
)


class TestFidelityTrackerInit:
    """Tests for FidelityTracker initialization."""

    def test_init_without_repository(self):
        """Test initialization without repository."""
        tracker = FidelityTracker()

        assert tracker.repository is None
        assert tracker.records == {}
        assert tracker.model_fidelity_cache == {}

    def test_init_with_repository(self):
        """Test initialization with repository."""
        mock_repo = MagicMock()
        tracker = FidelityTracker(repository=mock_repo)

        assert tracker.repository is mock_repo
        assert tracker.records == {}

    def test_grade_thresholds_defined(self):
        """Test that grade thresholds are properly defined."""
        assert FidelityTracker.GRADE_THRESHOLDS[FidelityGrade.EXCELLENT] == 0.10
        assert FidelityTracker.GRADE_THRESHOLDS[FidelityGrade.GOOD] == 0.20
        assert FidelityTracker.GRADE_THRESHOLDS[FidelityGrade.FAIR] == 0.35
        assert FidelityTracker.GRADE_THRESHOLDS[FidelityGrade.POOR] == float("inf")

    def test_degradation_thresholds_defined(self):
        """Test that degradation detection thresholds are defined."""
        assert FidelityTracker.DEGRADATION_LOOKBACK_DAYS == 90
        assert FidelityTracker.DEGRADATION_THRESHOLD == 0.10
        assert FidelityTracker.MIN_VALIDATIONS_FOR_ALERT == 5


class TestRecordPrediction:
    """Tests for record_prediction method."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker without repository."""
        return FidelityTracker()

    @pytest.fixture
    def simulation_result(self):
        """Create a sample simulation result."""
        return SimulationResult(
            model_id=uuid4(),
            intervention_config=InterventionConfig(
                intervention_type="email_campaign",
                channel="email",
            ),
            twin_count=1000,
            simulated_ate=0.08,
            simulated_ci_lower=0.05,
            simulated_ci_upper=0.11,
            simulated_std_error=0.015,
            status=SimulationStatus.COMPLETED,
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Effect is significant",
            simulation_confidence=0.85,
            execution_time_ms=150,
        )

    def test_record_prediction_creates_fidelity_record(self, tracker, simulation_result):
        """Test that record_prediction creates a FidelityRecord."""
        record = tracker.record_prediction(simulation_result)

        assert isinstance(record, FidelityRecord)
        assert record.simulation_id == simulation_result.simulation_id
        assert record.simulated_ate == simulation_result.simulated_ate
        assert record.simulated_ci_lower == simulation_result.simulated_ci_lower
        assert record.simulated_ci_upper == simulation_result.simulated_ci_upper

    def test_record_prediction_stores_record(self, tracker, simulation_result):
        """Test that record is stored in tracker.records."""
        record = tracker.record_prediction(simulation_result)

        assert record.tracking_id in tracker.records
        assert tracker.records[record.tracking_id] is record

    def test_record_prediction_with_repository(self, simulation_result):
        """Test that record is saved to repository when available."""
        mock_repo = MagicMock()
        tracker = FidelityTracker(repository=mock_repo)

        record = tracker.record_prediction(simulation_result)

        mock_repo.save_fidelity_record.assert_called_once_with(record)

    def test_multiple_records_stored(self, tracker, simulation_result):
        """Test storing multiple prediction records."""
        record1 = tracker.record_prediction(simulation_result)

        # Create second result
        result2 = SimulationResult(
            model_id=uuid4(),
            intervention_config=InterventionConfig(
                intervention_type="call_frequency_increase",
                channel="phone",
            ),
            twin_count=500,
            simulated_ate=0.12,
            simulated_ci_lower=0.08,
            simulated_ci_upper=0.16,
            simulated_std_error=0.02,
            status=SimulationStatus.COMPLETED,
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Effect is positive",
            simulation_confidence=0.80,
            execution_time_ms=120,
        )
        record2 = tracker.record_prediction(result2)

        assert len(tracker.records) == 2
        assert record1.tracking_id in tracker.records
        assert record2.tracking_id in tracker.records


class TestValidate:
    """Tests for validate method."""

    @pytest.fixture
    def tracker_with_record(self):
        """Create tracker with a recorded prediction."""
        tracker = FidelityTracker()

        result = SimulationResult(
            model_id=uuid4(),
            intervention_config=InterventionConfig(
                intervention_type="email_campaign",
                channel="email",
            ),
            twin_count=1000,
            simulated_ate=0.10,
            simulated_ci_lower=0.06,
            simulated_ci_upper=0.14,
            simulated_std_error=0.02,
            status=SimulationStatus.COMPLETED,
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Effect is significant",
            simulation_confidence=0.85,
            execution_time_ms=150,
        )
        record = tracker.record_prediction(result)

        return tracker, result.simulation_id, record

    def test_validate_updates_record_with_actuals(self, tracker_with_record):
        """Test that validate updates record with actual results."""
        tracker, simulation_id, original_record = tracker_with_record

        validated_record = tracker.validate(
            simulation_id=simulation_id,
            actual_ate=0.09,
            actual_ci=(0.05, 0.13),
            actual_sample_size=800,
        )

        assert validated_record.actual_ate == 0.09
        assert validated_record.actual_ci_lower == 0.05
        assert validated_record.actual_ci_upper == 0.13
        assert validated_record.actual_sample_size == 800

    def test_validate_calculates_prediction_error(self, tracker_with_record):
        """Test that validation calculates prediction error."""
        tracker, simulation_id, _ = tracker_with_record

        validated_record = tracker.validate(
            simulation_id=simulation_id,
            actual_ate=0.09,  # Predicted was 0.10, error = 0.01/0.09 ≈ 11%
        )

        # Error should be calculated
        assert validated_record.prediction_error is not None
        assert validated_record.absolute_error is not None

    def test_validate_assigns_fidelity_grade(self, tracker_with_record):
        """Test that validation assigns fidelity grade."""
        tracker, simulation_id, _ = tracker_with_record

        validated_record = tracker.validate(
            simulation_id=simulation_id,
            actual_ate=0.09,  # Close to predicted 0.10, should be EXCELLENT
        )

        assert validated_record.fidelity_grade != FidelityGrade.UNVALIDATED

    def test_validate_with_experiment_id(self, tracker_with_record):
        """Test validation with experiment ID."""
        tracker, simulation_id, _ = tracker_with_record
        experiment_id = uuid4()

        validated_record = tracker.validate(
            simulation_id=simulation_id,
            actual_ate=0.10,
            actual_experiment_id=experiment_id,
        )

        assert validated_record.actual_experiment_id == experiment_id

    def test_validate_with_notes_and_validator(self, tracker_with_record):
        """Test validation with notes and validator."""
        tracker, simulation_id, _ = tracker_with_record

        validated_record = tracker.validate(
            simulation_id=simulation_id,
            actual_ate=0.10,
            notes="Experiment completed successfully",
            validated_by="analyst@company.com",
        )

        assert validated_record.validation_notes == "Experiment completed successfully"
        assert validated_record.validated_by == "analyst@company.com"

    def test_validate_with_confounding_factors(self, tracker_with_record):
        """Test validation with confounding factors."""
        tracker, simulation_id, _ = tracker_with_record
        factors = ["seasonality", "competitor_launch"]

        validated_record = tracker.validate(
            simulation_id=simulation_id,
            actual_ate=0.10,
            confounding_factors=factors,
        )

        assert validated_record.confounding_factors == factors

    def test_validate_nonexistent_simulation_raises_error(self):
        """Test that validating non-existent simulation raises ValueError."""
        tracker = FidelityTracker()

        with pytest.raises(ValueError, match="No fidelity record found"):
            tracker.validate(
                simulation_id=uuid4(),
                actual_ate=0.10,
            )

    def test_validate_with_repository_updates(self, tracker_with_record):
        """Test that validation updates repository when available."""
        tracker, simulation_id, _ = tracker_with_record
        mock_repo = MagicMock()
        tracker.repository = mock_repo

        validated_record = tracker.validate(
            simulation_id=simulation_id,
            actual_ate=0.10,
        )

        mock_repo.update_fidelity_record.assert_called_once_with(validated_record)


class TestGetGrade:
    """Tests for get_grade method."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker."""
        return FidelityTracker()

    def test_get_grade_excellent(self, tracker):
        """Test EXCELLENT grade for small errors."""
        assert tracker.get_grade(0.05) == FidelityGrade.EXCELLENT
        assert tracker.get_grade(-0.05) == FidelityGrade.EXCELLENT
        assert tracker.get_grade(0.09) == FidelityGrade.EXCELLENT

    def test_get_grade_good(self, tracker):
        """Test GOOD grade for moderate errors."""
        assert tracker.get_grade(0.10) == FidelityGrade.GOOD
        assert tracker.get_grade(0.15) == FidelityGrade.GOOD
        assert tracker.get_grade(-0.19) == FidelityGrade.GOOD

    def test_get_grade_fair(self, tracker):
        """Test FAIR grade for larger errors."""
        assert tracker.get_grade(0.20) == FidelityGrade.FAIR
        assert tracker.get_grade(0.30) == FidelityGrade.FAIR
        assert tracker.get_grade(-0.34) == FidelityGrade.FAIR

    def test_get_grade_poor(self, tracker):
        """Test POOR grade for large errors."""
        assert tracker.get_grade(0.35) == FidelityGrade.POOR
        assert tracker.get_grade(0.50) == FidelityGrade.POOR
        assert tracker.get_grade(-0.80) == FidelityGrade.POOR

    def test_get_grade_boundary_values(self, tracker):
        """Test boundary values for grade thresholds."""
        # Just under threshold
        assert tracker.get_grade(0.099) == FidelityGrade.EXCELLENT
        assert tracker.get_grade(0.199) == FidelityGrade.GOOD
        assert tracker.get_grade(0.349) == FidelityGrade.FAIR


class TestGetRecord:
    """Tests for get_record and get_simulation_record methods."""

    @pytest.fixture
    def tracker_with_records(self):
        """Create tracker with multiple records."""
        tracker = FidelityTracker()

        result1 = SimulationResult(
            model_id=uuid4(),
            intervention_config=InterventionConfig(
                intervention_type="email_campaign",
                channel="email",
            ),
            twin_count=1000,
            simulated_ate=0.08,
            simulated_ci_lower=0.05,
            simulated_ci_upper=0.11,
            simulated_std_error=0.015,
            status=SimulationStatus.COMPLETED,
            recommendation=SimulationRecommendation.DEPLOY,
            recommendation_rationale="Effect is significant",
            simulation_confidence=0.85,
            execution_time_ms=150,
        )
        record1 = tracker.record_prediction(result1)

        return tracker, record1, result1.simulation_id

    def test_get_record_by_tracking_id(self, tracker_with_records):
        """Test retrieving record by tracking ID."""
        tracker, record, _ = tracker_with_records

        retrieved = tracker.get_record(record.tracking_id)

        assert retrieved is record

    def test_get_record_nonexistent(self, tracker_with_records):
        """Test getting non-existent tracking ID returns None."""
        tracker, _, _ = tracker_with_records

        retrieved = tracker.get_record(uuid4())

        assert retrieved is None

    def test_get_simulation_record(self, tracker_with_records):
        """Test retrieving record by simulation ID."""
        tracker, record, simulation_id = tracker_with_records

        retrieved = tracker.get_simulation_record(simulation_id)

        assert retrieved is record

    def test_get_simulation_record_nonexistent(self, tracker_with_records):
        """Test getting non-existent simulation ID returns None."""
        tracker, _, _ = tracker_with_records

        retrieved = tracker.get_simulation_record(uuid4())

        assert retrieved is None


class TestFidelityScore:
    """Tests for _calculate_fidelity_score method."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker."""
        return FidelityTracker()

    def test_fidelity_score_with_no_errors(self, tracker):
        """Test fidelity score with no error data."""
        score = tracker._calculate_fidelity_score([], [])

        assert score == 0.5  # Unknown/default score

    def test_fidelity_score_perfect(self, tracker):
        """Test fidelity score with perfect predictions."""
        errors = [0.0, 0.0, 0.0]
        ci_coverages = [True, True, True]

        score = tracker._calculate_fidelity_score(errors, ci_coverages)

        assert score == 1.0  # Perfect score

    def test_fidelity_score_high_error(self, tracker):
        """Test fidelity score with high errors."""
        errors = [0.5, 0.6, 0.7]  # Very high errors
        ci_coverages = [False, False, False]

        score = tracker._calculate_fidelity_score(errors, ci_coverages)

        assert score < 0.5  # Low score due to high error

    def test_fidelity_score_moderate(self, tracker):
        """Test fidelity score with moderate errors."""
        errors = [0.10, 0.15, 0.12]
        ci_coverages = [True, True, False]

        score = tracker._calculate_fidelity_score(errors, ci_coverages)

        assert 0.5 < score < 1.0

    def test_fidelity_score_bounds(self, tracker):
        """Test that fidelity score is bounded 0-1."""
        # Very high errors
        score_low = tracker._calculate_fidelity_score([1.0, 1.5, 2.0], [])
        assert 0.0 <= score_low <= 1.0

        # Very low errors
        score_high = tracker._calculate_fidelity_score([0.0, 0.0], [True, True])
        assert 0.0 <= score_high <= 1.0


class TestDegradationDetection:
    """Tests for _check_degradation method."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker."""
        return FidelityTracker()

    def _create_validated_record(
        self,
        prediction_error: float,
        validated_at: datetime,
    ) -> FidelityRecord:
        """Helper to create a validated fidelity record."""
        record = FidelityRecord(
            simulation_id=uuid4(),
            simulated_ate=0.10,
            simulated_ci_lower=0.06,
            simulated_ci_upper=0.14,
        )
        record.actual_ate = 0.10 * (1 + prediction_error)
        record.prediction_error = prediction_error
        record.validated_at = validated_at
        record.fidelity_grade = FidelityGrade.GOOD
        return record

    def test_no_degradation_with_few_records(self, tracker):
        """Test no degradation alert with insufficient records."""
        records = [
            self._create_validated_record(0.05, datetime.now(timezone.utc))
            for _ in range(5)
        ]

        result = tracker._check_degradation(records)

        assert result is False  # Not enough records

    def test_no_degradation_with_stable_errors(self, tracker):
        """Test no degradation when errors are stable."""
        now = datetime.now(timezone.utc)
        records = []

        # Create 12 records with stable errors
        for i in range(12):
            records.append(
                self._create_validated_record(
                    prediction_error=0.10,  # Consistent 10% error
                    validated_at=now - timedelta(days=i * 5),
                )
            )

        result = tracker._check_degradation(records)

        assert result == False

    def test_degradation_detected_with_increasing_errors(self, tracker):
        """Test degradation detected when recent errors are higher."""
        now = datetime.now(timezone.utc)
        records = []

        # Older records with low error
        for i in range(6):
            records.append(
                self._create_validated_record(
                    prediction_error=0.05,  # 5% error historically
                    validated_at=now - timedelta(days=60 + i * 5),
                )
            )

        # Recent records with higher error (> 10% increase)
        for i in range(6):
            records.append(
                self._create_validated_record(
                    prediction_error=0.25,  # 25% error recently
                    validated_at=now - timedelta(days=i * 5),
                )
            )

        result = tracker._check_degradation(records)

        assert result == True


class TestModelFidelityReport:
    """Tests for get_model_fidelity_report method."""

    @pytest.fixture
    def tracker_with_validated_records(self):
        """Create tracker with validated records."""
        tracker = FidelityTracker()
        model_id = uuid4()

        # Add several validated records
        now = datetime.now(timezone.utc)
        for i in range(5):
            result = SimulationResult(
                model_id=model_id,
                intervention_config=InterventionConfig(
                    intervention_type="email_campaign",
                    channel="email",
                ),
                twin_count=1000,
                simulated_ate=0.10,
                simulated_ci_lower=0.06,
                simulated_ci_upper=0.14,
                simulated_std_error=0.02,
                status=SimulationStatus.COMPLETED,
                recommendation=SimulationRecommendation.DEPLOY,
                recommendation_rationale="Effect is significant",
                simulation_confidence=0.85,
                execution_time_ms=150,
            )
            record = tracker.record_prediction(result)

            # Manually validate (normally done through validate method)
            record.actual_ate = 0.10 + (i * 0.01)  # Varying actuals
            record.calculate_fidelity()
            record.validated_at = now - timedelta(days=i * 10)

        return tracker, model_id

    def test_report_with_no_records(self):
        """Test report with no validated records."""
        tracker = FidelityTracker()
        model_id = uuid4()

        report = tracker.get_model_fidelity_report(model_id)

        assert report["validation_count"] == 0
        assert "message" in report

    def test_report_includes_metrics(self, tracker_with_validated_records):
        """Test that report includes fidelity metrics."""
        tracker, model_id = tracker_with_validated_records

        report = tracker.get_model_fidelity_report(model_id)

        assert "metrics" in report
        assert "mean_absolute_error" in report["metrics"]
        assert "median_absolute_error" in report["metrics"]
        assert "max_error" in report["metrics"]
        assert "ci_coverage_rate" in report["metrics"]

    def test_report_includes_grade_distribution(self, tracker_with_validated_records):
        """Test that report includes grade distribution."""
        tracker, model_id = tracker_with_validated_records

        report = tracker.get_model_fidelity_report(model_id)

        assert "grade_distribution" in report
        assert isinstance(report["grade_distribution"], dict)

    def test_report_includes_fidelity_score(self, tracker_with_validated_records):
        """Test that report includes fidelity score."""
        tracker, model_id = tracker_with_validated_records

        report = tracker.get_model_fidelity_report(model_id)

        assert "fidelity_score" in report
        assert 0 <= report["fidelity_score"] <= 1

    def test_report_caching(self, tracker_with_validated_records):
        """Test that report is cached."""
        tracker, model_id = tracker_with_validated_records

        # First call
        report1 = tracker.get_model_fidelity_report(model_id)

        # Second call should use cache
        report2 = tracker.get_model_fidelity_report(model_id)

        # Should be same object (cached)
        assert report1["computed_at"] == report2["computed_at"]


class TestCheckDegradationAlerts:
    """Tests for check_degradation_alerts method."""

    def test_no_alert_when_no_degradation(self):
        """Test no alert returned when model is healthy."""
        tracker = FidelityTracker()
        model_id = uuid4()

        alert = tracker.check_degradation_alerts(model_id)

        # With no records, no alert
        assert alert is None

    def test_alert_structure_when_degradation(self):
        """Test alert structure when degradation is detected."""
        tracker = FidelityTracker()
        model_id = uuid4()

        # Mock the report to show degradation
        with patch.object(tracker, "get_model_fidelity_report") as mock_report:
            mock_report.return_value = {
                "degradation_alert": True,
                "metrics": {"mean_absolute_error": 0.30},
            }

            alert = tracker.check_degradation_alerts(model_id)

            assert alert is not None
            assert alert["alert_type"] == "fidelity_degradation"
            assert alert["model_id"] == str(model_id)
            assert "recommendation" in alert


class TestGetStatistics:
    """Tests for get_statistics method."""

    def test_statistics_empty_tracker(self):
        """Test statistics with no records."""
        tracker = FidelityTracker()

        stats = tracker.get_statistics()

        assert stats["total_predictions"] == 0
        assert stats["validated_predictions"] == 0
        assert stats["validation_rate"] == 0

    def test_statistics_with_records(self):
        """Test statistics with mixed records."""
        tracker = FidelityTracker()

        # Add 3 predictions
        for i in range(3):
            result = SimulationResult(
                model_id=uuid4(),
                intervention_config=InterventionConfig(
                    intervention_type="email_campaign",
                    channel="email",
                ),
                twin_count=1000,
                simulated_ate=0.10,
                simulated_ci_lower=0.06,
                simulated_ci_upper=0.14,
                simulated_std_error=0.02,
                status=SimulationStatus.COMPLETED,
                recommendation=SimulationRecommendation.DEPLOY,
                recommendation_rationale="Effect is significant",
                simulation_confidence=0.85,
                execution_time_ms=150,
            )
            record = tracker.record_prediction(result)

            # Validate only the first one
            if i == 0:
                record.actual_ate = 0.09
                record.calculate_fidelity()

        stats = tracker.get_statistics()

        assert stats["total_predictions"] == 3
        assert stats["validated_predictions"] == 1
        assert stats["validation_rate"] == pytest.approx(1 / 3)

    def test_statistics_grade_distribution(self):
        """Test that statistics include grade distribution."""
        tracker = FidelityTracker()

        # Add and validate records with different grades
        for error in [0.05, 0.15, 0.30]:  # EXCELLENT, GOOD, FAIR
            result = SimulationResult(
                model_id=uuid4(),
                intervention_config=InterventionConfig(
                    intervention_type="email_campaign",
                    channel="email",
                ),
                twin_count=1000,
                simulated_ate=0.10,
                simulated_ci_lower=0.06,
                simulated_ci_upper=0.14,
                simulated_std_error=0.02,
                status=SimulationStatus.COMPLETED,
                recommendation=SimulationRecommendation.DEPLOY,
                recommendation_rationale="Effect is significant",
                simulation_confidence=0.85,
                execution_time_ms=150,
            )
            record = tracker.record_prediction(result)
            record.actual_ate = 0.10 * (1 + error)
            record.calculate_fidelity()

        stats = tracker.get_statistics()

        assert "grade_distribution" in stats
        assert len(stats["grade_distribution"]) > 0
