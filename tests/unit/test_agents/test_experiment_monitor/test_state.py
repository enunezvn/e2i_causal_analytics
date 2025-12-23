"""Tests for Experiment Monitor State definitions.

Tests cover:
- TypedDict structure validation
- Type alias correctness
- Default values and optional fields
- State field groups (input, config, outputs, metadata)
"""

import pytest
from datetime import datetime, timezone

from src.agents.experiment_monitor.state import (
    AlertSeverity,
    AlertType,
    EnrollmentIssue,
    ErrorDetails,
    ExperimentMonitorState,
    ExperimentSummary,
    FidelityIssue,
    HealthStatus,
    InterimTrigger,
    MonitorAlert,
    MonitorStatus,
    SRMIssue,
)


class TestTypeAliases:
    """Tests for Literal type aliases."""

    def test_monitor_status_values(self):
        """Test MonitorStatus literal values."""
        valid_statuses = ["pending", "checking", "analyzing", "alerting", "completed", "failed"]
        # TypedDict doesn't enforce at runtime, but we can test the type
        assert all(isinstance(s, str) for s in valid_statuses)

    def test_health_status_values(self):
        """Test HealthStatus literal values."""
        valid_statuses = ["healthy", "warning", "critical", "unknown"]
        assert all(isinstance(s, str) for s in valid_statuses)

    def test_alert_severity_values(self):
        """Test AlertSeverity literal values."""
        valid_severities = ["info", "warning", "critical"]
        assert all(isinstance(s, str) for s in valid_severities)

    def test_alert_type_values(self):
        """Test AlertType literal values."""
        valid_types = ["srm", "enrollment", "stale_data", "fidelity", "interim_trigger"]
        assert all(isinstance(s, str) for s in valid_types)


class TestExperimentSummary:
    """Tests for ExperimentSummary TypedDict."""

    def test_create_valid_summary(self, sample_summary_healthy):
        """Test creating a valid ExperimentSummary."""
        summary = sample_summary_healthy
        assert summary["experiment_id"] == "exp-healthy"
        assert summary["name"] == "Healthy Test Experiment"
        assert summary["status"] == "running"
        assert summary["health_status"] == "healthy"
        assert summary["days_running"] == 7
        assert summary["total_enrolled"] == 500
        assert isinstance(summary["enrollment_rate"], float)
        assert isinstance(summary["current_information_fraction"], float)

    def test_summary_all_health_statuses(self):
        """Test ExperimentSummary with all health status values."""
        for status in ["healthy", "warning", "critical", "unknown"]:
            summary = ExperimentSummary(
                experiment_id="test",
                name="Test",
                status="running",
                health_status=status,
                days_running=1,
                total_enrolled=100,
                enrollment_rate=100.0,
                current_information_fraction=0.1,
            )
            assert summary["health_status"] == status


class TestSRMIssue:
    """Tests for SRMIssue TypedDict."""

    def test_create_valid_srm_issue(self, sample_srm_issue):
        """Test creating a valid SRMIssue."""
        issue = sample_srm_issue
        assert issue["experiment_id"] == "exp-warning"
        assert issue["detected"] is True
        assert isinstance(issue["p_value"], float)
        assert isinstance(issue["chi_squared"], float)
        assert isinstance(issue["expected_ratio"], dict)
        assert isinstance(issue["actual_counts"], dict)
        assert issue["severity"] in ["info", "warning", "critical"]

    def test_srm_issue_p_value_range(self):
        """Test SRMIssue with various p-values."""
        for p_value in [0.0001, 0.001, 0.05, 0.5, 1.0]:
            issue = SRMIssue(
                experiment_id="test",
                detected=p_value < 0.001,
                p_value=p_value,
                chi_squared=10.0,
                expected_ratio={"control": 0.5, "treatment": 0.5},
                actual_counts={"control": 100, "treatment": 100},
                severity="info",
            )
            assert 0 <= issue["p_value"] <= 1.0

    def test_srm_issue_severity_levels(self):
        """Test SRMIssue with different severity levels."""
        for severity in ["info", "warning", "critical"]:
            issue = SRMIssue(
                experiment_id="test",
                detected=True,
                p_value=0.0001,
                chi_squared=20.0,
                expected_ratio={"control": 0.5, "treatment": 0.5},
                actual_counts={"control": 150, "treatment": 50},
                severity=severity,
            )
            assert issue["severity"] == severity


class TestEnrollmentIssue:
    """Tests for EnrollmentIssue TypedDict."""

    def test_create_valid_enrollment_issue(self, sample_enrollment_issue):
        """Test creating a valid EnrollmentIssue."""
        issue = sample_enrollment_issue
        assert issue["experiment_id"] == "exp-critical"
        assert isinstance(issue["current_rate"], float)
        assert isinstance(issue["expected_rate"], float)
        assert isinstance(issue["days_below_threshold"], int)
        assert issue["severity"] in ["info", "warning", "critical"]

    def test_enrollment_issue_rates(self):
        """Test EnrollmentIssue with various rate comparisons."""
        # Current rate below expected
        issue = EnrollmentIssue(
            experiment_id="test",
            current_rate=2.0,
            expected_rate=5.0,
            days_below_threshold=7,
            severity="warning",
        )
        assert issue["current_rate"] < issue["expected_rate"]


class TestFidelityIssue:
    """Tests for FidelityIssue TypedDict."""

    def test_create_valid_fidelity_issue(self, sample_fidelity_issue):
        """Test creating a valid FidelityIssue."""
        issue = sample_fidelity_issue
        assert issue["experiment_id"] == "exp-fidelity"
        assert issue["twin_simulation_id"] == "sim-001"
        assert isinstance(issue["predicted_effect"], float)
        assert isinstance(issue["actual_effect"], float)
        assert isinstance(issue["prediction_error"], float)
        assert isinstance(issue["calibration_needed"], bool)

    def test_fidelity_issue_calibration_states(self):
        """Test FidelityIssue with and without calibration needed."""
        for calibration_needed in [True, False]:
            issue = FidelityIssue(
                experiment_id="test",
                twin_simulation_id="sim-001",
                predicted_effect=0.15,
                actual_effect=0.12,
                prediction_error=0.2,
                calibration_needed=calibration_needed,
                severity="warning" if calibration_needed else "info",
            )
            assert issue["calibration_needed"] == calibration_needed


class TestInterimTrigger:
    """Tests for InterimTrigger TypedDict."""

    def test_create_valid_interim_trigger(self, sample_interim_trigger):
        """Test creating a valid InterimTrigger."""
        trigger = sample_interim_trigger
        assert trigger["experiment_id"] == "exp-healthy"
        assert trigger["analysis_number"] == 2
        assert isinstance(trigger["information_fraction"], float)
        assert trigger["milestone_reached"] == "50%"
        assert trigger["triggered"] is True

    def test_interim_trigger_milestones(self):
        """Test InterimTrigger with different milestones."""
        milestones = ["25%", "50%", "75%"]
        for i, milestone in enumerate(milestones, 1):
            trigger = InterimTrigger(
                experiment_id="test",
                analysis_number=i,
                information_fraction=0.25 * i,
                milestone_reached=milestone,
                triggered=True,
            )
            assert trigger["milestone_reached"] == milestone
            assert trigger["analysis_number"] == i


class TestMonitorAlert:
    """Tests for MonitorAlert TypedDict."""

    def test_create_valid_monitor_alert(self, sample_alert_srm):
        """Test creating a valid MonitorAlert."""
        alert = sample_alert_srm
        assert alert["alert_id"] == "alert-001"
        assert alert["alert_type"] == "srm"
        assert alert["severity"] == "critical"
        assert alert["experiment_id"] == "exp-warning"
        assert alert["experiment_name"] == "Warning Test Experiment"
        assert isinstance(alert["message"], str)
        assert isinstance(alert["details"], dict)
        assert isinstance(alert["recommended_action"], str)
        assert isinstance(alert["timestamp"], str)

    def test_alert_all_types(self):
        """Test MonitorAlert with all alert types."""
        alert_types = ["srm", "enrollment", "stale_data", "fidelity", "interim_trigger"]
        for alert_type in alert_types:
            alert = MonitorAlert(
                alert_id="test-alert",
                alert_type=alert_type,
                severity="info",
                experiment_id="test",
                experiment_name="Test Experiment",
                message="Test message",
                details={},
                recommended_action="Take action",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            assert alert["alert_type"] == alert_type


class TestErrorDetails:
    """Tests for ErrorDetails TypedDict."""

    def test_create_valid_error_details(self):
        """Test creating valid ErrorDetails."""
        error = ErrorDetails(
            node="health_checker",
            error="Database connection failed",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        assert error["node"] == "health_checker"
        assert "Database" in error["error"]
        assert isinstance(error["timestamp"], str)

    def test_error_details_all_nodes(self):
        """Test ErrorDetails for each node type."""
        nodes = ["health_checker", "srm_detector", "interim_analyzer", "alert_generator"]
        for node in nodes:
            error = ErrorDetails(
                node=node, error=f"Error in {node}", timestamp=datetime.now(timezone.utc).isoformat()
            )
            assert error["node"] == node


class TestExperimentMonitorState:
    """Tests for ExperimentMonitorState TypedDict."""

    def test_create_minimal_state(self, base_monitor_state):
        """Test creating a minimal valid state."""
        state = base_monitor_state
        # Input fields
        assert state["query"] == "Check experiment health"
        assert state["check_all_active"] is True
        # Configuration
        assert state["srm_threshold"] == 0.001
        assert state["enrollment_threshold"] == 5.0
        # Status
        assert state["status"] == "pending"

    def test_state_field_groups(self, base_monitor_state):
        """Test that all field groups are present."""
        state = base_monitor_state

        # Input fields (3)
        assert "query" in state
        assert "experiment_ids" in state
        assert "check_all_active" in state

        # Configuration fields (4)
        assert "srm_threshold" in state
        assert "enrollment_threshold" in state
        assert "fidelity_threshold" in state
        assert "check_interim" in state

        # Monitoring outputs (4)
        assert "experiments" in state
        assert "srm_issues" in state
        assert "enrollment_issues" in state
        assert "fidelity_issues" in state

        # Trigger outputs (1)
        assert "interim_triggers" in state

        # Alerts (1)
        assert "alerts" in state

        # Summary (2)
        assert "monitor_summary" in state
        assert "recommended_actions" in state

        # Execution metadata (2)
        assert "check_latency_ms" in state
        assert "experiments_checked" in state

        # Error handling (3)
        assert "errors" in state
        assert "warnings" in state
        assert "status" in state

    def test_state_with_all_issues(self, state_with_all_issues):
        """Test state with all types of issues populated."""
        state = state_with_all_issues

        assert len(state["experiments"]) == 3
        assert len(state["srm_issues"]) == 1
        assert len(state["enrollment_issues"]) == 1
        assert len(state["fidelity_issues"]) == 1
        assert len(state["interim_triggers"]) == 1
        assert state["status"] == "analyzing"

    def test_state_status_transitions(self):
        """Test valid status transitions."""
        statuses = ["pending", "checking", "analyzing", "alerting", "completed"]
        for status in statuses:
            state = ExperimentMonitorState(
                query="test",
                check_all_active=True,
                srm_threshold=0.001,
                enrollment_threshold=5.0,
                fidelity_threshold=0.2,
                check_interim=True,
                errors=[],
                warnings=[],
                status=status,
            )
            assert state["status"] == status

    def test_state_copy_preserves_data(self, state_with_all_issues):
        """Test that copying state preserves all data."""
        original = state_with_all_issues
        copied = original.copy()

        assert copied["experiments"] == original["experiments"]
        assert copied["srm_issues"] == original["srm_issues"]
        assert copied["status"] == original["status"]

        # Modify copy shouldn't affect original (shallow copy for lists)
        copied["status"] = "completed"
        assert original["status"] == "analyzing"


class TestStateHelpers:
    """Tests for conftest helper functions."""

    def test_create_experiment_summary_helper(self):
        """Test the create_experiment_summary helper."""
        from tests.unit.test_agents.test_experiment_monitor.conftest import create_experiment_summary

        summary = create_experiment_summary(
            experiment_id="test-exp",
            name="Custom Experiment",
            health_status="warning",
            days_running=14,
        )
        assert summary["experiment_id"] == "test-exp"
        assert summary["name"] == "Custom Experiment"
        assert summary["health_status"] == "warning"
        assert summary["days_running"] == 14

    def test_create_srm_issue_helper(self):
        """Test the create_srm_issue helper."""
        from tests.unit.test_agents.test_experiment_monitor.conftest import create_srm_issue

        issue = create_srm_issue(
            experiment_id="test-exp", p_value=0.0005, severity="warning"
        )
        assert issue["experiment_id"] == "test-exp"
        assert issue["p_value"] == 0.0005
        assert issue["severity"] == "warning"

    def test_create_monitor_state_helper(self):
        """Test the create_monitor_state helper."""
        from tests.unit.test_agents.test_experiment_monitor.conftest import (
            create_monitor_state,
            create_experiment_summary,
        )

        summary = create_experiment_summary("exp-1")
        state = create_monitor_state(
            experiments=[summary],
            status="checking",
            query="Custom query",
        )
        assert len(state["experiments"]) == 1
        assert state["status"] == "checking"
        assert state["query"] == "Custom query"
