"""Tests for Alert Generator Node.

Tests cover:
- Node initialization
- Alert generation for SRM, enrollment, interim, fidelity issues
- Summary creation
- Recommendation generation
- Edge cases and error handling
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode
from src.agents.experiment_monitor.state import ExperimentMonitorState


class TestAlertGeneratorNodeInit:
    """Tests for AlertGeneratorNode initialization."""

    def test_node_initialization(self):
        """Test that node initializes correctly."""
        node = AlertGeneratorNode()
        assert node is not None

    def test_multiple_node_instances(self):
        """Test creating multiple node instances."""
        node1 = AlertGeneratorNode()
        node2 = AlertGeneratorNode()
        assert node1 is not node2


class TestAlertGeneratorExecute:
    """Tests for execute method."""

    @pytest.fixture
    def base_state(self):
        """Base state for testing."""
        return {
            "query": "Check experiments",
            "check_all_active": True,
            "experiment_ids": [],
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "experiments": [
                {
                    "experiment_id": "exp-001",
                    "name": "Test Experiment",
                    "status": "running",
                    "health_status": "healthy",
                    "days_running": 7,
                    "total_enrolled": 500,
                    "enrollment_rate": 71.43,
                    "current_information_fraction": 0.5,
                }
            ],
            "srm_issues": [],
            "enrollment_issues": [],
            "fidelity_issues": [],
            "interim_triggers": [],
            "alerts": [],
            "monitor_summary": "",
            "recommended_actions": [],
            "check_latency_ms": 0,
            "experiments_checked": 1,
            "errors": [],
            "warnings": [],
            "status": "analyzing",
        }

    @pytest.mark.asyncio
    async def test_execute_sets_status_to_alerting_then_completed(self, base_state):
        """Test that execute sets status correctly."""
        node = AlertGeneratorNode()

        result = await node.execute(base_state)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_execute_generates_summary(self, base_state):
        """Test that execute generates summary."""
        node = AlertGeneratorNode()

        result = await node.execute(base_state)

        assert result["monitor_summary"] != ""
        assert "Experiment Monitor Summary" in result["monitor_summary"]

    @pytest.mark.asyncio
    async def test_execute_accumulates_latency(self, base_state):
        """Test that latency is accumulated."""
        node = AlertGeneratorNode()
        base_state["check_latency_ms"] = 100

        result = await node.execute(base_state)

        assert result["check_latency_ms"] >= 100

    @pytest.mark.asyncio
    async def test_execute_handles_exceptions(self, base_state):
        """Test that exceptions are caught and recorded."""
        node = AlertGeneratorNode()

        with patch.object(node, "_generate_srm_alerts") as mock_srm:
            mock_srm.side_effect = Exception("Alert generation failed")

            result = await node.execute(base_state)

            assert len(result["errors"]) >= 1
            assert "Alert generation failed" in result["errors"][0]["error"]
            assert result["status"] == "failed"


class TestGenerateSRMAlerts:
    """Tests for _generate_srm_alerts method."""

    @pytest.fixture
    def node(self):
        return AlertGeneratorNode()

    def test_no_alerts_when_no_issues(self, node):
        """Test no alerts generated when no SRM issues."""
        state = {
            "experiments": [],
            "srm_issues": [],
        }

        alerts = node._generate_srm_alerts(state)

        assert alerts == []

    def test_generates_alert_for_detected_issue(self, node):
        """Test alert generated for detected SRM issue."""
        state = {
            "experiments": [{"experiment_id": "exp-001", "name": "Test Experiment"}],
            "srm_issues": [
                {
                    "experiment_id": "exp-001",
                    "detected": True,
                    "p_value": 0.0001,
                    "chi_squared": 25.5,
                    "expected_ratio": {"control": 0.5, "treatment": 0.5},
                    "actual_counts": {"control": 700, "treatment": 300},
                    "severity": "critical",
                }
            ],
        }

        alerts = node._generate_srm_alerts(state)

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "srm"
        assert alerts[0]["severity"] == "critical"
        assert "Sample Ratio Mismatch" in alerts[0]["message"]
        assert alerts[0]["experiment_name"] == "Test Experiment"

    def test_skips_non_detected_issues(self, node):
        """Test skips issues where detected is False."""
        state = {
            "experiments": [],
            "srm_issues": [
                {
                    "experiment_id": "exp-001",
                    "detected": False,
                    "p_value": 0.5,
                    "chi_squared": 0.5,
                    "expected_ratio": {},
                    "actual_counts": {},
                    "severity": "info",
                }
            ],
        }

        alerts = node._generate_srm_alerts(state)

        assert alerts == []

    def test_handles_unknown_experiment(self, node):
        """Test handles SRM issue for unknown experiment."""
        state = {
            "experiments": [],  # No matching experiment
            "srm_issues": [
                {
                    "experiment_id": "exp-unknown",
                    "detected": True,
                    "p_value": 0.0001,
                    "chi_squared": 25.5,
                    "expected_ratio": {},
                    "actual_counts": {},
                    "severity": "warning",
                }
            ],
        }

        alerts = node._generate_srm_alerts(state)

        assert len(alerts) == 1
        assert alerts[0]["experiment_name"] == "Unknown Experiment"


class TestGenerateEnrollmentAlerts:
    """Tests for _generate_enrollment_alerts method."""

    @pytest.fixture
    def node(self):
        # Disable DSPy prompts to use deterministic fallback messages
        return AlertGeneratorNode(use_dspy_prompts=False)

    def test_no_alerts_when_no_issues(self, node):
        """Test no alerts generated when no enrollment issues."""
        state = {
            "experiments": [],
            "enrollment_issues": [],
        }

        alerts = node._generate_enrollment_alerts(state)

        assert alerts == []

    def test_generates_alert_for_enrollment_issue(self, node):
        """Test alert generated for enrollment issue."""
        state = {
            "experiments": [{"experiment_id": "exp-001", "name": "Test Experiment"}],
            "enrollment_issues": [
                {
                    "experiment_id": "exp-001",
                    "current_rate": 2.5,
                    "expected_rate": 5.0,
                    "days_below_threshold": 7,
                    "severity": "warning",
                }
            ],
        }

        alerts = node._generate_enrollment_alerts(state)

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "enrollment"
        assert alerts[0]["severity"] == "warning"
        assert "Low enrollment rate" in alerts[0]["message"]
        assert "2.5/day" in alerts[0]["message"]

    def test_alert_contains_details(self, node):
        """Test alert contains correct details."""
        state = {
            "experiments": [{"experiment_id": "exp-001", "name": "Test"}],
            "enrollment_issues": [
                {
                    "experiment_id": "exp-001",
                    "current_rate": 2.5,
                    "expected_rate": 5.0,
                    "days_below_threshold": 10,
                    "severity": "critical",
                }
            ],
        }

        alerts = node._generate_enrollment_alerts(state)

        assert alerts[0]["details"]["current_rate"] == 2.5
        assert alerts[0]["details"]["expected_rate"] == 5.0
        assert alerts[0]["details"]["days_below_threshold"] == 10


class TestGenerateInterimAlerts:
    """Tests for _generate_interim_alerts method."""

    @pytest.fixture
    def node(self):
        return AlertGeneratorNode()

    def test_no_alerts_when_no_triggers(self, node):
        """Test no alerts when no interim triggers."""
        state = {
            "experiments": [],
            "interim_triggers": [],
        }

        alerts = node._generate_interim_alerts(state)

        assert alerts == []

    def test_generates_alert_for_trigger(self, node):
        """Test alert generated for interim trigger."""
        state = {
            "experiments": [{"experiment_id": "exp-001", "name": "Test Experiment"}],
            "interim_triggers": [
                {
                    "experiment_id": "exp-001",
                    "analysis_number": 2,
                    "information_fraction": 0.5,
                    "milestone_reached": "50%",
                    "triggered": True,
                }
            ],
        }

        alerts = node._generate_interim_alerts(state)

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "interim_trigger"
        assert alerts[0]["severity"] == "info"
        assert "Interim analysis #2" in alerts[0]["message"]
        assert "50%" in alerts[0]["message"]

    def test_skips_non_triggered(self, node):
        """Test skips triggers where triggered is False."""
        state = {
            "experiments": [],
            "interim_triggers": [
                {
                    "experiment_id": "exp-001",
                    "analysis_number": 1,
                    "information_fraction": 0.25,
                    "milestone_reached": "25%",
                    "triggered": False,
                }
            ],
        }

        alerts = node._generate_interim_alerts(state)

        assert alerts == []


class TestGenerateFidelityAlerts:
    """Tests for _generate_fidelity_alerts method."""

    @pytest.fixture
    def node(self):
        # Disable DSPy prompts to use deterministic fallback messages
        return AlertGeneratorNode(use_dspy_prompts=False)

    def test_no_alerts_when_no_issues(self, node):
        """Test no alerts when no fidelity issues."""
        state = {
            "experiments": [],
            "fidelity_issues": [],
        }

        alerts = node._generate_fidelity_alerts(state)

        assert alerts == []

    def test_warning_alert_when_calibration_needed(self, node):
        """Test warning alert when calibration needed."""
        state = {
            "experiments": [{"experiment_id": "exp-001", "name": "Test Experiment"}],
            "fidelity_issues": [
                {
                    "experiment_id": "exp-001",
                    "twin_simulation_id": "sim-001",
                    "predicted_effect": 0.15,
                    "actual_effect": 0.08,
                    "prediction_error": 0.35,
                    "calibration_needed": True,
                    "severity": "warning",
                }
            ],
        }

        alerts = node._generate_fidelity_alerts(state)

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "fidelity"
        assert alerts[0]["severity"] == "warning"
        assert "calibration needed" in alerts[0]["message"]

    def test_info_alert_when_no_calibration_needed(self, node):
        """Test info alert when no calibration needed."""
        state = {
            "experiments": [{"experiment_id": "exp-001", "name": "Test Experiment"}],
            "fidelity_issues": [
                {
                    "experiment_id": "exp-001",
                    "twin_simulation_id": "sim-001",
                    "predicted_effect": 0.15,
                    "actual_effect": 0.14,
                    "prediction_error": 0.05,
                    "calibration_needed": False,
                    "severity": "info",
                }
            ],
        }

        alerts = node._generate_fidelity_alerts(state)

        assert len(alerts) == 1
        assert alerts[0]["severity"] == "info"
        assert "fidelity check" in alerts[0]["message"]


class TestCreateSummary:
    """Tests for _create_summary method."""

    @pytest.fixture
    def node(self):
        return AlertGeneratorNode()

    def test_summary_includes_experiments_checked(self, node):
        """Test summary includes experiments checked count."""
        state = {
            "experiments_checked": 5,
            "experiments": [],
            "srm_issues": [],
            "enrollment_issues": [],
            "interim_triggers": [],
        }

        summary = node._create_summary(state, [])

        assert "Experiments checked: 5" in summary

    def test_summary_includes_health_counts(self, node):
        """Test summary includes health status counts."""
        state = {
            "experiments_checked": 3,
            "experiments": [
                {"health_status": "healthy"},
                {"health_status": "warning"},
                {"health_status": "critical"},
            ],
            "srm_issues": [],
            "enrollment_issues": [],
            "interim_triggers": [],
        }

        summary = node._create_summary(state, [])

        assert "1 healthy" in summary
        assert "1 warning" in summary
        assert "1 critical" in summary

    def test_summary_includes_alert_counts(self, node):
        """Test summary includes alert counts by severity."""
        state = {
            "experiments_checked": 2,
            "experiments": [],
            "srm_issues": [],
            "enrollment_issues": [],
            "interim_triggers": [],
        }
        alerts = [
            {"severity": "critical"},
            {"severity": "warning"},
            {"severity": "info"},
            {"severity": "info"},
        ]

        summary = node._create_summary(state, alerts)

        assert "1 critical" in summary
        assert "1 warning" in summary
        assert "2 info" in summary

    def test_summary_includes_srm_issues_count(self, node):
        """Test summary includes SRM issues count."""
        state = {
            "experiments_checked": 1,
            "experiments": [],
            "srm_issues": [{"experiment_id": "exp-1"}, {"experiment_id": "exp-2"}],
            "enrollment_issues": [],
            "interim_triggers": [],
        }

        summary = node._create_summary(state, [])

        assert "SRM issues detected: 2" in summary

    def test_summary_includes_enrollment_issues_count(self, node):
        """Test summary includes enrollment issues count."""
        state = {
            "experiments_checked": 1,
            "experiments": [],
            "srm_issues": [],
            "enrollment_issues": [{"experiment_id": "exp-1"}],
            "interim_triggers": [],
        }

        summary = node._create_summary(state, [])

        assert "Enrollment issues: 1" in summary

    def test_summary_includes_interim_triggers_count(self, node):
        """Test summary includes interim triggers count."""
        state = {
            "experiments_checked": 1,
            "experiments": [],
            "srm_issues": [],
            "enrollment_issues": [],
            "interim_triggers": [{"experiment_id": "exp-1"}],
        }

        summary = node._create_summary(state, [])

        assert "Interim analyses triggered: 1" in summary


class TestGenerateRecommendations:
    """Tests for _generate_recommendations method."""

    @pytest.fixture
    def node(self):
        return AlertGeneratorNode()

    def test_urgent_recommendation_for_critical_srm(self, node):
        """Test urgent recommendation for critical SRM."""
        state = {"experiments": []}
        alerts = [{"alert_type": "srm", "severity": "critical"}]

        recommendations = node._generate_recommendations(state, alerts)

        assert any("URGENT" in r for r in recommendations)
        assert any("SRM" in r for r in recommendations)

    def test_recommendation_for_critical_enrollment(self, node):
        """Test recommendation for critical enrollment."""
        state = {"experiments": []}
        alerts = [{"alert_type": "enrollment", "severity": "critical"}]

        recommendations = node._generate_recommendations(state, alerts)

        assert any("enrollment" in r for r in recommendations)

    def test_recommendation_for_interim_alerts(self, node):
        """Test recommendation for interim alerts."""
        state = {"experiments": []}
        alerts = [{"alert_type": "interim_trigger", "severity": "info"}]

        recommendations = node._generate_recommendations(state, alerts)

        assert any("interim analysis" in r for r in recommendations)

    def test_recommendation_for_fidelity_warnings(self, node):
        """Test recommendation for fidelity warnings."""
        state = {"experiments": []}
        alerts = [{"alert_type": "fidelity", "severity": "warning"}]

        recommendations = node._generate_recommendations(state, alerts)

        assert any("Digital Twin" in r or "calibration" in r for r in recommendations)

    def test_all_healthy_recommendation(self, node):
        """Test recommendation when all experiments healthy."""
        state = {
            "experiments": [
                {"health_status": "healthy"},
                {"health_status": "healthy"},
            ]
        }

        recommendations = node._generate_recommendations(state, [])

        assert any("no action required" in r for r in recommendations)


class TestAlertStructure:
    """Tests for alert structure validation."""

    @pytest.fixture
    def node(self):
        return AlertGeneratorNode()

    def test_alert_has_uuid(self, node):
        """Test that alert has valid UUID."""
        state = {
            "experiments": [{"experiment_id": "exp-001", "name": "Test"}],
            "srm_issues": [
                {
                    "experiment_id": "exp-001",
                    "detected": True,
                    "p_value": 0.0001,
                    "chi_squared": 25.0,
                    "expected_ratio": {},
                    "actual_counts": {},
                    "severity": "warning",
                }
            ],
        }

        alerts = node._generate_srm_alerts(state)

        assert len(alerts[0]["alert_id"]) == 36  # UUID length

    def test_alert_has_timestamp(self, node):
        """Test that alert has timestamp."""
        state = {
            "experiments": [{"experiment_id": "exp-001", "name": "Test"}],
            "enrollment_issues": [
                {
                    "experiment_id": "exp-001",
                    "current_rate": 2.0,
                    "expected_rate": 5.0,
                    "days_below_threshold": 7,
                    "severity": "warning",
                }
            ],
        }

        alerts = node._generate_enrollment_alerts(state)

        # Verify timestamp is valid ISO format
        assert "T" in alerts[0]["timestamp"]

    def test_alert_has_recommended_action(self, node):
        """Test that alert has recommended action."""
        state = {
            "experiments": [{"experiment_id": "exp-001", "name": "Test"}],
            "interim_triggers": [
                {
                    "experiment_id": "exp-001",
                    "analysis_number": 1,
                    "information_fraction": 0.25,
                    "milestone_reached": "25%",
                    "triggered": True,
                }
            ],
        }

        alerts = node._generate_interim_alerts(state)

        assert alerts[0]["recommended_action"] != ""


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def node(self):
        return AlertGeneratorNode()

    @pytest.mark.asyncio
    async def test_empty_state(self, node):
        """Test handling of minimal state."""
        state = {
            "query": "",
            "check_all_active": True,
            "experiment_ids": [],
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "experiments": [],
            "srm_issues": [],
            "enrollment_issues": [],
            "fidelity_issues": [],
            "interim_triggers": [],
            "alerts": [],
            "monitor_summary": "",
            "recommended_actions": [],
            "check_latency_ms": 0,
            "experiments_checked": 0,
            "errors": [],
            "warnings": [],
            "status": "analyzing",
        }

        result = await node.execute(state)

        assert result["status"] == "completed"
        assert result["alerts"] == []

    def test_multiple_alerts_same_experiment(self, node):
        """Test multiple alerts for same experiment."""
        state = {
            "experiments": [{"experiment_id": "exp-001", "name": "Test Experiment"}],
            "srm_issues": [
                {
                    "experiment_id": "exp-001",
                    "detected": True,
                    "p_value": 0.0001,
                    "chi_squared": 25.0,
                    "expected_ratio": {},
                    "actual_counts": {},
                    "severity": "critical",
                }
            ],
            "enrollment_issues": [
                {
                    "experiment_id": "exp-001",
                    "current_rate": 2.0,
                    "expected_rate": 5.0,
                    "days_below_threshold": 14,
                    "severity": "critical",
                }
            ],
            "interim_triggers": [
                {
                    "experiment_id": "exp-001",
                    "analysis_number": 2,
                    "information_fraction": 0.5,
                    "milestone_reached": "50%",
                    "triggered": True,
                }
            ],
            "fidelity_issues": [],
        }

        srm_alerts = node._generate_srm_alerts(state)
        enrollment_alerts = node._generate_enrollment_alerts(state)
        interim_alerts = node._generate_interim_alerts(state)

        total_alerts = len(srm_alerts) + len(enrollment_alerts) + len(interim_alerts)
        assert total_alerts == 3

    def test_unknown_health_status(self, node):
        """Test handling of unknown health status."""
        state = {
            "experiments_checked": 1,
            "experiments": [{"health_status": "unknown"}],
            "srm_issues": [],
            "enrollment_issues": [],
            "interim_triggers": [],
        }

        summary = node._create_summary(state, [])

        # Should still work without crashing
        assert "Experiment Monitor Summary" in summary

    def test_missing_severity_defaults(self, node):
        """Test handling of missing severity field."""
        state = {
            "experiments": [{"experiment_id": "exp-001", "name": "Test"}],
            "srm_issues": [
                {
                    "experiment_id": "exp-001",
                    "detected": True,
                    "p_value": 0.0001,
                    "chi_squared": 25.0,
                    "expected_ratio": {},
                    "actual_counts": {},
                    # No severity field
                }
            ],
        }

        alerts = node._generate_srm_alerts(state)

        # Should default to "warning"
        assert alerts[0]["severity"] == "warning"
