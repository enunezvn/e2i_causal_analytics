"""TDD tests for experiment_monitor self-emission of recipient training signals (Task B1).

Tests that AlertGeneratorNode emits recipient training signals after generating
SRM, alert, and summary outputs. Offline-only: no real LM or DB.

Asserts:
- emit_recipient_signal is called with agent_name="experiment_monitor"
- template_field matches the generating method
- signature_inputs keys EXACTLY match recipient_required_input_keys("experiment_monitor")[field]
- reward is a float in [0, 1]
- A failure in emit (stub raises) does NOT break the node's normal output
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

EXPECTED_SRM_KEYS = frozenset(
    ["experiment_name", "chi_squared", "p_value", "expected_ratio", "actual_counts"]
)
EXPECTED_SUMMARY_KEYS = frozenset(
    ["experiments_checked", "healthy_count", "warning_count", "critical_count", "issue_types"]
)
EXPECTED_ALERT_KEYS = frozenset(["experiment_name", "alert_type", "severity", "details"])


def _srm_state() -> Dict[str, Any]:
    """Minimal state that produces one SRM alert."""
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
                "experiment_id": "exp-srm-01",
                "name": "Kisqali-SRM-Test",
                "status": "running",
                "health_status": "warning",
                "days_running": 10,
                "total_enrolled": 200,
                "enrollment_rate": 20.0,
                "current_information_fraction": 0.4,
            }
        ],
        "srm_issues": [
            {
                "experiment_id": "exp-srm-01",
                "detected": True,
                "p_value": 0.00012,
                "chi_squared": 14.56,
                "expected_ratio": {"control": 0.5, "treatment": 0.5},
                "actual_counts": {"control": 140, "treatment": 60},
                "severity": "critical",
            }
        ],
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


def _alert_state() -> Dict[str, Any]:
    """Minimal state that produces one fidelity alert (uses alert_template)."""
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
                "experiment_id": "exp-fidelity-01",
                "name": "Kisqali-Fidelity-Test",
                "status": "running",
                "health_status": "warning",
                "days_running": 14,
                "total_enrolled": 400,
                "enrollment_rate": 28.5,
                "current_information_fraction": 0.4,
            }
        ],
        "srm_issues": [],
        "enrollment_issues": [],
        "fidelity_issues": [
            {
                "experiment_id": "exp-fidelity-01",
                "twin_simulation_id": "sim-001",
                "predicted_effect": 0.15,
                "actual_effect": 0.08,
                "prediction_error": 0.467,
                "calibration_needed": True,
                "severity": "warning",
            }
        ],
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


def _multi_issue_state() -> Dict[str, Any]:
    """State with SRM + fidelity issues to test multiple emit calls."""
    state = _srm_state()
    state["fidelity_issues"] = [
        {
            "experiment_id": "exp-srm-01",
            "twin_simulation_id": "sim-002",
            "predicted_effect": 0.20,
            "actual_effect": 0.11,
            "prediction_error": 0.45,
            "calibration_needed": True,
            "severity": "warning",
        }
    ]
    return state


# --------------------------------------------------------------------------- #
# Tests: srm_template emission
# --------------------------------------------------------------------------- #


class TestSRMTemplateEmission:
    """Tests that SRM message generation emits an srm_template signal."""

    @pytest.mark.asyncio
    async def test_srm_emit_called_with_correct_agent_name(self, monkeypatch):
        """emit_recipient_signal is called with agent_name='experiment_monitor'."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        mock_emit = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            mock_emit,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        await node.execute(_srm_state())

        srm_calls = [
            c for c in mock_emit.call_args_list if c.kwargs.get("template_field") == "srm_template"
        ]
        assert srm_calls, "Expected at least one srm_template emit call"
        assert srm_calls[0].kwargs["agent_name"] == "experiment_monitor"

    @pytest.mark.asyncio
    async def test_srm_emit_signature_inputs_keys_match_contract(self, monkeypatch):
        """signature_inputs keys EXACTLY match recipient_required_input_keys for srm_template."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        captured: List[Dict[str, Any]] = []

        async def _capture(**kwargs):
            captured.append(kwargs)
            return True

        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            _capture,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        await node.execute(_srm_state())

        srm_calls = [c for c in captured if c.get("template_field") == "srm_template"]
        assert srm_calls, "Expected at least one srm_template emit"
        sig_inputs = srm_calls[0]["signature_inputs"]
        assert set(sig_inputs.keys()) == EXPECTED_SRM_KEYS, (
            f"srm_template signature_inputs keys {set(sig_inputs.keys())} != {EXPECTED_SRM_KEYS}"
        )

    @pytest.mark.asyncio
    async def test_srm_emit_reward_in_unit_interval(self, monkeypatch):
        """reward is a float in [0, 1]."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        captured: List[Dict[str, Any]] = []

        async def _capture(**kwargs):
            captured.append(kwargs)
            return True

        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            _capture,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        await node.execute(_srm_state())

        srm_calls = [c for c in captured if c.get("template_field") == "srm_template"]
        assert srm_calls, "Expected at least one srm_template emit"
        reward = srm_calls[0]["reward"]
        assert isinstance(reward, float), f"reward must be float, got {type(reward)}"
        assert 0.0 <= reward <= 1.0, f"reward must be in [0,1], got {reward}"

    @pytest.mark.asyncio
    async def test_srm_emit_includes_experiment_name_value(self, monkeypatch):
        """signature_inputs['experiment_name'] matches the experiment name."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        captured: List[Dict[str, Any]] = []

        async def _capture(**kwargs):
            captured.append(kwargs)
            return True

        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            _capture,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        await node.execute(_srm_state())

        srm_calls = [c for c in captured if c.get("template_field") == "srm_template"]
        assert srm_calls
        sig_inputs = srm_calls[0]["signature_inputs"]
        assert sig_inputs["experiment_name"] == "Kisqali-SRM-Test"
        assert sig_inputs["chi_squared"] == pytest.approx(14.56)
        assert sig_inputs["p_value"] == pytest.approx(0.00012)


# --------------------------------------------------------------------------- #
# Tests: summary_template emission (via _create_summary path through execute)
# --------------------------------------------------------------------------- #


class TestSummaryTemplateEmission:
    """Tests that summary generation emits a summary_template signal."""

    @pytest.mark.asyncio
    async def test_summary_emit_called_with_correct_template_field(self, monkeypatch):
        """emit_recipient_signal is called with template_field='summary_template'."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        mock_emit = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            mock_emit,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        await node.execute(_srm_state())

        summary_calls = [
            c
            for c in mock_emit.call_args_list
            if c.kwargs.get("template_field") == "summary_template"
        ]
        assert summary_calls, "Expected at least one summary_template emit call"

    @pytest.mark.asyncio
    async def test_summary_emit_signature_inputs_keys_match_contract(self, monkeypatch):
        """signature_inputs keys EXACTLY match recipient_required_input_keys for summary_template."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        captured: List[Dict[str, Any]] = []

        async def _capture(**kwargs):
            captured.append(kwargs)
            return True

        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            _capture,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        await node.execute(_srm_state())

        summary_calls = [c for c in captured if c.get("template_field") == "summary_template"]
        assert summary_calls, "Expected at least one summary_template emit"
        sig_inputs = summary_calls[0]["signature_inputs"]
        assert set(sig_inputs.keys()) == EXPECTED_SUMMARY_KEYS, (
            f"summary_template signature_inputs keys {set(sig_inputs.keys())} "
            f"!= {EXPECTED_SUMMARY_KEYS}"
        )

    @pytest.mark.asyncio
    async def test_summary_emit_reward_in_unit_interval(self, monkeypatch):
        """reward is a float in [0, 1] for summary_template."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        captured: List[Dict[str, Any]] = []

        async def _capture(**kwargs):
            captured.append(kwargs)
            return True

        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            _capture,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        await node.execute(_srm_state())

        summary_calls = [c for c in captured if c.get("template_field") == "summary_template"]
        assert summary_calls
        reward = summary_calls[0]["reward"]
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0


# --------------------------------------------------------------------------- #
# Tests: alert_template emission (fidelity alerts use AlertGenerationSignature)
# --------------------------------------------------------------------------- #


class TestAlertTemplateEmission:
    """Tests that fidelity alert generation emits an alert_template signal."""

    @pytest.mark.asyncio
    async def test_fidelity_alert_emits_alert_template(self, monkeypatch):
        """emit_recipient_signal is called with template_field='alert_template' for fidelity."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        mock_emit = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            mock_emit,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        await node.execute(_alert_state())

        alert_calls = [
            c
            for c in mock_emit.call_args_list
            if c.kwargs.get("template_field") == "alert_template"
        ]
        assert alert_calls, "Expected at least one alert_template emit call"

    @pytest.mark.asyncio
    async def test_alert_emit_signature_inputs_keys_match_contract(self, monkeypatch):
        """signature_inputs keys EXACTLY match recipient_required_input_keys for alert_template."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        captured: List[Dict[str, Any]] = []

        async def _capture(**kwargs):
            captured.append(kwargs)
            return True

        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            _capture,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        await node.execute(_alert_state())

        alert_calls = [c for c in captured if c.get("template_field") == "alert_template"]
        assert alert_calls, "Expected at least one alert_template emit"
        sig_inputs = alert_calls[0]["signature_inputs"]
        assert set(sig_inputs.keys()) == EXPECTED_ALERT_KEYS, (
            f"alert_template signature_inputs keys {set(sig_inputs.keys())} "
            f"!= {EXPECTED_ALERT_KEYS}"
        )

    @pytest.mark.asyncio
    async def test_alert_emit_reward_in_unit_interval(self, monkeypatch):
        """reward is a float in [0, 1] for alert_template."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        captured: List[Dict[str, Any]] = []

        async def _capture(**kwargs):
            captured.append(kwargs)
            return True

        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            _capture,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        await node.execute(_alert_state())

        alert_calls = [c for c in captured if c.get("template_field") == "alert_template"]
        assert alert_calls
        reward = alert_calls[0]["reward"]
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0


# --------------------------------------------------------------------------- #
# Tests: best-effort contract — emit failure does NOT break node output
# --------------------------------------------------------------------------- #


class TestEmitBestEffort:
    """Tests that a failure in emit never breaks alert generation."""

    @pytest.mark.asyncio
    async def test_emit_failure_does_not_break_srm_alert_generation(self, monkeypatch):
        """Node still produces SRM alert even if emit raises an exception."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        async def _boom(**kwargs):
            raise RuntimeError("DB down during emit")

        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            _boom,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        result = await node.execute(_srm_state())

        # Node must complete successfully
        assert result["status"] == "completed", (
            f"Node failed when emit raised: status={result['status']}, "
            f"errors={result.get('errors')}"
        )
        # Alert must still be generated
        assert len(result["alerts"]) >= 1
        assert any(a["alert_type"] == "srm" for a in result["alerts"])

    @pytest.mark.asyncio
    async def test_emit_failure_does_not_break_summary_generation(self, monkeypatch):
        """Node still produces a summary even if emit raises."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        async def _boom(**kwargs):
            raise RuntimeError("Network error")

        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            _boom,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        result = await node.execute(_srm_state())

        assert result["status"] == "completed"
        assert "Experiment Monitor Summary" in result["monitor_summary"]

    @pytest.mark.asyncio
    async def test_emit_failure_does_not_break_fidelity_alert_generation(self, monkeypatch):
        """Node still produces fidelity alert even if emit raises."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        async def _boom(**kwargs):
            raise RuntimeError("Timeout")

        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            _boom,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        result = await node.execute(_alert_state())

        assert result["status"] == "completed"
        assert any(a["alert_type"] == "fidelity" for a in result["alerts"])


# --------------------------------------------------------------------------- #
# Tests: _signal_reward helper is deterministic and range-correct
# --------------------------------------------------------------------------- #


class TestSignalRewardHelper:
    """Tests for the _signal_reward deterministic helper."""

    def test_reward_is_float_in_unit_interval(self):
        """_signal_reward always returns float in [0, 1]."""
        from src.agents.experiment_monitor.nodes.alert_generator import _signal_reward

        reward = _signal_reward(
            generated_output="SRM detected in Kisqali-SRM-Test (p=0.000120)",
            signature_inputs={
                "experiment_name": "Kisqali-SRM-Test",
                "chi_squared": 14.56,
                "p_value": 0.00012,
                "expected_ratio": "{'control': 0.5, 'treatment': 0.5}",
                "actual_counts": "{'control': 140, 'treatment': 60}",
            },
        )
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0

    def test_reward_higher_for_non_empty_output(self):
        """Non-empty output scores higher than empty output."""
        from src.agents.experiment_monitor.nodes.alert_generator import _signal_reward

        inputs = {"experiment_name": "TestExp", "chi_squared": 5.0}
        good = _signal_reward("Sample Ratio Mismatch in TestExp (chi^2=5.0)", inputs)
        bad = _signal_reward("", inputs)
        assert good > bad

    def test_reward_higher_when_output_references_key_inputs(self):
        """Output referencing key input values scores higher than generic text."""
        from src.agents.experiment_monitor.nodes.alert_generator import _signal_reward

        inputs = {
            "experiment_name": "KisqaliNE",
            "p_value": 0.0001,
        }
        specific = _signal_reward(
            "Critical SRM in KisqaliNE: p=0.0001, investigate immediately", inputs
        )
        generic = _signal_reward("Something happened in the experiment", inputs)
        assert specific >= generic

    def test_reward_is_deterministic(self):
        """Same inputs always produce the same reward."""
        from src.agents.experiment_monitor.nodes.alert_generator import _signal_reward

        inputs = {"experiment_name": "Exp1", "chi_squared": 3.0}
        output = "SRM in Exp1 (chi_sq=3.0)"
        assert _signal_reward(output, inputs) == _signal_reward(output, inputs)

    def test_empty_output_scores_low(self):
        """Completely empty output scores below 0.3."""
        from src.agents.experiment_monitor.nodes.alert_generator import _signal_reward

        reward = _signal_reward("", {"experiment_name": "Test"})
        assert reward < 0.3

    def test_reasonable_length_output_scores_above_threshold(self):
        """A reasonable-length, relevant output scores above 0.3."""
        from src.agents.experiment_monitor.nodes.alert_generator import _signal_reward

        inputs = {
            "experiment_name": "Kisqali-NE",
            "chi_squared": 14.56,
            "p_value": 0.00012,
            "expected_ratio": "50/50",
            "actual_counts": "140/60",
        }
        output = (
            "Critical Sample Ratio Mismatch detected in Kisqali-NE experiment. "
            "Chi-squared value of 14.56 with p-value 0.00012 strongly indicates "
            "non-random assignment. Expected 50/50 split but observed 140/60."
        )
        reward = _signal_reward(output, inputs)
        assert reward > 0.3


# --------------------------------------------------------------------------- #
# Tests: multiple alerts trigger multiple emits
# --------------------------------------------------------------------------- #


class TestMultipleAlertEmissions:
    """Tests that each alert type produces its own emit call."""

    @pytest.mark.asyncio
    async def test_srm_and_fidelity_both_emit(self, monkeypatch):
        """Both SRM and fidelity alerts each trigger their respective emit calls."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        captured: List[Dict[str, Any]] = []

        async def _capture(**kwargs):
            captured.append(kwargs)
            return True

        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            _capture,
        )

        node = AlertGeneratorNode(use_dspy_prompts=False)
        result = await node.execute(_multi_issue_state())

        assert result["status"] == "completed"
        template_fields = {c.get("template_field") for c in captured}
        # At minimum we expect srm_template and summary_template
        assert "srm_template" in template_fields
        assert "summary_template" in template_fields

    @pytest.mark.asyncio
    async def test_no_emit_when_no_issues(self, monkeypatch):
        """No srm_template or alert_template emit calls when there are no issues."""
        from src.agents.experiment_monitor.nodes.alert_generator import AlertGeneratorNode

        captured: List[Dict[str, Any]] = []

        async def _capture(**kwargs):
            captured.append(kwargs)
            return True

        monkeypatch.setattr(
            "src.agents.experiment_monitor.nodes.alert_generator.emit_recipient_signal",
            _capture,
        )

        empty_state = {
            "query": "Check",
            "check_all_active": True,
            "experiment_ids": [],
            "srm_threshold": 0.001,
            "enrollment_threshold": 5.0,
            "fidelity_threshold": 0.2,
            "check_interim": True,
            "experiments": [
                {
                    "experiment_id": "exp-ok",
                    "name": "AllGood",
                    "status": "running",
                    "health_status": "healthy",
                    "days_running": 7,
                    "total_enrolled": 500,
                    "enrollment_rate": 71.0,
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

        node = AlertGeneratorNode(use_dspy_prompts=False)
        await node.execute(empty_state)

        # No SRM or alert_template calls when there are no alerts
        srm_calls = [c for c in captured if c.get("template_field") == "srm_template"]
        alert_calls = [c for c in captured if c.get("template_field") == "alert_template"]
        assert srm_calls == []
        assert alert_calls == []
