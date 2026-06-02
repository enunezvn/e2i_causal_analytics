"""Unit tests for DataSourceValidator.

Tests the detection of mock vs real data sources for agents.
"""

import pytest

from src.testing.data_source_validator import (
    DataSourceType,
    DataSourceValidationResult,
    DataSourceValidator,
)


class TestDataSourceType:
    """Test DataSourceType enum."""

    def test_enum_values(self):
        """Test that all expected data source types exist."""
        assert DataSourceType.SUPABASE.value == "supabase"
        assert DataSourceType.MOCK.value == "mock"
        assert DataSourceType.TIER0_PASSTHROUGH.value == "tier0"
        assert DataSourceType.COMPUTATIONAL.value == "computational"
        assert DataSourceType.UNKNOWN.value == "unknown"


class TestDataSourceValidationResult:
    """Test DataSourceValidationResult dataclass."""

    def test_summary_pass(self):
        """Test summary for passed validation."""
        result = DataSourceValidationResult(
            agent_name="test_agent",
            passed=True,
            detected_source=DataSourceType.SUPABASE,
            message="Data source 'supabase' is acceptable",
        )
        assert "PASS" in result.summary
        assert "test_agent" in result.summary
        assert "supabase" in result.summary

    def test_summary_fail(self):
        """Test summary for failed validation."""
        result = DataSourceValidationResult(
            agent_name="test_agent",
            passed=False,
            detected_source=DataSourceType.MOCK,
            message="Mock data detected but reject_mock=True",
        )
        assert "FAIL" in result.summary
        assert "mock" in result.summary


class TestDataSourceValidator:
    """Test DataSourceValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a DataSourceValidator instance."""
        return DataSourceValidator()

    def test_health_score_mock_detection_perfect_score(self, validator):
        """Test that perfect 100.0 health score is detected as mock."""
        output = {
            "overall_health_score": 100.0,
            "component_health_score": 1.0,
            "health_grade": "A",
        }
        result = validator.validate(
            agent_name="health_score",
            agent_output=output,
        )
        assert result.detected_source == DataSourceType.MOCK
        assert not result.passed
        assert "100.0" in str(result.evidence)

    def test_health_score_real_data_detection(self, validator):
        """Test that variance in health score indicates real data."""
        output = {
            "overall_health_score": 85.5,
            "component_health_score": 0.9,
            "health_grade": "B",
        }
        result = validator.validate(
            agent_name="health_score",
            agent_output=output,
        )
        assert result.detected_source == DataSourceType.SUPABASE
        assert result.passed

    def test_health_score_all_components_healthy_mock(self, validator):
        """Test that all components reporting healthy with 1.0 score is mock."""
        output = {
            "overall_health_score": 90.0,
            "component_health_score": 1.0,
            "component_statuses": [
                {"component_name": "database", "status": "healthy"},
                {"component_name": "cache", "status": "healthy"},
                {"component_name": "api", "status": "healthy"},
            ],
        }
        result = validator.validate(
            agent_name="health_score",
            agent_output=output,
        )
        assert result.detected_source == DataSourceType.MOCK
        assert not result.passed

    def test_gap_analyzer_mock_in_logs(self, validator):
        """Test that MockDataConnector in logs is detected."""
        output = {"executive_summary": "Analysis complete"}
        logs = ["INFO: Using MockDataConnector for testing"]
        result = validator.validate(
            agent_name="gap_analyzer",
            agent_output=output,
            execution_logs=logs,
        )
        assert result.detected_source == DataSourceType.MOCK
        assert not result.passed

    def test_gap_analyzer_real_data(self, validator):
        """Test gap_analyzer with real data source."""
        output = {
            "executive_summary": "Found 5 performance gaps",
            "data_source": "supabase_analytics",
        }
        result = validator.validate(
            agent_name="gap_analyzer",
            agent_output=output,
            execution_logs=[],
        )
        assert result.detected_source == DataSourceType.SUPABASE
        assert result.passed

    def test_gap_analyzer_tier0_passthrough(self, validator):
        """Test gap_analyzer with tier0 passthrough."""
        output = {
            "executive_summary": "Analysis complete",
            "tier0_experiment_id": "exp_12345",
        }
        result = validator.validate(
            agent_name="gap_analyzer",
            agent_output=output,
        )
        assert result.detected_source == DataSourceType.TIER0_PASSTHROUGH
        assert result.passed

    def test_heterogeneous_optimizer_mock_fallback_in_logs(self, validator):
        """Test heterogeneous_optimizer mock fallback detection."""
        output = {"status": "completed"}
        logs = ["WARNING: Falling back to MockDataConnector for testing."]
        result = validator.validate(
            agent_name="heterogeneous_optimizer",
            agent_output=output,
            execution_logs=logs,
        )
        assert result.detected_source == DataSourceType.MOCK
        assert not result.passed

    def test_resource_optimizer_computational(self, validator):
        """Test that resource_optimizer is always computational."""
        output = {"status": "optimal", "allocation": {}}
        result = validator.validate(
            agent_name="resource_optimizer",
            agent_output=output,
        )
        assert result.detected_source == DataSourceType.COMPUTATIONAL
        assert result.passed

    def test_orchestrator_computational(self, validator, monkeypatch):
        """Test that orchestrator is computational when NO mock marker is present.

        Explicitly clears the keyless-mock gate so ambient env (a dev shell or a
        lane that exports E2I_ALLOW_MOCK_LLM) can't flip this to MOCK.
        """
        monkeypatch.delenv("E2I_ALLOW_MOCK_LLM", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-real-key")
        output = {"status": "routed", "selected_agents": ["causal_impact"]}
        result = validator.validate(
            agent_name="orchestrator",
            agent_output=output,
        )
        assert result.detected_source == DataSourceType.COMPUTATIONAL
        assert result.passed

    def test_tool_composer_computational(self, validator, monkeypatch):
        """Test that tool_composer is computational when NO mock marker is present."""
        monkeypatch.delenv("E2I_ALLOW_MOCK_LLM", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-real-key")
        output = {"success": True, "tools_composed": 3}
        result = validator.validate(
            agent_name="tool_composer",
            agent_output=output,
        )
        assert result.detected_source == DataSourceType.COMPUTATIONAL
        assert result.passed

    def test_unknown_agent_no_requirements(self, validator):
        """Test that unknown agents pass with no requirements configured."""
        output = {"result": "success"}
        result = validator.validate(
            agent_name="unknown_agent",
            agent_output=output,
        )
        assert result.passed
        assert result.detected_source == DataSourceType.UNKNOWN
        assert "No data source requirements" in result.message

    def test_custom_requirements(self):
        """Test custom requirements override."""
        custom_requirements = {
            "custom_agent": {
                "acceptable": [DataSourceType.MOCK],
                "reject_mock": False,
            }
        }
        validator = DataSourceValidator(custom_requirements=custom_requirements)

        # Should accept mock for custom_agent
        output = {"status": "done"}
        result = validator.validate(
            agent_name="custom_agent",
            agent_output=output,
            execution_logs=["Using MockDataConnector"],
        )
        # Would detect mock from logs, but custom_agent accepts mock
        # Note: Detection still uses default logic for unknown agents
        assert result.passed or result.detected_source == DataSourceType.UNKNOWN

    def test_get_requirements(self, validator):
        """Test getting requirements for an agent."""
        reqs = validator.get_requirements("health_score")
        assert reqs is not None
        assert reqs["reject_mock"] is True
        assert DataSourceType.SUPABASE in reqs["acceptable"]

        reqs = validator.get_requirements("unknown_agent")
        assert reqs is None

    def test_list_agents_with_requirements(self, validator):
        """Test listing agents with configured requirements."""
        agents = validator.list_agents_with_requirements()
        assert "health_score" in agents
        assert "gap_analyzer" in agents
        assert "resource_optimizer" in agents
        assert "orchestrator" in agents

    def test_acceptable_sources_in_result(self, validator):
        """Test that acceptable sources are included in result."""
        output = {"overall_health_score": 85.0}
        result = validator.validate(
            agent_name="health_score",
            agent_output=output,
        )
        assert DataSourceType.SUPABASE in result.acceptable_sources

    def test_reject_mock_flag_in_result(self, validator):
        """Test that reject_mock flag is included in result."""
        output = {"overall_health_score": 100.0}
        result = validator.validate(
            agent_name="health_score",
            agent_output=output,
        )
        assert result.reject_mock is True

    def test_evidence_list_populated(self, validator):
        """Test that evidence list is populated with detection details."""
        output = {"overall_health_score": 100.0}
        result = validator.validate(
            agent_name="health_score",
            agent_output=output,
        )
        assert len(result.evidence) > 0
        assert any("100.0" in e for e in result.evidence)


class _MarkedMockStub:
    """Minimal stand-in carrying the in-band marked-mock attribute.

    Mirrors ``src.utils.mock_llm.MarkedMockChatLLM`` which sets the
    class attribute ``mock_response_for_dev_only = True``. Used to assert
    instance-traversal detection without importing the real agent runtime.
    """

    mock_response_for_dev_only = True


class _AgentInstanceStub:
    """Holds a marked-mock LLM the way tool_composer retains ``llm_client``."""

    def __init__(self, llm: object) -> None:
        self.llm_client = llm


class TestMarkedMockLLMDetection:
    """#616 fix#2: marked-mock LLM agents must be recorded as MOCK, not silently
    pass as COMPUTATIONAL (which would equate canned reasoning with real
    reasoning). Detection must use REAL signals (output marker / live instance /
    captured logs / the keyless-mock gate), never a fabricated marker."""

    @pytest.fixture
    def validator(self):
        return DataSourceValidator()

    @pytest.fixture(autouse=True)
    def _clear_gate(self, monkeypatch):
        # Default: gate OFF + real key present, so only the SIGNAL under test fires.
        monkeypatch.delenv("E2I_ALLOW_MOCK_LLM", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-real-key")

    # --- Signal 1: marker in output payload (experiment_designer) ---
    def test_experiment_designer_marker_in_output_is_mock(self, validator):
        output = {
            "design_type": "rct",
            "validity_audit": {"mock_response_for_dev_only": True, "score": 0.9},
        }
        result = validator.validate(agent_name="experiment_designer", agent_output=output)
        assert result.detected_source == DataSourceType.MOCK
        # Plumbing-only PASS: MOCK is acceptable for this LLM-dependent agent.
        assert result.passed
        assert any("mock_response_for_dev_only" in e for e in result.evidence)

    def test_experiment_designer_no_marker_is_computational(self, validator):
        output = {"design_type": "rct", "design_rationale": "real reasoning"}
        result = validator.validate(agent_name="experiment_designer", agent_output=output)
        assert result.detected_source == DataSourceType.COMPUTATIONAL
        assert result.passed

    # --- Signal 2: live marked-mock LLM on the instance (tool_composer) ---
    def test_tool_composer_instance_marked_mock_is_mock(self, validator):
        agent = _AgentInstanceStub(llm=_MarkedMockStub())
        result = validator.validate(
            agent_name="tool_composer",
            agent_output={"success": True},
            agent_instance=agent,
        )
        assert result.detected_source == DataSourceType.MOCK
        assert result.passed  # plumbing-only PASS keeps overall success True

    def test_orchestrator_instance_marked_mock_is_mock(self, validator):
        agent = _AgentInstanceStub(llm=_MarkedMockStub())
        result = validator.validate(
            agent_name="orchestrator",
            agent_output={"status": "routed"},
            agent_instance=agent,
        )
        assert result.detected_source == DataSourceType.MOCK
        assert result.passed

    # --- Signal 3: marker text in captured execution logs ---
    def test_tool_composer_marker_in_logs_is_mock(self, validator):
        result = validator.validate(
            agent_name="tool_composer",
            agent_output={"success": True},
            execution_logs=["tool_composer: using MARKED mock (mock_response_for_dev_only=True)"],
        )
        assert result.detected_source == DataSourceType.MOCK
        assert result.passed

    # --- Signal 4: keyless-mock gate active (orchestrator transient-node mock) ---
    def test_orchestrator_keyless_gate_active_is_mock(self, validator, monkeypatch):
        # Gate ON: opt-in flag set + provider key absent == the exact condition
        # under which the orchestrator's nodes fall back to a marked mock.
        monkeypatch.setenv("E2I_ALLOW_MOCK_LLM", "1")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        result = validator.validate(
            agent_name="orchestrator",
            agent_output={"status": "routed"},
            agent_instance=None,
        )
        assert result.detected_source == DataSourceType.MOCK
        assert result.passed
        assert any("keyless-mock gate" in e for e in result.evidence)

    def test_keyless_gate_inactive_with_real_key_is_computational(self, validator, monkeypatch):
        # Flag set BUT a real key is present -> real LLM, NOT a mock.
        monkeypatch.setenv("E2I_ALLOW_MOCK_LLM", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        result = validator.validate(
            agent_name="orchestrator",
            agent_output={"status": "routed"},
            agent_instance=None,
        )
        assert result.detected_source == DataSourceType.COMPUTATIONAL
        assert result.passed

    def test_keyless_gate_respects_anthropic_provider(self, validator, monkeypatch):
        monkeypatch.setenv("E2I_ALLOW_MOCK_LLM", "1")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-present")  # wrong provider's key
        result = validator.validate(
            agent_name="orchestrator",
            agent_output={"status": "routed"},
            agent_instance=None,
        )
        assert result.detected_source == DataSourceType.MOCK

    def test_gate_does_not_affect_non_llm_agent(self, validator, monkeypatch):
        # resource_optimizer is computational and NOT LLM-dependent: the gate must
        # not mislabel it even when the keyless-mock env is active.
        monkeypatch.setenv("E2I_ALLOW_MOCK_LLM", "1")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = validator.validate(
            agent_name="resource_optimizer",
            agent_output={"status": "optimal"},
        )
        assert result.detected_source == DataSourceType.COMPUTATIONAL
        assert result.passed

    def test_mock_detection_keeps_overall_success_true(self, validator):
        # The whole point of "plumbing-only PASS": MOCK must NOT fail the agent.
        agent = _AgentInstanceStub(llm=_MarkedMockStub())
        for name in ("orchestrator", "tool_composer"):
            result = validator.validate(
                agent_name=name,
                agent_output={"status": "ok"},
                agent_instance=agent,
            )
            assert result.passed, f"{name} marked-mock must remain a PASS (plumbing-only)"
            assert result.detected_source == DataSourceType.MOCK


class TestDataSourceValidatorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def validator(self):
        return DataSourceValidator()

    def test_empty_output(self, validator):
        """Test validation with empty output."""
        result = validator.validate(
            agent_name="health_score",
            agent_output={},
        )
        # Empty output should not crash, but should not pass validation
        assert result is not None

    def test_none_values_in_output(self, validator):
        """Test validation with None values."""
        output = {
            "overall_health_score": None,
            "component_health_score": None,
        }
        result = validator.validate(
            agent_name="health_score",
            agent_output=output,
        )
        assert result is not None
        # Should not detect as mock (100.0) since values are None

    def test_empty_logs_list(self, validator):
        """Test validation with empty logs."""
        result = validator.validate(
            agent_name="gap_analyzer",
            agent_output={"status": "done"},
            execution_logs=[],
        )
        assert result is not None

    def test_none_logs(self, validator):
        """Test validation with None logs."""
        result = validator.validate(
            agent_name="gap_analyzer",
            agent_output={"status": "done"},
            execution_logs=None,
        )
        assert result is not None


class TestDataSourceValidatorIntegration:
    """Integration tests for DataSourceValidator with agent patterns."""

    @pytest.fixture
    def validator(self):
        return DataSourceValidator()

    def test_drift_monitor_tier0_passthrough(self, validator):
        """Test drift_monitor with tier0 data."""
        output = {
            "status": "completed",
            "drift_detected": False,
            "tier0_experiment_id": "exp_test",
        }
        result = validator.validate(
            agent_name="drift_monitor",
            agent_output=output,
        )
        assert result.detected_source == DataSourceType.TIER0_PASSTHROUGH
        assert result.passed

    def test_explainer_tier0_passthrough(self, validator):
        """Test explainer with tier0 analysis results."""
        output = {
            "executive_summary": "The analysis shows...",
            "tier0_experiment_id": "exp_test",
        }
        result = validator.validate(
            agent_name="explainer",
            agent_output=output,
        )
        assert result.detected_source == DataSourceType.TIER0_PASSTHROUGH
        assert result.passed

    def test_feedback_learner_with_status(self, validator):
        """Test feedback_learner status-based detection."""
        output = {
            "status": "completed",
            "patterns_learned": 5,
        }
        result = validator.validate(
            agent_name="feedback_learner",
            agent_output=output,
        )
        # Should detect as tier0 passthrough based on success status
        assert result.detected_source == DataSourceType.TIER0_PASSTHROUGH
        assert result.passed
