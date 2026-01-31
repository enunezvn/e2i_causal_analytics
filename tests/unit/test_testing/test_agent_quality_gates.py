"""Unit tests for agent_quality_gates module.

Tests quality gate configurations and semantic validators.
"""

import os

import pytest

from src.testing.agent_quality_gates import (
    AGENT_QUALITY_GATES,
    AgentQualityGate,
    DataQualityCheck,
    DataSourceRequirement,
    _safe_get,
    _validate_causal_impact,
    _validate_drift_monitor,
    _validate_experiment_designer,
    _validate_explainer,
    _validate_feedback_learner,
    _validate_gap_analyzer,
    _validate_health_score,
    _validate_heterogeneous_optimizer,
    _validate_orchestrator,
    _validate_prediction_synthesizer,
    _validate_resource_optimizer,
    _validate_tool_composer,
    get_quality_gate,
    list_configured_agents,
    run_semantic_validation,
)

# Set testing mode
os.environ["E2I_TESTING_MODE"] = "true"


@pytest.mark.unit
class TestTypedDicts:
    """Test TypedDict definitions."""

    def test_data_quality_check_structure(self):
        """Test DataQualityCheck TypedDict."""
        check: DataQualityCheck = {
            "type": "str",
            "not_null": True,
            "min_length": 10,
        }
        assert check["type"] == "str"

    def test_data_source_requirement_structure(self):
        """Test DataSourceRequirement TypedDict."""
        req: DataSourceRequirement = {
            "reject_mock": True,
            "acceptable_sources": ["supabase", "tier0"],
        }
        assert req["reject_mock"] is True

    def test_agent_quality_gate_structure(self):
        """Test AgentQualityGate TypedDict."""
        gate: AgentQualityGate = {
            "description": "Test agent",
            "required_output_fields": ["status"],
            "fail_on_status": ["error"],
        }
        assert "status" in gate["required_output_fields"]


@pytest.mark.unit
class TestSafeGet:
    """Test _safe_get utility function."""

    def test_safe_get_from_dict(self):
        """Test getting value from dict."""
        obj = {"key": "value", "nested": {"inner": 42}}
        assert _safe_get(obj, "key") == "value"
        assert _safe_get(obj, "missing") is None
        assert _safe_get(obj, "missing", "default") == "default"

    def test_safe_get_from_object(self):
        """Test getting value from object attributes."""

        class TestObj:
            def __init__(self):
                self.attr = "value"

        obj = TestObj()
        assert _safe_get(obj, "attr") == "value"
        assert _safe_get(obj, "missing") is None

    def test_safe_get_from_none(self):
        """Test getting value from None."""
        assert _safe_get(None, "key") is None
        assert _safe_get(None, "key", "default") == "default"


@pytest.mark.unit
class TestToolComposerValidator:
    """Test _validate_tool_composer semantic validator."""

    def test_valid_tool_execution(self):
        """Test validation passes for valid tool execution."""
        output = {
            "response": "Successfully executed causal analysis",
            "confidence": 0.85,
            "tools_executed": 3,
            "tools_succeeded": 3,
            "success": True,
        }
        is_valid, reason = _validate_tool_composer(output)
        assert is_valid is True

    def test_failure_with_unable_to(self):
        """Test validation fails when 'unable to' with low confidence."""
        output = {
            "response": "Unable to assess the causal impact",
            "confidence": 0.3,
        }
        is_valid, reason = _validate_tool_composer(output)
        assert is_valid is False
        assert "unable to" in reason.lower()

    def test_low_confidence_threshold(self):
        """Test validation fails below confidence threshold."""
        output = {"confidence": 0.2}
        is_valid, reason = _validate_tool_composer(output)
        assert is_valid is False
        assert "confidence" in reason.lower()

    def test_all_tools_failed(self):
        """Test validation fails when all tools failed."""
        output = {
            "tools_executed": 5,
            "tools_succeeded": 0,
            "confidence": 0.5,  # Above threshold to avoid early exit
        }
        is_valid, reason = _validate_tool_composer(output)
        assert is_valid is False
        assert "tool" in reason.lower() and ("failed" in reason.lower() or "no successful" in reason.lower())

    def test_reports_success_but_no_tools(self):
        """Test validation fails when reporting success but no tools executed."""
        output = {
            "success": True,
            "tools_executed": 0,
        }
        is_valid, reason = _validate_tool_composer(output)
        assert is_valid is False

    def test_fabricated_sample_size(self):
        """Test detection of fabricated sample sizes."""
        output = {
            "response": "Analysis complete with sample size of 50000 patients",
            "confidence": 0.8,
            "tools_executed": 2,
            "tools_succeeded": 2,
        }
        is_valid, reason = _validate_tool_composer(output)
        assert is_valid is False
        assert "fabricated" in reason.lower()


@pytest.mark.unit
class TestPredictionSynthesizerValidator:
    """Test _validate_prediction_synthesizer semantic validator."""

    def test_single_model_with_warning(self):
        """Test validation passes for single model with warning."""
        output = {
            "models_succeeded": 1,
            "prediction_interpretation": {
                "recommendations": ["Caution: single model, cannot validate predictions"],
                "reliability_assessment": "UNVALIDATED",
            },
        }
        is_valid, reason = _validate_prediction_synthesizer(output)
        assert is_valid is True

    def test_single_model_without_warning(self):
        """Test validation fails for single model without warning."""
        output = {
            "models_succeeded": 1,
            "status": "completed",
        }
        is_valid, reason = _validate_prediction_synthesizer(output)
        assert is_valid is False
        assert "single model" in reason.lower()

    def test_zero_prediction_single_model(self):
        """Test validation fails for zero prediction from single model."""
        output = {
            "models_succeeded": 1,
            "ensemble_prediction": {"point_estimate": 0.0},
            "prediction_interpretation": {"reliability_assessment": "RELIABLE"},
        }
        is_valid, reason = _validate_prediction_synthesizer(output)
        assert is_valid is False

    def test_cannot_assess_in_output(self):
        """Test validation fails for CANNOT_ASSESS in output."""
        output = {
            "prediction_summary": "CANNOT_ASSESS risk level",
            "models_succeeded": 2,
        }
        is_valid, reason = _validate_prediction_synthesizer(output)
        assert is_valid is False


@pytest.mark.unit
class TestDriftMonitorValidator:
    """Test _validate_drift_monitor semantic validator."""

    def test_high_drift_with_actions(self):
        """Test validation passes for high drift with recommended actions."""
        output = {
            "overall_drift_score": 0.85,
            "recommended_actions": ["Retrain model", "Update feature set"],
        }
        is_valid, reason = _validate_drift_monitor(output)
        assert is_valid is True

    def test_high_drift_without_actions(self):
        """Test validation fails for high drift without actions."""
        output = {
            "overall_drift_score": 0.9,
            "recommended_actions": [],
        }
        is_valid, reason = _validate_drift_monitor(output)
        assert is_valid is False

    def test_completed_with_insufficient_drift_types(self):
        """Test validation fails when not enough drift types analyzed."""
        output = {
            "status": "completed",
            "data_drift_results": {"feature_1": 0.5},
            "model_drift_results": {},
            "concept_drift_results": {},
        }
        is_valid, reason = _validate_drift_monitor(output)
        assert is_valid is False


@pytest.mark.unit
class TestExperimentDesignerValidator:
    """Test _validate_experiment_designer semantic validator."""

    def test_valid_sample_size_and_power(self):
        """Test validation passes with calculated values."""
        output = {
            "required_sample_size": 500,
            "statistical_power": 0.8,
        }
        is_valid, reason = _validate_experiment_designer(output)
        assert is_valid is True

    def test_sample_size_na(self):
        """Test validation fails for N/A sample size."""
        output = {
            "required_sample_size": "N/A",
            "statistical_power": 0.8,
        }
        is_valid, reason = _validate_experiment_designer(output)
        assert is_valid is False

    def test_power_na(self):
        """Test validation fails for N/A power."""
        output = {
            "required_sample_size": 500,
            "statistical_power": "N/A",
        }
        is_valid, reason = _validate_experiment_designer(output)
        assert is_valid is False

    def test_nested_power_analysis(self):
        """Test validation with nested power_analysis structure."""
        output = {
            "power_analysis": {
                "required_sample_size": 500,
                "achieved_power": 0.85,
            }
        }
        is_valid, reason = _validate_experiment_designer(output)
        assert is_valid is True


@pytest.mark.unit
class TestHealthScoreValidator:
    """Test _validate_health_score semantic validator."""

    def test_low_score_with_diagnostics(self):
        """Test validation passes with diagnostics for low score."""
        output = {
            "component_health_score": 0.6,
            "health_diagnosis": {
                "root_causes": ["Database latency", "Cache miss rate"],
            },
        }
        is_valid, reason = _validate_health_score(output)
        assert is_valid is True

    def test_low_score_without_diagnostics(self):
        """Test validation fails without diagnostics for low score."""
        output = {
            "component_health_score": 0.5,
        }
        is_valid, reason = _validate_health_score(output)
        assert is_valid is False

    def test_low_score_with_critical_issues(self):
        """Test validation passes with critical_issues as diagnostics."""
        output = {
            "component_health_score": 0.7,
            "critical_issues": ["API timeout", "Database connection pool exhausted"],
        }
        is_valid, reason = _validate_health_score(output)
        assert is_valid is True


@pytest.mark.unit
class TestResourceOptimizerValidator:
    """Test _validate_resource_optimizer semantic validator."""

    def test_completed_with_savings(self):
        """Test validation passes with projected savings."""
        output = {
            "status": "optimal",
            "projected_savings": 50000,
            "projected_roi": 0.25,
        }
        is_valid, reason = _validate_resource_optimizer(output)
        assert is_valid is True

    def test_completed_without_savings(self):
        """Test validation fails without savings or ROI."""
        output = {
            "status": "completed",
            "projected_savings": "N/A",
            "projected_roi": "N/A",
        }
        is_valid, reason = _validate_resource_optimizer(output)
        assert is_valid is False

    def test_negligible_roi(self):
        """Test validation fails for negligible ROI."""
        output = {
            "status": "optimal",
            "projected_roi": 0.02,
        }
        is_valid, reason = _validate_resource_optimizer(output)
        assert is_valid is False


@pytest.mark.unit
class TestExplainerValidator:
    """Test _validate_explainer semantic validator."""

    def test_recommendations_surfaced(self):
        """Test validation passes when recommendations are surfaced."""
        output = {
            "recommendations_count": 5,
            "recommendations": [
                "Increase HCP engagement",
                "Target high-risk segments",
            ],
        }
        is_valid, reason = _validate_explainer(output)
        assert is_valid is True

    def test_recommendations_not_surfaced(self):
        """Test validation fails when recommendations not surfaced."""
        output = {
            "recommendations_count": 5,
            "executive_summary": "Analysis complete with findings.",
        }
        is_valid, reason = _validate_explainer(output)
        assert is_valid is False

    def test_meta_description_rejection(self):
        """Test rejection of meta-descriptions."""
        output = {
            "executive_summary": "Analysis complete with 5 findings",
        }
        is_valid, reason = _validate_explainer(output)
        assert is_valid is False

    def test_raw_float_rejection(self):
        """Test rejection of excessive raw floats."""
        output = {
            "executive_summary": "Score: 0.6583333333 vs baseline 0.7291666667",
            "detailed_explanation": "Confidence: 0.8541666667",
        }
        is_valid, reason = _validate_explainer(output)
        assert is_valid is False


@pytest.mark.unit
class TestHeterogeneousOptimizerValidator:
    """Test _validate_heterogeneous_optimizer semantic validator."""

    def test_valid_with_interpretation(self):
        """Test validation passes with strategic interpretation."""
        output = {
            "overall_ate": 0.15,
            "heterogeneity_score": 0.45,
            "strategic_interpretation": "Significant heterogeneity detected across segments",
        }
        is_valid, reason = _validate_heterogeneous_optimizer(output)
        assert is_valid is True

    def test_missing_interpretation(self):
        """Test validation fails without interpretation."""
        output = {
            "overall_ate": 0.15,
            "heterogeneity_score": 0.45,
        }
        is_valid, reason = _validate_heterogeneous_optimizer(output)
        assert is_valid is False

    def test_no_responder_segments(self):
        """Test validation fails when no segments identified."""
        output = {
            "status": "completed",
            "high_responders": [],
            "low_responders": [],
        }
        is_valid, reason = _validate_heterogeneous_optimizer(output)
        assert is_valid is False


@pytest.mark.unit
class TestFeedbackLearnerValidator:
    """Test _validate_feedback_learner semantic validator."""

    def test_partial_status_accepted(self):
        """Test validation passes for partial status."""
        output = {"status": "partial"}
        is_valid, reason = _validate_feedback_learner(output)
        assert is_valid is True

    def test_completed_with_learning(self):
        """Test validation passes for completed with learning."""
        output = {
            "status": "completed",
            "detected_patterns": ["Pattern 1", "Pattern 2"],
            "learning_recommendations": ["Rec 1"],
        }
        is_valid, reason = _validate_feedback_learner(output)
        assert is_valid is True

    def test_completed_without_learning(self):
        """Test validation fails for completed without learning."""
        output = {
            "status": "completed",
            "detected_patterns": [],
            "learning_recommendations": [],
            "feedback_count": 0,
        }
        is_valid, reason = _validate_feedback_learner(output)
        assert is_valid is False


@pytest.mark.unit
class TestOrchestratorValidator:
    """Test _validate_orchestrator semantic validator."""

    def test_valid_dispatch(self):
        """Test validation passes for valid dispatch."""
        output = {
            "status": "completed",
            "agents_dispatched": ["causal_impact", "gap_analyzer"],
            "response_text": "Analysis completed. The causal impact analysis reveals...",
        }
        is_valid, reason = _validate_orchestrator(output)
        assert is_valid is True

    def test_no_agents_dispatched(self):
        """Test validation fails when no agents dispatched."""
        output = {
            "status": "completed",
            "agents_dispatched": [],
            "response_text": "Query processed",
        }
        is_valid, reason = _validate_orchestrator(output)
        assert is_valid is False

    def test_duplicate_agents(self):
        """Test validation fails for duplicate agent dispatches."""
        output = {
            "status": "completed",
            "agents_dispatched": ["causal_impact", "causal_impact", "gap_analyzer"],
            "response_text": "Analysis completed successfully",
        }
        is_valid, reason = _validate_orchestrator(output)
        assert is_valid is False

    def test_short_response(self):
        """Test validation fails for short response."""
        output = {
            "status": "completed",
            "agents_dispatched": ["causal_impact"],
            "response_text": "Done",
        }
        is_valid, reason = _validate_orchestrator(output)
        assert is_valid is False


@pytest.mark.unit
class TestCausalImpactValidator:
    """Test _validate_causal_impact semantic validator."""

    def test_valid_ate_and_ci(self):
        """Test validation passes for valid ATE and CI."""
        output = {
            "status": "completed",
            "ate_estimate": 0.15,
            "confidence_interval": [0.10, 0.20],
        }
        is_valid, reason = _validate_causal_impact(output)
        assert is_valid is True

    def test_ate_outside_ci(self):
        """Test validation fails when ATE outside its own CI."""
        output = {
            "status": "completed",
            "ate_estimate": 0.30,
            "confidence_interval": [0.10, 0.20],
        }
        is_valid, reason = _validate_causal_impact(output)
        assert is_valid is False

    def test_invalid_ci_format(self):
        """Test validation fails for invalid CI format."""
        output = {
            "status": "completed",
            "ate_estimate": 0.15,
            "confidence_interval": [0.10],  # Only one element
        }
        is_valid, reason = _validate_causal_impact(output)
        assert is_valid is False


@pytest.mark.unit
class TestGapAnalyzerValidator:
    """Test _validate_gap_analyzer semantic validator."""

    def test_valid_opportunities(self):
        """Test validation passes with opportunities."""
        output = {
            "status": "completed",
            "prioritized_opportunities": [
                {"segment": "NE", "value": 50000},
            ],
            "total_addressable_value": 50000,
            "executive_summary": "Identified significant ROI opportunity in the NE region...",
        }
        is_valid, reason = _validate_gap_analyzer(output)
        assert is_valid is True

    def test_no_opportunities(self):
        """Test validation fails without opportunities."""
        output = {
            "status": "completed",
            "prioritized_opportunities": [],
        }
        is_valid, reason = _validate_gap_analyzer(output)
        assert is_valid is False

    def test_zero_value(self):
        """Test validation fails for zero total value."""
        output = {
            "status": "completed",
            "prioritized_opportunities": [{"segment": "NE"}],
            "total_addressable_value": 0,
        }
        is_valid, reason = _validate_gap_analyzer(output)
        assert is_valid is False


@pytest.mark.unit
class TestAgentQualityGates:
    """Test AGENT_QUALITY_GATES configuration."""

    def test_all_agents_have_gates(self):
        """Test that all expected agents have quality gates."""
        expected_agents = [
            "orchestrator",
            "tool_composer",
            "causal_impact",
            "gap_analyzer",
            "heterogeneous_optimizer",
            "drift_monitor",
            "experiment_designer",
            "health_score",
            "prediction_synthesizer",
            "resource_optimizer",
            "explainer",
            "feedback_learner",
        ]

        for agent in expected_agents:
            assert agent in AGENT_QUALITY_GATES

    def test_gates_have_required_fields(self):
        """Test that gates have required structure."""
        for agent_name, gate in AGENT_QUALITY_GATES.items():
            assert "description" in gate
            assert "required_output_fields" in gate
            assert isinstance(gate["required_output_fields"], list)

    def test_gates_have_semantic_validators(self):
        """Test that all gates have semantic validators."""
        for agent_name, gate in AGENT_QUALITY_GATES.items():
            assert "semantic_validator" in gate
            assert callable(gate["semantic_validator"])


@pytest.mark.unit
class TestHelperFunctions:
    """Test helper functions."""

    def test_get_quality_gate(self):
        """Test get_quality_gate function."""
        gate = get_quality_gate("causal_impact")
        assert gate is not None
        assert "required_output_fields" in gate

    def test_get_quality_gate_unknown(self):
        """Test get_quality_gate for unknown agent."""
        gate = get_quality_gate("unknown_agent")
        assert gate is None

    def test_list_configured_agents(self):
        """Test list_configured_agents function."""
        agents = list_configured_agents()
        assert isinstance(agents, list)
        assert "causal_impact" in agents
        assert "orchestrator" in agents

    def test_run_semantic_validation(self):
        """Test run_semantic_validation function."""
        output = {
            "status": "completed",
            "ate_estimate": 0.15,
            "confidence_interval": [0.10, 0.20],
        }
        is_valid, reason = run_semantic_validation("causal_impact", output)
        assert is_valid is True

    def test_run_semantic_validation_no_gate(self):
        """Test run_semantic_validation for agent without gate."""
        is_valid, reason = run_semantic_validation("unknown_agent", {})
        assert is_valid is True
        assert "No quality gate" in reason

    def test_run_semantic_validation_exception(self):
        """Test run_semantic_validation handles exceptions."""

        def failing_validator(output):
            raise ValueError("Test error")

        # Temporarily patch a validator
        original = AGENT_QUALITY_GATES["causal_impact"]["semantic_validator"]
        AGENT_QUALITY_GATES["causal_impact"]["semantic_validator"] = failing_validator

        try:
            is_valid, reason = run_semantic_validation("causal_impact", {})
            assert is_valid is False
            assert "error" in reason.lower()
        finally:
            AGENT_QUALITY_GATES["causal_impact"]["semantic_validator"] = original
