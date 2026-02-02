"""
Integration tests for audit chain across all agent tiers.

Tests verify:
1. Audit chain initialization in LangGraph agents (Tiers 1-5)
2. Audit chain integration in class-based Tier 0 pipeline
3. Cross-tier workflow verification
4. Graceful degradation when audit service unavailable
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.agents.base.audit_chain_mixin import (
    create_workflow_initializer,
    set_audit_chain_service,
)
from src.utils.audit_chain import (
    AgentTier,
    AuditChainEntry,
    AuditChainService,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client for testing."""
    client = MagicMock()
    mock_table = MagicMock()
    client.table.return_value = mock_table
    mock_table.insert.return_value = mock_table
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.order.return_value = mock_table
    mock_table.limit.return_value = mock_table
    mock_table.execute.return_value = MagicMock(data=[])
    return client


@pytest.fixture
def audit_service(mock_supabase_client):
    """Create an AuditChainService with mock client."""
    return AuditChainService(mock_supabase_client)


@pytest.fixture
def sample_workflow_id():
    """Return a sample workflow ID."""
    return uuid4()


@pytest.fixture
def sample_entry(sample_workflow_id):
    """Create a sample AuditChainEntry."""
    return AuditChainEntry(
        entry_id=uuid4(),
        workflow_id=sample_workflow_id,
        sequence_number=1,
        agent_name="test_agent",
        agent_tier=AgentTier.CAUSAL_ANALYTICS.value,
        action_type="initialization",
        created_at=datetime.now(),
        entry_hash="abc123",
    )


@pytest.fixture(autouse=True)
def reset_global_service():
    """Reset the global audit service before/after each test."""
    set_audit_chain_service(None)
    yield
    set_audit_chain_service(None)


# =============================================================================
# Tier Integration Tests - LangGraph Agents
# =============================================================================


class TestTier1CoordinationIntegration:
    """Tests for Tier 1 (Coordination) agent audit chain integration."""

    def test_orchestrator_graph_has_audit_init(self):
        """Verify orchestrator graph includes audit_init node."""
        from src.agents.orchestrator.graph import create_orchestrator_graph

        graph = create_orchestrator_graph()
        # Get nodes from the compiled graph
        nodes = list(graph.nodes.keys())

        assert "audit_init" in nodes, "Orchestrator should have audit_init node"

    def test_orchestrator_initializer_tier(self, audit_service, sample_entry):
        """Verify orchestrator uses correct tier."""
        audit_service.start_workflow = MagicMock(return_value=sample_entry)
        set_audit_chain_service(audit_service)

        initializer = create_workflow_initializer("orchestrator", AgentTier.COORDINATION)
        state = {"query": "test", "session_id": "test-123"}
        initializer(state)

        # Verify workflow was started with correct tier
        call_kwargs = audit_service.start_workflow.call_args.kwargs
        assert call_kwargs["agent_tier"] == AgentTier.COORDINATION


class TestTier2CausalIntegration:
    """Tests for Tier 2 (Causal Analytics) agent audit chain integration."""

    def test_causal_impact_graph_has_audit_init(self):
        """Verify causal_impact graph includes audit_init node."""
        from src.agents.causal_impact.graph import create_causal_impact_graph

        graph = create_causal_impact_graph()
        nodes = list(graph.nodes.keys())

        assert "audit_init" in nodes, "Causal Impact should have audit_init node"

    def test_gap_analyzer_graph_has_audit_init(self):
        """Verify gap_analyzer graph includes audit_init node."""
        from src.agents.gap_analyzer.graph import create_gap_analyzer_graph

        graph = create_gap_analyzer_graph()
        nodes = list(graph.nodes.keys())

        assert "audit_init" in nodes, "Gap Analyzer should have audit_init node"

    def test_heterogeneous_optimizer_graph_has_audit_init(self):
        """Verify heterogeneous_optimizer graph includes audit_init node."""
        from src.agents.heterogeneous_optimizer.graph import (
            create_heterogeneous_optimizer_graph,
        )

        graph = create_heterogeneous_optimizer_graph()
        nodes = list(graph.nodes.keys())

        assert "audit_init" in nodes, "Heterogeneous Optimizer should have audit_init node"

    def test_causal_initializer_tier(self, audit_service, sample_entry):
        """Verify causal agents use correct tier."""
        audit_service.start_workflow = MagicMock(return_value=sample_entry)
        set_audit_chain_service(audit_service)

        initializer = create_workflow_initializer("causal_impact", AgentTier.CAUSAL_ANALYTICS)
        state = {"query": "test", "treatment_var": "x", "outcome_var": "y"}
        initializer(state)

        call_kwargs = audit_service.start_workflow.call_args.kwargs
        assert call_kwargs["agent_tier"] == AgentTier.CAUSAL_ANALYTICS


class TestTier3MonitoringIntegration:
    """Tests for Tier 3 (Monitoring) agent audit chain integration."""

    def test_drift_monitor_graph_has_audit_init(self):
        """Verify drift_monitor graph includes audit_init node."""
        from src.agents.drift_monitor.graph import create_drift_monitor_graph

        graph = create_drift_monitor_graph()
        nodes = list(graph.nodes.keys())

        assert "audit_init" in nodes, "Drift Monitor should have audit_init node"

    def test_experiment_designer_graph_has_audit_init(self):
        """Verify experiment_designer graph includes audit_init node."""
        from src.agents.experiment_designer.graph import create_experiment_designer_graph

        graph = create_experiment_designer_graph()
        nodes = list(graph.nodes.keys())

        assert "audit_init" in nodes, "Experiment Designer should have audit_init node"

    def test_health_score_graph_has_audit_init(self):
        """Verify health_score graph includes audit_init node."""
        from src.agents.health_score.graph import build_health_score_graph

        graph = build_health_score_graph()
        nodes = list(graph.nodes.keys())

        assert "audit_init" in nodes, "Health Score should have audit_init node"

    def test_monitoring_initializer_tier(self, audit_service, sample_entry):
        """Verify monitoring agents use correct tier."""
        audit_service.start_workflow = MagicMock(return_value=sample_entry)
        set_audit_chain_service(audit_service)

        initializer = create_workflow_initializer("drift_monitor", AgentTier.MONITORING)
        state = {"model_id": "test-model"}
        initializer(state)

        call_kwargs = audit_service.start_workflow.call_args.kwargs
        assert call_kwargs["agent_tier"] == AgentTier.MONITORING


class TestTier4MLPredictionsIntegration:
    """Tests for Tier 4 (ML Predictions) agent audit chain integration."""

    def test_prediction_synthesizer_graph_has_audit_init(self):
        """Verify prediction_synthesizer graph includes audit_init node."""
        from src.agents.prediction_synthesizer.graph import (
            build_prediction_synthesizer_graph,
        )

        graph = build_prediction_synthesizer_graph()
        nodes = list(graph.nodes.keys())

        assert "audit_init" in nodes, "Prediction Synthesizer should have audit_init node"

    def test_resource_optimizer_graph_has_audit_init(self):
        """Verify resource_optimizer graph includes audit_init node."""
        from src.agents.resource_optimizer.graph import build_resource_optimizer_graph

        graph = build_resource_optimizer_graph()
        nodes = list(graph.nodes.keys())

        assert "audit_init" in nodes, "Resource Optimizer should have audit_init node"

    def test_ml_predictions_initializer_tier(self, audit_service, sample_entry):
        """Verify ML predictions agents use correct tier."""
        audit_service.start_workflow = MagicMock(return_value=sample_entry)
        set_audit_chain_service(audit_service)

        initializer = create_workflow_initializer(
            "prediction_synthesizer", AgentTier.ML_PREDICTIONS
        )
        state = {"entity_id": "test-entity"}
        initializer(state)

        call_kwargs = audit_service.start_workflow.call_args.kwargs
        assert call_kwargs["agent_tier"] == AgentTier.ML_PREDICTIONS


class TestTier5SelfImprovementIntegration:
    """Tests for Tier 5 (Self-Improvement) agent audit chain integration."""

    def test_explainer_graph_has_audit_init(self):
        """Verify explainer graph includes audit_init node."""
        from src.agents.explainer.graph import build_explainer_graph

        graph = build_explainer_graph(use_default_checkpointer=False)
        nodes = list(graph.nodes.keys())

        assert "audit_init" in nodes, "Explainer should have audit_init node"

    def test_feedback_learner_graph_has_audit_init(self):
        """Verify feedback_learner graph includes audit_init node."""
        from src.agents.feedback_learner.graph import build_feedback_learner_graph

        graph = build_feedback_learner_graph()
        nodes = list(graph.nodes.keys())

        assert "audit_init" in nodes, "Feedback Learner should have audit_init node"

    def test_self_improvement_initializer_tier(self, audit_service, sample_entry):
        """Verify self-improvement agents use correct tier."""
        audit_service.start_workflow = MagicMock(return_value=sample_entry)
        set_audit_chain_service(audit_service)

        initializer = create_workflow_initializer("explainer", AgentTier.SELF_IMPROVEMENT)
        state = {"analysis_results": {}}
        initializer(state)

        call_kwargs = audit_service.start_workflow.call_args.kwargs
        assert call_kwargs["agent_tier"] == AgentTier.SELF_IMPROVEMENT


# =============================================================================
# Tier 0 (ML Foundation) Integration Tests - Class-Based Pipeline
# =============================================================================


class TestTier0MLFoundationIntegration:
    """Tests for Tier 0 (ML Foundation) class-based pipeline audit chain."""

    def test_pipeline_result_has_audit_workflow_id_field(self):
        """Verify PipelineResult dataclass includes audit_workflow_id."""
        from src.agents.tier_0.pipeline import PipelineResult, PipelineStage

        # Create a minimal result to check the field exists
        result = PipelineResult(
            pipeline_run_id=str(uuid4()),
            status="pending",
            current_stage=PipelineStage.SCOPE_DEFINITION,
        )

        assert hasattr(result, "audit_workflow_id"), (
            "PipelineResult should have audit_workflow_id field"
        )

    def test_pipeline_has_audit_service_attribute(self):
        """Verify MLFoundationPipeline has _audit_service attribute."""
        from src.agents.tier_0.pipeline import MLFoundationPipeline

        pipeline = MLFoundationPipeline()

        assert hasattr(pipeline, "_audit_service"), (
            "MLFoundationPipeline should have _audit_service attribute"
        )

    def test_pipeline_has_get_audit_service_method(self):
        """Verify MLFoundationPipeline has _get_audit_service method."""
        from src.agents.tier_0.pipeline import MLFoundationPipeline

        pipeline = MLFoundationPipeline()

        assert hasattr(pipeline, "_get_audit_service"), (
            "MLFoundationPipeline should have _get_audit_service method"
        )
        assert callable(pipeline._get_audit_service)

    def test_pipeline_has_record_audit_entry_method(self):
        """Verify MLFoundationPipeline has _record_audit_entry method."""
        from src.agents.tier_0.pipeline import MLFoundationPipeline

        pipeline = MLFoundationPipeline()

        assert hasattr(pipeline, "_record_audit_entry"), (
            "MLFoundationPipeline should have _record_audit_entry method"
        )
        assert callable(pipeline._record_audit_entry)


# =============================================================================
# Cross-Tier Workflow Tests
# =============================================================================


class TestCrossTierWorkflow:
    """Tests for audit chain across multiple tiers."""

    def test_workflow_id_propagates_through_initializer(self, audit_service, sample_entry):
        """Verify workflow_id is added to state by initializer."""
        audit_service.start_workflow = MagicMock(return_value=sample_entry)
        set_audit_chain_service(audit_service)

        initializer = create_workflow_initializer("orchestrator", AgentTier.COORDINATION)
        state: Dict[str, Any] = {"query": "test"}
        result = initializer(state)

        assert "audit_workflow_id" in result
        assert result["audit_workflow_id"] == sample_entry.workflow_id

    def test_initializer_preserves_existing_state(self, audit_service, sample_entry):
        """Verify initializer doesn't overwrite existing state fields."""
        audit_service.start_workflow = MagicMock(return_value=sample_entry)
        set_audit_chain_service(audit_service)

        initializer = create_workflow_initializer("causal_impact", AgentTier.CAUSAL_ANALYTICS)
        state = {
            "query": "original query",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "custom_field": "preserved",
        }
        result = initializer(state)

        assert result["query"] == "original query"
        assert result["treatment_var"] == "treatment"
        assert result["custom_field"] == "preserved"


# =============================================================================
# Graceful Degradation Tests
# =============================================================================


class TestGracefulDegradation:
    """Tests for graceful degradation when audit service unavailable."""

    def test_initializer_returns_original_state_when_service_unavailable(self):
        """Initializer should return original state when service not available."""
        # Service is None (not set)
        initializer = create_workflow_initializer("orchestrator", AgentTier.COORDINATION)
        state = {"query": "test"}
        result = initializer(state)

        # Should return original state unchanged
        assert result == state
        assert "audit_workflow_id" not in result

    def test_initializer_handles_service_exception(self, audit_service):
        """Initializer should handle exceptions gracefully."""
        audit_service.start_workflow = MagicMock(
            side_effect=Exception("Database connection failed")
        )
        set_audit_chain_service(audit_service)

        initializer = create_workflow_initializer("causal_impact", AgentTier.CAUSAL_ANALYTICS)
        state = {"query": "test"}
        result = initializer(state)

        # Should return original state, not crash
        assert result == state

    def test_agents_work_without_audit_workflow_id(self):
        """Verify agents function when audit_workflow_id is not in state."""
        # This tests that agents don't crash when audit chain is disabled
        from src.agents.causal_impact.state import CausalImpactState

        # Create a state without audit_workflow_id
        state = CausalImpactState(
            query="What is the impact of treatment?",
            treatment_var="treatment",
            outcome_var="outcome",
        )

        # Should not raise - audit_workflow_id is Optional
        assert state.get("audit_workflow_id") is None


# =============================================================================
# All-Tiers Coverage Summary Tests
# =============================================================================


class TestAllTiersCoverage:
    """Summary tests verifying all tiers have audit chain integration."""

    def test_all_langgraph_agents_have_audit_init(self):
        """Verify all LangGraph-based agents have audit_init node."""
        # Tier 1 - Coordination (note: tool_composer uses different architecture)
        # Tier 2 - Causal Analytics
        from src.agents.causal_impact.graph import create_causal_impact_graph

        # Tier 3 - Monitoring
        from src.agents.drift_monitor.graph import create_drift_monitor_graph
        from src.agents.experiment_designer.graph import create_experiment_designer_graph

        # Tier 5 - Self-Improvement
        from src.agents.explainer.graph import build_explainer_graph
        from src.agents.feedback_learner.graph import build_feedback_learner_graph
        from src.agents.gap_analyzer.graph import create_gap_analyzer_graph
        from src.agents.health_score.graph import build_health_score_graph
        from src.agents.heterogeneous_optimizer.graph import (
            create_heterogeneous_optimizer_graph,
        )
        from src.agents.orchestrator.graph import create_orchestrator_graph

        # Tier 4 - ML Predictions
        from src.agents.prediction_synthesizer.graph import (
            build_prediction_synthesizer_graph,
        )
        from src.agents.resource_optimizer.graph import build_resource_optimizer_graph

        agents = {
            # Tier 1 - Coordination (tool_composer uses composer.py, not LangGraph)
            "orchestrator": create_orchestrator_graph(),
            # Tier 2 - Causal Analytics
            "causal_impact": create_causal_impact_graph(),
            "gap_analyzer": create_gap_analyzer_graph(),
            "heterogeneous_optimizer": create_heterogeneous_optimizer_graph(),
            # Tier 3 - Monitoring
            "drift_monitor": create_drift_monitor_graph(),
            "experiment_designer": create_experiment_designer_graph(),
            "health_score": build_health_score_graph(),
            # Tier 4 - ML Predictions
            "prediction_synthesizer": build_prediction_synthesizer_graph(),
            "resource_optimizer": build_resource_optimizer_graph(),
            # Tier 5 - Self-Improvement
            "explainer": build_explainer_graph(use_default_checkpointer=False),
            "feedback_learner": build_feedback_learner_graph(),
        }

        missing_audit_init = []
        for agent_name, graph in agents.items():
            nodes = list(graph.nodes.keys())
            if "audit_init" not in nodes:
                missing_audit_init.append(agent_name)

        assert not missing_audit_init, (
            f"The following agents are missing audit_init node: {missing_audit_init}"
        )

    def test_tier0_pipeline_has_audit_integration(self):
        """Verify Tier 0 ML Foundation Pipeline has audit chain integration."""
        from src.agents.tier_0.pipeline import (
            MLFoundationPipeline,
            PipelineResult,
            PipelineStage,
        )

        pipeline = MLFoundationPipeline()

        # Check pipeline has required audit methods/attributes
        required = [
            "_audit_service",
            "_get_audit_service",
            "_record_audit_entry",
        ]
        missing = [attr for attr in required if not hasattr(pipeline, attr)]

        assert not missing, f"MLFoundationPipeline missing audit attributes: {missing}"

        # Check PipelineResult has audit_workflow_id
        result = PipelineResult(
            pipeline_run_id=str(uuid4()),
            status="pending",
            current_stage=PipelineStage.SCOPE_DEFINITION,
        )
        assert hasattr(result, "audit_workflow_id"), (
            "PipelineResult missing audit_workflow_id field"
        )

    def test_correct_tier_mapping(self, audit_service, sample_entry):
        """Verify each agent uses the correct AgentTier."""
        audit_service.start_workflow = MagicMock(return_value=sample_entry)
        set_audit_chain_service(audit_service)

        tier_mappings = [
            # Tier 1
            ("orchestrator", AgentTier.COORDINATION),
            ("tool_composer", AgentTier.COORDINATION),
            # Tier 2
            ("causal_impact", AgentTier.CAUSAL_ANALYTICS),
            ("gap_analyzer", AgentTier.CAUSAL_ANALYTICS),
            ("heterogeneous_optimizer", AgentTier.CAUSAL_ANALYTICS),
            # Tier 3
            ("drift_monitor", AgentTier.MONITORING),
            ("experiment_designer", AgentTier.MONITORING),
            ("health_score", AgentTier.MONITORING),
            # Tier 4
            ("prediction_synthesizer", AgentTier.ML_PREDICTIONS),
            ("resource_optimizer", AgentTier.ML_PREDICTIONS),
            # Tier 5
            ("explainer", AgentTier.SELF_IMPROVEMENT),
            ("feedback_learner", AgentTier.SELF_IMPROVEMENT),
        ]

        for agent_name, expected_tier in tier_mappings:
            audit_service.start_workflow.reset_mock()

            initializer = create_workflow_initializer(agent_name, expected_tier)
            initializer({"query": "test"})

            call_kwargs = audit_service.start_workflow.call_args.kwargs
            actual_tier = call_kwargs["agent_tier"]

            assert actual_tier == expected_tier, (
                f"{agent_name} should use {expected_tier}, got {actual_tier}"
            )
