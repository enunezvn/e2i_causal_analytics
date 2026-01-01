"""Digital Twin Workflow Integration Tests.

Tests the end-to-end digital twin workflow integration:
1. Design experiment with twin pre-screening
2. Simulation leads to "deploy" → continues to design
3. Simulation leads to "skip" → early exit
4. Fidelity validation after real experiment

Phase 15: Digital Twin Pre-Screening for A/B Tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.agents.experiment_designer.graph import (
    create_experiment_designer_graph,
    create_initial_state,
)
from src.agents.experiment_designer.nodes import TwinSimulationNode
from src.agents.experiment_designer.state import ExperimentDesignState
from src.digital_twin.models.simulation_models import (
    SimulationRecommendation,
    SimulationResult,
    SimulationStatus,
    InterventionConfig,
    PopulationFilter,
    EffectHeterogeneity,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_simulation_deploy():
    """Mock simulation result that recommends DEPLOY."""
    return {
        "simulation_id": str(uuid4()),
        "recommendation": "deploy",
        "recommendation_rationale": "Simulation predicts positive effect (ATE=0.08). Proceed with A/B test.",
        "simulated_ate": 0.08,
        "confidence_interval": (0.05, 0.11),
        "recommended_sample_size": 1500,
        "recommended_duration_weeks": 8,
        "simulation_confidence": 0.85,
        "fidelity_warning": False,
        "fidelity_warning_reason": None,
        "top_segments": [
            {"dimension": "decile", "segment": "1-2", "ate": 0.12, "n": 2000},
            {"dimension": "specialty", "segment": "oncology", "ate": 0.10, "n": 3000},
        ],
    }


@pytest.fixture
def mock_simulation_skip():
    """Mock simulation result that recommends SKIP."""
    return {
        "simulation_id": str(uuid4()),
        "recommendation": "skip",
        "recommendation_rationale": "Simulated ATE (0.02) below minimum threshold (0.05). Predicted impact insufficient.",
        "simulated_ate": 0.02,
        "confidence_interval": (-0.01, 0.05),
        "recommended_sample_size": None,
        "recommended_duration_weeks": 8,
        "simulation_confidence": 0.75,
        "fidelity_warning": False,
        "fidelity_warning_reason": None,
        "top_segments": [],
    }


@pytest.fixture
def mock_simulation_refine():
    """Mock simulation result that recommends REFINE."""
    return {
        "simulation_id": str(uuid4()),
        "recommendation": "refine",
        "recommendation_rationale": "Effect not statistically significant (CI includes zero). Consider refining intervention.",
        "simulated_ate": 0.04,
        "confidence_interval": (-0.01, 0.09),
        "recommended_sample_size": 2500,
        "recommended_duration_weeks": 12,
        "simulation_confidence": 0.65,
        "fidelity_warning": True,
        "fidelity_warning_reason": "Model fidelity degraded for this intervention type.",
        "top_segments": [],
    }


@pytest.fixture
def initial_state_with_twin():
    """Create initial state with twin simulation enabled."""
    return create_initial_state(
        business_question="Does increasing email frequency improve HCP engagement for Kisqali?",
        intervention_type="email_campaign",
        brand="Kisqali",
        enable_twin_simulation=True,
        constraints={
            "channel": "email",
            "frequency": "weekly",
            "duration_weeks": 8,
            "target_deciles": [1, 2, 3],
        },
    )


@pytest.fixture
def initial_state_without_twin():
    """Create initial state with twin simulation disabled."""
    return create_initial_state(
        business_question="Does increasing email frequency improve HCP engagement for Kisqali?",
        enable_twin_simulation=False,
    )


# =============================================================================
# Test TwinSimulationNode
# =============================================================================


@pytest.mark.xdist_group(name="digital_twin_workflow")
class TestTwinSimulationNode:
    """Test the TwinSimulationNode in isolation."""

    @pytest.mark.asyncio
    async def test_skip_when_twin_simulation_disabled(self, initial_state_without_twin):
        """Test that twin simulation is skipped when disabled."""
        node = TwinSimulationNode()

        result = await node.execute(initial_state_without_twin)

        assert result["status"] == "reasoning"
        assert "twin_simulation_result" not in result

    @pytest.mark.asyncio
    async def test_skip_when_missing_intervention_type(self):
        """Test that twin simulation is skipped when intervention_type is missing."""
        state = create_initial_state(
            business_question="Test question",
            enable_twin_simulation=True,
            brand="Kisqali",
            # No intervention_type
        )

        node = TwinSimulationNode()
        result = await node.execute(state)

        assert result["status"] == "reasoning"
        assert any("missing intervention_type" in w for w in result.get("warnings", []))

    @pytest.mark.asyncio
    async def test_skip_when_missing_brand(self):
        """Test that twin simulation is skipped when brand is missing."""
        state = create_initial_state(
            business_question="Test question",
            enable_twin_simulation=True,
            intervention_type="email_campaign",
            # No brand
        )

        node = TwinSimulationNode()
        result = await node.execute(state)

        assert result["status"] == "reasoning"
        assert any("missing intervention_type or brand" in w for w in result.get("warnings", []))

    @pytest.mark.asyncio
    async def test_deploy_recommendation(self, initial_state_with_twin, mock_simulation_deploy):
        """Test that DEPLOY recommendation continues workflow."""
        node = TwinSimulationNode()

        with patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            return_value=mock_simulation_deploy,
        ):
            result = await node.execute(initial_state_with_twin)

        assert result["status"] == "reasoning"
        assert result.get("skip_experiment") is False
        assert result["twin_recommendation"] == "deploy"
        assert result["twin_simulated_ate"] == 0.08
        assert result["twin_recommended_sample_size"] == 1500

    @pytest.mark.asyncio
    async def test_skip_recommendation_auto_skip(self, initial_state_with_twin, mock_simulation_skip):
        """Test that SKIP recommendation exits workflow when auto_skip is enabled."""
        node = TwinSimulationNode(auto_skip_on_low_effect=True)

        with patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            return_value=mock_simulation_skip,
        ):
            result = await node.execute(initial_state_with_twin)

        assert result["status"] == "skipped"
        assert result.get("skip_experiment") is True
        assert result["twin_recommendation"] == "skip"
        assert any("skipped based on twin simulation" in w for w in result.get("warnings", []))

    @pytest.mark.asyncio
    async def test_skip_recommendation_no_auto_skip(self, initial_state_with_twin, mock_simulation_skip):
        """Test that SKIP recommendation continues workflow when auto_skip is disabled."""
        node = TwinSimulationNode(auto_skip_on_low_effect=False)

        with patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            return_value=mock_simulation_skip,
        ):
            result = await node.execute(initial_state_with_twin)

        assert result["status"] == "reasoning"
        assert result.get("skip_experiment") is False
        assert any("proceeding anyway" in w for w in result.get("warnings", []))

    @pytest.mark.asyncio
    async def test_refine_recommendation(self, initial_state_with_twin, mock_simulation_refine):
        """Test that REFINE recommendation continues with warnings."""
        node = TwinSimulationNode()

        with patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            return_value=mock_simulation_refine,
        ):
            result = await node.execute(initial_state_with_twin)

        assert result["status"] == "reasoning"
        assert result.get("skip_experiment") is False
        assert result["twin_recommendation"] == "refine"
        # Should have fidelity warning
        assert any("fidelity" in w.lower() for w in result.get("warnings", []))

    @pytest.mark.asyncio
    async def test_error_handling(self, initial_state_with_twin):
        """Test that simulation errors are recoverable."""
        node = TwinSimulationNode()

        with patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            side_effect=Exception("Simulation failed"),
        ):
            result = await node.execute(initial_state_with_twin)

        # Should continue with design despite error
        assert result["status"] == "reasoning"
        assert result.get("skip_experiment") is False
        assert len(result.get("errors", [])) > 0
        assert any("Simulation failed" in w for w in result.get("warnings", []))

    @pytest.mark.asyncio
    async def test_latency_tracking(self, initial_state_with_twin, mock_simulation_deploy):
        """Test that node latency is tracked."""
        node = TwinSimulationNode()

        with patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            return_value=mock_simulation_deploy,
        ):
            result = await node.execute(initial_state_with_twin)

        assert "node_latencies_ms" in result
        assert "twin_simulation" in result["node_latencies_ms"]
        assert result["node_latencies_ms"]["twin_simulation"] >= 0


# =============================================================================
# Test Graph Integration
# =============================================================================


@pytest.mark.xdist_group(name="digital_twin_workflow")
class TestGraphIntegration:
    """Test the full experiment designer graph with twin simulation."""

    def test_graph_includes_twin_simulation_node(self):
        """Test that the graph includes the twin_simulation node."""
        graph = create_experiment_designer_graph(enable_twin_simulation=True)

        assert "twin_simulation" in graph.nodes

    def test_graph_workflow_order(self):
        """Test that nodes are in correct order."""
        graph = create_experiment_designer_graph()

        nodes = list(graph.nodes.keys())

        # Verify order: context_loader comes before twin_simulation
        assert nodes.index("context_loader") < nodes.index("twin_simulation")
        # twin_simulation comes before design_reasoning
        assert nodes.index("twin_simulation") < nodes.index("design_reasoning")

    def test_create_initial_state_with_twin_params(self):
        """Test that initial state includes twin simulation parameters."""
        state = create_initial_state(
            business_question="Test question",
            intervention_type="email_campaign",
            brand="Kisqali",
            enable_twin_simulation=True,
        )

        assert state["enable_twin_simulation"] is True
        assert state["intervention_type"] == "email_campaign"
        assert state["brand"] == "Kisqali"

    def test_create_initial_state_without_twin_params(self):
        """Test that initial state works without twin simulation parameters."""
        state = create_initial_state(
            business_question="Test question",
            enable_twin_simulation=False,
        )

        assert state["enable_twin_simulation"] is False
        assert "intervention_type" not in state
        assert "brand" not in state


# =============================================================================
# Test End-to-End Workflow Scenarios
# =============================================================================


@pytest.mark.xdist_group(name="digital_twin_workflow")
class TestWorkflowScenarios:
    """Test complete workflow scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_deploy_continues_to_design(self, initial_state_with_twin, mock_simulation_deploy):
        """Scenario: Simulation recommends DEPLOY, workflow continues to design."""
        # Create node
        node = TwinSimulationNode()

        with patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            return_value=mock_simulation_deploy,
        ):
            result = await node.execute(initial_state_with_twin)

        # Verify state transitions correctly
        assert result["status"] == "reasoning"
        assert not result.get("skip_experiment", False)

        # Verify simulation data is available for power analysis
        assert "twin_simulated_ate" in result
        assert "twin_recommended_sample_size" in result
        assert result["twin_simulated_ate"] > 0.05  # Above threshold

    @pytest.mark.asyncio
    async def test_scenario_skip_exits_early(self, initial_state_with_twin, mock_simulation_skip):
        """Scenario: Simulation recommends SKIP, workflow exits early."""
        node = TwinSimulationNode(auto_skip_on_low_effect=True)

        with patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            return_value=mock_simulation_skip,
        ):
            result = await node.execute(initial_state_with_twin)

        # Verify workflow exits early
        assert result["status"] == "skipped"
        assert result.get("skip_experiment") is True

        # Verify reason is recorded
        assert len(result.get("warnings", [])) > 0

    @pytest.mark.asyncio
    async def test_scenario_refine_continues_with_warnings(
        self, initial_state_with_twin, mock_simulation_refine
    ):
        """Scenario: Simulation recommends REFINE, workflow continues with warnings."""
        node = TwinSimulationNode()

        with patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            return_value=mock_simulation_refine,
        ):
            result = await node.execute(initial_state_with_twin)

        # Verify workflow continues
        assert result["status"] == "reasoning"

        # Verify warnings are present
        warnings = result.get("warnings", [])
        assert any("refine" in w.lower() or "fidelity" in w.lower() for w in warnings)

    @pytest.mark.asyncio
    async def test_scenario_twin_disabled_bypasses_simulation(self, initial_state_without_twin):
        """Scenario: Twin simulation disabled, workflow bypasses simulation."""
        node = TwinSimulationNode()

        result = await node.execute(initial_state_without_twin)

        # Verify workflow proceeds directly to reasoning
        assert result["status"] == "reasoning"
        assert "twin_simulation_result" not in result


# =============================================================================
# Test State Management
# =============================================================================


@pytest.mark.xdist_group(name="digital_twin_workflow")
class TestStateManagement:
    """Test state field management during twin simulation."""

    @pytest.mark.asyncio
    async def test_twin_fields_populated_on_success(self, initial_state_with_twin, mock_simulation_deploy):
        """Test that all twin fields are populated on successful simulation."""
        node = TwinSimulationNode()

        with patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            return_value=mock_simulation_deploy,
        ):
            result = await node.execute(initial_state_with_twin)

        assert "twin_simulation_result" in result
        assert "twin_recommendation" in result
        assert "twin_simulated_ate" in result
        assert "twin_recommended_sample_size" in result
        assert "twin_top_segments" in result

    @pytest.mark.asyncio
    async def test_errors_preserved_on_failure(self, initial_state_with_twin):
        """Test that existing errors are preserved on simulation failure."""
        initial_state_with_twin["errors"] = [
            {"node": "previous_node", "error": "Previous error", "timestamp": "2024-01-01T00:00:00Z"}
        ]

        node = TwinSimulationNode()

        with patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            side_effect=Exception("New error"),
        ):
            result = await node.execute(initial_state_with_twin)

        # Should have both old and new errors
        assert len(result.get("errors", [])) >= 2

    @pytest.mark.asyncio
    async def test_warnings_accumulated(self, initial_state_with_twin, mock_simulation_refine):
        """Test that warnings are accumulated, not replaced."""
        initial_state_with_twin["warnings"] = ["Previous warning"]

        node = TwinSimulationNode()

        with patch(
            "src.agents.experiment_designer.nodes.twin_simulation.simulate_intervention",
            return_value=mock_simulation_refine,
        ):
            result = await node.execute(initial_state_with_twin)

        warnings = result.get("warnings", [])
        assert "Previous warning" in warnings
        assert len(warnings) > 1  # Should have accumulated new warnings


# =============================================================================
# Test Simulation Parameter Extraction
# =============================================================================


@pytest.mark.xdist_group(name="digital_twin_workflow")
class TestParameterExtraction:
    """Test extraction of simulation parameters from state/constraints."""

    @pytest.mark.asyncio
    async def test_extracts_channel_from_constraints(self):
        """Test that channel is extracted from constraints."""
        state = create_initial_state(
            business_question="Test",
            intervention_type="email_campaign",
            brand="Kisqali",
            enable_twin_simulation=True,
            constraints={"channel": "email"},
        )

        node = TwinSimulationNode()
        params = node._extract_simulation_params(state, state.get("constraints", {}))

        assert params.get("channel") == "email"

    @pytest.mark.asyncio
    async def test_extracts_target_filters(self):
        """Test that targeting filters are extracted."""
        state = create_initial_state(
            business_question="Test",
            intervention_type="email_campaign",
            brand="Kisqali",
            enable_twin_simulation=True,
            constraints={
                "target_deciles": [1, 2],
                "target_specialties": ["oncology"],
                "target_regions": ["northeast"],
            },
        )

        node = TwinSimulationNode()
        params = node._extract_simulation_params(state, state.get("constraints", {}))

        assert params.get("target_deciles") == [1, 2]
        assert params.get("target_specialties") == ["oncology"]
        assert params.get("target_regions") == ["northeast"]

    @pytest.mark.asyncio
    async def test_uses_default_twin_count(self):
        """Test that default twin count is used when not specified."""
        state = create_initial_state(
            business_question="Test",
            intervention_type="email_campaign",
            brand="Kisqali",
            enable_twin_simulation=True,
        )

        node = TwinSimulationNode()
        params = node._extract_simulation_params(state, state.get("constraints", {}))

        # Default should be used (not in params, will use class default)
        assert "twin_count" not in params or params.get("twin_count") == TwinSimulationNode.DEFAULT_TWIN_COUNT
