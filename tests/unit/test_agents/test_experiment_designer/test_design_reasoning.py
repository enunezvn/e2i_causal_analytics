"""Tests for Design Reasoning Node.

Tests the LLM-based experiment design reasoning functionality.
LLM calls are mocked via the conftest.py autouse fixture (mock_llm_factory).
"""

import pytest

from src.agents.experiment_designer.graph import create_initial_state
from src.agents.experiment_designer.nodes.design_reasoning import DesignReasoningNode


@pytest.mark.xdist_group(name="design_reasoning")
class TestDesignReasoningNode:
    """Test DesignReasoningNode functionality.

    Uses xdist_group to prevent event loop conflicts in parallel execution.
    """

    def test_create_node(self):
        """Test creating node."""
        node = DesignReasoningNode()

        assert node is not None
        assert hasattr(node, "llm")
        assert hasattr(node, "fallback_llm")

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic execution."""
        node = DesignReasoningNode()
        state = create_initial_state(
            business_question="Does increasing visit frequency improve engagement?"
        )
        state["status"] = "designing"

        result = await node.execute(state)

        assert result["status"] == "calculating"
        assert result.get("design_type") is not None
        assert result.get("design_rationale") is not None
        assert "design_reasoning" in result.get("node_latencies_ms", {})

    @pytest.mark.asyncio
    async def test_execute_returns_treatments(self):
        """Test that treatments are returned."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Test treatment definition")
        state["status"] = "designing"

        result = await node.execute(state)

        assert "treatments" in result
        assert len(result["treatments"]) >= 1

    @pytest.mark.asyncio
    async def test_execute_returns_outcomes(self):
        """Test that outcomes are returned."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Test outcome definition")
        state["status"] = "designing"

        result = await node.execute(state)

        assert "outcomes" in result
        assert len(result["outcomes"]) >= 1

    @pytest.mark.asyncio
    async def test_execute_returns_randomization(self):
        """Test that randomization settings are returned."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Test randomization settings")
        state["status"] = "designing"

        result = await node.execute(state)

        assert "randomization_unit" in result
        assert "randomization_method" in result

    @pytest.mark.asyncio
    async def test_execute_returns_stratification(self):
        """Test that stratification variables are returned."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Test stratification variables")
        state["status"] = "designing"

        result = await node.execute(state)

        assert "stratification_variables" in result
        assert isinstance(result["stratification_variables"], list)

    @pytest.mark.asyncio
    async def test_execute_skip_on_failed(self):
        """Test execution skips on failed status."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Test skip on failed")
        state["status"] = "failed"

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_with_constraints(self):
        """Test execution with constraints."""
        node = DesignReasoningNode()
        state = create_initial_state(
            business_question="Test with constraints",
            constraints={"expected_effect_size": 0.25, "power": 0.80, "budget": 100000},
        )
        state["status"] = "designing"

        result = await node.execute(state)

        assert result["status"] == "calculating"

    @pytest.mark.asyncio
    async def test_execute_with_historical_context(self):
        """Test execution with historical experiment context."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Test with historical context")
        state["status"] = "designing"
        state["historical_experiments"] = [
            {
                "experiment_id": "exp_001",
                "design_type": "RCT",
                "outcome": "Positive",
                "lessons": "Use stratified randomization",
            }
        ]

        result = await node.execute(state)

        assert result["status"] == "calculating"

    @pytest.mark.asyncio
    async def test_execute_records_latency(self):
        """Test that node latency is recorded."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Test latency recording")
        state["status"] = "designing"

        result = await node.execute(state)

        assert "node_latencies_ms" in result
        assert "design_reasoning" in result["node_latencies_ms"]
        assert result["node_latencies_ms"]["design_reasoning"] >= 0


@pytest.mark.xdist_group(name="design_reasoning")
class TestDesignReasoningDesignTypes:
    """Test different design type outputs."""

    @pytest.mark.asyncio
    async def test_rct_design(self):
        """Test RCT design output."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Randomized experiment on visit frequency")
        state["status"] = "designing"

        result = await node.execute(state)

        # Design type comparison is case-insensitive
        design_type_normalized = result["design_type"].lower().replace("-", "_")
        assert design_type_normalized in [
            "rct",
            "cluster_rct",
            "quasi_experimental",
            "observational",
        ]

    @pytest.mark.asyncio
    async def test_cluster_rct_context(self):
        """Test cluster RCT design with cluster context."""
        node = DesignReasoningNode()
        state = create_initial_state(
            business_question="Territory-level marketing intervention",
            constraints={"cluster_size": 50, "expected_icc": 0.05},
        )
        state["status"] = "designing"

        result = await node.execute(state)

        assert result["status"] == "calculating"

    @pytest.mark.asyncio
    async def test_quasi_experimental_context(self):
        """Test quasi-experimental design context."""
        node = DesignReasoningNode()
        state = create_initial_state(
            business_question="Impact of policy change on prescribing",
            constraints={"ethical": "Cannot randomize due to policy rollout"},
        )
        state["status"] = "designing"

        result = await node.execute(state)

        assert result["status"] == "calculating"


@pytest.mark.xdist_group(name="design_reasoning")
class TestDesignReasoningOutputValidation:
    """Test design reasoning output validation."""

    @pytest.mark.asyncio
    async def test_treatment_structure(self):
        """Test treatment structure is valid."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Test treatment structure")
        state["status"] = "designing"

        result = await node.execute(state)

        for treatment in result.get("treatments", []):
            assert "name" in treatment
            assert "description" in treatment

    @pytest.mark.asyncio
    async def test_outcome_structure(self):
        """Test outcome structure is valid."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Test outcome structure")
        state["status"] = "designing"

        result = await node.execute(state)

        for outcome in result.get("outcomes", []):
            assert "name" in outcome
            assert "metric_type" in outcome

    @pytest.mark.asyncio
    async def test_has_primary_outcome(self):
        """Test that at least one primary outcome exists."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Test primary outcome")
        state["status"] = "designing"

        result = await node.execute(state)

        outcomes = result.get("outcomes", [])
        if outcomes:
            primary_outcomes = [o for o in outcomes if o.get("is_primary")]
            assert len(primary_outcomes) >= 1, "Should have at least one primary outcome"


@pytest.mark.xdist_group(name="design_reasoning")
class TestDesignReasoningErrorHandling:
    """Test design reasoning error handling."""

    @pytest.mark.asyncio
    async def test_recoverable_error(self):
        """Test recoverable error handling."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Test error handling")
        state["status"] = "designing"
        # Inject invalid historical data to potentially cause issues
        state["historical_experiments"] = "invalid"

        result = await node.execute(state)

        # Should either succeed or record error
        assert result["status"] in ["calculating", "failed"]

    @pytest.mark.asyncio
    async def test_preserves_state_on_error(self):
        """Test that input state is preserved on error."""
        node = DesignReasoningNode()
        state = create_initial_state(
            business_question="Test state preservation", constraints={"budget": 50000}
        )
        state["status"] = "designing"

        result = await node.execute(state)

        assert result["business_question"] == "Test state preservation"
        assert result["constraints"]["budget"] == 50000


@pytest.mark.xdist_group(name="design_reasoning")
class TestDesignReasoningPerformance:
    """Test design reasoning performance characteristics."""

    @pytest.mark.asyncio
    async def test_latency_under_target(self):
        """Test design reasoning completes under 30s target."""
        node = DesignReasoningNode()
        state = create_initial_state(business_question="Test latency performance")
        state["status"] = "designing"

        result = await node.execute(state)

        latency = result["node_latencies_ms"]["design_reasoning"]
        # Mock LLM should be very fast; real target is 30s
        assert latency < 30_000, f"Design reasoning took {latency}ms, exceeds 30s target"
