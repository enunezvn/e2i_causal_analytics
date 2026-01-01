"""Tests for Context Loader Node.

Tests the organizational learning context loading functionality.
"""


import pytest

from src.agents.experiment_designer.graph import create_initial_state
from src.agents.experiment_designer.nodes.context_loader import (
    ContextLoaderNode,
    MockKnowledgeStore,
)


class TestContextLoaderNode:
    """Test ContextLoaderNode functionality."""

    def test_create_node_default(self):
        """Test creating node with default settings."""
        node = ContextLoaderNode()

        assert node is not None
        assert node.knowledge_store is not None  # Uses MockKnowledgeStore by default

    def test_create_node_with_store(self):
        """Test creating node with knowledge store."""
        store = MockKnowledgeStore()
        node = ContextLoaderNode(knowledge_store=store)

        assert node.knowledge_store is store

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic execution with default mock store."""
        node = ContextLoaderNode()
        state = create_initial_state(business_question="Test question here")

        result = await node.execute(state)

        # Uses MockKnowledgeStore by default which returns mock data
        assert result["status"] == "reasoning"  # Next status after context loading
        assert "context_loader" in result.get("node_latencies_ms", {})
        assert "domain_knowledge" in result
        assert "historical_experiments" in result

    @pytest.mark.asyncio
    async def test_execute_loads_historical_experiments(self):
        """Test that historical experiments are loaded."""
        node = ContextLoaderNode()
        state = create_initial_state(business_question="Test experiment question")

        result = await node.execute(state)

        # MockKnowledgeStore returns mock experiments
        assert len(result.get("historical_experiments", [])) > 0

    @pytest.mark.asyncio
    async def test_execute_with_brand(self):
        """Test execution with brand filter."""
        node = ContextLoaderNode()
        state = create_initial_state(business_question="Test Remibrutinib experiment")
        state["brand"] = "Remibrutinib"

        result = await node.execute(state)

        assert result["status"] == "reasoning"

    @pytest.mark.asyncio
    async def test_execute_loads_defaults(self):
        """Test that organizational defaults are loaded."""
        node = ContextLoaderNode()
        state = create_initial_state(business_question="Test defaults loading")

        result = await node.execute(state)

        # MockKnowledgeStore loads defaults into domain_knowledge
        domain = result.get("domain_knowledge", {})
        assert "organizational_defaults" in domain

    @pytest.mark.asyncio
    async def test_execute_loads_violations(self):
        """Test that past violations are loaded as warnings."""
        node = ContextLoaderNode()
        state = create_initial_state(business_question="Test violations loading")

        result = await node.execute(state)

        # Violations should be loaded as warnings
        warnings = result.get("warnings", [])
        # MockKnowledgeStore returns assumption violations that become warnings
        assert len(warnings) > 0 or "regulatory_requirements" in result

    @pytest.mark.asyncio
    async def test_execute_records_latency(self):
        """Test that node latency is recorded."""
        node = ContextLoaderNode()
        state = create_initial_state(business_question="Test latency recording")

        result = await node.execute(state)

        assert "node_latencies_ms" in result
        assert "context_loader" in result["node_latencies_ms"]
        assert result["node_latencies_ms"]["context_loader"] >= 0

    @pytest.mark.asyncio
    async def test_execute_updates_status(self):
        """Test that node updates status correctly."""
        node = ContextLoaderNode()
        state = create_initial_state(business_question="Test status update")

        result = await node.execute(state)

        # Status should transition to "reasoning" for next node
        assert result["status"] == "reasoning"

    @pytest.mark.asyncio
    async def test_execute_skip_on_failed(self):
        """Test execution skips on failed status."""
        node = ContextLoaderNode()
        state = create_initial_state(business_question="Test skip on failed")
        state["status"] = "failed"

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_error_handling(self):
        """Test error handling during execution."""

        class FailingStore:
            async def get_similar_experiments(self, *args, **kwargs):
                raise Exception("Store error")

            async def get_organizational_defaults(self, *args, **kwargs):
                raise Exception("Store error")

            async def get_recent_assumption_violations(self, *args, **kwargs):
                raise Exception("Store error")

            async def get_domain_knowledge(self, *args, **kwargs):
                raise Exception("Store error")

        node = ContextLoaderNode(knowledge_store=FailingStore())
        state = create_initial_state(business_question="Test error handling")

        result = await node.execute(state)

        # Should handle error gracefully - continues but adds error/warning
        assert len(result.get("errors", [])) > 0 or len(result.get("warnings", [])) > 0
        # Should still transition to reasoning despite error (recoverable)
        assert result["status"] == "reasoning"


class TestContextLoaderWithVariousInputs:
    """Test context loader with various input configurations."""

    @pytest.mark.asyncio
    async def test_execute_with_constraints(self):
        """Test execution with constraints in state."""
        node = ContextLoaderNode()
        state = create_initial_state(
            business_question="Test with constraints",
            constraints={"expected_effect_size": 0.25, "power": 0.80},
        )

        result = await node.execute(state)

        assert result["status"] == "reasoning"
        assert result["constraints"] == {"expected_effect_size": 0.25, "power": 0.80}

    @pytest.mark.asyncio
    async def test_execute_with_available_data(self):
        """Test execution with available data in state."""
        node = ContextLoaderNode()
        state = create_initial_state(
            business_question="Test with available data",
            available_data={"variables": ["var1", "var2"], "sample_size": 5000},
        )

        result = await node.execute(state)

        assert result["status"] == "reasoning"
        assert result["available_data"]["sample_size"] == 5000

    @pytest.mark.asyncio
    async def test_execute_preserves_input_fields(self):
        """Test that input fields are preserved through execution."""
        node = ContextLoaderNode()
        state = create_initial_state(
            business_question="Original question",
            constraints={"budget": 50000},
            available_data={"variables": ["x", "y"]},
            preregistration_formality="heavy",
            max_redesign_iterations=3,
            enable_validity_audit=False,
        )

        result = await node.execute(state)

        assert result["business_question"] == "Original question"
        assert result["constraints"]["budget"] == 50000
        assert result["available_data"]["variables"] == ["x", "y"]
        assert result["preregistration_formality"] == "heavy"
        assert result["max_redesign_iterations"] == 3
        assert result["enable_validity_audit"] is False


class TestContextLoaderPerformance:
    """Test context loader performance characteristics."""

    @pytest.mark.asyncio
    async def test_latency_under_target(self):
        """Test context loading completes under threshold.

        Production target: 100ms for optimal UX
        Test threshold: 500ms to accommodate CI/WSL environments

        Note: use_validation_learnings=False avoids Supabase calls for fast unit tests.
        """
        # Disable validation learnings to avoid Supabase calls in unit tests
        node = ContextLoaderNode(use_validation_learnings=False)
        state = create_initial_state(business_question="Test latency performance")

        result = await node.execute(state)

        latency = result["node_latencies_ms"]["context_loader"]
        # Relaxed threshold for CI environments; production target is 100ms
        assert latency < 500, f"Context loading took {latency}ms, exceeds 500ms threshold"

    @pytest.mark.asyncio
    async def test_latency_with_store(self):
        """Test context loading latency with knowledge store."""
        # Uses MockKnowledgeStore only, no Supabase calls
        node = ContextLoaderNode(use_validation_learnings=False)
        state = create_initial_state(business_question="Test latency with store")

        result = await node.execute(state)

        latency = result["node_latencies_ms"]["context_loader"]
        # Should still be fast even with store (target 500ms)
        assert latency < 500
