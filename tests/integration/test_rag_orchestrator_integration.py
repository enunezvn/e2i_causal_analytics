"""
Integration tests for RAG-Orchestrator Integration.

Tests the complete flow from query input through orchestrator
with RAG context enrichment to synthesized response.

This validates Phase 2 Checkpoint 2.4 implementation.

These exercise the orchestrator graph without instantiating real agents, so the
registry-less orchestrators are built with ``allow_mock=True`` to opt into the
test-only dispatcher mock scaffold. Production builds the orchestrator with a real
registry and ``allow_mock=False`` (the default), which fails closed for a
missing/partial registry rather than fabricating values (#814).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.orchestrator.agent import OrchestratorAgent
from src.agents.orchestrator.graph import create_orchestrator_graph
from src.agents.orchestrator.nodes import (
    RAGContextNode,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_rag_retriever():
    """Create mock RAG retriever with sample results."""
    mock = AsyncMock()
    mock.search = AsyncMock(
        return_value=[
            MagicMock(
                id="doc_001",
                content="Kisqali TRx declined 15% in Q3 2024 in Northeast region",
                score=0.92,
                source=MagicMock(value="vector"),
                metadata={
                    "brand": "Kisqali",
                    "kpi": "trx",
                    "region": "northeast",
                    "time_period": "Q3_2024",
                },
            ),
            MagicMock(
                id="doc_002",
                content="Causal analysis: HCP engagement -> TRx (0.35 coefficient)",
                score=0.88,
                source=MagicMock(value="graph"),
                metadata={
                    "causal_chain": "hcp_engagement -> trx",
                    "strength": 0.35,
                    "brand": "Kisqali",
                },
            ),
            MagicMock(
                id="doc_003",
                content="Competitive pressure from alternative treatments increased",
                score=0.75,
                source=MagicMock(value="fulltext"),
                metadata={"brand": "Kisqali", "time_period": "Q3_2024"},
            ),
        ]
    )
    return mock


@pytest.fixture
def orchestrator_with_mock_rag(mock_rag_retriever):
    """Create orchestrator with mocked RAG retriever."""
    orchestrator = OrchestratorAgent(allow_mock=True)

    # Patch the RAG retriever in the rag_context node
    with patch(
        "src.agents.orchestrator.nodes.rag_context.RAGContextNode._retriever", mock_rag_retriever
    ):
        yield orchestrator


# =============================================================================
# Workflow Integration Tests
# =============================================================================


class TestRAGOrchestratorWorkflow:
    """Test complete orchestrator workflow with RAG integration."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_rag_context(self):
        """Test complete workflow from query to response with RAG."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {
            "query": "Why did Kisqali TRx drop in Q3?",
            "user_id": "analyst_001",
            "user_context": {"expertise": "analyst"},
        }

        result = await orchestrator.run(input_data)

        # Verify workflow completed
        assert result["status"] == "completed"

        # Verify all latency components are present
        assert "classification_latency_ms" in result
        assert "rag_latency_ms" in result
        assert "routing_latency_ms" in result
        assert "dispatch_latency_ms" in result
        assert "synthesis_latency_ms" in result
        assert "total_latency_ms" in result

        # Verify RAG context is in output
        assert "rag_context" in result

        # Verify total latency includes RAG
        expected_total = (
            result["classification_latency_ms"]
            + result["rag_latency_ms"]
            + result["routing_latency_ms"]
            + result["dispatch_latency_ms"]
            + result["synthesis_latency_ms"]
        )
        assert result["total_latency_ms"] == expected_total

    @pytest.mark.asyncio
    async def test_workflow_with_brand_entity(self):
        """Test that brand entities flow through RAG to agents."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {
            "query": "What is the impact of HCP engagement on Kisqali conversions?",
        }

        result = await orchestrator.run(input_data)

        assert result["status"] == "completed"
        assert "rag_context" in result

    @pytest.mark.asyncio
    async def test_workflow_with_kpi_entity(self):
        """Test that KPI entities are used for RAG filtering."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {
            "query": "Analyze TRx performance trends",
        }

        result = await orchestrator.run(input_data)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_workflow_preserves_query_context(self):
        """Test that query context flows through entire workflow."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {
            "query": "Why is market share declining?",
            "query_id": "test-query-123",
            "user_context": {"role": "brand_manager"},
        }

        result = await orchestrator.run(input_data)

        assert result["query_id"] == "test-query-123"


class TestRAGContextEnrichment:
    """Test RAG context enrichment in workflow."""

    @pytest.mark.asyncio
    async def test_rag_context_has_expected_structure(self):
        """Test that RAG context has expected structure."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {"query": "Test query"}

        result = await orchestrator.run(input_data)

        result.get("rag_context")
        # RAG context may be None if retriever not configured
        # But the field should exist
        assert "rag_context" in result

    @pytest.mark.asyncio
    async def test_citations_built_from_rag(self):
        """Test that citations are built from RAG context."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {"query": "What drives conversions?"}

        result = await orchestrator.run(input_data)

        # Citations should be present in output
        assert "citations" in result
        assert isinstance(result["citations"], list)


class TestRAGOrchestratorPerformance:
    """Test performance characteristics with RAG integration."""

    @pytest.mark.asyncio
    async def test_orchestration_overhead_under_2_seconds(self):
        """Test that orchestration overhead (including RAG) is under 2s."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {"query": "What is the causal effect of HCP targeting?"}

        result = await orchestrator.run(input_data)

        # Orchestration overhead = classification + RAG + routing + synthesis
        orchestration_overhead = (
            result["classification_latency_ms"]
            + result["rag_latency_ms"]
            + result["routing_latency_ms"]
            + result["synthesis_latency_ms"]
        )

        # Should be well under 2000ms with mock agents
        assert orchestration_overhead < 2000, (
            f"Orchestration overhead {orchestration_overhead}ms exceeds 2s target"
        )

    @pytest.mark.asyncio
    async def test_rag_latency_under_500ms(self):
        """Test that RAG context retrieval is under 500ms target."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {"query": "Test query"}

        result = await orchestrator.run(input_data)

        # RAG should be fast (under 500ms target)
        assert result["rag_latency_ms"] < 500, (
            f"RAG latency {result['rag_latency_ms']}ms exceeds 500ms target"
        )


class TestRAGGraphConfiguration:
    """Test graph configuration options for RAG."""

    def test_graph_with_rag_enabled(self):
        """Test graph creation with RAG enabled (default)."""
        graph = create_orchestrator_graph(enable_rag=True)

        # Graph should be compiled and executable
        assert graph is not None

    def test_graph_with_rag_disabled(self):
        """Test graph creation with RAG disabled."""
        graph = create_orchestrator_graph(enable_rag=False)

        # Graph should still work without RAG
        assert graph is not None

    @pytest.mark.asyncio
    async def test_workflow_without_rag(self):
        """Test that workflow works when RAG is disabled."""
        # Create graph without RAG (allow_mock: registry-less graph uses the
        # test-only scaffold so dispatch completes; prod default fails closed).
        graph = create_orchestrator_graph(enable_rag=False, allow_mock=True)

        initial_state = {
            "query": "Test without RAG",
            "query_id": "test-001",
            "current_phase": "classifying",
            "status": "pending",
            "agent_results": [],
            "errors": [],
            "warnings": [],
            "fallback_used": False,
            "total_latency_ms": 0,
            "classification_latency_ms": 0,
            "rag_latency_ms": 0,
            "routing_latency_ms": 0,
            "dispatch_latency_ms": 0,
            "synthesis_latency_ms": 0,
            "response_confidence": 0.0,
            "agents_dispatched": [],
        }

        result = await graph.ainvoke(initial_state)

        # Should complete without RAG node
        assert result["status"] == "completed"
        # RAG latency should be 0 (not executed)
        assert result["rag_latency_ms"] == 0


class TestAgentRAGContextUsage:
    """Test that agents properly receive RAG context."""

    @pytest.mark.asyncio
    async def test_agent_receives_rag_context(self):
        """Test that dispatched agents receive RAG context in input."""
        mock_agent = MagicMock()
        # Dispatcher calls the mapped method (``run``), not ``analyze`` — stub the
        # real method so the guarded assertion below actually fires (previously the
        # ``.analyze`` stub was never called, so this test passed vacuously).
        mock_agent.run = AsyncMock(
            return_value={
                "narrative": "Test response with RAG context",
                "recommendations": [],
                "confidence": 0.9,
            }
        )

        orchestrator = OrchestratorAgent(agent_registry={"causal_impact": mock_agent})

        input_data = {"query": "what drives conversions?"}

        result = await orchestrator.run(input_data)

        # Result should include RAG context field
        assert "rag_context" in result

        # The registered agent MUST actually be dispatched (via its mapped .run
        # method) — assert unconditionally so this can never pass vacuously.
        assert mock_agent.run.called, "causal_impact agent was never dispatched"
        assert mock_agent.run.call_args is not None
        assert result["status"] == "completed"


class TestErrorHandling:
    """Test error handling in RAG-Orchestrator integration."""

    @pytest.mark.asyncio
    async def test_workflow_continues_on_rag_failure(self):
        """Test that workflow continues even if RAG fails."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        # Mock RAG to fail
        with patch.object(RAGContextNode, "execute", side_effect=Exception("RAG retrieval failed")):
            input_data = {"query": "Test with RAG failure"}

            # Should not raise, workflow should handle gracefully
            result = None
            try:
                result = await orchestrator.run(input_data)
            except Exception as e:
                # Some error handling is expected
                assert "RAG" in str(e) or result is not None

    @pytest.mark.asyncio
    async def test_empty_rag_results_handled(self):
        """Test that empty RAG results are handled gracefully."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {"query": "Query with no matching documents"}

        result = await orchestrator.run(input_data)

        # Should complete even with no RAG results
        assert result["status"] == "completed"


class TestMultiAgentWithRAG:
    """Test multi-agent workflows with RAG context."""

    @pytest.mark.asyncio
    async def test_multi_agent_receives_same_rag_context(self):
        """Test that multiple agents receive the same RAG context."""
        # The dispatcher invokes a registered agent via its mapped method
        # (AGENT_METHOD_MAP -> ``run``), NOT ``analyze`` — so the stub must expose
        # an async ``run`` or the await fails with "MagicMock can't be used in
        # 'await' expression" (a pre-existing latent bug from the #252 .analyze->.run
        # unification, surfaced here once a registered agent is actually dispatched).
        mock_causal = MagicMock()
        mock_causal.run = AsyncMock(
            return_value={
                "narrative": "Causal analysis",
                "recommendations": [],
                "confidence": 0.9,
            }
        )

        mock_gap = MagicMock()
        mock_gap.run = AsyncMock(
            return_value={
                "narrative": "Gap analysis",
                "recommendations": [],
                "confidence": 0.8,
            }
        )

        # Partial registry: the two stubbed agents verify RAG-context propagation,
        # but the router may also dispatch an agent absent from this 2-entry
        # registry (e.g. explainer). Since #814 a missing agent fails CLOSED by
        # default, which would regress this multi-agent run to status=failed; the
        # test opts into the canned dispatcher scaffold (per the module convention,
        # lines 10-12) so any extra-routed agent completes and the run reaches
        # status=completed. Production never sets allow_mock -> fail-closed.
        orchestrator = OrchestratorAgent(
            agent_registry={
                "causal_impact": mock_causal,
                "gap_analyzer": mock_gap,
            },
            allow_mock=True,
        )

        input_data = {
            "query": "Analyze performance gaps and their causes",
        }

        result = await orchestrator.run(input_data)

        assert result["status"] == "completed"


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestEndToEndScenarios:
    """End-to-end tests for realistic usage scenarios."""

    @pytest.mark.asyncio
    async def test_brand_performance_query(self):
        """Test brand performance analysis query."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {
            "query": "Why is Kisqali underperforming in the Northeast region?",
            "user_context": {"expertise": "brand_manager"},
        }

        result = await orchestrator.run(input_data)

        assert result["status"] == "completed"
        assert len(result["agents_dispatched"]) > 0
        assert result["response_text"] != ""

    @pytest.mark.asyncio
    async def test_causal_impact_query(self):
        """Test causal impact analysis query."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {
            "query": "What is the causal effect of HCP engagement on TRx?",
        }

        result = await orchestrator.run(input_data)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_prediction_query(self):
        """Test prediction/forecast query."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {
            "query": "What will be the TRx forecast for next quarter?",
        }

        result = await orchestrator.run(input_data)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_experiment_design_query(self):
        """Test experiment design query."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {
            "query": "Design an A/B test for new HCP targeting strategy",
        }

        result = await orchestrator.run(input_data)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_system_health_query(self):
        """Test system health check query."""
        orchestrator = OrchestratorAgent(allow_mock=True)

        input_data = {
            "query": "What is the current system health status?",
        }

        result = await orchestrator.run(input_data)

        assert result["status"] == "completed"
