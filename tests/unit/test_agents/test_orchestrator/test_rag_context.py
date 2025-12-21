"""Unit tests for RAG context retrieval node.

Tests the RAG context node that enriches orchestrator state with
relevant context from the hybrid RAG system.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.orchestrator.nodes.rag_context import (
    RAGContextNode,
    retrieve_rag_context,
)
from src.agents.orchestrator.state import OrchestratorState


class TestRAGContextNode:
    """Test RAGContextNode class."""

    @pytest.mark.asyncio
    async def test_execute_with_entities(self):
        """Test RAG context retrieval with extracted entities."""
        node = RAGContextNode()

        # Mock the retriever
        mock_results = [
            MagicMock(
                id="doc_001",
                content="Kisqali TRx dropped 15% in Q3",
                score=0.92,
                source=MagicMock(value="vector"),
                metadata={"brand": "Kisqali"},
            ),
            MagicMock(
                id="doc_002",
                content="HCP engagement drives conversions",
                score=0.85,
                source=MagicMock(value="graph"),
                metadata={"causal_path": "hcp_engagement -> conversions"},
            ),
        ]

        with patch.object(node, "_retriever") as mock_retriever:
            mock_retriever.search = AsyncMock(return_value=mock_results)

            state: OrchestratorState = {
                "query": "Why did Kisqali TRx drop in Q3?",
                "entities_extracted": {
                    "brands": ["Kisqali"],
                    "kpis": ["trx"],
                    "time_references": ["Q3"],
                },
                "intent": {
                    "primary_intent": "causal_effect",
                    "confidence": 0.9,
                },
            }

            result = await node.execute(state)

            # Verify RAG context was added
            assert "rag_context" in result
            assert "rag_results" in result
            assert "rag_latency_ms" in result
            assert result["rag_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_without_entities(self):
        """Test RAG context retrieval without extracted entities."""
        node = RAGContextNode()

        with patch.object(node, "_retriever") as mock_retriever:
            mock_retriever.search = AsyncMock(return_value=[])

            state: OrchestratorState = {
                "query": "What is happening?",
                "entities_extracted": {},
            }

            result = await node.execute(state)

            # Should still set RAG fields
            assert "rag_context" in result
            assert "rag_latency_ms" in result
            assert result["rag_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_with_retrieval_failure(self):
        """Test graceful handling of retrieval failures."""
        node = RAGContextNode()

        with patch.object(node, "_retriever") as mock_retriever:
            mock_retriever.search = AsyncMock(side_effect=Exception("Connection failed"))

            state: OrchestratorState = {
                "query": "Test query",
                "entities_extracted": {},
            }

            # Should not raise, should handle gracefully
            result = await node.execute(state)

            # Should still set RAG fields with defaults
            assert "rag_context" in result
            assert "rag_latency_ms" in result

    @pytest.mark.asyncio
    async def test_context_format(self):
        """Test that RAG context has expected format."""
        node = RAGContextNode()

        mock_results = [
            MagicMock(
                id="doc_001",
                content="Test content with causal insights",
                score=0.9,
                source=MagicMock(value="vector"),
                metadata={"brand": "Kisqali"},
            ),
        ]

        with patch.object(node, "_retriever") as mock_retriever:
            mock_retriever.search = AsyncMock(return_value=mock_results)

            state: OrchestratorState = {
                "query": "Test query",
                "entities_extracted": {"brands": ["Kisqali"]},
            }

            result = await node.execute(state)

            rag_context = result.get("rag_context", {})

            # Verify context structure
            assert "summary" in rag_context or "sources" in rag_context

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test that RAG latency is properly measured."""
        node = RAGContextNode()

        with patch.object(node, "_retriever") as mock_retriever:
            # Simulate some processing time
            async def slow_search(*args, **kwargs):
                import asyncio

                await asyncio.sleep(0.01)  # 10ms
                return []

            mock_retriever.search = slow_search

            state: OrchestratorState = {
                "query": "Test query",
                "entities_extracted": {},
            }

            result = await node.execute(state)

            # Latency should be at least 10ms
            assert result["rag_latency_ms"] >= 10


class TestRetrieveRAGContextFunction:
    """Test the retrieve_rag_context node function."""

    @pytest.mark.asyncio
    async def test_retrieve_rag_context_creates_node(self):
        """Test that retrieve_rag_context creates and executes node."""
        state = {
            "query": "Test query",
            "entities_extracted": {},
        }

        # Mock the RAGContextNode
        with patch("src.agents.orchestrator.nodes.rag_context.RAGContextNode") as MockNode:
            mock_instance = MagicMock()
            mock_instance.execute = AsyncMock(
                return_value={
                    **state,
                    "rag_context": {"summary": "Test"},
                    "rag_latency_ms": 50,
                }
            )
            MockNode.return_value = mock_instance

            result = await retrieve_rag_context(state)

            # Verify node was created and executed
            MockNode.assert_called_once()
            mock_instance.execute.assert_called_once_with(state)
            assert "rag_context" in result


class TestEntityFiltering:
    """Test entity-based filtering for RAG queries."""

    @pytest.mark.asyncio
    async def test_brand_filter_applied(self):
        """Test that brand entities are used for filtering."""
        node = RAGContextNode()

        with patch.object(node, "_retriever") as mock_retriever:
            mock_retriever.search = AsyncMock(return_value=[])

            state: OrchestratorState = {
                "query": "Kisqali performance",
                "entities_extracted": {
                    "brands": ["Kisqali", "Fabhalta"],
                },
            }

            await node.execute(state)

            # Verify search was called with entities
            mock_retriever.search.assert_called_once()
            call_kwargs = mock_retriever.search.call_args.kwargs
            assert "entities" in call_kwargs or "filters" in call_kwargs

    @pytest.mark.asyncio
    async def test_kpi_filter_applied(self):
        """Test that KPI entities are used for filtering."""
        node = RAGContextNode()

        with patch.object(node, "_retriever") as mock_retriever:
            mock_retriever.search = AsyncMock(return_value=[])

            state: OrchestratorState = {
                "query": "TRx trends",
                "entities_extracted": {
                    "kpis": ["trx", "nrx"],
                },
            }

            await node.execute(state)

            mock_retriever.search.assert_called_once()


class TestRAGContextIntegration:
    """Integration tests for RAG context in orchestrator workflow."""

    @pytest.mark.asyncio
    async def test_rag_context_in_workflow(self):
        """Test that RAG context is available in orchestrator output."""
        from src.agents.orchestrator.agent import OrchestratorAgent

        orchestrator = OrchestratorAgent()

        # Run a query
        input_data = {"query": "What drives conversions?"}

        result = await orchestrator.run(input_data)

        # Verify RAG fields in output
        assert "rag_latency_ms" in result
        assert result["rag_latency_ms"] >= 0

        # RAG context may be None if no retriever configured
        assert "rag_context" in result

    @pytest.mark.asyncio
    async def test_rag_latency_in_total(self):
        """Test that RAG latency is included in total latency."""
        from src.agents.orchestrator.agent import OrchestratorAgent

        orchestrator = OrchestratorAgent()

        input_data = {"query": "Test query"}

        result = await orchestrator.run(input_data)

        # Calculate expected total
        expected_total = (
            result["classification_latency_ms"]
            + result["rag_latency_ms"]
            + result["routing_latency_ms"]
            + result["dispatch_latency_ms"]
            + result["synthesis_latency_ms"]
        )

        assert result["total_latency_ms"] == expected_total
