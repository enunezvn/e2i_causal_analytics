"""
Unit tests for Memory Backend Adapters.

Tests the adapters that bridge real memory implementations
(MemoryConnector, FalkorDB, Procedural Memory) to the DSPy workflow interface.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.memory_adapters import (
    CollectedSignal,
    EpisodicMemoryAdapter,
    EpisodicMemoryProtocol,
    ProceduralMemoryAdapter,
    ProceduralMemoryProtocol,
    SemanticMemoryAdapter,
    SemanticMemoryProtocol,
    SignalCollectorAdapter,
    SignalCollectorProtocol,
    create_memory_adapters,
)

# =============================================================================
# Test Fixtures and Mock Classes
# =============================================================================


@dataclass
class MockRetrievalResult:
    """Mock result from MemoryConnector.vector_search."""

    content: str
    source: str = "episodic"
    source_id: Optional[str] = None
    score: float = 0.85
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MockMemoryConnector:
    """Mock MemoryConnector for testing."""

    def __init__(self, results: Optional[List[Any]] = None):
        self._results = results or []
        self.calls = []

    async def vector_search(self, query_embedding: List[float], limit: int = 10) -> List[Any]:
        self.calls.append(("vector_search", query_embedding, limit))
        return self._results

    async def vector_search_by_text(self, query_text: str, limit: int = 10) -> List[Any]:
        self.calls.append(("vector_search_by_text", query_text, limit))
        return self._results

    async def graph_traverse(
        self,
        start_node_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        self.calls.append(("graph_traverse", start_node_id, relationship_types, max_depth))
        return self._results


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, embedding: Optional[List[float]] = None):
        self._embedding = embedding or [0.1] * 384

    async def embed(self, text: str) -> List[float]:
        return self._embedding

    def encode(self, text: str) -> List[float]:
        return self._embedding


class MockFalkorDB:
    """Mock FalkorDB semantic memory for testing."""

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None):
        self._results = results or []
        self.calls = []

    async def find_related(
        self,
        entity_type: str,
        entity_id: str,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        self.calls.append(("find_related", entity_type, entity_id, max_hops))
        return self._results

    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        self.calls.append(("semantic_search", query, limit))
        return self._results


class MockSupabaseClient:
    """Mock Supabase client for testing."""

    def __init__(self, rpc_data: Optional[List] = None, table_data: Optional[List] = None):
        self._rpc_data = rpc_data
        self._table_data = table_data or []
        self.calls = []
        self._inserted = []

    def rpc(self, fn_name: str, params: Dict[str, Any]):
        self.calls.append(("rpc", fn_name, params))
        return MockSupabaseResponse(self._rpc_data)

    def table(self, table_name: str):
        self.calls.append(("table", table_name))
        return MockTableQuery(self._table_data, self._inserted)


class MockTableQuery:
    """Mock Supabase table query builder."""

    def __init__(self, data: List, inserted_list: List):
        self._data = data
        self._inserted_list = inserted_list
        self._filters = {}

    def select(self, columns: str):
        return self

    def limit(self, count: int):
        return self

    def eq(self, column: str, value: Any):
        self._filters[column] = value
        return self

    def gte(self, column: str, value: Any):
        self._filters[f"{column}__gte"] = value
        return self

    def insert(self, records: List[Dict]):
        self._inserted_list.extend(records)
        return self

    def execute(self):
        return MockSupabaseResponse(self._data)


class MockSupabaseResponse:
    """Mock Supabase response."""

    def __init__(self, data: Optional[List] = None):
        self.data = data


# =============================================================================
# EpisodicMemoryAdapter Tests
# =============================================================================


class TestEpisodicMemoryAdapter:
    """Tests for EpisodicMemoryAdapter."""

    @pytest.mark.asyncio
    async def test_vector_search_with_text_returns_results(self):
        """Test vector search using text query."""
        mock_results = [
            MockRetrievalResult(content="Kisqali adoption increased 15%", score=0.92),
            MockRetrievalResult(content="HCP engagement metrics improved", score=0.85),
        ]
        connector = MockMemoryConnector(results=mock_results)
        adapter = EpisodicMemoryAdapter(memory_connector=connector)

        results = await adapter.vector_search("Kisqali adoption trends", limit=10)

        assert len(results) == 2
        assert results[0]["content"] == "Kisqali adoption increased 15%"
        assert results[0]["score"] == 0.92
        assert results[0]["source"] == "episodic"
        assert ("vector_search_by_text", "Kisqali adoption trends", 10) in connector.calls

    @pytest.mark.asyncio
    async def test_vector_search_with_embedding_model(self):
        """Test vector search using embedding model."""
        mock_results = [MockRetrievalResult(content="TRx volume up", score=0.88)]
        connector = MockMemoryConnector(results=mock_results)
        embedding_model = MockEmbeddingModel()
        adapter = EpisodicMemoryAdapter(memory_connector=connector, embedding_model=embedding_model)

        results = await adapter.vector_search("TRx trends", limit=5)

        assert len(results) == 1
        assert results[0]["content"] == "TRx volume up"
        # Should use vector_search with embedding, not text
        assert any(call[0] == "vector_search" for call in connector.calls)

    @pytest.mark.asyncio
    async def test_vector_search_no_connector_returns_empty(self):
        """Test that missing connector returns empty list."""
        adapter = EpisodicMemoryAdapter(memory_connector=None)

        results = await adapter.vector_search("any query")

        assert results == []

    @pytest.mark.asyncio
    async def test_vector_search_handles_dict_results(self):
        """Test handling of dict-formatted results."""
        connector = MockMemoryConnector(
            results=[{"content": "Dict result", "source": "test", "score": 0.9}]
        )
        adapter = EpisodicMemoryAdapter(memory_connector=connector)

        results = await adapter.vector_search("query")

        assert len(results) == 1
        assert results[0]["content"] == "Dict result"
        assert results[0]["source"] == "test"

    @pytest.mark.asyncio
    async def test_vector_search_handles_string_results(self):
        """Test handling of plain string results."""
        connector = MockMemoryConnector(results=["Plain string result"])
        adapter = EpisodicMemoryAdapter(memory_connector=connector)

        results = await adapter.vector_search("query")

        assert len(results) == 1
        assert results[0]["content"] == "Plain string result"
        assert results[0]["source"] == "episodic"

    @pytest.mark.asyncio
    async def test_vector_search_handles_exception(self):
        """Test graceful error handling."""
        connector = MockMemoryConnector()
        connector.vector_search_by_text = AsyncMock(side_effect=Exception("DB error"))
        adapter = EpisodicMemoryAdapter(memory_connector=connector)

        results = await adapter.vector_search("query")

        assert results == []

    def test_implements_protocol(self):
        """Test that adapter implements EpisodicMemoryProtocol."""
        adapter = EpisodicMemoryAdapter()
        assert isinstance(adapter, EpisodicMemoryProtocol)


# =============================================================================
# SemanticMemoryAdapter Tests
# =============================================================================


class TestSemanticMemoryAdapter:
    """Tests for SemanticMemoryAdapter."""

    @pytest.mark.asyncio
    async def test_graph_query_with_falkordb(self):
        """Test graph query using FalkorDB."""
        mock_results = [
            {"type": "Drug", "id": "kisqali", "relationship": "TARGETS", "target": {"type": "HCP"}}
        ]
        falkordb = MockFalkorDB(results=mock_results)
        adapter = SemanticMemoryAdapter(falkordb_memory=falkordb)

        results = await adapter.graph_query("Kisqali HCP targeting", max_depth=2)

        assert len(results) > 0
        assert "semantic_graph" in results[0]["source"]
        assert any("find_related" in str(call) for call in falkordb.calls)

    @pytest.mark.asyncio
    async def test_graph_query_fallback_to_connector(self):
        """Test fallback to MemoryConnector when FalkorDB unavailable."""
        mock_results = [{"type": "Node", "name": "test"}]
        connector = MockMemoryConnector(results=mock_results)
        adapter = SemanticMemoryAdapter(falkordb_memory=None, memory_connector=connector)

        await adapter.graph_query("Kisqali relationships")

        assert any(call[0] == "graph_traverse" for call in connector.calls)

    @pytest.mark.asyncio
    async def test_graph_query_no_backends_returns_empty(self):
        """Test that missing backends return empty list."""
        adapter = SemanticMemoryAdapter(falkordb_memory=None, memory_connector=None)

        results = await adapter.graph_query("any query")

        assert results == []

    @pytest.mark.asyncio
    async def test_entity_extraction_from_query(self):
        """Test entity extraction for known terms."""
        adapter = SemanticMemoryAdapter()

        # Test known entities
        entities = adapter._extract_entities("Kisqali TRx trends for HCP")

        entity_ids = [e["id"] for e in entities]
        assert "kisqali" in entity_ids
        assert "trx" in entity_ids
        assert "hcp_generic" in entity_ids

    @pytest.mark.asyncio
    async def test_entity_extraction_unknown_query(self):
        """Test entity extraction for unknown terms."""
        adapter = SemanticMemoryAdapter()

        entities = adapter._extract_entities("completely unknown query")

        assert len(entities) == 1
        assert entities[0]["type"] == "Query"

    def test_build_content_from_graph_node(self):
        """Test content building from graph node."""
        adapter = SemanticMemoryAdapter()

        node = {
            "type": "Drug",
            "name": "Kisqali",
            "relationship": "TARGETS",
            "target": {"type": "HCP", "name": "Oncologist"},
            "strength": 0.9,
        }

        content = adapter._build_content_from_graph_node(node)

        assert "Drug" in content
        assert "Kisqali" in content
        assert "TARGETS" in content
        assert "HCP" in content

    def test_implements_protocol(self):
        """Test that adapter implements SemanticMemoryProtocol."""
        adapter = SemanticMemoryAdapter()
        assert isinstance(adapter, SemanticMemoryProtocol)


# =============================================================================
# ProceduralMemoryAdapter Tests
# =============================================================================


class TestProceduralMemoryAdapter:
    """Tests for ProceduralMemoryAdapter."""

    @pytest.mark.asyncio
    async def test_procedure_search_via_rpc(self):
        """Test procedure search using RPC function."""
        mock_procedures = [
            {
                "name": "Adoption Analysis",
                "procedure_type": "analysis",
                "steps": ["Query data", "Analyze trends", "Generate report"],
                "success_rate": 0.95,
            }
        ]

        # Mock the adapter's internal search method directly
        adapter = ProceduralMemoryAdapter(supabase_client=MagicMock())
        adapter._execute_procedure_search = AsyncMock(return_value=mock_procedures)

        results = await adapter.procedure_search("adoption analysis workflow", limit=5)

        assert len(results) == 1
        assert "Adoption Analysis" in results[0]["content"]
        assert results[0]["source"] == "procedural"
        assert results[0]["success_rate"] == 0.95

    @pytest.mark.asyncio
    async def test_procedure_search_no_client_returns_fallback(self):
        """Test that missing client gracefully returns fallback procedures.

        With semantic search integration, when the database is unavailable,
        the adapter returns hardcoded fallback procedures for common queries.
        """
        adapter = ProceduralMemoryAdapter(supabase_client=None)

        results = await adapter.procedure_search("adoption trends")

        # Should return fallback procedures when database unavailable
        assert len(results) > 0
        assert all(r.get("source") == "procedural_fallback" for r in results)

    @pytest.mark.asyncio
    async def test_procedure_search_fallback_procedures(self):
        """Test fallback procedures when database unavailable."""
        # Create adapter with a mock client, but mock the internal method to raise exception
        adapter = ProceduralMemoryAdapter(supabase_client=MagicMock())
        adapter._execute_procedure_search = AsyncMock(side_effect=Exception("DB unavailable"))

        results = await adapter.procedure_search("adoption trends", limit=5)

        # Should return fallback procedures since _execute_procedure_search raised an exception
        assert len(results) > 0
        assert any("adoption" in r["content"].lower() for r in results)

    def test_build_procedure_content(self):
        """Test building content from procedure dict."""
        adapter = ProceduralMemoryAdapter()

        proc = {
            "name": "TRx Analysis",
            "procedure_type": "analysis",
            "steps": ["Step 1", "Step 2", "Step 3"],
            "context": "Monthly reporting",
            "success_rate": 0.88,
        }

        content = adapter._build_procedure_content(proc)

        assert "TRx Analysis" in content
        assert "Step 1" in content
        assert "88%" in content

    def test_fallback_procedures_relevant_filtering(self):
        """Test that fallback procedures are filtered by relevance."""
        adapter = ProceduralMemoryAdapter()

        # Adoption-related query
        results = adapter._get_fallback_procedures("adoption trends")
        assert any("adoption" in r["content"].lower() for r in results)

        # Causal query
        results = adapter._get_fallback_procedures("causal impact analysis")
        assert any("causal" in r["content"].lower() for r in results)

    def test_implements_protocol(self):
        """Test that adapter implements ProceduralMemoryProtocol."""
        adapter = ProceduralMemoryAdapter()
        assert isinstance(adapter, ProceduralMemoryProtocol)


# =============================================================================
# SignalCollectorAdapter Tests
# =============================================================================


class TestSignalCollectorAdapter:
    """Tests for SignalCollectorAdapter."""

    @pytest.mark.asyncio
    async def test_collect_signals_adds_to_buffer(self):
        """Test that signals are added to buffer."""
        adapter = SignalCollectorAdapter(buffer_size=100)

        signals = [
            {"type": "response", "query": "test query", "response": "test response", "reward": 0.8},
            {"type": "feedback", "query": "query 2", "response": "response 2", "reward": 0.9},
        ]

        await adapter.collect(signals)

        assert len(adapter._signal_buffer) == 2
        assert adapter._signal_buffer[0].signal_type == "response"
        assert adapter._signal_buffer[0].reward == 0.8

    @pytest.mark.asyncio
    async def test_collect_auto_flush_on_buffer_full(self):
        """Test automatic flush when buffer is full."""
        client = MockSupabaseClient()
        adapter = SignalCollectorAdapter(supabase_client=client, buffer_size=3)

        # Add 4 signals (exceeds buffer of 3)
        for i in range(4):
            await adapter.collect(
                [{"type": "test", "query": f"q{i}", "response": f"r{i}", "reward": 0.5}]
            )

        # Should have flushed once (3 signals) and have 1 remaining
        assert len(adapter._signal_buffer) == 1 or len(client._inserted) >= 3

    @pytest.mark.asyncio
    async def test_flush_persists_to_database(self):
        """Test that flush persists signals to database."""
        client = MockSupabaseClient()
        adapter = SignalCollectorAdapter(supabase_client=client, buffer_size=100)

        await adapter.collect(
            [{"type": "response", "query": "q1", "response": "r1", "reward": 0.8}]
        )

        count = await adapter.flush()

        assert count == 1
        assert len(client._inserted) == 1
        # Schema uses source_agent instead of signal_type
        assert client._inserted[0]["source_agent"] == "response"
        assert client._inserted[0]["reward"] == 0.8
        # Verify new schema fields exist
        assert "input_context" in client._inserted[0]
        assert client._inserted[0]["input_context"]["query"] == "q1"

    @pytest.mark.asyncio
    async def test_flush_empty_buffer_returns_zero(self):
        """Test flush with empty buffer."""
        adapter = SignalCollectorAdapter()

        count = await adapter.flush()

        assert count == 0

    @pytest.mark.asyncio
    async def test_flush_no_client_returns_zero(self):
        """Test flush without client configured."""
        adapter = SignalCollectorAdapter(supabase_client=None)

        await adapter.collect([{"type": "test", "query": "q", "response": "r", "reward": 0.5}])
        count = await adapter.flush()

        assert count == 0

    @pytest.mark.asyncio
    async def test_get_signals_for_optimization(self):
        """Test retrieving signals for optimization."""
        mock_signals = [
            {"signal_type": "response", "query": "q1", "response": "r1", "reward": 0.9},
            {"signal_type": "response", "query": "q2", "response": "r2", "reward": 0.85},
        ]
        client = MockSupabaseClient(table_data=mock_signals)
        adapter = SignalCollectorAdapter(supabase_client=client)

        signals = await adapter.get_signals_for_optimization(
            signal_type="response", min_reward=0.8, limit=100
        )

        assert len(signals) == 2

    @pytest.mark.asyncio
    async def test_get_signals_no_client_returns_empty(self):
        """Test that missing client returns empty signals."""
        adapter = SignalCollectorAdapter(supabase_client=None)

        signals = await adapter.get_signals_for_optimization()

        assert signals == []

    def test_collected_signal_dataclass(self):
        """Test CollectedSignal dataclass."""
        signal = CollectedSignal(
            signal_type="response",
            query="test query",
            response="test response",
            reward=0.9,
        )

        assert signal.signal_type == "response"
        assert signal.query == "test query"
        assert signal.reward == 0.9
        assert isinstance(signal.timestamp, datetime)
        assert signal.metadata == {}

    def test_implements_protocol(self):
        """Test that adapter implements SignalCollectorProtocol."""
        adapter = SignalCollectorAdapter()
        assert isinstance(adapter, SignalCollectorProtocol)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateMemoryAdapters:
    """Tests for create_memory_adapters factory function."""

    def test_create_all_adapters(self):
        """Test creating all adapters."""
        adapters = create_memory_adapters()

        assert "episodic" in adapters
        assert "semantic" in adapters
        assert "procedural" in adapters
        assert "signals" in adapters

        assert isinstance(adapters["episodic"], EpisodicMemoryAdapter)
        assert isinstance(adapters["semantic"], SemanticMemoryAdapter)
        assert isinstance(adapters["procedural"], ProceduralMemoryAdapter)
        assert isinstance(adapters["signals"], SignalCollectorAdapter)

    def test_create_with_dependencies(self):
        """Test creating adapters with dependencies injected."""
        connector = MockMemoryConnector()
        falkordb = MockFalkorDB()
        client = MockSupabaseClient()
        embedding = MockEmbeddingModel()

        adapters = create_memory_adapters(
            supabase_client=client,
            falkordb_memory=falkordb,
            memory_connector=connector,
            embedding_model=embedding,
        )

        # Verify dependencies were passed
        assert adapters["episodic"]._connector is connector
        assert adapters["episodic"]._embedding_model is embedding
        assert adapters["semantic"]._falkordb is falkordb
        assert adapters["semantic"]._connector is connector
        assert adapters["procedural"]._client is client
        assert adapters["signals"]._client is client

    def test_adapters_implement_protocols(self):
        """Test that all adapters implement their protocols."""
        adapters = create_memory_adapters()

        assert isinstance(adapters["episodic"], EpisodicMemoryProtocol)
        assert isinstance(adapters["semantic"], SemanticMemoryProtocol)
        assert isinstance(adapters["procedural"], ProceduralMemoryProtocol)
        assert isinstance(adapters["signals"], SignalCollectorProtocol)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMemoryAdaptersIntegration:
    """Integration tests for memory adapters."""

    @pytest.mark.asyncio
    async def test_episodic_semantic_combined_search(self):
        """Test combined episodic and semantic search."""
        episodic_results = [
            MockRetrievalResult(content="Kisqali Q3 performance: +15% TRx", score=0.9)
        ]
        semantic_results = [
            {"type": "Drug", "name": "Kisqali", "relationship": "DRIVES", "target": "TRx Growth"}
        ]

        connector = MockMemoryConnector(results=episodic_results)
        falkordb = MockFalkorDB(results=semantic_results)

        adapters = create_memory_adapters(
            memory_connector=connector,
            falkordb_memory=falkordb,
        )

        # Search both memories
        episodic = await adapters["episodic"].vector_search("Kisqali TRx trends")
        semantic = await adapters["semantic"].graph_query("Kisqali TRx relationship")

        assert len(episodic) == 1
        assert "15% TRx" in episodic[0]["content"]

        assert len(semantic) > 0
        assert any("Kisqali" in str(r) for r in semantic)

    @pytest.mark.asyncio
    async def test_signal_collection_workflow(self):
        """Test complete signal collection workflow."""
        client = MockSupabaseClient()
        adapter = SignalCollectorAdapter(supabase_client=client, buffer_size=100)

        # Simulate workflow: collect signals from multiple interactions
        for i in range(5):
            await adapter.collect(
                [
                    {
                        "type": "response",
                        "query": f"Query {i}",
                        "response": f"Response {i}",
                        "reward": 0.7 + (i * 0.05),
                        "metadata": {"iteration": i},
                    }
                ]
            )

        # Flush all signals
        count = await adapter.flush()

        assert count == 5
        assert len(client._inserted) == 5
        # Verify rewards are captured correctly
        rewards = [s["reward"] for s in client._inserted]
        assert min(rewards) >= 0.7
        assert max(rewards) <= 0.95

    @pytest.mark.asyncio
    async def test_procedure_guides_investigation(self):
        """Test using procedures to guide investigation."""
        procedures = [
            {
                "name": "Adoption Investigation",
                "procedure_type": "investigation",
                "steps": ["Query episodic for trends", "Check semantic graph", "Synthesize"],
                "success_rate": 0.92,
            }
        ]
        # Mock the internal search method directly
        adapter = ProceduralMemoryAdapter(supabase_client=MagicMock())
        adapter._execute_procedure_search = AsyncMock(return_value=procedures)

        results = await adapter.procedure_search("how to investigate adoption trends")

        assert len(results) == 1
        assert "Query episodic" in results[0]["content"]
        assert results[0]["success_rate"] == 0.92
