"""
Comprehensive unit tests for src/rag/memory_adapters.py

Tests cover:
- EpisodicMemoryAdapter
- SemanticMemoryAdapter
- ProceduralMemoryAdapter
- SignalCollectorAdapter
- Memory protocols
- Factory function
"""

import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Mock dependencies before importing
sys.modules["src.memory.procedural_memory"] = MagicMock()

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
# Test Protocols
# =============================================================================


class TestMemoryProtocols:
    def test_episodic_protocol(self):
        # Test that protocol is defined correctly
        assert hasattr(EpisodicMemoryProtocol, "vector_search")

    def test_semantic_protocol(self):
        assert hasattr(SemanticMemoryProtocol, "graph_query")

    def test_procedural_protocol(self):
        assert hasattr(ProceduralMemoryProtocol, "procedure_search")

    def test_signal_collector_protocol(self):
        assert hasattr(SignalCollectorProtocol, "collect")


# =============================================================================
# Test EpisodicMemoryAdapter
# =============================================================================


class TestEpisodicMemoryAdapter:
    @pytest.fixture
    def mock_connector(self):
        connector = Mock()
        connector.vector_search_by_text = AsyncMock(return_value=[])
        connector.vector_search = AsyncMock(return_value=[])
        return connector

    @pytest.fixture
    def adapter(self, mock_connector):
        return EpisodicMemoryAdapter(memory_connector=mock_connector)

    def test_init_no_connector(self):
        adapter = EpisodicMemoryAdapter()
        assert adapter._connector is None
        assert adapter._embedding_model is None

    def test_init_with_connector(self, mock_connector):
        adapter = EpisodicMemoryAdapter(memory_connector=mock_connector)
        assert adapter._connector is mock_connector

    @pytest.mark.asyncio
    async def test_vector_search_no_connector(self):
        adapter = EpisodicMemoryAdapter()
        results = await adapter.vector_search("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_vector_search_by_text(self, adapter, mock_connector):
        # Mock result with object
        mock_result = Mock()
        mock_result.content = "Test content"
        mock_result.source = "episodic"
        mock_result.source_id = "id_001"
        mock_result.score = 0.9
        mock_result.metadata = {"key": "value"}

        mock_connector.vector_search_by_text = AsyncMock(return_value=[mock_result])

        results = await adapter.vector_search("test query", limit=5)

        assert len(results) == 1
        assert results[0]["content"] == "Test content"
        assert results[0]["source"] == "episodic"
        assert results[0]["score"] == 0.9
        mock_connector.vector_search_by_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_search_with_embedding_model(self, mock_connector):
        # Mock embedding model
        mock_embeddings = AsyncMock()
        mock_embeddings.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])

        adapter = EpisodicMemoryAdapter(
            memory_connector=mock_connector, embedding_model=mock_embeddings
        )

        mock_result = {"content": "test", "score": 0.8}
        mock_connector.vector_search = AsyncMock(return_value=[mock_result])

        await adapter.vector_search("test", limit=10)

        mock_embeddings.embed.assert_called_once_with("test")
        mock_connector.vector_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_search_dict_result(self, adapter, mock_connector):
        # Mock result as dict
        mock_result = {
            "content": "Test content",
            "source": "test_source",
            "score": 0.85,
        }

        mock_connector.vector_search_by_text = AsyncMock(return_value=[mock_result])

        results = await adapter.vector_search("query")

        assert len(results) == 1
        assert results[0]["content"] == "Test content"
        assert results[0]["source"] == "test_source"

    @pytest.mark.asyncio
    async def test_vector_search_exception(self, adapter, mock_connector):
        mock_connector.vector_search_by_text = AsyncMock(side_effect=Exception("DB error"))

        results = await adapter.vector_search("query")

        assert results == []

    @pytest.mark.asyncio
    async def test_generate_embedding_sync_model(self, mock_connector):
        # Mock synchronous embedding model with only encode method
        mock_model = Mock(spec=["encode"])
        mock_model.encode = Mock(return_value=[0.1, 0.2])

        adapter = EpisodicMemoryAdapter(memory_connector=mock_connector, embedding_model=mock_model)

        embedding = await adapter._generate_embedding("test text")

        assert embedding == [0.1, 0.2]
        # Note: encode is called in executor, so call count may vary

    @pytest.mark.asyncio
    async def test_generate_embedding_no_method(self, mock_connector):
        mock_model = Mock(spec=[])  # No embed or encode method

        adapter = EpisodicMemoryAdapter(memory_connector=mock_connector, embedding_model=mock_model)

        with pytest.raises(ValueError, match="embed.*encode"):
            await adapter._generate_embedding("test")

    def test_transform_results_object(self, adapter):
        # Mock result object
        mock_result = Mock()
        mock_result.content = "content"
        mock_result.source = "source"
        mock_result.source_id = "id"
        mock_result.score = 0.8
        mock_result.metadata = {}

        transformed = adapter._transform_results([mock_result])

        assert len(transformed) == 1
        assert transformed[0]["content"] == "content"

    def test_transform_results_dict(self, adapter):
        result_dict = {"content": "test", "score": 0.7}

        transformed = adapter._transform_results([result_dict])

        assert len(transformed) == 1
        assert transformed[0]["content"] == "test"
        assert transformed[0]["score"] == 0.7

    def test_transform_results_string(self, adapter):
        transformed = adapter._transform_results(["plain string"])

        assert len(transformed) == 1
        assert transformed[0]["content"] == "plain string"
        assert transformed[0]["score"] == 0.0


# =============================================================================
# Test SemanticMemoryAdapter
# =============================================================================


class TestSemanticMemoryAdapter:
    @pytest.fixture
    def mock_falkordb(self):
        db = Mock()
        db.find_related = AsyncMock(return_value=[])
        db.semantic_search = AsyncMock(return_value=[])
        return db

    @pytest.fixture
    def mock_connector(self):
        connector = Mock()
        connector.graph_traverse = AsyncMock(return_value=[])
        return connector

    @pytest.fixture
    def adapter(self, mock_falkordb, mock_connector):
        return SemanticMemoryAdapter(falkordb_memory=mock_falkordb, memory_connector=mock_connector)

    def test_init(self, mock_falkordb, mock_connector):
        adapter = SemanticMemoryAdapter(
            falkordb_memory=mock_falkordb, memory_connector=mock_connector
        )

        assert adapter._falkordb is mock_falkordb
        assert adapter._connector is mock_connector

    @pytest.mark.asyncio
    async def test_graph_query_no_backends(self):
        adapter = SemanticMemoryAdapter()
        results = await adapter.graph_query("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_graph_query_falkordb_success(self, adapter, mock_falkordb):
        mock_result = {"type": "Drug", "id": "kisqali", "relationship": "CAUSES"}
        mock_falkordb.find_related = AsyncMock(return_value=[mock_result])
        mock_falkordb.semantic_search = AsyncMock(return_value=[])

        results = await adapter.graph_query("kisqali causes", max_depth=2)

        assert len(results) > 0
        mock_falkordb.find_related.assert_called()

    @pytest.mark.asyncio
    async def test_graph_query_falkordb_failure_connector_fallback(
        self, adapter, mock_falkordb, mock_connector
    ):
        mock_falkordb.find_related = AsyncMock(side_effect=Exception("DB error"))

        mock_result = {
            "type": "Drug",
            "id": "kisqali",
            "relationship": "IMPACTS",
            "target": {"type": "KPI", "id": "TRx"},
        }
        mock_connector.graph_traverse = AsyncMock(return_value=[mock_result])

        results = await adapter.graph_query("kisqali", max_depth=1)

        assert len(results) > 0
        mock_connector.graph_traverse.assert_called()

    @pytest.mark.asyncio
    async def test_query_falkordb(self, adapter, mock_falkordb):
        mock_result = {"type": "Drug", "name": "Kisqali"}
        mock_falkordb.find_related = AsyncMock(return_value=[mock_result])

        results = await adapter._query_falkordb("kisqali", max_depth=2)

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_query_connector_graph(self, adapter, mock_connector):
        mock_result = {"id": "entity_1"}
        mock_connector.graph_traverse = AsyncMock(return_value=[mock_result])

        await adapter._query_connector_graph("test", max_depth=2)

        # Should return results for extracted entities
        mock_connector.graph_traverse.assert_called()

    def test_extract_entities_brands(self, adapter):
        entities = adapter._extract_entities("What about Kisqali and Fabhalta?")

        assert len(entities) >= 2
        entity_ids = [e["id"] for e in entities]
        assert "kisqali" in entity_ids
        assert "fabhalta" in entity_ids

    def test_extract_entities_kpis(self, adapter):
        entities = adapter._extract_entities("TRx and NRx trends")

        entity_ids = [e["id"] for e in entities]
        assert "trx" in entity_ids
        assert "nrx" in entity_ids

    def test_extract_entities_default(self, adapter):
        entities = adapter._extract_entities("some unknown query")

        assert len(entities) > 0  # Should have default

    def test_transform_graph_results_dict(self, adapter):
        result = {"type": "Drug", "id": "kisqali", "relationship": "CAUSES"}

        transformed = adapter._transform_graph_results([result])

        assert len(transformed) == 1
        assert "content" in transformed[0]
        assert transformed[0]["source"] == "semantic_graph"

    def test_transform_graph_results_object(self, adapter):
        mock_result = Mock()
        mock_result.to_dict = Mock(return_value={"type": "Drug", "name": "Kisqali"})

        transformed = adapter._transform_graph_results([mock_result])

        assert len(transformed) == 1

    def test_build_content_from_graph_node_full(self, adapter):
        node = {
            "type": "Drug",
            "id": "kisqali",
            "name": "Kisqali",
            "relationship": "CAUSES",
            "target": {"type": "KPI", "id": "TRx", "name": "TRx"},
            "confidence": 0.9,
        }

        content = adapter._build_content_from_graph_node(node)

        assert "Drug" in content
        assert "Kisqali" in content
        assert "CAUSES" in content
        assert "KPI" in content

    def test_build_content_from_graph_node_minimal(self, adapter):
        node = {"unknown": "data"}

        content = adapter._build_content_from_graph_node(node)

        assert isinstance(content, str)


# =============================================================================
# Test ProceduralMemoryAdapter
# =============================================================================


class TestProceduralMemoryAdapter:
    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.rpc = Mock()
        client.table = Mock()
        return client

    @pytest.fixture
    def adapter(self, mock_client):
        return ProceduralMemoryAdapter(supabase_client=mock_client)

    def test_init(self, mock_client):
        adapter = ProceduralMemoryAdapter(supabase_client=mock_client)
        assert adapter._client is mock_client

    @pytest.mark.asyncio
    @patch("src.rag.memory_adapters.find_relevant_procedures_by_text")
    async def test_procedure_search_semantic_success(self, mock_find, adapter):
        mock_procedure = {
            "id": "proc_001",
            "procedure_name": "Adoption Analysis",
            "tool_sequence": [{"tool": "query_episodic"}, {"tool": "analyze"}],
            "success_rate": 0.85,
        }

        # Return awaitable
        async def async_return():
            return [mock_procedure]

        mock_find.return_value = async_return()

        results = await adapter.procedure_search("adoption", limit=5)

        assert len(results) >= 1
        assert "content" in results[0]
        # Content might be the procedure name or fallback
        assert "source" in results[0]

    @pytest.mark.asyncio
    @patch("src.rag.memory_adapters.find_relevant_procedures_by_text")
    async def test_procedure_search_fallback_rpc(self, mock_find, adapter, mock_client):
        mock_find.side_effect = Exception("Semantic search failed")

        # Mock RPC response
        mock_response = Mock()
        mock_response.data = [{"procedure_name": "Test Procedure", "tool_sequence": []}]
        mock_client.rpc.return_value.execute = Mock(return_value=mock_response)

        results = await adapter.procedure_search("test", limit=3)

        assert len(results) > 0

    @pytest.mark.asyncio
    @patch("src.rag.memory_adapters.find_relevant_procedures_by_text")
    async def test_procedure_search_fallback_table(self, mock_find, adapter, mock_client):
        mock_find.side_effect = Exception("Search failed")

        # RPC fails, table query succeeds
        mock_client.rpc.side_effect = Exception("RPC failed")

        mock_response = Mock()
        mock_response.data = [{"procedure_name": "Table Procedure"}]
        mock_client.table.return_value.select.return_value.limit.return_value.execute = Mock(
            return_value=mock_response
        )

        results = await adapter.procedure_search("test", limit=3)

        # Should use hardcoded fallback procedures
        assert isinstance(results, list)

    @pytest.mark.asyncio
    @patch("src.rag.memory_adapters.find_relevant_procedures_by_text")
    async def test_procedure_search_no_results(self, mock_find, adapter):
        mock_find.return_value = []

        adapter._client = None  # No client for fallback

        results = await adapter.procedure_search("test", limit=5)

        # Should return hardcoded fallback
        assert len(results) > 0
        assert results[0]["source"] == "procedural_fallback"

    @pytest.mark.asyncio
    async def test_semantic_procedure_search_with_embedding_model(self, mock_client):
        mock_model = Mock(spec=["encode"])
        mock_model.encode = Mock(return_value=[0.1, 0.2])

        adapter = ProceduralMemoryAdapter(supabase_client=mock_client, embedding_model=mock_model)

        with patch("src.rag.memory_adapters.find_relevant_procedures") as mock_find_embed:
            # Return awaitable
            async def async_return():
                return []

            mock_find_embed.return_value = async_return()

            await adapter._semantic_procedure_search("test", limit=5)

            # encode called in executor
            mock_find_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embedding_encode_async(self, mock_client):
        mock_model = Mock(spec=["encode_async"])
        mock_model.encode_async = AsyncMock(return_value=[0.1, 0.2, 0.3])

        adapter = ProceduralMemoryAdapter(supabase_client=mock_client, embedding_model=mock_model)

        result = await adapter._generate_embedding("test")

        assert result == [0.1, 0.2, 0.3]
        mock_model.encode_async.assert_called_once_with("test")

    def test_transform_procedure_results(self, adapter):
        procedures = [
            {
                "procedure_name": "Test Proc",
                "steps": ["step1", "step2"],
                "success_rate": 0.9,
            }
        ]

        transformed = adapter._transform_procedure_results(procedures)

        assert len(transformed) == 1
        assert "content" in transformed[0]
        assert transformed[0]["source"] == "procedural"

    def test_build_procedure_content_full(self, adapter):
        proc = {
            "name": "Adoption Analysis",
            "steps": ["query", "analyze", "synthesize"],
            "context": "Kisqali adoption",
            "pattern": "adoption_pattern",
            "success_rate": 0.88,
        }

        content = adapter._build_procedure_content(proc)

        assert "Adoption Analysis" in content
        assert "query" in content
        assert "0.88" in content or "88%" in content

    def test_build_procedure_content_minimal(self, adapter):
        proc = {"procedure_type": "unknown"}

        content = adapter._build_procedure_content(proc)

        assert "unknown" in content

    def test_get_fallback_procedures_relevant(self, adapter):
        results = adapter._get_fallback_procedures("TRx adoption trends")

        assert len(results) > 0
        assert any("TRx" in r["content"] or "adoption" in r["content"] for r in results)

    def test_get_fallback_procedures_generic(self, adapter):
        results = adapter._get_fallback_procedures("random query")

        assert len(results) > 0  # Should return default procedures


# =============================================================================
# Test SignalCollectorAdapter
# =============================================================================


class TestSignalCollectorAdapter:
    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.table = Mock()
        return client

    @pytest.fixture
    def adapter(self, mock_client):
        return SignalCollectorAdapter(supabase_client=mock_client, buffer_size=3)

    def test_init(self, mock_client):
        adapter = SignalCollectorAdapter(supabase_client=mock_client, buffer_size=10)
        assert adapter._client is mock_client
        assert adapter._buffer_size == 10
        assert len(adapter._signal_buffer) == 0

    @pytest.mark.asyncio
    async def test_collect_signals(self, adapter):
        signals = [
            {"type": "response", "query": "test", "response": "answer", "reward": 0.8},
            {"type": "tool", "query": "test2", "response": "ans2", "reward": 0.9},
        ]

        await adapter.collect(signals)

        assert len(adapter._signal_buffer) == 2
        assert adapter._signal_buffer[0].signal_type == "response"
        assert adapter._signal_buffer[1].reward == 0.9

    @pytest.mark.asyncio
    async def test_collect_auto_flush(self, adapter, mock_client):
        # Buffer size is 3, so 3rd signal should trigger flush
        signals = [
            {"type": f"sig_{i}", "query": "q", "response": "r", "reward": 0.5} for i in range(3)
        ]

        mock_response = Mock()
        mock_client.table.return_value.insert.return_value.execute = Mock(
            return_value=mock_response
        )

        await adapter.collect(signals)

        # Buffer should be flushed
        mock_client.table.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_empty(self, adapter):
        count = await adapter.flush()
        assert count == 0

    @pytest.mark.asyncio
    async def test_flush_success(self, adapter, mock_client):
        # Add signals to buffer
        adapter._signal_buffer = [
            CollectedSignal(
                signal_type="test",
                query="q",
                response="r",
                reward=0.8,
            )
        ]

        mock_response = Mock()
        mock_client.table.return_value.insert.return_value.execute = Mock(
            return_value=mock_response
        )

        count = await adapter.flush()

        assert count == 1
        assert len(adapter._signal_buffer) == 0
        mock_client.table.assert_called_once_with("dspy_agent_training_signals")

    @pytest.mark.asyncio
    async def test_flush_no_client(self, adapter):
        adapter._client = None
        adapter._signal_buffer = [CollectedSignal(signal_type="test", query="q", response="r")]

        count = await adapter.flush()

        assert count == 0

    @pytest.mark.asyncio
    async def test_flush_failure(self, adapter, mock_client):
        adapter._signal_buffer = [CollectedSignal(signal_type="test", query="q", response="r")]

        mock_client.table.return_value.insert.return_value.execute = Mock(
            side_effect=Exception("DB error")
        )

        count = await adapter.flush()

        assert count == 0
        # Signal should be re-added to buffer
        assert len(adapter._signal_buffer) == 1

    @pytest.mark.asyncio
    async def test_get_signals_for_optimization(self, adapter, mock_client):
        mock_response = Mock()
        mock_response.data = [{"signal_type": "response", "query": "test", "reward": 0.9}]

        mock_query = Mock()
        mock_query.execute = Mock(return_value=mock_response)
        mock_query.gte = Mock(return_value=mock_query)
        mock_query.limit = Mock(return_value=mock_query)
        mock_query.eq = Mock(return_value=mock_query)

        mock_client.table.return_value.select = Mock(return_value=mock_query)

        signals = await adapter.get_signals_for_optimization(
            signal_type="response", min_reward=0.5, limit=100
        )

        assert len(signals) == 1
        assert signals[0]["reward"] == 0.9

    @pytest.mark.asyncio
    async def test_get_signals_no_client(self, adapter):
        adapter._client = None

        signals = await adapter.get_signals_for_optimization()

        assert signals == []

    @pytest.mark.asyncio
    async def test_get_signals_exception(self, adapter, mock_client):
        mock_client.table.return_value.select.side_effect = Exception("Query failed")

        signals = await adapter.get_signals_for_optimization()

        assert signals == []


# =============================================================================
# Test CollectedSignal
# =============================================================================


class TestCollectedSignal:
    def test_create_signal(self):
        signal = CollectedSignal(
            signal_type="response",
            query="test query",
            response="test response",
            feedback={"score": 0.9},
            reward=0.85,
        )

        assert signal.signal_type == "response"
        assert signal.query == "test query"
        assert signal.reward == 0.85
        assert signal.feedback["score"] == 0.9

    def test_signal_defaults(self):
        signal = CollectedSignal(signal_type="test", query="q", response="r")

        assert signal.feedback is None
        assert signal.reward == 0.0
        assert isinstance(signal.timestamp, datetime)
        assert signal.metadata == {}


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateMemoryAdapters:
    def test_create_adapters_all_args(self):
        mock_client = Mock()
        mock_falkordb = Mock()
        mock_connector = Mock()
        mock_embeddings = Mock()

        adapters = create_memory_adapters(
            supabase_client=mock_client,
            falkordb_memory=mock_falkordb,
            memory_connector=mock_connector,
            embedding_model=mock_embeddings,
        )

        assert "episodic" in adapters
        assert "semantic" in adapters
        assert "procedural" in adapters
        assert "signals" in adapters

        assert isinstance(adapters["episodic"], EpisodicMemoryAdapter)
        assert isinstance(adapters["semantic"], SemanticMemoryAdapter)
        assert isinstance(adapters["procedural"], ProceduralMemoryAdapter)
        assert isinstance(adapters["signals"], SignalCollectorAdapter)

    def test_create_adapters_minimal(self):
        adapters = create_memory_adapters()

        # Should create adapters with None clients
        assert len(adapters) == 4

    def test_episodic_adapter_has_connector(self):
        mock_connector = Mock()
        adapters = create_memory_adapters(memory_connector=mock_connector)

        assert adapters["episodic"]._connector is mock_connector

    def test_semantic_adapter_has_backends(self):
        mock_falkordb = Mock()
        mock_connector = Mock()

        adapters = create_memory_adapters(
            falkordb_memory=mock_falkordb, memory_connector=mock_connector
        )

        assert adapters["semantic"]._falkordb is mock_falkordb
        assert adapters["semantic"]._connector is mock_connector

    def test_procedural_adapter_has_client(self):
        mock_client = Mock()
        adapters = create_memory_adapters(supabase_client=mock_client)

        assert adapters["procedural"]._client is mock_client

    def test_signal_collector_has_client(self):
        mock_client = Mock()
        adapters = create_memory_adapters(supabase_client=mock_client)

        assert adapters["signals"]._client is mock_client
