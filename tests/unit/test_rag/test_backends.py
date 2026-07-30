"""
Unit tests for E2I Hybrid RAG Backend Clients.

Tests for VectorBackend, FulltextBackend, and GraphBackend.
All external dependencies are mocked.
"""

from unittest.mock import MagicMock

import pytest

from src.rag.backends import FulltextBackend, GraphBackend, VectorBackend
from src.rag.config import FalkorDBConfig, HybridSearchConfig
from src.rag.exceptions import (
    VectorSearchError,
)
from src.rag.types import ExtractedEntities, RetrievalSource

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_falkordb_client():
    """Create a mock FalkorDB client."""
    client = MagicMock()
    mock_graph = MagicMock()
    mock_graph.query = MagicMock(return_value=MagicMock(result_set=[]))
    client.select_graph = MagicMock(return_value=mock_graph)
    return client


@pytest.fixture
def search_config():
    """Create a search configuration for tests."""
    return HybridSearchConfig(
        vector_top_k=10,
        fulltext_top_k=10,
        graph_top_k=10,
        vector_timeout_ms=5000,
        fulltext_timeout_ms=3000,
        graph_timeout_ms=5000,
        vector_min_similarity=0.5,
        fulltext_min_rank=0.1,
        graph_min_relevance=0.2,
    )


@pytest.fixture
def falkordb_config():
    """Create FalkorDB configuration for tests."""
    return FalkorDBConfig(
        host="localhost",
        port=6381,
        graph_name="test_graph",
        max_path_length=5,
    )


# ============================================================================
# VectorBackend Tests
# ============================================================================


class TestVectorBackendInit:
    """Tests for VectorBackend initialization."""

    def test_init_with_defaults(self, mock_supabase_client):
        """Test initialization with default config."""
        backend = VectorBackend(supabase_client=mock_supabase_client)
        assert backend.client == mock_supabase_client
        assert backend.config is not None
        assert backend._last_latency_ms == 0.0

    def test_init_with_custom_config(self, mock_supabase_client, search_config):
        """Test initialization with custom config."""
        backend = VectorBackend(supabase_client=mock_supabase_client, config=search_config)
        assert backend.config == search_config
        assert backend.config.vector_top_k == 10

    def test_repr(self, mock_supabase_client, search_config):
        """Test string representation."""
        backend = VectorBackend(supabase_client=mock_supabase_client, config=search_config)
        repr_str = repr(backend)
        assert "VectorBackend" in repr_str
        assert "top_k=10" in repr_str


class TestVectorBackendSearch:
    """Tests for VectorBackend search operations."""

    @pytest.mark.asyncio
    async def test_search_success(self, mock_supabase_client, search_config):
        """Test successful vector search."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.data = [
            {
                "id": "doc-1",
                "content": "Test content 1",
                "similarity": 0.95,
                "metadata": {"brand": "Remibrutinib"},
            },
            {
                "id": "doc-2",
                "content": "Test content 2",
                "similarity": 0.85,
                "metadata": {"brand": "Kisqali"},
            },
        ]
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        backend = VectorBackend(supabase_client=mock_supabase_client, config=search_config)

        embedding = [0.1] * 1536
        results = await backend.search(embedding=embedding)

        assert len(results) == 2
        assert results[0].id == "doc-1"
        assert results[0].score == 0.95
        assert results[0].source == RetrievalSource.VECTOR

    @pytest.mark.asyncio
    async def test_search_filters_low_similarity(self, mock_supabase_client, search_config):
        """Test that low similarity results are filtered."""
        mock_response = MagicMock()
        mock_response.data = [
            {"id": "doc-1", "content": "High sim", "similarity": 0.9, "metadata": {}},
            {
                "id": "doc-2",
                "content": "Low sim",
                "similarity": 0.3,
                "metadata": {},
            },  # Below threshold
        ]
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        backend = VectorBackend(supabase_client=mock_supabase_client, config=search_config)

        results = await backend.search(embedding=[0.1] * 1536)

        assert len(results) == 1
        assert results[0].id == "doc-1"

    @pytest.mark.asyncio
    async def test_search_timeout(self, mock_supabase_client, search_config):
        """Test search timeout handling."""
        import time as time_module

        def slow_execute():
            """Synchronous blocking call to simulate slow RPC."""
            time_module.sleep(1)  # Block for 1 second
            return MagicMock(data=[])

        mock_supabase_client.rpc.return_value.execute = slow_execute

        # Use very short timeout (50ms)
        search_config.vector_timeout_ms = 50
        backend = VectorBackend(supabase_client=mock_supabase_client, config=search_config)

        with pytest.raises(VectorSearchError) as exc_info:
            await backend.search(embedding=[0.1] * 1536)

        assert "timeout" in str(exc_info.value.message).lower()

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_supabase_client, search_config):
        """Test search with filters passed through."""
        mock_response = MagicMock()
        mock_response.data = []
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        backend = VectorBackend(supabase_client=mock_supabase_client, config=search_config)

        filters = {"brand": "Remibrutinib", "region": "West"}
        await backend.search(embedding=[0.1] * 1536, filters=filters)

        # Verify RPC was called with filters
        # RPC is called as: client.rpc("hybrid_vector_search", {"query_embedding": ..., "filters": ...})
        mock_supabase_client.rpc.assert_called_once()
        call_args = mock_supabase_client.rpc.call_args
        assert call_args[0][0] == "rag_vector_search"
        # Second positional argument is the params dict. #896: the RPC boundary
        # makes the provenance default EXPLICIT (include_synthetic="false")
        # while preserving every caller-supplied filter key.
        rpc_params = call_args[0][1]
        assert rpc_params["filters"] == {**filters, "include_synthetic": "false"}


class TestVectorBackendProvenance:
    """#896: rag_vector_search callers pass include_synthetic explicitly.

    The RPC default-excludes synthetic episodic rows via
    filters->>'include_synthetic' (migration rag/004, mirroring memory/044+045).
    The Python boundary strict-parses the caller's value through the
    coerce_provenance_flag SSOT (#883 §4) so loose values ("false", "0",
    non-bool types) FAIL CLOSED instead of leaking into the jsonb payload.
    """

    async def _rpc_filters(self, mock_supabase_client, search_config, filters):
        mock_response = MagicMock()
        mock_response.data = []
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response
        backend = VectorBackend(supabase_client=mock_supabase_client, config=search_config)
        await backend.search(embedding=[0.1] * 1536, filters=filters)
        return mock_supabase_client.rpc.call_args[0][1]["filters"]

    @pytest.mark.asyncio
    async def test_default_is_explicit_false(self, mock_supabase_client, search_config):
        """No filters at all -> include_synthetic explicitly 'false'."""
        rpc_filters = await self._rpc_filters(mock_supabase_client, search_config, None)
        assert rpc_filters == {"include_synthetic": "false"}

    @pytest.mark.asyncio
    async def test_opt_in_bool_true(self, mock_supabase_client, search_config):
        rpc_filters = await self._rpc_filters(
            mock_supabase_client, search_config, {"include_synthetic": True}
        )
        assert rpc_filters["include_synthetic"] == "true"

    @pytest.mark.asyncio
    async def test_opt_in_string_true(self, mock_supabase_client, search_config):
        rpc_filters = await self._rpc_filters(
            mock_supabase_client, search_config, {"include_synthetic": "true"}
        )
        assert rpc_filters["include_synthetic"] == "true"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("loose", [False, "false", "0", "no", None, 1, [], {}])
    async def test_loose_values_fail_closed(self, mock_supabase_client, search_config, loose):
        rpc_filters = await self._rpc_filters(
            mock_supabase_client, search_config, {"include_synthetic": loose}
        )
        assert rpc_filters["include_synthetic"] == "false"

    @pytest.mark.asyncio
    async def test_caller_filters_dict_not_mutated(self, mock_supabase_client, search_config):
        filters = {"brands": ["Kisqali"], "include_synthetic": True}
        snapshot = {"brands": ["Kisqali"], "include_synthetic": True}
        await self._rpc_filters(mock_supabase_client, search_config, filters)
        assert filters == snapshot

    @pytest.mark.asyncio
    async def test_plural_keys_pass_through(self, mock_supabase_client, search_config):
        """Plural entity-derived keys reach the RPC unchanged (the SQL-side
        rag_filter_values normalizer honors both shapes; migration rag/004)."""
        rpc_filters = await self._rpc_filters(
            mock_supabase_client, search_config, {"brands": ["Kisqali", "Fabhalta"]}
        )
        assert rpc_filters["brands"] == ["Kisqali", "Fabhalta"]


class TestVectorBackendHealthCheck:
    """Tests for VectorBackend health check."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_supabase_client, search_config):
        """Test healthy backend."""
        mock_response = MagicMock()
        mock_response.data = []
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        backend = VectorBackend(supabase_client=mock_supabase_client, config=search_config)

        health = await backend.health_check()

        assert health["status"] == "healthy"
        assert health["error"] is None
        assert "latency_ms" in health

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, mock_supabase_client, search_config):
        """Test unhealthy backend."""
        mock_supabase_client.rpc.return_value.execute.side_effect = Exception(
            "DB connection failed"
        )

        backend = VectorBackend(supabase_client=mock_supabase_client, config=search_config)

        health = await backend.health_check()

        assert health["status"] == "unhealthy"
        assert "DB connection failed" in health["error"]


# ============================================================================
# FulltextBackend Tests
# ============================================================================


class TestFulltextBackendInit:
    """Tests for FulltextBackend initialization."""

    def test_init_with_defaults(self, mock_supabase_client):
        """Test initialization with default config."""
        backend = FulltextBackend(supabase_client=mock_supabase_client)
        assert backend.client == mock_supabase_client
        assert backend.config is not None

    def test_repr(self, mock_supabase_client, search_config):
        """Test string representation."""
        backend = FulltextBackend(supabase_client=mock_supabase_client, config=search_config)
        repr_str = repr(backend)
        assert "FulltextBackend" in repr_str


class TestFulltextBackendSearch:
    """Tests for FulltextBackend search operations."""

    @pytest.mark.asyncio
    async def test_search_success(self, mock_supabase_client, search_config):
        """Test successful fulltext search."""
        mock_response = MagicMock()
        mock_response.data = [
            {
                "id": "doc-1",
                "content": "TRx conversion rate improved",
                "rank": 0.8,
                "metadata": {},
                "source_table": "causal_paths",
            }
        ]
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        backend = FulltextBackend(supabase_client=mock_supabase_client, config=search_config)

        results = await backend.search(query="TRx conversion")

        assert len(results) == 1
        assert results[0].source == RetrievalSource.FULLTEXT
        assert results[0].score == 0.8

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_empty(self, mock_supabase_client, search_config):
        """Test that empty query returns empty results."""
        backend = FulltextBackend(supabase_client=mock_supabase_client, config=search_config)

        results = await backend.search(query="")
        assert results == []

        results = await backend.search(query="   ")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_filters_low_rank(self, mock_supabase_client, search_config):
        """Test that low rank results are filtered."""
        mock_response = MagicMock()
        mock_response.data = [
            {"id": "doc-1", "content": "High rank", "rank": 0.5, "metadata": {}},
            {"id": "doc-2", "content": "Low rank", "rank": 0.05, "metadata": {}},  # Below threshold
        ]
        mock_supabase_client.rpc.return_value.execute.return_value = mock_response

        backend = FulltextBackend(supabase_client=mock_supabase_client, config=search_config)

        results = await backend.search(query="test")

        assert len(results) == 1
        assert results[0].id == "doc-1"


# ============================================================================
# GraphBackend Tests
# ============================================================================


class TestGraphBackendInit:
    """Tests for GraphBackend initialization."""

    def test_init_with_defaults(self, mock_falkordb_client):
        """Test initialization with default config."""
        backend = GraphBackend(falkordb_client=mock_falkordb_client)
        assert backend.client == mock_falkordb_client
        assert backend.falkordb_config is not None
        assert backend.search_config is not None

    def test_init_with_custom_configs(self, mock_falkordb_client, falkordb_config, search_config):
        """Test initialization with custom configs."""
        backend = GraphBackend(
            falkordb_client=mock_falkordb_client,
            falkordb_config=falkordb_config,
            search_config=search_config,
        )
        assert backend.falkordb_config.graph_name == "test_graph"
        assert backend.search_config.graph_top_k == 10

    def test_repr(self, mock_falkordb_client, falkordb_config, search_config):
        """Test string representation."""
        backend = GraphBackend(
            falkordb_client=mock_falkordb_client,
            falkordb_config=falkordb_config,
            search_config=search_config,
        )
        repr_str = repr(backend)
        assert "GraphBackend" in repr_str
        assert "test_graph" in repr_str


class TestGraphBackendSearch:
    """Tests for GraphBackend search operations."""

    @pytest.mark.asyncio
    async def test_search_with_entities(self, mock_falkordb_client, falkordb_config, search_config):
        """Test search with extracted entities."""
        # Setup mock node
        mock_node = MagicMock()
        mock_node.id = 123
        mock_node.labels = ["Brand"]
        mock_node.properties = {"name": "Remibrutinib", "description": "CSU treatment"}

        mock_graph = MagicMock()
        mock_graph.query.return_value = MagicMock(result_set=[[mock_node, None, 0]])
        mock_falkordb_client.select_graph.return_value = mock_graph

        backend = GraphBackend(
            falkordb_client=mock_falkordb_client,
            falkordb_config=falkordb_config,
            search_config=search_config,
        )

        entities = ExtractedEntities(brands=["Remibrutinib"])
        await backend.search(entities=entities)

        assert mock_graph.query.called
        # Check that a Cypher query was built with brand match
        query_arg = mock_graph.query.call_args[0][0]
        assert "Brand" in query_arg or "MATCH" in query_arg

    @pytest.mark.asyncio
    async def test_search_empty_entities_uses_default_query(
        self, mock_falkordb_client, falkordb_config, search_config
    ):
        """Test that empty entities uses default causal path query."""
        mock_graph = MagicMock()
        mock_graph.query.return_value = MagicMock(result_set=[])
        mock_falkordb_client.select_graph.return_value = mock_graph

        backend = GraphBackend(
            falkordb_client=mock_falkordb_client,
            falkordb_config=falkordb_config,
            search_config=search_config,
        )

        entities = ExtractedEntities()  # Empty
        await backend.search(entities=entities)

        query_arg = mock_graph.query.call_args[0][0]
        assert "CAUSES" in query_arg or "AFFECTS" in query_arg


class TestGraphBackendEntityQueryContract:
    """#1376: the entity Cypher must be case-insensitive and must not
    cross-product entity labels.

    The extractor emits normalized lowercase canonicals ('west', 'trx') while
    the knowledge graph stores display-cased node names ('West', 'TRx'); a
    case-sensitive ``IN`` plus a comma cross-product over Brand×Region×KPI
    meant ANY entity-bearing query silently returned 0 graph rows (live g:0).
    """

    def _backend(self, mock_falkordb_client, falkordb_config, search_config):
        return GraphBackend(
            falkordb_client=mock_falkordb_client,
            falkordb_config=falkordb_config,
            search_config=search_config,
        )

    def test_entity_names_compared_case_insensitively(
        self, mock_falkordb_client, falkordb_config, search_config
    ):
        """Node-name comparison must lowercase BOTH sides."""
        backend = self._backend(mock_falkordb_client, falkordb_config, search_config)
        entities = ExtractedEntities(brands=["Kisqali"], regions=["west"], kpis=["trx"])
        cypher = backend._build_entity_query(entities, 10)
        assert "toLower" in cypher
        for term in ("'kisqali'", "'west'", "'trx'"):
            assert term in cypher
        # A raw cased literal would reintroduce the case-sensitive miss.
        assert "'Kisqali'" not in cypher

    def test_no_cross_product_between_entity_labels(
        self, mock_falkordb_client, falkordb_config, search_config
    ):
        """One unmatched entity type must not zero the whole result set.

        The old shape ``MATCH (b:Brand), (r:Region), (k:KPI)`` AND-ed the
        labels together, so a query mentioning a KPI absent from the graph got
        0 rows even when its brand and region both matched.
        """
        import re

        backend = self._backend(mock_falkordb_client, falkordb_config, search_config)
        entities = ExtractedEntities(brands=["Kisqali"], regions=["west"], kpis=["trx"])
        cypher = backend._build_entity_query(entities, 10)
        assert re.search(r"MATCH\s*\([^)]*\)\s*,\s*\(", cypher) is None

    def test_all_entity_labels_reachable_in_single_pattern(
        self, mock_falkordb_client, falkordb_config, search_config
    ):
        backend = self._backend(mock_falkordb_client, falkordb_config, search_config)
        entities = ExtractedEntities(
            brands=["Kisqali"], regions=["west"], kpis=["trx"], agents=["causal_impact"]
        )
        cypher = backend._build_entity_query(entities, 10)
        for label in ("Brand", "Region", "KPI", "Agent"):
            assert label in cypher

    def test_empty_entities_generic_fallback_unchanged(
        self, mock_falkordb_client, falkordb_config, search_config
    ):
        backend = self._backend(mock_falkordb_client, falkordb_config, search_config)
        cypher = backend._build_entity_query(ExtractedEntities(), 10)
        assert "CORRELATES" in cypher
        assert "toLower" not in cypher

    def test_single_quote_in_entity_name_is_escaped(
        self, mock_falkordb_client, falkordb_config, search_config
    ):
        backend = self._backend(mock_falkordb_client, falkordb_config, search_config)
        entities = ExtractedEntities(brands=["O'Brien"])
        cypher = backend._build_entity_query(entities, 10)
        assert "\\'" in cypher
        assert "'o'brien'" not in cypher  # unescaped literal would break the Cypher

    def test_backslash_in_entity_name_is_escaped(
        self, mock_falkordb_client, falkordb_config, search_config
    ):
        backend = self._backend(mock_falkordb_client, falkordb_config, search_config)
        entities = ExtractedEntities(brands=["a\\b"])
        cypher = backend._build_entity_query(entities, 10)
        assert "'a\\\\b'" in cypher

    @pytest.mark.asyncio
    async def test_search_parses_entity_row_shape(
        self, mock_falkordb_client, falkordb_config, search_config
    ):
        """The parser must handle the entity-query row shape
        ``[entity, path, path_nodes, path_rels, path_length]``."""
        mock_node = MagicMock()
        mock_node.id = 7
        mock_node.labels = ["Region"]
        mock_node.properties = {"name": "West"}

        mock_graph = MagicMock()
        mock_graph.query.return_value = MagicMock(result_set=[[mock_node, None, [], [], 0]])
        mock_falkordb_client.select_graph.return_value = mock_graph

        backend = self._backend(mock_falkordb_client, falkordb_config, search_config)
        results = await backend.search(entities=ExtractedEntities(regions=["west"]))
        assert len(results) == 1
        assert results[0].source == RetrievalSource.GRAPH
        assert results[0].score == 1.0

    @pytest.mark.asyncio
    async def test_search_extracts_relationship_types_from_path_rels(
        self, mock_falkordb_client, falkordb_config, search_config
    ):
        """falkordb Edge exposes BOTH .id and .relation (never .type); the
        parser must classify edges as relationships, not neighbors.

        Uses plain stand-in classes (not MagicMock, which answers hasattr()
        True for everything) so attribute-based classification is tested
        honestly.
        """

        class FakeNode:
            def __init__(self, node_id, name):
                self.id = node_id
                self.labels = ["KPI"]
                self.properties = {"name": name}

        class FakeEdge:
            def __init__(self, edge_id, relation):
                self.id = edge_id
                self.relation = relation

        entity = FakeNode(1, "TRx")
        target = FakeNode(2, "Market_Share")
        edge = FakeEdge(10, "CAUSES")

        mock_graph = MagicMock()
        mock_graph.query.return_value = MagicMock(
            result_set=[[entity, None, [entity, target], [edge], 1]]
        )
        mock_falkordb_client.select_graph.return_value = mock_graph

        backend = self._backend(mock_falkordb_client, falkordb_config, search_config)
        results = await backend.search(entities=ExtractedEntities(kpis=["trx"]))
        assert len(results) == 1
        ctx = results[0].graph_context
        assert ctx["relationship_types"] == ["CAUSES"]
        assert set(ctx["connected_nodes"]) == {"1", "2"}


class TestGraphBackendCausalSubgraph:
    """Tests for GraphBackend causal subgraph retrieval."""

    @pytest.mark.asyncio
    async def test_get_causal_subgraph(self, mock_falkordb_client, falkordb_config, search_config):
        """Test getting a causal subgraph for visualization."""
        # Setup mock nodes and relationships
        mock_node1 = MagicMock()
        mock_node1.id = 1
        mock_node1.labels = ["Brand"]
        mock_node1.properties = {"name": "Kisqali"}

        mock_node2 = MagicMock()
        mock_node2.id = 2
        mock_node2.labels = ["KPI"]
        mock_node2.properties = {"name": "TRx"}

        mock_rel = MagicMock()
        mock_rel.type = "AFFECTS"
        mock_rel.src_node = 1
        mock_rel.dest_node = 2
        mock_rel.properties = {}

        mock_graph = MagicMock()
        mock_graph.query.return_value = MagicMock(result_set=[[mock_node1, mock_rel]])
        mock_falkordb_client.select_graph.return_value = mock_graph

        backend = GraphBackend(
            falkordb_client=mock_falkordb_client,
            falkordb_config=falkordb_config,
            search_config=search_config,
        )

        result = await backend.get_causal_subgraph(max_depth=2, limit=50)

        assert "nodes" in result
        assert "edges" in result
        assert "metadata" in result


class TestGraphBackendHealthCheck:
    """Tests for GraphBackend health check."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_falkordb_client, falkordb_config, search_config):
        """Test healthy graph backend."""
        mock_graph = MagicMock()
        mock_graph.query.return_value = MagicMock(result_set=[[100]])
        mock_falkordb_client.select_graph.return_value = mock_graph

        backend = GraphBackend(
            falkordb_client=mock_falkordb_client,
            falkordb_config=falkordb_config,
            search_config=search_config,
        )

        health = await backend.health_check()

        assert health["status"] == "healthy"
        assert health["error"] is None
        assert health["node_count"] == 100

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(
        self, mock_falkordb_client, falkordb_config, search_config
    ):
        """Test unhealthy graph backend."""
        mock_falkordb_client.select_graph.side_effect = Exception("Connection refused")

        backend = GraphBackend(
            falkordb_client=mock_falkordb_client,
            falkordb_config=falkordb_config,
            search_config=search_config,
        )

        health = await backend.health_check()

        assert health["status"] == "unhealthy"
        assert "Connection refused" in health["error"]
