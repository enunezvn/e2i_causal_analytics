"""
Unit tests for src/api/routes/rag.py

Tests cover:
- RAGService class methods
- All endpoints (search, get_causal_subgraph, get_causal_path, extract_entities, get_health, get_stats)
- Happy paths, error paths, edge cases
- Mock all external dependencies (HybridRetriever, EntityExtractor, HealthMonitor, Supabase)
"""

import pytest
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from fastapi import HTTPException

from src.api.routes.rag import (
    RAGService,
    SearchMode,
    ResultFormat,
    SearchRequest,
    search,
    get_causal_subgraph,
    get_causal_path,
    extract_entities,
    get_health,
    get_stats,
    get_rag_service,
)
from src.rag import RetrievalResult, ExtractedEntities
from src.rag.types import RetrievalSource
from src.rag.exceptions import CircuitBreakerOpenError, BackendTimeoutError, RAGError


# =============================================================================
# TEST DATA CLASSES
# =============================================================================


@dataclass
class QueryStats:
    """Mock QueryStats dataclass for testing."""
    vector_count: int = 0
    fulltext_count: int = 0
    graph_count: int = 0
    total_latency_ms: float = 0.0


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_hybrid_retriever():
    """Mock HybridRetriever."""
    retriever = AsyncMock()
    retriever.search = AsyncMock(return_value=[
        RetrievalResult(
            id="doc1",
            content="Test content 1",
            score=0.9,
            source=RetrievalSource.VECTOR,
            metadata={"brand": "Kisqali"},
        ),
        RetrievalResult(
            id="doc2",
            content="Test content 2",
            score=0.8,
            source=RetrievalSource.FULLTEXT,
            metadata={"region": "northeast"},
        ),
    ])
    retriever.get_last_query_stats = MagicMock(return_value=MagicMock(
        vector_count=1,
        fulltext_count=1,
        graph_count=0,
        total_latency_ms=150.0,
    ))
    retriever.get_causal_subgraph = AsyncMock(return_value={
        "nodes": [
            {"id": "node1", "label": "Kisqali", "type": "brand", "properties": {}},
        ],
        "edges": [
            {"source": "node1", "target": "node2", "relationship": "affects", "weight": 1.0},
        ],
    })
    retriever.get_causal_path = AsyncMock(return_value={
        "paths": [["node1", "node2", "node3"]],
    })
    return retriever


@pytest.fixture
def mock_entity_extractor():
    """Mock EntityExtractor."""
    extractor = MagicMock()
    extractor.extract = MagicMock(return_value=ExtractedEntities(
        brands=["Kisqali"],
        regions=["northeast"],
        kpis=["trx"],
        agents=[],
        journey_stages=[],
        time_references=["Q3"],
        hcp_segments=[],
    ))
    return extractor


@pytest.fixture
def mock_health_monitor():
    """Mock HealthMonitor."""
    monitor = AsyncMock()
    monitor.get_health_status = AsyncMock(return_value={
        "status": "healthy",
        "backends": {
            "vector": {
                "status": "healthy",
                "latency_ms": 50.0,
                "last_check": datetime.now(timezone.utc).isoformat(),
                "consecutive_failures": 0,
            },
        },
        "monitoring_enabled": True,
    })
    return monitor


@pytest.fixture
def rag_service(mock_hybrid_retriever, mock_entity_extractor, mock_health_monitor):
    """Create RAGService with mocked dependencies."""
    with patch("src.api.routes.rag.RAGConfig") as mock_config:
        mock_config.from_env.return_value = MagicMock()

        service = RAGService()
        service._retriever = mock_hybrid_retriever
        service.entity_extractor = mock_entity_extractor
        service._health_monitor = mock_health_monitor

        return service


@pytest.fixture
def sample_search_request():
    """Sample search request."""
    return SearchRequest(
        query="Why did Kisqali TRx drop in the West during Q3?",
        mode=SearchMode.HYBRID,
        top_k=10,
        min_score=0.5,
    )


# =============================================================================
# RAGService Tests
# =============================================================================


class TestRAGService:
    """Tests for RAGService class."""

    def test_service_initialization(self):
        """Test RAGService initialization."""
        with patch("src.api.routes.rag.RAGConfig") as mock_config:
            mock_config.from_env.return_value = MagicMock()

            # Reset class-level state
            RAGService._initialized = False
            RAGService._instance = None

            service = RAGService()

            assert RAGService._initialized is True
            assert service.entity_extractor is not None

    def test_service_singleton(self):
        """Test RAGService singleton pattern."""
        # Reset singleton
        RAGService._instance = None

        service1 = RAGService.get_instance()
        service2 = RAGService.get_instance()

        assert service1 is service2

    def test_retriever_property(self, rag_service):
        """Test retriever property access."""
        retriever = rag_service.retriever

        assert retriever is not None
        assert retriever == rag_service._retriever

    def test_retriever_lazy_creation(self):
        """Test retriever is created lazily."""
        # Reset singleton state
        RAGService._initialized = False
        RAGService._instance = None

        with patch("src.api.routes.rag.RAGConfig") as mock_config, \
             patch("src.api.routes.rag.HybridRetriever") as mock_retriever_class:

            mock_config.from_env.return_value = MagicMock()
            mock_retriever_class.return_value = MagicMock()

            service = RAGService()
            service._retriever = None

            # Access retriever for first time
            retriever = service.retriever

            assert retriever is not None
            mock_retriever_class.assert_called_once()

    def test_health_monitor_property(self, rag_service):
        """Test health_monitor property access."""
        monitor = rag_service.health_monitor

        assert monitor is not None

    def test_health_monitor_lazy_creation(self):
        """Test health monitor is created lazily."""
        with patch("src.api.routes.rag.RAGConfig") as mock_config, \
             patch("src.api.routes.rag.HealthMonitor") as mock_monitor_class:

            mock_config.from_env.return_value = MagicMock()
            mock_monitor_class.return_value = MagicMock()

            service = RAGService()
            service._health_monitor = None

            # Access health_monitor for first time
            monitor = service.health_monitor

            assert monitor is not None
            mock_monitor_class.assert_called_once()

    def test_extract_entities(self, rag_service):
        """Test entity extraction from query."""
        entities = rag_service.extract_entities("Why did Kisqali TRx drop?")

        assert "Kisqali" in entities.brands
        assert "trx" in entities.kpis

    @pytest.mark.asyncio
    async def test_search_success(self, rag_service):
        """Test successful search execution."""
        results, stats = await rag_service.search(
            query="Test query",
            mode=SearchMode.HYBRID,
            top_k=10,
            min_score=0.0,
            include_graph_boost=True,
        )

        assert len(results) == 2
        assert results[0].id == "doc1"
        assert "vector_count" in stats

    @pytest.mark.asyncio
    async def test_search_with_min_score_filter(self, rag_service, mock_hybrid_retriever):
        """Test search applies minimum score filter."""
        # Set up results with different scores
        mock_hybrid_retriever.search.return_value = [
            RetrievalResult(id="doc1", content="High score", score=0.9, source=RetrievalSource.VECTOR),
            RetrievalResult(id="doc2", content="Low score", score=0.3, source=RetrievalSource.VECTOR),
        ]

        results, _ = await rag_service.search(
            query="Test",
            mode=SearchMode.HYBRID,
            top_k=10,
            min_score=0.5,
            include_graph_boost=True,
        )

        assert len(results) == 1
        assert results[0].score >= 0.5

    @pytest.mark.asyncio
    async def test_search_with_filters(self, rag_service, mock_hybrid_retriever):
        """Test search with metadata filters."""
        filters = {"brand": "Kisqali", "region": "northeast"}

        await rag_service.search(
            query="Test",
            mode=SearchMode.HYBRID,
            top_k=10,
            min_score=0.0,
            include_graph_boost=True,
            filters=filters,
        )

        # Verify filters were passed to retriever
        call_args = mock_hybrid_retriever.search.call_args
        assert call_args.kwargs["filters"] == filters

    @pytest.mark.asyncio
    async def test_get_causal_subgraph(self, rag_service):
        """Test causal subgraph retrieval."""
        result = await rag_service.get_causal_subgraph(
            entity="Kisqali",
            depth=2,
            limit=100,
        )

        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) > 0

    @pytest.mark.asyncio
    async def test_get_causal_path(self, rag_service):
        """Test causal path finding."""
        result = await rag_service.get_causal_path(
            source="Kisqali",
            target="TRx",
            max_depth=5,
        )

        assert "paths" in result
        assert len(result["paths"]) > 0

    @pytest.mark.asyncio
    async def test_get_health_status(self, rag_service):
        """Test health status retrieval."""
        health = await rag_service.get_health_status()

        assert health["status"] == "healthy"
        assert "backends" in health


# =============================================================================
# Endpoint Tests
# =============================================================================


class TestSearchEndpoint:
    """Tests for /api/v1/rag/search endpoint."""

    @pytest.mark.asyncio
    async def test_search_success(self, sample_search_request):
        """Test successful search request."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(return_value=ExtractedEntities(
                brands=["Kisqali"],
                regions=["west"],
                kpis=["trx"],
                agents=[],
                journey_stages=[],
                time_references=["Q3"],
                hcp_segments=[],
            ))
            mock_service.search = AsyncMock(return_value=(
                [
                    RetrievalResult(
                        id="doc1",
                        content="Test content",
                        score=0.9,
                        source=RetrievalSource.VECTOR,
                        metadata={"brand": "Kisqali"},
                    ),
                ],
                QueryStats(vector_count=1, fulltext_count=0),
            ))
            mock_get_service.return_value = mock_service

            response = await search(sample_search_request, mock_service)

            assert response.query == sample_search_request.query
            assert len(response.results) == 1
            assert "Kisqali" in response.entities.brands

    @pytest.mark.asyncio
    async def test_search_full_format(self, sample_search_request):
        """Test search with full result format."""
        sample_search_request.format = ResultFormat.FULL

        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(return_value=ExtractedEntities())
            mock_service.search = AsyncMock(return_value=(
                [
                    RetrievalResult(
                        id="doc1",
                        content="Long content " * 50,
                        score=0.9,
                        source=RetrievalSource.VECTOR,
                        metadata={"key": "value"},
                    ),
                ],
                QueryStats(),
            ))
            mock_get_service.return_value = mock_service

            response = await search(sample_search_request, mock_service)

            # Full format includes all content and metadata
            assert len(response.results[0].content) > 200
            assert response.results[0].metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_search_compact_format(self, sample_search_request):
        """Test search with compact result format."""
        sample_search_request.format = ResultFormat.COMPACT

        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(return_value=ExtractedEntities())
            mock_service.search = AsyncMock(return_value=(
                [
                    RetrievalResult(
                        id="doc1",
                        content="Long content " * 50,
                        score=0.9,
                        source=RetrievalSource.VECTOR,
                        metadata={"key": "value"},
                    ),
                ],
                QueryStats(),
            ))
            mock_get_service.return_value = mock_service

            response = await search(sample_search_request, mock_service)

            # Compact format truncates content and excludes metadata
            assert len(response.results[0].content) <= 203  # 200 + "..."
            assert response.results[0].metadata == {}

    @pytest.mark.asyncio
    async def test_search_ids_only_format(self, sample_search_request):
        """Test search with IDs only format."""
        sample_search_request.format = ResultFormat.IDS_ONLY

        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(return_value=ExtractedEntities())
            mock_service.search = AsyncMock(return_value=(
                [
                    RetrievalResult(
                        id="doc1",
                        content="Content",
                        score=0.9,
                        source=RetrievalSource.VECTOR,
                    ),
                ],
                QueryStats(),
            ))
            mock_get_service.return_value = mock_service

            response = await search(sample_search_request, mock_service)

            # IDs only format excludes content
            assert response.results[0].content == ""

    @pytest.mark.asyncio
    async def test_search_circuit_breaker_error(self, sample_search_request):
        """Test search handles circuit breaker errors."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(return_value=ExtractedEntities())
            mock_service.search = AsyncMock(side_effect=CircuitBreakerOpenError(
                backend="vector",
                reset_time_seconds=30.0,
            ))
            mock_get_service.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await search(sample_search_request, mock_service)

            assert exc_info.value.status_code == 503
            assert "temporarily unavailable" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_search_timeout_error(self, sample_search_request):
        """Test search handles timeout errors."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(return_value=ExtractedEntities())
            mock_service.search = AsyncMock(side_effect=BackendTimeoutError(
                backend="vector",
                timeout_ms=5000.0,
            ))
            mock_get_service.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await search(sample_search_request, mock_service)

            assert exc_info.value.status_code == 504

    @pytest.mark.asyncio
    async def test_search_rag_error(self, sample_search_request):
        """Test search handles RAG errors."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(return_value=ExtractedEntities())
            mock_service.search = AsyncMock(side_effect=RAGError("RAG error"))
            mock_get_service.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await search(sample_search_request, mock_service)

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_search_unexpected_error(self, sample_search_request):
        """Test search handles unexpected errors."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(return_value=ExtractedEntities())
            mock_service.search = AsyncMock(side_effect=Exception("Unexpected"))
            mock_get_service.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await search(sample_search_request, mock_service)

            assert exc_info.value.status_code == 500


class TestGetCausalSubgraphEndpoint:
    """Tests for /api/v1/rag/graph/{entity} endpoint."""

    @pytest.mark.asyncio
    async def test_get_subgraph_success(self):
        """Test successful subgraph retrieval."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_causal_subgraph = AsyncMock(return_value={
                "nodes": [
                    {"id": "node1", "label": "Kisqali", "type": "brand", "properties": {"key": "value"}},
                ],
                "edges": [
                    {"source": "node1", "target": "node2", "relationship": "affects", "weight": 0.8, "properties": {}},
                ],
            })
            mock_get_service.return_value = mock_service

            response = await get_causal_subgraph(
                entity="Kisqali",
                depth=2,
                limit=100,
                service=mock_service,
            )

            assert response.entity == "Kisqali"
            assert response.node_count == 1
            assert response.edge_count == 1

    @pytest.mark.asyncio
    async def test_get_subgraph_with_custom_depth(self):
        """Test subgraph with custom depth parameter."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_causal_subgraph = AsyncMock(return_value={
                "nodes": [],
                "edges": [],
            })
            mock_get_service.return_value = mock_service

            await get_causal_subgraph(
                entity="test",
                depth=3,
                limit=50,
                service=mock_service,
            )

            call_args = mock_service.get_causal_subgraph.call_args
            assert call_args.kwargs["depth"] == 3
            assert call_args.kwargs["limit"] == 50

    @pytest.mark.asyncio
    async def test_get_subgraph_error(self):
        """Test subgraph error handling."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_causal_subgraph = AsyncMock(side_effect=RAGError("Graph error"))
            mock_get_service.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await get_causal_subgraph(
                    entity="test",
                    service=mock_service,
                )

            assert exc_info.value.status_code == 500


class TestGetCausalPathEndpoint:
    """Tests for /api/v1/rag/causal-path endpoint."""

    @pytest.mark.asyncio
    async def test_get_path_success(self):
        """Test successful path finding."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_causal_path = AsyncMock(return_value={
                "paths": [
                    ["Kisqali", "HCP", "TRx"],
                    ["Kisqali", "Market", "TRx"],
                ],
            })
            mock_get_service.return_value = mock_service

            response = await get_causal_path(
                source="Kisqali",
                target="TRx",
                max_depth=5,
                service=mock_service,
            )

            assert response.source == "Kisqali"
            assert response.target == "TRx"
            assert response.total_paths == 2
            assert response.shortest_path_length == 3

    @pytest.mark.asyncio
    async def test_get_path_no_paths_found(self):
        """Test path finding when no paths exist."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_causal_path = AsyncMock(return_value={
                "paths": [],
            })
            mock_get_service.return_value = mock_service

            response = await get_causal_path(
                source="A",
                target="B",
                service=mock_service,
            )

            assert response.total_paths == 0
            assert response.shortest_path_length == 0

    @pytest.mark.asyncio
    async def test_get_path_error(self):
        """Test path finding error handling."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_causal_path = AsyncMock(side_effect=Exception("Path error"))
            mock_get_service.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await get_causal_path(
                    source="A",
                    target="B",
                    service=mock_service,
                )

            assert exc_info.value.status_code == 500


class TestExtractEntitiesEndpoint:
    """Tests for /api/v1/rag/entities endpoint."""

    @pytest.mark.asyncio
    async def test_extract_entities_success(self):
        """Test successful entity extraction."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(return_value=ExtractedEntities(
                brands=["Kisqali", "Fabhalta"],
                regions=["northeast"],
                kpis=["trx", "nrx"],
                agents=["causal_impact"],
                journey_stages=["awareness"],
                time_references=["Q3", "2024"],
                hcp_segments=["oncologist"],
            ))
            mock_get_service.return_value = mock_service

            response = await extract_entities(
                query="Why did Kisqali and Fabhalta TRx drop in northeast during Q3 2024?",
                service=mock_service,
            )

            assert len(response.brands) == 2
            assert len(response.kpis) == 2
            assert "northeast" in response.regions

    @pytest.mark.asyncio
    async def test_extract_entities_empty(self):
        """Test entity extraction with no entities found."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(return_value=ExtractedEntities())
            mock_get_service.return_value = mock_service

            response = await extract_entities(
                query="Generic question",
                service=mock_service,
            )

            assert len(response.brands) == 0
            assert len(response.kpis) == 0

    @pytest.mark.asyncio
    async def test_extract_entities_error(self):
        """Test entity extraction error handling."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(side_effect=Exception("Extraction error"))
            mock_get_service.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await extract_entities(
                    query="test",
                    service=mock_service,
                )

            assert exc_info.value.status_code == 500


class TestGetHealthEndpoint:
    """Tests for /api/v1/rag/health endpoint."""

    @pytest.mark.asyncio
    async def test_get_health_healthy(self):
        """Test health check when system is healthy."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_health_status = AsyncMock(return_value={
                "status": "healthy",
                "backends": {
                    "vector": {
                        "status": "healthy",
                        "latency_ms": 50.0,
                        "last_check": datetime.now(timezone.utc).isoformat(),
                        "consecutive_failures": 0,
                        "circuit_breaker": {"state": "closed"},
                    },
                },
                "monitoring_enabled": True,
            })
            mock_get_service.return_value = mock_service

            response = await get_health(mock_service)

            assert response.status == "healthy"
            assert response.monitoring_enabled is True
            assert "vector" in response.backends

    @pytest.mark.asyncio
    async def test_get_health_degraded(self):
        """Test health check when system is degraded."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_health_status = AsyncMock(return_value={
                "status": "degraded",
                "backends": {},
                "monitoring_enabled": False,
            })
            mock_get_service.return_value = mock_service

            response = await get_health(mock_service)

            assert response.status == "degraded"

    @pytest.mark.asyncio
    async def test_get_health_error(self):
        """Test health check error handling."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_health_status = AsyncMock(side_effect=Exception("Health error"))
            mock_get_service.return_value = mock_service

            response = await get_health(mock_service)

            # Should return degraded status on error
            assert response.status == "degraded"


class TestGetStatsEndpoint:
    """Tests for /api/v1/rag/stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self):
        """Test successful stats retrieval."""
        with patch("src.api.routes.rag.get_supabase_client") as mock_get_sb:
            mock_client = MagicMock()
            mock_rpc = MagicMock()
            mock_execute = MagicMock()
            mock_execute.data = {
                "period_hours": 24,
                "total_searches": 100,
                "avg_latency_ms": 150.5,
                "p95_latency_ms": 450.2,
                "avg_results": 8.5,
                "error_rate": 0.02,
                "backend_usage": {"vector": 80, "fulltext": 50, "graph": 30},
                "top_queries": [{"query": "test", "count": 10, "avg_latency": 120.0}],
            }
            mock_rpc.execute.return_value = mock_execute
            mock_client.rpc.return_value = mock_rpc
            mock_get_sb.return_value = mock_client

            response = await get_stats(hours=24)

            assert response["total_searches"] == 100
            assert response["avg_latency_ms"] == 150.5
            assert response["error_rate"] == 0.02

    @pytest.mark.asyncio
    async def test_get_stats_no_data(self):
        """Test stats when no data available."""
        with patch("src.api.routes.rag.get_supabase_client") as mock_get_sb:
            mock_client = MagicMock()
            mock_rpc = MagicMock()
            mock_execute = MagicMock()
            mock_execute.data = None
            mock_rpc.execute.return_value = mock_execute
            mock_client.rpc.return_value = mock_rpc
            mock_get_sb.return_value = mock_client

            response = await get_stats(hours=24)

            assert response["total_searches"] == 0
            assert response["avg_latency_ms"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_custom_period(self):
        """Test stats with custom time period."""
        with patch("src.api.routes.rag.get_supabase_client") as mock_get_sb:
            mock_client = MagicMock()
            mock_rpc = MagicMock()
            mock_execute = MagicMock()
            mock_execute.data = {
                "period_hours": 72,
                "total_searches": 300,
            }
            mock_rpc.execute.return_value = mock_execute
            mock_client.rpc.return_value = mock_rpc
            mock_get_sb.return_value = mock_client

            response = await get_stats(hours=72)

            mock_client.rpc.assert_called_with("get_rag_search_stats", {"hours_lookback": 72})

    @pytest.mark.asyncio
    async def test_get_stats_error(self):
        """Test stats error handling."""
        with patch("src.api.routes.rag.get_supabase_client") as mock_get_sb:
            mock_get_sb.side_effect = Exception("DB error")

            response = await get_stats(hours=24)

            # Should return empty stats with error
            assert response["total_searches"] == 0
            assert "error" in response


class TestGetRagServiceFunction:
    """Tests for get_rag_service singleton function."""

    def test_get_rag_service_singleton(self):
        """Test get_rag_service returns singleton."""
        # Reset singleton
        RAGService._instance = None

        service1 = get_rag_service()
        service2 = get_rag_service()

        assert service1 is service2


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        """Test search with no results."""
        request = SearchRequest(query="nonexistent query")

        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(return_value=ExtractedEntities())
            mock_service.search = AsyncMock(return_value=([], QueryStats()))
            mock_get_service.return_value = mock_service

            response = await search(request, mock_service)

            assert response.total_results == 0
            assert len(response.results) == 0

    @pytest.mark.asyncio
    async def test_search_with_max_top_k(self):
        """Test search with maximum top_k value."""
        request = SearchRequest(query="test", top_k=50)  # Max allowed

        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.extract_entities = MagicMock(return_value=ExtractedEntities())
            mock_service.search = AsyncMock(return_value=([], QueryStats()))
            mock_get_service.return_value = mock_service

            await search(request, mock_service)

            call_args = mock_service.search.call_args
            assert call_args.kwargs["top_k"] == 50

    @pytest.mark.asyncio
    async def test_search_different_modes(self):
        """Test search with different search modes."""
        modes = [SearchMode.HYBRID, SearchMode.VECTOR_ONLY, SearchMode.FULLTEXT_ONLY, SearchMode.GRAPH_ONLY]

        for mode in modes:
            request = SearchRequest(query="test", mode=mode)

            with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
                mock_service = MagicMock()
                mock_service.extract_entities = MagicMock(return_value=ExtractedEntities())
                mock_service.search = AsyncMock(return_value=([], QueryStats()))
                mock_get_service.return_value = mock_service

                response = await search(request, mock_service)

                assert response is not None

    @pytest.mark.asyncio
    async def test_get_subgraph_empty_graph(self):
        """Test subgraph retrieval when no graph exists."""
        with patch("src.api.routes.rag.get_rag_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_causal_subgraph = AsyncMock(return_value={
                "nodes": [],
                "edges": [],
            })
            mock_get_service.return_value = mock_service

            response = await get_causal_subgraph(
                entity="nonexistent",
                depth=2,  # Pass explicit value instead of Query default
                limit=100,  # Pass explicit value instead of Query default
                service=mock_service,
            )

            assert response.node_count == 0
            assert response.edge_count == 0
