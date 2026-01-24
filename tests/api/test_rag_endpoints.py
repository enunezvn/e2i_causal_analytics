"""
Tests for RAG API endpoints.

Phase 3B of API Audit - RAG API
Tests organized by batch as per api-endpoints-audit-plan.md

Endpoints covered:
- Batch 3B.1: Hybrid Search (POST /api/v1/rag/search, GET /api/v1/rag/entities, GET /api/v1/rag/health)
- Batch 3B.2: Graph Operations (GET /api/v1/rag/graph/{entity}, GET /api/v1/rag/causal-path, GET /api/v1/rag/stats)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes.rag import get_rag_service
from src.rag.types import SearchStats

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_extracted_entities():
    """Mock extracted entities from query."""
    entities = MagicMock()
    entities.brands = ["Kisqali"]
    entities.regions = ["west"]
    entities.kpis = ["trx"]
    entities.agents = []
    entities.journey_stages = []
    entities.time_references = ["Q3"]
    entities.hcp_segments = []
    return entities


@pytest.fixture
def mock_retrieval_result():
    """Mock retrieval result."""
    result = MagicMock()
    result.id = "doc_001"
    result.content = "Kisqali TRx declined in the West region during Q3 2024."
    result.score = 0.85
    result.source = MagicMock(value="vector")
    result.metadata = {"brand": "Kisqali", "region": "west"}
    return result


@pytest.fixture
def mock_query_stats():
    """Mock query statistics as proper SearchStats dataclass."""
    return SearchStats(
        query="test query",
        total_latency_ms=150.5,
        vector_count=5,
        fulltext_count=3,
        graph_count=2,
        fused_count=10,
        sources_used={"vector": True, "fulltext": True, "graph": True},
        vector_latency_ms=50.0,
        fulltext_latency_ms=30.0,
        graph_latency_ms=40.0,
        fusion_latency_ms=30.5,
    )


@pytest.fixture
def mock_subgraph_result():
    """Mock causal subgraph result."""
    return {
        "nodes": [
            {"id": "kisqali", "label": "Kisqali", "type": "brand", "properties": {}},
            {"id": "trx", "label": "TRx", "type": "kpi", "properties": {}},
            {"id": "west", "label": "West", "type": "region", "properties": {}},
        ],
        "edges": [
            {"source": "kisqali", "target": "trx", "relationship": "has_kpi", "weight": 1.0, "properties": {}},
            {"source": "west", "target": "trx", "relationship": "affects", "weight": 0.8, "properties": {}},
        ],
    }


@pytest.fixture
def mock_path_result():
    """Mock causal path result."""
    return {
        "paths": [
            ["kisqali", "west_territory", "trx_decline"],
            ["kisqali", "hcp_engagement", "west_territory", "trx_decline"],
        ],
    }


@pytest.fixture
def mock_health_result():
    """Mock health status result."""
    return {
        "status": "healthy",
        "backends": {
            "vector": {
                "status": "healthy",
                "latency_ms": 25.5,
                "last_check": datetime.now(timezone.utc).isoformat(),
                "consecutive_failures": 0,
                "circuit_breaker": {"state": "closed"},
                "error": None,
            },
            "fulltext": {
                "status": "healthy",
                "latency_ms": 15.2,
                "last_check": datetime.now(timezone.utc).isoformat(),
                "consecutive_failures": 0,
                "circuit_breaker": {"state": "closed"},
                "error": None,
            },
            "graph": {
                "status": "healthy",
                "latency_ms": 35.8,
                "last_check": datetime.now(timezone.utc).isoformat(),
                "consecutive_failures": 0,
                "circuit_breaker": {"state": "closed"},
                "error": None,
            },
        },
        "monitoring_enabled": True,
    }


@pytest.fixture
def mock_rag_service(
    mock_extracted_entities,
    mock_retrieval_result,
    mock_query_stats,
    mock_subgraph_result,
    mock_path_result,
    mock_health_result,
):
    """Mock RAGService instance."""
    service = MagicMock()

    # Mock entity extraction
    service.extract_entities = MagicMock(return_value=mock_extracted_entities)

    # Mock search
    service.search = AsyncMock(return_value=([mock_retrieval_result], mock_query_stats))

    # Mock subgraph
    service.get_causal_subgraph = AsyncMock(return_value=mock_subgraph_result)

    # Mock path finding
    service.get_causal_path = AsyncMock(return_value=mock_path_result)

    # Mock health
    service.get_health_status = AsyncMock(return_value=mock_health_result)

    return service


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Clean up dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


# =============================================================================
# BATCH 3B.1 - HYBRID SEARCH TESTS
# =============================================================================


class TestHybridSearch:
    """Tests for POST /api/v1/rag/search."""

    def test_search_success(self, mock_rag_service):
        """Should return search results with hybrid mode."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "Why did Kisqali TRx drop in the West during Q3?",
                "mode": "hybrid",
                "top_k": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "search_id" in data
        assert data["query"] == "Why did Kisqali TRx drop in the West during Q3?"
        assert "results" in data
        assert "total_results" in data
        assert "entities" in data
        assert "stats" in data
        assert "latency_ms" in data

    def test_search_with_filters(self, mock_rag_service):
        """Should accept optional filters."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "TRx trends for Remibrutinib",
                "filters": {"brand": "Remibrutinib", "time_period": "Q4_2024"},
            },
        )

        assert response.status_code == 200

    def test_search_vector_only_mode(self, mock_rag_service):
        """Should support vector-only search mode."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "Market share analysis",
                "mode": "vector",
                "top_k": 5,
            },
        )

        assert response.status_code == 200

    def test_search_with_min_score_filter(self, mock_rag_service):
        """Should filter results by minimum score."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "Fabhalta adoption rate",
                "min_score": 0.7,
            },
        )

        assert response.status_code == 200

    def test_search_compact_format(self, mock_rag_service):
        """Should return compact format when requested."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "Conversion rates by region",
                "format": "compact",
            },
        )

        assert response.status_code == 200

    def test_search_circuit_breaker_open(self, mock_rag_service):
        """Should return 503 when circuit breaker is open."""
        from src.rag.exceptions import CircuitBreakerOpenError

        mock_rag_service.search = AsyncMock(
            side_effect=CircuitBreakerOpenError(backend="vector", reset_time_seconds=30)
        )
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.post(
            "/api/v1/rag/search",
            json={"query": "Test query"},
        )

        assert response.status_code == 503
        assert "unavailable" in response.json()["message"].lower()

    def test_search_timeout(self, mock_rag_service):
        """Should return 504 on timeout."""
        from src.rag.exceptions import BackendTimeoutError

        mock_rag_service.search = AsyncMock(
            side_effect=BackendTimeoutError(backend="vector", timeout_ms=30000)
        )
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.post(
            "/api/v1/rag/search",
            json={"query": "Test query"},
        )

        assert response.status_code == 504
        assert "timed out" in response.json()["message"].lower()


class TestEntityExtraction:
    """Tests for GET /api/v1/rag/entities."""

    def test_extract_entities_success(self, mock_rag_service):
        """Should extract entities from query."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get(
            "/api/v1/rag/entities",
            params={"query": "Why did Kisqali TRx drop in the West during Q3?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "brands" in data
        assert "regions" in data
        assert "kpis" in data
        assert "time_references" in data
        assert data["brands"] == ["Kisqali"]
        assert data["regions"] == ["west"]
        assert data["kpis"] == ["trx"]

    def test_extract_entities_empty_query(self, mock_rag_service):
        """Should reject empty query."""
        response = client.get("/api/v1/rag/entities", params={"query": ""})

        assert response.status_code == 422  # Validation error

    def test_extract_entities_error_handling(self, mock_rag_service):
        """Should return 500 on extraction error."""
        mock_rag_service.extract_entities = MagicMock(
            side_effect=Exception("Extraction failed")
        )
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get(
            "/api/v1/rag/entities",
            params={"query": "Test query"},
        )

        assert response.status_code == 500
        assert "extraction failed" in response.json()["detail"].lower()


class TestRAGHealth:
    """Tests for GET /api/v1/rag/health."""

    def test_health_check_healthy(self, mock_rag_service):
        """Should return healthy status."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get("/api/v1/rag/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "backends" in data
        assert "monitoring_enabled" in data

    def test_health_check_includes_backends(self, mock_rag_service):
        """Should include status for all backends."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get("/api/v1/rag/health")

        assert response.status_code == 200
        data = response.json()
        backends = data["backends"]
        assert "vector" in backends
        assert "fulltext" in backends
        assert "graph" in backends

        # Check backend structure
        for backend in backends.values():
            assert "status" in backend
            assert "latency_ms" in backend
            assert "last_check" in backend

    def test_health_check_degraded(self, mock_rag_service):
        """Should return degraded status when backend is down."""
        mock_rag_service.get_health_status = AsyncMock(
            return_value={
                "status": "degraded",
                "backends": {
                    "vector": {
                        "status": "unhealthy",
                        "latency_ms": 0,
                        "last_check": datetime.now(timezone.utc).isoformat(),
                        "consecutive_failures": 5,
                        "error": "Connection refused",
                    },
                    "fulltext": {
                        "status": "healthy",
                        "latency_ms": 15.0,
                        "last_check": datetime.now(timezone.utc).isoformat(),
                    },
                    "graph": {
                        "status": "healthy",
                        "latency_ms": 30.0,
                        "last_check": datetime.now(timezone.utc).isoformat(),
                    },
                },
                "monitoring_enabled": True,
            }
        )
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get("/api/v1/rag/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["backends"]["vector"]["status"] == "unhealthy"

    def test_health_check_error_returns_degraded(self, mock_rag_service):
        """Should return degraded status on health check error."""
        mock_rag_service.get_health_status = AsyncMock(
            side_effect=Exception("Health check failed")
        )
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get("/api/v1/rag/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"


# =============================================================================
# BATCH 3B.2 - GRAPH OPERATIONS TESTS
# =============================================================================


class TestCausalSubgraph:
    """Tests for GET /api/v1/rag/graph/{entity}."""

    def test_get_subgraph_success(self, mock_rag_service):
        """Should return causal subgraph for entity."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get("/api/v1/rag/graph/kisqali")

        assert response.status_code == 200
        data = response.json()
        assert data["entity"] == "kisqali"
        assert "nodes" in data
        assert "edges" in data
        assert "depth" in data
        assert "node_count" in data
        assert "edge_count" in data
        assert "query_time_ms" in data
        assert data["node_count"] == 3
        assert data["edge_count"] == 2

    def test_get_subgraph_with_depth(self, mock_rag_service):
        """Should respect depth parameter."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get("/api/v1/rag/graph/trx", params={"depth": 3})

        assert response.status_code == 200
        data = response.json()
        assert data["depth"] == 3
        mock_rag_service.get_causal_subgraph.assert_called_with(
            entity="trx",
            depth=3,
            limit=100,
        )

    def test_get_subgraph_with_limit(self, mock_rag_service):
        """Should respect limit parameter."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get("/api/v1/rag/graph/west", params={"limit": 50})

        assert response.status_code == 200
        mock_rag_service.get_causal_subgraph.assert_called_with(
            entity="west",
            depth=2,
            limit=50,
        )

    def test_get_subgraph_error_handling(self, mock_rag_service):
        """Should return 500 on graph error."""
        from src.rag.exceptions import RAGError

        mock_rag_service.get_causal_subgraph = AsyncMock(
            side_effect=RAGError("Graph query failed")
        )
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get("/api/v1/rag/graph/unknown_entity")

        assert response.status_code == 500


class TestCausalPath:
    """Tests for GET /api/v1/rag/causal-path."""

    def test_find_path_success(self, mock_rag_service):
        """Should find causal paths between entities."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get(
            "/api/v1/rag/causal-path",
            params={"source": "kisqali", "target": "trx_decline"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "kisqali"
        assert data["target"] == "trx_decline"
        assert "paths" in data
        assert "shortest_path_length" in data
        assert "total_paths" in data
        assert "query_time_ms" in data
        assert data["total_paths"] == 2

    def test_find_path_with_max_depth(self, mock_rag_service):
        """Should respect max_depth parameter."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get(
            "/api/v1/rag/causal-path",
            params={
                "source": "brand_a",
                "target": "kpi_b",
                "max_depth": 3,
            },
        )

        assert response.status_code == 200
        mock_rag_service.get_causal_path.assert_called_with(
            source="brand_a",
            target="kpi_b",
            max_depth=3,
        )

    def test_find_path_no_paths_found(self, mock_rag_service):
        """Should return empty paths when no connection found."""
        mock_rag_service.get_causal_path = AsyncMock(return_value={"paths": []})
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get(
            "/api/v1/rag/causal-path",
            params={"source": "entity_a", "target": "entity_b"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_paths"] == 0
        assert data["paths"] == []
        assert data["shortest_path_length"] == 0

    def test_find_path_missing_params(self):
        """Should reject request without required params."""
        response = client.get("/api/v1/rag/causal-path")

        assert response.status_code == 422  # Validation error

    def test_find_path_error_handling(self, mock_rag_service):
        """Should return 500 on path finding error."""
        from src.rag.exceptions import RAGError

        mock_rag_service.get_causal_path = AsyncMock(
            side_effect=RAGError("Path query failed")
        )
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get(
            "/api/v1/rag/causal-path",
            params={"source": "a", "target": "b"},
        )

        assert response.status_code == 500


class TestRAGStats:
    """Tests for GET /api/v1/rag/stats."""

    def test_get_stats_success(self, mock_rag_service):
        """Should return usage statistics."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get("/api/v1/rag/stats")

        assert response.status_code == 200
        data = response.json()
        assert "period_hours" in data
        assert "total_searches" in data
        assert "avg_latency_ms" in data
        assert "backend_usage" in data

    def test_get_stats_with_hours_param(self, mock_rag_service):
        """Should respect hours parameter."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get("/api/v1/rag/stats", params={"hours": 48})

        assert response.status_code == 200
        data = response.json()
        assert data["period_hours"] == 48

    def test_get_stats_includes_backend_usage(self, mock_rag_service):
        """Should include per-backend usage breakdown."""
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
        response = client.get("/api/v1/rag/stats")

        assert response.status_code == 200
        data = response.json()
        backend_usage = data["backend_usage"]
        assert "vector" in backend_usage
        assert "fulltext" in backend_usage
        assert "graph" in backend_usage
