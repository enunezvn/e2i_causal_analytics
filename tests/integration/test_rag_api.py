"""
Integration tests for E2I Hybrid RAG API Endpoints.

Tests the RAG API endpoints using FastAPI TestClient with mocked backends.
Verifies request/response handling, error cases, and endpoint behavior.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes.rag import get_rag_service, RAGService
from src.rag.types import (
    RetrievalResult,
    RetrievalSource,
    ExtractedEntities,
    BackendHealth,
    BackendStatus,
    SearchStats,
    GraphPath,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_search_results():
    """Create mock search results using correct field names."""
    return [
        RetrievalResult(
            id="doc_001",
            content="Kisqali TRx declined 15% in Q3 due to competitive pressure",
            score=0.92,
            source=RetrievalSource.VECTOR,
            metadata={
                "brand": "Kisqali",
                "kpi": "trx",
                "region": "northeast",
                "time_period": "Q3_2024"
            }
        ),
        RetrievalResult(
            id="doc_002",
            content="Causal analysis shows HCP targeting changes impacted adoption",
            score=0.85,
            source=RetrievalSource.GRAPH,
            metadata={
                "causal_chain": "hcp_targeting -> adoption -> trx",
                "brand": "Kisqali"
            }
        ),
        RetrievalResult(
            id="doc_003",
            content="Regional TRx patterns in Q3 2024",
            score=0.78,
            source=RetrievalSource.FULLTEXT,
            metadata={
                "time_period": "Q3_2024"
            }
        ),
    ]


@pytest.fixture
def mock_search_stats():
    """Create mock search stats."""
    return SearchStats(
        query="Test query",
        total_latency_ms=156.0,
        vector_count=10,
        fulltext_count=5,
        graph_count=3,
        fused_count=10,
        sources_used={"vector": True, "fulltext": True, "graph": True},
        vector_latency_ms=80.0,
        fulltext_latency_ms=30.0,
        graph_latency_ms=45.0
    )


@pytest.fixture
def mock_extracted_entities():
    """Create mock extracted entities."""
    return ExtractedEntities(
        brands=["Kisqali"],
        regions=["northeast"],
        kpis=["trx"],
        agents=[],
        journey_stages=[],
        time_references=["Q3"],
        hcp_segments=[]
    )


@pytest.fixture
def mock_backend_health():
    """Create mock backend health status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backends": {
            "vector": {
                "status": "healthy",
                "latency_ms": 45.2,
                "available": True
            },
            "fulltext": {
                "status": "healthy",
                "latency_ms": 12.5,
                "available": True
            },
            "graph": {
                "status": "healthy",
                "latency_ms": 28.7,
                "available": True
            }
        },
        "monitoring_enabled": True
    }


@pytest.fixture
def mock_rag_service(mock_search_results, mock_search_stats, mock_extracted_entities, mock_backend_health):
    """Create a fully mocked RAG service."""
    service = MagicMock(spec=RAGService)

    # Mock retriever
    service._retriever = AsyncMock()
    service._retriever.search = AsyncMock(return_value=mock_search_results)

    # Mock entity extractor - need to mock extract_entities method directly
    service.entity_extractor = MagicMock()
    service.entity_extractor.extract = MagicMock(return_value=mock_extracted_entities)
    # Also mock the service-level method that the endpoint calls
    service.extract_entities = MagicMock(return_value=mock_extracted_entities)

    # Mock health monitor
    service._health_monitor = AsyncMock()
    service._health_monitor.get_health_status = AsyncMock(return_value=mock_backend_health)

    # Mock search logger
    service._search_logger = MagicMock()
    service._search_logger.get_stats = MagicMock(return_value={
        "total_searches": 1250,
        "searches_today": 45,
        "avg_latency_ms": 156.3,
        "cache_hit_rate": 0.68
    })

    # Mock the search method
    async def mock_search(*args, **kwargs):
        return mock_search_results, mock_search_stats

    service.search = mock_search

    # Mock initialized flag
    service._initialized = True

    return service


@pytest.fixture
def client(mock_rag_service):
    """Create FastAPI test client with mocked service."""
    # Override the dependency
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    yield TestClient(app)
    # Clean up
    app.dependency_overrides.clear()


# =============================================================================
# Search Endpoint Tests
# =============================================================================


class TestSearchEndpoint:
    """Tests for POST /api/v1/rag/search endpoint."""

    def test_search_basic_query(self, client):
        """Test basic hybrid search query."""
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "Why did Kisqali TRx drop in Q3?",
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "query" in data
        assert "stats" in data
        assert data["query"] == "Why did Kisqali TRx drop in Q3?"
        assert len(data["results"]) > 0

    def test_search_with_filters(self, client):
        """Test search with brand and region filters."""
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "Show TRx trends",
                "top_k": 10,
                "filters": {
                    "brands": ["Kisqali"],
                    "regions": ["northeast", "west"]
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_search_vector_only_mode(self, client):
        """Test vector-only search mode."""
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "Semantic search query",
                "mode": "vector"
            }
        )

        assert response.status_code == 200

    def test_search_empty_query(self, client):
        """Test search with empty query returns validation error."""
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "",
                "top_k": 5
            }
        )

        # FastAPI validation should reject empty query
        assert response.status_code in [400, 422]

    def test_search_invalid_top_k(self, client):
        """Test search with invalid top_k value."""
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "Valid query",
                "top_k": 0
            }
        )

        assert response.status_code == 422  # Validation error


# =============================================================================
# Entity Extraction Endpoint Tests
# =============================================================================


class TestEntityExtractEndpoint:
    """Tests for GET /api/v1/rag/entities endpoint."""

    def test_extract_entities_success(self, client, mock_extracted_entities):
        """Test successful entity extraction."""
        response = client.get(
            "/api/v1/rag/entities",
            params={"query": "Why did Kisqali TRx drop in the Northeast during Q3?"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "brands" in data
        assert "Kisqali" in data["brands"]
        assert "regions" in data
        assert "northeast" in data["regions"]
        assert "kpis" in data
        assert "trx" in data["kpis"]
        assert "time_references" in data
        assert "Q3" in data["time_references"]

    def test_extract_entities_empty_query(self, client):
        """Test entity extraction with empty query."""
        response = client.get(
            "/api/v1/rag/entities",
            params={"query": ""}
        )

        # Should either return empty or validation error
        assert response.status_code in [200, 400, 422]


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoint:
    """Tests for GET /api/v1/rag/health endpoint."""

    def test_health_endpoint_returns_status(self, client, mock_backend_health):
        """Test health endpoint returns status information."""
        response = client.get("/api/v1/rag/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "backends" in data


# =============================================================================
# Stats Endpoint Tests
# =============================================================================


class TestStatsEndpoint:
    """Tests for GET /api/v1/rag/stats endpoint."""

    def test_get_stats(self, client):
        """Test getting RAG usage statistics."""
        response = client.get("/api/v1/rag/stats")

        assert response.status_code == 200
        data = response.json()

        # Note: Stats endpoint returns hardcoded values until search logging is configured
        assert "total_searches" in data
        assert "period_hours" in data
        assert "backend_usage" in data


# =============================================================================
# Request Validation Tests
# =============================================================================


class TestRequestValidation:
    """Tests for request validation."""

    def test_missing_required_field(self, client):
        """Test that missing required fields return validation error."""
        response = client.post(
            "/api/v1/rag/search",
            json={}  # Missing query field
        )

        assert response.status_code == 422

    def test_invalid_search_mode(self, client):
        """Test that invalid search mode returns validation error."""
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "Valid query",
                "mode": "invalid_mode"
            }
        )

        assert response.status_code == 422

    def test_top_k_out_of_range(self, client):
        """Test that top_k out of valid range returns validation error."""
        response = client.post(
            "/api/v1/rag/search",
            json={
                "query": "Valid query",
                "top_k": 1000  # Too high
            }
        )

        assert response.status_code == 422


# =============================================================================
# Response Format Tests
# =============================================================================


class TestResponseFormats:
    """Tests for response format consistency."""

    def test_search_response_has_required_fields(self, client):
        """Test search response has all required fields."""
        response = client.post(
            "/api/v1/rag/search",
            json={"query": "Test query"}
        )

        assert response.status_code == 200
        data = response.json()

        # Check required top-level fields
        assert "results" in data
        assert "query" in data
        assert "stats" in data
        assert "entities" in data

    def test_search_result_item_format(self, client):
        """Test individual search result item has correct format."""
        response = client.post(
            "/api/v1/rag/search",
            json={"query": "Test query"}
        )

        assert response.status_code == 200
        data = response.json()

        if data["results"]:
            result = data["results"][0]
            # Check result item fields
            assert "id" in result or "document_id" in result
            assert "content" in result
            assert "score" in result
            assert "source" in result

    def test_entity_response_format(self, client):
        """Test entity extraction response format."""
        response = client.get(
            "/api/v1/rag/entities",
            params={"query": "Test Kisqali"}
        )

        assert response.status_code == 200
        data = response.json()

        # All entity types should be present
        assert "brands" in data
        assert "regions" in data
        assert "kpis" in data
        assert "agents" in data
        assert "journey_stages" in data
        assert "time_references" in data
        assert "hcp_segments" in data

        # All should be lists
        assert isinstance(data["brands"], list)
        assert isinstance(data["regions"], list)
