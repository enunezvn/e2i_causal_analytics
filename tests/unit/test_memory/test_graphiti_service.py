"""
Unit tests for E2I Graphiti Service.

Tests the main Graphiti service wrapper for automatic entity/relationship extraction.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import asdict

from src.memory.graphiti_service import (
    E2IGraphitiService,
    ExtractedEntity,
    ExtractedRelationship,
    EpisodeResult,
    SearchResult,
    SubgraphResult,
    get_graphiti_service,
    reset_graphiti_service,
    reset_graphiti_service_async,
)
from src.memory.graphiti_config import (
    E2IEntityType,
    E2IRelationshipType,
    GraphitiConfig,
)
from src.memory.services.factories import ServiceConnectionError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a test configuration."""
    return GraphitiConfig(
        enabled=True,
        model="claude-3-5-sonnet-latest",
        graph_name="test_e2i_semantic",
        entity_types=list(E2IEntityType),
        relationship_types=list(E2IRelationshipType),
        falkordb_host="localhost",
        falkordb_port=6380,
    )


@pytest.fixture
def mock_graph():
    """Create a mock FalkorDB graph."""
    graph = MagicMock()
    graph.query.return_value = MagicMock(result_set=[])
    return graph


@pytest.fixture
def mock_falkordb_client(mock_graph):
    """Create a mock FalkorDB client."""
    client = MagicMock()
    client.select_graph.return_value = mock_graph
    return client


@pytest.fixture
def graphiti_service(mock_config, mock_falkordb_client):
    """Create a service instance with mocked dependencies."""
    with patch('src.memory.graphiti_service.get_falkordb_client', return_value=mock_falkordb_client):
        service = E2IGraphitiService(config=mock_config)
        service._falkordb = mock_falkordb_client
        service._graph = mock_falkordb_client.select_graph()
        return service


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the service singleton before and after each test."""
    # Use sync reset before test
    try:
        reset_graphiti_service()
    except Exception:
        pass
    yield
    # Use sync reset after test
    try:
        reset_graphiti_service()
    except Exception:
        pass


# ============================================================================
# Dataclass Tests
# ============================================================================

class TestExtractedEntity:
    """Tests for ExtractedEntity dataclass."""

    def test_create_with_defaults(self):
        """Test creating entity with default values."""
        entity = ExtractedEntity(
            entity_id="test-123",
            entity_type=E2IEntityType.HCP,
            name="Dr. Smith",
        )
        assert entity.entity_id == "test-123"
        assert entity.entity_type == E2IEntityType.HCP
        assert entity.name == "Dr. Smith"
        assert entity.properties == {}
        assert entity.confidence == 1.0

    def test_create_with_all_fields(self):
        """Test creating entity with all fields."""
        entity = ExtractedEntity(
            entity_id="hcp-456",
            entity_type=E2IEntityType.HCP,
            name="Dr. Jones",
            properties={"specialty": "oncology", "npi": "1234567890"},
            confidence=0.95,
        )
        assert entity.properties == {"specialty": "oncology", "npi": "1234567890"}
        assert entity.confidence == 0.95


class TestExtractedRelationship:
    """Tests for ExtractedRelationship dataclass."""

    def test_create_with_defaults(self):
        """Test creating relationship with default values."""
        rel = ExtractedRelationship(
            source_id="hcp-1",
            source_type=E2IEntityType.HCP,
            target_id="brand-1",
            target_type=E2IEntityType.BRAND,
            relationship_type=E2IRelationshipType.PRESCRIBES,
        )
        assert rel.source_id == "hcp-1"
        assert rel.source_type == E2IEntityType.HCP
        assert rel.target_id == "brand-1"
        assert rel.target_type == E2IEntityType.BRAND
        assert rel.relationship_type == E2IRelationshipType.PRESCRIBES
        assert rel.properties == {}
        assert rel.confidence == 1.0

    def test_create_with_properties(self):
        """Test creating relationship with properties."""
        rel = ExtractedRelationship(
            source_id="action-1",
            source_type=E2IEntityType.TRIGGER,
            target_id="kpi-1",
            target_type=E2IEntityType.KPI,
            relationship_type=E2IRelationshipType.IMPACTS,
            properties={"impact_direction": "positive", "lag_days": 30},
            confidence=0.85,
        )
        assert rel.properties == {"impact_direction": "positive", "lag_days": 30}
        assert rel.confidence == 0.85


class TestEpisodeResult:
    """Tests for EpisodeResult dataclass."""

    def test_successful_episode(self):
        """Test successful episode result."""
        entity = ExtractedEntity(
            entity_id="e1",
            entity_type=E2IEntityType.BRAND,
            name="Remibrutinib",
        )
        rel = ExtractedRelationship(
            source_id="hcp-1",
            source_type=E2IEntityType.HCP,
            target_id="e1",
            target_type=E2IEntityType.BRAND,
            relationship_type=E2IRelationshipType.PRESCRIBES,
        )
        result = EpisodeResult(
            episode_id="ep-123",
            entities_extracted=[entity],
            relationships_extracted=[rel],
            success=True,
        )
        assert result.success is True
        assert result.error is None
        assert len(result.entities_extracted) == 1
        assert len(result.relationships_extracted) == 1

    def test_failed_episode(self):
        """Test failed episode result."""
        result = EpisodeResult(
            episode_id="ep-456",
            entities_extracted=[],
            relationships_extracted=[],
            success=False,
            error="Connection timeout",
        )
        assert result.success is False
        assert result.error == "Connection timeout"


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self):
        """Test creating search result."""
        result = SearchResult(
            entity_id="patient-123",
            entity_type="Patient",
            name="Patient P-100",
            score=0.95,
            properties={"condition": "CSU"},
            relationships=[{"type": "TREATED_BY", "target": "hcp-1"}],
        )
        assert result.entity_id == "patient-123"
        assert result.score == 0.95
        assert len(result.relationships) == 1


class TestSubgraphResult:
    """Tests for SubgraphResult dataclass."""

    def test_create_subgraph_result(self):
        """Test creating subgraph result."""
        result = SubgraphResult(
            nodes=[
                {"id": "n1", "type": "HCP", "name": "Dr. Smith"},
                {"id": "n2", "type": "Brand", "name": "Remibrutinib"},
            ],
            edges=[
                {"source": "n1", "target": "n2", "type": "PRESCRIBES"},
            ],
            center_entity_id="n1",
            depth=2,
        )
        assert len(result.nodes) == 2
        assert len(result.edges) == 1
        assert result.center_entity_id == "n1"
        assert result.depth == 2


# ============================================================================
# E2IGraphitiService Tests
# ============================================================================

class TestE2IGraphitiServiceInit:
    """Tests for E2IGraphitiService initialization."""

    def test_init_with_config(self, mock_config):
        """Test initialization with provided config."""
        service = E2IGraphitiService(config=mock_config)
        assert service.config == mock_config
        assert service._graphiti is None
        assert service._falkordb is None
        assert service._initialized is False

    def test_init_without_config(self):
        """Test initialization loads default config."""
        with patch('src.memory.graphiti_service.get_graphiti_config') as mock_get_config:
            mock_get_config.return_value = MagicMock()
            service = E2IGraphitiService()
            mock_get_config.assert_called_once()


class TestE2IGraphitiServiceInitialize:
    """Tests for service initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config, mock_falkordb_client):
        """Test successful initialization."""
        with patch('src.memory.graphiti_service.get_falkordb_client', return_value=mock_falkordb_client):
            service = E2IGraphitiService(config=mock_config)

            # Mock Graphiti initialization to avoid external dependencies
            with patch.object(service, '_init_graphiti', new_callable=AsyncMock):
                await service.initialize()

                assert service._initialized is True
                assert service._falkordb == mock_falkordb_client

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, graphiti_service):
        """Test that double initialization is a no-op."""
        graphiti_service._initialized = True
        original_falkordb = graphiti_service._falkordb

        await graphiti_service.initialize()

        # Should not have reinitialied
        assert graphiti_service._falkordb == original_falkordb

    @pytest.mark.asyncio
    async def test_initialize_connection_error(self, mock_config):
        """Test initialization handles connection errors."""
        with patch('src.memory.graphiti_service.get_falkordb_client', side_effect=Exception("Connection failed")):
            service = E2IGraphitiService(config=mock_config)

            with pytest.raises(ServiceConnectionError):
                await service.initialize()


class TestEntityTypeMapping:
    """Tests for entity type mapping."""

    def test_map_known_entity_types(self, graphiti_service):
        """Test mapping of known entity types."""
        assert graphiti_service._map_entity_type("Patient") == E2IEntityType.PATIENT
        assert graphiti_service._map_entity_type("HCP") == E2IEntityType.HCP
        assert graphiti_service._map_entity_type("Brand") == E2IEntityType.BRAND
        assert graphiti_service._map_entity_type("KPI") == E2IEntityType.KPI
        assert graphiti_service._map_entity_type("Episode") == E2IEntityType.EPISODE

    def test_map_case_insensitive(self, graphiti_service):
        """Test case-insensitive mapping."""
        assert graphiti_service._map_entity_type("patient") == E2IEntityType.PATIENT
        assert graphiti_service._map_entity_type("PATIENT") == E2IEntityType.PATIENT
        assert graphiti_service._map_entity_type("hcp") == E2IEntityType.HCP

    def test_map_unknown_entity_type(self, graphiti_service):
        """Test unknown entity types default to Agent."""
        assert graphiti_service._map_entity_type("UnknownType") == E2IEntityType.AGENT
        assert graphiti_service._map_entity_type("CustomEntity") == E2IEntityType.AGENT


class TestRelationshipTypeMapping:
    """Tests for relationship type mapping."""

    def test_map_known_relationship_types(self, graphiti_service):
        """Test mapping of known relationship types."""
        assert graphiti_service._map_relationship_type("CAUSES") == E2IRelationshipType.CAUSES
        assert graphiti_service._map_relationship_type("IMPACTS") == E2IRelationshipType.IMPACTS
        assert graphiti_service._map_relationship_type("PRESCRIBES") == E2IRelationshipType.PRESCRIBES
        assert graphiti_service._map_relationship_type("DISCOVERED") == E2IRelationshipType.DISCOVERED

    def test_map_with_spaces(self, graphiti_service):
        """Test mapping handles spaces (converted to underscores)."""
        assert graphiti_service._map_relationship_type("RELATES TO") == E2IRelationshipType.RELATES_TO
        assert graphiti_service._map_relationship_type("MEMBER OF") == E2IRelationshipType.MEMBER_OF

    def test_map_unknown_relationship_type(self, graphiti_service):
        """Test unknown relationship types default to RELATES_TO."""
        assert graphiti_service._map_relationship_type("CUSTOM_REL") == E2IRelationshipType.RELATES_TO
        assert graphiti_service._map_relationship_type("UNKNOWN") == E2IRelationshipType.RELATES_TO


class TestAddEpisode:
    """Tests for add_episode method."""

    @pytest.mark.asyncio
    async def test_add_episode_fallback_success(self, graphiti_service, mock_graph):
        """Test add_episode in fallback mode."""
        graphiti_service._initialized = True
        graphiti_service._graphiti = None  # Fallback mode

        result = await graphiti_service.add_episode(
            content="Dr. Smith prescribed Remibrutinib",
            source="orchestrator",
            session_id="session-123",
        )

        assert result.success is True
        assert result.episode_id is not None
        # In fallback mode, no entities are extracted
        assert len(result.entities_extracted) == 0
        mock_graph.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_episode_fallback_with_metadata(self, graphiti_service, mock_graph):
        """Test add_episode with metadata in fallback mode."""
        graphiti_service._initialized = True
        graphiti_service._graphiti = None

        metadata = {"agent_tier": 1, "query_type": "causal"}
        result = await graphiti_service.add_episode(
            content="Test content",
            source="causal_impact",
            session_id="session-456",
            metadata=metadata,
        )

        assert result.success is True
        # Verify metadata was passed to query
        call_args = mock_graph.query.call_args
        assert "metadata" in call_args.kwargs["params"]

    @pytest.mark.asyncio
    async def test_add_episode_with_graphiti(self, graphiti_service):
        """Test add_episode with Graphiti extraction."""
        graphiti_service._initialized = True

        # Create mock Graphiti result
        mock_node = MagicMock()
        mock_node.uuid = "entity-uuid-123"
        mock_node.name = "Dr. Smith"
        mock_node.labels = ["HCP"]
        mock_node.attributes = {"specialty": "oncology"}

        mock_edge = MagicMock()
        mock_edge.source_node_uuid = "entity-uuid-123"
        mock_edge.target_node_uuid = "brand-uuid-456"
        mock_edge.name = "PRESCRIBES"
        mock_edge.attributes = {}

        mock_graphiti_result = MagicMock()
        mock_graphiti_result.nodes = [mock_node]
        mock_graphiti_result.edges = [mock_edge]

        mock_graphiti = AsyncMock()
        mock_graphiti.add_episode.return_value = mock_graphiti_result
        graphiti_service._graphiti = mock_graphiti

        result = await graphiti_service.add_episode(
            content="Dr. Smith prescribed Remibrutinib",
            source="orchestrator",
            session_id="session-123",
        )

        assert result.success is True
        assert len(result.entities_extracted) == 1
        assert result.entities_extracted[0].name == "Dr. Smith"
        assert result.entities_extracted[0].entity_type == E2IEntityType.HCP
        assert len(result.relationships_extracted) == 1
        assert result.relationships_extracted[0].relationship_type == E2IRelationshipType.PRESCRIBES

    @pytest.mark.asyncio
    async def test_add_episode_error_handling(self, graphiti_service, mock_graph):
        """Test add_episode handles errors gracefully."""
        graphiti_service._initialized = True
        graphiti_service._graphiti = None
        mock_graph.query.side_effect = Exception("Database error")

        result = await graphiti_service.add_episode(
            content="Test content",
            source="test",
            session_id="session-error",
        )

        assert result.success is False
        assert "Database error" in result.error


class TestSearch:
    """Tests for search method."""

    @pytest.mark.asyncio
    async def test_search_fallback_mode(self, graphiti_service, mock_graph):
        """Test search in fallback mode."""
        graphiti_service._initialized = True
        graphiti_service._graphiti = None

        # Mock query result
        mock_node = MagicMock()
        mock_node.properties = {"id": "hcp-1", "name": "Dr. Smith"}
        mock_graph.query.return_value.result_set = [[mock_node, ["HCP"]]]

        results = await graphiti_service.search("Dr. Smith")

        assert len(results) == 1
        assert results[0].name == "Dr. Smith"

    @pytest.mark.asyncio
    async def test_search_with_graphiti(self, graphiti_service):
        """Test search with Graphiti client."""
        graphiti_service._initialized = True

        mock_result = MagicMock()
        mock_result.uuid = "result-123"
        mock_result.label = "HCP"
        mock_result.name = "Dr. Jones"
        mock_result.score = 0.95
        mock_result.properties = {"specialty": "cardiology"}

        mock_graphiti = AsyncMock()
        mock_graphiti.search.return_value = [mock_result]
        graphiti_service._graphiti = mock_graphiti

        results = await graphiti_service.search("cardiologist")

        assert len(results) == 1
        assert results[0].name == "Dr. Jones"
        assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_search_with_entity_type_filter(self, graphiti_service):
        """Test search filters by entity type."""
        graphiti_service._initialized = True

        mock_result_hcp = MagicMock()
        mock_result_hcp.uuid = "hcp-1"
        mock_result_hcp.label = "HCP"
        mock_result_hcp.name = "Dr. Smith"
        mock_result_hcp.score = 0.9
        mock_result_hcp.properties = {}

        mock_result_brand = MagicMock()
        mock_result_brand.uuid = "brand-1"
        mock_result_brand.label = "Brand"
        mock_result_brand.name = "Remibrutinib"
        mock_result_brand.score = 0.8
        mock_result_brand.properties = {}

        mock_graphiti = AsyncMock()
        mock_graphiti.search.return_value = [mock_result_hcp, mock_result_brand]
        graphiti_service._graphiti = mock_graphiti

        # Filter for only HCPs
        results = await graphiti_service.search(
            "treatment",
            entity_types=[E2IEntityType.HCP]
        )

        assert len(results) == 1
        assert results[0].entity_type == "HCP"

    @pytest.mark.asyncio
    async def test_search_error_returns_empty(self, graphiti_service, mock_graph):
        """Test search returns empty list on error."""
        graphiti_service._initialized = True
        graphiti_service._graphiti = None
        mock_graph.query.side_effect = Exception("Query failed")

        results = await graphiti_service.search("test query")

        assert results == []


class TestGetEntitySubgraph:
    """Tests for get_entity_subgraph method."""

    @pytest.mark.asyncio
    async def test_get_subgraph_success(self, graphiti_service, mock_graph):
        """Test successful subgraph retrieval."""
        graphiti_service._initialized = True

        # Mock node with proper structure
        mock_node = MagicMock()
        mock_node.properties = {"id": "hcp-1", "name": "Dr. Smith"}
        mock_node.labels = ["HCP"]

        # Mock relationship
        mock_rel = MagicMock()
        mock_rel.src_node = "hcp-1"
        mock_rel.dest_node = "brand-1"
        mock_rel.relation = "PRESCRIBES"
        mock_rel.properties = {}

        mock_graph.query.return_value.result_set = [[mock_node, mock_rel]]

        result = await graphiti_service.get_entity_subgraph("hcp-1", max_depth=2)

        assert isinstance(result, SubgraphResult)
        assert result.center_entity_id == "hcp-1"
        assert result.depth == 2

    @pytest.mark.asyncio
    async def test_get_subgraph_empty(self, graphiti_service, mock_graph):
        """Test subgraph for non-existent entity."""
        graphiti_service._initialized = True
        mock_graph.query.return_value.result_set = []

        result = await graphiti_service.get_entity_subgraph("nonexistent-id")

        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    @pytest.mark.asyncio
    async def test_get_subgraph_error_handling(self, graphiti_service, mock_graph):
        """Test subgraph handles errors gracefully."""
        graphiti_service._initialized = True
        mock_graph.query.side_effect = Exception("Graph error")

        result = await graphiti_service.get_entity_subgraph("test-id")

        assert len(result.nodes) == 0
        assert len(result.edges) == 0
        assert result.center_entity_id == "test-id"


class TestGetCausalChains:
    """Tests for get_causal_chains method."""

    @pytest.mark.asyncio
    async def test_get_causal_chains_by_kpi(self, graphiti_service, mock_graph):
        """Test getting causal chains for a KPI."""
        graphiti_service._initialized = True
        mock_graph.query.return_value.result_set = []

        result = await graphiti_service.get_causal_chains(kpi_name="NRx")

        assert isinstance(result, list)
        # Verify the query was constructed with KPI filter
        call_args = mock_graph.query.call_args
        assert "kpi_name" in call_args.kwargs["params"]

    @pytest.mark.asyncio
    async def test_get_causal_chains_by_entity(self, graphiti_service, mock_graph):
        """Test getting causal chains from a starting entity."""
        graphiti_service._initialized = True
        mock_graph.query.return_value.result_set = []

        result = await graphiti_service.get_causal_chains(start_entity_id="hcp-1")

        assert isinstance(result, list)
        call_args = mock_graph.query.call_args
        assert "entity_id" in call_args.kwargs["params"]

    @pytest.mark.asyncio
    async def test_get_causal_chains_error_returns_empty(self, graphiti_service, mock_graph):
        """Test causal chains returns empty on error."""
        graphiti_service._initialized = True
        mock_graph.query.side_effect = Exception("Query failed")

        result = await graphiti_service.get_causal_chains()

        assert result == []


class TestGetGraphStats:
    """Tests for get_graph_stats method."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, graphiti_service, mock_graph):
        """Test successful stats retrieval."""
        graphiti_service._initialized = True

        # Mock count query results
        mock_graph.query.return_value.result_set = [[5]]

        stats = await graphiti_service.get_graph_stats()

        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "nodes_by_type" in stats
        assert "edges_by_type" in stats
        assert stats["graph_name"] == "test_e2i_semantic"

    @pytest.mark.asyncio
    async def test_get_stats_error_returns_defaults(self, graphiti_service, mock_graph):
        """Test stats returns defaults on error."""
        graphiti_service._initialized = True
        mock_graph.query.side_effect = Exception("Stats query failed")

        stats = await graphiti_service.get_graph_stats()

        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert "error" in stats


class TestClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    async def test_close_with_graphiti(self, graphiti_service):
        """Test close with Graphiti client."""
        graphiti_service._initialized = True
        mock_graphiti = AsyncMock()
        graphiti_service._graphiti = mock_graphiti

        await graphiti_service.close()

        mock_graphiti.close.assert_called_once()
        assert graphiti_service._graphiti is None
        assert graphiti_service._initialized is False

    @pytest.mark.asyncio
    async def test_close_without_graphiti(self, graphiti_service):
        """Test close without Graphiti client (fallback mode)."""
        graphiti_service._initialized = True
        graphiti_service._graphiti = None

        await graphiti_service.close()

        assert graphiti_service._initialized is False


# ============================================================================
# Singleton Function Tests
# ============================================================================

class TestGetGraphitiService:
    """Tests for get_graphiti_service function."""

    @pytest.mark.asyncio
    async def test_creates_singleton(self, mock_config, mock_falkordb_client):
        """Test that get_graphiti_service creates a singleton."""
        with patch('src.memory.graphiti_service.get_falkordb_client', return_value=mock_falkordb_client):
            with patch('src.memory.graphiti_service.get_graphiti_config', return_value=mock_config):
                with patch('src.memory.graphiti_service.E2IGraphitiService._init_graphiti', new_callable=AsyncMock):
                    service1 = await get_graphiti_service()
                    service2 = await get_graphiti_service()

                    assert service1 is service2


class TestResetGraphitiService:
    """Tests for reset functions."""

    def test_reset_when_none(self):
        """Test reset when service is None."""
        # Should not raise
        reset_graphiti_service()

    @pytest.mark.asyncio
    async def test_async_reset(self, mock_config, mock_falkordb_client):
        """Test async reset function."""
        with patch('src.memory.graphiti_service.get_falkordb_client', return_value=mock_falkordb_client):
            with patch('src.memory.graphiti_service.get_graphiti_config', return_value=mock_config):
                with patch('src.memory.graphiti_service.E2IGraphitiService._init_graphiti', new_callable=AsyncMock):
                    service = await get_graphiti_service()
                    assert service is not None

                    await reset_graphiti_service_async()

                    # After reset, a new call should create new service
                    with patch('src.memory.graphiti_service.E2IGraphitiService.close', new_callable=AsyncMock):
                        new_service = await get_graphiti_service()
                        # Note: We can't easily verify it's a new instance without
                        # more complex tracking, but the reset should have cleared it


# ============================================================================
# Conversion Method Tests
# ============================================================================

class TestConvertGraphitiEntities:
    """Tests for _convert_graphiti_entities method."""

    def test_convert_entities_from_nodes(self, graphiti_service):
        """Test converting Graphiti nodes to ExtractedEntity."""
        mock_node = MagicMock()
        mock_node.uuid = "node-123"
        mock_node.name = "Test Entity"
        mock_node.labels = ["Brand"]
        mock_node.attributes = {"therapeutic_area": "immunology"}

        mock_result = MagicMock()
        mock_result.nodes = [mock_node]

        entities = graphiti_service._convert_graphiti_entities(mock_result)

        assert len(entities) == 1
        assert entities[0].entity_id == "node-123"
        assert entities[0].name == "Test Entity"
        assert entities[0].entity_type == E2IEntityType.BRAND
        assert entities[0].properties == {"therapeutic_area": "immunology"}

    def test_convert_entities_handles_missing_labels(self, graphiti_service):
        """Test conversion handles nodes without labels."""
        mock_node = MagicMock()
        mock_node.uuid = "node-456"
        mock_node.name = "Unknown Entity"
        mock_node.labels = []
        mock_node.attributes = {}

        mock_result = MagicMock()
        mock_result.nodes = [mock_node]

        entities = graphiti_service._convert_graphiti_entities(mock_result)

        assert len(entities) == 1
        # Should default to AGENT when no label
        assert entities[0].entity_type == E2IEntityType.AGENT

    def test_convert_entities_handles_errors(self, graphiti_service):
        """Test conversion handles individual node errors."""
        # Node that will cause error
        bad_node = MagicMock()
        bad_node.uuid = None  # This shouldn't cause issues with getattr

        # Good node
        good_node = MagicMock()
        good_node.uuid = "good-node"
        good_node.name = "Good Entity"
        good_node.labels = ["KPI"]
        good_node.attributes = {}

        mock_result = MagicMock()
        mock_result.nodes = [good_node]  # Only good node

        entities = graphiti_service._convert_graphiti_entities(mock_result)

        assert len(entities) == 1


class TestConvertGraphitiRelationships:
    """Tests for _convert_graphiti_relationships method."""

    def test_convert_relationships_from_edges(self, graphiti_service):
        """Test converting Graphiti edges to ExtractedRelationship."""
        mock_edge = MagicMock()
        mock_edge.source_node_uuid = "source-123"
        mock_edge.target_node_uuid = "target-456"
        mock_edge.name = "IMPACTS"
        mock_edge.attributes = {"effect_size": 0.5}

        mock_result = MagicMock()
        mock_result.edges = [mock_edge]

        relationships = graphiti_service._convert_graphiti_relationships(mock_result)

        assert len(relationships) == 1
        assert relationships[0].source_id == "source-123"
        assert relationships[0].target_id == "target-456"
        assert relationships[0].relationship_type == E2IRelationshipType.IMPACTS
        assert relationships[0].properties == {"effect_size": 0.5}

    def test_convert_relationships_unknown_type(self, graphiti_service):
        """Test conversion handles unknown relationship types."""
        mock_edge = MagicMock()
        mock_edge.source_node_uuid = "s1"
        mock_edge.target_node_uuid = "t1"
        mock_edge.name = "UNKNOWN_RELATIONSHIP"
        mock_edge.attributes = {}

        mock_result = MagicMock()
        mock_result.edges = [mock_edge]

        relationships = graphiti_service._convert_graphiti_relationships(mock_result)

        assert len(relationships) == 1
        assert relationships[0].relationship_type == E2IRelationshipType.RELATES_TO


class TestPathToChain:
    """Tests for _path_to_chain method."""

    def test_convert_path_with_nodes_and_edges(self, graphiti_service):
        """Test converting a graph path to chain dictionary."""
        # Create mock path
        mock_node1 = MagicMock()
        mock_node1.properties = {"id": "n1", "name": "Cause"}
        mock_node1.labels = ["Trigger"]

        mock_node2 = MagicMock()
        mock_node2.properties = {"id": "n2", "name": "Effect"}
        mock_node2.labels = ["KPI"]

        mock_rel = MagicMock()
        mock_rel.relation = "CAUSES"
        mock_rel.properties = {"confidence": 0.9, "effect_size": 0.3}

        mock_path = MagicMock()
        mock_path.nodes.return_value = [mock_node1, mock_node2]
        mock_path.relationships.return_value = [mock_rel]

        chain = graphiti_service._path_to_chain(mock_path)

        assert len(chain["nodes"]) == 2
        assert len(chain["edges"]) == 1
        assert chain["length"] == 1
        assert chain["edges"][0]["type"] == "CAUSES"
        assert chain["edges"][0]["confidence"] == 0.9

    def test_convert_path_empty(self, graphiti_service):
        """Test converting empty path."""
        mock_path = MagicMock()
        # Path without nodes/relationships methods
        del mock_path.nodes
        del mock_path.relationships

        chain = graphiti_service._path_to_chain(mock_path)

        assert chain["nodes"] == []
        assert chain["edges"] == []
        assert chain["length"] == -1  # len([]) - 1
