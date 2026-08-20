"""Tests for Graph API routes.

Version: 1.0.0
Tests the knowledge graph API endpoints for nodes, relationships, traversal,
causal chains, Cypher queries, and WebSocket streaming.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from src.api.routes.graph import router


@pytest.fixture
def app():
    """Create FastAPI app with graph router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_graphiti_service():
    """Create mock Graphiti service."""
    mock = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.add_episode = AsyncMock(return_value={"episode_id": str(uuid4())})
    mock.get_graph_stats = AsyncMock(
        return_value={
            "total_nodes": 100,
            "total_edges": 250,
            "nodes_by_type": {"HCP": 50, "Brand": 30, "Patient": 20},
            "edges_by_type": {"CAUSES": 100, "IMPACTS": 150},
            "total_episodes": 10,
            "total_communities": 5,
        }
    )
    return mock


@pytest.fixture
def mock_semantic_memory():
    """Create mock semantic memory."""
    mock = MagicMock()
    mock.list_nodes = MagicMock(
        return_value=[
            {"id": "node1", "name": "Test Node 1", "type": "HCP", "properties": {}},
            {"id": "node2", "name": "Test Node 2", "type": "Brand", "properties": {}},
        ]
    )
    mock.count_nodes = MagicMock(return_value=2)  # Required for list_nodes pagination
    mock.get_node = MagicMock(
        return_value={
            "id": "node1",
            "name": "Test Node",
            "type": "HCP",
            "properties": {"specialty": "Oncology"},
        }
    )
    mock.get_node_network = MagicMock(
        return_value={
            "node": {"id": "node1", "name": "Test Node", "type": "HCP"},
            "neighbors": [
                {"id": "node2", "name": "Related Node", "type": "Brand"},
            ],
            "relationships": [
                {"source": "node1", "target": "node2", "type": "PRESCRIBES"},
            ],
        }
    )
    mock.get_patient_network = MagicMock(
        return_value={
            "hcps": [{"id": "hcp1", "name": "Dr. Smith", "type": "HCP"}],
            "treatments": [],
            "events": [],
        }
    )
    mock.get_hcp_network = MagicMock(
        return_value={
            "patients": [{"id": "patient1", "name": "Patient 1", "type": "Patient"}],
            "brands": [],
            "territories": [],
        }
    )
    mock.get_hcp_influence_network = MagicMock(
        return_value={
            "influenced_hcps": [],
            "patients": [{"id": "patient1", "name": "Patient 1", "type": "Patient"}],
            "brands_prescribed": [],
        }
    )
    mock.list_relationships = MagicMock(
        return_value=[
            {
                "id": "rel1",
                "source_id": "node1",
                "target_id": "node2",
                "type": "CAUSES",
                "properties": {},
            },
        ]
    )
    mock.count_relationships = MagicMock(return_value=1)  # Required for relationship pagination
    mock.traverse_from_node = MagicMock(
        return_value={
            "path": ["node1", "node2", "node3"],
            "nodes": [
                {"id": "node1", "name": "Start", "type": "HCP"},
                {"id": "node2", "name": "Middle", "type": "Brand"},
                {"id": "node3", "name": "End", "type": "Patient"},
            ],
            "relationships": [],
        }
    )
    mock.find_causal_chains = MagicMock(
        return_value=[
            {
                "path": [
                    {"id": "cause", "name": "Cause", "type": "HCP"},
                    {"id": "effect", "name": "Effect", "type": "Brand"},
                ],
                "relationships": [
                    {"type": "CAUSES", "strength": 0.85, "confidence": 0.9},
                ],
                "total_strength": 0.85,
            }
        ]
    )
    mock.search_nodes = MagicMock(
        return_value=[
            {"id": "result1", "name": "Search Result", "type": "HCP", "score": 0.95},
        ]
    )
    mock.semantic_search = MagicMock(
        return_value=[
            {"id": "result1", "name": "Search Result", "type": "HCP", "score": 0.95},
        ]
    )
    # The real FalkorDBSemanticMemory exposes get_graph_stats() (not get_stats);
    # the /graph/stats route is wired to that name.
    mock.get_graph_stats = MagicMock(
        return_value={
            "total_nodes": 100,
            "total_relationships": 250,
            "nodes_by_type": {"HCP": 50, "Brand": 30},
            "relationships_by_type": {"CAUSES": 100, "IMPACTS": 150},
        }
    )
    mock.health_check = MagicMock(return_value={"status": "healthy", "connected": True})
    return mock


# =============================================================================
# List Nodes Tests
# =============================================================================


class TestListNodes:
    """Test GET /graph/nodes endpoint."""

    def test_list_nodes_success(self, client, mock_semantic_memory):
        """Test successful node listing."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "nodes" in data
            assert isinstance(data["nodes"], list)

    def test_list_nodes_with_type_filter(self, client, mock_semantic_memory):
        """Test node listing with entity type filter."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes?entity_type=HCP")

            assert response.status_code == status.HTTP_200_OK

    def test_list_nodes_with_pagination(self, client, mock_semantic_memory):
        """Test node listing with pagination parameters."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes?limit=10&offset=0")

            assert response.status_code == status.HTTP_200_OK

    def test_list_nodes_service_unavailable(self, client):
        """Test node listing when service is unavailable."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.get("/graph/nodes")

            # Should return error or empty list depending on implementation
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_503_SERVICE_UNAVAILABLE,
            ]

    def test_list_nodes_curated_only_passed_through(self, client, mock_semantic_memory):
        """?curated_only=true must reach list_nodes AND count_nodes so the
        gold-standard view excludes agent-written runtime nodes."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes?entity_types=Variable&curated_only=true")

            assert response.status_code == status.HTTP_200_OK
            assert mock_semantic_memory.list_nodes.call_args.kwargs["curated_only"] is True
            assert mock_semantic_memory.count_nodes.call_args.kwargs["curated_only"] is True

    def test_list_nodes_curated_only_defaults_false(self, client, mock_semantic_memory):
        """Omitting curated_only must default to False (unchanged behaviour)."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes?entity_types=Variable")

            assert response.status_code == status.HTTP_200_OK
            assert mock_semantic_memory.list_nodes.call_args.kwargs["curated_only"] is False


# =============================================================================
# Get Node Tests
# =============================================================================


class TestGetNode:
    """Test GET /graph/nodes/{node_id} endpoint."""

    def test_get_node_success(self, client, mock_semantic_memory):
        """Test successful node retrieval."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes/node1")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "id" in data or "node" in data

    def test_get_node_not_found(self, client, mock_semantic_memory):
        """Test node retrieval for non-existent node."""
        mock_semantic_memory.get_node = MagicMock(return_value=None)

        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes/nonexistent")

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_node_service_error(self, client, mock_semantic_memory):
        """Test node retrieval with service error."""
        mock_semantic_memory.get_node = MagicMock(side_effect=Exception("Database error"))

        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes/node1")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# =============================================================================
# Get Node Network Tests
# =============================================================================


class TestGetNodeNetwork:
    """Test GET /graph/nodes/{node_id}/network endpoint."""

    def test_get_node_network_success(self, client, mock_semantic_memory):
        """Test successful node network retrieval."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes/node1/network")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            # Response has node_id, node_type, connected_nodes format
            assert "node_id" in data or "connected_nodes" in data

    def test_get_node_network_with_depth(self, client, mock_semantic_memory):
        """Test node network with depth parameter."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes/node1/network?depth=2")

            assert response.status_code == status.HTTP_200_OK

    def test_get_node_network_not_found(self, client, mock_semantic_memory):
        """Test node network for non-existent node."""
        mock_semantic_memory.get_node_network = MagicMock(return_value=None)

        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes/nonexistent/network")

            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_200_OK,  # May return empty network
            ]


# =============================================================================
# List Relationships Tests
# =============================================================================


class TestListRelationships:
    """Test GET /graph/relationships endpoint."""

    def test_list_relationships_success(self, client, mock_semantic_memory):
        """Test successful relationship listing."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/relationships")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "relationships" in data
            assert isinstance(data["relationships"], list)

    def test_list_relationships_with_type_filter(self, client, mock_semantic_memory):
        """Test relationship listing with type filter."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/relationships?relationship_type=CAUSES")

            assert response.status_code == status.HTTP_200_OK

    def test_list_relationships_with_pagination(self, client, mock_semantic_memory):
        """Test relationship listing with pagination."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/relationships?limit=10&offset=0")

            assert response.status_code == status.HTTP_200_OK

    def test_list_relationships_curated_only_passed_through(self, client, mock_semantic_memory):
        """?curated_only=true must reach list_relationships so the gold-standard
        view excludes agent-written runtime edges."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get(
                "/graph/relationships?relationship_types=CAUSES&curated_only=true"
            )

            assert response.status_code == status.HTTP_200_OK
            assert mock_semantic_memory.list_relationships.call_args.kwargs["curated_only"] is True

    def test_list_relationships_curated_only_defaults_false(self, client, mock_semantic_memory):
        """Omitting curated_only must default to False (unchanged behaviour)."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/relationships?relationship_types=CAUSES")

            assert response.status_code == status.HTTP_200_OK
            assert mock_semantic_memory.list_relationships.call_args.kwargs["curated_only"] is False


# =============================================================================
# Traverse Graph Tests
# =============================================================================


class TestTraverseGraph:
    """Test POST /graph/traverse endpoint."""

    def test_traverse_graph_success(self, client, mock_semantic_memory):
        """Test successful graph traversal."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.post(
                "/graph/traverse",
                json={
                    "start_node_id": "node1",
                    "max_depth": 3,
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "path" in data or "nodes" in data

    def test_traverse_graph_with_filters(self, client, mock_semantic_memory):
        """Test graph traversal with relationship type filters."""
        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post(
                "/graph/traverse",
                json={
                    "start_node_id": "node1",
                    "max_depth": 2,
                    "relationship_types": ["CAUSES", "IMPACTS"],
                },
            )

            assert response.status_code == status.HTTP_200_OK

    def test_traverse_graph_with_direction(self, client, mock_semantic_memory):
        """Test graph traversal with direction parameter."""
        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post(
                "/graph/traverse",
                json={
                    "start_node_id": "node1",
                    "max_depth": 2,
                    "direction": "outbound",
                },
            )

            assert response.status_code == status.HTTP_200_OK

    def test_traverse_graph_invalid_node(self, client, mock_semantic_memory):
        """Test graph traversal with non-existent start node."""
        # Return empty structure instead of None to avoid NoneType error
        mock_semantic_memory.traverse_from_node = MagicMock(
            return_value={
                "nodes": [],
                "relationships": [],
                "path": [],
            }
        )

        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post(
                "/graph/traverse",
                json={
                    "start_node_id": "nonexistent",
                    "max_depth": 2,
                },
            )

            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_200_OK,  # May return empty result
            ]

    def test_traverse_graph_missing_start_node(self, client):
        """Test graph traversal without start node."""
        response = client.post(
            "/graph/traverse",
            json={
                "max_depth": 2,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# Query Causal Chains Tests
# =============================================================================


class TestQueryCausalChains:
    """Test POST /graph/causal-chains endpoint."""

    def test_query_causal_chains_success(self, client, mock_semantic_memory):
        """Test successful causal chain query."""
        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post(
                "/graph/causal-chains",
                json={
                    "source_entity_id": "cause_node",
                    "target_entity_id": "effect_node",
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "chains" in data

    def test_query_causal_chains_with_max_length(self, client, mock_semantic_memory):
        """Test causal chain query with max path length."""
        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post(
                "/graph/causal-chains",
                json={
                    "source_entity_id": "cause_node",
                    "target_entity_id": "effect_node",
                    "max_chain_length": 5,
                },
            )

            assert response.status_code == status.HTTP_200_OK

    def test_query_causal_chains_with_min_confidence(self, client, mock_semantic_memory):
        """Test causal chain query with minimum confidence threshold."""
        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post(
                "/graph/causal-chains",
                json={
                    "source_entity_id": "cause_node",
                    "target_entity_id": "effect_node",
                    "min_confidence": 0.7,
                },
            )

            assert response.status_code == status.HTTP_200_OK

    def test_query_causal_chains_no_path_found(self, client, mock_semantic_memory):
        """Test causal chain query when no path exists."""
        mock_semantic_memory.find_causal_chains = MagicMock(return_value=[])

        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post(
                "/graph/causal-chains",
                json={
                    "source_entity_id": "isolated_node1",
                    "target_entity_id": "isolated_node2",
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data.get("chains") == [] or len(data.get("chains", [])) == 0


# =============================================================================
# Execute Cypher Query Tests
# =============================================================================


class TestExecuteCypherQuery:
    """Test POST /graph/query endpoint — raw Cypher passthrough is DISABLED.

    The endpoint used to call an arbitrary-Cypher executor against a PHI graph
    behind a trivially-bypassable keyword "read-only" filter. It now refuses
    all requests with 501 rather than executing attacker-controlled Cypher.
    """

    def test_read_query_is_refused_not_executed(self, client, mock_semantic_memory):
        """Even a benign read query must be refused (501), never executed."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.post(
                "/graph/query",
                json={"query": "MATCH (n) RETURN n LIMIT 10"},
            )

            assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
            # The semantic memory must NOT have been asked to run anything.
            assert not mock_semantic_memory.method_calls

    def test_write_query_is_refused(self, client, mock_semantic_memory):
        """A write query must be refused (501) — arbitrary execution disabled."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.post(
                "/graph/query",
                json={"query": "MATCH (n) DETACH DELETE n"},
            )

            assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED

    def test_injection_style_query_is_refused(self, client, mock_semantic_memory):
        """An injection / read-only-bypass query is refused without execution."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            # A query that bypasses the old substring "CREATE/DELETE" filter
            # (no write keyword, but exfiltrates all PHI).
            response = client.post(
                "/graph/query",
                json={"query": "MATCH (p:Patient) RETURN p", "read_only": True},
            )

            assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
            assert not mock_semantic_memory.method_calls

    def test_refusal_detail_does_not_leak_internals(self, client, mock_semantic_memory):
        """The 501 detail should be a generic, actionable message (no stack)."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.post(
                "/graph/query",
                json={"query": "MATCH (n) RETURN n"},
            )

            assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
            detail = response.json().get("detail", "")
            assert "disabled" in detail.lower()
            assert "Traceback" not in detail

    def test_execute_cypher_query_empty_query(self, client):
        """Empty query string is still rejected by request validation (422)."""
        response = client.post(
            "/graph/query",
            json={"query": ""},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# Add Episode Tests
# =============================================================================


class TestAddEpisode:
    """Test POST /graph/episodes endpoint."""

    def test_add_episode_success(self, client, mock_graphiti_service):
        """Test successful episode addition."""
        # Mock the result object from graphiti
        mock_result = MagicMock()
        mock_result.episode_id = str(uuid4())
        mock_result.entities_extracted = []
        mock_result.relationships_extracted = []
        mock_graphiti_service.add_episode = AsyncMock(return_value=mock_result)

        with patch(
            "src.api.routes.graph._get_graphiti_service",
            new_callable=AsyncMock,
            return_value=mock_graphiti_service,
        ):
            response = client.post(
                "/graph/episodes",
                json={
                    "content": "This is a test episode content.",
                    "source": "test_source",
                },
            )

            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_201_CREATED,
            ]

    def test_add_episode_with_session_id(self, client, mock_graphiti_service):
        """Test episode addition with session ID."""
        # Mock the result object from graphiti
        mock_result = MagicMock()
        mock_result.episode_id = str(uuid4())
        mock_result.entities_extracted = []
        mock_result.relationships_extracted = []
        mock_graphiti_service.add_episode = AsyncMock(return_value=mock_result)

        with patch(
            "src.api.routes.graph._get_graphiti_service",
            new_callable=AsyncMock,
            return_value=mock_graphiti_service,
        ):
            response = client.post(
                "/graph/episodes",
                json={
                    "content": "Episode with session context.",
                    "source": "test_source",
                    "session_id": "sess_abc123",
                },
            )

            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_201_CREATED,
            ]

    def test_add_episode_missing_content(self, client):
        """Test episode addition without content."""
        response = client.post(
            "/graph/episodes",
            json={
                "source": "test_source",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_add_episode_missing_source(self, client):
        """Test episode addition without source."""
        response = client.post(
            "/graph/episodes",
            json={
                "content": "Test content without source",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_add_episode_service_unavailable(self, client):
        """Test episode addition when Graphiti service unavailable."""
        with patch(
            "src.api.routes.graph._get_graphiti_service",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.post(
                "/graph/episodes",
                json={
                    "content": "Test content",
                    "source": "test_source",
                },
            )

            assert response.status_code in [
                status.HTTP_503_SERVICE_UNAVAILABLE,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ]


# =============================================================================
# Search Graph Tests
# =============================================================================


class TestSearchGraph:
    """Test POST /graph/search endpoint."""

    def test_search_graph_success(self, client, mock_semantic_memory):
        """Test successful graph search."""
        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post(
                "/graph/search",
                json={
                    "query": "Find HCP nodes related to oncology",
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "results" in data

    def test_search_graph_with_entity_types(self, client, mock_semantic_memory):
        """Test graph search with entity type filter."""
        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post(
                "/graph/search",
                json={
                    "query": "oncology specialists",
                    "entity_types": ["HCP"],
                },
            )

            assert response.status_code == status.HTTP_200_OK

    def test_search_graph_with_limit(self, client, mock_semantic_memory):
        """Test graph search with result limit."""
        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post(
                "/graph/search",
                json={
                    "query": "test query",
                    "limit": 5,
                },
            )

            assert response.status_code == status.HTTP_200_OK

    def test_search_graph_empty_query(self, client):
        """Test graph search with empty query."""
        response = client.post(
            "/graph/search",
            json={
                "query": "",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_graph_no_results(self, client, mock_semantic_memory):
        """Test graph search with no matching results."""
        mock_semantic_memory.semantic_search = MagicMock(return_value=[])

        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post(
                "/graph/search",
                json={
                    "query": "nonexistent entity xyz123",
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data.get("results") == [] or len(data.get("results", [])) == 0


# =============================================================================
# Graph Stats Tests
# =============================================================================


class TestGetGraphStats:
    """Test GET /graph/stats endpoint."""

    def test_get_graph_stats_success(self, client, mock_semantic_memory):
        """Test successful graph stats retrieval."""
        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.get("/graph/stats")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            # Response format uses total_nodes, total_relationships
            assert "total_nodes" in data or "stats" in data

    def test_get_graph_stats_with_graphiti(self, client, mock_graphiti_service):
        """Test graph stats with Graphiti service."""
        with patch(
            "src.api.routes.graph._get_graphiti_service",
            new_callable=AsyncMock,
            return_value=mock_graphiti_service,
        ):
            response = client.get("/graph/stats")

            assert response.status_code == status.HTTP_200_OK

    def test_get_graph_stats_service_unavailable(self, client):
        """Test graph stats when services unavailable."""
        with (
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            response = client.get("/graph/stats")

            assert response.status_code in [
                status.HTTP_200_OK,  # May return empty stats
                status.HTTP_503_SERVICE_UNAVAILABLE,
            ]


# =============================================================================
# Graph Health Tests
# =============================================================================


class TestGraphHealth:
    """Test GET /graph/health endpoint."""

    def test_graph_health_success(self, client, mock_semantic_memory):
        """Test successful health check."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/health")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "status" in data

    def test_graph_health_service_down(self, client):
        """Test health check when services are down."""
        # Mock both services to return None to simulate services unavailable
        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            response = client.get("/graph/health")

            # Health endpoint returns 200 with degraded status when services unavailable
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data.get("status") == "degraded"
            assert data.get("graphiti") == "unavailable"
            assert data.get("falkordb") == "unavailable"

    def test_graph_health_degrades_when_graph_empty_1760(self, client, mock_semantic_memory):
        """Connected-but-EMPTY graph must degrade, not report healthy (#1760).

        The #1758 incident signature: FalkorDB reachable, every curated node
        wiped — and this endpoint stayed green for four days because it only
        checked connectivity.
        """
        with (
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
            patch(
                "src.api.dependencies.falkordb_client.falkordb_diagnostics",
                new_callable=AsyncMock,
                return_value={
                    "status": "healthy",
                    "current_graph": "e2i_causal",
                    "node_count": 0,
                    "edge_count": 0,
                    "cached": False,
                },
            ),
        ):
            response = client.get("/graph/health")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data.get("status") == "degraded"
            content = data.get("graph_content")
            assert content is not None, "health payload must carry graph_content (#1760)"
            assert content.get("empty") is True
            assert content.get("node_count") == 0

    def test_graph_health_healthy_with_content_1760(self, client, mock_semantic_memory):
        """A populated graph stays healthy and surfaces its counts (#1760)."""
        with (
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
            patch(
                "src.api.dependencies.falkordb_client.falkordb_diagnostics",
                new_callable=AsyncMock,
                return_value={
                    "status": "healthy",
                    "current_graph": "e2i_causal",
                    "node_count": 85,
                    "edge_count": 233,
                    "cached": True,
                },
            ),
        ):
            response = client.get("/graph/health")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data.get("status") == "healthy"
            content = data.get("graph_content")
            assert content is not None
            assert content.get("empty") is False
            assert content.get("node_count") == 85
            assert content.get("edge_count") == 233

    def test_graph_health_scan_failure_is_unknown_not_empty_1760(
        self, client, mock_semantic_memory
    ):
        """A failed count scan must read as UNKNOWN, never as empty (#1760).

        A transient query failure yielding node_count=0 would be a
        plausible-wrong value: it would flip status to degraded and mimic the
        wipe signature. Unknown content must leave the connectivity verdict
        alone.
        """
        with (
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
            patch(
                "src.api.dependencies.falkordb_client.falkordb_diagnostics",
                new_callable=AsyncMock,
                return_value={"status": "unknown", "error": "scan failed"},
            ),
        ):
            response = client.get("/graph/health")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data.get("status") == "healthy"
            content = data.get("graph_content")
            assert content is not None
            assert content.get("status") == "unknown"
            assert content.get("empty") is not True


# =============================================================================
# WebSocket Stream Tests
# =============================================================================


class TestGraphStream:
    """Test WebSocket /graph/stream endpoint."""

    def test_websocket_connect(self, client):
        """Test WebSocket connection establishment."""
        # Note: TestClient has limited WebSocket support
        # This tests basic connection capability
        try:
            with client.websocket_connect("/graph/stream") as websocket:
                # Send a subscription message
                websocket.send_json(
                    {
                        "action": "subscribe",
                        "node_ids": ["node1", "node2"],
                    }
                )
                # Connection successful if we get here
                assert True
        except Exception:
            # WebSocket may not be fully supported in test mode
            pytest.skip("WebSocket testing not supported in test environment")

    def test_websocket_subscription(self, client):
        """Test WebSocket node subscription."""
        try:
            with client.websocket_connect("/graph/stream") as websocket:
                websocket.send_json(
                    {
                        "action": "subscribe",
                        "node_ids": ["node1"],
                    }
                )
                # Expect acknowledgment
                data = websocket.receive_json()
                assert "subscribed" in data or "action" in data or "status" in data
        except Exception:
            pytest.skip("WebSocket testing not supported in test environment")

    def test_websocket_unsubscribe(self, client):
        """Test WebSocket node unsubscription."""
        try:
            with client.websocket_connect("/graph/stream") as websocket:
                # Subscribe first
                websocket.send_json(
                    {
                        "action": "subscribe",
                        "node_ids": ["node1"],
                    }
                )
                websocket.receive_json()

                # Then unsubscribe
                websocket.send_json(
                    {
                        "action": "unsubscribe",
                        "node_ids": ["node1"],
                    }
                )
                data = websocket.receive_json()
                assert "unsubscribed" in data or "action" in data or "status" in data
        except Exception:
            pytest.skip("WebSocket testing not supported in test environment")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestGraphErrorHandling:
    """Test error handling across graph endpoints."""

    def test_invalid_entity_type(self, client, mock_semantic_memory):
        """Test handling of invalid entity type."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes?entity_type=INVALID_TYPE")

            # Should either reject or ignore invalid type
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ]

    def test_invalid_relationship_type(self, client, mock_semantic_memory):
        """Test handling of invalid relationship type."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/relationships?relationship_type=INVALID")

            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ]

    def test_database_connection_error(self, client, mock_semantic_memory):
        """Test handling of database connection errors."""
        mock_semantic_memory.list_nodes = MagicMock(
            side_effect=Exception("Database connection failed")
        )

        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_timeout_handling(self, client, mock_semantic_memory):
        """Test handling of operation timeouts."""
        import asyncio

        mock_semantic_memory.traverse_from_node = MagicMock(
            side_effect=asyncio.TimeoutError("Operation timed out")
        )

        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post(
                "/graph/traverse",
                json={
                    "start_node_id": "node1",
                    "max_depth": 5,  # Use valid depth (1-5)
                },
            )

            assert response.status_code in [
                status.HTTP_408_REQUEST_TIMEOUT,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                status.HTTP_504_GATEWAY_TIMEOUT,
            ]


# =============================================================================
# Integration Tests
# =============================================================================


class TestGraphIntegration:
    """Integration tests for graph API workflows."""

    def test_node_to_network_workflow(self, client, mock_semantic_memory):
        """Test workflow: list nodes -> get node -> get network."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            # Step 1: List nodes
            list_response = client.get("/graph/nodes")
            assert list_response.status_code == status.HTTP_200_OK

            # Step 2: Get specific node
            node_response = client.get("/graph/nodes/node1")
            assert node_response.status_code == status.HTTP_200_OK

            # Step 3: Get node network
            network_response = client.get("/graph/nodes/node1/network")
            assert network_response.status_code == status.HTTP_200_OK

    def test_search_to_traverse_workflow(self, client, mock_semantic_memory):
        """Test workflow: search -> traverse from result."""
        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            # Step 1: Search for nodes
            search_response = client.post(
                "/graph/search",
                json={"query": "oncology"},
            )
            assert search_response.status_code == status.HTTP_200_OK

            # Step 2: Traverse from search result
            traverse_response = client.post(
                "/graph/traverse",
                json={
                    "start_node_id": "result1",
                    "max_depth": 2,
                },
            )
            assert traverse_response.status_code == status.HTTP_200_OK

    def test_causal_chain_discovery_workflow(self, client, mock_semantic_memory):
        """Test causal chain discovery workflow."""
        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            # Step 1: Find potential cause node
            cause_search = client.post(
                "/graph/search",
                json={"query": "marketing campaign"},
            )
            assert cause_search.status_code == status.HTTP_200_OK

            # Step 2: Find potential effect node
            effect_search = client.post(
                "/graph/search",
                json={"query": "prescription volume"},
            )
            assert effect_search.status_code == status.HTTP_200_OK

            # Step 3: Query causal chains
            chains_response = client.post(
                "/graph/causal-chains",
                json={
                    "source_entity_id": "cause_node",
                    "target_entity_id": "effect_node",
                },
            )
            assert chains_response.status_code == status.HTTP_200_OK


# =============================================================================
# Pagination Tests
# =============================================================================


class TestGraphPagination:
    """Test pagination across graph endpoints."""

    def test_nodes_pagination_params(self, client, mock_semantic_memory):
        """Test node listing pagination parameters."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            # First page
            page1 = client.get("/graph/nodes?limit=10&offset=0")
            assert page1.status_code == status.HTTP_200_OK

            # Second page
            page2 = client.get("/graph/nodes?limit=10&offset=10")
            assert page2.status_code == status.HTTP_200_OK

    def test_relationships_pagination_params(self, client, mock_semantic_memory):
        """Test relationship listing pagination parameters."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/relationships?limit=20&offset=0")
            assert response.status_code == status.HTTP_200_OK

    def test_invalid_pagination_params(self, client, mock_semantic_memory):
        """Test handling of invalid pagination parameters."""
        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            # Negative offset
            response = client.get("/graph/nodes?offset=-1")
            assert response.status_code in [
                status.HTTP_200_OK,  # May ignore invalid
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ]

            # Zero limit
            response = client.get("/graph/nodes?limit=0")
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ]


# =============================================================================
# Error-message hygiene (Finding 3 — no internal/stack disclosure)
# =============================================================================


class TestErrorMessageHygiene:
    """500 responses must return generic messages, not internal exception text."""

    def test_get_node_500_does_not_leak_exception_text(self, client, mock_semantic_memory):
        """An internal error must not echo the exception string into the body."""
        secret = "Database password=hunter2 at 10.0.0.5"
        mock_semantic_memory.get_node = MagicMock(side_effect=Exception(secret))

        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes/node1")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            body = response.text
            assert secret not in body
            assert "hunter2" not in body

    def test_list_nodes_500_does_not_leak_exception_text(self, client, mock_semantic_memory):
        """list_nodes internal error must return a generic message."""
        secret = "internal-stack-detail-xyz"
        mock_semantic_memory.list_nodes = MagicMock(side_effect=Exception(secret))

        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=mock_semantic_memory,
        ):
            response = client.get("/graph/nodes")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert secret not in response.text

    def test_search_500_does_not_leak_exception_text(self, client, mock_semantic_memory):
        """search internal error must return a generic message."""
        secret = "redis-connection-string-secret"
        mock_semantic_memory.semantic_search = MagicMock(side_effect=Exception(secret))

        with (
            patch(
                "src.api.routes.graph._get_graphiti_service",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "src.api.routes.graph._get_semantic_memory",
                new_callable=AsyncMock,
                return_value=mock_semantic_memory,
            ),
        ):
            response = client.post("/graph/search", json={"query": "anything"})

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert secret not in response.text


# =============================================================================
# Injection rejection at the route boundary (Findings 1 + 2 integration)
# =============================================================================


class TestRouteLevelInjectionRejection:
    """A real (non-mock) semantic memory rejects injected label/type filters."""

    def test_list_nodes_injection_label_rejected_with_generic_500(self, client):
        """An injection-style entity_types value reaches the real validator and
        is rejected (ValueError -> generic 500), never interpolated into Cypher."""
        from src.memory.semantic_memory import FalkorDBSemanticMemory

        real = FalkorDBSemanticMemory()
        # Stub the graph driver so any *executed* query would be observable;
        # the injection must be rejected BEFORE the driver is touched.
        real._graph = MagicMock()
        real._graph.query.return_value = MagicMock(result_set=[])

        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=real,
        ):
            response = client.get("/graph/nodes?entity_types=Patient' OR '1'='1")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            # Poisoned label never spliced into a query sent to the driver.
            for call in real._graph.query.call_args_list:
                assert "OR '1'='1" not in call[0][0]

    def test_list_nodes_known_label_passes_through(self, client):
        """A known label is accepted and parameterized (no interpolation)."""
        from src.memory.semantic_memory import FalkorDBSemanticMemory

        real = FalkorDBSemanticMemory()
        real._graph = MagicMock()
        real._graph.query.return_value = MagicMock(result_set=[])

        with patch(
            "src.api.routes.graph._get_semantic_memory",
            new_callable=AsyncMock,
            return_value=real,
        ):
            response = client.get("/graph/nodes?entity_types=Patient")

            assert response.status_code == status.HTTP_200_OK
            # The label reached the driver as a parameter, not a literal.
            list_call = real._graph.query.call_args_list[0]
            assert "'Patient' IN labels(n)" not in list_call[0][0]
            assert list_call[0][1].get("entity_types") == ["Patient"]
