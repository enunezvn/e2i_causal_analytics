"""
Tests for Graph API endpoints.

Tests the knowledge graph endpoints:
- GET /graph/stats
- GET /graph/nodes
- GET /graph/nodes/{node_id}
- GET /graph/nodes/{node_id}/network
- GET /graph/relationships
- POST /graph/search
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_semantic_memory():
    """Mock FalkorDB semantic memory."""
    memory = MagicMock()

    # Mock list_nodes
    memory.list_nodes = MagicMock(return_value=[
        {
            "id": "patient_001",
            "type": "Patient",
            "name": "Patient 001",
            "journey_stage": "adoption",
            "region": "Northeast"
        },
        {
            "id": "patient_002",
            "type": "Patient",
            "name": "Patient 002",
            "journey_stage": "awareness",
            "region": "Midwest"
        }
    ])

    # Mock count_nodes
    memory.count_nodes = MagicMock(return_value=5)

    # Mock get_node
    memory.get_node = MagicMock(return_value={
        "id": "patient_001",
        "type": "Patient",
        "name": "Patient 001",
        "journey_stage": "adoption",
        "region": "Northeast"
    })

    # Mock get_patient_network
    memory.get_patient_network = MagicMock(return_value={
        "patient_id": "patient_001",
        "hcps": [
            {"id": "hcp_001", "properties": {"name": "Dr. Sarah Chen", "specialty": "Oncology"}}
        ],
        "treatments": [
            {"id": "brand_kisqali", "properties": {"name": "Kisqali"}}
        ],
        "triggers": [
            {"id": "trigger_001", "properties": {"type": "peer_influence"}}
        ],
        "causal_paths": [],
        "brands": []
    })

    # Mock get_hcp_influence_network
    memory.get_hcp_influence_network = MagicMock(return_value={
        "hcp_id": "hcp_001",
        "influenced_hcps": [
            {"id": "hcp_002", "properties": {"name": "Dr. James Wilson"}}
        ],
        "patients": [
            {"id": "patient_001", "properties": {"journey_stage": "adoption"}}
        ],
        "brands_prescribed": [
            {"id": "brand_kisqali", "properties": {"name": "Kisqali"}}
        ]
    })

    # Mock list_relationships
    memory.list_relationships = MagicMock(return_value=[
        {
            "id": "rel_001",
            "type": "TREATED_BY",
            "source_id": "patient_001",
            "target_id": "hcp_001",
            "confidence": 0.95
        }
    ])

    # Mock count_relationships
    memory.count_relationships = MagicMock(return_value=10)

    # Mock get_stats
    memory.get_stats = MagicMock(return_value={
        "total_nodes": 30,
        "total_relationships": 22,
        "nodes_by_type": {"Patient": 5, "HCP": 5, "Trigger": 3},
        "relationships_by_type": {"TREATED_BY": 5, "PRESCRIBES": 5}
    })

    # Mock traverse_from_node for generic traversal
    memory.traverse_from_node = MagicMock(return_value={
        "nodes": [
            {"id": "node_001", "type": "Agent", "name": "orchestrator"}
        ],
        "relationships": []
    })

    return memory


@pytest.fixture
def mock_graphiti_service():
    """Mock Graphiti service."""
    service = AsyncMock()
    service.get_graph_stats = AsyncMock(return_value={
        "total_nodes": 30,
        "total_edges": 22,
        "nodes_by_type": {"Patient": 5, "HCP": 5},
        "edges_by_type": {"TREATED_BY": 5}
    })
    service.search = AsyncMock(return_value=[])
    return service


# =============================================================================
# GRAPH STATS TESTS
# =============================================================================

class TestGraphStats:
    """Tests for GET /graph/stats endpoint."""

    def test_get_stats_success(self, mock_semantic_memory, mock_graphiti_service):
        """Test successful retrieval of graph statistics."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory), \
             patch("src.api.routes.graph._get_graphiti_service", return_value=mock_graphiti_service):

            response = client.get("/graph/stats")

            assert response.status_code == 200
            data = response.json()
            assert "total_nodes" in data
            assert "total_relationships" in data
            assert "nodes_by_type" in data
            assert "timestamp" in data

    def test_get_stats_service_unavailable(self):
        """Test stats when graph service is unavailable."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=None), \
             patch("src.api.routes.graph._get_graphiti_service", return_value=None):

            response = client.get("/graph/stats")

            assert response.status_code == 200
            data = response.json()
            assert data["total_nodes"] == 0
            assert data["total_relationships"] == 0


# =============================================================================
# LIST NODES TESTS
# =============================================================================

class TestListNodes:
    """Tests for GET /graph/nodes endpoint."""

    def test_list_nodes_success(self, mock_semantic_memory):
        """Test successful listing of nodes."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/nodes")

            assert response.status_code == 200
            data = response.json()
            assert "nodes" in data
            assert "total_count" in data
            assert "has_more" in data
            assert len(data["nodes"]) == 2

    def test_list_nodes_with_entity_type_filter(self, mock_semantic_memory):
        """Test listing nodes with entity type filter."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/nodes?entity_types=Patient")

            assert response.status_code == 200
            mock_semantic_memory.list_nodes.assert_called_once()
            call_args = mock_semantic_memory.list_nodes.call_args
            assert call_args.kwargs["entity_types"] == ["Patient"]

    def test_list_nodes_with_search(self, mock_semantic_memory):
        """Test listing nodes with search query."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/nodes?search=Sarah")

            assert response.status_code == 200
            call_args = mock_semantic_memory.list_nodes.call_args
            assert call_args.kwargs["search"] == "Sarah"

    def test_list_nodes_with_pagination(self, mock_semantic_memory):
        """Test listing nodes with pagination parameters."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/nodes?limit=10&offset=5")

            assert response.status_code == 200
            call_args = mock_semantic_memory.list_nodes.call_args
            assert call_args.kwargs["limit"] == 10
            assert call_args.kwargs["offset"] == 5

    def test_list_nodes_service_unavailable(self):
        """Test listing nodes when service unavailable."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=None):

            response = client.get("/graph/nodes")

            assert response.status_code == 200
            data = response.json()
            assert data["nodes"] == []
            assert data["total_count"] == 0


# =============================================================================
# GET NODE TESTS
# =============================================================================

class TestGetNode:
    """Tests for GET /graph/nodes/{node_id} endpoint."""

    def test_get_node_success(self, mock_semantic_memory):
        """Test successful retrieval of a specific node."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/nodes/patient_001")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "patient_001"
            assert data["type"] == "Patient"

    def test_get_node_not_found(self, mock_semantic_memory):
        """Test retrieval of non-existent node."""
        mock_semantic_memory.get_node = MagicMock(return_value=None)

        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/nodes/nonexistent")

            assert response.status_code == 404

    def test_get_node_service_unavailable(self):
        """Test get node when service unavailable."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=None):

            response = client.get("/graph/nodes/patient_001")

            assert response.status_code == 503


# =============================================================================
# NODE NETWORK TESTS
# =============================================================================

class TestNodeNetwork:
    """Tests for GET /graph/nodes/{node_id}/network endpoint."""

    def test_patient_network_success(self, mock_semantic_memory):
        """Test successful retrieval of patient network."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/nodes/patient_001/network")

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert data["node_id"] == "patient_001"
            assert data["node_type"] == "Patient"
            assert "connected_nodes" in data
            assert "total_connections" in data
            assert "max_depth" in data
            assert "query_latency_ms" in data

            # Verify patient-specific connected node types
            connected = data["connected_nodes"]
            assert "hcps" in connected
            assert "treatments" in connected
            assert "triggers" in connected
            assert len(connected["hcps"]) == 1
            assert connected["hcps"][0]["id"] == "hcp_001"

            # Verify total connections
            assert data["total_connections"] == 3  # 1 hcp + 1 treatment + 1 trigger

    def test_hcp_network_success(self, mock_semantic_memory):
        """Test successful retrieval of HCP network."""
        # Configure mock to return HCP node
        mock_semantic_memory.get_node = MagicMock(return_value={
            "id": "hcp_001",
            "type": "HCP",
            "name": "Dr. Sarah Chen",
            "specialty": "Oncology"
        })

        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/nodes/hcp_001/network")

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert data["node_id"] == "hcp_001"
            assert data["node_type"] == "HCP"

            # Verify HCP-specific connected node types
            connected = data["connected_nodes"]
            assert "influenced_hcps" in connected
            assert "patients" in connected
            assert "brands_prescribed" in connected
            assert len(connected["influenced_hcps"]) == 1
            assert len(connected["patients"]) == 1

    def test_network_with_max_depth(self, mock_semantic_memory):
        """Test network retrieval with custom max_depth."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/nodes/patient_001/network?max_depth=3")

            assert response.status_code == 200
            data = response.json()
            assert data["max_depth"] == 3

            # Verify max_depth was passed to the method
            mock_semantic_memory.get_patient_network.assert_called_with(
                "patient_001", max_depth=3
            )

    def test_network_max_depth_validation(self, mock_semantic_memory):
        """Test max_depth parameter validation."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            # Test max_depth too high
            response = client.get("/graph/nodes/patient_001/network?max_depth=10")
            assert response.status_code == 422  # Validation error

            # Test max_depth too low
            response = client.get("/graph/nodes/patient_001/network?max_depth=0")
            assert response.status_code == 422

    def test_network_node_not_found(self, mock_semantic_memory):
        """Test network for non-existent node."""
        mock_semantic_memory.get_node = MagicMock(return_value=None)

        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/nodes/nonexistent/network")

            assert response.status_code == 404

    def test_network_service_unavailable(self):
        """Test network when service unavailable."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=None):

            response = client.get("/graph/nodes/patient_001/network")

            assert response.status_code == 503

    def test_network_generic_node_type(self, mock_semantic_memory):
        """Test network for non-Patient/HCP node types uses generic traversal."""
        # Configure mock to return a Trigger node
        mock_semantic_memory.get_node = MagicMock(return_value={
            "id": "trigger_001",
            "type": "Trigger",
            "name": "Peer Influence Trigger"
        })

        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/nodes/trigger_001/network")

            assert response.status_code == 200
            data = response.json()
            assert data["node_id"] == "trigger_001"
            assert data["node_type"] == "Trigger"

            # Verify generic traversal was used
            mock_semantic_memory.traverse_from_node.assert_called_once()

    def test_network_empty_connections(self, mock_semantic_memory):
        """Test network with no connections."""
        mock_semantic_memory.get_patient_network = MagicMock(return_value={
            "patient_id": "patient_isolated",
            "hcps": [],
            "treatments": [],
            "triggers": [],
            "causal_paths": [],
            "brands": []
        })

        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/nodes/patient_001/network")

            assert response.status_code == 200
            data = response.json()
            assert data["total_connections"] == 0


# =============================================================================
# LIST RELATIONSHIPS TESTS
# =============================================================================

class TestListRelationships:
    """Tests for GET /graph/relationships endpoint."""

    def test_list_relationships_success(self, mock_semantic_memory):
        """Test successful listing of relationships."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/relationships")

            assert response.status_code == 200
            data = response.json()
            assert "relationships" in data
            assert "total_count" in data
            assert len(data["relationships"]) == 1

    def test_list_relationships_with_type_filter(self, mock_semantic_memory):
        """Test listing relationships with type filter."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory):

            response = client.get("/graph/relationships?relationship_types=TREATED_BY")

            assert response.status_code == 200
            call_args = mock_semantic_memory.list_relationships.call_args
            assert call_args.kwargs["relationship_types"] == ["TREATED_BY"]


# =============================================================================
# SEARCH TESTS
# =============================================================================

class TestGraphSearch:
    """Tests for POST /graph/search endpoint."""

    def test_search_success(self, mock_semantic_memory, mock_graphiti_service):
        """Test successful graph search."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory), \
             patch("src.api.routes.graph._get_graphiti_service", return_value=mock_graphiti_service):

            response = client.post("/graph/search", json={
                "query": "What treatments does Dr. Chen prescribe?"
            })

            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert "query" in data
            assert data["query"] == "What treatments does Dr. Chen prescribe?"

    def test_search_with_parameters(self, mock_semantic_memory, mock_graphiti_service):
        """Test search with optional parameters."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory), \
             patch("src.api.routes.graph._get_graphiti_service", return_value=mock_graphiti_service):

            response = client.post("/graph/search", json={
                "query": "Find HCPs",
                "entity_types": ["HCP"],
                "k": 5,
                "min_score": 0.5
            })

            assert response.status_code == 200


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================

class TestGraphHealth:
    """Tests for GET /graph/health endpoint."""

    def test_health_check_healthy(self, mock_semantic_memory, mock_graphiti_service):
        """Test health check when services are available."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=mock_semantic_memory), \
             patch("src.api.routes.graph._get_graphiti_service", return_value=mock_graphiti_service):

            response = client.get("/graph/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "graphiti" in data
            assert "falkordb" in data
            assert "timestamp" in data

    def test_health_check_degraded(self):
        """Test health check when services are unavailable."""
        with patch("src.api.routes.graph._get_semantic_memory", return_value=None), \
             patch("src.api.routes.graph._get_graphiti_service", return_value=None):

            response = client.get("/graph/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
