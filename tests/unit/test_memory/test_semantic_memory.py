"""
Unit tests for E2I Semantic Memory module.

Tests FalkorDB graph operations for entity relationships and causal chains.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

from src.memory.episodic_memory import E2IEntityType
from src.memory.semantic_memory import (
    # Class
    FalkorDBSemanticMemory,
    # Label mappings
    E2I_TO_LABEL,
    LABEL_TO_E2I,
    # Singleton functions
    get_semantic_memory,
    reset_semantic_memory,
    # Utility functions
    query_semantic_graph,
    sync_to_semantic_graph,
    sync_data_layer_to_semantic_cache,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_graph():
    """Create a mock FalkorDB graph."""
    graph = MagicMock()
    graph.query.return_value = MagicMock(result_set=[], nodes_deleted=0)
    return graph


@pytest.fixture
def mock_falkordb_client(mock_graph):
    """Create a mock FalkorDB client."""
    client = MagicMock()
    client.select_graph.return_value = mock_graph
    return client


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock()
    config.semantic.graph_name = "e2i_semantic_graph"
    return config


@pytest.fixture
def semantic_memory(mock_falkordb_client, mock_config):
    """Create semantic memory instance with mocks."""
    with patch("src.memory.semantic_memory.get_config", return_value=mock_config):
        with patch("src.memory.semantic_memory.get_falkordb_client", return_value=mock_falkordb_client):
            memory = FalkorDBSemanticMemory()
            # Force lazy initialization
            _ = memory.client
            _ = memory.graph
            return memory


@pytest.fixture
def mock_node():
    """Create a mock graph node."""
    node = MagicMock()
    node.labels = ["Patient"]
    node.properties = {"id": "pat_123", "name": "Test Patient"}
    return node


@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value.data = {"synced": 10}
    return client


# ============================================================================
# LABEL MAPPING TESTS
# ============================================================================

class TestLabelMappings:
    """Tests for E2I to graph label mappings."""

    def test_e2i_to_label_mapping(self):
        """E2I_TO_LABEL should map all entity types."""
        assert E2I_TO_LABEL[E2IEntityType.PATIENT] == "Patient"
        assert E2I_TO_LABEL[E2IEntityType.HCP] == "HCP"
        assert E2I_TO_LABEL[E2IEntityType.TRIGGER] == "Trigger"
        assert E2I_TO_LABEL[E2IEntityType.CAUSAL_PATH] == "CausalPath"
        assert E2I_TO_LABEL[E2IEntityType.PREDICTION] == "Prediction"
        assert E2I_TO_LABEL[E2IEntityType.TREATMENT] == "Treatment"
        assert E2I_TO_LABEL[E2IEntityType.EXPERIMENT] == "Experiment"
        assert E2I_TO_LABEL[E2IEntityType.AGENT_ACTIVITY] == "AgentActivity"

    def test_label_to_e2i_mapping(self):
        """LABEL_TO_E2I should be reverse of E2I_TO_LABEL."""
        for entity_type, label in E2I_TO_LABEL.items():
            assert LABEL_TO_E2I[label] == entity_type

    def test_all_e2i_types_mapped(self):
        """All mapped E2I entity types should have bidirectional mapping."""
        # Verify all mapped types have valid reverse mappings
        assert len(E2I_TO_LABEL) == len(LABEL_TO_E2I)
        for entity_type in E2I_TO_LABEL:
            label = E2I_TO_LABEL[entity_type]
            assert LABEL_TO_E2I[label] == entity_type


# ============================================================================
# SEMANTIC MEMORY CLASS TESTS
# ============================================================================

class TestFalkorDBSemanticMemoryInit:
    """Tests for FalkorDBSemanticMemory initialization."""

    def test_init_creates_instance(self, mock_config):
        """FalkorDBSemanticMemory should initialize with config."""
        with patch("src.memory.semantic_memory.get_config", return_value=mock_config):
            memory = FalkorDBSemanticMemory()
            assert memory._client is None
            assert memory._graph is None

    def test_lazy_client_initialization(self, mock_falkordb_client, mock_config):
        """Client should be lazily initialized on first access."""
        with patch("src.memory.semantic_memory.get_config", return_value=mock_config):
            with patch("src.memory.semantic_memory.get_falkordb_client", return_value=mock_falkordb_client):
                memory = FalkorDBSemanticMemory()
                assert memory._client is None

                # Access client
                client = memory.client
                assert client is mock_falkordb_client

    def test_lazy_graph_initialization(self, mock_falkordb_client, mock_config):
        """Graph should be lazily initialized on first access."""
        with patch("src.memory.semantic_memory.get_config", return_value=mock_config):
            with patch("src.memory.semantic_memory.get_falkordb_client", return_value=mock_falkordb_client):
                memory = FalkorDBSemanticMemory()

                # Access graph
                graph = memory.graph
                mock_falkordb_client.select_graph.assert_called_once_with("e2i_semantic_graph")


# ============================================================================
# ENTITY MANAGEMENT TESTS
# ============================================================================

class TestAddE2IEntity:
    """Tests for add_e2i_entity method."""

    def test_add_patient_entity(self, semantic_memory, mock_graph):
        """add_e2i_entity should create Patient node with MERGE."""
        result = semantic_memory.add_e2i_entity(
            E2IEntityType.PATIENT,
            "pat_123",
            {"name": "John Doe"}
        )

        assert result is True
        mock_graph.query.assert_called_once()

        # Check query contains correct label and ID
        call_args = mock_graph.query.call_args
        query = call_args[0][0]
        params = call_args[0][1]

        assert "MERGE (e:Patient {id: $entity_id})" in query
        assert params["entity_id"] == "pat_123"
        assert params["name"] == "John Doe"
        assert params["e2i_entity_type"] == "patient"

    def test_add_hcp_entity(self, semantic_memory, mock_graph):
        """add_e2i_entity should create HCP node."""
        semantic_memory.add_e2i_entity(
            E2IEntityType.HCP,
            "hcp_456",
            {"specialty": "Oncology"}
        )

        call_args = mock_graph.query.call_args
        query = call_args[0][0]
        params = call_args[0][1]

        assert "MERGE (e:HCP {id: $entity_id})" in query
        assert params["specialty"] == "Oncology"

    def test_add_entity_without_properties(self, semantic_memory, mock_graph):
        """add_e2i_entity should work without additional properties."""
        result = semantic_memory.add_e2i_entity(
            E2IEntityType.TRIGGER,
            "trig_789"
        )

        assert result is True
        call_args = mock_graph.query.call_args
        params = call_args[0][1]

        assert params["entity_id"] == "trig_789"
        assert "e2i_entity_type" in params
        assert "updated_at" in params

    def test_add_entity_defaults_to_entity_label(self, semantic_memory, mock_graph):
        """add_e2i_entity should use 'Entity' for unknown types."""
        # Create a mock entity type that's not in the mapping
        with patch.dict(E2I_TO_LABEL, {}, clear=True):
            semantic_memory.add_e2i_entity(
                E2IEntityType.PATIENT,  # Will not find mapping
                "test_123"
            )

        call_args = mock_graph.query.call_args
        query = call_args[0][0]
        assert "MERGE (e:Entity {id: $entity_id})" in query


class TestGetEntity:
    """Tests for get_entity method."""

    def test_get_existing_entity(self, semantic_memory, mock_graph, mock_node):
        """get_entity should return entity properties."""
        mock_graph.query.return_value.result_set = [[mock_node]]

        result = semantic_memory.get_entity(E2IEntityType.PATIENT, "pat_123")

        assert result is not None
        assert result["id"] == "pat_123"
        assert result["name"] == "Test Patient"

    def test_get_nonexistent_entity(self, semantic_memory, mock_graph):
        """get_entity should return None for missing entity."""
        mock_graph.query.return_value.result_set = []

        result = semantic_memory.get_entity(E2IEntityType.PATIENT, "nonexistent")

        assert result is None

    def test_get_entity_empty_result_set(self, semantic_memory, mock_graph):
        """get_entity should handle empty result gracefully."""
        mock_graph.query.return_value.result_set = None

        result = semantic_memory.get_entity(E2IEntityType.PATIENT, "pat_123")

        assert result is None


class TestDeleteEntity:
    """Tests for delete_entity method."""

    def test_delete_existing_entity(self, semantic_memory, mock_graph):
        """delete_entity should delete node and return True."""
        mock_graph.query.return_value.nodes_deleted = 1

        result = semantic_memory.delete_entity(E2IEntityType.PATIENT, "pat_123")

        assert result is True
        call_args = mock_graph.query.call_args
        query = call_args[0][0]
        assert "DETACH DELETE" in query

    def test_delete_nonexistent_entity(self, semantic_memory, mock_graph):
        """delete_entity should return False for missing entity."""
        mock_graph.query.return_value.nodes_deleted = 0

        result = semantic_memory.delete_entity(E2IEntityType.PATIENT, "nonexistent")

        assert result is False


# ============================================================================
# RELATIONSHIP MANAGEMENT TESTS
# ============================================================================

class TestAddE2IRelationship:
    """Tests for add_e2i_relationship method."""

    def test_add_treated_by_relationship(self, semantic_memory, mock_graph):
        """add_e2i_relationship should create TREATED_BY relationship."""
        result = semantic_memory.add_e2i_relationship(
            E2IEntityType.PATIENT,
            "pat_123",
            E2IEntityType.HCP,
            "hcp_456",
            "TREATED_BY"
        )

        assert result is True

        # Should call query 3 times: 2 entity adds + 1 relationship
        assert mock_graph.query.call_count == 3

        # Check relationship query
        last_call = mock_graph.query.call_args_list[-1]
        query = last_call[0][0]
        params = last_call[0][1]

        assert "MERGE (s)-[r:TREATED_BY]->(t)" in query
        assert params["source_id"] == "pat_123"
        assert params["target_id"] == "hcp_456"

    def test_add_relationship_with_properties(self, semantic_memory, mock_graph):
        """add_e2i_relationship should include properties."""
        semantic_memory.add_e2i_relationship(
            E2IEntityType.CAUSAL_PATH,
            "cp_123",
            E2IEntityType.TRIGGER,
            "trig_456",
            "CAUSES",
            {"confidence": 0.85, "effect_size": 0.2}
        )

        last_call = mock_graph.query.call_args_list[-1]
        params = last_call[0][1]

        assert params["confidence"] == 0.85
        assert params["effect_size"] == 0.2

    def test_add_relationship_ensures_entities_exist(self, semantic_memory, mock_graph):
        """add_e2i_relationship should create entities if needed."""
        semantic_memory.add_e2i_relationship(
            E2IEntityType.PATIENT,
            "new_pat",
            E2IEntityType.HCP,
            "new_hcp",
            "TREATED_BY"
        )

        # First two calls should be MERGE for entities
        first_call = mock_graph.query.call_args_list[0][0][0]
        second_call = mock_graph.query.call_args_list[1][0][0]

        assert "MERGE (e:Patient" in first_call
        assert "MERGE (e:HCP" in second_call


class TestGetRelationships:
    """Tests for get_relationships method."""

    def test_get_outgoing_relationships(self, semantic_memory, mock_graph):
        """get_relationships should get outgoing relationships."""
        mock_graph.query.return_value.result_set = [
            ["pat_123", "TREATED_BY", "hcp_456", {"since": "2024-01-01"}]
        ]

        result = semantic_memory.get_relationships(
            E2IEntityType.PATIENT,
            "pat_123",
            direction="outgoing"
        )

        assert len(result) == 1
        assert result[0]["source"] == "pat_123"
        assert result[0]["rel_type"] == "TREATED_BY"
        assert result[0]["target"] == "hcp_456"

    def test_get_incoming_relationships(self, semantic_memory, mock_graph):
        """get_relationships should get incoming relationships."""
        mock_graph.query.return_value.result_set = [
            ["pat_123", "TREATED_BY", "hcp_456", None]
        ]

        result = semantic_memory.get_relationships(
            E2IEntityType.HCP,
            "hcp_456",
            direction="incoming"
        )

        assert len(result) == 1

    def test_get_both_directions(self, semantic_memory, mock_graph):
        """get_relationships should get both directions by default."""
        mock_graph.query.return_value.result_set = []

        semantic_memory.get_relationships(
            E2IEntityType.PATIENT,
            "pat_123",
            direction="both"
        )

        # Should not have specific direction in query
        call_args = mock_graph.query.call_args
        query = call_args[0][0]
        assert "-[r]-(" in query  # Undirected pattern


# ============================================================================
# NETWORK TRAVERSAL TESTS
# ============================================================================

class TestGetPatientNetwork:
    """Tests for get_patient_network method."""

    def test_get_patient_network_structure(self, semantic_memory, mock_graph):
        """get_patient_network should return structured network."""
        mock_graph.query.return_value.result_set = []

        result = semantic_memory.get_patient_network("pat_123")

        assert result["patient_id"] == "pat_123"
        assert "hcps" in result
        assert "treatments" in result
        assert "triggers" in result
        assert "causal_paths" in result
        assert "brands" in result

    def test_get_patient_network_with_connections(self, semantic_memory, mock_graph):
        """get_patient_network should categorize connected nodes."""
        hcp_node = MagicMock()
        hcp_node.labels = ["HCP"]
        hcp_node.properties = {"id": "hcp_123"}

        trigger_node = MagicMock()
        trigger_node.labels = ["Trigger"]
        trigger_node.properties = {"id": "trig_456"}

        mock_graph.query.return_value.result_set = [
            [MagicMock(), MagicMock(), hcp_node],
            [MagicMock(), MagicMock(), trigger_node]
        ]

        result = semantic_memory.get_patient_network("pat_123", max_depth=2)

        assert len(result["hcps"]) == 1
        assert len(result["triggers"]) == 1
        assert result["hcps"][0]["id"] == "hcp_123"

    def test_get_patient_network_respects_max_depth(self, semantic_memory, mock_graph):
        """get_patient_network should pass max_depth to query."""
        mock_graph.query.return_value.result_set = []

        semantic_memory.get_patient_network("pat_123", max_depth=5)

        call_args = mock_graph.query.call_args
        params = call_args[0][1]
        assert params["max_depth"] == 5


class TestGetHCPInfluenceNetwork:
    """Tests for get_hcp_influence_network method."""

    def test_get_hcp_network_structure(self, semantic_memory, mock_graph):
        """get_hcp_influence_network should return structured network."""
        mock_graph.query.return_value.result_set = []

        result = semantic_memory.get_hcp_influence_network("hcp_123")

        assert result["hcp_id"] == "hcp_123"
        assert "influenced_hcps" in result
        assert "patients" in result
        assert "brands_prescribed" in result

    def test_get_hcp_network_with_connections(self, semantic_memory, mock_graph):
        """get_hcp_influence_network should categorize connections."""
        patient_node = MagicMock()
        patient_node.labels = ["Patient"]
        patient_node.properties = {"id": "pat_123"}

        brand_node = MagicMock()
        brand_node.labels = ["Brand"]
        brand_node.properties = {"id": "Kisqali"}

        mock_graph.query.return_value.result_set = [
            [MagicMock(), MagicMock(), patient_node, "TREATS"],
            [MagicMock(), MagicMock(), brand_node, "PRESCRIBES"]
        ]

        result = semantic_memory.get_hcp_influence_network("hcp_123")

        assert len(result["patients"]) == 1
        assert len(result["brands_prescribed"]) == 1


# ============================================================================
# CAUSAL CHAIN TESTS
# ============================================================================

class TestTraverseCausalChain:
    """Tests for traverse_causal_chain method."""

    def test_traverse_returns_chains(self, semantic_memory, mock_graph):
        """traverse_causal_chain should return causal chains."""
        mock_graph.query.return_value.result_set = [
            [
                [{"id": "a", "type": "CausalPath"}, {"id": "b", "type": "KPI"}],
                [{"type": "IMPACTS", "conf": 0.8}]
            ]
        ]

        result = semantic_memory.traverse_causal_chain("a", max_depth=3)

        assert len(result) == 1
        assert result[0]["path_length"] == 1
        assert len(result[0]["nodes"]) == 2
        assert len(result[0]["relationships"]) == 1

    def test_traverse_empty_chains(self, semantic_memory, mock_graph):
        """traverse_causal_chain should handle no chains."""
        mock_graph.query.return_value.result_set = []

        result = semantic_memory.traverse_causal_chain("isolated_node")

        assert result == []


class TestFindCausalPathsForKPI:
    """Tests for find_causal_paths_for_kpi method."""

    def test_find_kpi_paths(self, semantic_memory, mock_graph):
        """find_causal_paths_for_kpi should return matching paths."""
        mock_graph.query.return_value.result_set = [
            ["cp_1", 0.15, 0.85, "DoWhy"],
            ["cp_2", 0.10, 0.70, "EconML"]
        ]

        result = semantic_memory.find_causal_paths_for_kpi("TRx", min_confidence=0.5)

        assert len(result) == 2
        assert result[0]["path_id"] == "cp_1"
        assert result[0]["effect_size"] == 0.15
        assert result[0]["confidence"] == 0.85
        assert result[0]["method"] == "DoWhy"

    def test_find_kpi_paths_respects_min_confidence(self, semantic_memory, mock_graph):
        """find_causal_paths_for_kpi should pass min_confidence to query."""
        mock_graph.query.return_value.result_set = []

        semantic_memory.find_causal_paths_for_kpi("NRx", min_confidence=0.8)

        call_args = mock_graph.query.call_args
        params = call_args[0][1]
        assert params["min_confidence"] == 0.8


class TestFindCommonPaths:
    """Tests for find_common_paths method."""

    def test_find_paths_between_entities(self, semantic_memory, mock_graph):
        """find_common_paths should return connecting paths."""
        mock_graph.query.return_value.result_set = [
            [
                [{"id": "a", "type": "Patient"}, {"id": "b", "type": "HCP"}],
                ["TREATED_BY"],
                1
            ]
        ]

        result = semantic_memory.find_common_paths("a", "b", max_depth=3)

        assert len(result) == 1
        assert result[0]["path_length"] == 1
        assert "TREATED_BY" in result[0]["relationship_types"]


# ============================================================================
# GRAPH STATISTICS TESTS
# ============================================================================

class TestGetGraphStats:
    """Tests for get_graph_stats method."""

    def test_get_graph_stats(self, semantic_memory, mock_graph):
        """get_graph_stats should aggregate counts."""
        # First query returns node counts
        mock_graph.query.return_value.result_set = [
            ["Patient", 100],
            ["HCP", 50]
        ]

        result = semantic_memory.get_graph_stats()

        assert "total_nodes" in result
        assert "total_relationships" in result
        assert "nodes_by_type" in result
        assert "relationships_by_type" in result


# ============================================================================
# SINGLETON TESTS
# ============================================================================

class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_semantic_memory_returns_singleton(self, mock_falkordb_client, mock_config):
        """get_semantic_memory should return same instance."""
        reset_semantic_memory()

        with patch("src.memory.semantic_memory.get_config", return_value=mock_config):
            with patch("src.memory.semantic_memory.get_falkordb_client", return_value=mock_falkordb_client):
                memory1 = get_semantic_memory()
                memory2 = get_semantic_memory()

                assert memory1 is memory2

        reset_semantic_memory()

    def test_reset_clears_singleton(self, mock_falkordb_client, mock_config):
        """reset_semantic_memory should clear singleton."""
        with patch("src.memory.semantic_memory.get_config", return_value=mock_config):
            with patch("src.memory.semantic_memory.get_falkordb_client", return_value=mock_falkordb_client):
                memory1 = get_semantic_memory()
                reset_semantic_memory()
                memory2 = get_semantic_memory()

                assert memory1 is not memory2

        reset_semantic_memory()


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestQuerySemanticGraph:
    """Tests for query_semantic_graph function."""

    @pytest.mark.asyncio
    async def test_query_patient_network(self, mock_falkordb_client, mock_config):
        """query_semantic_graph should query patient network."""
        reset_semantic_memory()

        with patch("src.memory.semantic_memory.get_config", return_value=mock_config):
            with patch("src.memory.semantic_memory.get_falkordb_client", return_value=mock_falkordb_client):
                mock_falkordb_client.select_graph.return_value.query.return_value.result_set = []

                result = await query_semantic_graph({
                    "start_nodes": ["pat_123"],
                    "entity_type": "patient",
                    "max_depth": 2
                })

                assert len(result) == 1
                assert result[0]["type"] == "patient_network"

        reset_semantic_memory()

    @pytest.mark.asyncio
    async def test_query_hcp_network(self, mock_falkordb_client, mock_config):
        """query_semantic_graph should query HCP network."""
        reset_semantic_memory()

        with patch("src.memory.semantic_memory.get_config", return_value=mock_config):
            with patch("src.memory.semantic_memory.get_falkordb_client", return_value=mock_falkordb_client):
                mock_falkordb_client.select_graph.return_value.query.return_value.result_set = []

                result = await query_semantic_graph({
                    "start_nodes": ["hcp_456"],
                    "entity_type": "hcp"
                })

                assert result[0]["type"] == "hcp_network"

        reset_semantic_memory()

    @pytest.mark.asyncio
    async def test_query_causal_chains(self, mock_falkordb_client, mock_config):
        """query_semantic_graph should query causal chains."""
        reset_semantic_memory()

        with patch("src.memory.semantic_memory.get_config", return_value=mock_config):
            with patch("src.memory.semantic_memory.get_falkordb_client", return_value=mock_falkordb_client):
                mock_falkordb_client.select_graph.return_value.query.return_value.result_set = [
                    [[{"id": "a"}], [{"type": "CAUSES"}]]
                ]

                result = await query_semantic_graph({
                    "start_nodes": ["cp_123"],
                    "follow_causal": True,
                    "max_depth": 3
                })

                assert result[0]["type"] == "causal_chain"

        reset_semantic_memory()


class TestSyncToSemanticGraph:
    """Tests for sync_to_semantic_graph function."""

    @pytest.mark.asyncio
    async def test_sync_e2i_triplet(self, mock_falkordb_client, mock_config):
        """sync_to_semantic_graph should add E2I triplet."""
        reset_semantic_memory()

        with patch("src.memory.semantic_memory.get_config", return_value=mock_config):
            with patch("src.memory.semantic_memory.get_falkordb_client", return_value=mock_falkordb_client):
                result = await sync_to_semantic_graph({
                    "subject": "pat_123",
                    "subject_type": "Patient",
                    "predicate": "TREATED_BY",
                    "object": "hcp_456",
                    "object_type": "HCP",
                    "confidence": 0.9
                })

                assert result is True

        reset_semantic_memory()

    @pytest.mark.asyncio
    async def test_sync_generic_triplet(self, mock_falkordb_client, mock_config):
        """sync_to_semantic_graph should handle unknown types."""
        reset_semantic_memory()

        with patch("src.memory.semantic_memory.get_config", return_value=mock_config):
            with patch("src.memory.semantic_memory.get_falkordb_client", return_value=mock_falkordb_client):
                result = await sync_to_semantic_graph({
                    "subject": "unknown_1",
                    "subject_type": "Unknown",
                    "predicate": "RELATES_TO",
                    "object": "unknown_2",
                    "object_type": "Unknown"
                })

                assert result is True

        reset_semantic_memory()


class TestSyncDataLayerToSemanticCache:
    """Tests for sync_data_layer_to_semantic_cache function."""

    @pytest.mark.asyncio
    async def test_sync_to_cache(self, mock_supabase):
        """sync_data_layer_to_semantic_cache should call RPC."""
        with patch("src.memory.semantic_memory.get_supabase_client", return_value=mock_supabase):
            result = await sync_data_layer_to_semantic_cache()

        mock_supabase.rpc.assert_called_once_with(
            "sync_hcp_patient_relationships_to_cache",
            {}
        )
        assert result == {"synced": 10}


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_all_e2i_entity_types_work(self, semantic_memory, mock_graph):
        """All E2I entity types should be addable."""
        for entity_type in E2IEntityType:
            result = semantic_memory.add_e2i_entity(entity_type, f"test_{entity_type.value}")
            assert result is True

    def test_relationship_common_types(self, semantic_memory, mock_graph):
        """Common relationship types should work."""
        rel_types = [
            "TREATED_BY",
            "PRESCRIBED",
            "PRESCRIBES",
            "GENERATED",
            "CAUSES",
            "IMPACTS"
        ]

        for rel_type in rel_types:
            result = semantic_memory.add_e2i_relationship(
                E2IEntityType.PATIENT,
                "source",
                E2IEntityType.HCP,
                "target",
                rel_type
            )
            assert result is True

    def test_node_without_labels_attribute(self, semantic_memory, mock_graph):
        """get_patient_network should handle nodes without labels."""
        node = MagicMock()
        del node.labels  # Remove labels attribute
        node.properties = {"id": "test"}

        mock_graph.query.return_value.result_set = [
            [MagicMock(), MagicMock(), node]
        ]

        # Should not raise
        result = semantic_memory.get_patient_network("pat_123")
        assert result["patient_id"] == "pat_123"

    def test_empty_properties_in_relationship(self, semantic_memory, mock_graph):
        """get_relationships should handle None properties."""
        mock_graph.query.return_value.result_set = [
            ["src", "REL", "tgt", None]
        ]

        result = semantic_memory.get_relationships(
            E2IEntityType.PATIENT,
            "pat_123"
        )

        assert len(result) == 1
        assert result[0]["properties"] == {}

    def test_graph_stats_handles_none_label(self, semantic_memory, mock_graph):
        """get_graph_stats should handle None label."""
        mock_graph.query.return_value.result_set = [
            [None, 10],
            ["Patient", 50]
        ]

        result = semantic_memory.get_graph_stats()

        assert "Unknown" in result["nodes_by_type"]
        assert result["nodes_by_type"]["Unknown"] == 10
