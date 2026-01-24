"""
Unit tests for semantic memory pagination functionality.

Tests:
- get_patient_network() pagination
- get_hcp_influence_network() pagination
- traverse_causal_chain() limit
- count methods for pagination
"""

from unittest.mock import MagicMock, patch

import pytest

from src.memory.semantic_memory import (
    FalkorDBSemanticMemory,
    reset_semantic_memory,
)


class TestPatientNetworkPagination:
    """Tests for get_patient_network pagination."""

    def setup_method(self):
        """Reset semantic memory before each test."""
        reset_semantic_memory()

    def test_default_pagination_parameters(self):
        """Test that default pagination parameters are applied."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_result = MagicMock()
            mock_result.result_set = []
            mock_graph.query.return_value = mock_result

            result = memory.get_patient_network("patient-1")

            # Check that pagination metadata is included
            assert "pagination" in result
            assert result["pagination"]["limit"] == 100
            assert result["pagination"]["offset"] == 0

    def test_custom_pagination_parameters(self):
        """Test custom limit and offset."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_result = MagicMock()
            mock_result.result_set = []
            mock_graph.query.return_value = mock_result

            result = memory.get_patient_network(
                "patient-1", limit=50, offset=100
            )

            assert result["pagination"]["limit"] == 50
            assert result["pagination"]["offset"] == 100

    def test_limit_capped_at_500(self):
        """Test that limit is capped at 500."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_result = MagicMock()
            mock_result.result_set = []
            mock_graph.query.return_value = mock_result

            result = memory.get_patient_network("patient-1", limit=1000)

            # Should be capped at 500
            assert result["pagination"]["limit"] == 500

    def test_has_more_when_at_limit(self):
        """Test has_more flag when results equal limit."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            # Create mock nodes that match the limit
            mock_nodes = []
            for i in range(10):
                node = MagicMock()
                node.labels = ["HCP"]
                node.properties = {"id": f"hcp-{i}"}
                mock_nodes.append([node])

            mock_result = MagicMock()
            mock_result.result_set = mock_nodes
            mock_graph.query.return_value = mock_result

            result = memory.get_patient_network("patient-1", limit=10)

            assert result["pagination"]["has_more"] is True
            assert result["pagination"]["returned"] == 10

    def test_no_more_when_under_limit(self):
        """Test has_more is False when results under limit."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_nodes = []
            for i in range(5):
                node = MagicMock()
                node.labels = ["HCP"]
                node.properties = {"id": f"hcp-{i}"}
                mock_nodes.append([node])

            mock_result = MagicMock()
            mock_result.result_set = mock_nodes
            mock_graph.query.return_value = mock_result

            result = memory.get_patient_network("patient-1", limit=10)

            assert result["pagination"]["has_more"] is False
            assert result["pagination"]["returned"] == 5


class TestHcpInfluenceNetworkPagination:
    """Tests for get_hcp_influence_network pagination."""

    def setup_method(self):
        """Reset semantic memory before each test."""
        reset_semantic_memory()

    def test_pagination_metadata_included(self):
        """Test that pagination metadata is included in HCP network."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_result = MagicMock()
            mock_result.result_set = []
            mock_graph.query.return_value = mock_result

            result = memory.get_hcp_influence_network("hcp-1", limit=25, offset=50)

            assert "pagination" in result
            assert result["pagination"]["limit"] == 25
            assert result["pagination"]["offset"] == 50


class TestCausalChainLimit:
    """Tests for traverse_causal_chain limit."""

    def setup_method(self):
        """Reset semantic memory before each test."""
        reset_semantic_memory()

    def test_default_limit(self):
        """Test that default limit is 50."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_result = MagicMock()
            mock_result.result_set = []
            mock_graph.query.return_value = mock_result

            memory.traverse_causal_chain("entity-1")

            # Verify LIMIT 50 is in the query
            call_args = mock_graph.query.call_args
            query = call_args[0][0]
            assert "LIMIT 50" in query

    def test_custom_limit(self):
        """Test custom limit parameter."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_result = MagicMock()
            mock_result.result_set = []
            mock_graph.query.return_value = mock_result

            memory.traverse_causal_chain("entity-1", limit=25)

            call_args = mock_graph.query.call_args
            query = call_args[0][0]
            assert "LIMIT 25" in query

    def test_limit_capped_at_200(self):
        """Test that limit is capped at 200."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_result = MagicMock()
            mock_result.result_set = []
            mock_graph.query.return_value = mock_result

            memory.traverse_causal_chain("entity-1", limit=500)

            call_args = mock_graph.query.call_args
            query = call_args[0][0]
            assert "LIMIT 200" in query


class TestCountMethods:
    """Tests for count methods."""

    def setup_method(self):
        """Reset semantic memory before each test."""
        reset_semantic_memory()

    def test_count_patient_network(self):
        """Test count_patient_network returns integer."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_result = MagicMock()
            mock_result.result_set = [[42]]
            mock_graph.query.return_value = mock_result

            count = memory.count_patient_network("patient-1")

            assert count == 42
            assert isinstance(count, int)

    def test_count_patient_network_empty(self):
        """Test count returns 0 for empty results."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_result = MagicMock()
            mock_result.result_set = []
            mock_graph.query.return_value = mock_result

            count = memory.count_patient_network("patient-1")

            assert count == 0

    def test_count_hcp_influence_network(self):
        """Test count_hcp_influence_network returns integer."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_result = MagicMock()
            mock_result.result_set = [[15]]
            mock_graph.query.return_value = mock_result

            count = memory.count_hcp_influence_network("hcp-1")

            assert count == 15

    def test_count_causal_chains(self):
        """Test count_causal_chains returns integer."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_result = MagicMock()
            mock_result.result_set = [[8]]
            mock_graph.query.return_value = mock_result

            count = memory.count_causal_chains("entity-1")

            assert count == 8

    def test_depth_parameter_sanitized_in_count(self):
        """Test that max_depth is sanitized in count methods."""
        memory = FalkorDBSemanticMemory()

        with patch.object(memory, "_graph", create=True) as mock_graph:
            mock_result = MagicMock()
            mock_result.result_set = [[0]]
            mock_graph.query.return_value = mock_result

            # Try to pass invalid depth
            memory.count_patient_network("patient-1", max_depth=10)

            # Should be capped at 5
            call_args = mock_graph.query.call_args
            query = call_args[0][0]
            assert "*1..5" in query
