"""Integration tests for GraphBuilderNode with causal discovery.

Version: 1.0.0
Tests the integration between GraphBuilderNode and the discovery module.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode, build_causal_graph
from src.agents.causal_impact.state import CausalImpactState
from src.causal_engine.discovery import (
    DiscoveredEdge,
    DiscoveryConfig,
    DiscoveryResult,
    GateDecision,
)
from src.causal_engine.discovery.gate import GateEvaluation


class TestGraphBuilderNodeBasic:
    """Test basic GraphBuilderNode functionality without discovery."""

    @pytest.fixture
    def node(self):
        """Create GraphBuilderNode instance."""
        return GraphBuilderNode()

    @pytest.fixture
    def basic_state(self) -> CausalImpactState:
        """Create basic state for testing."""
        return {
            "query": "What is the effect of HCP engagement on patient conversion?",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "auto_discover": False,
        }

    @pytest.mark.asyncio
    async def test_execute_without_discovery(self, node, basic_state):
        """Test execution without auto-discovery."""
        result = await node.execute(basic_state)

        assert "causal_graph" in result
        assert result["causal_graph"]["discovery_enabled"] is False
        assert result["causal_graph"]["treatment_nodes"] == ["hcp_engagement_level"]
        assert result["causal_graph"]["outcome_nodes"] == ["patient_conversion_rate"]

    @pytest.mark.asyncio
    async def test_dag_version_hash_computed(self, node, basic_state):
        """Test that DAG version hash is computed."""
        result = await node.execute(basic_state)

        assert "dag_version_hash" in result
        assert result["dag_version_hash"] is not None
        assert len(result["dag_version_hash"]) > 0

    @pytest.mark.asyncio
    async def test_adjustment_sets_computed(self, node, basic_state):
        """Test that adjustment sets are computed."""
        result = await node.execute(basic_state)

        assert "adjustment_sets" in result["causal_graph"]
        assert isinstance(result["causal_graph"]["adjustment_sets"], list)

    @pytest.mark.asyncio
    async def test_dot_format_generated(self, node, basic_state):
        """Test that DOT format is generated for visualization."""
        result = await node.execute(basic_state)

        assert "dag_dot" in result["causal_graph"]
        assert "digraph" in result["causal_graph"]["dag_dot"]


class TestGraphBuilderNodeDiscoveryIntegration:
    """Test GraphBuilderNode with auto-discovery enabled."""

    @pytest.fixture
    def node(self):
        """Create GraphBuilderNode instance."""
        return GraphBuilderNode()

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data with known causal structure."""
        np.random.seed(42)
        n = 200

        # A causes B causes Outcome
        # Treatment affects Outcome
        treatment = np.random.randn(n)
        a = np.random.randn(n)
        b = 0.7 * a + 0.3 * np.random.randn(n)
        outcome = 0.5 * treatment + 0.3 * b + 0.2 * np.random.randn(n)

        return pd.DataFrame(
            {
                "hcp_engagement_level": treatment,
                "marketing_spend": a,
                "hcp_meeting_frequency": b,
                "patient_conversion_rate": outcome,
            }
        )

    @pytest.fixture
    def discovery_state(self, synthetic_data) -> CausalImpactState:
        """Create state with discovery enabled."""
        return {
            "query": "What is the effect of HCP engagement on patient conversion?",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "auto_discover": True,
            "session_id": str(uuid4()),
            "data_cache": {"data": synthetic_data},
            "discovery_algorithms": ["ges", "pc"],
            "discovery_ensemble_threshold": 0.5,
        }

    @pytest.mark.asyncio
    async def test_discovery_runs_when_enabled(self, node, discovery_state):
        """Test that discovery runs when auto_discover=True."""
        result = await node.execute(discovery_state)

        assert "causal_graph" in result
        assert result["causal_graph"]["discovery_enabled"] is True
        assert "discovery_latency_ms" in result

    @pytest.mark.asyncio
    async def test_discovery_result_in_state(self, node, discovery_state):
        """Test that discovery result is included in output state."""
        result = await node.execute(discovery_state)

        assert "discovery_result" in result
        assert "success" in result["discovery_result"]

    @pytest.mark.asyncio
    async def test_discovery_gate_evaluation_in_state(self, node, discovery_state):
        """Test that gate evaluation is included in output state."""
        result = await node.execute(discovery_state)

        assert "discovery_gate_evaluation" in result
        assert "decision" in result["discovery_gate_evaluation"]
        assert "confidence" in result["discovery_gate_evaluation"]

    @pytest.mark.asyncio
    async def test_discovery_algorithms_used_recorded(self, node, discovery_state):
        """Test that algorithms used are recorded."""
        result = await node.execute(discovery_state)

        assert result["causal_graph"]["discovery_algorithms_used"] == ["ges", "pc"]

    @pytest.mark.asyncio
    async def test_discovery_without_data_falls_back(self, node):
        """Test that discovery falls back to manual DAG when no data."""
        state = {
            "query": "Test query",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "auto_discover": True,
            # No data_cache provided
        }

        result = await node.execute(state)

        # Should succeed but with manual DAG
        assert "causal_graph" in result
        # Discovery should have failed gracefully
        assert result["causal_graph"]["discovery_gate_decision"] is None or "error" in str(result)


class TestGraphBuilderGateDecisions:
    """Test handling of different gate decisions."""

    @pytest.fixture
    def node(self):
        """Create GraphBuilderNode instance."""
        return GraphBuilderNode()

    @pytest.fixture
    def base_state(self) -> CausalImpactState:
        """Create base state."""
        return {
            "query": "Test query",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "confounders": [],
            "auto_discover": True,
        }

    def _create_mock_discovery_result(self, edges):
        """Create mock discovery result with given edges."""
        dag = nx.DiGraph()
        dag.add_edges_from([(e.source, e.target) for e in edges])

        return DiscoveryResult(
            success=True,
            config=DiscoveryConfig(),
            ensemble_dag=dag,
            edges=edges,
        )

    @pytest.mark.asyncio
    async def test_accept_decision_uses_discovered_dag(self, node, base_state):
        """Test ACCEPT decision uses discovered DAG."""
        # Create mock discovery result
        edges = [
            DiscoveredEdge(source="treatment", target="outcome", confidence=0.95),
            DiscoveredEdge(source="confounder", target="treatment", confidence=0.90),
            DiscoveredEdge(source="confounder", target="outcome", confidence=0.88),
        ]
        mock_result = self._create_mock_discovery_result(edges)

        mock_evaluation = GateEvaluation(
            decision=GateDecision.ACCEPT,
            confidence=0.9,
            reasons=["High confidence"],
            high_confidence_edges=edges,
        )

        with patch.object(node, "_run_discovery", new_callable=AsyncMock) as mock_discovery:
            mock_discovery.return_value = (mock_result, mock_evaluation.to_dict())

            result = await node.execute(base_state)

        assert result["causal_graph"]["discovery_gate_decision"] == "accept"
        assert result["causal_graph"]["confidence"] >= 0.9

    @pytest.mark.asyncio
    async def test_augment_decision_adds_edges(self, node, base_state):
        """Test AUGMENT decision adds high-confidence discovered edges."""
        # Create mock discovery result
        edges = [
            DiscoveredEdge(source="treatment", target="outcome", confidence=0.95),
            DiscoveredEdge(source="new_edge_source", target="new_edge_target", confidence=0.92),
        ]
        mock_result = self._create_mock_discovery_result(edges)

        mock_evaluation = GateEvaluation(
            decision=GateDecision.AUGMENT,
            confidence=0.75,
            reasons=["Medium confidence, augmenting"],
            high_confidence_edges=[e for e in edges if e.confidence >= 0.9],
        )

        with patch.object(node, "_run_discovery", new_callable=AsyncMock) as mock_discovery:
            mock_discovery.return_value = (mock_result, mock_evaluation.to_dict())

            result = await node.execute(base_state)

        assert result["causal_graph"]["discovery_gate_decision"] == "augment"
        # Should have augmented_edges in the output
        assert "augmented_edges" in result["causal_graph"]

    @pytest.mark.asyncio
    async def test_review_decision_uses_manual_dag(self, node, base_state):
        """Test REVIEW decision uses manual DAG."""
        mock_result = DiscoveryResult(
            success=True,
            config=DiscoveryConfig(),
        )

        mock_evaluation = GateEvaluation(
            decision=GateDecision.REVIEW,
            confidence=0.6,
            reasons=["Medium confidence, needs review"],
            high_confidence_edges=[],
        )

        with patch.object(node, "_run_discovery", new_callable=AsyncMock) as mock_discovery:
            mock_discovery.return_value = (mock_result, mock_evaluation.to_dict())

            result = await node.execute(base_state)

        assert result["causal_graph"]["discovery_gate_decision"] == "review"

    @pytest.mark.asyncio
    async def test_reject_decision_uses_manual_dag(self, node, base_state):
        """Test REJECT decision uses manual DAG."""
        mock_result = DiscoveryResult(
            success=True,
            config=DiscoveryConfig(),
        )

        mock_evaluation = GateEvaluation(
            decision=GateDecision.REJECT,
            confidence=0.3,
            reasons=["Low confidence"],
            high_confidence_edges=[],
        )

        with patch.object(node, "_run_discovery", new_callable=AsyncMock) as mock_discovery:
            mock_discovery.return_value = (mock_result, mock_evaluation.to_dict())

            result = await node.execute(base_state)

        assert result["causal_graph"]["discovery_gate_decision"] == "reject"


class TestGraphBuilderDiscoveryErrorHandling:
    """Test error handling in discovery integration."""

    @pytest.fixture
    def node(self):
        """Create GraphBuilderNode instance."""
        return GraphBuilderNode()

    @pytest.mark.asyncio
    async def test_discovery_exception_falls_back_gracefully(self, node):
        """Test that discovery exceptions result in fallback to manual DAG."""
        state = {
            "query": "Test query",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "auto_discover": True,
            "data_cache": {"data": pd.DataFrame({"a": [1, 2], "b": [3, 4]})},
        }

        with patch.object(node, "_run_discovery", new_callable=AsyncMock) as mock_discovery:
            mock_discovery.side_effect = Exception("Discovery failed!")

            result = await node.execute(state)

        # Should still have a valid result with manual DAG
        assert "causal_graph" in result
        assert result["causal_graph"]["discovery_gate_decision"] is None

    @pytest.mark.asyncio
    async def test_empty_data_handled_gracefully(self, node):
        """Test handling of empty data."""
        state = {
            "query": "Test query",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "auto_discover": True,
            "data_cache": {"data": pd.DataFrame()},  # Empty DataFrame
        }

        # Should not crash
        result = await node.execute(state)

        assert "causal_graph" in result or "error" in str(result).lower()


class TestBuildCausalGraphFunction:
    """Test standalone build_causal_graph function."""

    @pytest.mark.asyncio
    async def test_standalone_function_works(self):
        """Test that standalone function creates node and executes."""
        state = {
            "query": "What is the effect of marketing on conversions?",
            "treatment_var": "marketing_spend",
            "outcome_var": "patient_conversion_rate",
            "auto_discover": False,
        }

        result = await build_causal_graph(state)

        assert "causal_graph" in result
        assert result["causal_graph"]["treatment_nodes"] == ["marketing_spend"]


class TestVariableInference:
    """Test variable inference from query."""

    @pytest.fixture
    def node(self):
        """Create GraphBuilderNode instance."""
        return GraphBuilderNode()

    @pytest.mark.asyncio
    async def test_infer_treatment_from_query(self, node):
        """Test treatment variable inference."""
        state = {
            "query": "How does marketing spend affect prescription volume?",
            "auto_discover": False,
        }

        result = await node.execute(state)

        assert result["causal_graph"]["treatment_nodes"] == ["marketing_spend"]
        assert result["causal_graph"]["outcome_nodes"] == ["prescription_volume"]

    @pytest.mark.asyncio
    async def test_infer_hcp_engagement(self, node):
        """Test HCP engagement inference."""
        state = {
            "query": "What is the impact of HCP engagement on conversion rates?",
            "auto_discover": False,
        }

        result = await node.execute(state)

        assert result["causal_graph"]["treatment_nodes"] == ["hcp_engagement_level"]
        assert result["causal_graph"]["outcome_nodes"] == ["patient_conversion_rate"]

    @pytest.mark.asyncio
    async def test_fallback_to_defaults(self, node):
        """Test fallback to default variables."""
        state = {
            "query": "Some generic query without keywords",
            "auto_discover": False,
        }

        result = await node.execute(state)

        # Should use defaults
        assert result["causal_graph"]["treatment_nodes"] == ["hcp_engagement_level"]
        assert result["causal_graph"]["outcome_nodes"] == ["patient_conversion_rate"]


class TestKnownCausalRelationships:
    """Test integration with known causal relationships."""

    @pytest.fixture
    def node(self):
        """Create GraphBuilderNode instance."""
        return GraphBuilderNode()

    @pytest.mark.asyncio
    async def test_known_relationships_added(self, node):
        """Test that known causal relationships are added to DAG."""
        state = {
            "query": "Test",
            "treatment_var": "marketing_spend",
            "outcome_var": "hcp_engagement_level",
            "auto_discover": False,
        }

        result = await node.execute(state)

        edges = result["causal_graph"]["edges"]

        # marketing_spend -> hcp_engagement_level should be in edges
        assert ("marketing_spend", "hcp_engagement_level") in edges

    @pytest.mark.asyncio
    async def test_confounders_affect_treatment_and_outcome(self, node):
        """Test that confounders are connected properly."""
        state = {
            "query": "Test",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "auto_discover": False,
        }

        result = await node.execute(state)

        edges = result["causal_graph"]["edges"]

        # Confounder should affect both treatment and outcome
        assert ("geographic_region", "hcp_engagement_level") in edges
        assert ("geographic_region", "patient_conversion_rate") in edges
