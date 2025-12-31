"""Tests for V4.4 DAG Validation in Heterogeneous Optimizer.

Tests the segment_analyzer's DAG validation methods:
- _has_dag_evidence
- _validate_segment_effects
- _has_causal_path
- _has_latent_confounder
"""

import pytest

from src.agents.heterogeneous_optimizer.nodes.segment_analyzer import (
    LATENT_CONFOUNDER_WARNING_THRESHOLD,
    SegmentAnalyzerNode,
)


@pytest.fixture
def segment_analyzer():
    """Create segment analyzer instance."""
    return SegmentAnalyzerNode()


@pytest.fixture
def sample_dag_adjacency():
    """Sample DAG: treatment -> segment -> outcome.

    Adjacency matrix (row -> col):
        treatment  segment  outcome
    treatment    0        1        0
    segment      0        0        1
    outcome      0        0        0
    """
    return [
        [0, 1, 0],  # treatment -> segment
        [0, 0, 1],  # segment -> outcome
        [0, 0, 0],  # outcome (no outgoing)
    ]


@pytest.fixture
def sample_dag_nodes():
    """Node names for the sample DAG."""
    return ["treatment", "segment", "outcome"]


@pytest.fixture
def sample_segments():
    """Sample segment results."""
    return [
        {
            "segment_var": "segment",
            "result": {
                "segment_name": "segment",
                "segment_value": "high",
                "cate_estimate": 0.15,
                "cate_ci_lower": 0.10,
                "cate_ci_upper": 0.20,
                "sample_size": 500,
                "statistical_significance": True,
            },
        },
        {
            "segment_var": "segment",
            "result": {
                "segment_name": "segment",
                "segment_value": "low",
                "cate_estimate": 0.05,
                "cate_ci_lower": 0.02,
                "cate_ci_upper": 0.08,
                "sample_size": 500,
                "statistical_significance": True,
            },
        },
    ]


class TestHasDagEvidence:
    """Tests for _has_dag_evidence method."""

    def test_has_dag_evidence_with_valid_dag(
        self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should return True when DAG evidence is valid."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "accept",
        }
        assert segment_analyzer._has_dag_evidence(state) is True

    def test_has_dag_evidence_with_review_decision(
        self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should return True when gate decision is 'review'."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "review",
        }
        assert segment_analyzer._has_dag_evidence(state) is True

    def test_has_dag_evidence_with_reject_decision(
        self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should return False when gate decision is 'reject'."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "reject",
        }
        assert segment_analyzer._has_dag_evidence(state) is False

    def test_has_dag_evidence_without_adjacency(
        self, segment_analyzer, sample_dag_nodes
    ):
        """Should return False when adjacency matrix is missing."""
        state = {
            "discovered_dag_adjacency": None,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "accept",
        }
        assert segment_analyzer._has_dag_evidence(state) is False

    def test_has_dag_evidence_with_empty_adjacency(
        self, segment_analyzer, sample_dag_nodes
    ):
        """Should return False when adjacency matrix is empty."""
        state = {
            "discovered_dag_adjacency": [],
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "accept",
        }
        assert segment_analyzer._has_dag_evidence(state) is False


class TestHasCausalPath:
    """Tests for _has_causal_path method."""

    def test_direct_path_exists(
        self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should find direct path from treatment to segment."""
        node_to_idx = {node: idx for idx, node in enumerate(sample_dag_nodes)}
        result = segment_analyzer._has_causal_path(
            "treatment", "segment", sample_dag_adjacency, node_to_idx
        )
        assert result is True

    def test_indirect_path_exists(
        self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should find indirect path from treatment to outcome via segment."""
        node_to_idx = {node: idx for idx, node in enumerate(sample_dag_nodes)}
        result = segment_analyzer._has_causal_path(
            "treatment", "outcome", sample_dag_adjacency, node_to_idx
        )
        assert result is True

    def test_no_path_reverse(
        self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should not find path in reverse direction."""
        node_to_idx = {node: idx for idx, node in enumerate(sample_dag_nodes)}
        result = segment_analyzer._has_causal_path(
            "outcome", "treatment", sample_dag_adjacency, node_to_idx
        )
        assert result is False

    def test_path_to_self(
        self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should return True for path to self."""
        node_to_idx = {node: idx for idx, node in enumerate(sample_dag_nodes)}
        result = segment_analyzer._has_causal_path(
            "treatment", "treatment", sample_dag_adjacency, node_to_idx
        )
        assert result is True

    def test_node_not_in_dag(
        self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should return False if node not in DAG."""
        node_to_idx = {node: idx for idx, node in enumerate(sample_dag_nodes)}
        result = segment_analyzer._has_causal_path(
            "nonexistent", "outcome", sample_dag_adjacency, node_to_idx
        )
        assert result is False


class TestHasLatentConfounder:
    """Tests for _has_latent_confounder method."""

    def test_no_bidirected_edges(self, segment_analyzer):
        """Should return False when no bidirected edges."""
        edge_types = {
            "treatment->segment": "DIRECTED",
            "segment->outcome": "DIRECTED",
        }
        result = segment_analyzer._has_latent_confounder(
            "segment", "treatment", "outcome", edge_types
        )
        assert result is False

    def test_bidirected_edge_detected(self, segment_analyzer):
        """Should detect bidirected edge between treatment and segment."""
        edge_types = {
            "treatment<->segment": "BIDIRECTED",
            "segment->outcome": "DIRECTED",
        }
        result = segment_analyzer._has_latent_confounder(
            "segment", "treatment", "outcome", edge_types
        )
        assert result is True

    def test_bidirected_edge_alternate_format(self, segment_analyzer):
        """Should detect bidirected edge with value BIDIRECTED on arrow key."""
        edge_types = {
            "treatment->segment": "BIDIRECTED",
            "segment->outcome": "DIRECTED",
        }
        result = segment_analyzer._has_latent_confounder(
            "segment", "treatment", "outcome", edge_types
        )
        assert result is True

    def test_bidirected_to_outcome(self, segment_analyzer):
        """Should detect bidirected edge between segment and outcome."""
        edge_types = {
            "treatment->segment": "DIRECTED",
            "segment<->outcome": "BIDIRECTED",
        }
        result = segment_analyzer._has_latent_confounder(
            "segment", "treatment", "outcome", edge_types
        )
        assert result is True


class TestValidateSegmentEffects:
    """Tests for _validate_segment_effects method."""

    def test_validate_segments_with_causal_path(
        self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes, sample_segments
    ):
        """Should validate segments that have causal paths."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": {},
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "discovery_gate_decision": "accept",
        }

        validated, invalid, latent, warnings = segment_analyzer._validate_segment_effects(
            sample_segments, state
        )

        assert len(validated) == 2  # Both segments validated
        assert len(invalid) == 0
        assert len(latent) == 0

    def test_invalid_segment_no_path(self, segment_analyzer):
        """Should invalidate segment with no causal path."""
        # DAG: treatment -> outcome (no segment connection)
        dag_adjacency = [
            [0, 0, 1],  # treatment -> outcome
            [0, 0, 0],  # segment (isolated)
            [0, 0, 0],  # outcome
        ]
        dag_nodes = ["treatment", "segment", "outcome"]

        segments = [
            {
                "segment_var": "segment",
                "result": {
                    "segment_value": "high",
                    "cate_estimate": 0.15,
                    "sample_size": 500,
                },
            }
        ]

        state = {
            "discovered_dag_adjacency": dag_adjacency,
            "discovered_dag_nodes": dag_nodes,
            "discovered_dag_edge_types": {},
            "treatment_var": "treatment",
            "outcome_var": "outcome",
        }

        validated, invalid, latent, warnings = segment_analyzer._validate_segment_effects(
            segments, state
        )

        assert len(invalid) == 1
        assert "segment_high" in invalid
        assert any("no causal path" in w for w in warnings)

    def test_latent_confounder_detection(
        self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes, sample_segments
    ):
        """Should detect latent confounders from bidirected edges."""
        edge_types = {
            "treatment<->segment": "BIDIRECTED",  # Latent confounder
        }

        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": edge_types,
            "treatment_var": "treatment",
            "outcome_var": "outcome",
        }

        validated, invalid, latent, warnings = segment_analyzer._validate_segment_effects(
            sample_segments, state
        )

        assert len(latent) == 2  # Both segments have latent confounder
        assert any("latent confounder" in w.lower() for w in warnings)

    def test_segment_not_in_dag_warning(self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes):
        """Should warn when segment variable not in DAG."""
        segments = [
            {
                "segment_var": "unknown_var",
                "result": {
                    "segment_value": "high",
                    "cate_estimate": 0.15,
                    "sample_size": 500,
                },
            }
        ]

        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": {},
            "treatment_var": "treatment",
            "outcome_var": "outcome",
        }

        validated, invalid, latent, warnings = segment_analyzer._validate_segment_effects(
            segments, state
        )

        # Segment is validated (not rejected) but warning is added
        assert len(validated) == 1
        assert any("not in discovered DAG" in w for w in warnings)


class TestDagValidationIntegration:
    """Integration tests for DAG validation in segment analysis."""

    @pytest.mark.asyncio
    async def test_execute_with_dag_validation(
        self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should include DAG validation results when DAG is available."""
        cate_by_segment = {
            "segment": [
                {
                    "segment_name": "segment",
                    "segment_value": "high",
                    "cate_estimate": 0.15,
                    "cate_ci_lower": 0.10,
                    "cate_ci_upper": 0.20,
                    "sample_size": 500,
                    "statistical_significance": True,
                },
                {
                    "segment_name": "segment",
                    "segment_value": "low",
                    "cate_estimate": 0.02,
                    "cate_ci_lower": -0.01,
                    "cate_ci_upper": 0.05,
                    "sample_size": 500,
                    "statistical_significance": False,
                },
            ]
        }

        state = {
            "overall_ate": 0.10,
            "cate_by_segment": cate_by_segment,
            "top_segments_count": 10,
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": {},
            "discovery_gate_decision": "accept",
            "discovery_gate_confidence": 0.85,
            "status": "estimating",
        }

        result = await segment_analyzer.execute(state)

        assert result["status"] == "optimizing"
        assert "dag_validated_segments" in result
        assert len(result["dag_validated_segments"]) == 2

    @pytest.mark.asyncio
    async def test_execute_without_dag_skips_validation(self, segment_analyzer):
        """Should skip DAG validation when no DAG evidence."""
        cate_by_segment = {
            "region": [
                {
                    "segment_name": "region",
                    "segment_value": "Northeast",
                    "cate_estimate": 0.12,
                    "cate_ci_lower": 0.08,
                    "cate_ci_upper": 0.16,
                    "sample_size": 300,
                    "statistical_significance": True,
                },
            ]
        }

        state = {
            "overall_ate": 0.10,
            "cate_by_segment": cate_by_segment,
            "top_segments_count": 10,
            "status": "estimating",
        }

        result = await segment_analyzer.execute(state)

        assert result["status"] == "optimizing"
        assert "dag_validated_segments" not in result
        assert "dag_invalid_segments" not in result

    @pytest.mark.asyncio
    async def test_execute_with_reject_decision_skips_validation(
        self, segment_analyzer, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should skip DAG validation when discovery gate rejected."""
        cate_by_segment = {
            "segment": [
                {
                    "segment_name": "segment",
                    "segment_value": "high",
                    "cate_estimate": 0.15,
                    "cate_ci_lower": 0.10,
                    "cate_ci_upper": 0.20,
                    "sample_size": 500,
                    "statistical_significance": True,
                },
            ]
        }

        state = {
            "overall_ate": 0.10,
            "cate_by_segment": cate_by_segment,
            "top_segments_count": 10,
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": {},
            "discovery_gate_decision": "reject",  # Rejected!
            "status": "estimating",
        }

        result = await segment_analyzer.execute(state)

        assert result["status"] == "optimizing"
        assert "dag_validated_segments" not in result
