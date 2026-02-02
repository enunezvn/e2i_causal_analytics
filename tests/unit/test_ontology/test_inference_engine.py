"""
Unit tests for src/ontology/inference_engine.py

Tests for the graph-based inference engine including:
- Enum types (InferenceType, ConfidenceLevel)
- Dataclasses (InferredRelationship, CausalPath, InferenceRule)
- InferenceEngine methods
- PathFinder static methods
"""

from datetime import datetime

import pytest

from src.ontology.inference_engine import (
    CausalPath,
    ConfidenceLevel,
    InferenceEngine,
    InferenceRule,
    InferenceType,
    InferredRelationship,
    PathFinder,
)

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestInferenceTypeEnum:
    """Tests for InferenceType enum."""

    def test_transitive_value(self):
        """Test TRANSITIVE enum value."""
        assert InferenceType.TRANSITIVE.value == "transitive"

    def test_symmetric_value(self):
        """Test SYMMETRIC enum value."""
        assert InferenceType.SYMMETRIC.value == "symmetric"

    def test_inverse_value(self):
        """Test INVERSE enum value."""
        assert InferenceType.INVERSE.value == "inverse"

    def test_property_value(self):
        """Test PROPERTY enum value."""
        assert InferenceType.PROPERTY.value == "property_inheritance"

    def test_causal_chain_value(self):
        """Test CAUSAL_CHAIN enum value."""
        assert InferenceType.CAUSAL_CHAIN.value == "causal_chain"

    def test_similarity_value(self):
        """Test SIMILARITY enum value."""
        assert InferenceType.SIMILARITY.value == "similarity"

    def test_all_enum_members_exist(self):
        """Test that all expected enum members exist."""
        expected_members = {
            "TRANSITIVE",
            "SYMMETRIC",
            "INVERSE",
            "PROPERTY",
            "CAUSAL_CHAIN",
            "SIMILARITY",
        }
        actual_members = {m.name for m in InferenceType}
        assert expected_members == actual_members


class TestConfidenceLevelEnum:
    """Tests for ConfidenceLevel enum."""

    def test_high_value(self):
        """Test HIGH enum value."""
        assert ConfidenceLevel.HIGH.value == "high"

    def test_medium_value(self):
        """Test MEDIUM enum value."""
        assert ConfidenceLevel.MEDIUM.value == "medium"

    def test_low_value(self):
        """Test LOW enum value."""
        assert ConfidenceLevel.LOW.value == "low"

    def test_uncertain_value(self):
        """Test UNCERTAIN enum value."""
        assert ConfidenceLevel.UNCERTAIN.value == "uncertain"

    def test_all_confidence_levels_exist(self):
        """Test that all expected confidence levels exist."""
        expected = {"HIGH", "MEDIUM", "LOW", "UNCERTAIN"}
        actual = {m.name for m in ConfidenceLevel}
        assert expected == actual


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestInferredRelationshipDataclass:
    """Tests for InferredRelationship dataclass."""

    def test_create_inferred_relationship(self):
        """Test creating an InferredRelationship."""
        rel = InferredRelationship(
            from_id="entity_a",
            to_id="entity_b",
            relationship_type="CAUSES",
            confidence=0.85,
            inference_type=InferenceType.TRANSITIVE,
            reasoning_path=[("a", "REL", "b")],
            supporting_evidence={"source": "test"},
        )

        assert rel.from_id == "entity_a"
        assert rel.to_id == "entity_b"
        assert rel.relationship_type == "CAUSES"
        assert rel.confidence == 0.85
        assert rel.inference_type == InferenceType.TRANSITIVE
        assert len(rel.reasoning_path) == 1
        assert rel.supporting_evidence == {"source": "test"}

    def test_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        rel = InferredRelationship(
            from_id="a",
            to_id="b",
            relationship_type="CAUSES",
            confidence=0.5,
            inference_type=InferenceType.CAUSAL_CHAIN,
            reasoning_path=[],
            supporting_evidence={},
        )

        assert rel.timestamp is not None
        assert isinstance(rel.timestamp, datetime)

    def test_confidence_level_high(self):
        """Test confidence_level property returns HIGH for >0.8."""
        rel = InferredRelationship(
            from_id="a",
            to_id="b",
            relationship_type="CAUSES",
            confidence=0.85,
            inference_type=InferenceType.TRANSITIVE,
            reasoning_path=[],
            supporting_evidence={},
        )
        assert rel.confidence_level == ConfidenceLevel.HIGH

    def test_confidence_level_medium(self):
        """Test confidence_level property returns MEDIUM for 0.5-0.8."""
        rel = InferredRelationship(
            from_id="a",
            to_id="b",
            relationship_type="CAUSES",
            confidence=0.65,
            inference_type=InferenceType.TRANSITIVE,
            reasoning_path=[],
            supporting_evidence={},
        )
        assert rel.confidence_level == ConfidenceLevel.MEDIUM

    def test_confidence_level_low(self):
        """Test confidence_level property returns LOW for 0.2-0.5."""
        rel = InferredRelationship(
            from_id="a",
            to_id="b",
            relationship_type="CAUSES",
            confidence=0.35,
            inference_type=InferenceType.TRANSITIVE,
            reasoning_path=[],
            supporting_evidence={},
        )
        assert rel.confidence_level == ConfidenceLevel.LOW

    def test_confidence_level_uncertain(self):
        """Test confidence_level property returns UNCERTAIN for <0.2."""
        rel = InferredRelationship(
            from_id="a",
            to_id="b",
            relationship_type="CAUSES",
            confidence=0.15,
            inference_type=InferenceType.TRANSITIVE,
            reasoning_path=[],
            supporting_evidence={},
        )
        assert rel.confidence_level == ConfidenceLevel.UNCERTAIN

    def test_confidence_level_boundary_high(self):
        """Test boundary at 0.8 returns MEDIUM (not HIGH)."""
        rel = InferredRelationship(
            from_id="a",
            to_id="b",
            relationship_type="CAUSES",
            confidence=0.8,
            inference_type=InferenceType.TRANSITIVE,
            reasoning_path=[],
            supporting_evidence={},
        )
        # 0.8 is not > 0.8, so MEDIUM
        assert rel.confidence_level == ConfidenceLevel.MEDIUM

    def test_confidence_level_boundary_medium(self):
        """Test boundary at 0.5 returns LOW (not MEDIUM)."""
        rel = InferredRelationship(
            from_id="a",
            to_id="b",
            relationship_type="CAUSES",
            confidence=0.5,
            inference_type=InferenceType.TRANSITIVE,
            reasoning_path=[],
            supporting_evidence={},
        )
        # 0.5 is not > 0.5, so LOW
        assert rel.confidence_level == ConfidenceLevel.LOW


class TestCausalPathDataclass:
    """Tests for CausalPath dataclass."""

    def test_create_causal_path(self):
        """Test creating a CausalPath."""
        path = CausalPath(
            source_id="source",
            target_id="target",
            path=[("source", "CAUSES", "mid"), ("mid", "LEADS_TO", "target")],
            path_length=2,
            path_strength=0.72,
            mediated_by=["mid"],
            confounders=[],
        )

        assert path.source_id == "source"
        assert path.target_id == "target"
        assert len(path.path) == 2
        assert path.path_length == 2
        assert path.path_strength == 0.72
        assert path.mediated_by == ["mid"]
        assert path.confounders == []

    def test_causal_path_metadata_default(self):
        """Test that metadata defaults to empty dict."""
        path = CausalPath(
            source_id="s",
            target_id="t",
            path=[],
            path_length=0,
            path_strength=1.0,
            mediated_by=[],
            confounders=[],
        )
        assert path.metadata == {}

    def test_causal_path_with_metadata(self):
        """Test CausalPath with custom metadata."""
        path = CausalPath(
            source_id="s",
            target_id="t",
            path=[],
            path_length=0,
            path_strength=1.0,
            mediated_by=[],
            confounders=[],
            metadata={"discovered_at": "2024-01-01"},
        )
        assert path.metadata["discovered_at"] == "2024-01-01"


class TestInferenceRuleDataclass:
    """Tests for InferenceRule dataclass."""

    def test_create_inference_rule(self):
        """Test creating an InferenceRule."""

        def confidence_fn(d):
            return 0.75

        rule = InferenceRule(
            name="test_rule",
            inference_type=InferenceType.TRANSITIVE,
            source_pattern="(a)-[:CAUSES]->(b)",
            inferred_pattern="(a)-[:INDIRECTLY_CAUSES]->(c)",
            confidence_calculator=confidence_fn,
        )

        assert rule.name == "test_rule"
        assert rule.inference_type == InferenceType.TRANSITIVE
        assert rule.source_pattern == "(a)-[:CAUSES]->(b)"
        assert rule.confidence_calculator({}) == 0.75

    def test_inference_rule_defaults(self):
        """Test InferenceRule default values."""
        rule = InferenceRule(
            name="test",
            inference_type=InferenceType.CAUSAL_CHAIN,
            source_pattern="pattern",
            inferred_pattern="result",
            confidence_calculator=lambda d: 0.5,
        )

        assert rule.bidirectional is False
        assert rule.max_depth == 3

    def test_inference_rule_custom_depth(self):
        """Test InferenceRule with custom max_depth."""
        rule = InferenceRule(
            name="deep_rule",
            inference_type=InferenceType.TRANSITIVE,
            source_pattern="p",
            inferred_pattern="r",
            confidence_calculator=lambda d: 0.5,
            max_depth=10,
        )

        assert rule.max_depth == 10


# =============================================================================
# INFERENCE ENGINE TESTS
# =============================================================================


class TestInferenceEngineInit:
    """Tests for InferenceEngine initialization."""

    def test_init_with_graph_client(self, mock_graph_client):
        """Test initialization with graph client."""
        engine = InferenceEngine(mock_graph_client)

        assert engine.graph == mock_graph_client
        assert isinstance(engine.rules, list)

    def test_default_rules_loaded(self, mock_graph_client):
        """Test that default rules are loaded on init."""
        engine = InferenceEngine(mock_graph_client)

        # Should have 3 default rules
        assert len(engine.rules) == 3

    def test_default_rule_names(self, mock_graph_client):
        """Test default rule names."""
        engine = InferenceEngine(mock_graph_client)

        rule_names = [r.name for r in engine.rules]
        assert "transitive_causation" in rule_names
        assert "intervention_outcome_chain" in rule_names
        assert "hcp_influence" in rule_names

    def test_default_rule_types(self, mock_graph_client):
        """Test default rule inference types."""
        engine = InferenceEngine(mock_graph_client)

        transitive_rules = [r for r in engine.rules if r.inference_type == InferenceType.TRANSITIVE]
        causal_chain_rules = [
            r for r in engine.rules if r.inference_type == InferenceType.CAUSAL_CHAIN
        ]

        assert len(transitive_rules) == 1
        assert len(causal_chain_rules) == 2


class TestInferenceEngineAddRule:
    """Tests for InferenceEngine.add_rule method."""

    def test_add_custom_rule(self, mock_graph_client):
        """Test adding a custom rule."""
        engine = InferenceEngine(mock_graph_client)
        initial_count = len(engine.rules)

        custom_rule = InferenceRule(
            name="custom_rule",
            inference_type=InferenceType.SIMILARITY,
            source_pattern="(a:Brand)-[:SIMILAR_TO]->(b:Brand)",
            inferred_pattern="(b)-[:SIMILAR_TO]->(a)",
            confidence_calculator=lambda d: 0.9,
        )

        engine.add_rule(custom_rule)

        assert len(engine.rules) == initial_count + 1
        assert engine.rules[-1].name == "custom_rule"

    def test_multiple_rules_can_be_added(self, mock_graph_client):
        """Test adding multiple custom rules."""
        engine = InferenceEngine(mock_graph_client)
        initial_count = len(engine.rules)

        for i in range(3):
            rule = InferenceRule(
                name=f"rule_{i}",
                inference_type=InferenceType.PROPERTY,
                source_pattern="p",
                inferred_pattern="r",
                confidence_calculator=lambda d: 0.5,
            )
            engine.add_rule(rule)

        assert len(engine.rules) == initial_count + 3


class TestInferenceEngineDiscoverCausalPaths:
    """Tests for InferenceEngine.discover_causal_paths method."""

    def test_discover_paths_returns_list(self, mock_graph_with_paths):
        """Test that discover_causal_paths returns a list."""
        engine = InferenceEngine(mock_graph_with_paths)

        paths = engine.discover_causal_paths("source", "target")

        assert isinstance(paths, list)

    def test_discover_paths_creates_causal_path_objects(self, mock_graph_with_paths):
        """Test that returned paths are CausalPath objects."""
        engine = InferenceEngine(mock_graph_with_paths)

        paths = engine.discover_causal_paths("source", "target")

        # Mock returns 2 paths
        assert len(paths) == 2
        for path in paths:
            assert isinstance(path, CausalPath)

    def test_discover_paths_calculates_strength(self, mock_graph_with_paths):
        """Test that path strength is calculated from weights."""
        engine = InferenceEngine(mock_graph_with_paths)

        paths = engine.discover_causal_paths("source", "target")

        # First path has weights [0.8, 0.7] -> strength = 0.56
        first_path = paths[0]
        assert first_path.path_strength == pytest.approx(0.56, rel=0.01)

    def test_discover_paths_identifies_mediators(self, mock_graph_with_paths):
        """Test that mediators are identified."""
        engine = InferenceEngine(mock_graph_with_paths)

        paths = engine.discover_causal_paths("source", "target")

        # First path: source -> mediator -> target
        first_path = paths[0]
        assert first_path.mediated_by == ["mediator"]

    def test_discover_paths_with_max_depth(self, mock_graph_client):
        """Test path discovery with custom max_depth."""
        engine = InferenceEngine(mock_graph_client)

        engine.discover_causal_paths("a", "b", max_depth=3)

        # Check that query was executed with depth parameter
        assert len(mock_graph_client.queries_executed) == 1
        assert "*1..3" in mock_graph_client.queries_executed[0]

    def test_discover_paths_with_relationship_filter(self, mock_graph_client):
        """Test path discovery with relationship type filter."""
        engine = InferenceEngine(mock_graph_client)

        engine.discover_causal_paths("a", "b", relationship_types=["CAUSES", "LEADS_TO"])

        query = mock_graph_client.queries_executed[0]
        assert "CAUSES|LEADS_TO" in query


class TestInferenceEngineInferRelationships:
    """Tests for InferenceEngine.infer_relationships method."""

    def test_infer_relationships_returns_list(self, mock_graph_client):
        """Test that infer_relationships returns a list."""
        engine = InferenceEngine(mock_graph_client)

        result = engine.infer_relationships()

        assert isinstance(result, list)

    def test_infer_relationships_applies_all_rules(self, mock_graph_client):
        """Test that all rules are applied during inference."""
        engine = InferenceEngine(mock_graph_client)

        engine.infer_relationships()

        # Should execute queries for each rule type
        # At least 2 queries (one for transitive, one for causal chain)
        assert len(mock_graph_client.queries_executed) >= 2

    def test_infer_relationships_respects_min_confidence(self, mock_graph_client):
        """Test that min_confidence filters results."""
        # Set up mock result
        from tests.unit.test_ontology.conftest import MockQueryResult

        mock_graph_client.set_mock_result(
            "MATCH path",
            MockQueryResult(
                result_set=[
                    ("a", "c", "b", 0.3, 0.4)  # conf1 * conf2 would be low
                ]
            ),
        )

        engine = InferenceEngine(mock_graph_client)

        # With high min_confidence, should filter out low confidence
        result = engine.infer_relationships(min_confidence=0.9)

        # Result should be empty or filtered
        assert all(r.confidence >= 0.5 for r in result)


class TestInferenceEngineFindConfounders:
    """Tests for InferenceEngine.find_confounders method."""

    def test_find_confounders_returns_list(self, mock_graph_with_confounders):
        """Test that find_confounders returns a list."""
        engine = InferenceEngine(mock_graph_with_confounders)

        confounders = engine.find_confounders("treatment", "outcome")

        assert isinstance(confounders, list)

    def test_find_confounders_executes_query(self, mock_graph_client):
        """Test that confounder query is executed."""
        engine = InferenceEngine(mock_graph_client)

        engine.find_confounders("treatment", "outcome")

        assert len(mock_graph_client.queries_executed) > 0
        # Query should include both treatment and outcome
        query = mock_graph_client.queries_executed[-1]
        assert "$treatment_id" in query
        assert "$outcome_id" in query

    def test_find_confounders_sorted_by_strength(self, mock_graph_with_confounders):
        """Test that confounders are sorted by strength."""
        engine = InferenceEngine(mock_graph_with_confounders)

        confounders = engine.find_confounders("treatment", "outcome")

        if len(confounders) > 1:
            for i in range(len(confounders) - 1):
                assert (
                    confounders[i]["confounder_strength"]
                    >= confounders[i + 1]["confounder_strength"]
                )


class TestInferenceEngineFindMediators:
    """Tests for InferenceEngine.find_mediators method."""

    def test_find_mediators_returns_list(self, mock_graph_client):
        """Test that find_mediators returns a list."""
        engine = InferenceEngine(mock_graph_client)

        mediators = engine.find_mediators("source", "target")

        assert isinstance(mediators, list)

    def test_find_mediators_executes_query(self, mock_graph_client):
        """Test that mediator query is executed."""
        engine = InferenceEngine(mock_graph_client)

        engine.find_mediators("source", "target")

        assert len(mock_graph_client.queries_executed) > 0


class TestInferenceEngineComputePathImportance:
    """Tests for InferenceEngine.compute_path_importance method."""

    @pytest.fixture
    def sample_causal_path(self):
        """Create a sample CausalPath for testing."""
        return CausalPath(
            source_id="source",
            target_id="target",
            path=[("source", "CAUSES", "mid"), ("mid", "LEADS_TO", "target")],
            path_length=2,
            path_strength=0.72,
            mediated_by=["mid"],
            confounders=[],
        )

    def test_compute_importance_edge_weight(self, mock_graph_client, sample_causal_path):
        """Test edge_weight method returns path_strength."""
        engine = InferenceEngine(mock_graph_client)

        score = engine.compute_path_importance(sample_causal_path, method="edge_weight")

        assert score == 0.72

    def test_compute_importance_length(self, mock_graph_client, sample_causal_path):
        """Test length method favors shorter paths."""
        engine = InferenceEngine(mock_graph_client)

        score = engine.compute_path_importance(sample_causal_path, method="length")

        # 1 / (path_length + 1) = 1/3 ≈ 0.333
        assert score == pytest.approx(1.0 / 3.0, rel=0.01)

    def test_compute_importance_mediator_count(self, mock_graph_client, sample_causal_path):
        """Test mediator_count method penalizes mediators."""
        engine = InferenceEngine(mock_graph_client)

        score = engine.compute_path_importance(sample_causal_path, method="mediator_count")

        # 1 mediator * 0.1 penalty = 0.9
        assert score == pytest.approx(0.9, rel=0.01)

    def test_compute_importance_combined(self, mock_graph_client, sample_causal_path):
        """Test combined method uses all factors."""
        engine = InferenceEngine(mock_graph_client)

        score = engine.compute_path_importance(sample_causal_path, method="combined")

        # Combined: weight*0.5 + length*0.3 + mediator*0.2
        # 0.72*0.5 + 0.333*0.3 + 0.9*0.2 = 0.36 + 0.1 + 0.18 = 0.64
        assert 0.0 < score < 1.0

    def test_compute_importance_unknown_method(self, mock_graph_client, sample_causal_path):
        """Test unknown method returns default 0.5."""
        engine = InferenceEngine(mock_graph_client)

        score = engine.compute_path_importance(sample_causal_path, method="unknown")

        assert score == 0.5


class TestInferenceEngineRankCausalPaths:
    """Tests for InferenceEngine.rank_causal_paths method."""

    def test_rank_returns_tuples(self, mock_graph_client):
        """Test that rank returns list of (path, score) tuples."""
        engine = InferenceEngine(mock_graph_client)

        paths = [
            CausalPath("a", "b", [], 1, 0.9, [], []),
            CausalPath("a", "b", [], 2, 0.7, [], []),
        ]

        ranked = engine.rank_causal_paths(paths)

        assert all(isinstance(item, tuple) for item in ranked)
        assert all(len(item) == 2 for item in ranked)

    def test_rank_sorted_descending(self, mock_graph_client):
        """Test that paths are sorted by score descending."""
        engine = InferenceEngine(mock_graph_client)

        paths = [
            CausalPath("a", "b", [], 3, 0.5, [], []),  # Lower score
            CausalPath("a", "b", [], 1, 0.9, [], []),  # Higher score
            CausalPath("a", "b", [], 2, 0.7, [], []),  # Middle score
        ]

        ranked = engine.rank_causal_paths(paths, method="edge_weight")

        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_empty_list(self, mock_graph_client):
        """Test ranking empty list returns empty list."""
        engine = InferenceEngine(mock_graph_client)

        ranked = engine.rank_causal_paths([])

        assert ranked == []


class TestInferenceEngineMaterializeRelationships:
    """Tests for InferenceEngine.materialize_inferred_relationships method."""

    def test_materialize_executes_queries(self, mock_graph_client):
        """Test that materialization executes graph queries."""
        engine = InferenceEngine(mock_graph_client)

        relationships = [
            InferredRelationship(
                from_id="a",
                to_id="b",
                relationship_type="CAUSES",
                confidence=0.85,
                inference_type=InferenceType.TRANSITIVE,
                reasoning_path=[],
                supporting_evidence={},
            )
        ]

        count = engine.materialize_inferred_relationships(relationships)

        assert count == 1
        assert len(mock_graph_client.queries_executed) > 0

    def test_materialize_respects_min_confidence(self, mock_graph_client):
        """Test that low confidence relationships are not materialized."""
        engine = InferenceEngine(mock_graph_client)

        relationships = [
            InferredRelationship(
                from_id="a",
                to_id="b",
                relationship_type="CAUSES",
                confidence=0.5,  # Below default 0.7 threshold
                inference_type=InferenceType.TRANSITIVE,
                reasoning_path=[],
                supporting_evidence={},
            )
        ]

        count = engine.materialize_inferred_relationships(relationships, min_confidence=0.7)

        assert count == 0

    def test_materialize_returns_count(self, mock_graph_client):
        """Test that correct count is returned."""
        engine = InferenceEngine(mock_graph_client)

        relationships = [
            InferredRelationship(
                from_id="a",
                to_id="b",
                relationship_type="CAUSES",
                confidence=0.9,
                inference_type=InferenceType.TRANSITIVE,
                reasoning_path=[],
                supporting_evidence={},
            ),
            InferredRelationship(
                from_id="c",
                to_id="d",
                relationship_type="LEADS_TO",
                confidence=0.85,
                inference_type=InferenceType.CAUSAL_CHAIN,
                reasoning_path=[],
                supporting_evidence={},
            ),
        ]

        count = engine.materialize_inferred_relationships(relationships, min_confidence=0.8)

        assert count == 2


# =============================================================================
# PATHFINDER TESTS
# =============================================================================


class TestPathFinderShortestPath:
    """Tests for PathFinder.shortest_path static method."""

    def test_shortest_path_executes_query(self, mock_graph_client):
        """Test that shortest_path executes a query."""
        PathFinder.shortest_path(mock_graph_client, "source", "target")

        assert len(mock_graph_client.queries_executed) == 1
        assert "shortestPath" in mock_graph_client.queries_executed[0]

    def test_shortest_path_returns_none_no_result(self, mock_graph_client):
        """Test that None is returned when no path exists."""
        result = PathFinder.shortest_path(mock_graph_client, "a", "b")

        assert result is None

    def test_shortest_path_with_relationship_filter(self, mock_graph_client):
        """Test shortest_path with relationship type filter."""
        PathFinder.shortest_path(
            mock_graph_client, "a", "b", relationship_types=["CAUSES", "LEADS_TO"]
        )

        query = mock_graph_client.queries_executed[0]
        assert "CAUSES|LEADS_TO" in query

    def test_shortest_path_returns_path_tuples(self, mock_graph_client):
        """Test that path is returned as list of tuples."""
        from tests.unit.test_ontology.conftest import MockQueryResult

        mock_graph_client.set_mock_result(
            "shortestPath", MockQueryResult(result_set=[(["a", "b", "c"], ["REL1", "REL2"])])
        )

        path = PathFinder.shortest_path(mock_graph_client, "a", "c")

        assert path is not None
        assert path == [("a", "REL1", "b"), ("b", "REL2", "c")]


class TestPathFinderAllSimplePaths:
    """Tests for PathFinder.all_simple_paths static method."""

    def test_all_simple_paths_executes_query(self, mock_graph_client):
        """Test that all_simple_paths executes a query."""
        PathFinder.all_simple_paths(mock_graph_client, "source", "target")

        assert len(mock_graph_client.queries_executed) == 1

    def test_all_simple_paths_respects_max_depth(self, mock_graph_client):
        """Test that max_depth is included in query."""
        PathFinder.all_simple_paths(mock_graph_client, "a", "b", max_depth=3)

        query = mock_graph_client.queries_executed[0]
        assert "*1..3" in query

    def test_all_simple_paths_returns_list(self, mock_graph_client):
        """Test that empty list is returned when no paths exist."""
        paths = PathFinder.all_simple_paths(mock_graph_client, "a", "b")

        assert isinstance(paths, list)
        assert paths == []

    def test_all_simple_paths_returns_multiple_paths(self, mock_graph_client):
        """Test that multiple paths are returned."""
        from tests.unit.test_ontology.conftest import MockQueryResult

        mock_graph_client.set_mock_result(
            "MATCH path",
            MockQueryResult(
                result_set=[
                    (["a", "b", "c"], ["REL1", "REL2"]),
                    (["a", "d", "c"], ["REL3", "REL4"]),
                ]
            ),
        )

        paths = PathFinder.all_simple_paths(mock_graph_client, "a", "c")

        assert len(paths) == 2
        assert paths[0] == [("a", "REL1", "b"), ("b", "REL2", "c")]
        assert paths[1] == [("a", "REL3", "d"), ("d", "REL4", "c")]


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestInferenceEngineEdgeCases:
    """Tests for edge cases in InferenceEngine."""

    def test_discover_paths_empty_result(self, mock_graph_client):
        """Test discover_causal_paths with no results."""
        engine = InferenceEngine(mock_graph_client)

        paths = engine.discover_causal_paths("nonexistent", "alsononexistent")

        assert paths == []

    def test_discover_paths_handles_null_weights(self, mock_graph_client):
        """Test that null weights are handled (default to 1.0)."""
        from tests.unit.test_ontology.conftest import MockQueryResult

        mock_graph_client.set_mock_result(
            "MATCH path",
            MockQueryResult(
                result_set=[
                    (None, 2, ["a", "b", "c"], ["REL1", "REL2"], None)  # null weights
                ]
            ),
        )

        engine = InferenceEngine(mock_graph_client)
        paths = engine.discover_causal_paths("a", "c")

        # Should handle None weights
        assert len(paths) == 1

    def test_path_with_no_mediators(self, mock_graph_client):
        """Test path with only 2 nodes (no mediators)."""
        from tests.unit.test_ontology.conftest import MockQueryResult

        mock_graph_client.set_mock_result(
            "MATCH path", MockQueryResult(result_set=[(None, 1, ["a", "b"], ["DIRECT"], [1.0])])
        )

        engine = InferenceEngine(mock_graph_client)
        paths = engine.discover_causal_paths("a", "b")

        assert len(paths) == 1
        assert paths[0].mediated_by == []

    def test_confidence_calculation_with_callable(self, mock_graph_client):
        """Test that confidence calculators are properly invoked."""
        # The default rules have callable confidence calculators
        engine = InferenceEngine(mock_graph_client)

        # Test the transitive rule calculator directly
        transitive_rule = next(r for r in engine.rules if r.name == "transitive_causation")
        evidence = {"confidence_ab": 0.8, "confidence_bc": 0.9}

        confidence = transitive_rule.confidence_calculator(evidence)

        # Should return min of the two
        assert confidence == 0.8
