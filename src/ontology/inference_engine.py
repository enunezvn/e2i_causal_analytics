"""
E2I Causal Analytics - Ontology Inference Engine
Graph-based reasoning engine for causal path discovery and relationship inference
"""

from typing import Dict, List, Set, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
from datetime import datetime
import heapq

logger = logging.getLogger(__name__)


class InferenceType(Enum):
    """Types of inference operations"""
    TRANSITIVE = "transitive"          # A->B, B->C implies A->C
    SYMMETRIC = "symmetric"            # A->B implies B->A
    INVERSE = "inverse"                # A->B implies B->A^-1
    PROPERTY = "property_inheritance"  # Inherit properties along paths
    CAUSAL_CHAIN = "causal_chain"      # Infer causal chains
    SIMILARITY = "similarity"          # Infer similar entities


class ConfidenceLevel(Enum):
    """Confidence levels for inferred relationships"""
    HIGH = "high"          # >0.8
    MEDIUM = "medium"      # 0.5-0.8
    LOW = "low"            # 0.2-0.5
    UNCERTAIN = "uncertain" # <0.2


@dataclass
class InferredRelationship:
    """Inferred relationship between entities"""
    from_id: str
    to_id: str
    relationship_type: str
    confidence: float
    inference_type: InferenceType
    reasoning_path: List[Tuple[str, str, str]]  # [(from, rel, to)]
    supporting_evidence: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence > 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence > 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN


@dataclass
class CausalPath:
    """Discovered causal path through the graph"""
    source_id: str
    target_id: str
    path: List[Tuple[str, str, str]]  # [(from, rel, to)]
    path_length: int
    path_strength: float  # Product of edge weights
    mediated_by: List[str]  # Mediator entity IDs
    confounders: List[str]  # Potential confounder IDs
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceRule:
    """Rule for inferring relationships"""
    name: str
    inference_type: InferenceType
    source_pattern: str  # Cypher-like pattern
    inferred_pattern: str  # Resulting pattern
    confidence_calculator: Callable[[Dict[str, Any]], float]
    bidirectional: bool = False
    max_depth: int = 3


class InferenceEngine:
    """
    Graph-based reasoning engine for the E2I semantic memory.
    Discovers causal paths, infers relationships, and identifies patterns.
    """

    # Default inference rules for E2I causal graphs
    DEFAULT_RULES = [
        {
            'name': 'transitive_causation',
            'type': InferenceType.TRANSITIVE,
            'pattern': '(a)-[:CAUSES]->(b)-[:CAUSES]->(c)',
            'infers': '(a)-[:INDIRECTLY_CAUSES]->(c)',
            'confidence_fn': lambda d: min(d.get('confidence_ab', 1.0),
                                          d.get('confidence_bc', 1.0))
        },
        {
            'name': 'intervention_outcome_chain',
            'type': InferenceType.CAUSAL_CHAIN,
            'pattern': '(i:Intervention)-[:APPLIED_TO]->(p:Patient)-[:HAS_OUTCOME]->(o:Outcome)',
            'infers': '(i)-[:LED_TO]->(o)',
            'confidence_fn': lambda d: 0.7 if d.get('temporal_order') else 0.4
        },
        {
            'name': 'hcp_influence',
            'type': InferenceType.CAUSAL_CHAIN,
            'pattern': '(h:HCP)-[:TREATED]->(p:Patient)-[:RECEIVED]->(i:Intervention)',
            'infers': '(h)-[:INFLUENCED]->(i)',
            'confidence_fn': lambda d: 0.8
        }
    ]

    def __init__(self, graph_client: Any):
        """
        Initialize inference engine

        Args:
            graph_client: FalkorDB client instance
        """
        self.graph = graph_client
        self.rules: List[InferenceRule] = []
        self._load_default_rules()

    def _load_default_rules(self) -> None:
        """Load default E2I inference rules"""
        # Convert default rule dicts to InferenceRule objects
        for rule_def in self.DEFAULT_RULES:
            # Create a simple confidence calculator
            confidence_fn = rule_def['confidence_fn']

            rule = InferenceRule(
                name=rule_def['name'],
                inference_type=rule_def['type'],
                source_pattern=rule_def['pattern'],
                inferred_pattern=rule_def['infers'],
                confidence_calculator=confidence_fn
            )
            self.rules.append(rule)

    def add_rule(self, rule: InferenceRule) -> None:
        """
        Add custom inference rule

        Args:
            rule: InferenceRule to add
        """
        self.rules.append(rule)
        logger.info(f"Added inference rule: {rule.name}")

    def discover_causal_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        relationship_types: Optional[List[str]] = None
    ) -> List[CausalPath]:
        """
        Discover causal paths between source and target entities

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_depth: Maximum path length
            relationship_types: Optional filter for relationship types

        Returns:
            List of discovered CausalPath objects
        """
        logger.info(f"Discovering causal paths: {source_id} -> {target_id}")

        # Build Cypher query for path discovery
        if relationship_types:
            rel_filter = "|".join(relationship_types)
            rel_pattern = f"[:{rel_filter}*1..{max_depth}]"
        else:
            rel_pattern = f"[*1..{max_depth}]"

        query = f"""
        MATCH path = (source {{id: $source_id}})-{rel_pattern}->(target {{id: $target_id}})
        WHERE ALL(r IN relationships(path) WHERE r.causal = true OR NOT exists(r.causal))
        RETURN path,
               length(path) as path_length,
               [n IN nodes(path) | n.id] as node_ids,
               [r IN relationships(path) | type(r)] as rel_types,
               [r IN relationships(path) | r.weight] as weights
        ORDER BY path_length ASC
        LIMIT 100
        """

        result = self.graph.query(query, {
            'source_id': source_id,
            'target_id': target_id
        })

        causal_paths = []
        for record in result.result_set:
            path_length = record[1]
            node_ids = record[2]
            rel_types = record[3]
            weights = record[4] or [1.0] * len(rel_types)

            # Build path tuples
            path = []
            for i in range(len(rel_types)):
                path.append((node_ids[i], rel_types[i], node_ids[i + 1]))

            # Calculate path strength
            path_strength = 1.0
            for weight in weights:
                if weight is not None:
                    path_strength *= weight

            # Identify mediators (nodes in middle of path)
            mediators = node_ids[1:-1] if len(node_ids) > 2 else []

            # Identify potential confounders (would need separate query)
            confounders = []  # TODO: Implement confounder detection

            causal_path = CausalPath(
                source_id=source_id,
                target_id=target_id,
                path=path,
                path_length=path_length,
                path_strength=path_strength,
                mediated_by=mediators,
                confounders=confounders
            )
            causal_paths.append(causal_path)

        logger.info(f"Found {len(causal_paths)} causal paths")
        return causal_paths

    def infer_relationships(
        self,
        max_depth: int = 3,
        min_confidence: float = 0.5
    ) -> List[InferredRelationship]:
        """
        Apply inference rules to discover new relationships

        Args:
            max_depth: Maximum depth for transitive inference
            min_confidence: Minimum confidence threshold

        Returns:
            List of inferred relationships
        """
        logger.info("Starting relationship inference")

        inferred_relationships = []

        for rule in self.rules:
            logger.debug(f"Applying rule: {rule.name}")

            # Apply rule based on type
            if rule.inference_type == InferenceType.TRANSITIVE:
                inferred = self._apply_transitive_rule(rule, max_depth, min_confidence)
                inferred_relationships.extend(inferred)

            elif rule.inference_type == InferenceType.CAUSAL_CHAIN:
                inferred = self._apply_causal_chain_rule(rule, min_confidence)
                inferred_relationships.extend(inferred)

        logger.info(f"Inferred {len(inferred_relationships)} new relationships")
        return inferred_relationships

    def _apply_transitive_rule(
        self,
        rule: InferenceRule,
        max_depth: int,
        min_confidence: float
    ) -> List[InferredRelationship]:
        """Apply transitive closure rule"""
        inferred = []

        # Extract relationship type from rule pattern
        # Simplified: assumes pattern like (a)-[:REL]->(b)-[:REL]->(c)
        # TODO: Proper pattern parsing

        query = f"""
        MATCH path = (a)-[r1:CAUSES]->(b)-[r2:CAUSES]->(c)
        WHERE NOT (a)-[:INDIRECTLY_CAUSES]->(c)
        RETURN a.id as from_id,
               c.id as to_id,
               b.id as mediator,
               r1.confidence as conf1,
               r2.confidence as conf2
        LIMIT 1000
        """

        result = self.graph.query(query)

        for record in result.result_set:
            from_id = record[0]
            to_id = record[1]
            mediator = record[2]
            conf1 = record[3] or 1.0
            conf2 = record[4] or 1.0

            # Calculate confidence
            evidence = {'confidence_ab': conf1, 'confidence_bc': conf2}
            confidence = rule.confidence_calculator(evidence)

            if confidence >= min_confidence:
                inferred_rel = InferredRelationship(
                    from_id=from_id,
                    to_id=to_id,
                    relationship_type='INDIRECTLY_CAUSES',
                    confidence=confidence,
                    inference_type=InferenceType.TRANSITIVE,
                    reasoning_path=[
                        (from_id, 'CAUSES', mediator),
                        (mediator, 'CAUSES', to_id)
                    ],
                    supporting_evidence=evidence
                )
                inferred.append(inferred_rel)

        return inferred

    def _apply_causal_chain_rule(
        self,
        rule: InferenceRule,
        min_confidence: float
    ) -> List[InferredRelationship]:
        """Apply causal chain inference rule"""
        inferred = []

        # Build query from rule pattern
        query = f"""
        MATCH {rule.source_pattern}
        WHERE NOT exists((i)-[:LED_TO]->(o))
        RETURN i.id as from_id,
               o.id as to_id,
               p.id as through_id,
               i.timestamp as i_time,
               o.timestamp as o_time
        LIMIT 1000
        """

        result = self.graph.query(query)

        for record in result.result_set:
            from_id = record[0]
            to_id = record[1]
            through_id = record[2]
            i_time = record[3]
            o_time = record[4]

            # Check temporal ordering
            temporal_order = False
            if i_time and o_time:
                temporal_order = i_time < o_time

            evidence = {'temporal_order': temporal_order}
            confidence = rule.confidence_calculator(evidence)

            if confidence >= min_confidence:
                inferred_rel = InferredRelationship(
                    from_id=from_id,
                    to_id=to_id,
                    relationship_type='LED_TO',
                    confidence=confidence,
                    inference_type=InferenceType.CAUSAL_CHAIN,
                    reasoning_path=[
                        (from_id, 'APPLIED_TO', through_id),
                        (through_id, 'HAS_OUTCOME', to_id)
                    ],
                    supporting_evidence=evidence
                )
                inferred.append(inferred_rel)

        return inferred

    def find_confounders(
        self,
        treatment_id: str,
        outcome_id: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Identify potential confounders between treatment and outcome

        Args:
            treatment_id: Treatment entity ID
            outcome_id: Outcome entity ID
            max_depth: Maximum search depth

        Returns:
            List of potential confounder entities with evidence
        """
        logger.info(f"Finding confounders: {treatment_id} -> {outcome_id}")

        # A confounder affects both treatment and outcome
        query = f"""
        MATCH (t {{id: $treatment_id}}), (o {{id: $outcome_id}})
        MATCH (c)-[r1]->(t)
        MATCH (c)-[r2]->(o)
        WHERE c.id <> t.id AND c.id <> o.id
        RETURN DISTINCT c.id as confounder_id,
               c.label as confounder_label,
               type(r1) as rel_to_treatment,
               type(r2) as rel_to_outcome,
               r1.confidence as conf_treatment,
               r2.confidence as conf_outcome
        """

        result = self.graph.query(query, {
            'treatment_id': treatment_id,
            'outcome_id': outcome_id
        })

        confounders = []
        for record in result.result_set:
            confounder = {
                'id': record[0],
                'label': record[1],
                'relationship_to_treatment': record[2],
                'relationship_to_outcome': record[3],
                'confidence_treatment': record[4] or 0.5,
                'confidence_outcome': record[5] or 0.5,
                'confounder_strength': (record[4] or 0.5) * (record[5] or 0.5)
            }
            confounders.append(confounder)

        # Sort by confounder strength
        confounders.sort(key=lambda x: x['confounder_strength'], reverse=True)

        logger.info(f"Found {len(confounders)} potential confounders")
        return confounders

    def find_mediators(
        self,
        source_id: str,
        target_id: str
    ) -> List[Dict[str, Any]]:
        """
        Identify mediators in causal path from source to target

        Args:
            source_id: Source entity ID
            target_id: Target entity ID

        Returns:
            List of mediator entities with evidence
        """
        logger.info(f"Finding mediators: {source_id} -> {target_id}")

        # Mediator is on the path between source and target
        query = """
        MATCH path = (s {id: $source_id})-[*1..3]->(m)-[*1..3]->(t {id: $target_id})
        WHERE m.id <> s.id AND m.id <> t.id
        RETURN DISTINCT m.id as mediator_id,
               m.label as mediator_label,
               length(path) as total_path_length,
               COUNT(*) as path_count
        ORDER BY path_count DESC
        LIMIT 20
        """

        result = self.graph.query(query, {
            'source_id': source_id,
            'target_id': target_id
        })

        mediators = []
        for record in result.result_set:
            mediator = {
                'id': record[0],
                'label': record[1],
                'path_length': record[2],
                'path_count': record[3],
                'mediation_strength': record[3] / (record[2] or 1)
            }
            mediators.append(mediator)

        logger.info(f"Found {len(mediators)} potential mediators")
        return mediators

    def compute_path_importance(
        self,
        causal_path: CausalPath,
        method: str = 'edge_weight'
    ) -> float:
        """
        Compute importance score for a causal path

        Args:
            causal_path: CausalPath to score
            method: Scoring method ('edge_weight', 'length', 'mediator_count')

        Returns:
            Importance score (0-1)
        """
        if method == 'edge_weight':
            return causal_path.path_strength

        elif method == 'length':
            # Shorter paths are more important
            return 1.0 / (causal_path.path_length + 1)

        elif method == 'mediator_count':
            # Fewer mediators = more direct = more important
            mediator_penalty = len(causal_path.mediated_by) * 0.1
            return max(0.0, 1.0 - mediator_penalty)

        elif method == 'combined':
            # Combine multiple factors
            weight_score = causal_path.path_strength
            length_score = 1.0 / (causal_path.path_length + 1)
            mediator_penalty = len(causal_path.mediated_by) * 0.1

            return (weight_score * 0.5 +
                   length_score * 0.3 +
                   max(0.0, 1.0 - mediator_penalty) * 0.2)

        else:
            return 0.5  # Default

    def rank_causal_paths(
        self,
        causal_paths: List[CausalPath],
        method: str = 'combined'
    ) -> List[Tuple[CausalPath, float]]:
        """
        Rank causal paths by importance

        Args:
            causal_paths: List of CausalPath objects
            method: Ranking method

        Returns:
            List of (CausalPath, score) tuples sorted by score
        """
        scored_paths = []
        for path in causal_paths:
            score = self.compute_path_importance(path, method)
            scored_paths.append((path, score))

        # Sort by score descending
        scored_paths.sort(key=lambda x: x[1], reverse=True)

        return scored_paths

    def materialize_inferred_relationships(
        self,
        inferred_relationships: List[InferredRelationship],
        min_confidence: float = 0.7
    ) -> int:
        """
        Materialize high-confidence inferred relationships into the graph

        Args:
            inferred_relationships: List of inferred relationships
            min_confidence: Minimum confidence to materialize

        Returns:
            Number of relationships materialized
        """
        logger.info(f"Materializing relationships with confidence >= {min_confidence}")

        materialized = 0

        for rel in inferred_relationships:
            if rel.confidence >= min_confidence:
                query = f"""
                MATCH (from {{id: $from_id}}), (to {{id: $to_id}})
                MERGE (from)-[r:{rel.relationship_type}]->(to)
                SET r.confidence = $confidence,
                    r.inferred = true,
                    r.inference_type = $inference_type,
                    r.timestamp = $timestamp
                RETURN r
                """

                self.graph.query(query, {
                    'from_id': rel.from_id,
                    'to_id': rel.to_id,
                    'confidence': rel.confidence,
                    'inference_type': rel.inference_type.value,
                    'timestamp': rel.timestamp.isoformat()
                })

                materialized += 1

        logger.info(f"Materialized {materialized} relationships")
        return materialized


class PathFinder:
    """
    Specialized path-finding algorithms for causal graphs
    """

    @staticmethod
    def shortest_path(
        graph: Any,
        source_id: str,
        target_id: str,
        relationship_types: Optional[List[str]] = None
    ) -> Optional[List[Tuple[str, str, str]]]:
        """
        Find shortest path using BFS

        Args:
            graph: Graph client
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_types: Optional relationship type filter

        Returns:
            Path as list of (from, rel, to) tuples, or None
        """
        if relationship_types:
            rel_filter = "|".join(relationship_types)
            rel_pattern = f"[:{rel_filter}]"
        else:
            rel_pattern = "[]"

        query = f"""
        MATCH path = shortestPath((source {{id: $source_id}})-{rel_pattern}*->(target {{id: $target_id}}))
        RETURN [n IN nodes(path) | n.id] as node_ids,
               [r IN relationships(path) | type(r)] as rel_types
        """

        result = graph.query(query, {
            'source_id': source_id,
            'target_id': target_id
        })

        if not result.result_set:
            return None

        record = result.result_set[0]
        node_ids = record[0]
        rel_types = record[1]

        path = []
        for i in range(len(rel_types)):
            path.append((node_ids[i], rel_types[i], node_ids[i + 1]))

        return path

    @staticmethod
    def all_simple_paths(
        graph: Any,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Find all simple paths (no repeated nodes)

        Args:
            graph: Graph client
            source_id: Source entity ID
            target_id: Target entity ID
            max_depth: Maximum path depth

        Returns:
            List of paths
        """
        query = f"""
        MATCH path = (source {{id: $source_id}})-[*1..{max_depth}]->(target {{id: $target_id}})
        WHERE ALL(n IN nodes(path) WHERE size([x IN nodes(path) WHERE x = n]) = 1)
        RETURN [n IN nodes(path) | n.id] as node_ids,
               [r IN relationships(path) | type(r)] as rel_types
        LIMIT 100
        """

        result = graph.query(query, {
            'source_id': source_id,
            'target_id': target_id
        })

        paths = []
        for record in result.result_set:
            node_ids = record[0]
            rel_types = record[1]

            path = []
            for i in range(len(rel_types)):
                path.append((node_ids[i], rel_types[i], node_ids[i + 1]))

            paths.append(path)

        return paths


# CLI interface
if __name__ == "__main__":
    import sys

    print("E2I Ontology Inference Engine")
    print("=" * 70)
    print()
    print("This module provides:")
    print("  - Causal path discovery")
    print("  - Relationship inference")
    print("  - Confounder detection")
    print("  - Mediator identification")
    print("  - Path ranking and importance scoring")
    print()
    print("Usage: Import and use with FalkorDB client")
    print()

    # Example usage
    print("Example:")
    print("-" * 70)
    print("""
from src.ontology.inference_engine import InferenceEngine
from falkordb import FalkorDB

# Initialize
client = FalkorDB(host='localhost', port=6379)
graph = client.select_graph('e2i_causal')
engine = InferenceEngine(graph)

# Discover causal paths
paths = engine.discover_causal_paths('hcp_123', 'outcome_456', max_depth=5)
print(f"Found {len(paths)} causal paths")

# Rank paths
ranked = engine.rank_causal_paths(paths, method='combined')
best_path, score = ranked[0]
print(f"Best path score: {score:.3f}")

# Infer new relationships
inferred = engine.infer_relationships(min_confidence=0.7)
print(f"Inferred {len(inferred)} new relationships")

# Materialize high-confidence inferences
materialized = engine.materialize_inferred_relationships(inferred, min_confidence=0.8)
print(f"Materialized {materialized} relationships")
    """)
