"""
E2I Causal Analytics - Discovery Gate
=====================================

Gating logic for causal discovery results.

The DiscoveryGate evaluates discovery results and makes decisions about
whether to accept, review, reject, or augment the discovered structure.

Gate Decisions:
- ACCEPT: High confidence, use discovered DAG directly
- REVIEW: Medium confidence, requires expert validation
- REJECT: Low confidence, use manual DAG instead
- AUGMENT: Supplement manual DAG with high-confidence edges

Author: E2I Causal Analytics Team
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from loguru import logger

from .base import DiscoveredEdge, DiscoveryResult, GateDecision


@dataclass
class GateConfig:
    """Configuration for DiscoveryGate thresholds.

    Attributes:
        accept_threshold: Minimum overall confidence for ACCEPT
        review_threshold: Minimum overall confidence for REVIEW (below = REJECT)
        augment_edge_threshold: Minimum edge confidence for AUGMENT
        min_algorithm_agreement: Minimum fraction of algorithms that must agree
        max_rejected_edges_fraction: Max fraction of edges to reject before REJECT
        min_edges: Minimum edges required (below = REJECT)
        require_dag: Whether to require acyclic graph
    """

    accept_threshold: float = 0.8
    review_threshold: float = 0.5
    augment_edge_threshold: float = 0.9
    min_algorithm_agreement: float = 0.5
    max_rejected_edges_fraction: float = 0.3
    min_edges: int = 1
    require_dag: bool = True


@dataclass
class GateEvaluation:
    """Result of gate evaluation.

    Attributes:
        decision: Final gate decision
        confidence: Overall confidence score
        reasons: List of reasons for decision
        high_confidence_edges: Edges with confidence >= augment threshold
        rejected_edges: Edges rejected due to low confidence
        warnings: Any warnings about the discovery
        metadata: Additional evaluation metadata
    """

    decision: GateDecision
    confidence: float
    reasons: List[str] = field(default_factory=list)
    high_confidence_edges: List[DiscoveredEdge] = field(default_factory=list)
    rejected_edges: List[DiscoveredEdge] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "n_high_confidence_edges": len(self.high_confidence_edges),
            "n_rejected_edges": len(self.rejected_edges),
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class DiscoveryGate:
    """Evaluates discovery results and makes gating decisions.

    The gate uses multiple criteria to determine confidence:
    1. Algorithm agreement: How many algorithms found the same edges
    2. Edge confidence: Average confidence across edges
    3. Graph structure: Whether result is a valid DAG
    4. Coverage: Whether important variables are connected

    Example:
        >>> gate = DiscoveryGate()
        >>> evaluation = gate.evaluate(discovery_result)
        >>> if evaluation.decision == GateDecision.ACCEPT:
        ...     use_discovered_dag(discovery_result.ensemble_dag)
        >>> elif evaluation.decision == GateDecision.AUGMENT:
        ...     augment_manual_dag(evaluation.high_confidence_edges)
    """

    def __init__(self, config: Optional[GateConfig] = None):
        """Initialize DiscoveryGate.

        Args:
            config: Gate configuration. Uses defaults if None.
        """
        self.config = config or GateConfig()

    def evaluate(
        self,
        result: DiscoveryResult,
        expected_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> GateEvaluation:
        """Evaluate discovery result and make gating decision.

        Args:
            result: Discovery result to evaluate
            expected_edges: Optional list of expected edges for validation

        Returns:
            GateEvaluation with decision and reasoning
        """
        reasons = []
        warnings = []

        # Check for failed discovery
        if not result.success:
            return GateEvaluation(
                decision=GateDecision.REJECT,
                confidence=0.0,
                reasons=["Discovery failed"],
                metadata={"error": result.metadata.get("error", "Unknown error")},
            )

        # Check minimum edges
        if result.n_edges < self.config.min_edges:
            return GateEvaluation(
                decision=GateDecision.REJECT,
                confidence=0.0,
                reasons=[
                    f"Too few edges discovered: {result.n_edges} < {self.config.min_edges}"
                ],
            )

        # Calculate component scores
        agreement_score = self._calculate_agreement_score(result)
        edge_confidence_score = self._calculate_edge_confidence(result)
        structure_score = self._calculate_structure_score(result)

        # Overall confidence is weighted average
        confidence = (
            0.4 * agreement_score + 0.4 * edge_confidence_score + 0.2 * structure_score
        )

        # Categorize edges
        high_conf_edges = [
            e for e in result.edges if e.confidence >= self.config.augment_edge_threshold
        ]
        low_conf_edges = [
            e for e in result.edges if e.confidence < self.config.review_threshold
        ]

        # Calculate rejected fraction
        rejected_fraction = len(low_conf_edges) / len(result.edges) if result.edges else 0

        # Build reasons
        reasons.append(f"Algorithm agreement: {agreement_score:.2%}")
        reasons.append(f"Average edge confidence: {edge_confidence_score:.2%}")
        reasons.append(f"Structure score: {structure_score:.2%}")

        if rejected_fraction > self.config.max_rejected_edges_fraction:
            warnings.append(
                f"High fraction of low-confidence edges: {rejected_fraction:.1%}"
            )

        # Validate against expected edges if provided
        if expected_edges:
            recall, precision = self._calculate_edge_metrics(result, expected_edges)
            reasons.append(f"Edge recall: {recall:.2%}, precision: {precision:.2%}")
            if recall < 0.5:
                warnings.append(f"Low recall of expected edges: {recall:.1%}")

        # Make decision
        if confidence >= self.config.accept_threshold:
            decision = GateDecision.ACCEPT
            reasons.append("High confidence - accepting discovered structure")
        elif confidence >= self.config.review_threshold:
            if len(high_conf_edges) >= self.config.min_edges:
                decision = GateDecision.AUGMENT
                reasons.append(
                    f"Medium confidence but {len(high_conf_edges)} high-confidence edges available for augmentation"
                )
            else:
                decision = GateDecision.REVIEW
                reasons.append("Medium confidence - expert review recommended")
        else:
            if len(high_conf_edges) >= self.config.min_edges:
                decision = GateDecision.AUGMENT
                reasons.append(
                    f"Low overall confidence but {len(high_conf_edges)} high-confidence edges available"
                )
            else:
                decision = GateDecision.REJECT
                reasons.append("Low confidence - recommend using manual DAG")

        logger.info(
            f"Gate decision: {decision.value} (confidence: {confidence:.2%}, "
            f"edges: {result.n_edges}, high-conf: {len(high_conf_edges)})"
        )

        return GateEvaluation(
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            high_confidence_edges=high_conf_edges,
            rejected_edges=low_conf_edges,
            warnings=warnings,
            metadata={
                "agreement_score": agreement_score,
                "edge_confidence_score": edge_confidence_score,
                "structure_score": structure_score,
                "rejected_fraction": rejected_fraction,
            },
        )

    def _calculate_agreement_score(self, result: DiscoveryResult) -> float:
        """Calculate algorithm agreement score.

        Higher score when algorithms agree on more edges.

        Args:
            result: Discovery result

        Returns:
            Agreement score [0, 1]
        """
        if not result.edges or not result.algorithm_results:
            return 0.0

        n_algorithms = len(
            [r for r in result.algorithm_results if r.converged]
        )
        if n_algorithms == 0:
            return 0.0

        # Average votes per edge normalized by number of algorithms
        total_agreement = sum(e.algorithm_votes / n_algorithms for e in result.edges)
        return total_agreement / len(result.edges)

    def _calculate_edge_confidence(self, result: DiscoveryResult) -> float:
        """Calculate average edge confidence.

        Args:
            result: Discovery result

        Returns:
            Average confidence [0, 1]
        """
        if not result.edges:
            return 0.0

        return sum(e.confidence for e in result.edges) / len(result.edges)

    def _calculate_structure_score(self, result: DiscoveryResult) -> float:
        """Calculate graph structure score.

        Checks for valid DAG properties.

        Args:
            result: Discovery result

        Returns:
            Structure score [0, 1]
        """
        if result.ensemble_dag is None:
            return 0.0

        score = 1.0

        # Check if DAG is connected (weakly)
        if not nx.is_weakly_connected(result.ensemble_dag):
            # Penalize disconnected components
            n_components = nx.number_weakly_connected_components(result.ensemble_dag)
            score *= 0.8 ** (n_components - 1)

        # Check for isolated nodes
        isolated = list(nx.isolates(result.ensemble_dag))
        if isolated:
            isolated_fraction = len(isolated) / result.ensemble_dag.number_of_nodes()
            score *= 1 - isolated_fraction * 0.5

        # Bonus for having root nodes (potential treatments)
        roots = [n for n in result.ensemble_dag.nodes() if result.ensemble_dag.in_degree(n) == 0]
        if roots:
            score *= 1.1  # Small bonus

        return min(score, 1.0)

    def _calculate_edge_metrics(
        self,
        result: DiscoveryResult,
        expected_edges: List[Tuple[str, str]],
    ) -> Tuple[float, float]:
        """Calculate recall and precision against expected edges.

        Args:
            result: Discovery result
            expected_edges: Expected edges

        Returns:
            Tuple of (recall, precision)
        """
        expected_set = set(expected_edges)
        discovered_set = {(e.source, e.target) for e in result.edges}

        true_positives = len(expected_set & discovered_set)
        recall = true_positives / len(expected_set) if expected_set else 0.0
        precision = true_positives / len(discovered_set) if discovered_set else 0.0

        return recall, precision

    def should_accept(self, result: DiscoveryResult) -> bool:
        """Quick check if result should be accepted.

        Args:
            result: Discovery result

        Returns:
            True if should accept
        """
        evaluation = self.evaluate(result)
        return evaluation.decision == GateDecision.ACCEPT

    def get_augmentation_edges(
        self,
        result: DiscoveryResult,
        manual_dag: nx.DiGraph,
    ) -> List[DiscoveredEdge]:
        """Get edges to add to manual DAG.

        Returns high-confidence edges that are not already in the manual DAG.

        Args:
            result: Discovery result
            manual_dag: Manual DAG to augment

        Returns:
            List of edges to add
        """
        evaluation = self.evaluate(result)

        if evaluation.decision not in [GateDecision.ACCEPT, GateDecision.AUGMENT]:
            return []

        # Filter edges not already in manual DAG
        existing_edges = set(manual_dag.edges())
        new_edges = [
            e
            for e in evaluation.high_confidence_edges
            if (e.source, e.target) not in existing_edges
        ]

        # Check that adding edges won't create cycles
        valid_edges = []
        temp_dag = manual_dag.copy()

        for edge in new_edges:
            temp_dag.add_edge(edge.source, edge.target)
            if nx.is_directed_acyclic_graph(temp_dag):
                valid_edges.append(edge)
            else:
                temp_dag.remove_edge(edge.source, edge.target)
                logger.warning(
                    f"Skipping edge {edge.source} -> {edge.target}: would create cycle"
                )

        return valid_edges
