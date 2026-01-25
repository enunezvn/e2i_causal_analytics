"""
E2I Feedback Learner Agent - Discovery Feedback Node
Version: 4.4
Purpose: Process feedback specific to causal discovery results

This node handles the discovery feedback loop by:
1. Collecting feedback on causal discovery results (DAGs)
2. Tracking accuracy by algorithm
3. Generating parameter recommendations
4. Updating discovery configuration stores

Discovery feedback types:
- user_correction: User corrects discovered edges
- expert_review: Domain expert validates/rejects DAG
- outcome_validation: Observed outcomes vs predicted
- gate_override: User overrides gate decision
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..state import (
    DiscoveryAccuracyTracking,
    DiscoveryFeedbackItem,
    DiscoveryParameterRecommendation,
    FeedbackLearnerState,
)

logger = logging.getLogger(__name__)


class DiscoveryFeedbackNode:
    """
    Node for processing discovery-specific feedback.

    Responsibilities:
    - Collect and aggregate discovery feedback
    - Track accuracy metrics by algorithm
    - Detect patterns in discovery failures
    - Generate parameter recommendations for algorithms
    """

    def __init__(
        self,
        discovery_store: Optional[Any] = None,
        min_runs_for_recommendation: int = 10,
        accuracy_threshold: float = 0.7,
    ):
        """
        Initialize discovery feedback node.

        Args:
            discovery_store: Store for discovery feedback data
            min_runs_for_recommendation: Minimum runs before recommending changes
            accuracy_threshold: Accuracy below this triggers recommendations
        """
        self._discovery_store = discovery_store
        self._min_runs = min_runs_for_recommendation
        self._accuracy_threshold = accuracy_threshold

    async def execute(self, state: FeedbackLearnerState) -> FeedbackLearnerState:
        """
        Process discovery feedback from state.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with discovery analysis
        """
        start_time = time.time()

        # Check if there's discovery feedback to process
        discovery_items = state.get("discovery_feedback_items") or []
        if not discovery_items:
            logger.debug("No discovery feedback to process")
            return state

        logger.info(f"Processing {len(discovery_items)} discovery feedback items")

        try:
            # Track accuracy by algorithm
            accuracy_tracking = self._compute_accuracy_tracking(discovery_items)

            # Generate parameter recommendations
            recommendations = self._generate_recommendations(
                discovery_items, accuracy_tracking
            )

            # Update patterns with discovery-specific issues
            existing_patterns = state.get("detected_patterns") or []
            discovery_patterns = self._detect_discovery_patterns(
                discovery_items, accuracy_tracking
            )
            all_patterns = existing_patterns + discovery_patterns

            # Update state
            latency_ms = int((time.time() - start_time) * 1000)

            return {
                **state,
                "discovery_accuracy_tracking": accuracy_tracking,
                "discovery_parameter_recommendations": recommendations,
                "detected_patterns": all_patterns,
                "analysis_latency_ms": state.get("analysis_latency_ms", 0) + latency_ms,
                "warnings": state.get("warnings", [])
                + (
                    [f"Discovery accuracy issues detected for {len(recommendations)} algorithms"]
                    if recommendations
                    else []
                ),
            }

        except Exception as e:
            logger.error(f"Error processing discovery feedback: {e}")
            return {
                **state,
                "errors": state.get("errors", [])
                + [
                    {
                        "node": "discovery_feedback",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            }

    def _compute_accuracy_tracking(
        self, items: List[DiscoveryFeedbackItem]
    ) -> Dict[str, DiscoveryAccuracyTracking]:
        """
        Compute accuracy metrics by algorithm.

        Args:
            items: Discovery feedback items

        Returns:
            Dictionary mapping algorithm to accuracy tracking
        """
        # Group by algorithm
        by_algorithm: Dict[str, List[DiscoveryFeedbackItem]] = defaultdict(list)
        for item in items:
            algorithm = item.get("algorithm_used", "unknown")
            by_algorithm[algorithm].append(item)

        tracking = {}
        for algorithm, algo_items in by_algorithm.items():
            total = len(algo_items)
            accepted = sum(
                1
                for item in algo_items
                if item.get("user_decision") == "accept"
                or (
                    item.get("user_decision") is None
                    and item.get("original_gate_decision") == "accept"
                )
            )
            rejected = sum(
                1
                for item in algo_items
                if item.get("user_decision") == "reject"
            )
            modified = sum(
                1
                for item in algo_items
                if item.get("user_decision") == "modify"
                or item.get("edge_corrections")
            )

            # Calculate accuracy scores
            accuracy_scores = [
                item.get("accuracy_score", 0.0)
                for item in algo_items
                if item.get("accuracy_score") is not None
            ]
            avg_accuracy = (
                sum(accuracy_scores) / len(accuracy_scores)
                if accuracy_scores
                else 0.0
            )

            # Calculate edge precision/recall from corrections
            total_predicted_edges = 0
            correct_edges = 0
            total_true_edges = 0

            for item in algo_items:
                corrections = item.get("edge_corrections") or []
                dag = item.get("dag_adjacency") or {}

                # Count predicted edges
                for source, targets in dag.items():
                    total_predicted_edges += len(targets)

                # Process corrections
                for correction in corrections:
                    if correction.get("action") == "remove":
                        # False positive
                        pass
                    elif correction.get("action") == "add":
                        # False negative
                        total_true_edges += 1
                    elif correction.get("action") == "confirm":
                        correct_edges += 1
                        total_true_edges += 1

                # If no corrections, assume all edges correct
                if not corrections and item.get("user_decision") == "accept":
                    for source, targets in dag.items():
                        correct_edges += len(targets)
                        total_true_edges += len(targets)

            precision = (
                correct_edges / total_predicted_edges
                if total_predicted_edges > 0
                else 0.0
            )
            recall = (
                correct_edges / total_true_edges if total_true_edges > 0 else 0.0
            )

            tracking[algorithm] = DiscoveryAccuracyTracking(
                algorithm=algorithm,
                total_runs=total,
                accepted_runs=accepted,
                rejected_runs=rejected,
                modified_runs=modified,
                average_accuracy=avg_accuracy,
                edge_precision=precision,
                edge_recall=recall,
            )

        return tracking

    def _generate_recommendations(
        self,
        items: List[DiscoveryFeedbackItem],
        accuracy_tracking: Dict[str, DiscoveryAccuracyTracking],
    ) -> List[DiscoveryParameterRecommendation]:
        """
        Generate parameter recommendations for underperforming algorithms.

        Args:
            items: Discovery feedback items
            accuracy_tracking: Accuracy metrics by algorithm

        Returns:
            List of parameter recommendations
        """
        recommendations = []

        for algorithm, tracking in accuracy_tracking.items():
            # Skip if not enough runs
            if tracking["total_runs"] < self._min_runs:
                continue

            # Check if accuracy is below threshold
            if tracking["average_accuracy"] >= self._accuracy_threshold:
                continue

            # Analyze common issues
            rejection_rate = tracking["rejected_runs"] / tracking["total_runs"]
            modification_rate = tracking["modified_runs"] / tracking["total_runs"]

            # Generate recommendations based on issues
            if rejection_rate > 0.3:
                # High rejection rate - need stricter thresholds
                recommendations.append(
                    DiscoveryParameterRecommendation(
                        recommendation_id=f"rec_{uuid.uuid4().hex[:8]}",
                        algorithm=algorithm,
                        parameter_name="significance_threshold",
                        current_value=0.05,
                        recommended_value=0.01,
                        justification=f"High rejection rate ({rejection_rate:.1%}) indicates false positives. "
                        f"Recommend stricter significance threshold.",
                        expected_accuracy_improvement=0.1,
                        confidence=0.7,
                    )
                )

            if modification_rate > 0.4:
                # High modification rate - need edge pruning
                recommendations.append(
                    DiscoveryParameterRecommendation(
                        recommendation_id=f"rec_{uuid.uuid4().hex[:8]}",
                        algorithm=algorithm,
                        parameter_name="edge_strength_threshold",
                        current_value=0.1,
                        recommended_value=0.2,
                        justification=f"High modification rate ({modification_rate:.1%}) indicates weak edges. "
                        f"Recommend stricter edge strength threshold.",
                        expected_accuracy_improvement=0.08,
                        confidence=0.6,
                    )
                )

            if tracking["edge_precision"] < 0.6:
                # Low precision - too many false positive edges
                recommendations.append(
                    DiscoveryParameterRecommendation(
                        recommendation_id=f"rec_{uuid.uuid4().hex[:8]}",
                        algorithm=algorithm,
                        parameter_name="max_degree",
                        current_value=None,
                        recommended_value=5,
                        justification=f"Low precision ({tracking['edge_precision']:.1%}) suggests overfitting. "
                        f"Recommend limiting maximum node degree.",
                        expected_accuracy_improvement=0.12,
                        confidence=0.65,
                    )
                )

            if tracking["edge_recall"] < 0.5:
                # Low recall - missing true edges
                recommendations.append(
                    DiscoveryParameterRecommendation(
                        recommendation_id=f"rec_{uuid.uuid4().hex[:8]}",
                        algorithm=algorithm,
                        parameter_name="min_samples",
                        current_value=None,
                        recommended_value=100,
                        justification=f"Low recall ({tracking['edge_recall']:.1%}) suggests underfitting. "
                        f"Recommend more samples or relaxed constraints.",
                        expected_accuracy_improvement=0.1,
                        confidence=0.55,
                    )
                )

        return recommendations

    def _detect_discovery_patterns(
        self,
        items: List[DiscoveryFeedbackItem],
        accuracy_tracking: Dict[str, DiscoveryAccuracyTracking],
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns in discovery feedback.

        Args:
            items: Discovery feedback items
            accuracy_tracking: Accuracy metrics by algorithm

        Returns:
            List of detected patterns
        """
        patterns = []

        # Check for systematic algorithm failures
        for algorithm, tracking in accuracy_tracking.items():
            if tracking["total_runs"] >= self._min_runs:
                acceptance_rate = tracking["accepted_runs"] / tracking["total_runs"]

                if acceptance_rate < 0.5:
                    patterns.append(
                        {
                            "pattern_id": f"discovery_{uuid.uuid4().hex[:8]}",
                            "pattern_type": "accuracy_issue",
                            "description": f"Algorithm {algorithm} has low acceptance rate ({acceptance_rate:.1%})",
                            "frequency": tracking["total_runs"],
                            "severity": "high" if acceptance_rate < 0.3 else "medium",
                            "affected_agents": ["causal_discovery"],
                            "example_feedback_ids": [
                                item["feedback_id"]
                                for item in items[:3]
                                if item.get("algorithm_used") == algorithm
                            ],
                            "root_cause_hypothesis": f"Algorithm {algorithm} parameters may not be tuned for this domain",
                        }
                    )

        # Check for gate override patterns
        overrides = [
            item for item in items if item.get("feedback_type") == "gate_override"
        ]
        if len(overrides) >= 5:
            override_rate = len(overrides) / len(items)
            if override_rate > 0.2:
                patterns.append(
                    {
                        "pattern_id": f"discovery_{uuid.uuid4().hex[:8]}",
                        "pattern_type": "coverage_gap",
                        "description": f"High gate override rate ({override_rate:.1%}) indicates gate criteria may be too strict",
                        "frequency": len(overrides),
                        "severity": "medium",
                        "affected_agents": ["causal_discovery", "orchestrator"],
                        "example_feedback_ids": [o["feedback_id"] for o in overrides[:3]],
                        "root_cause_hypothesis": "Gate acceptance criteria may not align with user expectations",
                    }
                )

        return patterns


# Factory function
def create_discovery_feedback_node(
    discovery_store: Optional[Any] = None,
    **kwargs,
) -> DiscoveryFeedbackNode:
    """
    Create a discovery feedback node.

    Args:
        discovery_store: Optional store for discovery data
        **kwargs: Additional configuration

    Returns:
        Configured DiscoveryFeedbackNode
    """
    return DiscoveryFeedbackNode(discovery_store=discovery_store, **kwargs)
