"""
E2I Causal Analytics - Driver Ranker
====================================

Compares causal vs predictive feature importance.

The DriverRanker:
1. Computes causal importance from discovered DAG
2. Computes predictive importance from SHAP values
3. Compares rankings to identify discrepancies
4. Highlights features that are predictive but not causal

Author: E2I Causal Analytics Team
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from .base import DiscoveryResult


class ImportanceType(str, Enum):
    """Types of feature importance."""

    CAUSAL = "causal"  # Based on DAG structure
    PREDICTIVE = "predictive"  # Based on SHAP values
    COMBINED = "combined"  # Weighted combination


@dataclass
class FeatureRanking:
    """Ranking information for a single feature.

    Attributes:
        feature_name: Name of the feature
        causal_rank: Rank based on causal importance (1 = most important)
        predictive_rank: Rank based on predictive importance
        causal_score: Causal importance score [0, 1]
        predictive_score: Predictive importance score [0, 1]
        rank_difference: predictive_rank - causal_rank (positive = more predictive)
        is_direct_cause: Whether feature is direct cause of target
        path_length: Shortest path length to target in DAG
    """

    feature_name: str
    causal_rank: int
    predictive_rank: int
    causal_score: float
    predictive_score: float
    rank_difference: int = 0
    is_direct_cause: bool = False
    path_length: Optional[int] = None

    def __post_init__(self):
        """Calculate derived fields."""
        self.rank_difference = self.predictive_rank - self.causal_rank

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "causal_rank": self.causal_rank,
            "predictive_rank": self.predictive_rank,
            "causal_score": self.causal_score,
            "predictive_score": self.predictive_score,
            "rank_difference": self.rank_difference,
            "is_direct_cause": self.is_direct_cause,
            "path_length": self.path_length,
        }


@dataclass
class DriverRankingResult:
    """Result of driver ranking analysis.

    Attributes:
        rankings: List of feature rankings sorted by causal importance
        target_variable: Target variable used for ranking
        causal_only_features: Features important causally but not predictively
        predictive_only_features: Features important predictively but not causally
        concordant_features: Features with similar causal and predictive rank
        rank_correlation: Spearman correlation between causal and predictive ranks
        metadata: Additional analysis metadata
    """

    rankings: List[FeatureRanking]
    target_variable: str
    causal_only_features: List[str] = field(default_factory=list)
    predictive_only_features: List[str] = field(default_factory=list)
    concordant_features: List[str] = field(default_factory=list)
    rank_correlation: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_by_causal_rank(self, top_k: Optional[int] = None) -> List[FeatureRanking]:
        """Get rankings sorted by causal importance.

        Args:
            top_k: Return only top k features

        Returns:
            Sorted list of rankings
        """
        sorted_rankings = sorted(self.rankings, key=lambda r: r.causal_rank)
        return sorted_rankings[:top_k] if top_k else sorted_rankings

    def get_by_predictive_rank(self, top_k: Optional[int] = None) -> List[FeatureRanking]:
        """Get rankings sorted by predictive importance.

        Args:
            top_k: Return only top k features

        Returns:
            Sorted list of rankings
        """
        sorted_rankings = sorted(self.rankings, key=lambda r: r.predictive_rank)
        return sorted_rankings[:top_k] if top_k else sorted_rankings

    def get_discordant_features(self, threshold: int = 3) -> List[FeatureRanking]:
        """Get features with large rank differences.

        Args:
            threshold: Minimum absolute rank difference

        Returns:
            Features with rank_difference > threshold
        """
        return [r for r in self.rankings if abs(r.rank_difference) >= threshold]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rankings": [r.to_dict() for r in self.rankings],
            "target_variable": self.target_variable,
            "causal_only_features": self.causal_only_features,
            "predictive_only_features": self.predictive_only_features,
            "concordant_features": self.concordant_features,
            "rank_correlation": self.rank_correlation,
            "metadata": self.metadata,
        }


class DriverRanker:
    """Compares causal vs predictive feature importance.

    Combines DAG structure information with SHAP values to identify:
    - Features that are true causal drivers
    - Features that are predictive due to correlation/confounding
    - Discrepancies between causal and predictive importance

    Example:
        >>> ranker = DriverRanker()
        >>> result = ranker.rank_drivers(
        ...     dag=discovery_result.ensemble_dag,
        ...     target="outcome",
        ...     shap_values=shap_values,
        ...     feature_names=feature_names,
        ... )
        >>> print("Top causal drivers:", [r.feature_name for r in result.get_by_causal_rank(5)])
        >>> print("Predictive but not causal:", result.predictive_only_features)
    """

    def __init__(
        self,
        concordance_threshold: int = 2,
        importance_percentile: float = 0.25,
    ):
        """Initialize DriverRanker.

        Args:
            concordance_threshold: Max rank difference for concordant features
            importance_percentile: Top percentile to consider as "important"
        """
        self.concordance_threshold = concordance_threshold
        self.importance_percentile = importance_percentile

    def rank_drivers(
        self,
        dag: nx.DiGraph,
        target: str,
        shap_values: NDArray[np.float64],
        feature_names: List[str],
    ) -> DriverRankingResult:
        """Rank features by causal and predictive importance.

        Args:
            dag: Discovered or manual DAG
            target: Target variable name
            shap_values: SHAP values array (n_samples, n_features)
            feature_names: Names of features

        Returns:
            DriverRankingResult with rankings and analysis
        """
        if target not in dag.nodes():
            logger.warning(f"Target '{target}' not in DAG, adding as isolated node")
            dag.add_node(target)

        # Compute causal importance from DAG
        causal_scores = self._compute_causal_importance(dag, target, feature_names)

        # Compute predictive importance from SHAP
        predictive_scores = self._compute_predictive_importance(shap_values, feature_names)

        # Create rankings
        causal_ranks = self._scores_to_ranks(causal_scores)
        predictive_ranks = self._scores_to_ranks(predictive_scores)

        # Build FeatureRanking objects
        rankings = []
        for feature in feature_names:
            if feature == target:
                continue

            # Check if direct cause
            is_direct = feature in dag.predecessors(target) if target in dag.nodes() else False

            # Calculate path length to target
            try:
                path_length = nx.shortest_path_length(dag, feature, target)
            except nx.NetworkXNoPath:
                path_length = None

            rankings.append(
                FeatureRanking(
                    feature_name=feature,
                    causal_rank=causal_ranks.get(feature, len(feature_names)),
                    predictive_rank=predictive_ranks.get(feature, len(feature_names)),
                    causal_score=causal_scores.get(feature, 0.0),
                    predictive_score=predictive_scores.get(feature, 0.0),
                    is_direct_cause=is_direct,
                    path_length=path_length,
                )
            )

        # Categorize features
        n_important = max(1, int(len(rankings) * self.importance_percentile))

        top_causal = {r.feature_name for r in sorted(rankings, key=lambda x: x.causal_rank)[:n_important]}
        top_predictive = {r.feature_name for r in sorted(rankings, key=lambda x: x.predictive_rank)[:n_important]}

        causal_only = list(top_causal - top_predictive)
        predictive_only = list(top_predictive - top_causal)
        concordant = [
            r.feature_name
            for r in rankings
            if abs(r.rank_difference) <= self.concordance_threshold
        ]

        # Calculate rank correlation
        rank_corr = self._spearman_correlation(
            [r.causal_rank for r in rankings],
            [r.predictive_rank for r in rankings],
        )

        logger.info(
            f"Driver ranking: {len(rankings)} features, "
            f"correlation: {rank_corr:.2f}, "
            f"causal-only: {len(causal_only)}, "
            f"predictive-only: {len(predictive_only)}"
        )

        return DriverRankingResult(
            rankings=rankings,
            target_variable=target,
            causal_only_features=causal_only,
            predictive_only_features=predictive_only,
            concordant_features=concordant,
            rank_correlation=rank_corr,
            metadata={
                "n_features": len(rankings),
                "n_important": n_important,
                "concordance_threshold": self.concordance_threshold,
            },
        )

    def _compute_causal_importance(
        self,
        dag: nx.DiGraph,
        target: str,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Compute causal importance scores from DAG structure.

        Importance is based on:
        1. Direct causal relationship to target
        2. Path length to target
        3. Number of paths to target
        4. Centrality in causal structure

        Args:
            dag: Causal DAG
            target: Target variable
            feature_names: Feature names

        Returns:
            Dict mapping feature name to causal importance score [0, 1]
        """
        scores = {}

        for feature in feature_names:
            if feature == target or feature not in dag.nodes():
                scores[feature] = 0.0
                continue

            score = 0.0

            # Check if direct cause (highest importance)
            if target in dag.successors(feature):
                score += 0.5

            # Calculate path-based importance
            try:
                # Shortest path length (shorter = more important)
                path_length = nx.shortest_path_length(dag, feature, target)
                path_score = 1.0 / (1.0 + path_length)
                score += 0.3 * path_score

                # Number of paths (more paths = more important)
                all_paths = list(nx.all_simple_paths(dag, feature, target, cutoff=4))
                n_paths = len(all_paths)
                score += 0.1 * min(n_paths / 5, 1.0)

            except nx.NetworkXNoPath:
                pass

            # Add centrality contribution
            try:
                betweenness = nx.betweenness_centrality(dag)
                score += 0.1 * betweenness.get(feature, 0.0)
            except Exception:
                pass

            scores[feature] = min(score, 1.0)

        # Normalize to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        return scores

    def _compute_predictive_importance(
        self,
        shap_values: NDArray[np.float64],
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Compute predictive importance from SHAP values.

        Args:
            shap_values: SHAP values (n_samples, n_features)
            feature_names: Feature names

        Returns:
            Dict mapping feature name to predictive importance [0, 1]
        """
        # Mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # Normalize to [0, 1]
        max_shap = mean_abs_shap.max() if mean_abs_shap.max() > 0 else 1.0
        normalized = mean_abs_shap / max_shap

        return {name: float(score) for name, score in zip(feature_names, normalized)}

    def _scores_to_ranks(self, scores: Dict[str, float]) -> Dict[str, int]:
        """Convert scores to ranks (1 = highest score).

        Args:
            scores: Feature name to score mapping

        Returns:
            Feature name to rank mapping
        """
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {name: rank + 1 for rank, (name, _) in enumerate(sorted_features)}

    def _spearman_correlation(
        self,
        ranks1: List[int],
        ranks2: List[int],
    ) -> float:
        """Calculate Spearman rank correlation.

        Args:
            ranks1: First ranking
            ranks2: Second ranking

        Returns:
            Correlation coefficient [-1, 1]
        """
        if len(ranks1) != len(ranks2) or len(ranks1) == 0:
            return 0.0

        n = len(ranks1)
        d_squared = sum((r1 - r2) ** 2 for r1, r2 in zip(ranks1, ranks2))

        # Spearman's rho formula
        rho = 1 - (6 * d_squared) / (n * (n**2 - 1))
        return rho

    def rank_from_discovery_result(
        self,
        result: DiscoveryResult,
        target: str,
        shap_values: NDArray[np.float64],
    ) -> DriverRankingResult:
        """Convenience method to rank from DiscoveryResult.

        Args:
            result: Discovery result with ensemble DAG
            target: Target variable
            shap_values: SHAP values

        Returns:
            DriverRankingResult
        """
        if result.ensemble_dag is None:
            raise ValueError("DiscoveryResult has no ensemble DAG")

        feature_names = result.metadata.get("node_names", list(result.ensemble_dag.nodes()))
        return self.rank_drivers(result.ensemble_dag, target, shap_values, feature_names)
