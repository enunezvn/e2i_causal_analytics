"""
E2I Explainer Agent - Configuration
Version: 4.3
Purpose: Configuration for smart LLM mode selection and agent settings

Complexity-based LLM selection allows the agent to automatically determine
whether to use LLM-based reasoning or deterministic fallback based on
input characteristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExplainerConfig:
    """Configuration for Explainer agent behavior."""

    # LLM Mode Selection
    llm_threshold: float = 0.5  # Complexity score threshold for LLM usage
    auto_llm: bool = True  # Enable automatic LLM mode selection

    # Complexity scoring weights
    result_count_weight: float = 0.25  # Weight for number of analysis results
    query_complexity_weight: float = 0.30  # Weight for query complexity
    causal_discovery_weight: float = 0.25  # Weight for causal discovery presence
    expertise_weight: float = 0.20  # Weight for expertise level complexity

    # Complexity thresholds
    high_result_count: int = 3  # Results above this count are considered complex
    complex_query_min_words: int = 10  # Queries with more words are considered complex

    # Complex query indicators (trigger LLM mode)
    complex_query_patterns: List[str] = field(
        default_factory=lambda: [
            r"\bwhy\b",
            r"\bexplain\b",
            r"\bcompare\b",
            r"\bcontrast\b",
            r"\banalyze\b",
            r"\bimpact\b",
            r"\bcause\b",
            r"\beffect\b",
            r"\brelationship\b",
            r"\bcorrelation\b",
            r"\btrend\b",
            r"\bpattern\b",
            r"\binsight\b",
            r"\brecommend\b",
            r"\bstrategy\b",
            r"\boptimize\b",
        ]
    )

    # Simple query indicators (prefer deterministic mode)
    simple_query_patterns: List[str] = field(
        default_factory=lambda: [
            r"\bwhat is\b",
            r"\bshow me\b",
            r"\blist\b",
            r"\bget\b",
            r"\bfetch\b",
            r"\bdisplay\b",
        ]
    )


# Default configuration instance
_default_config: Optional[ExplainerConfig] = None


def get_default_config() -> ExplainerConfig:
    """Get the default explainer configuration."""
    global _default_config
    if _default_config is None:
        _default_config = ExplainerConfig()
    return _default_config


def set_default_config(config: ExplainerConfig) -> None:
    """Set the default explainer configuration."""
    global _default_config
    _default_config = config


class ComplexityScorer:
    """
    Scores input complexity to determine LLM usage.

    The scorer evaluates multiple factors:
    1. Number of analysis results (more results = more complex)
    2. Query complexity (length, keywords, patterns)
    3. Presence of causal discovery data
    4. Target expertise level
    """

    def __init__(self, config: Optional[ExplainerConfig] = None):
        """
        Initialize complexity scorer.

        Args:
            config: Optional configuration. Uses default if not provided.
        """
        self.config = config or get_default_config()
        self._compiled_complex_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.complex_query_patterns
        ]
        self._compiled_simple_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.simple_query_patterns
        ]

    def compute_complexity(
        self,
        analysis_results: List[Dict[str, Any]],
        query: str,
        user_expertise: str = "analyst",
        has_causal_discovery: bool = False,
    ) -> float:
        """
        Compute complexity score for the input.

        Args:
            analysis_results: List of analysis results to explain
            query: User's query string
            user_expertise: Target audience expertise level
            has_causal_discovery: Whether causal discovery data is present

        Returns:
            Complexity score between 0.0 and 1.0
        """
        scores = {
            "result_count": self._score_result_count(analysis_results),
            "query_complexity": self._score_query_complexity(query),
            "causal_discovery": self._score_causal_discovery(
                has_causal_discovery, analysis_results
            ),
            "expertise": self._score_expertise_complexity(user_expertise),
        }

        # Weighted average
        weighted_score = (
            scores["result_count"] * self.config.result_count_weight
            + scores["query_complexity"] * self.config.query_complexity_weight
            + scores["causal_discovery"] * self.config.causal_discovery_weight
            + scores["expertise"] * self.config.expertise_weight
        )

        return min(1.0, max(0.0, weighted_score))

    def should_use_llm(
        self,
        analysis_results: List[Dict[str, Any]],
        query: str,
        user_expertise: str = "analyst",
        has_causal_discovery: bool = False,
    ) -> tuple[bool, float, str]:
        """
        Determine if LLM should be used based on complexity.

        Args:
            analysis_results: List of analysis results
            query: User's query
            user_expertise: Target audience
            has_causal_discovery: Whether causal discovery data present

        Returns:
            Tuple of (should_use_llm, complexity_score, reason)
        """
        score = self.compute_complexity(
            analysis_results, query, user_expertise, has_causal_discovery
        )

        if score >= self.config.llm_threshold:
            reason = f"Complexity score {score:.2f} >= threshold {self.config.llm_threshold}"
            return True, score, reason
        else:
            reason = f"Complexity score {score:.2f} < threshold {self.config.llm_threshold}"
            return False, score, reason

    def _score_result_count(self, analysis_results: List[Dict[str, Any]]) -> float:
        """Score based on number of analysis results."""
        count = len(analysis_results)
        if count == 0:
            return 0.0
        elif count <= 1:
            return 0.2
        elif count <= self.config.high_result_count:
            return 0.5
        else:
            # Scale up to 1.0 for very high counts
            return min(1.0, 0.5 + (count - self.config.high_result_count) * 0.1)

    def _score_query_complexity(self, query: str) -> float:
        """Score based on query complexity."""
        if not query:
            return 0.3  # No query means we rely on other factors

        query_lower = query.lower()
        word_count = len(query.split())

        # Check for simple patterns (reduce score)
        simple_matches = sum(
            1 for p in self._compiled_simple_patterns if p.search(query_lower)
        )

        # Check for complex patterns (increase score)
        complex_matches = sum(
            1 for p in self._compiled_complex_patterns if p.search(query_lower)
        )

        # Base score from word count
        if word_count <= 5:
            base_score = 0.2
        elif word_count <= self.config.complex_query_min_words:
            base_score = 0.4
        else:
            base_score = 0.6

        # Adjust for pattern matches
        pattern_adjustment = (complex_matches * 0.15) - (simple_matches * 0.1)
        final_score = base_score + pattern_adjustment

        return min(1.0, max(0.0, final_score))

    def _score_causal_discovery(
        self, has_causal_discovery: bool, analysis_results: List[Dict[str, Any]]
    ) -> float:
        """Score based on causal discovery presence."""
        if has_causal_discovery:
            return 1.0

        # Check if any result contains causal discovery indicators
        for result in analysis_results:
            if any(
                key in result
                for key in [
                    "discovered_dag",
                    "causal_graph",
                    "dag_adjacency",
                    "causal_discovery",
                    "discovery_gate_decision",
                ]
            ):
                return 1.0

            # Check nested structures
            if isinstance(result.get("data"), dict):
                if any(
                    key in result["data"]
                    for key in ["dag", "graph", "causal_structure"]
                ):
                    return 0.8

        return 0.0

    def _score_expertise_complexity(self, user_expertise: str) -> float:
        """
        Score based on expertise level complexity.

        Executive explanations require more synthesis (higher complexity).
        Data scientist explanations can be more technical/direct.
        """
        expertise_scores = {
            "executive": 0.8,  # Needs synthesis, simplification
            "analyst": 0.5,  # Balanced approach
            "data_scientist": 0.3,  # Can handle raw data
        }
        return expertise_scores.get(user_expertise, 0.5)


# Module-level convenience function
def compute_complexity(
    analysis_results: List[Dict[str, Any]],
    query: str,
    user_expertise: str = "analyst",
    has_causal_discovery: bool = False,
    config: Optional[ExplainerConfig] = None,
) -> float:
    """
    Compute complexity score for input.

    Args:
        analysis_results: List of analysis results
        query: User's query
        user_expertise: Target audience
        has_causal_discovery: Whether causal discovery data present
        config: Optional configuration

    Returns:
        Complexity score between 0.0 and 1.0
    """
    scorer = ComplexityScorer(config)
    return scorer.compute_complexity(
        analysis_results, query, user_expertise, has_causal_discovery
    )


def should_use_llm(
    analysis_results: List[Dict[str, Any]],
    query: str,
    user_expertise: str = "analyst",
    has_causal_discovery: bool = False,
    config: Optional[ExplainerConfig] = None,
) -> tuple[bool, float, str]:
    """
    Determine if LLM should be used based on complexity.

    Args:
        analysis_results: List of analysis results
        query: User's query
        user_expertise: Target audience
        has_causal_discovery: Whether causal discovery data present
        config: Optional configuration

    Returns:
        Tuple of (should_use_llm, complexity_score, reason)
    """
    scorer = ComplexityScorer(config)
    return scorer.should_use_llm(
        analysis_results, query, user_expertise, has_causal_discovery
    )
