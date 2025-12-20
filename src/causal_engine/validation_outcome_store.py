"""
E2I Causal Engine - Validation Outcome Store
Version: 4.3
Purpose: Persist and query validation outcomes for learning

This module provides storage and retrieval of ValidationOutcome objects,
enabling the Feedback Learner to learn from validation failures and
the Experiment Designer to query past failures.

Phase 4 of Causal Validation Protocol:
- Connect Feedback Learner to validation outcomes
- Build ExperimentKnowledgeStore query for past failures

Reference: docs/E2I_Causal_Validation_Protocol.html
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .validation_outcome import (
    ValidationOutcome,
    ValidationOutcomeType,
    FailureCategory,
    ValidationFailurePattern,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================


class ValidationOutcomeStoreBase(ABC):
    """Abstract base class for validation outcome storage.

    Implementations can use different backends:
    - InMemoryValidationOutcomeStore: For testing and development
    - SupabaseValidationOutcomeStore: For production (database)
    """

    @abstractmethod
    async def store(self, outcome: ValidationOutcome) -> str:
        """Store a validation outcome.

        Args:
            outcome: ValidationOutcome to store

        Returns:
            Outcome ID
        """
        pass

    @abstractmethod
    async def get(self, outcome_id: str) -> Optional[ValidationOutcome]:
        """Retrieve a validation outcome by ID.

        Args:
            outcome_id: Unique outcome identifier

        Returns:
            ValidationOutcome or None if not found
        """
        pass

    @abstractmethod
    async def query_failures(
        self,
        limit: int = 10,
        treatment_variable: Optional[str] = None,
        outcome_variable: Optional[str] = None,
        brand: Optional[str] = None,
        failure_category: Optional[FailureCategory] = None,
        since: Optional[str] = None,
    ) -> List[ValidationOutcome]:
        """Query validation failures for learning.

        Args:
            limit: Maximum number of outcomes to return
            treatment_variable: Filter by treatment variable
            outcome_variable: Filter by outcome variable
            brand: Filter by brand
            failure_category: Filter by failure category
            since: ISO timestamp to filter outcomes after

        Returns:
            List of ValidationOutcome objects matching criteria
        """
        pass

    @abstractmethod
    async def get_failure_patterns(
        self,
        limit: int = 10,
        category: Optional[FailureCategory] = None,
    ) -> List[Dict[str, Any]]:
        """Get aggregated failure patterns for learning.

        Args:
            limit: Maximum number of patterns to return
            category: Optional category filter

        Returns:
            List of pattern dictionaries with frequency counts
        """
        pass

    @abstractmethod
    async def get_similar_failures(
        self,
        treatment_variable: str,
        outcome_variable: str,
        limit: int = 5,
    ) -> List[ValidationOutcome]:
        """Get similar past validation failures.

        Args:
            treatment_variable: Treatment variable to match
            outcome_variable: Outcome variable to match
            limit: Maximum number of outcomes to return

        Returns:
            List of similar ValidationOutcome objects
        """
        pass


# ============================================================================
# IN-MEMORY IMPLEMENTATION (FOR TESTING)
# ============================================================================


class InMemoryValidationOutcomeStore(ValidationOutcomeStoreBase):
    """In-memory implementation of ValidationOutcomeStore.

    Suitable for testing and development. For production,
    use SupabaseValidationOutcomeStore.
    """

    def __init__(self):
        """Initialize in-memory store."""
        self._outcomes: Dict[str, ValidationOutcome] = {}
        self._pattern_counts: Dict[str, int] = defaultdict(int)

    async def store(self, outcome: ValidationOutcome) -> str:
        """Store a validation outcome."""
        self._outcomes[outcome.outcome_id] = outcome

        # Update pattern counts
        for pattern in outcome.failure_patterns:
            key = f"{pattern.category.value}:{pattern.test_name}"
            self._pattern_counts[key] += 1

        logger.info(
            f"Stored validation outcome {outcome.outcome_id}: "
            f"{outcome.outcome_type.value}"
        )

        return outcome.outcome_id

    async def get(self, outcome_id: str) -> Optional[ValidationOutcome]:
        """Retrieve a validation outcome by ID."""
        return self._outcomes.get(outcome_id)

    async def query_failures(
        self,
        limit: int = 10,
        treatment_variable: Optional[str] = None,
        outcome_variable: Optional[str] = None,
        brand: Optional[str] = None,
        failure_category: Optional[FailureCategory] = None,
        since: Optional[str] = None,
    ) -> List[ValidationOutcome]:
        """Query validation failures for learning."""
        results = []

        for outcome in self._outcomes.values():
            # Skip passed outcomes
            if outcome.outcome_type == ValidationOutcomeType.PASSED:
                continue

            # Apply filters
            if treatment_variable and outcome.treatment_variable != treatment_variable:
                continue
            if outcome_variable and outcome.outcome_variable != outcome_variable:
                continue
            if brand and outcome.brand != brand:
                continue
            if since and outcome.timestamp < since:
                continue

            # Filter by failure category if specified
            if failure_category:
                has_category = any(
                    p.category == failure_category
                    for p in outcome.failure_patterns
                )
                if not has_category:
                    continue

            results.append(outcome)

        # Sort by timestamp (most recent first) and limit
        results.sort(key=lambda o: o.timestamp, reverse=True)
        return results[:limit]

    async def get_failure_patterns(
        self,
        limit: int = 10,
        category: Optional[FailureCategory] = None,
    ) -> List[Dict[str, Any]]:
        """Get aggregated failure patterns for learning."""
        # Aggregate patterns across all outcomes
        pattern_data: Dict[str, Dict[str, Any]] = {}

        for outcome in self._outcomes.values():
            for pattern in outcome.failure_patterns:
                if category and pattern.category != category:
                    continue

                key = f"{pattern.category.value}:{pattern.test_name}"
                if key not in pattern_data:
                    pattern_data[key] = {
                        "category": pattern.category.value,
                        "test_name": pattern.test_name,
                        "count": 0,
                        "recommendations": set(),
                        "avg_delta_percent": 0,
                        "total_delta": 0,
                    }

                pattern_data[key]["count"] += 1
                pattern_data[key]["recommendations"].add(pattern.recommendation)
                pattern_data[key]["total_delta"] += abs(pattern.delta_percent)

        # Compute averages and convert to list
        results = []
        for key, data in pattern_data.items():
            count = data["count"]
            results.append({
                "category": data["category"],
                "test_name": data["test_name"],
                "count": count,
                "avg_delta_percent": data["total_delta"] / count if count > 0 else 0,
                "recommendations": list(data["recommendations"]),
            })

        # Sort by frequency and limit
        results.sort(key=lambda x: x["count"], reverse=True)
        return results[:limit]

    async def get_similar_failures(
        self,
        treatment_variable: str,
        outcome_variable: str,
        limit: int = 5,
    ) -> List[ValidationOutcome]:
        """Get similar past validation failures."""
        # Simple similarity: exact match on treatment or outcome variable
        results = []

        for outcome in self._outcomes.values():
            if outcome.outcome_type == ValidationOutcomeType.PASSED:
                continue

            # Score similarity
            score = 0
            if outcome.treatment_variable == treatment_variable:
                score += 2
            if outcome.outcome_variable == outcome_variable:
                score += 2
            # Partial match on variable names
            if treatment_variable and outcome.treatment_variable:
                if treatment_variable in outcome.treatment_variable or \
                   outcome.treatment_variable in treatment_variable:
                    score += 1
            if outcome_variable and outcome.outcome_variable:
                if outcome_variable in outcome.outcome_variable or \
                   outcome.outcome_variable in outcome_variable:
                    score += 1

            if score > 0:
                results.append((score, outcome))

        # Sort by similarity score and timestamp
        results.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)
        return [r[1] for r in results[:limit]]

    def clear(self):
        """Clear all stored outcomes (for testing)."""
        self._outcomes.clear()
        self._pattern_counts.clear()

    @property
    def count(self) -> int:
        """Number of stored outcomes."""
        return len(self._outcomes)


# ============================================================================
# EXPERIMENT KNOWLEDGE STORE INTERFACE
# ============================================================================


@dataclass
class ValidationLearning:
    """Learning extracted from validation failures.

    Used by Experiment Designer to avoid similar mistakes.
    """

    failure_category: str
    description: str
    recommendation: str
    frequency: int
    example_variables: List[Tuple[str, str]]  # (treatment, outcome) pairs
    avg_effect_change: float


class ExperimentKnowledgeStore:
    """Knowledge store for Experiment Designer integration.

    This wraps ValidationOutcomeStore to provide experiment-design-specific
    queries for the Experiment Designer agent.

    Integration point for Phase 4:
    - Query past validation failures
    - Extract learnings for experiment design
    - Provide recommendations based on historical patterns
    """

    def __init__(self, outcome_store: Optional[ValidationOutcomeStoreBase] = None):
        """Initialize experiment knowledge store.

        Args:
            outcome_store: ValidationOutcomeStore for querying failures.
                          Uses InMemoryValidationOutcomeStore if not provided.
        """
        self._outcome_store = outcome_store or InMemoryValidationOutcomeStore()

    async def get_similar_experiments(
        self,
        business_question: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve similar past experiments based on business question.

        This extends the mock implementation in context_loader.py
        with actual validation outcome data.

        Args:
            business_question: Business question to match
            limit: Maximum number of experiments to return

        Returns:
            List of experiment dictionaries with lessons learned
        """
        # Extract potential variable names from business question
        # (Simple heuristic - in production, use NLP)
        keywords = business_question.lower().split()

        # Get all failures and score by relevance
        all_failures = await self._outcome_store.query_failures(limit=50)

        experiments = []
        for outcome in all_failures:
            # Score relevance
            relevance = 0
            if outcome.treatment_variable:
                for kw in keywords:
                    if kw in outcome.treatment_variable.lower():
                        relevance += 1
            if outcome.outcome_variable:
                for kw in keywords:
                    if kw in outcome.outcome_variable.lower():
                        relevance += 1

            if relevance > 0 or len(experiments) < limit:
                lessons = [p.recommendation for p in outcome.failure_patterns[:3]]
                experiments.append({
                    "experiment_id": outcome.estimate_id or outcome.outcome_id,
                    "hypothesis": f"{outcome.treatment_variable} → {outcome.outcome_variable}",
                    "design_type": "observational",  # From validation context
                    "sample_size": outcome.sample_size,
                    "outcome": outcome.outcome_variable,
                    "result": "failed_validation" if outcome.outcome_type != ValidationOutcomeType.PASSED else "passed",
                    "effect_size": outcome.effect_size,
                    "lessons_learned": lessons if lessons else ["No specific lessons"],
                    "relevance_score": relevance,
                })

        # Sort by relevance and limit
        experiments.sort(key=lambda x: x["relevance_score"], reverse=True)
        return experiments[:limit]

    async def get_recent_assumption_violations(
        self,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get recent experiments where key assumptions were violated.

        Args:
            limit: Maximum number of violations to return

        Returns:
            List of violation dictionaries
        """
        patterns = await self._outcome_store.get_failure_patterns(limit=limit)

        violations = []
        for pattern in patterns:
            violations.append({
                "experiment_id": f"pattern_{pattern['category']}",
                "violation_type": pattern["category"],
                "description": f"Test {pattern['test_name']} failed {pattern['count']} times",
                "impact": f"Average effect change: {pattern['avg_delta_percent']:.1f}%",
                "recommendation": pattern["recommendations"][0] if pattern["recommendations"] else "Review test results",
            })

        return violations

    async def get_validation_learnings(
        self,
        category: Optional[FailureCategory] = None,
        limit: int = 10,
    ) -> List[ValidationLearning]:
        """Get structured learnings from validation failures.

        Args:
            category: Optional failure category filter
            limit: Maximum number of learnings to return

        Returns:
            List of ValidationLearning objects
        """
        patterns = await self._outcome_store.get_failure_patterns(
            limit=limit,
            category=category,
        )

        learnings = []
        for pattern in patterns:
            # Get example variables from outcomes with this pattern
            failures = await self._outcome_store.query_failures(
                limit=3,
                failure_category=FailureCategory(pattern["category"]) if pattern["category"] != "unknown" else None,
            )

            example_variables = [
                (o.treatment_variable or "unknown", o.outcome_variable or "unknown")
                for o in failures
                if any(p.test_name == pattern["test_name"] for p in o.failure_patterns)
            ][:3]

            learnings.append(ValidationLearning(
                failure_category=pattern["category"],
                description=f"Test {pattern['test_name']} commonly fails",
                recommendation=pattern["recommendations"][0] if pattern["recommendations"] else "Review methodology",
                frequency=pattern["count"],
                example_variables=example_variables,
                avg_effect_change=pattern["avg_delta_percent"],
            ))

        return learnings

    async def should_warn_for_design(
        self,
        treatment_variable: str,
        outcome_variable: str,
    ) -> List[str]:
        """Check if there are warnings for a proposed design.

        Args:
            treatment_variable: Proposed treatment variable
            outcome_variable: Proposed outcome variable

        Returns:
            List of warning messages based on past failures
        """
        similar = await self._outcome_store.get_similar_failures(
            treatment_variable=treatment_variable,
            outcome_variable=outcome_variable,
            limit=5,
        )

        warnings = []
        seen_recommendations = set()

        for outcome in similar:
            for pattern in outcome.failure_patterns:
                if pattern.recommendation not in seen_recommendations:
                    warnings.append(
                        f"⚠️ Past failure ({pattern.category.value}): {pattern.recommendation}"
                    )
                    seen_recommendations.add(pattern.recommendation)

        return warnings[:3]  # Limit to top 3 warnings


# ============================================================================
# GLOBAL INSTANCE (SINGLETON PATTERN)
# ============================================================================

_global_outcome_store: Optional[InMemoryValidationOutcomeStore] = None
_global_knowledge_store: Optional[ExperimentKnowledgeStore] = None


def get_validation_outcome_store() -> InMemoryValidationOutcomeStore:
    """Get the global validation outcome store instance.

    Returns:
        InMemoryValidationOutcomeStore instance
    """
    global _global_outcome_store
    if _global_outcome_store is None:
        _global_outcome_store = InMemoryValidationOutcomeStore()
    return _global_outcome_store


def get_experiment_knowledge_store() -> ExperimentKnowledgeStore:
    """Get the global experiment knowledge store instance.

    Returns:
        ExperimentKnowledgeStore instance
    """
    global _global_knowledge_store
    if _global_knowledge_store is None:
        _global_knowledge_store = ExperimentKnowledgeStore(
            outcome_store=get_validation_outcome_store()
        )
    return _global_knowledge_store


async def log_validation_outcome(outcome: ValidationOutcome) -> str:
    """Convenience function to log a validation outcome.

    Args:
        outcome: ValidationOutcome to store

    Returns:
        Outcome ID
    """
    store = get_validation_outcome_store()
    return await store.store(outcome)
