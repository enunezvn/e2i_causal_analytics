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

import logging
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .validation_outcome import (
    FailureCategory,
    ValidationFailurePattern,
    ValidationOutcome,
    ValidationOutcomeType,
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

        logger.info(f"Stored validation outcome {outcome.outcome_id}: {outcome.outcome_type.value}")

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
                has_category = any(p.category == failure_category for p in outcome.failure_patterns)
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
            results.append(
                {
                    "category": data["category"],
                    "test_name": data["test_name"],
                    "count": count,
                    "avg_delta_percent": data["total_delta"] / count if count > 0 else 0,
                    "recommendations": list(data["recommendations"]),
                }
            )

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
                if (
                    treatment_variable in outcome.treatment_variable
                    or outcome.treatment_variable in treatment_variable
                ):
                    score += 1
            if outcome_variable and outcome.outcome_variable:
                if (
                    outcome_variable in outcome.outcome_variable
                    or outcome.outcome_variable in outcome_variable
                ):
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
# SUPABASE IMPLEMENTATION (FOR PRODUCTION)
# ============================================================================


class SupabaseValidationOutcomeStore(ValidationOutcomeStoreBase):
    """Supabase-backed implementation of ValidationOutcomeStore.

    Uses the validation_outcomes table for persistent storage.
    Falls back to InMemoryValidationOutcomeStore if Supabase is unavailable.
    """

    def __init__(self):
        """Initialize Supabase store with lazy client loading."""
        self._client = None
        self._fallback_store: Optional[InMemoryValidationOutcomeStore] = None

    def _get_client(self):
        """Get Supabase client (lazy initialization)."""
        if self._client is None:
            try:
                from src.memory.services.factories import get_supabase_client

                self._client = get_supabase_client()
            except Exception as e:
                logger.warning(f"Could not initialize Supabase client: {e}")
                self._client = None
        return self._client

    def _get_fallback(self) -> InMemoryValidationOutcomeStore:
        """Get fallback in-memory store."""
        if self._fallback_store is None:
            self._fallback_store = InMemoryValidationOutcomeStore()
        return self._fallback_store

    def _outcome_to_row(self, outcome: ValidationOutcome) -> Dict[str, Any]:
        """Convert ValidationOutcome to database row."""
        return {
            "outcome_id": outcome.outcome_id,
            "estimate_id": outcome.estimate_id,
            "outcome_type": outcome.outcome_type.value,
            "treatment_variable": outcome.treatment_variable,
            "outcome_variable": outcome.outcome_variable,
            "brand": outcome.brand,
            "sample_size": outcome.sample_size,
            "effect_size": float(outcome.effect_size) if outcome.effect_size else None,
            "gate_decision": outcome.gate_decision,
            "confidence_score": outcome.confidence_score,
            "tests_passed": outcome.tests_passed,
            "tests_failed": outcome.tests_failed,
            "tests_total": outcome.tests_total,
            "failure_patterns": [
                {
                    "category": p.category.value,
                    "test_name": p.test_name,
                    "description": p.description,
                    "severity": p.severity,
                    "original_effect": p.original_effect,
                    "refuted_effect": p.refuted_effect,
                    "delta_percent": p.delta_percent,
                    "recommendation": p.recommendation,
                }
                for p in outcome.failure_patterns
            ],
            "raw_suite": outcome.raw_suite or {},
            "agent_context": outcome.agent_context or {},
            "dag_hash": outcome.dag_hash,
            "timestamp": outcome.timestamp,
        }

    def _row_to_outcome(self, row: Dict[str, Any]) -> ValidationOutcome:
        """Convert database row to ValidationOutcome."""
        patterns = []
        for p in row.get("failure_patterns", []):
            patterns.append(
                ValidationFailurePattern(
                    category=FailureCategory(p.get("category", "unknown")),
                    test_name=p.get("test_name", ""),
                    description=p.get("description", ""),
                    severity=p.get("severity", "medium"),
                    original_effect=p.get("original_effect", 0.0),
                    refuted_effect=p.get("refuted_effect", 0.0),
                    delta_percent=p.get("delta_percent", 0.0),
                    recommendation=p.get("recommendation", ""),
                )
            )

        return ValidationOutcome(
            outcome_id=row.get("outcome_id", ""),
            estimate_id=row.get("estimate_id"),
            outcome_type=ValidationOutcomeType(row.get("outcome_type", "passed")),
            treatment_variable=row.get("treatment_variable"),
            outcome_variable=row.get("outcome_variable"),
            brand=row.get("brand"),
            sample_size=row.get("sample_size"),
            effect_size=row.get("effect_size"),
            gate_decision=row.get("gate_decision", "unknown"),
            confidence_score=row.get("confidence_score", 0.0),
            tests_passed=row.get("tests_passed", 0),
            tests_failed=row.get("tests_failed", 0),
            tests_total=row.get("tests_total", 0),
            failure_patterns=patterns,
            raw_suite=row.get("raw_suite", {}),
            agent_context=row.get("agent_context", {}),
            dag_hash=row.get("dag_hash"),
            timestamp=row.get("timestamp", ""),
        )

    async def store(self, outcome: ValidationOutcome) -> str:
        """Store a validation outcome."""
        client = self._get_client()
        if not client:
            logger.warning("Supabase unavailable, using in-memory fallback")
            return await self._get_fallback().store(outcome)

        try:
            row = self._outcome_to_row(outcome)
            result = client.table("validation_outcomes").insert(row).execute()

            if result.data:
                logger.info(
                    f"Stored validation outcome {outcome.outcome_id}: {outcome.outcome_type.value}"
                )
                return outcome.outcome_id
            else:
                raise Exception("Insert returned no data")

        except Exception as e:
            logger.error(f"Failed to store outcome in Supabase: {e}")
            # Fall back to in-memory
            return await self._get_fallback().store(outcome)

    async def get(self, outcome_id: str) -> Optional[ValidationOutcome]:
        """Retrieve a validation outcome by ID."""
        client = self._get_client()
        if not client:
            return await self._get_fallback().get(outcome_id)

        try:
            result = (
                client.table("validation_outcomes")
                .select("*")
                .eq("outcome_id", outcome_id)
                .execute()
            )

            if result.data and len(result.data) > 0:
                return self._row_to_outcome(result.data[0])
            return None

        except Exception as e:
            logger.error(f"Failed to get outcome from Supabase: {e}")
            return await self._get_fallback().get(outcome_id)

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
        client = self._get_client()
        if not client:
            return await self._get_fallback().query_failures(
                limit=limit,
                treatment_variable=treatment_variable,
                outcome_variable=outcome_variable,
                brand=brand,
                failure_category=failure_category,
                since=since,
            )

        try:
            query = client.table("validation_outcomes").select("*")

            # Exclude passed outcomes
            query = query.neq("outcome_type", "passed")

            # Apply filters
            if treatment_variable:
                query = query.eq("treatment_variable", treatment_variable)
            if outcome_variable:
                query = query.eq("outcome_variable", outcome_variable)
            if brand:
                query = query.eq("brand", brand)
            if since:
                query = query.gte("timestamp", since)

            # Order by timestamp (most recent first) and limit
            query = query.order("timestamp", desc=True).limit(limit)

            result = query.execute()

            outcomes = [self._row_to_outcome(row) for row in result.data]

            # Post-filter by failure category if specified (needs JSONB containment)
            if failure_category:
                outcomes = [
                    o
                    for o in outcomes
                    if any(p.category == failure_category for p in o.failure_patterns)
                ]

            return outcomes

        except Exception as e:
            logger.error(f"Failed to query failures from Supabase: {e}")
            return await self._get_fallback().query_failures(
                limit=limit,
                treatment_variable=treatment_variable,
                outcome_variable=outcome_variable,
                brand=brand,
                failure_category=failure_category,
                since=since,
            )

    async def get_failure_patterns(
        self,
        limit: int = 10,
        category: Optional[FailureCategory] = None,
    ) -> List[Dict[str, Any]]:
        """Get aggregated failure patterns for learning."""
        client = self._get_client()
        if not client:
            return await self._get_fallback().get_failure_patterns(limit=limit, category=category)

        try:
            # Use the v_validation_failure_patterns view
            query = client.table("v_validation_failure_patterns").select("*")

            if category:
                query = query.eq("category", category.value)

            query = query.limit(limit)
            result = query.execute()

            patterns = []
            for row in result.data:
                patterns.append(
                    {
                        "category": row.get("category", "unknown"),
                        "test_name": row.get("test_name", ""),
                        "count": row.get("failure_count", 0),
                        "avg_delta_percent": float(row.get("avg_delta_percent", 0) or 0),
                        "recommendations": row.get("recommendations", []),
                    }
                )

            return patterns

        except Exception as e:
            logger.error(f"Failed to get failure patterns from Supabase: {e}")
            return await self._get_fallback().get_failure_patterns(limit=limit, category=category)

    async def get_similar_failures(
        self,
        treatment_variable: str,
        outcome_variable: str,
        limit: int = 5,
    ) -> List[ValidationOutcome]:
        """Get similar past validation failures."""
        client = self._get_client()
        if not client:
            return await self._get_fallback().get_similar_failures(
                treatment_variable=treatment_variable,
                outcome_variable=outcome_variable,
                limit=limit,
            )

        try:
            # Use the query_similar_validation_failures function
            result = client.rpc(
                "query_similar_validation_failures",
                {
                    "p_treatment_variable": treatment_variable,
                    "p_outcome_variable": outcome_variable,
                    "p_limit": limit,
                },
            ).execute()

            outcomes = []
            for row in result.data:
                # Convert RPC result to ValidationOutcome
                patterns = []
                for p in row.get("failure_patterns", []):
                    patterns.append(
                        ValidationFailurePattern(
                            category=FailureCategory(p.get("category", "unknown")),
                            test_name=p.get("test_name", ""),
                            description=p.get("description", ""),
                            severity=p.get("severity", "medium"),
                            original_effect=p.get("original_effect", 0.0),
                            refuted_effect=p.get("refuted_effect", 0.0),
                            delta_percent=p.get("delta_percent", 0.0),
                            recommendation=p.get("recommendation", ""),
                        )
                    )

                outcomes.append(
                    ValidationOutcome(
                        outcome_id=row.get("outcome_id", ""),
                        outcome_type=ValidationOutcomeType(row.get("outcome_type", "passed")),
                        treatment_variable=row.get("treatment_variable"),
                        outcome_variable=row.get("outcome_variable"),
                        effect_size=row.get("effect_size"),
                        failure_patterns=patterns,
                        timestamp=row.get("timestamp", ""),
                    )
                )

            return outcomes

        except Exception as e:
            logger.error(f"Failed to get similar failures from Supabase: {e}")
            return await self._get_fallback().get_similar_failures(
                treatment_variable=treatment_variable,
                outcome_variable=outcome_variable,
                limit=limit,
            )


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

        experiments: List[Dict[str, Any]] = []
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
                experiments.append(
                    {
                        "experiment_id": outcome.estimate_id or outcome.outcome_id,
                        "hypothesis": f"{outcome.treatment_variable} → {outcome.outcome_variable}",
                        "design_type": "observational",  # From validation context
                        "sample_size": outcome.sample_size,
                        "outcome": outcome.outcome_variable,
                        "result": (
                            "failed_validation"
                            if outcome.outcome_type != ValidationOutcomeType.PASSED
                            else "passed"
                        ),
                        "effect_size": outcome.effect_size,
                        "lessons_learned": lessons if lessons else ["No specific lessons"],
                        "relevance_score": relevance,
                    }
                )

        # Sort by relevance and limit
        experiments.sort(key=lambda x: int(x.get("relevance_score", 0)), reverse=True)
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
            violations.append(
                {
                    "experiment_id": f"pattern_{pattern['category']}",
                    "violation_type": pattern["category"],
                    "description": f"Test {pattern['test_name']} failed {pattern['count']} times",
                    "impact": f"Average effect change: {pattern['avg_delta_percent']:.1f}%",
                    "recommendation": (
                        pattern["recommendations"][0]
                        if pattern["recommendations"]
                        else "Review test results"
                    ),
                }
            )

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
                failure_category=(
                    FailureCategory(pattern["category"])
                    if pattern["category"] != "unknown"
                    else None
                ),
            )

            example_variables = [
                (o.treatment_variable or "unknown", o.outcome_variable or "unknown")
                for o in failures
                if any(p.test_name == pattern["test_name"] for p in o.failure_patterns)
            ][:3]

            learnings.append(
                ValidationLearning(
                    failure_category=pattern["category"],
                    description=f"Test {pattern['test_name']} commonly fails",
                    recommendation=(
                        pattern["recommendations"][0]
                        if pattern["recommendations"]
                        else "Review methodology"
                    ),
                    frequency=pattern["count"],
                    example_variables=example_variables,
                    avg_effect_change=pattern["avg_delta_percent"],
                )
            )

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

_global_outcome_store: Optional[ValidationOutcomeStoreBase] = None
_global_knowledge_store: Optional[ExperimentKnowledgeStore] = None


def get_validation_outcome_store(
    use_supabase: bool = True,
) -> ValidationOutcomeStoreBase:
    """Get the global validation outcome store instance.

    Args:
        use_supabase: If True, use Supabase backend (falls back to in-memory
                     if Supabase is unavailable). If False, use in-memory.

    Returns:
        ValidationOutcomeStoreBase instance (Supabase or InMemory)
    """
    global _global_outcome_store

    # Check if we need to create/recreate the store
    if _global_outcome_store is None:
        if use_supabase:
            # Check if Supabase is configured
            supabase_url = os.getenv("SUPABASE_URL")
            if supabase_url:
                logger.info("Using Supabase validation outcome store")
                _global_outcome_store = SupabaseValidationOutcomeStore()
            else:
                logger.info("SUPABASE_URL not set, using in-memory validation outcome store")
                _global_outcome_store = InMemoryValidationOutcomeStore()
        else:
            logger.info("Using in-memory validation outcome store (requested)")
            _global_outcome_store = InMemoryValidationOutcomeStore()

    return _global_outcome_store


def reset_validation_outcome_store():
    """Reset the global validation outcome store.

    Used for testing to ensure a fresh store.
    """
    global _global_outcome_store, _global_knowledge_store
    _global_outcome_store = None
    _global_knowledge_store = None


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
