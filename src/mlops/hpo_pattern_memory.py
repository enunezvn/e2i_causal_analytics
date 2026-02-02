"""HPO Pattern Memory for Warm-Starting Hyperparameter Optimization.

This module provides procedural memory for successful HPO patterns,
enabling warm-starting of future optimization runs.

Features:
- Store successful HPO patterns after optimization
- Retrieve similar patterns based on algorithm, problem type, dataset size
- Track pattern effectiveness over time
- Integrate with procedural_memories table for unified memory

Usage:
    from src.mlops.hpo_pattern_memory import (
        store_hpo_pattern,
        find_similar_patterns,
        get_warmstart_hyperparameters,
        record_warmstart_outcome,
    )

    # After successful HPO
    pattern_id = await store_hpo_pattern(
        algorithm_name="XGBoost",
        problem_type="binary_classification",
        search_space={...},
        best_hyperparameters={...},
        best_value=0.92,
        metric="roc_auc",
        n_samples=10000,
        n_features=50,
    )

    # Before starting HPO - get warm-start suggestions
    warmstart = await get_warmstart_hyperparameters(
        algorithm_name="XGBoost",
        problem_type="binary_classification",
        n_samples=12000,
        n_features=45,
    )

Version: 1.0.0
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class HPOPatternInput:
    """Input for storing an HPO pattern."""

    algorithm_name: str
    problem_type: str
    search_space: Dict[str, Any]
    best_hyperparameters: Dict[str, Any]
    best_value: float
    optimization_metric: str
    n_trials: int
    n_completed: int
    n_pruned: int = 0
    duration_seconds: float = 0.0
    study_name: Optional[str] = None
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    n_classes: Optional[int] = None
    class_balance: Optional[float] = None
    feature_types: Optional[Dict[str, int]] = None
    experiment_id: Optional[str] = None


@dataclass
class HPOPatternMatch:
    """Result from pattern similarity search."""

    pattern_id: str
    algorithm_name: str
    problem_type: str
    best_hyperparameters: Dict[str, Any]
    best_value: float
    optimization_metric: str
    n_samples: Optional[int]
    n_features: Optional[int]
    similarity_score: float
    times_used: int


@dataclass
class WarmStartConfig:
    """Configuration for warm-starting HPO."""

    initial_hyperparameters: Dict[str, Any]
    pattern_id: str
    similarity_score: float
    original_best_value: float
    algorithm_name: str


# ============================================================================
# STORAGE FUNCTIONS
# ============================================================================


def _get_supabase_client():
    """Get Supabase client with lazy import."""
    try:
        from src.memory.services.factories import get_supabase_client

        return get_supabase_client()
    except ImportError:
        logger.warning("Supabase client not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get Supabase client: {e}")
        return None


async def store_hpo_pattern(pattern: HPOPatternInput) -> Optional[str]:
    """Store a successful HPO pattern for future warm-starting.

    Args:
        pattern: HPOPatternInput with pattern details

    Returns:
        Pattern ID if stored successfully, None otherwise
    """
    client = _get_supabase_client()
    if client is None:
        logger.debug("Supabase not available, skipping pattern storage")
        return None

    try:
        pattern_id = str(uuid.uuid4())

        # First, create a procedural memory entry
        procedure_record = {
            "procedure_id": pattern_id,
            "procedure_name": f"hpo_{pattern.algorithm_name}_{pattern.problem_type}",
            "procedure_type": "optimization",  # Use existing enum value
            "tool_sequence": json.dumps(
                {
                    "type": "hpo_pattern",
                    "algorithm": pattern.algorithm_name,
                    "best_params": pattern.best_hyperparameters,
                }
            ),
            "trigger_pattern": f"{pattern.algorithm_name} {pattern.problem_type} hyperparameter optimization",
            "detected_intent": "optimization",
            "applicable_agents": ["model_trainer"],
            "usage_count": 1,
            "success_count": 1,
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        client.table("procedural_memories").insert(procedure_record).execute()

        # Then, create the HPO-specific pattern record
        hpo_record = {
            "pattern_id": pattern_id,
            "procedure_id": pattern_id,
            "algorithm_name": pattern.algorithm_name,
            "problem_type": pattern.problem_type,
            "search_space": json.dumps(pattern.search_space),
            "best_hyperparameters": json.dumps(pattern.best_hyperparameters),
            "best_value": pattern.best_value,
            "optimization_metric": pattern.optimization_metric,
            "n_trials": pattern.n_trials,
            "n_completed": pattern.n_completed,
            "n_pruned": pattern.n_pruned,
            "duration_seconds": pattern.duration_seconds,
            "study_name": pattern.study_name,
            "n_samples": pattern.n_samples,
            "n_features": pattern.n_features,
            "n_classes": pattern.n_classes,
            "class_balance": pattern.class_balance,
            "feature_types": json.dumps(pattern.feature_types) if pattern.feature_types else None,
            "times_used_as_warmstart": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Remove None values
        hpo_record = {k: v for k, v in hpo_record.items() if v is not None}

        client.table("ml_hpo_patterns").insert(hpo_record).execute()

        logger.info(
            f"Stored HPO pattern {pattern_id}: {pattern.algorithm_name} "
            f"{pattern.problem_type} best_value={pattern.best_value:.4f}"
        )
        return pattern_id

    except Exception as e:
        logger.warning(f"Failed to store HPO pattern: {e}")
        return None


# ============================================================================
# RETRIEVAL FUNCTIONS
# ============================================================================


async def find_similar_patterns(
    algorithm_name: str,
    problem_type: str,
    n_samples: Optional[int] = None,
    n_features: Optional[int] = None,
    metric: Optional[str] = None,
    limit: int = 5,
) -> List[HPOPatternMatch]:
    """Find similar HPO patterns for potential warm-starting.

    Args:
        algorithm_name: Target algorithm (XGBoost, LightGBM, etc.)
        problem_type: Problem type (binary_classification, regression, etc.)
        n_samples: Number of samples in dataset
        n_features: Number of features in dataset
        metric: Optimization metric (roc_auc, f1, rmse, etc.)
        limit: Maximum patterns to return

    Returns:
        List of matching patterns ordered by similarity
    """
    client = _get_supabase_client()
    if client is None:
        logger.debug("Supabase not available, no patterns found")
        return []

    try:
        # Use the database function for similarity search
        result = client.rpc(
            "find_similar_hpo_patterns",
            {
                "p_algorithm_name": algorithm_name,
                "p_problem_type": problem_type,
                "p_n_samples": n_samples,
                "p_n_features": n_features,
                "p_metric": metric,
                "p_limit": limit,
            },
        ).execute()

        patterns = []
        for row in result.data or []:
            # Parse JSON fields
            best_hp = row.get("best_hyperparameters", {})
            if isinstance(best_hp, str):
                best_hp = json.loads(best_hp)

            patterns.append(
                HPOPatternMatch(
                    pattern_id=row["pattern_id"],
                    algorithm_name=row["algorithm_name"],
                    problem_type=row["problem_type"],
                    best_hyperparameters=best_hp,
                    best_value=row["best_value"],
                    optimization_metric=row["optimization_metric"],
                    n_samples=row.get("n_samples"),
                    n_features=row.get("n_features"),
                    similarity_score=row.get("similarity_score", 0.0),
                    times_used=row.get("times_used", 0),
                )
            )

        logger.debug(f"Found {len(patterns)} similar HPO patterns for {algorithm_name}")
        return patterns

    except Exception as e:
        logger.warning(f"Failed to find similar patterns: {e}")
        return []


async def get_warmstart_hyperparameters(
    algorithm_name: str,
    problem_type: str,
    n_samples: Optional[int] = None,
    n_features: Optional[int] = None,
    metric: Optional[str] = None,
    min_similarity: float = 0.5,
) -> Optional[WarmStartConfig]:
    """Get warm-start hyperparameters from the most similar successful pattern.

    Args:
        algorithm_name: Target algorithm
        problem_type: Problem type
        n_samples: Number of samples in dataset
        n_features: Number of features in dataset
        metric: Optimization metric
        min_similarity: Minimum similarity threshold

    Returns:
        WarmStartConfig if suitable pattern found, None otherwise
    """
    patterns = await find_similar_patterns(
        algorithm_name=algorithm_name,
        problem_type=problem_type,
        n_samples=n_samples,
        n_features=n_features,
        metric=metric,
        limit=1,
    )

    if not patterns:
        logger.debug(f"No HPO patterns found for {algorithm_name} {problem_type}")
        return None

    best_match = patterns[0]

    if best_match.similarity_score < min_similarity:
        logger.debug(
            f"Best pattern similarity {best_match.similarity_score:.2f} "
            f"below threshold {min_similarity}"
        )
        return None

    logger.info(
        f"Found warm-start pattern {best_match.pattern_id[:8]} "
        f"(similarity={best_match.similarity_score:.2f}, "
        f"original_value={best_match.best_value:.4f})"
    )

    return WarmStartConfig(
        initial_hyperparameters=best_match.best_hyperparameters,
        pattern_id=best_match.pattern_id,
        similarity_score=best_match.similarity_score,
        original_best_value=best_match.best_value,
        algorithm_name=best_match.algorithm_name,
    )


# ============================================================================
# OUTCOME TRACKING
# ============================================================================


async def record_warmstart_outcome(
    pattern_id: str,
    new_best_value: float,
    original_best_value: float,
) -> None:
    """Record the outcome of using a pattern for warm-starting.

    Args:
        pattern_id: ID of the pattern that was used
        new_best_value: Best value achieved in new HPO run
        original_best_value: Best value from the original pattern
    """
    client = _get_supabase_client()
    if client is None:
        return

    try:
        # Calculate improvement (positive = better)
        improvement = new_best_value - original_best_value

        client.rpc(
            "record_hpo_warmstart_usage",
            {"p_pattern_id": pattern_id, "p_improvement": improvement},
        ).execute()

        logger.info(
            f"Recorded warmstart outcome for {pattern_id[:8]}: improvement={improvement:+.4f}"
        )

    except Exception as e:
        logger.warning(f"Failed to record warmstart outcome: {e}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


async def get_pattern_stats(algorithm_name: Optional[str] = None) -> Dict[str, Any]:
    """Get statistics about stored HPO patterns.

    Args:
        algorithm_name: Optional filter by algorithm

    Returns:
        Dict with pattern statistics
    """
    client = _get_supabase_client()
    if client is None:
        return {"available": False, "reason": "Supabase not available"}

    try:
        query = client.table("ml_hpo_patterns").select(
            "algorithm_name, problem_type, best_value, times_used_as_warmstart"
        )

        if algorithm_name:
            query = query.eq("algorithm_name", algorithm_name)

        result = query.execute()
        patterns = result.data or []

        if not patterns:
            return {"total_patterns": 0, "by_algorithm": {}}

        # Aggregate stats
        by_algorithm = {}
        total_warmstarts = 0

        for p in patterns:
            algo = p["algorithm_name"]
            if algo not in by_algorithm:
                by_algorithm[algo] = {
                    "count": 0,
                    "problem_types": set(),
                    "avg_best_value": 0.0,
                    "values": [],
                }
            by_algorithm[algo]["count"] += 1
            by_algorithm[algo]["problem_types"].add(p["problem_type"])
            by_algorithm[algo]["values"].append(p["best_value"])
            total_warmstarts += p.get("times_used_as_warmstart", 0)

        # Calculate averages and convert sets
        for algo in by_algorithm:
            values = by_algorithm[algo]["values"]
            by_algorithm[algo]["avg_best_value"] = sum(values) / len(values) if values else 0
            by_algorithm[algo]["problem_types"] = list(by_algorithm[algo]["problem_types"])
            del by_algorithm[algo]["values"]

        return {
            "total_patterns": len(patterns),
            "total_warmstarts": total_warmstarts,
            "by_algorithm": by_algorithm,
        }

    except Exception as e:
        logger.warning(f"Failed to get pattern stats: {e}")
        return {"error": str(e)}


async def cleanup_old_patterns(
    days_old: int = 90,
    min_usage: int = 0,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Clean up old patterns that haven't been used.

    Args:
        days_old: Remove patterns older than this many days
        min_usage: Only remove if usage count below this
        dry_run: If True, just report what would be deleted

    Returns:
        Dict with cleanup results
    """
    client = _get_supabase_client()
    if client is None:
        return {"cleaned": 0, "reason": "Supabase not available"}

    try:
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()

        # Find patterns to clean
        query = (
            client.table("ml_hpo_patterns")
            .select("pattern_id, algorithm_name, created_at, times_used_as_warmstart")
            .lt("created_at", cutoff)
            .lte("times_used_as_warmstart", min_usage)
        )

        result = query.execute()
        patterns_to_clean = result.data or []

        if dry_run:
            return {
                "would_clean": len(patterns_to_clean),
                "patterns": [p["pattern_id"] for p in patterns_to_clean],
                "dry_run": True,
            }

        # Actually delete
        for pattern in patterns_to_clean:
            client.table("ml_hpo_patterns").delete().eq(
                "pattern_id", pattern["pattern_id"]
            ).execute()
            client.table("procedural_memories").delete().eq(
                "procedure_id", pattern["pattern_id"]
            ).execute()

        logger.info(f"Cleaned up {len(patterns_to_clean)} old HPO patterns")
        return {"cleaned": len(patterns_to_clean), "dry_run": False}

    except Exception as e:
        logger.warning(f"Failed to cleanup patterns: {e}")
        return {"error": str(e)}
