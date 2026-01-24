"""
E2I Agentic Memory - Evidence Evaluation Cache

Cache for LLM-based evidence evaluation results during multi-hop investigation.
Reduces redundant LLM calls by caching evaluation results based on goal + evidence content.

Configuration via environment:
    E2I_EVIDENCE_CACHE: "true" (default) to enable caching
    E2I_EVIDENCE_CACHE_TTL: TTL in seconds (default: 3600)
    E2I_EVIDENCE_CACHE_SIZE: Max entries (default: 1000)
"""

import hashlib
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class EvidenceEvaluationCache:
    """
    Cache for LLM-based evidence evaluation results.

    Reduces redundant LLM calls during multi-hop investigation by caching
    evaluation results based on goal + evidence content. Uses a TTL-based
    expiration and LRU eviction strategy.

    Configuration via environment:
        E2I_EVIDENCE_CACHE: "true" (default) to enable caching
        E2I_EVIDENCE_CACHE_TTL: TTL in seconds (default: 3600)
        E2I_EVIDENCE_CACHE_SIZE: Max entries (default: 1000)

    Usage:
        cache = EvidenceEvaluationCache()

        # Check cache before LLM call
        cached = cache.get(goal, evidence_summary)
        if cached:
            return cached

        # After LLM call, store result
        result = await llm.complete(prompt)
        cache.set(goal, evidence_summary, result)
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        ttl_seconds: Optional[float] = None,
    ):
        """
        Initialize the evidence evaluation cache.

        Args:
            max_size: Maximum cache entries. Defaults to E2I_EVIDENCE_CACHE_SIZE or 1000.
            ttl_seconds: Entry TTL in seconds. Defaults to E2I_EVIDENCE_CACHE_TTL or 3600.
        """
        if max_size is None:
            max_size = int(os.environ.get("E2I_EVIDENCE_CACHE_SIZE", "1000"))
        if ttl_seconds is None:
            ttl_seconds = float(os.environ.get("E2I_EVIDENCE_CACHE_TTL", "3600"))

        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: Dict[str, Tuple[str, float]] = {}  # key -> (result, timestamp)
        self._hits = 0
        self._misses = 0

    def _make_key(self, goal: str, evidence_summary: str) -> str:
        """
        Create a cache key from goal and evidence summary.

        Uses SHA-256 hash for consistent key length and collision resistance.
        """
        content = f"{goal.strip().lower()}|{evidence_summary.strip()}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, goal: str, evidence_summary: str) -> Optional[str]:
        """
        Get cached evaluation result.

        Args:
            goal: The investigation goal
            evidence_summary: Summary of evidence collected

        Returns:
            Cached result string or None if not found/expired
        """
        key = self._make_key(goal, evidence_summary)

        if key in self._cache:
            result, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp < self._ttl:
                self._hits += 1
                logger.debug(f"Evidence cache hit (hits={self._hits})")
                return result

            # Expired - remove entry
            del self._cache[key]

        self._misses += 1
        return None

    def set(self, goal: str, evidence_summary: str, result: str) -> None:
        """
        Cache an evaluation result.

        Args:
            goal: The investigation goal
            evidence_summary: Summary of evidence collected
            result: The evaluation result to cache
        """
        # Evict oldest entries if at capacity
        if len(self._cache) >= self._max_size:
            self._evict_oldest(count=self._max_size // 10)  # Evict 10%

        key = self._make_key(goal, evidence_summary)
        self._cache[key] = (result, time.time())

    def _evict_oldest(self, count: int) -> None:
        """Evict the oldest entries from cache."""
        if not self._cache or count <= 0:
            return

        # Sort by timestamp and remove oldest
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k][1],
        )

        for key in sorted_keys[: min(count, len(sorted_keys))]:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Evidence evaluation cache cleared")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "ttl_seconds": self._ttl,
        }


# Module-level cache singleton
_evidence_cache: Optional[EvidenceEvaluationCache] = None


def get_evidence_cache() -> EvidenceEvaluationCache:
    """Get or create the evidence evaluation cache singleton."""
    global _evidence_cache
    if _evidence_cache is None:
        _evidence_cache = EvidenceEvaluationCache()
    return _evidence_cache


def is_evidence_cache_enabled() -> bool:
    """Check if evidence caching is enabled."""
    return os.environ.get("E2I_EVIDENCE_CACHE", "true").lower() == "true"
