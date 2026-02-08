"""
E2I Tool Composer - Performance Cache
Version: 4.3
Purpose: Caching layer for composition performance optimization (G6)

Features:
- Decomposition result caching with TTL
- Plan similarity matching
- Deterministic tool output caching
- LRU eviction policy
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# CACHE ENTRY
# ============================================================================


@dataclass
class CacheEntry:
    """A single cache entry with TTL support."""

    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 300.0  # 5 minutes default
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds

    def access(self) -> Any:
        """Mark entry as accessed and return value."""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value


# ============================================================================
# LRU CACHE WITH TTL
# ============================================================================


class LRUCache:
    """
    LRU cache with TTL support.

    Features:
    - Time-based expiration
    - Least-recently-used eviction
    - Hit/miss statistics
    """

    def __init__(
        self,
        max_size: int = 100,
        default_ttl_seconds: float = 300.0,
    ):
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired."""
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return entry.access()

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """Set value in cache with optional TTL."""
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            ttl_seconds=ttl_seconds or self.default_ttl_seconds,
        )

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find entry with oldest last_accessed time
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )
        del self._cache[oldest_key]
        logger.debug(f"Evicted LRU cache entry: {oldest_key[:50]}...")

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


# ============================================================================
# DECOMPOSITION CACHE
# ============================================================================


class DecompositionCache:
    """
    Cache for query decomposition results.

    Caches decomposition results to avoid re-processing identical
    or very similar queries. Uses normalized query hash as key.
    """

    def __init__(
        self,
        max_size: int = 50,
        ttl_seconds: float = 600.0,  # 10 minutes
    ):
        self._cache = LRUCache(max_size=max_size, default_ttl_seconds=ttl_seconds)

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent hashing."""
        # Lowercase, strip, collapse whitespace
        normalized = " ".join(query.lower().split())
        return normalized

    def _hash_query(self, query: str) -> str:
        """Create hash key from normalized query."""
        normalized = self._normalize_query(query)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, query: str) -> Optional[Any]:
        """Get cached decomposition for query."""
        key = self._hash_query(query)
        result = self._cache.get(key)

        if result:
            logger.debug(f"Decomposition cache hit for query hash: {key}")
        return result

    def set(self, query: str, decomposition: Any) -> None:
        """Cache decomposition result."""
        key = self._hash_query(query)
        self._cache.set(key, decomposition)
        logger.debug(f"Cached decomposition with key: {key}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "type": "decomposition",
            **self._cache.get_stats(),
        }


# ============================================================================
# PLAN SIMILARITY CACHE
# ============================================================================


class PlanSimilarityCache:
    """
    Cache for execution plans with similarity matching.

    Allows reusing plans for queries with similar structure
    even if not exactly identical.
    """

    def __init__(
        self,
        max_size: int = 30,
        ttl_seconds: float = 900.0,  # 15 minutes
        similarity_threshold: float = 0.8,
    ):
        self._cache = LRUCache(max_size=max_size, default_ttl_seconds=ttl_seconds)
        self.similarity_threshold = similarity_threshold

    def _extract_signature(self, decomposition: Any) -> Tuple[frozenset, int]:
        """Extract signature from decomposition for matching."""
        # Extract key features for similarity comparison
        intents = set()
        entities = set()
        dep_count = 0

        if hasattr(decomposition, "sub_questions"):
            for sq in decomposition.sub_questions:
                if hasattr(sq, "intent"):
                    intents.add(str(sq.intent))
                if hasattr(sq, "entities"):
                    entities.update(sq.entities)
                if hasattr(sq, "depends_on") and sq.depends_on:
                    dep_count += len(sq.depends_on)

        # Create frozen signature
        signature = frozenset(intents | entities)
        return signature, dep_count

    def _compute_similarity(
        self,
        sig1: Tuple[frozenset, int],
        sig2: Tuple[frozenset, int],
    ) -> float:
        """Compute similarity score between two signatures."""
        set1, deps1 = sig1
        set2, deps2 = sig2

        if not set1 or not set2:
            return 0.0

        # Jaccard similarity for sets
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = intersection / union if union > 0 else 0.0

        # Dependency count similarity
        max_deps = max(deps1, deps2, 1)
        dep_sim = 1.0 - abs(deps1 - deps2) / max_deps

        # Weighted combination
        return 0.7 * jaccard + 0.3 * dep_sim

    def _hash_signature(self, signature: Tuple[frozenset, int]) -> str:
        """Hash signature for cache key."""
        sig_set, dep_count = signature
        sig_str = f"{sorted(sig_set)}:{dep_count}"
        return hashlib.sha256(sig_str.encode()).hexdigest()[:16]

    def get_similar(self, decomposition: Any) -> Optional[Tuple[Any, float]]:
        """
        Find similar cached plan.

        Args:
            decomposition: The decomposition to find similar plans for

        Returns:
            Tuple of (cached_plan, similarity_score) if found, None otherwise
        """
        query_sig = self._extract_signature(decomposition)

        best_match = None
        best_similarity = 0.0

        # Check all cached entries for similarity
        for key in list(self._cache._cache.keys()):
            entry = self._cache._cache.get(key)
            if entry is None or entry.is_expired():
                continue

            cached_sig, cached_plan = entry.value
            similarity = self._compute_similarity(query_sig, cached_sig)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_plan

        if best_match and best_similarity >= self.similarity_threshold:
            logger.info(f"Plan similarity match found: {best_similarity:.2f}")
            return best_match, best_similarity

        return None

    def set(self, decomposition: Any, plan: Any) -> None:
        """Cache plan with its decomposition signature."""
        signature = self._extract_signature(decomposition)
        key = self._hash_signature(signature)
        self._cache.set(key, (signature, plan))
        logger.debug(f"Cached plan with signature key: {key}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "type": "plan_similarity",
            "similarity_threshold": self.similarity_threshold,
            **self._cache.get_stats(),
        }


# ============================================================================
# TOOL OUTPUT CACHE
# ============================================================================


class ToolOutputCache:
    """
    Cache for deterministic tool outputs.

    Caches results for tools marked as deterministic (same inputs
    always produce same outputs) to avoid redundant computation.
    """

    def __init__(
        self,
        max_size: int = 200,
        ttl_seconds: float = 300.0,  # 5 minutes
    ):
        self._cache = LRUCache(max_size=max_size, default_ttl_seconds=ttl_seconds)
        self._deterministic_tools: set = set()

    def register_deterministic_tool(self, tool_name: str) -> None:
        """Mark a tool as deterministic (cacheable)."""
        self._deterministic_tools.add(tool_name)
        logger.debug(f"Registered deterministic tool: {tool_name}")

    def is_deterministic(self, tool_name: str) -> bool:
        """Check if tool is deterministic."""
        return tool_name in self._deterministic_tools

    def _hash_inputs(self, tool_name: str, inputs: Dict[str, Any]) -> str:
        """Create cache key from tool name and inputs."""
        # Sort inputs for consistent hashing
        try:
            input_str = json.dumps(inputs, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback for non-serializable inputs
            input_str = str(sorted(inputs.items()))

        combined = f"{tool_name}:{input_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:20]

    def get(self, tool_name: str, inputs: Dict[str, Any]) -> Optional[Any]:
        """Get cached output for tool with given inputs."""
        if not self.is_deterministic(tool_name):
            return None

        key = self._hash_inputs(tool_name, inputs)
        result = self._cache.get(key)

        if result:
            logger.debug(f"Tool output cache hit: {tool_name}")
        return result

    def set(self, tool_name: str, inputs: Dict[str, Any], output: Any) -> None:
        """Cache tool output if tool is deterministic."""
        if not self.is_deterministic(tool_name):
            return

        key = self._hash_inputs(tool_name, inputs)
        self._cache.set(key, output)
        logger.debug(f"Cached output for deterministic tool: {tool_name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "type": "tool_output",
            "deterministic_tools": list(self._deterministic_tools),
            **self._cache.get_stats(),
        }


# ============================================================================
# UNIFIED CACHE MANAGER
# ============================================================================


class ToolComposerCacheManager:
    """
    Unified cache manager for Tool Composer performance optimization.

    Coordinates all caching layers:
    - Decomposition caching
    - Plan similarity matching
    - Tool output caching
    """

    _instance: Optional[ToolComposerCacheManager] = None
    _initialized: bool = False

    def __new__(cls, **kwargs) -> ToolComposerCacheManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        decomposition_max_size: int = 50,
        decomposition_ttl: float = 600.0,
        plan_max_size: int = 30,
        plan_ttl: float = 900.0,
        plan_similarity_threshold: float = 0.8,
        output_max_size: int = 200,
        output_ttl: float = 300.0,
    ):
        if self._initialized:
            return

        self.decomposition_cache = DecompositionCache(
            max_size=decomposition_max_size,
            ttl_seconds=decomposition_ttl,
        )

        self.plan_cache = PlanSimilarityCache(
            max_size=plan_max_size,
            ttl_seconds=plan_ttl,
            similarity_threshold=plan_similarity_threshold,
        )

        self.output_cache = ToolOutputCache(
            max_size=output_max_size,
            ttl_seconds=output_ttl,
        )

        # Default deterministic tools
        default_deterministic = [
            "psi_calculator",
            "power_calculator",
            "segment_ranker",
        ]
        for tool in default_deterministic:
            self.output_cache.register_deterministic_tool(tool)

        self._initialized = True
        logger.info("ToolComposerCacheManager initialized")

    def get_decomposition(self, query: str) -> Optional[Any]:
        """Get cached decomposition."""
        return self.decomposition_cache.get(query)

    def cache_decomposition(self, query: str, decomposition: Any) -> None:
        """Cache decomposition result."""
        self.decomposition_cache.set(query, decomposition)

    def get_similar_plan(self, decomposition: Any) -> Optional[Tuple[Any, float]]:
        """Get similar cached plan."""
        return self.plan_cache.get_similar(decomposition)

    def cache_plan(self, decomposition: Any, plan: Any) -> None:
        """Cache execution plan."""
        self.plan_cache.set(decomposition, plan)

    def get_tool_output(self, tool_name: str, inputs: Dict[str, Any]) -> Optional[Any]:
        """Get cached tool output."""
        return self.output_cache.get(tool_name, inputs)

    def cache_tool_output(self, tool_name: str, inputs: Dict[str, Any], output: Any) -> None:
        """Cache tool output."""
        self.output_cache.set(tool_name, inputs, output)

    def register_deterministic_tool(self, tool_name: str) -> None:
        """Mark tool as deterministic for caching."""
        self.output_cache.register_deterministic_tool(tool_name)

    def cleanup(self) -> Dict[str, int]:
        """Cleanup expired entries from all caches."""
        return {
            "decomposition": self.decomposition_cache._cache.cleanup_expired(),
            "plan": self.plan_cache._cache.cleanup_expired(),
            "output": self.output_cache._cache.cleanup_expired(),
        }

    def clear_all(self) -> None:
        """Clear all caches."""
        self.decomposition_cache._cache.clear()
        self.plan_cache._cache.clear()
        self.output_cache._cache.clear()
        logger.info("All caches cleared")

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches."""
        return {
            "decomposition": self.decomposition_cache.get_stats(),
            "plan": self.plan_cache.get_stats(),
            "output": self.output_cache.get_stats(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================================
# SINGLETON ACCESS
# ============================================================================


def get_cache_manager() -> ToolComposerCacheManager:
    """Get the global cache manager instance."""
    return ToolComposerCacheManager()


__all__ = [
    "CacheEntry",
    "LRUCache",
    "DecompositionCache",
    "PlanSimilarityCache",
    "ToolOutputCache",
    "ToolComposerCacheManager",
    "get_cache_manager",
]
