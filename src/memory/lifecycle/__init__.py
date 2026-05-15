"""Insight lifecycle: consolidation (working->semantic->procedural) and cascading invalidation."""

from src.memory.lifecycle.consolidator import (
    ConsolidationResult,
    Consolidator,
    consolidate_insights,
)
from src.memory.lifecycle.invalidator import (
    CascadeResult,
    cascade_invalidate,
)

__all__ = [
    "ConsolidationResult",
    "Consolidator",
    "consolidate_insights",
    "CascadeResult",
    "cascade_invalidate",
]
