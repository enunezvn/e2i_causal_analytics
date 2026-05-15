"""Sentinel watcher registry and dispatcher."""

from src.memory.sentinels.registry import (
    SentinelDispatchResult,
    SentinelEvaluationError,
    dispatch_sentinels,
    evaluate_sentinel,
    register_sentinel,
)

__all__ = [
    "SentinelDispatchResult",
    "SentinelEvaluationError",
    "dispatch_sentinels",
    "evaluate_sentinel",
    "register_sentinel",
]
