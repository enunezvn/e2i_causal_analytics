"""Repeated train/val/test splitting for the model_trainer agent.

Phase 1 W3-lite Day 4 (shard 17 W3 row Day 4, shard 21 §A). Provides the
``RepeatedStratifiedSplitter`` consumed by the ``_run_repeated_splits``
orchestrator in ``agent.py`` when ``evaluation_mode == "repeated_k10"``.
"""

from __future__ import annotations

from .repeated_splitter import FoldSpec, RepeatedStratifiedSplitter

__all__ = ["FoldSpec", "RepeatedStratifiedSplitter"]
