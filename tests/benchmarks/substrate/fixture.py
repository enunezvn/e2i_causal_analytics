"""Substrate readiness + connector injection for the hybrid latency benchmark (#414)."""

from __future__ import annotations

import os

import src.rag.memory_connector as _mc
from tests.benchmarks.substrate.direct_sql_connector import DirectSQLMemoryConnector


def substrate_ready() -> bool:
    """True when the local pg substrate is configured for this run."""
    return os.getenv("BENCH_SUBSTRATE") == "local_pg" and bool(os.getenv("BENCH_PG_DSN"))


def make_connector() -> DirectSQLMemoryConnector:
    return DirectSQLMemoryConnector(os.environ["BENCH_PG_DSN"])


def inject(connector) -> None:
    """Install a connector as the process-wide singleton."""
    _mc._memory_connector = connector


def reset() -> None:
    _mc.reset_memory_connector()
