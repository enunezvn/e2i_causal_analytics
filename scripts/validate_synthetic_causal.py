"""Acceptance-gate harness for the synthetic causal-validation dataset (Shard 11).

Runs INDEX gates 1-11 against the FAITHFUL docker Supabase. Each gate is a discrete,
individually-runnable function returning a GateResult(name, ok, measured, expected).
OOM-safe: LOKY_MAX_CPU_COUNT=1; no full-tree work. Usage::

    LOKY_MAX_CPU_COUNT=1 python scripts/validate_synthetic_causal.py            # all
    LOKY_MAX_CPU_COUNT=1 python scripts/validate_synthetic_causal.py --gate 3   # one

REASON-BEFORE-RULES / anti-fabrication: every column, RPC id, agent entrypoint, and
orchestrator state key below is VERIFIED against the live docker Supabase + real
source (2026-06-10). Where the plan's sketch diverged from reality the harness aligns
to reality with an inline ``RECONCILED:`` note. A gate that cannot recover a real
value FAILS (ok=False) — it never papers over absent substrate with a fabricated pass.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")  # OOM discipline (INDEX §SHARED)

from src.api.dependencies.supabase_client import get_supabase  # noqa: E402

# INDEX §SHARED brand_type enum (verified: src/ml/synthetic/config.py Brand).
BRANDS = ["Remibrutinib", "Kisqali", "Fabhalta"]


@dataclass
class GateResult:
    name: str
    ok: bool
    measured: object
    expected: str


def _kpi(client, query_id: str, params: list) -> list:
    """Call the kpi_query allowlist RPC exactly as the runtime does.

    VERIFIED: ``kpi_query(query_id text, params jsonb DEFAULT '[]')`` RETURNS SETOF
    json (database/migrations/044_kpi_query_allowlist.sql:49). supabase-py JSON-encodes
    the ``params`` list into the jsonb arg; ``.data`` is the row list of json objects.
    """
    return (
        client.rpc("kpi_query", {"query_id": query_id, "params": params}).execute().data
    ) or []


def _banner(r: GateResult) -> str:
    tag = "PASS" if r.ok else "FAIL"
    return f"[{tag}] {r.name}: measured={r.measured!r} expected={r.expected}"
