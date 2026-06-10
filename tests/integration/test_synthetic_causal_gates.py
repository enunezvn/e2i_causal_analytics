"""Shard 11 — faithful acceptance-gate harness for the synthetic causal-validation
dataset. NO MOCKING. Each test wraps one INDEX global-gate (1-11) and asserts a
MEASURED value against the LIVE docker Supabase. Run gate: E2I_DB_INTEGRATION=1 -n0.

Many gates only go GREEN after the orchestrator runs the full synthetic load against
the faithful docker Supabase (rolling dates, recoverable DGP, two-arm triggers, the
input_model dispatcher bridges, the ground-truth sidecar). Before that load they fail
RED for the RIGHT reason — no-substrate / stale-window / missing-artifact — never on a
typo. The harness binds only to columns/RPCs/signatures verified to exist (2026-06-10).
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 to run (-n0)",
)

from scripts.validate_synthetic_causal import _kpi  # noqa: E402
from src.api.dependencies.supabase_client import get_supabase  # noqa: E402


@pytest.fixture(scope="module")
def client():
    c = get_supabase()
    if c is None:
        pytest.skip("no Supabase client (SUPABASE_URL/ANON_KEY unset)")
    return c


def test_disproof_synthetic_substrate_present(client):
    """CHEAPEST DISPROOF: synthetic rows must EXIST + be tagged before any gate
    can recover a known truth. Fail loud if a producing shard (02-09) regressed."""
    # RECONCILED: business_metrics has no `id` column (PK is `metric_id`); selecting
    # a non-existent column 42703-crashes. `select("*", count="exact")` counts rows
    # without binding a column name (verified \d business_metrics, 2026-06-10).
    resp = (
        client.table("business_metrics")
        .select("*", count="exact")
        .eq("is_synthetic", True)
        .limit(1)
        .execute()
    )
    assert (resp.count or 0) > 0, (
        "no synthetic business_metrics rows tagged is_synthetic=true; "
        "run scripts/load_synthetic_data.py (Shard 02) before the gates"
    )
