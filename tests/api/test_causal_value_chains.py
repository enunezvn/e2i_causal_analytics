"""Tests for GET /api/causal/value-chains.

The Home dashboard "Primary Causal Value Chains" section sources the REAL
discovered chains from the ``causal_paths`` table — ranked by |effect| x
confidence, de-duplicated by pathway, scoped by the brand/region dropdowns.
These tests pin the pure mappers and the endpoint's filter/rank/dedup behavior
against a mocked Supabase client (no live DB).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes.causal import (
    _causal_path_to_graphpath,
    _chain_node_sequence,
    _chain_score,
)

client = TestClient(app)


def _row(start, mids, end, eff, conf, brand="Kisqali", region="south",
         method="backdoor.linear_regression"):
    return {
        "path_id": f"scp_{start}_{end}_{'_'.join(mids)}",
        "start_node": start,
        "intermediate_nodes": list(mids),
        "end_node": end,
        "causal_chain": {"nodes": [start, *mids, end]},
        "causal_effect_size": eff,
        "confidence_level": conf,
        "method_used": method,
        "validation_status": "validated",
        "brand": brand,
        "region": region,
        "path_length": len(mids) + 1,
    }


class _FakeQuery:
    """Records .eq() calls; every builder method returns self; execute() is async."""

    def __init__(self, rows):
        self._rows = rows
        self.eq_calls = []

    def select(self, *a, **k):
        return self

    def eq(self, col, val):
        self.eq_calls.append((col, val))
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    async def execute(self):
        return MagicMock(data=self._rows)


class _FakeClient:
    def __init__(self, q):
        self._q = q

    def table(self, name):
        return self._q


def _patch_client(rows):
    q = _FakeQuery(rows)
    cm = patch(
        "src.memory.services.factories.get_async_supabase_client",
        AsyncMock(return_value=_FakeClient(q)),
    )
    return cm, q


# ---------------------------------------------------------------------------
# Pure mappers
# ---------------------------------------------------------------------------

def test_chain_node_sequence_prefers_causal_chain():
    row = _row("treatment_arm", ["engagement_score"], "treatment_initiated", 0.55, 0.86)
    assert _chain_node_sequence(row) == [
        "treatment_arm", "engagement_score", "treatment_initiated"
    ]


def test_chain_node_sequence_fallback_without_causal_chain():
    row = _row("a", ["m1", "m2"], "z", 0.3, 0.7)
    row["causal_chain"] = None
    assert _chain_node_sequence(row) == ["a", "m1", "m2", "z"]


def test_path_to_graphpath_puts_ate_on_terminal_edge():
    row = _row("treatment_arm", ["engagement_score"], "treatment_initiated", 0.549, 0.86)
    gp = _causal_path_to_graphpath(row)
    assert [n.name for n in gp.nodes] == [
        "treatment_arm", "engagement_score", "treatment_initiated"
    ]
    assert gp.total_confidence == pytest.approx(0.86)
    # The chain-level ATE rides the TERMINAL edge (what the dashboard reads).
    assert gp.relationships[-1].properties["ate_estimate"] == pytest.approx(0.549)
    assert gp.relationships[0].properties["method"] == "backdoor.linear_regression"
    # effect_size is categorical in this platform — never emitted as a number.
    assert "effect_size" not in gp.relationships[-1].properties


def test_chain_score_is_abs_effect_times_confidence():
    assert _chain_score({"causal_effect_size": -0.5, "confidence_level": 0.8}) == pytest.approx(0.4)
    assert _chain_score({"causal_effect_size": None, "confidence_level": 0.9}) == 0.0


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

def test_value_chains_ranks_and_dedupes_and_scopes():
    rows = [
        _row("treatment_arm", ["engagement_score"], "treatment_initiated", 0.55, 0.86),  # .473
        _row("treatment_arm", ["prior_therapy"], "treatment_initiated", 0.54, 0.94),     # .508 top
        _row("treatment_arm", ["engagement_score"], "treatment_initiated", 0.40, 0.50),  # dup -> drop
        _row("hcp_reach", [], "nrx", 0.20, 0.60),                                         # .12
    ]
    cm, q = _patch_client(rows)
    with cm:
        resp = client.get("/api/causal/value-chains?brand=Kisqali&region=south&limit=3")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total_chains"] == 3  # duplicate pathway collapsed
    # Ranked by |effect| x confidence: prior_therapy (.508) first.
    assert data["chains"][0]["nodes"][1]["name"] == "prior_therapy"
    assert data["aggregate_effect"] is None  # no fabricated scalar aggregate
    # Brand/region scope applied.
    assert ("brand", "Kisqali") in q.eq_calls
    assert ("region", "south") in q.eq_calls


def test_value_chains_portfolio_skips_brand_region_filter():
    rows = [_row("treatment_arm", ["adherence"], "treatment_initiated", 0.5, 0.9)]
    cm, q = _patch_client(rows)
    with cm:
        resp = client.get("/api/causal/value-chains?brand=All&region=All%20US&limit=3")
    assert resp.status_code == 200, resp.text
    # 'All' sentinels => no brand/region predicate (portfolio view).
    assert ("brand", "All") not in q.eq_calls
    assert not any(col == "region" for col, _ in q.eq_calls)
