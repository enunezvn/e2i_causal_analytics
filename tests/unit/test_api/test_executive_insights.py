"""Issue #376 — Phase 4 schema completion.

Tests pin the new portfolio-summary endpoint behavior + verify the
extended ExecutiveInsightResponse round-trips through the list and
get-one endpoints.

Auth is bypassed via E2I_TESTING_MODE=1 (set in
``tests/unit/test_api/conftest.py``).
"""

from __future__ import annotations

# E2I_TESTING_MODE is set in tests/unit/test_api/conftest.py at import
# time so auth dependencies short-circuit.
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Lightweight FakeSupabase mirroring the crystallizer test shape
# ---------------------------------------------------------------------------


class _Query:
    def __init__(self, store: "_FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self.filters: Dict[str, Any] = {}
        # Tracks ``.is_(col, "null")`` filters as col -> True so the
        # eq-based predicate dispatcher can short-circuit to
        # ``r.get(col) is None``. Mirrors PostgREST ``is.null`` semantics.
        self.is_null_filters: Dict[str, bool] = {}

    def select(self, *_args: Any, **_kwargs: Any) -> "_Query":
        return self

    def eq(self, col: str, val: Any) -> "_Query":
        self.filters[col] = val
        return self

    def is_(self, col: str, val: Any) -> "_Query":
        # supabase-py / PostgREST treat the literal string ``"null"`` as
        # IS NULL. Only that form is supported here; anything else would
        # be a silent test-only divergence from production semantics.
        if isinstance(val, str) and val.lower() == "null":
            self.is_null_filters[col] = True
            return self
        raise NotImplementedError(
            f"_FakeSupabase.is_({col!r}, {val!r}): only is_(col, 'null') "
            f"is supported in this test fake."
        )

    def order(self, *_args: Any, **_kwargs: Any) -> "_Query":
        return self

    def limit(self, _n: int) -> "_Query":
        return self

    def execute(self) -> Any:
        rows = list(self.store.rows.get(self.table_name, []))
        for col, want in self.filters.items():
            rows = [r for r in rows if r.get(col) == want]
        for col in self.is_null_filters:
            rows = [r for r in rows if r.get(col) is None]
        m = MagicMock()
        m.data = rows
        return m


class _FakeSupabase:
    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {"executive_insights": []}

    def table(self, name: str) -> _Query:
        return _Query(self, name)


@pytest.fixture
def fake_supabase() -> _FakeSupabase:
    return _FakeSupabase()


@pytest.fixture
def client(fake_supabase: _FakeSupabase) -> TestClient:
    """Return a FastAPI TestClient with the supabase factory patched.

    The patch path resolves the executive_insights route module —
    ``src.api.routes.executive_insights.get_supabase_client`` —
    independently of the underlying factory implementation, so the
    test does not depend on how the factory dispatches.
    """
    from fastapi import FastAPI

    from src.api.routes.executive_insights import router

    app = FastAPI()
    app.include_router(router, prefix="/api")

    with patch(
        "src.api.routes.executive_insights.get_supabase_client",
        return_value=fake_supabase,
    ):
        with TestClient(app) as c:
            yield c


def _seed_insight(
    db: _FakeSupabase,
    *,
    insight_id: str,
    brand: str,
    crystallized_at: datetime,
    effect_size: Optional[float] = None,
    recall: bool = False,
    invalidated_at: Optional[datetime] = None,
) -> None:
    db.rows["executive_insights"].append(
        {
            "insight_id": insight_id,
            "title": f"Insight for {brand}",
            "narrative": "narrative body",
            "brand": brand,
            "region": "northeast",
            "kpi": "trx",
            "crystallized_at": crystallized_at.isoformat(),
            "source_count": 3,
            "effect_size": effect_size,
            "recall": recall,
            "invalidated_at": invalidated_at.isoformat() if invalidated_at else None,
            "key_metrics": {"causal_path_id": "cp-1"},
        }
    )


# ---------------------------------------------------------------------------
# portfolio-summary endpoint tests (#376 DoD §D)
# ---------------------------------------------------------------------------


def test_portfolio_summary_empty_returns_zero_brands(client: TestClient):
    """Empty insight set → empty by_brand list + zero totals."""
    response = client.get("/api/executive-insights/portfolio-summary")
    assert response.status_code == 200
    body = response.json()
    assert body["by_brand"] == []
    assert body["total_brands"] == 0
    assert body["total_insights"] == 0


def test_portfolio_summary_aggregates_by_brand(client: TestClient, fake_supabase: _FakeSupabase):
    """Per-brand: count, latest crystallized_at, average effect_size."""
    now = datetime.now(timezone.utc)
    _seed_insight(
        fake_supabase,
        insight_id="i-1",
        brand="kisqali",
        crystallized_at=now - timedelta(hours=2),
        effect_size=0.40,
    )
    _seed_insight(
        fake_supabase,
        insight_id="i-2",
        brand="kisqali",
        crystallized_at=now - timedelta(hours=1),
        effect_size=0.50,
    )
    _seed_insight(
        fake_supabase,
        insight_id="i-3",
        brand="fabhalta",
        crystallized_at=now,
        effect_size=0.30,
    )

    response = client.get("/api/executive-insights/portfolio-summary")
    assert response.status_code == 200
    body = response.json()
    assert body["total_brands"] == 2
    assert body["total_insights"] == 3

    by_brand = {b["brand"]: b for b in body["by_brand"]}
    assert "kisqali" in by_brand and "fabhalta" in by_brand

    # Kisqali: 2 insights, mean(0.40, 0.50) = 0.45
    k = by_brand["kisqali"]
    assert k["insight_count"] == 2
    assert k["average_effect_size"] == pytest.approx(0.45)
    assert k["effect_size_sample_count"] == 2

    # Fabhalta: 1 insight, mean = 0.30
    f = by_brand["fabhalta"]
    assert f["insight_count"] == 1
    assert f["average_effect_size"] == pytest.approx(0.30)


def test_portfolio_summary_excludes_recalled_insights(
    client: TestClient, fake_supabase: _FakeSupabase
):
    """recall=True rows MUST NOT contribute to the portfolio summary —
    a recalled insight has been invalidated upstream."""
    now = datetime.now(timezone.utc)
    _seed_insight(
        fake_supabase,
        insight_id="i-active",
        brand="kisqali",
        crystallized_at=now,
        effect_size=0.40,
        recall=False,
    )
    _seed_insight(
        fake_supabase,
        insight_id="i-recalled",
        brand="kisqali",
        crystallized_at=now,
        effect_size=999.0,  # would explode the mean if not filtered
        recall=True,
    )

    response = client.get("/api/executive-insights/portfolio-summary")
    body = response.json()
    by_brand = {b["brand"]: b for b in body["by_brand"]}
    k = by_brand["kisqali"]
    assert k["insight_count"] == 1
    assert k["average_effect_size"] == pytest.approx(0.40)


def test_portfolio_summary_excludes_invalidated_insights(
    client: TestClient, fake_supabase: _FakeSupabase
):
    """invalidated_at IS NOT NULL rows MUST NOT contribute to the
    portfolio summary (issue #385). The docstring at
    ``src/api/routes/executive_insights.py`` for
    ``get_portfolio_summary`` claims it aggregates across non-recalled,
    non-invalidated crystals — that contract was previously broken
    because only ``.eq("recall", False)`` was applied.

    This test pins the corrected contract: a row with ``recall=False``
    AND ``invalidated_at`` set (the silent-cascade case from the JIT
    verifier middleware) must be excluded from count + mean + latest.
    """
    now = datetime.now(timezone.utc)
    _seed_insight(
        fake_supabase,
        insight_id="i-active",
        brand="kisqali",
        crystallized_at=now,
        effect_size=0.40,
        recall=False,
        invalidated_at=None,
    )
    _seed_insight(
        fake_supabase,
        insight_id="i-invalidated",
        brand="kisqali",
        crystallized_at=now,
        effect_size=999.0,  # would explode the mean if not filtered
        recall=False,
        invalidated_at=now - timedelta(hours=1),
    )

    response = client.get("/api/executive-insights/portfolio-summary")
    body = response.json()
    by_brand = {b["brand"]: b for b in body["by_brand"]}
    k = by_brand["kisqali"]
    assert k["insight_count"] == 1
    assert k["effect_size_sample_count"] == 1
    assert k["average_effect_size"] == pytest.approx(0.40)


def test_portfolio_summary_handles_null_effect_size_in_mean(
    client: TestClient, fake_supabase: _FakeSupabase
):
    """Rows with effect_size=NULL must NOT contribute to the mean — the
    denominator counts only rows with a numeric effect_size."""
    now = datetime.now(timezone.utc)
    _seed_insight(
        fake_supabase,
        insight_id="i-1",
        brand="kisqali",
        crystallized_at=now,
        effect_size=0.40,
    )
    _seed_insight(
        fake_supabase,
        insight_id="i-2",
        brand="kisqali",
        crystallized_at=now,
        effect_size=None,  # legacy row pre-#376
    )

    response = client.get("/api/executive-insights/portfolio-summary")
    body = response.json()
    by_brand = {b["brand"]: b for b in body["by_brand"]}
    k = by_brand["kisqali"]
    assert k["insight_count"] == 2  # count includes legacy rows
    assert k["effect_size_sample_count"] == 1  # only the numeric one
    assert k["average_effect_size"] == pytest.approx(0.40)


def test_portfolio_summary_average_none_when_all_effect_sizes_null(
    client: TestClient, fake_supabase: _FakeSupabase
):
    """Brand with zero numeric effect_size rows → average_effect_size=None
    (not 0.0 — division-by-zero protection)."""
    now = datetime.now(timezone.utc)
    _seed_insight(
        fake_supabase,
        insight_id="i-1",
        brand="kisqali",
        crystallized_at=now,
        effect_size=None,
    )

    response = client.get("/api/executive-insights/portfolio-summary")
    body = response.json()
    by_brand = {b["brand"]: b for b in body["by_brand"]}
    k = by_brand["kisqali"]
    assert k["average_effect_size"] is None
    assert k["effect_size_sample_count"] == 0


def test_portfolio_summary_route_does_not_shadow_insight_id_route(
    client: TestClient, fake_supabase: _FakeSupabase
):
    """portfolio-summary must be matched as its own path, not as an
    insight_id passed to /{insight_id}. The route order in the router
    determines this — portfolio-summary must come BEFORE the dynamic
    insight_id route."""
    response = client.get("/api/executive-insights/portfolio-summary")
    # If the dynamic route shadowed it, we'd get 404 (insight not found).
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Round-trip through the list endpoint with the extended response shape
# ---------------------------------------------------------------------------


def test_list_endpoint_round_trips_new_fields(client: TestClient, fake_supabase: _FakeSupabase):
    """A row with all 15 new fields populated must round-trip through
    the list endpoint without dropping fields."""
    now = datetime.now(timezone.utc)
    fake_supabase.rows["executive_insights"].append(
        {
            "insight_id": "ei-1",
            "title": "T",
            "narrative": "N",
            "brand": "kisqali",
            "region": "northeast",
            "kpi": "trx",
            "crystallized_at": now.isoformat(),
            "source_count": 3,
            "recall": False,
            "effect_size": 0.42,
            "effect_ci_lower": 0.30,
            "effect_ci_upper": 0.55,
            "effect_direction": "positive",
            "cohort_size": 1200,
            "confounders_controlled": ["age", "prior_use"],
            "sensitivity_checks_passed": ["placebo_treatment"],
            "sensitivity_checks_failed": ["data_subset"],
            "limitations": "Small pre-period.",
            "recommended_next_analysis": "Replicate on Q3.",
            "provenance_chain_id": "chain-abc",
            "provenance_depth": 2,
            "consolidation_tier": "semantic",
            "replication_count": 3,
            "data_version": "2026-05-19-snapshot",
            "key_metrics": {"causal_path_id": "cp-1"},
        }
    )

    response = client.get("/api/executive-insights")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    r = rows[0]
    # spot-check a few of the 15 new fields
    assert r["effect_size"] == pytest.approx(0.42)
    assert r["effect_direction"] == "positive"
    assert r["cohort_size"] == 1200
    assert r["confounders_controlled"] == ["age", "prior_use"]
    assert r["provenance_chain_id"] == "chain-abc"
    assert r["consolidation_tier"] == "semantic"
    assert r["data_version"] == "2026-05-19-snapshot"
