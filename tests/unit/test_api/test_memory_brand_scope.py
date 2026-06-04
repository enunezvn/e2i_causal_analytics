"""Brand-scoping enforcement on the memory READ routes (H1).

Memory-review finding H1 (cross-tenant PHI reads):

* ``POST /memory/search`` forwarded ``request.filters`` straight into
  ``hybrid_search`` with no grant check. Omitting ``brand`` (RPC predicate
  ``filters->>'brand' IS NULL`` => every brand) or naming another tenant's
  brand returned cross-tenant episodic memory content.
* ``GET /memory/semantic/paths`` traversed the FalkorDB causal graph with no
  tenant scope at all. The graph nodes carry no ``brand`` property (verified
  on the live ``e2i_causal`` graph), so per-brand scoping is not possible at
  the graph level — the fail-closed decision is to gate graph traversal to
  cross-brand admins until ``brand`` is added to the causal-graph nodes.

These tests invoke the route handlers directly with patched dependencies,
mirroring ``test_sentinels_brand_auth.py``. Imports are module-level (matching
``test_routes/test_memory.py``) so the heavy ``src.rag.retriever`` import
happens at collection, not under the per-test timeout.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.routes.memory import (
    MemorySearchRequest,
    query_semantic_paths,
    search_memory,
)


def _viewer(*brands: str) -> Dict[str, Any]:
    return {"sub": "u-view", "role": "viewer", "brands": list(brands)}


def _admin() -> Dict[str, Any]:
    return {"sub": "u-admin", "role": "admin", "brands": []}


# ---------------------------------------------------------------------------
# search_memory
# ---------------------------------------------------------------------------


async def test_search_non_admin_injects_brand_filter_and_strips_graph() -> None:
    """A scoped viewer's search is pinned to their granted brand and the
    (non-brand-scopable) graph-traversal params are dropped."""
    req = MemorySearchRequest(query="why did TRx drop", entities=["e1"], kpi_name="TRx")
    with patch("src.api.routes.memory.hybrid_search", new=AsyncMock(return_value=[])) as hs:
        await search_memory(req, user=_viewer("Brand-X"))

    hs.assert_awaited_once()
    kwargs = hs.await_args.kwargs
    assert kwargs["filters"]["brand"] == "Brand-X"
    assert kwargs["entities"] is None
    assert kwargs["kpi_name"] is None


async def test_search_non_admin_out_of_grant_brand_returns_empty() -> None:
    """Requesting an out-of-grant brand returns a defensive empty result and
    never reaches ``hybrid_search`` (no existence leak, no cross-tenant read)."""
    req = MemorySearchRequest(query="q", filters={"brand": "Brand-Y"})
    with patch("src.api.routes.memory.hybrid_search", new=AsyncMock(return_value=[])) as hs:
        resp = await search_memory(req, user=_viewer("Brand-X"))

    assert resp.results == []
    assert resp.total_results == 0
    hs.assert_not_awaited()


async def test_search_non_admin_no_grants_returns_empty() -> None:
    """A viewer with no brand grants cannot tenant-scope and gets nothing."""
    req = MemorySearchRequest(query="q")
    with patch("src.api.routes.memory.hybrid_search", new=AsyncMock(return_value=[])) as hs:
        resp = await search_memory(req, user=_viewer())

    assert resp.results == []
    hs.assert_not_awaited()


async def test_search_admin_passthrough_keeps_brand_and_graph() -> None:
    """A cross-brand admin's search is unrestricted: requested brand, entities
    and kpi_name all pass through unchanged."""
    req = MemorySearchRequest(
        query="q", filters={"brand": "Brand-Z"}, entities=["e1"], kpi_name="TRx"
    )
    with patch("src.api.routes.memory.hybrid_search", new=AsyncMock(return_value=[])) as hs:
        await search_memory(req, user=_admin())

    kwargs = hs.await_args.kwargs
    assert kwargs["filters"]["brand"] == "Brand-Z"
    assert kwargs["entities"] == ["e1"]
    assert kwargs["kpi_name"] == "TRx"


# ---------------------------------------------------------------------------
# query_semantic_paths
# ---------------------------------------------------------------------------


async def test_semantic_paths_non_admin_forbidden() -> None:
    """The brand-unscopable causal-graph traversal is admin-only (fail-closed)."""
    with pytest.raises(HTTPException) as exc:
        await query_semantic_paths(kpi_name="TRx", user=_viewer("Brand-X"))
    assert exc.value.status_code == 403


async def test_semantic_paths_admin_allowed() -> None:
    """A cross-brand admin may traverse the causal graph."""
    sem = MagicMock()
    sem.find_causal_paths_for_kpi.return_value = []
    with patch("src.api.routes.memory.get_semantic_memory", return_value=sem):
        resp = await query_semantic_paths(kpi_name="TRx", user=_admin())

    assert resp.total_paths == 0
    sem.find_causal_paths_for_kpi.assert_called_once()
