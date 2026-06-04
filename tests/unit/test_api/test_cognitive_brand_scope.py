"""Brand-scoping enforcement on the cognitive query route (H1).

``POST /cognitive/query`` forwarded the caller-supplied ``request.brand`` into
``hybrid_search`` (via ``_build_filters``) with no grant check, so a non-admin
could retrieve another tenant's episodic memories by naming their brand, or
retrieve every brand by omitting it. The fix mirrors the memory-search route:
scope the episodic filter to the caller's grant and drop the (non-brand-
scopable) graph-traversal ``kpi_name`` for non-admins; deny out-of-grant.

Handler exercised directly with the heavy collaborators (working memory,
hybrid search, orchestrator) mocked so the brand-scoping branch is isolated.
Imports are module-level (matching ``test_routes/test_cognitive.py``) so the
heavy ``src.rag.retriever`` import happens at collection, not under the
per-test timeout.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from src.api.routes.cognitive import CognitiveQueryRequest, process_cognitive_query


def _viewer(*brands: str) -> Dict[str, Any]:
    return {"sub": "u-view", "role": "viewer", "brands": list(brands)}


def _admin() -> Dict[str, Any]:
    return {"sub": "u-admin", "role": "admin", "brands": []}


def _mock_working_memory() -> MagicMock:
    wm = MagicMock()
    wm.create_session = AsyncMock()
    wm.get_session = AsyncMock(return_value={"user_id": "u-view"})
    wm.add_message = AsyncMock()
    wm.append_evidence = AsyncMock()
    return wm


async def test_cognitive_non_admin_scopes_brand_and_strips_graph() -> None:
    req = CognitiveQueryRequest(query="why did TRx drop in northeast", brand=None)
    with (
        patch("src.api.routes.cognitive.get_working_memory", return_value=_mock_working_memory()),
        patch("src.api.routes.cognitive.hybrid_search", new=AsyncMock(return_value=[])) as hs,
        patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
    ):
        await process_cognitive_query(req, BackgroundTasks(), user=_viewer("Brand-X"))

    kwargs = hs.await_args.kwargs
    assert kwargs["filters"] == {"brand": "Brand-X"}
    assert kwargs["kpi_name"] is None


async def test_cognitive_non_admin_out_of_grant_forbidden() -> None:
    req = CognitiveQueryRequest(query="q", brand="Brand-Y")
    with (
        patch("src.api.routes.cognitive.get_working_memory", return_value=_mock_working_memory()),
        patch("src.api.routes.cognitive.hybrid_search", new=AsyncMock(return_value=[])) as hs,
        patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
    ):
        with pytest.raises(HTTPException) as exc:
            await process_cognitive_query(req, BackgroundTasks(), user=_viewer("Brand-X"))

    assert exc.value.status_code == 403
    hs.assert_not_awaited()


async def test_cognitive_no_grant_viewer_proceeds_without_memory() -> None:
    """A non-admin with NO brand grants and no brand requested cannot be
    tenant-scoped, so the query proceeds (session/orchestration still work for
    grant-less users) but NO memory is retrieved — hybrid_search is never called
    with an unscoped all-brand filter. (Mirrors the session-ownership tests that
    legitimately use grant-less viewers; only an *explicit* out-of-grant brand
    is a 403.)"""
    req = CognitiveQueryRequest(query="q", brand=None)
    with (
        patch("src.api.routes.cognitive.get_working_memory", return_value=_mock_working_memory()),
        patch("src.api.routes.cognitive.hybrid_search", new=AsyncMock(return_value=[])) as hs,
        patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
    ):
        await process_cognitive_query(req, BackgroundTasks(), user=_viewer())

    hs.assert_not_awaited()


async def test_cognitive_admin_passthrough_brand() -> None:
    req = CognitiveQueryRequest(query="q", brand="Brand-Z")
    with (
        patch("src.api.routes.cognitive.get_working_memory", return_value=_mock_working_memory()),
        patch("src.api.routes.cognitive.hybrid_search", new=AsyncMock(return_value=[])) as hs,
        patch("src.api.routes.cognitive.get_orchestrator", return_value=None),
    ):
        await process_cognitive_query(req, BackgroundTasks(), user=_admin())

    assert hs.await_args.kwargs["filters"] == {"brand": "Brand-Z"}
