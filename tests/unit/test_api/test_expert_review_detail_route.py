"""Lane 1 (spec §4.2): GET /expert-reviews/{review_id} returns a review in ANY
status plus the same-structure history, so the drill-down's deep link resolves
for pending, approved and rejected structures alike."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest
from fastapi import HTTPException

import src.api.routes.expert_review as route_mod

ROW: Dict[str, Any] = {
    "review_id": "rev-1",
    "review_type": "dag_approval",
    "dag_version_hash": "h" * 64,
    "brand": "Kisqali",
    "treatment_variable": "treatment_arm",
    "outcome_variable": "persistent_180d",
    "approval_status": "rejected",
    "reviewer_name": "Dr. No",
    "concerns_raised": ["collider"],
    "created_at": "2026-07-13T10:00:00+00:00",
    "dag_structure_json": json.dumps({"nodes": ["t", "y"], "edges": [["t", "y"]]}),
    "comments_json": json.dumps({"note": "engagement is post-treatment"}),
}


class _Repo:
    def __init__(
        self,
        row: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None,
        fail: Optional[str] = None,
    ):
        self.row, self.history, self.fail = row, history or [], fail
        self.history_calls: List[tuple] = []

    async def get_by_id(self, review_id: str):
        if self.fail == "row":
            raise RuntimeError("connection refused")
        return self.row if self.row and self.row["review_id"] == review_id else None

    async def get_reviews_for_dag(
        self, dag_hash: str, include_expired: bool = False, brand: Optional[str] = None
    ):
        self.history_calls.append((dag_hash, include_expired, brand))
        if self.fail == "history":
            raise RuntimeError("connection refused")
        return self.history


def _install(monkeypatch, repo: _Repo) -> None:
    async def _factory():
        return repo

    monkeypatch.setattr(route_mod, "_get_expert_review_repo", _factory)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_returns_the_row_and_its_same_structure_history(monkeypatch):
    repo = _Repo(ROW, history=[ROW, {**ROW, "review_id": "rev-0", "approval_status": "pending"}])
    _install(monkeypatch, repo)
    resp = await route_mod.get_expert_review("rev-1", user={})
    assert resp.review.review_id == "rev-1"
    assert resp.review.approval_status == "rejected"
    assert resp.review.reviewer_name == "Dr. No"
    assert resp.review.dag_structure_json == {"nodes": ["t", "y"], "edges": [["t", "y"]]}
    assert resp.review.comments_json == {"note": "engagement is post-treatment"}
    assert [r.review_id for r in resp.history] == ["rev-1", "rev-0"]
    # the same read the gate's rejection probe performs: expired included, brand-scoped
    assert repo.history_calls == [("h" * 64, True, "Kisqali")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_review_is_404(monkeypatch):
    _install(monkeypatch, _Repo(None))
    with pytest.raises(HTTPException) as ei:
        await route_mod.get_expert_review("nope", user={})
    assert ei.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("fail", ["row", "history"])
async def test_store_failure_is_503_never_an_empty_200(monkeypatch, fail):
    _install(monkeypatch, _Repo(ROW, history=[ROW], fail=fail))
    with pytest.raises(HTTPException) as ei:
        await route_mod.get_expert_review("rev-1", user={})
    assert ei.value.status_code == 503


@pytest.mark.unit
@pytest.mark.asyncio
async def test_client_factory_failure_is_503(monkeypatch):
    """``_get_expert_review_repo`` raises when Supabase is unset/unreachable; that
    must be the same honest 503, not an unhandled 500."""

    async def _boom():
        raise RuntimeError("supabase unavailable")

    monkeypatch.setattr(route_mod, "_get_expert_review_repo", _boom)
    with pytest.raises(HTTPException) as ei:
        await route_mod.get_expert_review("rev-1", user={})
    assert ei.value.status_code == 503


@pytest.mark.unit
@pytest.mark.asyncio
async def test_row_without_a_hash_has_empty_history(monkeypatch):
    repo = _Repo({**ROW, "dag_version_hash": None})
    _install(monkeypatch, repo)
    resp = await route_mod.get_expert_review("rev-1", user={})
    assert resp.history == []
    assert repo.history_calls == []


@pytest.mark.unit
def test_lookup_route_is_declared_after_pending_and_summary():
    """FastAPI matches in declaration order: /{review_id} must not shadow the
    two literal GET routes."""
    paths = [r.path for r in route_mod.router.routes]
    assert paths.index("/expert-reviews/pending") < paths.index("/expert-reviews/{review_id}")
    assert paths.index("/expert-reviews/summary") < paths.index("/expert-reviews/{review_id}")
