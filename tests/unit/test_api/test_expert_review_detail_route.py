"""Lane 1 (spec §4.2): GET /expert-reviews/{review_id} returns a review in ANY
status plus the same-structure history, so the drill-down's deep link resolves
for pending, approved and rejected structures alike."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import src.api.routes.expert_review as route_mod
from src.api.errors import SAFE_503_DETAIL_PREFIX

# Real UUID literals: ``expert_reviews.review_id`` is a uuid column, and the
# route canonicalises the path id before the store read (review F1).
RID = "1c8f3d6a-5b7e-4c21-9f0a-2d4e6b8a0c13"
RID_OLDER = "7a2b9c40-3d1e-4f65-8a7b-0c9d2e1f3a58"
RID_UNKNOWN = "9e8d7c6b-5a49-4382-b1c0-d9e8f7a6b5c4"
DAG_HASH = "h" * 64

ROW: Dict[str, Any] = {
    "review_id": RID,
    "review_type": "dag_approval",
    "dag_version_hash": DAG_HASH,
    "brand": "Kisqali",
    "treatment_variable": "treatment_arm",
    "outcome_variable": "persistent_180d",
    "approval_status": "rejected",
    "reviewer_name": "Dr. No",
    "concerns_raised": ["collider"],
    "created_at": "2026-07-13T10:00:00+00:00",
    "valid_from": "2026-07-13",
    "approved_at": "2026-07-14T09:30:00+00:00",
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
        self.get_calls: List[str] = []
        self.history_calls: List[tuple] = []

    async def get_by_id(self, review_id: str):
        self.get_calls.append(review_id)
        if self.fail == "row":
            raise RuntimeError("connection refused")
        try:
            uuid.UUID(review_id)
        except ValueError:  # what the live uuid column does: PostgREST APIError 22P02
            raise RuntimeError(f'invalid input syntax for type uuid: "{review_id}"') from None
        # Matches on the CANONICAL id only, like the uuid column would.
        return self.row if self.row and self.row["review_id"] == review_id else None

    async def get_reviews_for_dag(
        self, dag_hash: str, include_expired: bool = False, brand: Optional[str] = None
    ):
        self.history_calls.append((dag_hash, include_expired, brand))
        if self.fail == "history":
            raise RuntimeError("connection refused")
        return self.history

    async def get_pending_reviews(self, brand=None, reviewer_id=None, limit=50):
        return []


def _install(monkeypatch, repo: _Repo) -> None:
    async def _factory():
        return repo

    monkeypatch.setattr(route_mod, "_get_expert_review_repo", _factory)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_returns_the_row_and_its_same_structure_history(monkeypatch):
    repo = _Repo(ROW, history=[ROW, {**ROW, "review_id": RID_OLDER, "approval_status": "pending"}])
    _install(monkeypatch, repo)
    resp = await route_mod.get_expert_review(RID, user={})
    assert resp.review.review_id == RID
    assert resp.review.approval_status == "rejected"
    assert resp.review.reviewer_name == "Dr. No"
    assert resp.review.dag_structure_json == {"nodes": ["t", "y"], "edges": [["t", "y"]]}
    assert resp.review.comments_json == {"note": "engagement is post-treatment"}
    assert [r.review_id for r in resp.history] == [RID, RID_OLDER]
    # the same read the gate's rejection probe performs: expired included, brand-scoped
    assert repo.history_calls == [(DAG_HASH, True, "Kisqali")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_review_is_404(monkeypatch):
    """A VALID but unknown uuid: the store answers None -> 404."""
    repo = _Repo(None)
    _install(monkeypatch, repo)
    with pytest.raises(HTTPException) as ei:
        await route_mod.get_expert_review(RID_UNKNOWN, user={})
    assert ei.value.status_code == 404
    assert repo.get_calls == [RID_UNKNOWN]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_malformed_id_is_404_without_touching_the_store(monkeypatch):
    """A non-uuid id would make PostgREST raise 22P02 (a 503 through the
    store-failure guard); the pre-check answers 404 and never reads the store."""

    class _NeverRead(_Repo):
        async def get_by_id(self, review_id: str):
            raise AssertionError("store must not be read")

    _install(monkeypatch, _NeverRead(ROW))
    with pytest.raises(HTTPException) as ei:
        await route_mod.get_expert_review("nope", user={})
    assert ei.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_canonical_id_reaches_the_store_canonical(monkeypatch):
    repo = _Repo(ROW, history=[ROW])
    _install(monkeypatch, repo)
    resp = await route_mod.get_expert_review(RID.upper(), user={})
    assert resp.review.review_id == RID
    assert repo.get_calls == [RID]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("fail", ["row", "history"])
async def test_store_failure_is_503_never_an_empty_200(monkeypatch, fail):
    _install(monkeypatch, _Repo(ROW, history=[ROW], fail=fail))
    with pytest.raises(HTTPException) as ei:
        await route_mod.get_expert_review(RID, user={})
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
        await route_mod.get_expert_review(RID, user={})
    assert ei.value.status_code == 503


@pytest.mark.unit
@pytest.mark.asyncio
async def test_row_without_a_hash_has_empty_history(monkeypatch):
    repo = _Repo({**ROW, "dag_version_hash": None})
    _install(monkeypatch, repo)
    resp = await route_mod.get_expert_review(RID, user={})
    assert resp.history == []
    assert repo.history_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_row_without_a_brand_reads_an_unfiltered_history(monkeypatch):
    """A brand-less row's history is cross-brand -- exactly the gate's read."""
    repo = _Repo({**ROW, "brand": None}, history=[{**ROW, "brand": None}])
    _install(monkeypatch, repo)
    resp = await route_mod.get_expert_review(RID, user={})
    assert [r.review_id for r in resp.history] == [RID]
    assert repo.history_calls == [(DAG_HASH, True, None)]


@pytest.mark.unit
def test_lookup_route_is_declared_after_pending_and_summary():
    """FastAPI matches in declaration order: /{review_id} must not shadow the
    two literal GET routes."""
    paths = [r.path for r in route_mod.router.routes]
    assert paths.index("/expert-reviews/pending") < paths.index("/expert-reviews/{review_id}")
    assert paths.index("/expert-reviews/summary") < paths.index("/expert-reviews/{review_id}")


# --- in-process client: Depends, HTTP serialisation, error bodies, routing ---
# Minimal app (no ``src.api.main`` import, so no lifespan); precedent
# tests/unit/test_api/test_executive_insights.py. E2I_TESTING_MODE=1 (conftest)
# makes ``require_operator`` yield the mock user.


def _client(monkeypatch, repo: _Repo) -> TestClient:
    _install(monkeypatch, repo)
    app = FastAPI()
    app.include_router(route_mod.router, prefix="/api")
    return TestClient(app)


@pytest.mark.unit
def test_http_200_serialises_dates_and_parsed_json(monkeypatch):
    client = _client(monkeypatch, _Repo(ROW, history=[ROW, {**ROW, "review_id": RID_OLDER}]))
    r = client.get(f"/api/expert-reviews/{RID}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["review"]["review_id"] == RID
    assert body["review"]["valid_from"] == "2026-07-13"
    assert datetime.fromisoformat(body["review"]["approved_at"]) == datetime(
        2026, 7, 14, 9, 30, tzinfo=timezone.utc
    )
    assert body["review"]["comments_json"] == {"note": "engagement is post-treatment"}
    assert len(body["history"]) == 2


@pytest.mark.unit
def test_http_unknown_valid_uuid_is_404_with_detail(monkeypatch):
    r = _client(monkeypatch, _Repo(None)).get(f"/api/expert-reviews/{RID_UNKNOWN}")
    assert r.status_code == 404
    assert isinstance(r.json()["detail"], str) and RID_UNKNOWN in r.json()["detail"]


@pytest.mark.unit
def test_http_malformed_id_is_404(monkeypatch):
    r = _client(monkeypatch, _Repo(ROW)).get("/api/expert-reviews/nope")
    assert r.status_code == 404, r.text


@pytest.mark.unit
def test_http_store_failure_is_a_safe_503(monkeypatch):
    r = _client(monkeypatch, _Repo(ROW, fail="row")).get(f"/api/expert-reviews/{RID}")
    assert r.status_code == 503
    detail = r.json()["detail"]
    # The marker the app's global handler surfaces verbatim -- pin it.
    assert detail.startswith(SAFE_503_DETAIL_PREFIX)
    assert "Expert-review store unavailable" in detail


@pytest.mark.unit
def test_http_pending_is_not_shadowed_by_the_lookup(monkeypatch):
    r = _client(monkeypatch, _Repo(ROW)).get("/api/expert-reviews/pending")
    # The 200 + the pending body are the proof. A store-call count could not
    # discriminate: were /{review_id} declared first, "pending" would fail the
    # uuid pre-check and answer 404 BEFORE any store read (measured), which the
    # status assertion catches.
    assert r.status_code == 200, r.text
    assert r.json() == {"reviews": [], "total": 0}
