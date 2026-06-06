"""
Tests for Expert Review API endpoints (R6-F2 Phase A2).

The human-in-the-loop review queue consumer:
- GET  /api/expert-reviews/pending          -> oldest-first pending queue
- POST /api/expert-reviews/{review_id}/resolve -> approve/reject a review
- GET  /api/expert-reviews/summary          -> status counts

Auth: ``require_operator`` (OD-1). ``E2I_TESTING_MODE`` (set in tests/api/conftest.py)
bypasses JWT, so these tests exercise the route wiring/shape, not the auth gate.

NO live PostgREST insert: the route's ``_get_expert_review_repo`` helper is
monkeypatched to return a FAKE repo whose methods capture the kwargs they are
called with. This keeps the live ``expert_reviews`` table un-polluted (the async
repo uses PostgREST HTTP, so BEGIN..ROLLBACK is not available).
"""

from typing import Any, Dict, List, Optional

import pytest

import src.api.routes.expert_review as expert_review_route
from src.api.main import app


class _FakeExpertReviewRepo:
    """In-memory fake mirroring the ExpertReviewRepository surface the route uses."""

    def __init__(self) -> None:
        self.pending_rows: List[Dict[str, Any]] = []
        self.summary: Dict[str, int] = {
            "pending": 0,
            "approved": 0,
            "rejected": 0,
            "expired": 0,
            "expiring_soon": 0,
        }
        self.submit_calls: List[Dict[str, Any]] = []
        self.submit_return: bool = True

    async def get_pending_reviews(
        self,
        brand: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return list(self.pending_rows)

    async def submit_review(
        self,
        review_id: str,
        approval_status: str,
        checklist: Dict[str, Any],
        comments: Optional[Dict[str, Any]] = None,
        concerns_raised: Optional[List[str]] = None,
        conditions: Optional[str] = None,
        validity_days: int = 90,
    ) -> bool:
        self.submit_calls.append(
            {
                "review_id": review_id,
                "approval_status": approval_status,
                "checklist": checklist,
                "comments": comments,
                "concerns_raised": concerns_raised,
                "conditions": conditions,
                "validity_days": validity_days,
            }
        )
        return self.submit_return

    async def get_review_summary(self, brand: Optional[str] = None) -> Dict[str, int]:
        return dict(self.summary)


@pytest.fixture
def fake_repo(monkeypatch):
    """Patch the route's repo helper to return a fake repo (no live DB)."""
    repo = _FakeExpertReviewRepo()

    async def _fake_get_repo():
        return repo

    monkeypatch.setattr(expert_review_route, "_get_expert_review_repo", _fake_get_repo)
    return repo


@pytest.fixture
def client(fake_repo):
    from fastapi.testclient import TestClient

    return TestClient(app)


class TestPendingReviews:
    def test_route_is_registered_not_404(self, client):
        """RED until A2/A3: route file + main.py registration absent => 404."""
        resp = client.get("/api/expert-reviews/pending")
        assert resp.status_code != 404, "route /api/expert-reviews/pending not registered"

    def test_returns_pending_queue_shape(self, client, fake_repo):
        fake_repo.pending_rows = [
            {
                "review_id": "11111111-1111-1111-1111-111111111111",
                "review_type": "dag_approval",
                "dag_version_hash": "abc123",
                "brand": "Remibrutinib",
                "treatment_variable": "email_frequency",
                "outcome_variable": "trx",
                "analysis_context": "confidence=0.60, gate=review",
                "created_at": "2026-06-01T00:00:00+00:00",
                "days_pending": 5.0,
            }
        ]
        resp = client.get("/api/expert-reviews/pending")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] == 1
        assert len(body["reviews"]) == 1
        item = body["reviews"][0]
        assert item["review_id"] == "11111111-1111-1111-1111-111111111111"
        assert item["review_type"] == "dag_approval"
        assert item["treatment_variable"] == "email_frequency"
        assert item["outcome_variable"] == "trx"

    def test_empty_queue(self, client, fake_repo):
        resp = client.get("/api/expert-reviews/pending")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {"reviews": [], "total": 0}


class TestResolveReview:
    def test_resolve_approved_calls_submit_once(self, client, fake_repo):
        review_id = "22222222-2222-2222-2222-222222222222"
        resp = client.post(
            f"/api/expert-reviews/{review_id}/resolve",
            json={
                "approval_status": "approved",
                "checklist": {"conf_complete": True, "edge_plausible": True},
                "comments": {"note": "looks good"},
                "validity_days": 90,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {
            "review_id": review_id,
            "approval_status": "approved",
            "success": True,
        }
        assert len(fake_repo.submit_calls) == 1
        call = fake_repo.submit_calls[0]
        assert call["review_id"] == review_id
        assert call["approval_status"] == "approved"
        assert call["checklist"] == {"conf_complete": True, "edge_plausible": True}
        assert call["comments"] == {"note": "looks good"}
        assert call["validity_days"] == 90

    def test_resolve_rejected(self, client, fake_repo):
        review_id = "33333333-3333-3333-3333-333333333333"
        resp = client.post(
            f"/api/expert-reviews/{review_id}/resolve",
            json={"approval_status": "rejected", "checklist": {}},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["approval_status"] == "rejected"
        assert fake_repo.submit_calls[0]["approval_status"] == "rejected"

    def test_bad_approval_status_is_422(self, client, fake_repo):
        resp = client.post(
            "/api/expert-reviews/44444444-4444-4444-4444-444444444444/resolve",
            json={"approval_status": "blocked", "checklist": {}},
        )
        assert resp.status_code == 422
        # repo must NOT have been called when validation fails
        assert fake_repo.submit_calls == []

    def test_zero_row_resolve_is_404_not_200(self, client, fake_repo):
        """FIX B (codex HIGH): a zero-row resolve must be 404, never a fake 200.

        submit_review returns False for a nonexistent / already-resolved
        review_id (zero-row update). The route must surface that as 404 (the
        honest 'not found / not resolvable' code), NOT a fabricated 200 — and not
        the old generic 502 either.
        """
        fake_repo.submit_return = False
        resp = client.post(
            "/api/expert-reviews/55555555-5555-5555-5555-555555555555/resolve",
            json={"approval_status": "approved", "checklist": {}},
        )
        assert resp.status_code == 404, resp.text


class TestReviewSummary:
    def test_summary_shape(self, client, fake_repo):
        fake_repo.summary = {
            "pending": 3,
            "approved": 7,
            "rejected": 1,
            "expired": 2,
            "expiring_soon": 1,
        }
        resp = client.get("/api/expert-reviews/summary")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "pending": 3,
            "approved": 7,
            "rejected": 1,
            "expired": 2,
            "expiring_soon": 1,
        }
