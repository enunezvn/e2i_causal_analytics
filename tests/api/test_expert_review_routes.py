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
        self.rows_by_id: Dict[str, Dict[str, Any]] = {}
        self.assessment_writes: List[Dict[str, Any]] = []
        self.assessment_write_return: bool = True

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

    async def get_by_id(self, id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return self.rows_by_id.get(id)

    async def update_agent_assessment(self, review_id: str, assessment: Dict[str, Any]) -> bool:
        self.assessment_writes.append({"review_id": review_id, "assessment": assessment})
        return self.assessment_write_return


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


_STRUCTURE = {
    "nodes": ["t", "y", "c"],
    "edges": [["t", "y"], ["c", "t"], ["c", "y"]],
    "treatment_nodes": ["t"],
    "outcome_nodes": ["y"],
}

_ASSESSMENT = {
    "items": [
        {
            "id": "conf_complete",
            "question": "Are all known confounders included?",
            "verdict": "supports",
            "rationale": "confounder refuters passed",
        }
    ],
    "is_fallback": True,
    "evidence": {"refutation_tests": 1, "has_dag_structure": True},
}


class TestPendingReviewsCarryDagStructure:
    """Mig 097: the queue rows must expose the renderable DAG snapshot and any
    cached agent assessment — parsed to OBJECTS even when the repo row holds a
    JSON string (json.dumps write path -> JSONB string scalar)."""

    def test_structure_and_assessment_surfaced_as_objects(self, client, fake_repo):
        import json as _json

        fake_repo.pending_rows = [
            {
                "review_id": "33333333-3333-3333-3333-333333333333",
                "review_type": "dag_approval",
                "dag_structure_json": _json.dumps(_STRUCTURE),  # string form
                "agent_assessment_json": _ASSESSMENT,  # dict form
            }
        ]
        resp = client.get("/api/expert-reviews/pending")
        assert resp.status_code == 200, resp.text
        item = resp.json()["reviews"][0]
        assert item["dag_structure_json"] == _STRUCTURE
        assert item["agent_assessment_json"]["items"][0]["verdict"] == "supports"

    def test_absent_structure_is_null_not_fabricated(self, client, fake_repo):
        fake_repo.pending_rows = [{"review_id": "44444444-4444-4444-4444-444444444444"}]
        resp = client.get("/api/expert-reviews/pending")
        item = resp.json()["reviews"][0]
        assert item["dag_structure_json"] is None
        assert item["agent_assessment_json"] is None


class TestAgentAssessmentEndpoint:
    """POST /expert-reviews/{id}/assessment — on-demand advisory assessment,
    cached in agent_assessment_json (never regenerated unless force=true)."""

    @pytest.fixture
    def stub_generation(self, monkeypatch):
        """Stub the LM/evidence seams: no live DB reads, no LM call."""
        calls = {"generate": 0}

        async def _fake_validation_rows(ids):
            return [{"test_type": "random_common_cause", "status": "passed"}]

        def _fake_build(review, validations):
            calls["generate"] += 1
            return dict(_ASSESSMENT)

        monkeypatch.setattr(expert_review_route, "_get_validation_rows", _fake_validation_rows)
        monkeypatch.setattr(expert_review_route, "_build_assessment", _fake_build)
        return calls

    def test_unknown_review_404(self, client, fake_repo, stub_generation):
        resp = client.post("/api/expert-reviews/99999999-9999-9999-9999-999999999999/assessment")
        assert resp.status_code == 404

    def test_generates_persists_and_returns(self, client, fake_repo, stub_generation):
        rid = "55555555-5555-5555-5555-555555555555"
        fake_repo.rows_by_id[rid] = {
            "review_id": rid,
            "dag_structure_json": _STRUCTURE,
            "related_validation_ids": ["val-1"],
        }
        resp = client.post(f"/api/expert-reviews/{rid}/assessment")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["cached"] is False
        assert body["persisted"] is True
        assert body["assessment"]["items"][0]["id"] == "conf_complete"
        assert stub_generation["generate"] == 1
        assert fake_repo.assessment_writes[0]["review_id"] == rid

    def test_cached_assessment_is_returned_without_regenerating(
        self, client, fake_repo, stub_generation
    ):
        import json as _json

        rid = "66666666-6666-6666-6666-666666666666"
        fake_repo.rows_by_id[rid] = {
            "review_id": rid,
            # stored as a JSON string (json.dumps write path) -> must come back parsed
            "agent_assessment_json": _json.dumps(_ASSESSMENT),
        }
        resp = client.post(f"/api/expert-reviews/{rid}/assessment")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["cached"] is True
        assert body["assessment"]["items"][0]["verdict"] == "supports"
        assert stub_generation["generate"] == 0
        assert fake_repo.assessment_writes == []

    def test_force_regenerates_over_cache(self, client, fake_repo, stub_generation):
        rid = "77777777-7777-7777-7777-777777777777"
        fake_repo.rows_by_id[rid] = {
            "review_id": rid,
            "agent_assessment_json": dict(_ASSESSMENT),
        }
        resp = client.post(f"/api/expert-reviews/{rid}/assessment?force=true")
        assert resp.status_code == 200, resp.text
        assert resp.json()["cached"] is False
        assert stub_generation["generate"] == 1

    def test_persistence_failure_is_honest(self, client, fake_repo, stub_generation):
        """Cache write failing must NOT fabricate persisted=True (the assessment
        itself is still returned — it is valid, just not cached)."""
        rid = "88888888-8888-8888-8888-888888888888"
        fake_repo.rows_by_id[rid] = {"review_id": rid}
        fake_repo.assessment_write_return = False
        resp = client.post(f"/api/expert-reviews/{rid}/assessment")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["persisted"] is False
        assert body["assessment"]["items"]
