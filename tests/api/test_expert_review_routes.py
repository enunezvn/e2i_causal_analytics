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
from src.api.dependencies.auth import require_operator
from src.api.main import app

#: The structure version a resolve body must name (#1991 debt 3, codex round-1
#: HIGH): the resolution is bound to the version the reviewer's form displayed,
#: so every resolve request carries it and the fake records it.
HASH = "h" * 64


class _FakeExpertReviewRepo:
    """In-memory fake mirroring the ExpertReviewRepository surface the route uses."""

    def __init__(self) -> None:
        self.pending_rows: List[Dict[str, Any]] = []
        self.summary: Dict[str, int] = {
            "pending": 0,
            "approved": 0,
            "rejected": 0,
            # Stored status since migration 140 (#1991 debt 3): a resolution,
            # counted apart from ``pending``.
            "superseded": 0,
            "expired": 0,
            "expiring_soon": 0,
        }
        self.versions_by_review: Dict[str, List[Dict[str, Any]]] = {}
        self.submit_calls: List[Dict[str, Any]] = []
        self.submit_return: bool = True
        self.rows_by_id: Dict[str, Dict[str, Any]] = {}
        self.assessment_writes: List[Dict[str, Any]] = []
        self.assessment_write_return: bool = True
        # R3: when set, every READ raises it (a store outage). The real repo
        # re-raises client errors instead of returning []/zeros.
        self.read_error: Optional[BaseException] = None

    async def get_pending_reviews(
        self,
        brand: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        if self.read_error is not None:
            raise self.read_error
        return list(self.pending_rows)

    async def get_versions_for_reviews(
        self, review_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """#1991 debt 3: the queue reads every row's structure versions in ONE
        call. ``versions_by_review`` is empty by default, so each row reports
        the never-moved shape (version 1, changed at creation)."""
        if self.read_error is not None:
            raise self.read_error
        return {
            rid: list(self.versions_by_review[rid])
            for rid in review_ids
            if rid in self.versions_by_review
        }

    async def submit_review(
        self,
        review_id: str,
        approval_status: str,
        checklist: Dict[str, Any],
        comments: Optional[Dict[str, Any]] = None,
        concerns_raised: Optional[List[str]] = None,
        conditions: Optional[str] = None,
        validity_days: int = 90,
        reviewer_name: Optional[str] = None,
        reviewer_email: Optional[str] = None,
        *,
        expected_dag_version_hash: str,
        expected_adjustment_set_hash: Optional[str],
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
                "reviewer_name": reviewer_name,
                "reviewer_email": reviewer_email,
                "expected_dag_version_hash": expected_dag_version_hash,
                "expected_adjustment_set_hash": expected_adjustment_set_hash,
            }
        )
        return self.submit_return

    async def get_review_summary(self, brand: Optional[str] = None) -> Dict[str, int]:
        if self.read_error is not None:
            raise self.read_error
        return dict(self.summary)

    async def get_by_id(self, id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return self.rows_by_id.get(id)

    async def update_agent_assessment(
        self,
        review_id: str,
        assessment: Dict[str, Any],
        *,
        for_dag_version_hash: Optional[str] = None,
        for_adjustment_set_hash: Optional[str] = None,
    ) -> bool:
        """#1991 debt 3 (codex rounds 1 and 2): the cache write is filtered on
        the WHOLE version the build GRADED -- the DAG hash and the adjustment-set
        hash -- so a build that finished after the review advanced writes
        nothing, including after an advance that changed only the covariates."""
        self.assessment_writes.append(
            {
                "review_id": review_id,
                "assessment": assessment,
                "for_dag_version_hash": for_dag_version_hash,
                "for_adjustment_set_hash": for_adjustment_set_hash,
            }
        )
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
                "dag_version_hash": HASH,
                "adjustment_set_hash": None,
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
            json={
                "approval_status": "rejected",
                "checklist": {},
                "dag_version_hash": HASH,
                "adjustment_set_hash": None,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["approval_status"] == "rejected"
        assert fake_repo.submit_calls[0]["approval_status"] == "rejected"

    def test_resolve_records_the_authenticated_operator(self, client, fake_repo):
        """Codex whole-diff HIGH F1: the operator the route authenticated is the
        resolver, so their identity must reach ``submit_review`` (call-kwargs
        pin). The override carries BOTH a metadata name and an email so the
        assertion cannot pass vacuously on the testing-mode default user."""
        app.dependency_overrides[require_operator] = lambda: {
            "id": "op-1",
            "email": "operator@example.com",
            "app_metadata": {"role": "admin"},
            "user_metadata": {"name": "Dr. Operator"},
        }
        resp = client.post(
            "/api/expert-reviews/66666666-6666-6666-6666-666666666666/resolve",
            json={
                "approval_status": "rejected",
                "checklist": {},
                "dag_version_hash": HASH,
                "adjustment_set_hash": None,
            },
        )
        assert resp.status_code == 200, resp.text
        call = fake_repo.submit_calls[0]
        assert call["reviewer_name"] == "Dr. Operator"
        assert call["reviewer_email"] == "operator@example.com"

    @pytest.mark.parametrize(
        ("user", "expected_name", "expected_email"),
        [
            # No metadata name -> the email stands in as the display name.
            (
                {"id": "op-2", "email": "op2@example.com", "user_metadata": {}},
                "op2@example.com",
                "op2@example.com",
            ),
            # No name, no email -> the id; email stays unknown.
            ({"id": "op-3", "user_metadata": {}}, "op-3", None),
            # Nothing usable -> unknown stays unknown (None, never a placeholder).
            ({"user_metadata": {}}, None, None),
        ],
    )
    def test_resolve_operator_identity_fallbacks(
        self, client, fake_repo, user, expected_name, expected_email
    ):
        app.dependency_overrides[require_operator] = lambda: {
            "app_metadata": {"role": "admin"},
            **user,
        }
        resp = client.post(
            "/api/expert-reviews/77777777-7777-7777-7777-777777777777/resolve",
            json={
                "approval_status": "approved",
                "checklist": {},
                "dag_version_hash": HASH,
                "adjustment_set_hash": None,
            },
        )
        assert resp.status_code == 200, resp.text
        call = fake_repo.submit_calls[0]
        assert call["reviewer_name"] == expected_name
        assert call["reviewer_email"] == expected_email

    def test_bad_approval_status_is_422(self, client, fake_repo):
        resp = client.post(
            "/api/expert-reviews/44444444-4444-4444-4444-444444444444/resolve",
            json={
                "approval_status": "blocked",
                "checklist": {},
                "dag_version_hash": HASH,
                "adjustment_set_hash": None,
            },
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
            json={
                "approval_status": "approved",
                "checklist": {},
                "dag_version_hash": HASH,
                "adjustment_set_hash": None,
            },
        )
        assert resp.status_code == 404, resp.text

    def test_stale_version_409_detail_reaches_the_frontends_message_field(self, client, fake_repo):
        """Codex round-1 HIGH: the 409 must TELL the reviewer to reload.

        This is the only resolve test mounted on the REAL app, so it is where the
        handler chain is observable: src/api/main.py maps a 409 to ConflictError
        and preserves the detail verbatim as ``message`` -- the field
        frontend/src/lib/api-client.ts reads into ``ApiError.message``, which is
        what ResolveForm's "Failed to submit review" banner renders. A body that
        carried only ``detail`` would show the reviewer a bare status code.
        """
        rid = "88888888-8888-8888-8888-888888888888"
        fake_repo.submit_return = False
        fake_repo.rows_by_id[rid] = {
            "review_id": rid,
            "approval_status": "pending",
            "dag_version_hash": "a" * 64,  # advanced since the form was opened
        }
        resp = client.post(
            f"/api/expert-reviews/{rid}/resolve",
            json={
                "approval_status": "approved",
                "checklist": {},
                "dag_version_hash": HASH,
                "adjustment_set_hash": None,
            },
        )
        assert resp.status_code == 409, resp.text
        message = resp.json()["message"]
        assert "advanced to a new structure version" in message
        assert "reload the review" in message


class TestStoreOutageIsHonest:
    """R3: a store failure must be a 503 with a plain detail, never a 200 with
    an empty queue or zero counts, and never a raw 500. The app's catch-all
    classifies by message keyword ("connection"/"unavailable" -> 503, else
    500), which is not a contract -- so the route maps it itself, mirroring
    routes/causal/catalog.py ("Causal data store unavailable", 503) and
    routes/digital_twin.py (503 + Retry-After)."""

    @pytest.fixture
    def outage_client(self, fake_repo):
        from fastapi.testclient import TestClient

        # A generic message: on the OLD code this reached the catch-all as a
        # plain 500 (positive control), because it carries no keyword.
        fake_repo.read_error = RuntimeError("store down")
        return TestClient(app, raise_server_exceptions=False)

    # The app's StarletteHTTPException handler reshapes a 503 into the
    # DependencyError envelope ({"error", "category", "message", ...}) and masks
    # the detail unless the route marked it user-safe -- so the plain message
    # landing in ``message`` proves both the status and the marking.
    def test_pending_queue_is_503_not_empty_200(self, outage_client):
        resp = outage_client.get("/api/expert-reviews/pending")
        assert resp.status_code == 503, resp.text
        body = resp.json()
        assert body["category"] == "dependency_error"
        assert body["message"] == "Expert-review store unavailable. Retry shortly."
        assert "reviews" not in body, "no empty queue may ride an error response"

    def test_summary_is_503_not_zeros_200(self, outage_client):
        resp = outage_client.get("/api/expert-reviews/summary")
        assert resp.status_code == 503, resp.text
        body = resp.json()
        assert body["category"] == "dependency_error"
        assert body["message"] == "Expert-review store unavailable. Retry shortly."
        assert "pending" not in body, "no zero counts may ride an error response"


class TestReviewSummary:
    def test_summary_shape(self, client, fake_repo):
        fake_repo.summary = {
            "pending": 3,
            "approved": 7,
            "rejected": 1,
            "superseded": 38,
            "expired": 2,
            "expiring_soon": 1,
        }
        resp = client.get("/api/expert-reviews/summary")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "pending": 3,
            "approved": 7,
            "rejected": 1,
            # Migration 140 resolved the BLOCK-band backlog to this status; the
            # page must be able to show it, not fold it into another bucket.
            "superseded": 38,
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
        # #1991: dag_structure_json is now a typed DagStructureSnapshot, so the
        # response carries every snapshot key (adjustment_sets, confidence, ...)
        # defaulted to null. Compare non-null keys only (mirrors the
        # exclude_none form in test_expert_review_detail_route.py) rather than
        # a fixed subset -- a subset comprehension would let a fabricated
        # non-null value in an extra key (e.g. confidence) pass silently.
        assert {k: v for k, v in item["dag_structure_json"].items() if v is not None} == _STRUCTURE
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
