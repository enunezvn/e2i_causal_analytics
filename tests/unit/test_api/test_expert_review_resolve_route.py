"""POST /expert-reviews/{review_id}/resolve records the authenticated operator as
the resolver (lane 1, codex whole-diff HIGH F1).

``reviewer_id`` holds the REQUESTER (the originating query id the gate wrote),
so the resolver's identity must come from the verified token: ``user_metadata
.name`` (else email, else id) as ``reviewer_name`` and ``email`` as
``reviewer_email``, passed to ``submit_review`` (call-kwargs pin). An identity
the token does not carry stays None -- never a placeholder.

In-process bare app (no ``src.api.main`` import, no lifespan), the pattern of
test_expert_review_detail_route.py; ``require_operator`` is overridden per
test with an EXPLICIT operator dict so the pins cannot pass vacuously on the
testing-mode default user. This directory is on backend-tests.yml's allowlist;
tests/api/test_expert_review_routes.py (the older resolve tests) is not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routes.expert_review as route_mod
from src.api.dependencies.auth import require_operator

RID = "1c8f3d6a-5b7e-4c21-9f0a-2d4e6b8a0c13"


class _Repo:
    """Captures every kwarg ``resolve_review`` passes to ``submit_review``."""

    def __init__(self) -> None:
        self.submit_calls: List[Dict[str, Any]] = []

    async def submit_review(self, **kwargs: Any) -> bool:
        self.submit_calls.append(kwargs)
        return True


def _client(monkeypatch, repo: _Repo, operator: Optional[Dict[str, Any]]) -> TestClient:
    async def _factory():
        return repo

    monkeypatch.setattr(route_mod, "_get_expert_review_repo", _factory)
    app = FastAPI()
    app.include_router(route_mod.router, prefix="/api")
    if operator is not None:
        app.dependency_overrides[require_operator] = lambda: operator
    return TestClient(app)


def _resolve(client: TestClient, status: str = "rejected") -> Dict[str, Any]:
    r = client.post(
        f"/api/expert-reviews/{RID}/resolve",
        json={"approval_status": status, "checklist": {"confounders_complete": True}},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"review_id": RID, "approval_status": status, "success": True}
    return r.json()


@pytest.mark.unit
def test_operator_name_and_email_reach_submit_review(monkeypatch):
    repo = _Repo()
    client = _client(
        monkeypatch,
        repo,
        {
            "id": "op-1",
            "email": "operator@example.com",
            "app_metadata": {"role": "admin"},
            "user_metadata": {"name": "Dr. Operator"},
        },
    )
    _resolve(client)
    assert len(repo.submit_calls) == 1
    call = repo.submit_calls[0]
    assert call["review_id"] == RID
    assert call["approval_status"] == "rejected"
    assert call["reviewer_name"] == "Dr. Operator"
    assert call["reviewer_email"] == "operator@example.com"


@pytest.mark.unit
def test_operator_with_only_an_id_is_named_by_it(monkeypatch):
    """No profile name and no email: the id is the only recorded handle; the
    email stays None rather than a stand-in."""
    repo = _Repo()
    _resolve(_client(monkeypatch, repo, {"id": "op-3", "user_metadata": {}}), status="approved")
    call = repo.submit_calls[0]
    assert call["approval_status"] == "approved"
    assert call["reviewer_name"] == "op-3"
    assert call["reviewer_email"] is None


@pytest.mark.unit
@pytest.mark.parametrize("operator", [{}, {"user_metadata": None, "email": "", "id": ""}])
def test_operator_without_usable_identity_records_nothing(monkeypatch, operator):
    """Nothing usable (no keys at all, or a None metadata block and empty
    strings): both fields are None and the resolve still succeeds -- unknown
    stays unknown, and the route never crashes on a sparse token."""
    repo = _Repo()
    _resolve(_client(monkeypatch, repo, operator))
    call = repo.submit_calls[0]
    assert call["reviewer_name"] is None
    assert call["reviewer_email"] is None
