"""#1991 debt 3: the expert-review API surfaces the ESTIMAND's structure timeline.

The detail route returns the review's ``expert_review_versions`` rows (mig 141)
oldest-first, each carrying the diff against the version before it, and its
``history`` is now every review of the same ESTIMAND (mig 140) rather than every
review of the same DAG hash -- a covariate change advances the SAME review, so a
hash-keyed history would lose the earlier structure entirely. Pre-140 rows (no
``estimand_key``) keep the same-hash history.

The queue carries ``version_count`` / ``last_changed_at`` so an operator can see
that a pending review has moved under them, and the summary counts the
``superseded`` partition migration 140 introduced.

Client shape mirrors ``test_expert_review_detail_route.py``: a minimal FastAPI
app over ``route_mod.router`` (no ``src.api.main`` import, so no lifespan), with
``_get_expert_review_repo`` monkeypatched to a fake. ``E2I_TESTING_MODE=1``
(conftest) makes ``require_operator`` yield the mock user.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routes.expert_review as route_mod
from src.api.errors import SAFE_503_DETAIL_PREFIX

RID = "1c8f3d6a-5b7e-4c21-9f0a-2d4e6b8a0c13"
RID_OLDER = "7a2b9c40-3d1e-4f65-8a7b-0c9d2e1f3a58"
RID_PRE140 = "3f5e7d90-1a2b-4c3d-8e9f-0a1b2c3d4e5f"
RID_NO_VERSIONS = "5b4a3c2d-1e0f-4a9b-8c7d-6e5f4a3b2c1d"

ESTIMAND = "b:t:y"
HASH_V1 = "1" * 64
HASH_V2 = "2" * 64

SNAP_V1: Dict[str, Any] = {"nodes": ["T", "Y"], "edges": [["T", "Y"]], "adjustment_sets": []}
SNAP_V2: Dict[str, Any] = {
    "nodes": ["T", "Y", "W"],
    "edges": [["W", "T"], ["T", "Y"]],
    "adjustment_sets": [["W"]],
}

V1_AT = "2026-07-13T10:00:00+00:00"
V2_AT = "2026-07-14T10:00:00+00:00"


def _version(version_id: str, dag_hash: str, snapshot: Optional[Dict[str, Any]], created_at: str):
    return {
        "version_id": version_id,
        "review_id": RID,
        "dag_version_hash": dag_hash,
        "adjustment_set_hash": None,
        "dag_structure_json": snapshot,
        "query_id": "q-1",
        "created_at": created_at,
    }


V1 = _version("aaaaaaa1-0000-4000-8000-000000000001", HASH_V1, SNAP_V1, V1_AT)
V2 = _version("aaaaaaa2-0000-4000-8000-000000000002", HASH_V2, SNAP_V2, V2_AT)

ROW: Dict[str, Any] = {
    "review_id": RID,
    "review_type": "dag_approval",
    "estimand_key": ESTIMAND,
    "dag_version_hash": HASH_V2,
    "brand": "Kisqali",
    "treatment_variable": "treatment_arm",
    "outcome_variable": "persistent_180d",
    "approval_status": "pending",
    "created_at": V1_AT,
    "dag_structure_json": SNAP_V2,
}

ROW_OLDER: Dict[str, Any] = {
    **ROW,
    "review_id": RID_OLDER,
    "dag_version_hash": HASH_V1,
    "approval_status": "approved",
    "dag_structure_json": SNAP_V1,
    "created_at": "2026-07-01T10:00:00+00:00",
}

ROW_PRE140: Dict[str, Any] = {k: v for k, v in ROW.items() if k != "estimand_key"}
ROW_PRE140["review_id"] = RID_PRE140


class _Repo:
    """Fake repo mirroring ``test_expert_review_detail_route.py:_Repo``, extended
    with the estimand/version readers."""

    def __init__(
        self,
        row: Optional[Dict[str, Any]] = None,
        *,
        estimand_history: Optional[List[Dict[str, Any]]] = None,
        dag_history: Optional[List[Dict[str, Any]]] = None,
        versions: Optional[List[Dict[str, Any]]] = None,
        pending: Optional[List[Dict[str, Any]]] = None,
        versions_by_review: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ):
        self.row = row
        self.estimand_history = estimand_history or []
        self.dag_history = dag_history or []
        self.versions = versions or []
        self.pending = pending or []
        self.versions_by_review = versions_by_review or {}
        self.summary = summary or {}
        self.estimand_calls: List[str] = []
        self.dag_calls: List[tuple] = []
        self.version_calls: List[str] = []
        self.batch_calls: List[List[str]] = []

    async def get_by_id(self, review_id: str):
        uuid.UUID(review_id)
        return self.row if self.row and self.row["review_id"] == review_id else None

    async def get_reviews_for_estimand(self, estimand_key: str, include_expired: bool = True):
        self.estimand_calls.append(estimand_key)
        return self.estimand_history

    async def get_reviews_for_dag(
        self, dag_hash: str, include_expired: bool = False, brand: Optional[str] = None
    ):
        self.dag_calls.append((dag_hash, include_expired, brand))
        return self.dag_history

    async def get_versions(self, review_id: str):
        self.version_calls.append(review_id)
        return self.versions

    async def get_versions_for_reviews(self, review_ids: List[str]):
        self.batch_calls.append(list(review_ids))
        return self.versions_by_review

    async def get_pending_reviews(self, brand=None, reviewer_id=None, limit=50):
        return self.pending

    async def get_review_summary(self, brand=None):
        return self.summary


def _client(monkeypatch, repo: _Repo) -> TestClient:
    async def _factory():
        return repo

    monkeypatch.setattr(route_mod, "_get_expert_review_repo", _factory)
    app = FastAPI()
    app.include_router(route_mod.router, prefix="/api")
    return TestClient(app)


def _detail_repo(**overrides) -> _Repo:
    kwargs: Dict[str, Any] = {
        "estimand_history": [ROW, ROW_OLDER],
        "versions": [V1, V2],
    }
    kwargs.update(overrides)
    return _Repo(ROW, **kwargs)


# --------------------------------------------------------------------------
# GET /{review_id}: the version timeline and its diffs
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_detail_returns_versions_oldest_first_with_diffs(monkeypatch):
    r = _client(monkeypatch, _detail_repo()).get(f"/api/expert-reviews/{RID}")
    assert r.status_code == 200, r.text
    versions = r.json()["versions"]
    assert len(versions) == 2
    assert [v["dag_version_hash"] for v in versions] == [HASH_V1, HASH_V2]
    # The first version has nothing to diff against -- None, never an
    # everything-added diff that would read as a real structure change.
    assert versions[0]["changes"] is None
    changes = versions[1]["changes"]
    assert changes["nodes_added"] == ["W"]
    assert changes["edges_added"] == [["W", "T"]]
    assert changes["adjustment_sets_added"] == [["W"]]
    assert changes["is_changed"] is True


@pytest.mark.unit
def test_detail_history_is_the_estimands_reviews_with_diffs(monkeypatch):
    repo = _detail_repo()
    body = _client(monkeypatch, repo).get(f"/api/expert-reviews/{RID}").json()
    assert repo.estimand_calls == [ESTIMAND]
    assert repo.dag_calls == []  # the hash-keyed read is the pre-140 fallback only
    history = body["history"]
    assert [h["review_id"] for h in history] == [RID, RID_OLDER]
    assert history[0]["changes_from_previous"]["nodes_added"] == ["W"]
    # The oldest review has no predecessor of this estimand.
    assert history[-1]["changes_from_previous"] is None


@pytest.mark.unit
def test_a_double_append_of_the_same_snapshot_diffs_to_no_change(monkeypatch):
    """mig 141 is a TIMELINE, not a set: a revert appends again. Two adjacent
    rows carrying the same structure must diff to is_changed False, not 500."""
    twin = _version("aaaaaaa3-0000-4000-8000-000000000003", HASH_V1, SNAP_V1, V2_AT)
    repo = _detail_repo(versions=[V1, twin])
    r = _client(monkeypatch, repo).get(f"/api/expert-reviews/{RID}")
    assert r.status_code == 200, r.text
    changes = r.json()["versions"][1]["changes"]
    assert changes["is_changed"] is False
    assert changes["nodes_added"] == [] and changes["nodes_removed"] == []
    assert changes["edges_added"] == [] and changes["edges_removed"] == []
    assert changes["adjustment_sets_added"] == [] and changes["adjustment_sets_removed"] == []


@pytest.mark.unit
def test_a_null_snapshot_diffs_against_an_empty_graph(monkeypatch):
    """mig 141 backfilled rows can carry a NULL ``dag_structure_json``; the
    diff against it is everything-added, and never a 500."""
    null_v = _version("aaaaaaa4-0000-4000-8000-000000000004", HASH_V1, None, V1_AT)
    repo = _detail_repo(versions=[null_v, V2])
    r = _client(monkeypatch, repo).get(f"/api/expert-reviews/{RID}")
    assert r.status_code == 200, r.text
    body = r.json()["versions"]
    assert body[0]["dag_structure_json"] is None and body[0]["changes"] is None
    changes = body[1]["changes"]
    assert changes["nodes_added"] == ["T", "W", "Y"]
    assert changes["edges_added"] == [["T", "Y"], ["W", "T"]]
    assert changes["adjustment_sets_added"] == [["W"]]
    assert changes["is_changed"] is True
    # A version with NO recorded structure has NO hash. Hashing the ``{}`` the
    # diff falls back to would emit a real-looking sha256 that matches nothing
    # in the database -- an operator comparing it to the row's stored
    # ``dag_version_hash`` would be reading a fabricated value.
    assert changes["old_hash"] is None
    assert isinstance(changes["new_hash"], str) and len(changes["new_hash"]) == 64


@pytest.mark.unit
def test_a_pre_140_row_falls_back_to_the_same_hash_history(monkeypatch):
    """No ``estimand_key`` (a row older than migration 140): the hash-keyed
    history is still served, and an empty timeline is an empty list."""
    repo = _Repo(ROW_PRE140, dag_history=[ROW_PRE140], versions=[])
    r = _client(monkeypatch, repo).get(f"/api/expert-reviews/{RID_PRE140}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert repo.estimand_calls == []
    assert repo.dag_calls == [(HASH_V2, True, "Kisqali")]
    assert body["versions"] == []
    assert [h["review_id"] for h in body["history"]] == [RID_PRE140]


# --------------------------------------------------------------------------
# GET /pending: version_count / last_changed_at
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_pending_items_carry_version_count_and_last_changed_at(monkeypatch):
    no_versions = {**ROW, "review_id": RID_NO_VERSIONS}
    repo = _Repo(
        pending=[ROW, no_versions],
        versions_by_review={RID: [V1, V2]},
    )
    r = _client(monkeypatch, repo).get("/api/expert-reviews/pending")
    assert r.status_code == 200, r.text
    items = {i["review_id"]: i for i in r.json()["reviews"]}
    assert items[RID]["version_count"] == 2
    assert datetime.fromisoformat(items[RID]["last_changed_at"]) == datetime(
        2026, 7, 14, 10, 0, tzinfo=timezone.utc
    )
    # A review that never moved is version 1, changed when it was created --
    # never 0 versions, which would read as "this review has no structure".
    assert items[RID_NO_VERSIONS]["version_count"] == 1
    assert datetime.fromisoformat(items[RID_NO_VERSIONS]["last_changed_at"]) == datetime(
        2026, 7, 13, 10, 0, tzinfo=timezone.utc
    )


@pytest.mark.unit
def test_pending_reads_every_reviews_versions_in_one_query(monkeypatch):
    """One batched read, not one per row: the queue is up to 200 rows."""
    repo = _Repo(pending=[ROW, {**ROW, "review_id": RID_NO_VERSIONS}], versions_by_review={})
    _client(monkeypatch, repo).get("/api/expert-reviews/pending")
    assert repo.batch_calls == [[RID, RID_NO_VERSIONS]]


# --------------------------------------------------------------------------
# GET /summary: the superseded partition (migration 140)
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_summary_carries_the_superseded_count(monkeypatch):
    repo = _Repo(
        summary={
            "pending": 3,
            "approved": 7,
            "rejected": 2,
            "superseded": 38,
            "expired": 1,
            "expiring_soon": 4,
        }
    )
    body = _client(monkeypatch, repo).get("/api/expert-reviews/summary").json()
    assert body["superseded"] == 38
    assert body["pending"] == 3 and body["approved"] == 7
    assert body["rejected"] == 2 and body["expired"] == 1 and body["expiring_soon"] == 4


@pytest.mark.unit
def test_pending_versions_store_failure_is_a_safe_503(monkeypatch):
    """The queue's SECOND read has its own outage path. The fake's other
    readers answer normally, so only a failing ``get_versions_for_reviews``
    can produce this 503 -- a queue served with every row at "version 1"
    would be a plausible-wrong page during an outage."""
    repo = _Repo(pending=[ROW])

    async def _boom(review_ids):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(repo, "get_versions_for_reviews", _boom)
    r = _client(monkeypatch, repo).get("/api/expert-reviews/pending")
    assert r.status_code == 503, r.text
    detail = r.json()["detail"]
    assert detail.startswith(SAFE_503_DETAIL_PREFIX)
    assert "Expert-review store unavailable" in detail
    assert "reviews" not in r.json(), "no partial queue may ride an error response"


@pytest.mark.unit
def test_a_non_object_snapshot_is_a_named_500_not_an_unhandled_crash(monkeypatch):
    """``expert_reviews.dag_structure_json`` carries NO CHECK constraint (mig 137
    added none), so a stored ARRAY reaches the diff. ``get_dag_changes`` calls
    ``.get`` on it -- an AttributeError, which is a bare 500 with no detail. The
    guard makes it the same honest 500 ``_validate_review_row`` raises: it names
    the review id and the column, so an operator can find the bad row."""
    bad_older = {**ROW_OLDER, "dag_structure_json": ["T", "Y"]}
    repo = _detail_repo(estimand_history=[ROW, bad_older])
    r = _client(monkeypatch, repo).get(f"/api/expert-reviews/{RID}")
    assert r.status_code == 500, r.text
    detail = str(r.json()["detail"])
    assert RID_OLDER in detail
    assert "dag_structure_json" in detail
