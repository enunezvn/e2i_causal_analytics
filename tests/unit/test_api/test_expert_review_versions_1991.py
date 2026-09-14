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
from src.api.schemas.expert_review import ReviewVersion
from src.causal_engine.dag_hash import compute_adjustment_set_hash

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


# --------------------------------------------------------------------------
# GET /{review_id}: which timeline row is the review's CURRENT version
# (codex round 3 HIGH -- the panel must not diff a LOSING version)
# --------------------------------------------------------------------------

ADJ_A = "a" * 64
ADJ_B = "b" * 64
ADJ_C = "c" * 64

# DISTINCT from every ADJ_* above, deliberately: with HASH_C == ADJ_C the orphan
# pins below would survive a helper that compared the DAG hash against the
# adjustment field and vice versa.
HASH_B = "5" * 64
HASH_C = "6" * 64

# The orphan timeline the gate's CAS permits (gate test ~:1295): both runs read
# A; C inserted AND advanced; B inserted afterwards and LOST its advance. The
# review is on C while the timeline ends ... C, B.
SNAP_A: Dict[str, Any] = {"nodes": ["T", "Y"], "edges": [["T", "Y"]], "adjustment_sets": []}
SNAP_C: Dict[str, Any] = {
    "nodes": ["T", "Y", "W"],
    "edges": [["T", "Y"], ["W", "T"]],
    "adjustment_sets": [["W"]],
}
SNAP_B: Dict[str, Any] = {
    "nodes": ["T", "Y", "Z"],
    "edges": [["T", "Y"], ["Z", "T"]],
    "adjustment_sets": [["Z"]],
}

# The hash a run WOULD have stored for each snapshot above -- what a
# NULL-adjustment row carrying that snapshot proves (codex round 4).
ADJ_SNAP_A = compute_adjustment_set_hash(SNAP_A["adjustment_sets"])
ADJ_SNAP_C = compute_adjustment_set_hash(SNAP_C["adjustment_sets"])
ADJ_SNAP_B = compute_adjustment_set_hash(SNAP_B["adjustment_sets"])

VID_A = "bbbbbbb1-0000-4000-8000-000000000001"
VID_C = "bbbbbbb2-0000-4000-8000-000000000002"
VID_B = "bbbbbbb3-0000-4000-8000-000000000003"


def _paired(
    version_id: str,
    dag_hash: str,
    adjustment_hash: Optional[str],
    snapshot: Optional[Dict[str, Any]] = None,
    created_at: str = V1_AT,
) -> Dict[str, Any]:
    """A timeline row carrying BOTH halves of a version identity."""
    return {
        **_version(version_id, dag_hash, snapshot, created_at),
        "adjustment_set_hash": adjustment_hash,
    }


def _rv(
    version_id: str,
    dag_hash: str,
    adjustment_hash: Optional[str],
    snapshot: Optional[Dict[str, Any]] = None,
) -> ReviewVersion:
    return ReviewVersion(
        version_id=version_id,
        dag_version_hash=dag_hash,
        adjustment_set_hash=adjustment_hash,
        dag_structure_json=snapshot,
    )


@pytest.mark.unit
def test_current_version_id_picks_the_pair_match_not_the_last_row():
    """The orphan state: the review is on C, the timeline ends on B."""
    versions = [_rv(VID_A, HASH_V1, ADJ_A), _rv(VID_C, HASH_C, ADJ_C), _rv(VID_B, HASH_B, ADJ_B)]
    assert route_mod._current_version_id(versions, HASH_C, ADJ_C, None) == VID_C


@pytest.mark.unit
def test_current_version_id_matches_a_null_adjustment_on_both_sides():
    """A review whose adjustment is UNKNOWN matches the row that is also
    unknown -- None equals None, it is not a wildcard."""
    versions = [_rv(VID_A, HASH_V1, None), _rv(VID_B, HASH_V2, ADJ_B)]
    assert route_mod._current_version_id(versions, HASH_V1, None, None) == VID_A


@pytest.mark.unit
def test_current_version_id_falls_back_to_a_null_adjustment_row_that_proves_it():
    """The review learned its adjustment hash (migration 142) but the only
    recorded version of that structure was backfilled with NULL. The row's
    SNAPSHOT still names the covariate set, and it derives to the review's
    hash -- same structure, same covariates: that row IS the current version."""
    versions = [_rv(VID_A, HASH_V1, None, SNAP_C)]
    assert route_mod._current_version_id(versions, HASH_V1, ADJ_SNAP_C, None) == VID_A


@pytest.mark.unit
def test_current_version_id_prefers_the_exact_pair_over_the_null_fallback():
    """The NULL row is COMPATIBLE -- its snapshot derives to the review's hash,
    so both rows name the same covariate set. Equal effective values, so this
    pins the tie-break (the LAST one wins), not a disqualification."""
    versions = [_rv(VID_A, HASH_V1, None, SNAP_C), _rv(VID_C, HASH_V1, ADJ_SNAP_C)]
    assert route_mod._current_version_id(versions, HASH_V1, ADJ_SNAP_C, None) == VID_C


@pytest.mark.unit
def test_current_version_id_is_none_when_no_row_carries_the_reviews_structure():
    versions = [_rv(VID_A, HASH_V1, ADJ_A)]
    assert route_mod._current_version_id(versions, HASH_V2, ADJ_B, None) is None
    # A review with no hash at all names no version either.
    assert route_mod._current_version_id(versions, None, None, None) is None
    assert route_mod._current_version_id([], HASH_V1, ADJ_A, None) is None


@pytest.mark.unit
def test_current_version_id_takes_the_LAST_match_on_a_revert():
    """A revert A -> B -> A appends A again (the table is a timeline, not a
    set). The review points at the NEWER A, whose ``changes`` is the B -> A
    delta the reviewer is being asked to approve."""
    a2 = "bbbbbbb4-0000-4000-8000-000000000004"
    versions = [_rv(VID_A, HASH_V1, ADJ_A), _rv(VID_B, HASH_V2, ADJ_B), _rv(a2, HASH_V1, ADJ_A)]
    assert route_mod._current_version_id(versions, HASH_V1, ADJ_A, None) == a2
    # and the null-adjustment fallback takes the last PROVABLE one too
    n1 = "bbbbbbb5-0000-4000-8000-000000000005"
    n2 = "bbbbbbb6-0000-4000-8000-000000000006"
    assert (
        route_mod._current_version_id(
            [_rv(n1, HASH_V1, None, SNAP_C), _rv(n2, HASH_V1, None, SNAP_C)],
            HASH_V1,
            ADJ_SNAP_C,
            None,
        )
        == n2
    )


@pytest.mark.unit
def test_detail_names_the_current_version_when_the_timeline_ends_on_an_orphan(monkeypatch):
    """End to end: the response names C, and C's own ``changes`` is the A -> C
    delta -- while B stays in the timeline with its (losing) C -> B delta."""
    row = {
        **ROW,
        "dag_version_hash": HASH_C,
        "adjustment_set_hash": ADJ_C,
        "dag_structure_json": SNAP_C,
    }
    versions = [
        _paired(VID_A, HASH_V1, ADJ_A, SNAP_A, V1_AT),
        _paired(VID_C, HASH_C, ADJ_C, SNAP_C, V2_AT),
        _paired(VID_B, HASH_B, ADJ_B, SNAP_B, V2_AT),
    ]
    repo = _Repo(row, estimand_history=[row], versions=versions)
    r = _client(monkeypatch, repo).get(f"/api/expert-reviews/{RID}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["current_version_id"] == VID_C

    by_id = {v["version_id"]: v for v in body["versions"]}
    current = by_id[VID_C]["changes"]
    assert current["nodes_added"] == ["W"]
    assert current["adjustment_sets_added"] == [["W"]]
    # The orphan's delta is a TIMELINE FACT and stays on its own row; it is
    # simply no longer the one the panel renders.
    assert by_id[VID_B]["changes"]["nodes_added"] == ["Z"]
    assert by_id[VID_B]["changes"]["nodes_removed"] == ["W"]


@pytest.mark.unit
def test_detail_current_version_id_is_null_when_no_row_matches(monkeypatch):
    """A review minted before the versions table (or whose first version was
    never recorded) names NO current version -- the panel then shows silence
    rather than a possibly-wrong delta."""
    row = {**ROW, "dag_version_hash": HASH_C, "adjustment_set_hash": ADJ_C}
    repo = _Repo(row, estimand_history=[row], versions=[_paired(VID_A, HASH_V1, ADJ_A, SNAP_A)])
    body = _client(monkeypatch, repo).get(f"/api/expert-reviews/{RID}").json()
    assert body["current_version_id"] is None


@pytest.mark.unit
def test_detail_current_version_id_falls_back_to_a_backfilled_version(monkeypatch):
    """Migration 141 backfilled versions carry NULL adjustment hashes; a review
    that has since learned its own (migration 142) still resolves to them --
    when the row's snapshot derives that hash.

    Seeded with a NON-EMPTY adjustment set on purpose: the empty-set hash is
    what ANY dict snapshot derives, so an empty seed could not tell "derived
    from THIS snapshot's covariates" from "any dict qualifies"."""
    row = {**ROW, "dag_version_hash": HASH_V1, "adjustment_set_hash": ADJ_SNAP_C}
    repo = _Repo(row, estimand_history=[row], versions=[_paired(VID_A, HASH_V1, None, SNAP_C)])
    body = _client(monkeypatch, repo).get(f"/api/expert-reviews/{RID}").json()
    assert body["current_version_id"] == VID_A


# --------------------------------------------------------------------------
# GET /pending: last_changed_at follows the CURRENT version, not the tail
# --------------------------------------------------------------------------

V3_AT = "2026-07-15T10:00:00+00:00"


@pytest.mark.unit
def test_pending_last_changed_at_is_the_current_versions_timestamp(monkeypatch):
    """The queue's "last changed" is when the structure UNDER REVIEW landed.

    In the orphan state the timeline ends on the losing row, whose timestamp is
    the NEWEST -- reporting it would tell the operator their review moved at a
    moment it did not.
    """
    row = {**ROW, "dag_version_hash": HASH_C, "adjustment_set_hash": ADJ_C}
    versions = [
        _paired(VID_A, HASH_V1, ADJ_A, SNAP_A, V1_AT),
        _paired(VID_C, HASH_C, ADJ_C, SNAP_C, V2_AT),
        _paired(VID_B, HASH_B, ADJ_B, SNAP_B, V3_AT),
    ]
    repo = _Repo(pending=[row], versions_by_review={RID: versions})
    item = _client(monkeypatch, repo).get("/api/expert-reviews/pending").json()["reviews"][0]
    # The COUNT stays a timeline fact: three rows were recorded for this review.
    assert item["version_count"] == 3
    assert datetime.fromisoformat(item["last_changed_at"]) == datetime(
        2026, 7, 14, 10, 0, tzinfo=timezone.utc
    )


@pytest.mark.unit
def test_pending_last_changed_at_falls_back_to_the_row_when_no_version_matches(monkeypatch):
    """No recorded version carries the review's pair, so WHEN the current
    structure landed is unknown -- the review's own creation is the honest
    answer, exactly as for a review with no timeline at all. Never the newest
    row's timestamp, which belongs to a structure this review is not on."""
    row = {**ROW, "dag_version_hash": HASH_C, "adjustment_set_hash": ADJ_C}
    versions = [
        _paired(VID_A, HASH_V1, ADJ_A, SNAP_A, V2_AT),
        _paired(VID_B, HASH_B, ADJ_B, SNAP_B, V3_AT),
    ]
    repo = _Repo(pending=[row], versions_by_review={RID: versions})
    item = _client(monkeypatch, repo).get("/api/expert-reviews/pending").json()["reviews"][0]
    assert item["version_count"] == 2
    assert datetime.fromisoformat(item["last_changed_at"]) == datetime(
        2026, 7, 13, 10, 0, tzinfo=timezone.utc
    )


# --------------------------------------------------------------------------
# The NULL-adjustment fallback must PROVE compatibility (codex round 4)
# --------------------------------------------------------------------------
#
# A same-hash row whose ``adjustment_set_hash`` is NULL used to win the
# fallback unexamined. Its SNAPSHOT is the evidence: derived with the writer's
# own function it either names the review's covariate set or it does not, and a
# NULL snapshot names nothing at all.

# The sequence codex executed. The timeline holds (h, A) and then a (h, NULL)
# row with a NULL snapshot -- a run that had no structure in scope. A later run
# supplied structure B while ``get_latest_version`` was raising, so it did not
# APPEND (the row asserted no change: same DAG hash, unknown adjustment) but it
# did ADVANCE the review onto (h, B). The (h, NULL) row is then the only
# same-hash candidate left.
_UNPROVABLE_TIMELINE = [
    (VID_A, HASH_V1, ADJ_A, SNAP_A, V1_AT),
    (VID_B, HASH_V1, None, None, V3_AT),
]


@pytest.mark.unit
def test_current_version_id_refuses_a_null_adjustment_row_that_proves_nothing():
    """Naming that row would render "the previous snapshot disappeared" beside
    B's populated graph -- the wrong-delta/right-graph defect, rebuilt. No
    provable current version is the honest answer; the panel then shows the
    graph with no diff."""
    versions = [_rv(vid, h, adj, snap) for vid, h, adj, snap, _at in _UNPROVABLE_TIMELINE]
    assert route_mod._current_version_id(versions, HASH_V1, ADJ_SNAP_B, None) is None


@pytest.mark.unit
def test_current_version_id_refuses_a_backfilled_row_naming_other_covariates():
    """Same DAG hash, but the snapshot's adjustment sets are not the review's:
    ``compute_dag_hash`` excludes covariates, so the hash alone cannot see it."""
    versions = [_rv(VID_A, HASH_V1, None, SNAP_C)]
    assert route_mod._current_version_id(versions, HASH_V1, ADJ_SNAP_B, None) is None


@pytest.mark.unit
def test_a_dict_snapshot_without_adjustment_sets_proves_the_EMPTY_set():
    """Absent covariates inside a real snapshot is a fact -- the empty set --
    unlike a NULL snapshot, which is the absence of any structure at all."""
    versions = [_rv(VID_A, HASH_V1, None, {"nodes": ["T", "Y"], "edges": [["T", "Y"]]})]
    assert route_mod._current_version_id(versions, HASH_V1, ADJ_SNAP_A, None) == VID_A
    assert ADJ_SNAP_A == compute_adjustment_set_hash([])


@pytest.mark.unit
def test_detail_current_version_id_is_null_when_the_only_candidate_proves_nothing(monkeypatch):
    """End to end on codex's sequence: the response names NO current version,
    while the timeline itself is unchanged -- the rows WERE recorded."""
    row = {
        **ROW,
        "dag_version_hash": HASH_V1,
        "adjustment_set_hash": ADJ_SNAP_B,
        "dag_structure_json": SNAP_B,
    }
    versions = [_paired(vid, h, adj, snap, at) for vid, h, adj, snap, at in _UNPROVABLE_TIMELINE]
    repo = _Repo(row, estimand_history=[row], versions=versions)
    r = _client(monkeypatch, repo).get(f"/api/expert-reviews/{RID}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["current_version_id"] is None
    assert [v["version_id"] for v in body["versions"]] == [VID_A, VID_B]


@pytest.mark.unit
def test_pending_last_changed_at_ignores_a_null_row_that_proves_nothing(monkeypatch):
    """The queue reads the same rule over RAW rows: with no provable current
    version the change date is unknown, so it is the review's own creation --
    never the unprovable row's, which is the NEWEST timestamp here."""
    row = {**ROW, "dag_version_hash": HASH_V1, "adjustment_set_hash": ADJ_SNAP_B}
    versions = [_paired(vid, h, adj, snap, at) for vid, h, adj, snap, at in _UNPROVABLE_TIMELINE]
    repo = _Repo(pending=[row], versions_by_review={RID: versions})
    item = _client(monkeypatch, repo).get("/api/expert-reviews/pending").json()["reviews"][0]
    assert item["version_count"] == 2
    assert datetime.fromisoformat(item["last_changed_at"]) == datetime(
        2026, 7, 13, 10, 0, tzinfo=timezone.utc
    )


@pytest.mark.unit
def test_pending_last_changed_at_accepts_a_backfilled_row_whose_snapshot_proves_it(monkeypatch):
    """The other half of the rule on the queue's raw rows: a NULL-adjustment
    row that DOES derive to the review's hash still dates the change."""
    row = {**ROW, "dag_version_hash": HASH_V1, "adjustment_set_hash": ADJ_SNAP_C}
    versions = [_paired(VID_A, HASH_V1, None, SNAP_C, V2_AT)]
    repo = _Repo(pending=[row], versions_by_review={RID: versions})
    item = _client(monkeypatch, repo).get("/api/expert-reviews/pending").json()["reviews"][0]
    assert datetime.fromisoformat(item["last_changed_at"]) == datetime(
        2026, 7, 14, 10, 0, tzinfo=timezone.utc
    )


# --------------------------------------------------------------------------
# The rule compares the EFFECTIVE adjustment on BOTH sides (codex round 5)
# --------------------------------------------------------------------------
#
# ``effective(adjustment, snapshot)`` is the adjustment hash when the column
# carries one, else what the snapshot DERIVES, else None. Comparing it on both
# sides subsumes the round-4 fallback (a known review vs a NULL row that proves
# it) AND closes round 5's finding 1: a review whose own adjustment column is
# NULL but whose SNAPSHOT names its covariates is not "unknown", so an
# information-free ``(h, NULL, NULL)`` row no longer matches it by default.

# Codex round 5's sequence, as the timeline it leaves behind. A post-141/142
# review carries (h, NULL, snapshot A); its backfilled version carries the
# same; a structureless run of h then recorded (h, NULL, NULL snapshot).
_STRUCTURELESS_TAIL = [
    (VID_A, HASH_V1, None, SNAP_C, V1_AT),
    (VID_B, HASH_V1, None, None, V3_AT),
]


@pytest.mark.unit
def test_current_version_id_prefers_the_backfill_that_proves_the_reviews_snapshot():
    """The review's adjustment column is NULL, but its SNAPSHOT names the
    covariate set -- so the review is NOT unknown, and the row that proves the
    same set wins over the later row that proves nothing.

    Before this rule the exact NULL == NULL match took the LAST such row, which
    is the information-free one: the panel showed the review's real graph beside
    "the previous snapshot disappeared"."""
    versions = [_rv(vid, h, adj, snap) for vid, h, adj, snap, _at in _STRUCTURELESS_TAIL]
    assert route_mod._current_version_id(versions, HASH_V1, None, SNAP_C) == VID_A


@pytest.mark.unit
def test_current_version_id_refuses_a_null_row_when_the_review_has_a_snapshot():
    """The same sequence with the backfill absent: nothing proves the review's
    covariate set, so there is no current version. Silence, not the wrong
    delta."""
    versions = [_rv(VID_B, HASH_V1, None, None)]
    assert route_mod._current_version_id(versions, HASH_V1, None, SNAP_C) is None


@pytest.mark.unit
def test_current_version_id_matches_when_BOTH_sides_are_genuinely_unknown():
    """The legitimate no-structure recording, preserved: a review that carries
    no snapshot and no adjustment hash still matches the row that carries
    neither. None == None means "both sides are genuinely unknown", which is a
    match -- it is only not a WILDCARD."""
    versions = [_rv(VID_B, HASH_V1, None, None)]
    assert route_mod._current_version_id(versions, HASH_V1, None, None) == VID_B


@pytest.mark.unit
def test_current_version_id_matches_a_known_row_from_the_reviews_snapshot_alone():
    """The other direction the effective rule subsumes: the review's column is
    NULL but its snapshot derives C, and the recorded row states C outright.
    Same structure, same covariates -- the round-4 fallback could not see this,
    because it only ran when the REVIEW's hash was known."""
    versions = [_rv(VID_C, HASH_V1, ADJ_SNAP_C, None)]
    assert route_mod._current_version_id(versions, HASH_V1, None, SNAP_C) == VID_C


@pytest.mark.unit
def test_detail_names_the_backfill_not_the_structureless_tail(monkeypatch):
    """End to end on codex round 5's sequence: the detail names the row that
    proves the review's covariates, and the timeline keeps both rows."""
    row = {
        **ROW,
        "dag_version_hash": HASH_V1,
        "adjustment_set_hash": None,
        "dag_structure_json": SNAP_C,
    }
    versions = [_paired(vid, h, adj, snap, at) for vid, h, adj, snap, at in _STRUCTURELESS_TAIL]
    repo = _Repo(row, estimand_history=[row], versions=versions)
    r = _client(monkeypatch, repo).get(f"/api/expert-reviews/{RID}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["current_version_id"] == VID_A
    assert [v["version_id"] for v in body["versions"]] == [VID_A, VID_B]


@pytest.mark.unit
def test_detail_current_version_id_is_null_when_only_a_structureless_row_remains(monkeypatch):
    row = {
        **ROW,
        "dag_version_hash": HASH_V1,
        "adjustment_set_hash": None,
        "dag_structure_json": SNAP_C,
    }
    versions = [_paired(VID_B, HASH_V1, None, None, V3_AT)]
    repo = _Repo(row, estimand_history=[row], versions=versions)
    body = _client(monkeypatch, repo).get(f"/api/expert-reviews/{RID}").json()
    assert body["current_version_id"] is None


@pytest.mark.unit
def test_pending_last_changed_at_dates_the_backfill_not_the_structureless_tail(monkeypatch):
    """The queue reads the same rule over RAW rows, and its review snapshot is
    the raw column read through ``parse_json_column``. The structureless row is
    the NEWEST, so taking it would date a move that never happened."""
    row = {
        **ROW,
        "dag_version_hash": HASH_V1,
        "adjustment_set_hash": None,
        "dag_structure_json": SNAP_C,
    }
    versions = [_paired(vid, h, adj, snap, at) for vid, h, adj, snap, at in _STRUCTURELESS_TAIL]
    repo = _Repo(pending=[row], versions_by_review={RID: versions})
    item = _client(monkeypatch, repo).get("/api/expert-reviews/pending").json()["reviews"][0]
    assert item["version_count"] == 2
    assert datetime.fromisoformat(item["last_changed_at"]) == datetime(
        2026, 7, 13, 10, 0, tzinfo=timezone.utc
    )


@pytest.mark.unit
def test_pending_last_changed_at_falls_back_when_only_a_structureless_row_remains(monkeypatch):
    """No provable current version, so the change date is unknown and the
    review's own creation is the honest answer -- never the tail's."""
    row = {
        **ROW,
        "dag_version_hash": HASH_V1,
        "adjustment_set_hash": None,
        "dag_structure_json": SNAP_C,
        "created_at": V2_AT,
    }
    versions = [_paired(VID_B, HASH_V1, None, None, V3_AT)]
    repo = _Repo(pending=[row], versions_by_review={RID: versions})
    item = _client(monkeypatch, repo).get("/api/expert-reviews/pending").json()["reviews"][0]
    assert datetime.fromisoformat(item["last_changed_at"]) == datetime(
        2026, 7, 14, 10, 0, tzinfo=timezone.utc
    )


@pytest.mark.unit
def test_pending_last_changed_at_still_dates_a_genuinely_unknown_pair(monkeypatch):
    """The legitimate case through the queue: a review with no snapshot and no
    adjustment hash still dates itself by the row that recorded the same."""
    row = {
        **ROW,
        "dag_version_hash": HASH_V1,
        "adjustment_set_hash": None,
        "dag_structure_json": None,
    }
    versions = [_paired(VID_B, HASH_V1, None, None, V3_AT)]
    repo = _Repo(pending=[row], versions_by_review={RID: versions})
    item = _client(monkeypatch, repo).get("/api/expert-reviews/pending").json()["reviews"][0]
    assert datetime.fromisoformat(item["last_changed_at"]) == datetime(
        2026, 7, 15, 10, 0, tzinfo=timezone.utc
    )


@pytest.mark.unit
def test_pending_survives_a_version_row_with_malformed_adjustment_data(monkeypatch):
    """Codex round 5, finding 2. ``{"adjustment_sets": [null]}`` satisfies
    migration 141's outer-object CHECK, so the table can hold it -- and the
    queue derives hashes OUTSIDE its 503-guarded database read, so one such row
    raising TypeError aborted the WHOLE pending response.

    It proves nothing, so it names no version and the review dates itself by its
    own creation. The rows are still counted: they WERE recorded."""
    row = {
        **ROW,
        "dag_version_hash": HASH_V1,
        "adjustment_set_hash": ADJ_SNAP_C,
        "created_at": V2_AT,
    }
    versions = [_paired(VID_B, HASH_V1, None, {"adjustment_sets": [None]}, V3_AT)]
    repo = _Repo(pending=[row], versions_by_review={RID: versions})
    r = _client(monkeypatch, repo).get("/api/expert-reviews/pending")
    assert r.status_code == 200, r.text
    item = r.json()["reviews"][0]
    assert item["version_count"] == 1
    assert datetime.fromisoformat(item["last_changed_at"]) == datetime(
        2026, 7, 14, 10, 0, tzinfo=timezone.utc
    )


@pytest.mark.unit
def test_a_malformed_snapshot_on_the_REVIEW_row_is_the_named_500_not_a_TypeError(monkeypatch):
    """The other side the derivation now reads: the review's OWN snapshot.

    ``expert_reviews.dag_structure_json`` carries no CHECK constraint, so this
    shape is storable, and ``_last_changed_at`` derives from it BEFORE the row
    is validated -- the derivation raising TypeError there aborted the queue
    with an unhandled error. It now proves nothing, so the route reaches its own
    validation and answers the named 500 it already had for a malformed review
    snapshot. That behaviour is unchanged and deliberate; only the unhandled
    crash in front of it is gone."""
    row = {
        **ROW,
        "dag_version_hash": HASH_V1,
        "adjustment_set_hash": None,
        "dag_structure_json": {"adjustment_sets": [["A", 1]]},
    }
    versions = [_paired(VID_B, HASH_V1, None, None, V3_AT)]
    repo = _Repo(pending=[row], versions_by_review={RID: versions})
    r = _client(monkeypatch, repo).get("/api/expert-reviews/pending")
    assert r.status_code == 500
    assert "malformed dag_structure_json" in r.json()["detail"]
