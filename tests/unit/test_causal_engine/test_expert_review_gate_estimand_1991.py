"""#1991 debt 3 -- expert reviews are keyed on the ESTIMAND, not on one DAG hash.

Identity vs version (migrations 140/141):

* IDENTITY is the estimand ``lower(brand):treatment:outcome`` -- the generated
  ``expert_reviews.estimand_key`` column, with a partial UNIQUE index allowing
  ONE pending review per estimand. A covariate or structure change must UPDATE
  the pending review of an estimand, never mint a sibling that splits a
  reviewer's attention across rows nobody can reconcile.
* VERSION is the DAG hash. ``expert_review_versions`` is a TIMELINE, so a
  differing hash APPENDS a version row (and advances the pending review), an
  unchanged hash appends nothing, and a revert (A -> B -> A) appends a third
  row rather than being suppressed -- same-hash idempotence is this gate's
  job, not a constraint's.

An APPROVAL stays scoped to the hash it was granted on (spec §7): the same
estimand on a NEW structure re-opens a pending review that records the earlier
approval in ``supersedes_review_id``, and the old approval keeps its validity
for its own hash. ``get_dag_approval`` is therefore no longer consulted by
``check_approval`` -- it answers "is THIS hash approved" without the estimand's
chronology, and the history read already carries the answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.causal_engine.expert_review_gate import ExpertReviewGate, ReviewGateDecision
from src.repositories.expert_review import estimand_key_for


class _EstimandRepo:
    """ExpertReviewRepository stand-in keyed on the estimand (no live DB).

    Records every write so the negatives ("no second row", "nothing appended")
    are positive-controlled by the calls that DID happen. ``create_review``
    records the estimand_key the DB would GENERATE for the row's brand /
    treatment / outcome -- the real writer never sends it (migration 140 made
    the column ``GENERATED ALWAYS``), so deriving it here is the only way to
    assert which estimand the row lands on.
    """

    def __init__(self, history: Optional[List[Dict[str, Any]]] = None) -> None:
        self.history = list(history or [])
        self.created: List[Dict[str, Any]] = []
        self.appended: List[tuple] = []
        self.append_kwargs: List[Dict[str, Any]] = []
        self.structure_updates: List[tuple] = []
        self.dag_approval_calls = 0
        self.estimand_lookups: List[str] = []

    async def get_reviews_for_estimand(
        self, estimand_key: str, include_expired: bool = True
    ) -> List[Dict[str, Any]]:
        self.estimand_lookups.append(estimand_key)
        return [r for r in self.history if r.get("estimand_key") == estimand_key]

    async def get_reviews_for_dag(
        self, dag_hash: str, include_expired: bool = False, brand: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return [r for r in self.history if r.get("dag_version_hash") == dag_hash]

    async def get_dag_approval(self, dag_hash: str, brand: Optional[str] = None):
        self.dag_approval_calls += 1
        return None

    async def create_review(self, **kwargs: Any) -> str:
        recorded = dict(kwargs)
        recorded["estimand_key"] = estimand_key_for(
            kwargs.get("brand"),
            kwargs.get("treatment_variable"),
            kwargs.get("outcome_variable"),
        )
        self.created.append(recorded)
        return "rev-new"

    async def append_version(self, review_id: str, **kwargs: Any) -> bool:
        self.appended.append((review_id, kwargs.get("dag_version_hash")))
        self.append_kwargs.append({"review_id": review_id, **kwargs})
        return True

    async def update_dag_structure(
        self, review_id: str, dag_structure, related_validation_ids=None
    ) -> bool:
        self.structure_updates.append((review_id, dag_structure, related_validation_ids))
        return True


def _pending(review_id: str, dag_hash: str, created_at: str = "2026-09-01T00:00:00+00:00"):
    return {
        "review_id": review_id,
        "approval_status": "pending",
        "dag_version_hash": dag_hash,
        "estimand_key": "b:t:y",
        "created_at": created_at,
    }


def _approved(
    review_id: str,
    dag_hash: str,
    valid_until: str = "2099-01-01",
    created_at: str = "2026-09-01T00:00:00+00:00",
):
    return {
        "review_id": review_id,
        "approval_status": "approved",
        "dag_version_hash": dag_hash,
        "estimand_key": "b:t:y",
        "valid_until": valid_until,
        "created_at": created_at,
    }


def _rejected(review_id: str, dag_hash: str, created_at: str = "2026-09-01T00:00:00+00:00"):
    return {
        "review_id": review_id,
        "approval_status": "rejected",
        "dag_version_hash": dag_hash,
        "estimand_key": "b:t:y",
        "created_at": created_at,
    }


_GRAPH = {"nodes": ["T", "Y"], "edges": [["T", "Y"]]}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_review_band_new_estimand_mints_once_with_first_version():
    repo = _EstimandRepo(history=[])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure=_GRAPH,
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW and r.review_id == "rev-new"
    assert len(repo.created) == 1 and repo.created[0]["estimand_key"] == "b:t:y"
    assert repo.appended == [("rev-new", "h1")]  # version 1 recorded on mint
    assert repo.dag_approval_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_review_band_same_estimand_new_hash_appends_version_not_row():
    repo = _EstimandRepo(history=[_pending("r1", "h1")])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h2",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={"nodes": ["T", "Y", "W"], "edges": [["W", "T"], ["T", "Y"]]},
        related_validation_ids=["v1", "v2"],
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW and r.review_id == "r1"
    assert repo.created == [] and repo.appended == [("r1", "h2")]
    # The advanced review must point at THIS run's evidence: its hash is now h2,
    # so related_validation_ids left on h1's run would render the wrong evidence
    # in the review UI (src/api/routes/expert_review.py reads that column).
    assert repo.append_kwargs[0]["related_validation_ids"] == ["v1", "v2"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_same_hash_appends_nothing():
    repo = _EstimandRepo(history=[_pending("r1", "h1")])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1", brand="B", treatment="T", outcome="Y", requester_id="q"
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW
    assert repo.appended == [] and repo.created == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_revert_to_an_earlier_hash_appends_again():
    """Timeline semantics (migration 141): A -> B -> A appends a third version."""
    repo = _EstimandRepo(history=[_pending("r1", "hB")])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="hA", brand="B", treatment="T", outcome="Y", requester_id="q"
    )

    assert repo.appended == [("r1", "hA")]
    # No structure in scope means the adjustment set is UNKNOWN, which migration
    # 141 stores as NULL. sha256("[]") is the canonical EMPTY set -- a different
    # fact, and one this call has no basis to assert.
    assert repo.append_kwargs[0]["adjustment_set_hash"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_valid_approval_on_the_same_hash_is_approved():
    repo = _EstimandRepo(history=[_approved("r0", "h1")])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1", brand="B", treatment="T", outcome="Y", requester_id="q"
    )

    assert r.decision == ReviewGateDecision.PROCEED and r.is_approved
    assert repo.created == []
    assert repo.dag_approval_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_approved_estimand_on_new_hash_reopens_with_supersedes():
    repo = _EstimandRepo(history=[_approved("r0", "h1")])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h2", brand="B", treatment="T", outcome="Y", requester_id="q"
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW
    assert repo.created[0]["supersedes_review_id"] == "r0"
    assert repo.appended == [("rev-new", "h2")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expired_approval_on_another_hash_does_not_supersede():
    """Positive control for the re-open: only a VALID approval is superseded."""
    repo = _EstimandRepo(history=[_approved("r0", "h1", valid_until="2020-01-01")])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h2", brand="B", treatment="T", outcome="Y", requester_id="q"
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW
    assert repo.created[0].get("supersedes_review_id") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rejection_on_the_estimand_is_still_authoritative():
    repo = _EstimandRepo(history=[_rejected("r0", "h1")])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1", brand="B", treatment="T", outcome="Y", requester_id="q"
    )

    assert r.decision == ReviewGateDecision.REJECTED and repo.created == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rejection_on_another_hash_of_the_estimand_does_not_block_a_new_structure():
    """A rejection covers the VERSION it was given on, not the estimand (§7).

    A reviewer who rejects hash A is saying "this structure is wrong", which is
    exactly the thing a revised structure fixes. Reading the rejection as a
    verdict on the estimand would make a rejected question un-re-openable by
    re-running, and would put ``check_approval`` (estimand-keyed) at odds with
    the read-only ``check_rejection`` probe (hash-keyed), which answers "not
    rejected" for B.
    """
    repo = _EstimandRepo(history=[_rejected("r0", "h1")])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h2", brand="B", treatment="T", outcome="Y", requester_id="q"
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW
    assert len(repo.created) == 1
    # supersedes_review_id links a SUPERSEDED APPROVAL, never a rejection (§7).
    assert repo.created[0].get("supersedes_review_id") is None
    assert repo.appended == [("rev-new", "h2")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_newer_approval_of_another_hash_does_not_revive_a_rejected_one():
    """Chronology is read over the SAME-HASH rows -- the rows the probe sees.

    Reject h1, revise to h2, approve h2, then re-run h1. Ranking the estimand's
    whole history would put the h2 approval on top, find no rejection "latest",
    and mint a pending review for the structure a human already turned down --
    while ``check_rejection(h1)`` still answers REJECTED. The two readers share
    one row set so they cannot disagree.
    """
    repo = _EstimandRepo(
        history=[
            _approved("r2", "h2", created_at="2026-09-05T00:00:00+00:00"),
            _rejected("r1", "h1"),
        ]
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1", brand="B", treatment="T", outcome="Y", requester_id="q"
    )

    assert r.decision == ReviewGateDecision.REJECTED and r.review_id == "r1"
    assert repo.created == [] and repo.appended == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_newer_rejection_of_another_hash_does_not_void_this_hashs_approval():
    """The mirror of the test above (§7): h1 keeps its own approval."""
    repo = _EstimandRepo(
        history=[
            _rejected("r2", "h2", created_at="2026-09-05T00:00:00+00:00"),
            _approved("r1", "h1"),
        ]
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1", brand="B", treatment="T", outcome="Y", requester_id="q"
    )

    assert r.decision == ReviewGateDecision.PROCEED and r.is_approved
    assert r.review_id == "r1"
    assert repo.created == [] and repo.appended == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_pending_review_of_another_hash_does_not_void_this_hashs_approval():
    """An approved structure still clears while a REVISED one is under review.

    The approval search precedes the pending branch, so re-running h1 proceeds
    on its own approval and appends nothing to r2's timeline; re-running h2
    lands on the open review, unchanged.
    """
    repo = _EstimandRepo(
        history=[
            _pending("r2", "h2", created_at="2026-09-05T00:00:00+00:00"),
            _approved("r1", "h1"),
        ]
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    on_h1 = await gate.check_approval(
        dag_hash="h1", brand="B", treatment="T", outcome="Y", requester_id="q"
    )
    assert on_h1.decision == ReviewGateDecision.PROCEED and on_h1.review_id == "r1"
    assert repo.appended == []

    on_h2 = await gate.check_approval(
        dag_hash="h2", brand="B", treatment="T", outcome="Y", requester_id="q"
    )
    assert on_h2.decision == ReviewGateDecision.PENDING_REVIEW and on_h2.review_id == "r2"
    assert repo.appended == [] and repo.created == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rejection_on_the_same_hash_beats_an_older_approval_on_that_hash():
    """Chronology still wins WITHIN a hash: approve h1, then reject h1."""
    repo = _EstimandRepo(
        history=[
            _rejected("r1", "h1", created_at="2026-09-05T00:00:00+00:00"),
            _approved("r0", "h1"),
        ]
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1", brand="B", treatment="T", outcome="Y", requester_id="q"
    )

    assert r.decision == ReviewGateDecision.REJECTED and r.review_id == "r1"
    assert repo.created == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_expired_approval_does_not_count_and_a_pending_row_wins_over_reopen():
    repo = _EstimandRepo(
        history=[
            _pending("r1", "h1", created_at="2026-09-05T00:00:00+00:00"),
            _approved("r0", "h1", valid_until="2020-01-01"),
        ]
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1", brand="B", treatment="T", outcome="Y", requester_id="q"
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW and r.review_id == "r1"
    assert repo.created == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_absent_brand_keys_on_the_empty_brand_bucket():
    """Mirrors migration 140's COALESCE(brand, '') -- a brandless run has its
    own estimand, never a sibling of some other brand's."""
    repo = _EstimandRepo(history=[])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1", brand=None, treatment="T", outcome="Y", requester_id="q"
    )

    assert repo.created[0]["estimand_key"] == ":t:y"
    assert repo.estimand_lookups == [":t:y"]
