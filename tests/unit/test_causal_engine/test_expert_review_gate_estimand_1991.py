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

    async def append_version(
        self,
        review_id: str,
        *,
        dag_version_hash: str,
        dag_structure: Optional[Dict[str, Any]] = None,
        adjustment_set_hash: Optional[str] = None,
        query_id: Optional[str] = None,
    ) -> bool:
        self.appended.append((review_id, dag_version_hash))
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
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW and r.review_id == "r1"
    assert repo.created == [] and repo.appended == [("r1", "h2")]


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
    repo = _EstimandRepo(
        history=[
            {
                "review_id": "r0",
                "approval_status": "rejected",
                "dag_version_hash": "h1",
                "estimand_key": "b:t:y",
                "created_at": "2026-09-01T00:00:00+00:00",
            }
        ]
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1", brand="B", treatment="T", outcome="Y", requester_id="q"
    )

    assert r.decision == ReviewGateDecision.REJECTED and repo.created == []


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
