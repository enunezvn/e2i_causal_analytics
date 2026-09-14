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

from src.causal_engine.dag_hash import compute_adjustment_set_hash
from src.causal_engine.expert_review_gate import (
    ExpertReviewGate,
    ReviewGateDecision,
    _VersionMatch,
)
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

    def __init__(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        latest_version: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.history = list(history or [])
        #: What ``get_latest_version`` answers -- None means "this review has no
        #: timeline yet", which is a fresh insert (or a pre-141 row).
        self.latest_version = latest_version
        self.latest_version_reads: List[str] = []
        self.created: List[Dict[str, Any]] = []
        self.appended: List[tuple] = []
        self.append_kwargs: List[Dict[str, Any]] = []
        self.advanced: List[tuple] = []
        self.advance_kwargs: List[Dict[str, Any]] = []
        self.advance_result = True
        self.recorded: List[tuple] = []
        self.record_kwargs: List[Dict[str, Any]] = []
        self.record_result = True
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

    async def get_latest_version(self, review_id: str) -> Optional[Dict[str, Any]]:
        self.latest_version_reads.append(review_id)
        return dict(self.latest_version) if self.latest_version is not None else None

    async def append_version(self, review_id: str, **kwargs: Any) -> bool:
        self.appended.append((review_id, kwargs.get("dag_version_hash")))
        self.append_kwargs.append({"review_id": review_id, **kwargs})
        return True

    async def record_version(self, review_id: str, **kwargs: Any) -> bool:
        """The timeline write WITHOUT the row advance. Recorded separately so a
        test can tell "the timeline grew" from "the review moved" -- a double
        that folded them could not see that a record-only run leaves the review's
        cached assessment alone."""
        self.recorded.append((review_id, kwargs.get("dag_version_hash")))
        self.record_kwargs.append({"review_id": review_id, **kwargs})
        return self.record_result

    async def advance_review(self, review_id: str, **kwargs: Any) -> bool:
        """The advance WITHOUT an append (codex round 2). Recorded separately
        from ``append_version`` on purpose: "the review moved" and "the timeline
        grew" are the two decisions the gate now makes independently, and a
        double that folded them together could not tell a repair from a
        duplicate row."""
        self.advanced.append((review_id, kwargs.get("dag_version_hash")))
        self.advance_kwargs.append({"review_id": review_id, **kwargs})
        return self.advance_result

    async def update_dag_structure(
        self, review_id: str, dag_structure, related_validation_ids=None
    ) -> bool:
        self.structure_updates.append((review_id, dag_structure, related_validation_ids))
        return True


def _pending(
    review_id: str,
    dag_hash: str,
    created_at: str = "2026-09-01T00:00:00+00:00",
    adjustment_set_hash: Optional[str] = None,
):
    """A pending review row. ``adjustment_set_hash`` is the row's OWN half of its
    current version identity (migration 142); None -- the default -- is the
    honest pre-142 shape, an UNKNOWN adjustment set."""
    return {
        "review_id": review_id,
        "approval_status": "pending",
        "dag_version_hash": dag_hash,
        "adjustment_set_hash": adjustment_set_hash,
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
    # The timeline is seeded with this review's own pair on purpose. An EMPTY
    # fixture now means NOT_RECORDED, a different case: the gate records version
    # 1 there. What this pins is same-structure idempotence on RE-ENCOUNTER --
    # already recorded, so there is nothing left to write.
    repo = _EstimandRepo(
        history=[_pending("r1", "h1")],
        latest_version=_version("h1", snapshot=None),
    )
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
    # r2's own pair is seeded on its timeline: an EMPTY fixture would mean
    # NOT_RECORDED, where the gate records version 1 -- a different case from the
    # one this test is about (an approval surviving a revised sibling).
    repo = _EstimandRepo(
        history=[
            _pending("r2", "h2", created_at="2026-09-05T00:00:00+00:00"),
            _approved("r1", "h1"),
        ],
        latest_version=_version("h2", snapshot=None),
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


# --------------------------------------------------------------------------
# H2 / H4 (codex round-1): what "changed" means is the (hash, adjustment) PAIR
# --------------------------------------------------------------------------


def _version(
    dag_hash: str, adjustment_sets=None, *, adjustment_set_hash="__derive__", snapshot="__default__"
):
    """A timeline row as ``get_latest_version`` returns it.

    ``snapshot=None`` is a migration-141 BACKFILLED row for a pre-097 review: the
    backfill copies ``expert_reviews.dag_structure_json``, which was NULL there,
    so both the snapshot and the adjustment hash are NULL. That combination is
    the only way to seed a recorded pair whose adjustment half is genuinely
    UNKNOWN -- with a snapshot present the gate DERIVES the hash from it, and an
    absent ``adjustment_sets`` key derives the canonical EMPTY set, a different
    fact.
    """
    if snapshot is None:
        return {
            "version_id": "v1",
            "review_id": "r1",
            "dag_version_hash": dag_hash,
            "adjustment_set_hash": None,
            "dag_structure_json": None,
        }
    snapshot = {"nodes": ["T", "Y"], "edges": [["T", "Y"]]}
    if adjustment_sets is not None:
        snapshot["adjustment_sets"] = adjustment_sets
    row = {
        "version_id": "v1",
        "review_id": "r1",
        "dag_version_hash": dag_hash,
        "dag_structure_json": snapshot,
    }
    if adjustment_set_hash == "__derive__":
        row["adjustment_set_hash"] = (
            compute_adjustment_set_hash(list(adjustment_sets)) if adjustment_sets else None
        )
    else:
        row["adjustment_set_hash"] = adjustment_set_hash
    return row


@pytest.mark.unit
@pytest.mark.asyncio
async def test_recovered_mint_does_not_append_the_winners_pair_twice():
    """Two concurrent mints of the SAME structure both recover the winner's
    review from the 23505. The winner already recorded (hash, adjustment set);
    the loser must add nothing -- a second identical row is not a version, it is
    a duplicate that makes the timeline lie about how often the DAG changed."""
    repo = _EstimandRepo(history=[], latest_version=_version("h1", [["W"]]))
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["W"]]},
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW
    assert repo.appended == []
    assert repo.latest_version_reads  # the check was actually made


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fresh_mint_still_records_version_one_and_pins_its_own_hash():
    """No timeline yet -> version 1, with the compare-and-set naming the hash
    ``create_review`` just stored."""
    repo = _EstimandRepo(history=[], latest_version=None)
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure=_GRAPH,
    )

    assert repo.appended == [("rev-new", "h1")]
    assert repo.append_kwargs[0]["expected_current_hash"] == "h1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adjustment_set_change_on_the_same_dag_appends_a_version():
    """H4: ``compute_dag_hash`` EXCLUDES adjustment sets, so a covariate change
    leaves the hash equal. Comparing only the hash kept that change off the
    timeline entirely -- the reviewer never saw the estimand's covariates move."""
    repo = _EstimandRepo(history=[_pending("r1", "h1")], latest_version=_version("h1", [["W"]]))
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["W", "X"]]},
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW and r.review_id == "r1"
    assert repo.appended == [("r1", "h1")]
    kwargs = repo.append_kwargs[0]
    assert kwargs["adjustment_set_hash"] == compute_adjustment_set_hash([["W", "X"]])
    # the snapshot must advance too, or the review renders the old covariates
    assert kwargs["dag_structure"]["adjustment_sets"] == [["W", "X"]]
    assert kwargs["expected_current_hash"] == "h1"
    assert "structure updated to a new version" in r.message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_identical_pair_on_a_pending_review_appends_nothing():
    repo = _EstimandRepo(history=[_pending("r1", "h1")], latest_version=_version("h1", [["W"]]))
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["W"]]},
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW
    assert repo.appended == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_backfilled_version_without_an_adjustment_hash_derives_it_from_the_snapshot():
    """Migration 141's backfill wrote ``adjustment_set_hash = NULL``. Reading
    that as "no adjustment set" would make every first re-encounter of an
    unchanged DAG look like a covariate change and append a spurious version."""
    repo = _EstimandRepo(
        history=[_pending("r1", "h1")],
        latest_version=_version("h1", [["W"]], adjustment_set_hash=None),
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["W"]]},
    )

    assert repo.appended == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_backfilled_version_derives_a_DIFFERENT_set_and_still_appends():
    """Positive control for the derivation: the same NULL-hash row whose
    snapshot carries OTHER covariates must not be read as unchanged."""
    repo = _EstimandRepo(
        history=[_pending("r1", "h1")],
        latest_version=_version("h1", [["W"]], adjustment_set_hash=None),
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["Z"]]},
    )

    assert repo.appended == [("r1", "h1")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_new_hash_pins_the_advance_to_the_hash_the_gate_read():
    """The compare-and-set the repository applies: of two interleaved appends,
    the one whose read is stale must not drag the review backwards."""
    repo = _EstimandRepo(history=[_pending("r1", "h1")], latest_version=_version("h1"))
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h2",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure=_GRAPH,
    )

    assert repo.appended == [("r1", "h2")]
    assert repo.append_kwargs[0]["expected_current_hash"] == "h1"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_unreadable_timeline_appends_nothing_and_still_answers(caplog):
    """A failed version read must not become a duplicate append.

    ``get_latest_version`` re-raises a client error (R1/R3) because None reads as
    "nothing recorded, append". The gate answers that with the conservative
    choice: append nothing and stay available, whereas appending on a comparison
    that never happened writes the duplicate row this check exists to prevent.

    The row here asserts NO change of its own: its hash matches the run's and its
    adjustment half is NULL, which is UNKNOWN rather than evidence. So there is
    nothing to fall back on and the append is skipped. The ADVANCE is a separate
    decision (codex round 2) and still fires -- the row learns its adjustment
    half, which is what every guard keyed on it needs -- and the next run
    re-reads the timeline and appends if it really was missing.
    """

    class _BlindRepo(_EstimandRepo):
        async def get_latest_version(self, review_id: str):
            raise RuntimeError("connection refused")

    repo = _BlindRepo(history=[_pending("r1", "h1")])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    with caplog.at_level("WARNING"):
        r = await gate.check_approval(
            dag_hash="h1",
            brand="B",
            treatment="T",
            outcome="Y",
            requester_id="q",
            dag_structure={**_GRAPH, "adjustment_sets": [["W"]]},
        )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW and r.review_id == "r1"
    assert repo.appended == []
    assert any("latest structure version" in rec.getMessage() for rec in caplog.records)
    # The advance is NOT suppressed by the read failure: it reads the review row,
    # which the outage does not hide.
    assert repo.advanced == [("r1", "h1")]
    assert repo.advance_kwargs[0]["adjustment_set_hash"] == compute_adjustment_set_hash([["W"]])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_unreadable_timeline_still_advances_a_review_on_another_hash(caplog):
    """Positive control: the review's OWN hash is read from the history, not from
    the timeline, so a genuinely new structure is still recorded when the version
    read fails -- the conservative skip covers only the pair comparison."""

    class _BlindRepo(_EstimandRepo):
        async def get_latest_version(self, review_id: str):
            raise RuntimeError("connection refused")

    repo = _BlindRepo(history=[_pending("r1", "h1")])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    with caplog.at_level("WARNING"):
        await gate.check_approval(
            dag_hash="h2",
            brand="B",
            treatment="T",
            outcome="Y",
            requester_id="q",
            dag_structure=_GRAPH,
        )

    assert repo.appended == [("r1", "h2")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_mint_records_version_one_even_when_the_timeline_is_unreadable(caplog):
    """On the MINT path the conservative skip is NOT self-repairing.

    The pending branch repairs itself because the review's OWN hash carries the
    change: the next run sees hash_changed and appends. A fresh mint has no such
    signal -- ``create_review`` just stored this very hash, so every later run of
    the same structure computes hash_changed=False and finds nothing recorded,
    and skips again. Version 1 would never be written; the first row the timeline
    ever got would be a LATER structure, which the diff renders as the origin
    (changes=None) instead of as the change it was.

    So an unreadable read is its own state: the mint appends. A duplicate row
    during a store outage is the lesser harm -- it is one extra row in a
    timeline, against an estimand whose first structure is missing for good.
    """

    class _BlindRepo(_EstimandRepo):
        async def get_latest_version(self, review_id: str):
            raise RuntimeError("connection refused")

    repo = _BlindRepo(history=[])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    with caplog.at_level("WARNING"):
        r = await gate.check_approval(
            dag_hash="h1",
            brand="B",
            treatment="T",
            outcome="Y",
            requester_id="q",
            dag_structure=_GRAPH,
        )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW and r.review_id == "rev-new"
    assert repo.created and repo.appended == [("rev-new", "h1")]
    assert repo.append_kwargs[0]["expected_current_hash"] == "h1"


# --------------------------------------------------------------------------
# M5 -- check_rejection reads the SAME rows as check_approval
# --------------------------------------------------------------------------


class _RecordingRepo(_EstimandRepo):
    """Also records what the legacy hash-keyed reader was asked for."""

    def __init__(self, history=None) -> None:
        super().__init__(history=history)
        self.dag_reads: List[tuple] = []

    async def get_reviews_for_dag(
        self, dag_hash: str, include_expired: bool = False, brand: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self.dag_reads.append((dag_hash, include_expired, brand))
        rows = [r for r in self.history if r.get("dag_version_hash") == dag_hash]
        return [r for r in rows if brand is None or r.get("brand") == brand]


def _rejected_row(review_id: str, dag_hash: str, brand: str, *, treatment="T", outcome="Y"):
    return {
        "review_id": review_id,
        "approval_status": "rejected",
        "dag_version_hash": dag_hash,
        "brand": brand,
        "treatment_variable": treatment,
        "outcome_variable": outcome,
        "estimand_key": estimand_key_for(brand, treatment, outcome),
        "created_at": "2026-09-01T00:00:00+00:00",
        "concerns_raised": ["formulary_status is a collider"],
        "reviewer_name": "Dr. No",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rejection_probe_is_case_insensitive_on_the_brand():
    """The probe missed rejections purely on brand CASE: rows are stored with the
    brand as the caller wrote it ("Remibrutinib") and probed with whatever the
    run carries ("remibrutinib"), and ``get_reviews_for_dag`` filters with an
    exact-case ``.eq``. The estimand key lowercases every operand, which is what
    ``check_approval`` already reads -- so the two readers now cannot disagree."""
    repo = _RecordingRepo(history=[_rejected_row("r0", "h1", "Remibrutinib")])
    gate = ExpertReviewGate(repository=repo)

    result = await gate.check_rejection("h1", brand="remibrutinib", treatment="T", outcome="Y")

    assert result is not None and result.decision == ReviewGateDecision.REJECTED
    assert result.review_id == "r0"
    assert repo.estimand_lookups == ["remibrutinib:t:y"]
    assert repo.dag_reads == []  # the hash-keyed reader is no longer consulted


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rejection_probe_without_a_brand_does_not_read_another_brands_rejection():
    """With brand None the old probe applied NO brand filter at all, so another
    brand's rejection of a structurally identical DAG halted this run. The empty
    brand is its own estimand bucket (``:t:y``)."""
    repo = _RecordingRepo(history=[_rejected_row("r0", "h1", "Kisqali")])
    gate = ExpertReviewGate(repository=repo)

    assert await gate.check_rejection("h1", treatment="T", outcome="Y") is None
    assert repo.estimand_lookups == [":t:y"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rejection_probe_matches_the_empty_brand_bucket_it_reads():
    """Positive control for the bucket: a rejection recorded WITHOUT a brand is
    found by a brand-less probe."""
    repo = _RecordingRepo(history=[_rejected_row("r0", "h1", "")])
    gate = ExpertReviewGate(repository=repo)

    result = await gate.check_rejection("h1", treatment="T", outcome="Y")
    assert result is not None and result.review_id == "r0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rejection_probe_only_counts_the_hash_under_analysis():
    """A rejection of ANOTHER structure of the same estimand does not halt this
    one (spec §7) -- the estimand read is filtered to this hash, exactly the
    ``same_hash_history`` slice ``check_approval`` ranks."""
    repo = _RecordingRepo(history=[_rejected_row("r0", "hOTHER", "B")])
    gate = ExpertReviewGate(repository=repo)

    assert await gate.check_rejection("h1", brand="B", treatment="T", outcome="Y") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rejection_probe_without_treatment_or_outcome_keeps_the_legacy_read():
    """Callers that have no estimand in hand (``can_proceed``-style, older tests)
    must keep working: with neither variable the hash-keyed reader is used, with
    the brand it was given."""
    repo = _RecordingRepo(history=[_rejected_row("r0", "h1", "B")])
    gate = ExpertReviewGate(repository=repo)

    result = await gate.check_rejection("h1", brand="B")
    assert result is not None and result.review_id == "r0"
    assert repo.dag_reads == [("h1", True, "B")]
    assert repo.estimand_lookups == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rejection_probe_takes_the_estimand_path_with_only_one_variable():
    """One variable is still an estimand (the key COALESCEs the missing operand),
    and the node always has both -- a partial call must not silently fall back to
    the brand-blind legacy read."""
    repo = _RecordingRepo(history=[_rejected_row("r0", "h1", "B", outcome="")])
    gate = ExpertReviewGate(repository=repo)

    result = await gate.check_rejection("h1", brand="B", treatment="T")
    assert result is not None and result.review_id == "r0"
    assert repo.estimand_lookups == ["b:t:"]
    assert repo.dag_reads == []


# --------------------------------------------------------------------------
# Codex round 2 -- APPEND and ADVANCE are two decisions, not one
#
# The pending branch used to ask one question ("hash_changed OR pair_differs")
# and answer it with one write. That conflated "the timeline is missing this
# structure" with "the review is not on this structure", and the two can differ:
# a lost compare-and-set leaves a review BEHIND its own timeline, with nothing
# new to append. The gate now decides them separately.
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_review_behind_its_own_timeline_is_advanced_without_appending():
    """Codex round-2 finding 3, the repair.

    Two adjustment-only advances raced from (h1, A): both appended, C's
    compare-and-set landed first, B's lost. The review is on C while the
    timeline's latest row is B. A later run of B must move the review to B --
    and must NOT append, because the timeline already holds (h1, B). Under the
    old single decision this run saw its own pair already recorded and skipped
    forever: the permanent strand.
    """
    run_pair_adjustment = compute_adjustment_set_hash([["B"]])
    repo = _EstimandRepo(
        # The review lost the race and sits on C; the timeline's latest is B,
        # which is what THIS run produces -- so its pair is already recorded.
        history=[_pending("r1", "h1", adjustment_set_hash="adj-C")],
        latest_version=_version("h1", adjustment_set_hash=run_pair_adjustment),
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["B"]]},
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW and r.review_id == "r1"
    # Nothing appended: the timeline is not missing anything this run produced.
    assert repo.appended == []
    # But the review moved, and the compare-and-set named the pair it READ.
    assert repo.advanced == [("r1", "h1")]
    kwargs = repo.advance_kwargs[0]
    assert kwargs["expected_current_hash"] == "h1"
    assert kwargs["expected_current_adjustment_hash"] == "adj-C"
    assert "structure updated to a new version" in r.message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_structure_the_timeline_lacks_is_appended_not_separately_advanced():
    """The other side of the same interleaving: this run's pair is NOT the
    timeline's latest, so it appends -- and the append carries the advance, so
    no second write is made. One structure, one row, one move."""
    run_adjustment = compute_adjustment_set_hash([["C"]])
    repo = _EstimandRepo(
        history=[_pending("r1", "h1", adjustment_set_hash="adj-C")],
        latest_version=_version("h1", adjustment_set_hash="adj-B"),
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    r = await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["C"]]},
    )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW
    assert repo.appended == [("r1", "h1")]
    assert repo.advanced == [], "the append already advances; a second write would be a duplicate"
    kwargs = repo.append_kwargs[0]
    assert kwargs["adjustment_set_hash"] == run_adjustment
    assert kwargs["expected_current_hash"] == "h1"
    assert kwargs["expected_current_adjustment_hash"] == "adj-C"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_row_with_an_unknown_adjustment_set_learns_it_without_appending():
    """A pre-142 row (or one the old image minted) carries NULL. When the
    timeline already holds this run's pair, there is nothing to append -- but the
    row must still LEARN its adjustment half, or every guard keyed on it keeps
    matching half an identity, which is the whole defect."""
    adjustment = compute_adjustment_set_hash([["W"]])
    repo = _EstimandRepo(
        history=[_pending("r1", "h1", adjustment_set_hash=None)],
        latest_version=_version("h1", adjustment_set_hash=adjustment),
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["W"]]},
    )

    assert repo.appended == []
    assert repo.advanced == [("r1", "h1")]
    kwargs = repo.advance_kwargs[0]
    assert kwargs["adjustment_set_hash"] == adjustment
    # The compare-and-set expects the UNKNOWN it read -- None, matched IS NULL.
    assert kwargs["expected_current_adjustment_hash"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_run_without_a_structure_on_an_unchanged_hash_writes_nothing():
    """FLIPPED (codex round 5, finding 1). T9 read this run as the honest
    direction of the rule above: no structure in scope means no adjustment set,
    which "differs from" the row's stored hash, so the row was advanced to
    unknown.

    That conflated two different things. UNKNOWN is not a different VALUE; it is
    the ABSENCE of a claim. A run that carried no structure has not discovered
    that the covariates changed -- it has discovered nothing about them. Letting
    it advance overwrote a KNOWN covariate set with unknown, and (codex's
    sequence, pinned below) recorded information-free version rows that the
    detail route then named, putting "the structure disappeared" beside the
    reviewer's real graph.

    So when the run knows no structure AND its DAG hash is the row's, it asserts
    nothing and writes nothing -- no append, no record, no advance. The review is
    still PENDING on the structure it already had. A CHANGED hash is a different
    case and still writes; see the pin two below.
    """
    repo = _EstimandRepo(
        history=[_pending("r1", "h1", adjustment_set_hash="adj-W")],
        latest_version={
            "version_id": "v1",
            "review_id": "r1",
            "dag_version_hash": "h1",
            # Both NULL: genuinely unknown. A snapshot here would be DERIVED from
            # instead, giving the canonical EMPTY hash -- a different fact.
            "adjustment_set_hash": None,
            "dag_structure_json": None,
        },
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    result = await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure=None,
    )

    assert repo.appended == [] and repo.recorded == [] and repo.advanced == []
    assert repo.created == [], "the open review is still the estimand's review"
    # Still the pending answer the caller needs, with the review id.
    assert result.decision is ReviewGateDecision.PENDING_REVIEW
    assert result.review_id == "r1"
    assert result.message == "DAG review pending expert approval"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_structureless_run_does_not_record_a_null_version_over_a_backfill():
    """Codex round 5's exact sequence, at the gate.

    Post-141/142 a pending review carries ``(h, NULL, snapshot A)`` and its
    backfilled version row carries the same. A REVIEW run supplies ``h`` WITHOUT
    structure. The gate DERIVES A's adjustment hash from the backfilled row, so
    the pair comparison called it DIFFERENT and appended -- while the review
    row's own pair already equalled the run's ``(h, None)``, so nothing advanced
    and ``record_version`` wrote ``(h, NULL, NULL snapshot)`` onto the timeline.
    The selector's exact NULL-pair match then named that row.

    The run knew no structure and the hash did not move: it asserts nothing, so
    the timeline keeps the backfill and the review keeps snapshot A.
    """
    snapshot_a = {"nodes": ["T", "Y", "W"], "edges": [["T", "Y"]], "adjustment_sets": [["W"]]}
    row = {**_pending("r1", "h1", adjustment_set_hash=None), "dag_structure_json": snapshot_a}
    repo = _EstimandRepo(
        history=[row],
        latest_version={
            "version_id": "v1",
            "review_id": "r1",
            "dag_version_hash": "h1",
            # Migration 141 backfilled the hash half as NULL but COPIED the
            # snapshot, which is why the gate can derive A from it at all.
            "adjustment_set_hash": None,
            "dag_structure_json": snapshot_a,
        },
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure=None,
    )

    assert repo.recorded == [], "an information-free version row is not a fact"
    assert repo.appended == [] and repo.advanced == []
    assert repo.structure_updates == [], "the row keeps snapshot A; nothing cleared it"
    assert row["dag_structure_json"] == snapshot_a


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_structureless_run_on_a_NEW_hash_still_appends_and_advances():
    """The other side of the reversal, unchanged. A new DAG hash with unknown
    structure is a real -- if incomplete -- fact: the run positively discovered
    that the structure moved, even though it cannot say to which covariates. The
    row and the appended version agree on ``(h2, NULL, NULL)``, so nothing is
    rendered against a snapshot it does not have."""
    repo = _EstimandRepo(
        history=[_pending("r1", "h1", adjustment_set_hash="adj-W")],
        latest_version=_version("h1", [["W"]], adjustment_set_hash="adj-W"),
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h2",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure=None,
    )

    assert repo.appended == [("r1", "h2")]
    kwargs = repo.append_kwargs[0]
    assert kwargs["adjustment_set_hash"] is None
    assert kwargs["dag_structure"] is None
    assert kwargs["expected_current_hash"] == "h1"
    assert kwargs["expected_current_adjustment_hash"] == "adj-W"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_identical_pair_on_a_row_that_already_knows_it_writes_nothing():
    """The no-op, now that the row can hold the whole identity: timeline latest
    and review row both equal the run's pair, so neither decision fires."""
    adjustment = compute_adjustment_set_hash([["W"]])
    repo = _EstimandRepo(
        history=[_pending("r1", "h1", adjustment_set_hash=adjustment)],
        latest_version=_version("h1", adjustment_set_hash=adjustment),
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["W"]]},
    )

    assert repo.appended == [] and repo.advanced == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_failed_advance_is_logged_and_still_returns_the_review(caplog):
    """Best-effort, like the append beside it: a lost repair race must not
    withhold the review id the caller needs."""
    repo = _EstimandRepo(
        history=[_pending("r1", "h1", adjustment_set_hash="adj-C")],
        latest_version=_version("h1", adjustment_set_hash=compute_adjustment_set_hash([["B"]])),
    )
    repo.advance_result = False
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    with caplog.at_level("WARNING"):
        r = await gate.check_approval(
            dag_hash="h1",
            brand="B",
            treatment="T",
            outcome="Y",
            requester_id="q",
            dag_structure={**_GRAPH, "adjustment_sets": [["B"]]},
        )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW and r.review_id == "r1"
    assert any("r1" in rec.getMessage() for rec in caplog.records)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_mint_stores_the_adjustment_half_on_the_new_row():
    """The minted row carries its WHOLE identity from the start, so the
    version-1 append's compare-and-set has a pair to expect -- and every guard
    keyed on the row works from its first moment, not from its first advance."""
    repo = _EstimandRepo(history=[], latest_version=None)
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["W"]]},
    )

    adjustment = compute_adjustment_set_hash([["W"]])
    assert repo.created[0]["adjustment_set_hash"] == adjustment
    kwargs = repo.append_kwargs[0]
    assert kwargs["expected_current_hash"] == "h1"
    assert kwargs["expected_current_adjustment_hash"] == adjustment


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_mint_without_a_structure_stores_an_unknown_adjustment_half():
    repo = _EstimandRepo(history=[], latest_version=None)
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure=None,
    )

    assert repo.created[0]["adjustment_set_hash"] is None
    assert repo.append_kwargs[0]["expected_current_adjustment_hash"] is None


# --------------------------------------------------------------------------
# UNKNOWN is not an empty timeline
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_unreadable_timeline_is_still_NOT_treated_as_an_empty_one():
    """The other half of the distinction, so the fix cannot be over-applied.

    UNKNOWN keeps its conservatism: there may be a version row this run cannot
    see, so an append on no row-side evidence would write the duplicate the read
    exists to prevent. Only NOT_RECORDED -- a POSITIVE empty timeline -- appends
    without evidence.
    """

    class _BlindRepo(_EstimandRepo):
        async def get_latest_version(self, review_id: str):
            raise RuntimeError("connection refused")

    adjustment = compute_adjustment_set_hash([["W"]])
    repo = _BlindRepo(history=[_pending("r1", "h1", adjustment_set_hash=adjustment)])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["W"]]},
    )

    assert repo.appended == [] and repo.advanced == []


# --------------------------------------------------------------------------
# NOT_RECORDED is an EMPTY timeline, not an unreadable one
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_row_that_only_learns_its_adjustment_half_still_records_version_one():
    """The gap: no timeline, and the ONLY difference is the row learning its
    adjustment half (the DAG hash matches, the row's half is NULL).

    ``row_asserts_change`` is False -- a NULL half is UNKNOWN, not evidence -- so
    the append was skipped and the advance made the row equal the run's pair.
    Every later run then found nothing to do, and that review never got a version
    row at all: invisible in the timeline forever, with no outage and no race.
    """
    adjustment = compute_adjustment_set_hash([["W"]])
    repo = _EstimandRepo(
        history=[_pending("r1", "h1", adjustment_set_hash=None)],
        latest_version=None,
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["W"]]},
    )

    assert repo.appended == [("r1", "h1")]
    kwargs = repo.append_kwargs[0]
    assert kwargs["adjustment_set_hash"] == adjustment
    # The compare-and-set names the pair it READ -- the unknown half included.
    assert kwargs["expected_current_hash"] == "h1"
    assert kwargs["expected_current_adjustment_hash"] is None
    # append_version advances too, so no second write.
    assert repo.advanced == []


# --------------------------------------------------------------------------
# THREE branches: record-only, record-and-advance, advance-only
#
# "The timeline is missing this structure" and "the review is not on it" are
# independent, so three combinations write, and WHICH one runs matters because
# they touch different rows. Folding record-only into append_version re-wrote
# the review with the pair it already had, clearing an advisory grading OF THAT
# VERY STRUCTURE for nothing.
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_unchanged_run_on_an_empty_timeline_records_without_advancing():
    """NOT_RECORDED + the row already on this pair: the timeline needs version 1,
    the review needs nothing.

    Live, this is an old-image mint from the deploy window, or a mint whose
    version-1 insert failed. Both self-repair here -- and once the row lands the
    timeline answers SAME, so it happens exactly once. That is a repair, not a
    duplicate.
    """
    adjustment = compute_adjustment_set_hash([["W"]])
    repo = _EstimandRepo(
        history=[_pending("r1", "h1", adjustment_set_hash=adjustment)],
        latest_version=None,
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["W"]]},
    )

    assert repo.recorded == [("r1", "h1")]
    kwargs = repo.record_kwargs[0]
    assert kwargs["adjustment_set_hash"] == adjustment
    # The review row is NOT rewritten, so its cached advisory grading of this
    # very structure survives -- the whole point of the record-only branch.
    assert repo.advanced == [] and repo.appended == []
    # record_version takes no compare-and-set: there is nothing to compare.
    assert "expected_current_hash" not in kwargs


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_orphan_latest_pair_records_only_when_the_row_is_already_correct():
    """The lost-race leftover from the other side: the timeline's latest is an
    orphan B while the review and this run are both on C. C is missing from the
    timeline, so record it -- but the review is already right, so leave it."""
    run_adjustment = compute_adjustment_set_hash([["C"]])
    repo = _EstimandRepo(
        history=[_pending("r1", "h1", adjustment_set_hash=run_adjustment)],
        latest_version=_version("h1", adjustment_set_hash="adj-B-orphan"),
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["C"]]},
    )

    assert repo.recorded == [("r1", "h1")]
    assert repo.record_kwargs[0]["adjustment_set_hash"] == run_adjustment
    assert repo.advanced == [] and repo.appended == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_failed_record_only_write_is_logged_and_still_returns_the_review(caplog):
    """Best-effort, like the other two writes: a failed timeline row must not
    withhold the review id the caller needs."""
    adjustment = compute_adjustment_set_hash([["W"]])
    repo = _EstimandRepo(
        history=[_pending("r1", "h1", adjustment_set_hash=adjustment)],
        latest_version=None,
    )
    repo.record_result = False
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    with caplog.at_level("WARNING"):
        r = await gate.check_approval(
            dag_hash="h1",
            brand="B",
            treatment="T",
            outcome="Y",
            requester_id="q",
            dag_structure={**_GRAPH, "adjustment_sets": [["W"]]},
        )

    assert r.decision == ReviewGateDecision.PENDING_REVIEW and r.review_id == "r1"
    assert any("r1" in rec.getMessage() for rec in caplog.records)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_latest_version_with_malformed_adjustment_data_does_not_break_the_gate():
    """Codex round 5, finding 2, at the gate's own reader.

    ``_match_last_recorded_version`` derives a NULL-adjustment version's hash
    from its snapshot, and ``{"adjustment_sets": [null]}`` -- a row migration
    141's outer-object CHECK admits -- used to raise TypeError out of
    ``check_approval`` entirely. The read's try/except covers the repository
    call, not the derivation after it.

    It proves nothing, so the recorded pair is (h1, UNKNOWN), which is not this
    run's pair: DIFFERENT, so the structure is appended and the review advanced.
    No exception anywhere."""
    repo = _EstimandRepo(
        history=[_pending("r1", "h1", adjustment_set_hash=None)],
        latest_version={
            "version_id": "v1",
            "review_id": "r1",
            "dag_version_hash": "h1",
            "adjustment_set_hash": None,
            "dag_structure_json": {"nodes": ["T", "Y"], "adjustment_sets": [None]},
        },
    )
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    result = await gate.check_approval(
        dag_hash="h1",
        brand="B",
        treatment="T",
        outcome="Y",
        requester_id="q",
        dag_structure={**_GRAPH, "adjustment_sets": [["W"]]},
    )

    assert result.decision is ReviewGateDecision.PENDING_REVIEW
    assert repo.appended == [("r1", "h1")]
    assert repo.append_kwargs[0]["adjustment_set_hash"] == compute_adjustment_set_hash([["W"]])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_version_match_is_unknown_without_a_repository(caplog):
    """A gate with no repository can read no timeline, so "is this structure
    already recorded" is UNKNOWN -- the same answer a failed read gets, and the
    one whose call sites are already written for it.

    ``check_approval`` returns UNAVAILABLE before it ever consults, so this is
    unreachable through the public path; the guard exists so the method's own
    contract holds for a direct caller (and so its type does).

    It must reach that answer from the GUARD, not by calling a method on None
    and letting the read's ``except`` catch the AttributeError. Both return
    UNKNOWN, so only the log tells them apart -- and the difference matters:
    that warning says the review STORE could not be read, which is an outage to
    investigate. "This gate was never given a repository" is a configuration
    fact the caller already knows, and logging it as a store failure would put
    a phantom outage in front of whoever reads the warnings.
    """
    gate = ExpertReviewGate(repository=None, auto_create_review=True)

    with caplog.at_level("WARNING"):
        match = await gate._match_last_recorded_version("r1", "h1", None)

    assert match is _VersionMatch.UNKNOWN
    assert not [r for r in caplog.records if "latest structure version" in r.getMessage()]
