"""#1991 debt 3 -- a BLOCK-band run never mints an expert review.

A BLOCK band is terminal for a STATISTICAL reason: the estimate failed its
robustness suite and is served as ``failed`` whatever a reviewer says about the
DAG. Queuing it asked a human to adjudicate a structure whose verdict could not
change the outcome, and (since migration 140 keys the queue on the estimand)
those rows compete for the ONE pending slot an estimand has -- a BLOCK run
could evict, or be evicted by, the REVIEW-band review a human can actually act
on. So the BLOCK branch no longer consults the gate and queues nothing.

What it still does: surface a rejection the read-only probe ALREADY observed.
That verdict is durable (#1970) and costs no write, so a BLOCK run on a
structure a human rejected still says so.

No DoWhy: the runner and the reconstruction are stand-ins (same device as
test_refutation_expert_review_enforcement_1971.py).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.agents.causal_impact.nodes.refutation as refutation_mod
from src.agents.causal_impact.nodes.refutation import RefutationNode
from src.causal_engine.expert_review_gate import ExpertReviewGate
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)

_QUERY_ID = "q-1991abc00001"
_PATH_ID = "cp_real_1991_00001"
_DAG_HASH = "c0ffee1991deadbeef"


def _suite(band: GateDecision) -> RefutationSuite:
    confidence = {GateDecision.PROCEED: 0.9, GateDecision.REVIEW: 0.6, GateDecision.BLOCK: 0.3}
    return RefutationSuite(
        passed=band != GateDecision.BLOCK,
        confidence_score=confidence[band],
        tests=[
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.5,
                refuted_effect=0.01,
            )
        ],
        gate_decision=band,
        treatment_variable="rep_visits",
        outcome_variable="trx",
        brand="Kisqali",
    )


class _Runner:
    def __init__(self, band: GateDecision) -> None:
        self.band = band

    def run_all_tests(self, **kwargs: Any) -> RefutationSuite:
        return _suite(self.band)


class _ReviewRepo:
    """ExpertReviewRepository stand-in recording every write (no live DB)."""

    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        self.rows = list(rows or [])
        self.create_calls: List[Dict[str, Any]] = []
        self.appended: List[tuple] = []

    async def get_reviews_for_dag(
        self, dag_hash: str, include_expired: bool = False, brand: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return list(self.rows)

    async def get_reviews_for_estimand(
        self, estimand_key: str, include_expired: bool = True
    ) -> List[Dict[str, Any]]:
        return list(self.rows)

    async def get_dag_approval(self, dag_hash: str, brand: Optional[str] = None):
        return None

    async def create_review(self, **kwargs: Any) -> str:
        self.create_calls.append(kwargs)
        return "rev-created"

    async def append_version(self, review_id: str, **kwargs: Any) -> bool:
        self.appended.append((review_id, kwargs.get("dag_version_hash")))
        return True

    async def get_latest_version(self, review_id: str) -> Optional[Dict[str, Any]]:
        """No timeline yet -- the gate reads this before appending and treats an
        UNREADABLE answer as "already recorded", so a missing method here would
        make every "nothing was appended" assertion pass vacuously."""
        return None

    async def update_dag_structure(self, *args: Any, **kwargs: Any) -> bool:
        return True


def _rejected_row() -> Dict[str, Any]:
    return {
        "review_id": "rev-rejected",
        "approval_status": "rejected",
        "dag_version_hash": _DAG_HASH,
        "reviewer_name": "Dr. No",
        "concerns_raised": ["formulary_status is a collider, not a confounder"],
    }


class _PathRepo:
    def __init__(self) -> None:
        self.status_calls: List[Dict[str, Any]] = []

    async def get_path_row(self, path_id: str) -> Optional[Dict[str, Any]]:
        if path_id != _PATH_ID:
            return None
        return {
            "path_id": _PATH_ID,
            "start_node": "rep_visits",
            "end_node": "trx",
            "brand": "Kisqali",
            "validation_status": "pending",
            "is_synthetic": False,
        }

    async def find_real_paths_for_pair(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return []

    async def set_validation_status(
        self, path_id: str, new_status: str, allowed_current, **kwargs: Any
    ) -> bool:
        self.status_calls.append({"path_id": path_id, "new_status": new_status})
        return True


def _validation_repo() -> MagicMock:
    repo = MagicMock()
    repo.save_suite = AsyncMock(return_value=["v-1"])
    return repo


def _state(**overrides: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "query": "effect of rep_visits on trx",
        "query_id": _QUERY_ID,
        "treatment_var": "rep_visits",
        "outcome_var": "trx",
        "brand": "Kisqali",
        "confounders": [],
        "dag_version_hash": _DAG_HASH,
        "causal_path_id": _PATH_ID,
        "data_source": "kpi_substrate:WS3-BI-009",
        "causal_graph": {
            "nodes": ["rep_visits", "trx"],
            "edges": [("rep_visits", "trx")],
            "treatment_nodes": ["rep_visits"],
            "outcome_nodes": ["trx"],
        },
        "estimation_result": {
            # #2155: the estimation node always records the columns it fitted on.
            "covariates_adjusted": [],
            "ate": 0.5,
            "ate_ci_lower": 0.3,
            "ate_ci_upper": 0.7,
            "method": "linear_regression",
            "selected_estimator": "ols",
            "statistical_significance": True,
        },
        "status": "in_progress",
    }
    state.update(overrides)
    return state


def _node(
    monkeypatch: pytest.MonkeyPatch,
    band: GateDecision,
    gate: Any,
    *,
    path_repo: Optional[_PathRepo] = None,
) -> RefutationNode:
    node = RefutationNode(
        validation_repo=_validation_repo(),
        expert_review_gate=gate,
        causal_path_repo=path_repo if path_repo is not None else _PathRepo(),
    )
    node.runner = _Runner(band)
    monkeypatch.setattr(
        refutation_mod,
        "_reconstruct_dowhy_artifacts",
        lambda **kwargs: (object(), object(), object()),
    )

    async def _no_signal(outcome: Any) -> None:
        return None

    monkeypatch.setattr(node, "_log_validation_outcome_signal", _no_signal)
    return node


@pytest.mark.unit
@pytest.mark.asyncio
async def test_block_band_never_consults_or_mints(monkeypatch):
    """A BLOCK run is terminal before the review is consulted (#1991 debt 3)."""
    repo = _ReviewRepo(rows=[])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)
    node = _node(monkeypatch, GateDecision.BLOCK, gate)

    consulted: List[int] = []

    async def _spy(*a: Any, **k: Any) -> Dict[str, Any]:
        consulted.append(1)
        return {}

    monkeypatch.setattr(node, "_consult_review_gate", _spy)

    result = await node.execute(_state())

    assert consulted == []
    assert result["status"] == "failed"
    assert result.get("expert_review_id") is None
    assert result.get("expert_review_decision") is None
    assert repo.create_calls == []
    assert repo.appended == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_block_band_without_the_spy_still_writes_no_row(monkeypatch):
    """Positive control for the spy above: the REAL gate is reachable from this
    node (a REVIEW band mints through it), so the empty create list on BLOCK is
    a decision, not a mis-wired fixture."""
    repo = _ReviewRepo(rows=[])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)

    blocked = await _node(monkeypatch, GateDecision.BLOCK, gate).execute(_state())
    assert blocked["status"] == "failed"
    assert repo.create_calls == []

    review = await _node(monkeypatch, GateDecision.REVIEW, gate).execute(_state())
    assert review["expert_review_decision"] == "pending_review"
    assert len(repo.create_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_block_band_still_surfaces_a_prior_rejection(monkeypatch):
    """The read-only probe's verdict is durable (#1970) and costs no write."""
    repo = _ReviewRepo(rows=[_rejected_row()])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)
    paths = _PathRepo()
    node = _node(monkeypatch, GateDecision.BLOCK, gate, path_repo=paths)

    result = await node.execute(_state())

    assert result["status"] == "failed"
    assert result["expert_review_decision"] == "rejected"
    assert result["expert_review_id"] == "rev-rejected"
    assert "REJECTED" in result["review_caveat"]
    assert repo.create_calls == []
    assert repo.appended == []
    assert paths.status_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_block_band_caveat_no_longer_promises_a_queue(monkeypatch):
    """The BLOCK band sentence claimed the estimate "has been routed to expert
    review for adjudication". Nothing is routed any more, so it must not."""
    repo = _ReviewRepo(rows=[_rejected_row()])
    gate = ExpertReviewGate(repository=repo, auto_create_review=True)
    node = _node(monkeypatch, GateDecision.BLOCK, gate)

    result = await node.execute(_state())
    caveat = result["review_caveat"]

    assert caveat, "positive control: a rejected BLOCK run always carries a caveat"
    assert "routed to expert review" not in caveat
    assert "is not queued for review" in caveat


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_rejection_probe_is_given_the_runs_estimand(monkeypatch):
    """M5 (codex round-1): the probe must read the rows ``check_approval`` reads.

    That read is keyed on the estimand, so the node has to hand over the
    treatment and outcome it already carries -- without them the probe falls back
    to the hash-keyed, exact-case-brand read that missed a rejection stored under
    "Remibrutinib" and probed as "remibrutinib".
    """
    calls: List[Dict[str, Any]] = []

    class _RecordingGate(ExpertReviewGate):
        async def check_rejection(self, dag_hash, brand=None, treatment=None, outcome=None):
            calls.append(
                {
                    "dag_hash": dag_hash,
                    "brand": brand,
                    "treatment": treatment,
                    "outcome": outcome,
                }
            )
            return None

    gate = _RecordingGate(repository=_ReviewRepo(), auto_create_review=True)
    await _node(monkeypatch, GateDecision.BLOCK, gate).execute(_state())

    assert calls == [
        {
            "dag_hash": _DAG_HASH,
            "brand": "Kisqali",
            "treatment": "rep_visits",
            "outcome": "trx",
        }
    ]
