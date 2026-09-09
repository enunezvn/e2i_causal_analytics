"""#1971 -- the expert-review gate is REAL on the live agent path.

Three behaviours, each pinned red-first:

1. A human REJECTION of the DAG structure is honoured on EVERY refutation band.
   Before this change the gate was consulted only on REVIEW/BLOCK, so a DAG a
   reviewer had explicitly rejected still yielded a PROCEED-band estimate that
   was promoted to ``causal_paths.validation_status='validated'`` and surfaced
   as ``completed``. Now the run halts (``status='failed'``,
   ``current_phase='awaiting_expert_review'``, ``expert_review_halt=True``),
   the verdict (reviewer, review id, reason) is in ``error_message`` and the
   caveat, the evidence is persisted UNLINKED (query-derived id, never under
   the path id) and no status transition fires. ``gate_decision`` keeps the
   statistical truth (``proceed``): a structural rejection changes no p-value.

2. ``CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL`` (env; default OFF; an explicit
   constructor argument wins) is the owner's real switch. OFF = today's
   behaviour byte-for-byte on a PROCEED band. ON = a REVIEW-band run whose gate
   decision is pending_review / blocked / unavailable halts honestly with the
   review id and how to resolve it; a real approval (proceed /
   renewal_required) still proceeds; a PROCEED band never needs approval --
   the same contract the retired SQL ``can_use_estimate`` promised.

3. Honest unavailability: the bare no-repository gate reports ``unavailable``
   (never ``proceed``), and the caveat says the gate was not consulted.

No DoWhy: the runner and the reconstruction are stand-ins (same device as
test_refutation_randomized.py). The fake review repository records every call
so the negatives ("no auto-create on PROCEED", "no promotion") are
positive-controlled by the lookups that DID happen.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.agents.causal_impact.nodes.refutation as refutation_mod
from src.agents.causal_impact.agent import CausalImpactAgent
from src.agents.causal_impact.nodes.refutation import (
    RefutationNode,
    _resolve_require_dag_approval,
)
from src.causal_engine.expert_review_gate import ExpertReviewGate
from src.causal_engine.refutation_runner import (
    GateDecision,
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
)
from src.repositories.causal_validation import (
    derive_causal_path_estimate_id,
    derive_query_estimate_id,
)

_ENV = "CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL"
_QUERY_ID = "q-1971abc00001"
_PATH_ID = "cp_real_1971_00001"
_DAG_HASH = "c0ffee1971deadbeef"

# The sentence the live caveat used to carry. It promised that expert review
# would let a borderline estimate be "used as a validated result" -- which the
# system cannot keep: approval is STRUCTURAL and never promotes needs_review
# to validated (#1969). Kept verbatim so the negative assertions below are
# positive-controlled against the real old wording, not a paraphrase.
_OLD_REVIEW_SENTENCE = "This estimate needs expert review before it is used as a validated result."


# ----------------------------------------------------------------- stand-ins


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
    """RefutationRunner stand-in returning a fixed band (no DoWhy refits)."""

    def __init__(self, band: GateDecision) -> None:
        self.band = band

    def run_all_tests(self, **kwargs: Any) -> RefutationSuite:
        return _suite(self.band)


class _ReviewRepo:
    """ExpertReviewRepository stand-in recording every call (no live DB)."""

    def __init__(
        self,
        approval: Optional[Dict[str, Any]] = None,
        rows: Optional[List[Dict[str, Any]]] = None,
        raise_on_lookup: bool = False,
    ) -> None:
        self.approval = approval
        self.rows = list(rows or [])
        self.raise_on_lookup = raise_on_lookup
        self.create_calls: List[Dict[str, Any]] = []
        self.lookups = 0

    async def get_dag_approval(self, dag_hash: str, brand: Optional[str] = None):
        self.lookups += 1
        if self.raise_on_lookup:
            raise RuntimeError("review store unreachable")
        return self.approval

    async def get_reviews_for_dag(
        self, dag_hash: str, include_expired: bool = False, brand: Optional[str] = None
    ):
        self.lookups += 1
        if self.raise_on_lookup:
            raise RuntimeError("review store unreachable")
        return list(self.rows)

    async def create_review(self, **kwargs: Any) -> str:
        self.create_calls.append(kwargs)
        return "rev-created"

    async def update_dag_structure(self, *args: Any, **kwargs: Any) -> bool:
        return True


def _rejected_row() -> Dict[str, Any]:
    return {
        "review_id": "rev-rejected",
        "approval_status": "rejected",
        "reviewer_name": "Dr. No",
        "concerns_raised": ["formulary_status is a collider, not a confounder"],
    }


def _approved_row() -> Dict[str, Any]:
    return {
        "review_id": "rev-approved",
        "approval_status": "approved",
        "reviewer_name": "Dr. Structure",
        "approved_at": "2026-01-01T00:00:00Z",
        "valid_until": "2099-01-01",
    }


class _PathRepo:
    """CausalPathRepository stand-in (explicit real row; records transitions)."""

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
        self.status_calls.append(
            {"path_id": path_id, "new_status": new_status, "allowed_current": allowed_current}
        )
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
    require_dag_approval: Optional[bool] = None,
    path_repo: Optional[_PathRepo] = None,
    validation_repo: Optional[MagicMock] = None,
) -> RefutationNode:
    node = RefutationNode(
        validation_repo=validation_repo if validation_repo is not None else _validation_repo(),
        expert_review_gate=gate,
        causal_path_repo=path_repo if path_repo is not None else _PathRepo(),
        require_dag_approval=require_dag_approval,
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


def _comparable(result: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the two per-run clocks (node latency, the suite's created_at stamp)
    so two otherwise identical results compare equal."""
    out = {k: v for k, v in result.items() if k != "refutation_latency_ms"}
    suite = dict(out.get("refutation_suite") or {})
    suite.pop("created_at", None)
    out["refutation_suite"] = suite
    return out


# ============================================================================
# 1. A human REJECTION is honoured on the PROCEED band (the harm-now fix)
# ============================================================================


class TestRejectionHonouredOnProceedBand:
    @pytest.mark.asyncio
    async def test_rejected_structure_halts_and_is_never_promoted(self, monkeypatch):
        repo = _ReviewRepo(rows=[_rejected_row()])
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)
        path_repo = _PathRepo()
        validation_repo = _validation_repo()
        node = _node(
            monkeypatch,
            GateDecision.PROCEED,
            gate,
            path_repo=path_repo,
            validation_repo=validation_repo,
        )

        result = await node.execute(_state())

        # Halted honestly -- never an estimate that looks completed.
        assert result["status"] == "failed"
        assert result["current_phase"] == "awaiting_expert_review"
        assert result["expert_review_halt"] is True
        # The statistical verdict is untouched: a structural rejection changes no p-value.
        assert result["gate_decision"] == "proceed"
        assert result["needs_review"] is False
        # The human verdict travels with the run.
        assert result["expert_review_decision"] == "rejected"
        assert result["expert_review_id"] == "rev-rejected"
        message = result["error_message"]
        assert "REJECTED" in message
        assert "Dr. No" in message
        assert "rev-rejected" in message
        assert "collider" in message, "the reviewer's stated reason must surface"
        assert "Dr. No" in result["review_caveat"]
        # Never promoted: no transition, evidence persisted UNLINKED so it can
        # never satisfy migration 119's evidence gate for this path.
        assert path_repo.status_calls == []
        assert result["causal_path_promotion"] == {}
        save_kwargs = validation_repo.save_suite.await_args.kwargs
        assert save_kwargs["estimate_id"] == derive_query_estimate_id(_QUERY_ID)
        assert save_kwargs["estimate_source"] == "causal_impact_query"
        assert save_kwargs["estimate_id"] != derive_causal_path_estimate_id(_PATH_ID)
        # A PROCEED band never queues a review row (#1970: rejection is durable).
        assert repo.create_calls == []
        assert repo.lookups >= 1, "positive control: the gate WAS consulted"

    @pytest.mark.asyncio
    async def test_switch_off_and_on_are_byte_identical_on_a_cleared_proceed_band(
        self, monkeypatch
    ):
        """Regression pin for "switch OFF = today's behaviour": with a review
        store that clears the structure, a PROCEED band is byte-identical
        whether enforcement is off or on -- same keys, same values, same
        promotion -- and the gate is never asked to queue anything."""
        off_repo = _ReviewRepo(rows=[])
        on_repo = _ReviewRepo(rows=[])
        off_paths = _PathRepo()
        on_paths = _PathRepo()
        off = _node(
            monkeypatch,
            GateDecision.PROCEED,
            ExpertReviewGate(repository=off_repo, auto_create_review=True),
            require_dag_approval=False,
            path_repo=off_paths,
        )
        on = _node(
            monkeypatch,
            GateDecision.PROCEED,
            ExpertReviewGate(repository=on_repo, auto_create_review=True),
            require_dag_approval=True,
            path_repo=on_paths,
        )

        with_off = await off.execute(_state())
        with_on = await on.execute(_state())

        assert _comparable(with_off) == _comparable(with_on)
        assert with_off["status"] != "failed"
        assert "expert_review_halt" not in with_off
        assert "expert_review_decision" not in with_off
        assert off_paths.status_calls == on_paths.status_calls
        assert off_paths.status_calls[0]["new_status"] == "validated"
        assert off_repo.create_calls == on_repo.create_calls == []
        assert off_repo.lookups >= 1 and on_repo.lookups >= 1, (
            "positive control: the rejection check ran in both"
        )

    @pytest.mark.asyncio
    async def test_no_review_store_never_blesses_a_real_path(self, monkeypatch):
        """codex iter-2 HIGH-3: a run that performed NO successful structural
        check (no gate, or a bare no-repository gate) may still serve its
        estimate under the advisory default, but it may not link evidence to a
        real causal_paths row or move its status. Only a run that actually
        cleared the structure promotes. In prod the gate and the persistence
        repos come from the same Supabase client, so this changes nothing on a
        healthy box and closes the split-factory flake."""
        for gate in (None, ExpertReviewGate(repository=None)):
            paths = _PathRepo()
            validation_repo = _validation_repo()
            node = _node(
                monkeypatch,
                GateDecision.PROCEED,
                gate,
                path_repo=paths,
                validation_repo=validation_repo,
            )

            result = await node.execute(_state())

            assert result["status"] != "failed", "the estimate itself is not withheld"
            assert "expert_review_halt" not in result
            assert paths.status_calls == [], f"gate={gate!r} promoted without a structural check"
            assert result["causal_path_promotion"] == {}
            save_kwargs = validation_repo.save_suite.await_args.kwargs
            assert save_kwargs["estimate_id"] == derive_query_estimate_id(_QUERY_ID)
            assert save_kwargs["estimate_source"] == "causal_impact_query"

    @pytest.mark.asyncio
    async def test_rejection_lookup_error_continues_but_never_blesses_the_path(
        self, monkeypatch, caplog
    ):
        """A review-store error on the probe means the structure's rejection
        status is UNKNOWN for this run. The estimate is not withheld (the
        default is advisory, and failing every PROCEED run on a transient
        store error would be a new failure mode) -- but a run that cannot
        prove the structure was not rejected must not bless the path: the
        evidence is persisted UNLINKED and no transition fires. A later run
        that can look promotes it. Observable in logs. (codex iter-1 HIGH-2)"""
        repo = _ReviewRepo(raise_on_lookup=True)
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)
        paths = _PathRepo()
        validation_repo = _validation_repo()
        node = _node(
            monkeypatch,
            GateDecision.PROCEED,
            gate,
            path_repo=paths,
            validation_repo=validation_repo,
        )

        with caplog.at_level(logging.WARNING):
            result = await node.execute(_state())

        assert result["status"] != "failed"
        assert "expert_review_halt" not in result
        assert paths.status_calls == []
        assert result["causal_path_promotion"] == {}
        save_kwargs = validation_repo.save_suite.await_args.kwargs
        assert save_kwargs["estimate_id"] == derive_query_estimate_id(_QUERY_ID)
        assert save_kwargs["estimate_source"] == "causal_impact_query"
        assert any(
            "rejection check" in r.getMessage().lower() and r.levelno == logging.WARNING
            for r in caplog.records
        ), "the degraded check must be observable in logs"


class _ProbeFindsRejectionThenConsultRaises(ExpertReviewGate):
    """check_rejection works (the repo answers), check_approval then raises."""

    async def check_approval(self, **kwargs):  # type: ignore[override]
        raise RuntimeError("review store went away between the two lookups")


class _ProbeRaisesThenConsultFindsRejection(ExpertReviewGate):
    """check_rejection raises, check_approval (real) then finds the rejection."""

    async def check_rejection(self, dag_hash, brand=None):  # type: ignore[override]
        raise RuntimeError("transient store error on the probe")


class TestOneStructuralVerdictPerRun:
    """codex iter-1: the structural verdict is resolved ONCE and reused.

    Two lookups (the read-only probe, then the REVIEW/BLOCK consult) can
    disagree when the review store flakes between them. Neither order may
    lose a rejection the run actually observed, and neither may move a path
    on evidence whose structure could not be cleared.
    """

    @pytest.mark.asyncio
    async def test_probe_rejection_is_authoritative_even_if_the_consult_raises(self, monkeypatch):
        """HIGH-1: probe says REJECTED, consult raises -> still halted as
        rejected (never 'unavailable' + continue), evidence unlinked."""
        repo = _ReviewRepo(rows=[_rejected_row()])
        gate = _ProbeFindsRejectionThenConsultRaises(repository=repo, auto_create_review=True)
        paths = _PathRepo()
        validation_repo = _validation_repo()
        node = _node(
            monkeypatch,
            GateDecision.REVIEW,
            gate,
            require_dag_approval=False,
            path_repo=paths,
            validation_repo=validation_repo,
        )

        result = await node.execute(_state())

        assert result["status"] == "failed"
        assert result["expert_review_halt"] is True
        assert result["expert_review_decision"] == "rejected"
        assert result["expert_review_id"] == "rev-rejected"
        assert "Dr. No" in result["error_message"]
        assert paths.status_calls == []
        save_kwargs = validation_repo.save_suite.await_args.kwargs
        assert save_kwargs["estimate_source"] == "causal_impact_query"
        assert repo.create_calls == []

    @pytest.mark.asyncio
    async def test_probe_error_then_consult_rejection_never_moved_the_path(
        self, monkeypatch, caplog
    ):
        """HIGH-2: probe errors (verdict unknown), consult then finds the
        rejection -> halted as rejected AND the evidence was persisted
        unlinked with no transition, because 'unknown' already withheld the
        linkage before anything was written."""
        repo = _ReviewRepo(rows=[_rejected_row()])
        gate = _ProbeRaisesThenConsultFindsRejection(repository=repo, auto_create_review=True)
        paths = _PathRepo()
        validation_repo = _validation_repo()
        node = _node(
            monkeypatch,
            GateDecision.REVIEW,
            gate,
            require_dag_approval=False,
            path_repo=paths,
            validation_repo=validation_repo,
        )

        with caplog.at_level(logging.WARNING):
            result = await node.execute(_state())

        assert result["status"] == "failed"
        assert result["expert_review_decision"] == "rejected"
        assert paths.status_calls == []
        assert result["causal_path_promotion"] == {}
        save_kwargs = validation_repo.save_suite.await_args.kwargs
        assert save_kwargs["estimate_id"] == derive_query_estimate_id(_QUERY_ID)
        assert save_kwargs["estimate_source"] == "causal_impact_query"
        assert repo.create_calls == []

    @pytest.mark.asyncio
    async def test_block_band_with_probe_rejection_skips_the_consult(self, monkeypatch):
        """Positive control for 'one verdict': on BLOCK the rejection found by
        the probe is reused -- no second lookup, no queue row, no demotion."""
        repo = _ReviewRepo(rows=[_rejected_row()])
        gate = _ProbeFindsRejectionThenConsultRaises(repository=repo, auto_create_review=True)
        paths = _PathRepo()
        node = _node(monkeypatch, GateDecision.BLOCK, gate, path_repo=paths)

        result = await node.execute(_state())

        assert result["status"] == "failed"
        assert result["expert_review_decision"] == "rejected"
        assert "REJECTED" in result["review_caveat"]
        assert paths.status_calls == []
        assert repo.create_calls == []


# ============================================================================
# 2. The switch: CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL
# ============================================================================


class TestRequireDagApprovalSwitch:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, False),
            ("", False),  # compose `${VAR:-}` puts "" in the container: same as unset
            ("   ", False),
            ("0", False),
            ("false", False),
            ("off", False),
            ("no", False),
            ("1", True),
            ("true", True),
            (" TRUE ", True),
            ("yes", True),
            ("on", True),
        ],
    )
    def test_env_values(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv(_ENV, raising=False)
        else:
            monkeypatch.setenv(_ENV, raw)
        assert _resolve_require_dag_approval(None) is expected

    def test_garbage_value_is_off_and_loud(self, monkeypatch, caplog):
        monkeypatch.setenv(_ENV, "maybe")
        with caplog.at_level(logging.WARNING):
            assert _resolve_require_dag_approval(None) is False
        assert any(_ENV in r.getMessage() for r in caplog.records)

    def test_explicit_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv(_ENV, "true")
        assert _resolve_require_dag_approval(False) is False
        monkeypatch.setenv(_ENV, "false")
        assert _resolve_require_dag_approval(True) is True

    def test_node_reads_the_env_switch(self, monkeypatch):
        monkeypatch.setenv(_ENV, "true")
        assert RefutationNode().require_dag_approval is True
        monkeypatch.delenv(_ENV, raising=False)
        assert RefutationNode().require_dag_approval is False

    @pytest.mark.asyncio
    async def test_on_review_band_pending_review_halts_with_resolve_hint(self, monkeypatch):
        repo = _ReviewRepo(rows=[])
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)
        paths = _PathRepo()
        node = _node(
            monkeypatch, GateDecision.REVIEW, gate, require_dag_approval=True, path_repo=paths
        )

        result = await node.execute(_state())

        assert result["status"] == "failed"
        assert result["current_phase"] == "awaiting_expert_review"
        assert result["expert_review_halt"] is True
        assert result["expert_review_decision"] == "pending_review"
        assert result["expert_review_id"] == "rev-created"
        assert result["gate_decision"] == "review"
        assert result["needs_review"] is True
        message = result["error_message"]
        assert "rev-created" in message
        assert "POST /expert-reviews/rev-created/resolve" in message
        assert _ENV in message, "the halt must name the switch that caused it"
        # The queue row WAS created (a human can resolve it) ...
        assert len(repo.create_calls) == 1
        # ... and a REVIEW band still moves pending -> needs_review, never validated.
        assert paths.status_calls and paths.status_calls[0]["new_status"] == "needs_review"

    @pytest.mark.asyncio
    async def test_off_review_band_pending_review_continues_as_today(self, monkeypatch):
        repo = _ReviewRepo(rows=[])
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)
        node = _node(monkeypatch, GateDecision.REVIEW, gate, require_dag_approval=False)

        result = await node.execute(_state())

        assert result["status"] == "in_progress"
        assert result["current_phase"] == "analyzing_sensitivity"
        assert "expert_review_halt" not in result
        assert result["expert_review_decision"] == "pending_review"
        assert result["expert_review_id"] == "rev-created"
        assert result["needs_review"] is True

    @pytest.mark.asyncio
    async def test_on_review_band_real_approval_proceeds(self, monkeypatch):
        repo = _ReviewRepo(approval=_approved_row())
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)
        node = _node(monkeypatch, GateDecision.REVIEW, gate, require_dag_approval=True)

        result = await node.execute(_state())

        assert result["status"] == "in_progress"
        assert "expert_review_halt" not in result
        assert result["expert_review_decision"] == "proceed"
        assert result["expert_review_id"] == "rev-approved"
        assert "expert-approved by Dr. Structure" in result["review_caveat"]

    @pytest.mark.asyncio
    async def test_on_review_band_bare_gate_halts_as_unavailable(self, monkeypatch):
        """Enforcement ON with no review store must not silently pass."""
        node = _node(
            monkeypatch,
            GateDecision.REVIEW,
            ExpertReviewGate(repository=None),
            require_dag_approval=True,
        )

        result = await node.execute(_state())

        assert result["status"] == "failed"
        assert result["expert_review_halt"] is True
        assert result["expert_review_decision"] == "unavailable"
        assert result["expert_review_id"] is None
        assert "could not be consulted" in result["error_message"]

    @pytest.mark.asyncio
    async def test_on_review_band_rejected_halts_with_verdict(self, monkeypatch):
        repo = _ReviewRepo(rows=[_rejected_row()])
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)
        node = _node(monkeypatch, GateDecision.REVIEW, gate, require_dag_approval=True)

        result = await node.execute(_state())

        assert result["status"] == "failed"
        assert result["expert_review_decision"] == "rejected"
        assert "Dr. No" in result["error_message"]
        assert repo.create_calls == []

    @pytest.mark.asyncio
    async def test_off_review_band_rejected_still_halts(self, monkeypatch):
        """A human rejection is honoured regardless of the switch (decision 2)."""
        repo = _ReviewRepo(rows=[_rejected_row()])
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)
        node = _node(monkeypatch, GateDecision.REVIEW, gate, require_dag_approval=False)

        result = await node.execute(_state())

        assert result["status"] == "failed"
        assert result["expert_review_halt"] is True
        assert result["expert_review_decision"] == "rejected"

    @pytest.mark.asyncio
    async def test_on_proceed_band_never_needs_approval(self, monkeypatch):
        """Mirrors the retired SQL can_use_estimate contract: proceed -> usable."""
        repo = _ReviewRepo(rows=[])
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)
        paths = _PathRepo()
        node = _node(
            monkeypatch, GateDecision.PROCEED, gate, require_dag_approval=True, path_repo=paths
        )

        result = await node.execute(_state())

        assert result["status"] != "failed"
        assert "expert_review_halt" not in result
        assert repo.create_calls == []
        assert paths.status_calls[0]["new_status"] == "validated"

    @pytest.mark.asyncio
    async def test_on_block_band_keeps_the_block_reason(self, monkeypatch):
        """BLOCK is already terminal-failed for a statistical reason; the switch
        must not overwrite that reason with an approval message."""
        repo = _ReviewRepo(rows=[])
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)
        node = _node(monkeypatch, GateDecision.BLOCK, gate, require_dag_approval=True)

        result = await node.execute(_state())

        assert result["status"] == "failed"
        assert result["current_phase"] == "failed"
        assert "expert_review_halt" not in result
        assert result["expert_review_decision"] == "pending_review"
        assert "blocked due to low confidence" in result["error_message"].lower()
        assert _ENV not in result["error_message"]


# ============================================================================
# 3. Honest unavailability + truthful caveat
# ============================================================================


class TestHonestUnavailability:
    @pytest.mark.asyncio
    async def test_bare_gate_reports_unavailable_never_proceed(self):
        node = RefutationNode(expert_review_gate=ExpertReviewGate(repository=None))
        fields = await node._consult_review_gate(_state(), _suite(GateDecision.REVIEW))
        assert fields["expert_review_decision"] == "unavailable"
        assert fields["expert_review_id"] is None
        caveat = fields["review_caveat"]
        assert "could not be consulted" in caveat
        assert "expert-approved" not in caveat

    @pytest.mark.asyncio
    async def test_gate_exception_reports_unavailable(self):
        class _Boom:
            async def check_approval(self, **kwargs):
                raise RuntimeError("review store down")

        node = RefutationNode(expert_review_gate=_Boom())
        fields = await node._consult_review_gate(_state(), _suite(GateDecision.REVIEW))
        assert fields["expert_review_decision"] == "unavailable"
        assert "could not be consulted" in fields["review_caveat"]


class TestCaveatTellsTheTruth:
    @pytest.mark.asyncio
    async def test_review_caveat_no_longer_promises_promotion_by_approval(self):
        repo = _ReviewRepo(rows=[])
        node = RefutationNode(
            expert_review_gate=ExpertReviewGate(repository=repo, auto_create_review=True)
        )
        fields = await node._consult_review_gate(_state(), _suite(GateDecision.REVIEW))
        caveat = fields["review_caveat"]

        assert caveat, "positive control: a REVIEW band always carries a caveat"
        assert _OLD_REVIEW_SENTENCE not in caveat
        assert "needs expert review before it is used" not in caveat
        # Positive control for the negative above: the substring DOES match the
        # real old wording, so the assertion is not vacuous.
        assert "needs expert review before it is used" in _OLD_REVIEW_SENTENCE
        # The truth: only a PROCEED re-run promotes; approval is structural.
        assert "Only a PROCEED re-run" in caveat
        assert "not this estimate's statistical robustness" in caveat
        assert "REVIEW" in caveat

    @pytest.mark.asyncio
    async def test_pending_caveat_says_queued_and_how_to_resolve(self):
        repo = _ReviewRepo(rows=[])
        node = RefutationNode(
            expert_review_gate=ExpertReviewGate(repository=repo, auto_create_review=True)
        )
        fields = await node._consult_review_gate(_state(), _suite(GateDecision.REVIEW))
        caveat = fields["review_caveat"]
        assert "queued for expert review" in caveat
        assert "rev-created" in caveat
        assert "POST /expert-reviews/rev-created/resolve" in caveat

    @pytest.mark.asyncio
    async def test_rejected_caveat_names_reviewer_review_and_reason(self):
        repo = _ReviewRepo(rows=[_rejected_row()])
        node = RefutationNode(
            expert_review_gate=ExpertReviewGate(repository=repo, auto_create_review=True)
        )
        fields = await node._consult_review_gate(_state(), _suite(GateDecision.REVIEW))
        caveat = fields["review_caveat"]
        assert "REJECTED" in caveat
        assert "Dr. No" in caveat
        assert "rev-rejected" in caveat
        assert "collider" in caveat
        assert fields["expert_review_decision"] == "rejected"

    @pytest.mark.asyncio
    async def test_proceed_band_rejected_caveat_names_the_proceed_band(self):
        repo = _ReviewRepo(rows=[_rejected_row()])
        node = RefutationNode(
            expert_review_gate=ExpertReviewGate(repository=repo, auto_create_review=True)
        )
        fields = await node._consult_review_gate(_state(), _suite(GateDecision.PROCEED))
        caveat = fields["review_caveat"]
        assert "PROCEED" in caveat
        assert "REVIEW (borderline" not in caveat
        assert "REJECTED" in caveat


# ============================================================================
# 4. The agent's output honours the halt (never "completed")
# ============================================================================


class TestBuildOutputHonoursHalt:
    def _final_state(self, halt: bool) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "query_id": "q1",
            "estimation_result": {
                "ate": 0.5,
                "ate_ci_lower": 0.3,
                "ate_ci_upper": 0.7,
                "statistical_significance": True,
                "method": "linear_regression",
            },
            "refutation_results": _suite(GateDecision.PROCEED).to_legacy_format(),
            "gate_decision": "proceed",
            "interpretation": {},
            "sensitivity_analysis": {},
            "causal_graph": {},
        }
        if halt:
            state["expert_review_halt"] = True
            state["expert_review_decision"] = "rejected"
            state["expert_review_id"] = "rev-rejected"
            state["error_message"] = "Estimate withheld: the DAG structure was REJECTED by Dr. No."
            state["status"] = "failed"
        return state

    def test_halted_run_is_failed_with_the_halt_as_narrative(self):
        agent = CausalImpactAgent()
        out = agent._build_output(self._final_state(halt=True), time.time())
        assert out["status"] == "failed"
        assert out["causal_narrative"].startswith("Estimate withheld")
        assert "Analysis completed successfully" not in out["causal_narrative"]
        assert out["requires_further_analysis"] is True
        # Statistical truth preserved: the suite DID pass.
        assert out["refutation_passed"] is True
        assert out["gate_decision"] == "proceed"

    def test_same_state_without_halt_completes(self):
        """Positive control: the only difference is the halt key."""
        agent = CausalImpactAgent()
        out = agent._build_output(self._final_state(halt=False), time.time())
        assert out["status"] == "completed"
        assert out["refutation_passed"] is True


# ============================================================================
# 5. The graph ends the run at the halt and records WHY
# ============================================================================


class TestGraphTerminatesAtTheHalt:
    def test_halt_routes_to_error_handler_not_sensitivity(self):
        from src.agents.causal_impact.graph import should_continue_after_refutation

        halted = {
            "status": "failed",
            "current_phase": "awaiting_expert_review",
            "expert_review_halt": True,
            "gate_decision": "proceed",
            "refutation_results": {"gate_decision": "proceed"},
        }
        assert should_continue_after_refutation(halted) == "error_handler"
        # Positive control: the same PROCEED state without the halt continues.
        assert (
            should_continue_after_refutation(
                {"gate_decision": "proceed", "refutation_results": {"gate_decision": "proceed"}}
            )
            == "sensitivity"
        )

    def test_error_handler_records_the_halt_phase_and_message(self):
        from src.agents.causal_impact.graph import handle_workflow_error

        halted = {
            "status": "failed",
            "current_phase": "awaiting_expert_review",
            "expert_review_halt": True,
            "expert_review_decision": "pending_review",
            "expert_review_id": "rev-created",
            "error_message": "Estimate withheld: ... POST /expert-reviews/rev-created/resolve",
            "errors": [],
            "warnings": [],
        }
        out = handle_workflow_error(halted)
        assert out["status"] == "failed"
        assert out["errors"] == [
            {
                "phase": "awaiting_expert_review",
                "message": "Estimate withheld: ... POST /expert-reviews/rev-created/resolve",
            }
        ]
        # The review fields survive the terminal node (LangGraph merges, and
        # they are declared in state.py so they are not dropped).
        assert out["expert_review_halt"] is True
        assert out["expert_review_id"] == "rev-created"
