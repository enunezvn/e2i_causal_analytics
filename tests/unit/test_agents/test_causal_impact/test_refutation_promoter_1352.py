"""RefutationNode as the SOLE promoter of causal_paths.validation_status (#1352 item 3).

Migration 119 (Lane V, PR #1385) pinned the semantics: 'validated' asserts
"RefutationSuite evidence exists and passed", enforced by a BEFORE trigger that
rejects a real path claiming 'validated' without passed causal_validations rows
under ``causal_path_estimate_id(path_id)``. This lane wires the producer side:

* evidence for a run tied to a REAL causal_paths row is persisted under
  ``derive_causal_path_estimate_id(path_id)`` (estimate_source='causal_paths')
  — the id the trigger's evidence gate looks up — and ONLY THEN is the row's
  status moved (the trigger enforces the same order; the code never relies on
  the trigger being installed);
* an UNLINKED run persists under a query-derived uuid
  (``derive_query_estimate_id``) with estimate_source='causal_impact_query' —
  the old ``estimate_id=query_id`` write could NEVER succeed (query_id is
  ``q-<hex12>``, the column is uuid; every insert failed the cast, which is
  half of why causal_validations measured 0 rows in #1352);
* gate→status mapping (demotion mechanics are this lane's documented call):
  PROCEED: pending/needs_review → validated; REVIEW: pending → needs_review
  (never downgrades validated on a borderline re-run); BLOCK:
  pending/needs_review/validated → refuted;
* promotion NEVER fires for a synthetic-fixture run (data_source='synthetic'),
  for a synthetic path row, or on an ambiguous (non-unique) auto-match;
* evidence rows are never deleted — demotion adds contradicting evidence and
  flips status, so migration 119's deliberately-unguarded DELETE surface stays
  untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.causal_impact.nodes.refutation import RefutationNode
from src.causal_engine.refutation_runner import GateDecision, RefutationSuite
from src.repositories.causal_path import CausalPathRepository
from src.repositories.causal_validation import (
    derive_causal_path_estimate_id,
    derive_query_estimate_id,
)


def _suite(gate: GateDecision) -> RefutationSuite:
    return RefutationSuite(
        passed=gate != GateDecision.BLOCK,
        confidence_score=0.9 if gate == GateDecision.PROCEED else 0.4,
        tests=[],
        gate_decision=gate,
        treatment_variable="rep_visits",
        outcome_variable="trx",
        brand="Kisqali",
    )


def _state(**overrides: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "query_id": "q-abc123def456",
        "treatment_var": "rep_visits",
        "outcome_var": "trx",
        "brand": "Kisqali",
        "data_source": "kpi_substrate:WS3-BI-009",
    }
    state.update(overrides)
    return state


class _FakePathRepo:
    """Deterministic stand-in for CausalPathRepository (no MagicMock hasattr traps)."""

    def __init__(
        self,
        rows_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
        pair_rows: Optional[List[Dict[str, Any]]] = None,
        fail_update: bool = False,
    ) -> None:
        self.rows_by_id = rows_by_id or {}
        self.pair_rows = pair_rows or []
        self.fail_update = fail_update
        self.status_calls: List[Dict[str, Any]] = []

    async def get_path_row(self, path_id: str) -> Optional[Dict[str, Any]]:
        return self.rows_by_id.get(path_id)

    async def find_real_paths_for_pair(
        self, treatment: str, outcome: str, brand: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        return list(self.pair_rows)

    async def set_validation_status(
        self, path_id: str, new_status: str, allowed_current: tuple, **kwargs: Any
    ) -> bool:
        if self.fail_update:
            raise RuntimeError("db write failed")
        self.status_calls.append(
            {
                "path_id": path_id,
                "new_status": new_status,
                "allowed_current": allowed_current,
                "dag_version_hash": kwargs.get("dag_version_hash"),
                "brand": kwargs.get("brand"),
                # Raw keywords: ``kwargs.get`` cannot tell "omitted" from "None";
                # the no-hash test pins that the keyword was explicitly forwarded.
                "kwargs": dict(kwargs),
            }
        )
        return True


def _real_row(path_id: str = "cp_real_000000001") -> Dict[str, Any]:
    return {
        "path_id": path_id,
        "start_node": "rep_visits",
        "end_node": "trx",
        "brand": "Kisqali",
        "validation_status": "pending",
        "is_synthetic": False,
    }


def _validation_repo() -> MagicMock:
    repo = MagicMock()
    repo.save_suite = AsyncMock(return_value=["v-1", "v-2"])
    return repo


class TestUnlinkedPersistence:
    @pytest.mark.asyncio
    async def test_unlinked_run_persists_under_query_derived_uuid(self) -> None:
        repo = _validation_repo()
        node = RefutationNode(validation_repo=repo, causal_path_repo=_FakePathRepo())
        ids, promotion = await node._persist_suite_and_promote(
            _state(),
            _suite(GateDecision.PROCEED),
            structure_verdict="clear",
        )
        assert ids == ["v-1", "v-2"]
        kwargs = repo.save_suite.call_args.kwargs
        assert kwargs["estimate_id"] == derive_query_estimate_id("q-abc123def456")
        assert kwargs["estimate_source"] == "causal_impact_query"
        assert promotion == {}

    @pytest.mark.asyncio
    async def test_no_repo_skips_persistence_and_promotion(self) -> None:
        node = RefutationNode(validation_repo=None, causal_path_repo=None)
        ids, promotion = await node._persist_suite_and_promote(
            _state(),
            _suite(GateDecision.PROCEED),
            structure_verdict="clear",
        )
        assert ids == []
        assert promotion == {}


class TestLinkedPromotion:
    @pytest.mark.asyncio
    async def test_explicit_path_id_proceed_persists_then_promotes(self) -> None:
        repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        order: List[str] = []
        repo.save_suite = AsyncMock(side_effect=lambda **kw: order.append("evidence") or ["v-1"])
        original_set = path_repo.set_validation_status

        async def _tracking_set(*a: Any, **kw: Any) -> bool:
            order.append("status")
            return await original_set(*a, **kw)

        path_repo.set_validation_status = _tracking_set  # type: ignore[method-assign]

        node = RefutationNode(validation_repo=repo, causal_path_repo=path_repo)
        ids, promotion = await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001"),
            _suite(GateDecision.PROCEED),
            structure_verdict="clear",
        )
        kwargs = repo.save_suite.call_args.kwargs
        assert kwargs["estimate_id"] == derive_causal_path_estimate_id("cp_real_000000001")
        assert kwargs["estimate_source"] == "causal_paths"
        # Migration-119 order: evidence FIRST, status flip second.
        assert order == ["evidence", "status"]
        assert promotion["path_id"] == "cp_real_000000001"
        assert promotion["new_status"] == "validated"
        assert path_repo.status_calls[0]["new_status"] == "validated"
        assert set(path_repo.status_calls[0]["allowed_current"]) == {
            "pending",
            "needs_review",
        }

    @pytest.mark.asyncio
    async def test_review_gate_marks_needs_review_from_pending_only(self) -> None:
        repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        node = RefutationNode(validation_repo=repo, causal_path_repo=path_repo)
        _ids, promotion = await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001"),
            _suite(GateDecision.REVIEW),
            structure_verdict="clear",
        )
        assert promotion["new_status"] == "needs_review"
        assert set(path_repo.status_calls[0]["allowed_current"]) == {"pending"}

    @pytest.mark.asyncio
    async def test_block_gate_demotes_to_refuted(self) -> None:
        repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        node = RefutationNode(validation_repo=repo, causal_path_repo=path_repo)
        _ids, promotion = await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001"),
            _suite(GateDecision.BLOCK),
            structure_verdict="clear",
        )
        assert promotion["new_status"] == "refuted"
        assert set(path_repo.status_calls[0]["allowed_current"]) == {
            "pending",
            "needs_review",
            "validated",
        }

    @pytest.mark.asyncio
    async def test_auto_match_unique_real_pending_row_promotes(self) -> None:
        repo = _validation_repo()
        path_repo = _FakePathRepo(pair_rows=[_real_row()])
        node = RefutationNode(validation_repo=repo, causal_path_repo=path_repo)
        _ids, promotion = await node._persist_suite_and_promote(
            _state(),
            _suite(GateDecision.PROCEED),
            structure_verdict="clear",
        )
        assert promotion.get("path_id") == "cp_real_000000001"
        kwargs = repo.save_suite.call_args.kwargs
        assert kwargs["estimate_id"] == derive_causal_path_estimate_id("cp_real_000000001")

    @pytest.mark.asyncio
    async def test_ambiguous_auto_match_does_not_promote(self) -> None:
        repo = _validation_repo()
        path_repo = _FakePathRepo(
            pair_rows=[_real_row("cp_a_00000000001"), _real_row("cp_b_00000000001")]
        )
        node = RefutationNode(validation_repo=repo, causal_path_repo=path_repo)
        _ids, promotion = await node._persist_suite_and_promote(
            _state(),
            _suite(GateDecision.PROCEED),
            structure_verdict="clear",
        )
        assert promotion == {}
        assert path_repo.status_calls == []
        # Evidence still lands, under the run's own query-derived id.
        kwargs = repo.save_suite.call_args.kwargs
        assert kwargs["estimate_source"] == "causal_impact_query"

    @pytest.mark.asyncio
    async def test_promote_carries_the_structure_hash_and_brand(self) -> None:
        """Lane 1 (spec §4.3): the guarded RPC needs the run's DAG hash and brand
        to evaluate the rejection rule inside the UPDATE."""
        repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        node = RefutationNode(validation_repo=repo, causal_path_repo=path_repo)
        await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001", dag_version_hash="h" * 64, brand="Kisqali"),
            _suite(GateDecision.PROCEED),
            structure_verdict="clear",
        )
        call = path_repo.status_calls[0]
        assert call["dag_version_hash"] == "h" * 64
        assert call["brand"] == "Kisqali"

    @pytest.mark.asyncio
    async def test_promote_without_a_hash_passes_none(self) -> None:
        repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        node = RefutationNode(validation_repo=repo, causal_path_repo=path_repo)
        state = _state(causal_path_id="cp_real_000000001")
        state.pop("dag_version_hash", None)
        await node._persist_suite_and_promote(
            state, _suite(GateDecision.PROCEED), structure_verdict="clear"
        )
        call = path_repo.status_calls[0]
        assert call["dag_version_hash"] is None
        # Explicitly forwarded as None (not merely omitted): the repository's
        # keyword-only parameter must receive it on every promote.
        assert "dag_version_hash" in call["kwargs"]
        assert call["kwargs"]["dag_version_hash"] is None

    @pytest.mark.asyncio
    async def test_promote_reaches_the_guarded_rpc_through_the_real_repository(self) -> None:
        """SEAM: the real node drives the real ``CausalPathRepository`` (stubbed
        client) so the node→repo keyword contract is pinned by executing code,
        not by two test files agreeing on a fake's signature. A drifted keyword
        at the call site raises TypeError inside the promoter's swallow and the
        RPC is never reached — this test then fails on ``call_args``."""
        client = MagicMock()
        # get_path_row → base get_many: select → eq → limit → offset → execute.
        query = MagicMock()
        query.eq.return_value = query
        query.limit.return_value = query
        query.offset.return_value = query
        query.execute = AsyncMock(return_value=MagicMock(data=[_real_row()]))
        client.table.return_value.select.return_value = query
        client.rpc.return_value.execute = AsyncMock(
            return_value=MagicMock(data={"moved": 1, "rejected": False})
        )
        real_repo = CausalPathRepository(supabase_client=client)

        node = RefutationNode(validation_repo=_validation_repo(), causal_path_repo=real_repo)
        _ids, promotion = await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001", dag_version_hash="h" * 64, brand="Kisqali"),
            _suite(GateDecision.PROCEED),
            structure_verdict="clear",
        )
        # _GATE_STATUS_TRANSITIONS[PROCEED] == ("validated", ("pending", "needs_review")).
        assert client.rpc.call_args.args == (
            "promote_causal_path_guarded",
            {
                "p_path_id": "cp_real_000000001",
                "p_new_status": "validated",
                "p_allowed_current": ["pending", "needs_review"],
                "p_dag_version_hash": "h" * 64,
                "p_brand": "Kisqali",
            },
        )
        assert promotion["path_id"] == "cp_real_000000001"
        assert promotion["new_status"] == "validated"


class TestPromotionGuards:
    @pytest.mark.asyncio
    async def test_synthetic_fixture_run_never_promotes(self) -> None:
        repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        node = RefutationNode(validation_repo=repo, causal_path_repo=path_repo)
        _ids, promotion = await node._persist_suite_and_promote(
            _state(data_source="synthetic", causal_path_id="cp_real_000000001"),
            _suite(GateDecision.PROCEED),
            structure_verdict="clear",
        )
        assert promotion == {}
        assert path_repo.status_calls == []

    @pytest.mark.asyncio
    async def test_synthetic_path_row_never_promoted(self) -> None:
        row = _real_row()
        row["is_synthetic"] = True
        repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": row})
        node = RefutationNode(validation_repo=repo, causal_path_repo=path_repo)
        _ids, promotion = await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001"),
            _suite(GateDecision.PROCEED),
            structure_verdict="clear",
        )
        assert promotion == {}
        assert path_repo.status_calls == []
        # And the evidence is NOT written under the synthetic path's id.
        kwargs = repo.save_suite.call_args.kwargs
        assert kwargs["estimate_source"] == "causal_impact_query"

    @pytest.mark.asyncio
    async def test_status_write_failure_degrades_without_raising(self) -> None:
        repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()}, fail_update=True)
        node = RefutationNode(validation_repo=repo, causal_path_repo=path_repo)
        ids, promotion = await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001"),
            _suite(GateDecision.PROCEED),
            structure_verdict="clear",
        )
        # Evidence persisted; the failed flip is reported as no promotion.
        assert ids == ["v-1", "v-2"]
        assert promotion == {}


class TestQueryEstimateDerivation:
    def test_query_derivation_is_deterministic_and_uuid(self) -> None:
        import uuid

        a = derive_query_estimate_id("q-abc123def456")
        b = derive_query_estimate_id("q-abc123def456")
        assert a == b
        uuid.UUID(a)  # must be a valid uuid (the estimate_id column is uuid)

    def test_query_and_path_namespaces_never_collide(self) -> None:
        assert derive_query_estimate_id("x") != derive_causal_path_estimate_id("x")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])


class TestRejectedStructureNeverBlessesAPath:
    """#1971: a run whose DAG structure a human REJECTED persists only UNLINKED
    evidence and never moves the path's status -- on every band.

    Why unlinked, not just "skip the transition": path-linked *passed* rows
    would satisfy migration 119's evidence gate for a later ``validated`` claim.
    A suite that passed on a structure a reviewer rejected is conditional on a
    premise the reviewer threw out; it must not be able to bless the path even
    in principle. Same mechanism as the synthetic-fixture rule above. The same
    holds when the verdict could not be determined (``"unknown"``): a run that
    cannot prove the structure was not rejected does not bless the path.
    """

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_unchecked_verdict_is_unlinked_and_not_promoted(self) -> None:
        """codex iter-2 HIGH-3: no structural check at all (no review store)
        may not authorise a real-path mutation either."""
        validation_repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        node = RefutationNode(validation_repo=validation_repo, causal_path_repo=path_repo)

        _ids, promotion = await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001"),
            _suite(GateDecision.PROCEED),
            structure_verdict="unchecked",
        )

        assert promotion == {}
        assert path_repo.status_calls == []
        assert validation_repo.save_suite.await_args.kwargs["estimate_source"] == (
            "causal_impact_query"
        )

    @pytest.mark.asyncio
    async def test_unknown_verdict_is_unlinked_and_not_promoted(self) -> None:
        validation_repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        node = RefutationNode(validation_repo=validation_repo, causal_path_repo=path_repo)

        _ids, promotion = await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001"),
            _suite(GateDecision.PROCEED),
            structure_verdict="unknown",
        )

        assert promotion == {}
        assert path_repo.status_calls == []
        assert validation_repo.save_suite.await_args.kwargs["estimate_source"] == (
            "causal_impact_query"
        )

    @pytest.mark.asyncio
    async def test_proceed_on_rejected_structure_is_unlinked_and_not_promoted(self) -> None:
        validation_repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        node = RefutationNode(validation_repo=validation_repo, causal_path_repo=path_repo)

        ids, promotion = await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001"),
            _suite(GateDecision.PROCEED),
            structure_verdict="rejected",
        )

        assert ids == ["v-1", "v-2"]
        assert promotion == {}
        assert path_repo.status_calls == []
        kwargs = validation_repo.save_suite.await_args.kwargs
        assert kwargs["estimate_id"] == derive_query_estimate_id("q-abc123def456")
        assert kwargs["estimate_source"] == "causal_impact_query"

    @pytest.mark.asyncio
    async def test_block_on_rejected_structure_does_not_demote_either(self) -> None:
        """A failed suite on a rejected structure says nothing about the path."""
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        node = RefutationNode(validation_repo=_validation_repo(), causal_path_repo=path_repo)

        _ids, promotion = await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001"),
            _suite(GateDecision.BLOCK),
            structure_verdict="rejected",
        )

        assert promotion == {}
        assert path_repo.status_calls == []

    @pytest.mark.asyncio
    async def test_positive_control_same_call_without_the_flag_promotes(self) -> None:
        validation_repo = _validation_repo()
        path_repo = _FakePathRepo(rows_by_id={"cp_real_000000001": _real_row()})
        node = RefutationNode(validation_repo=validation_repo, causal_path_repo=path_repo)

        _ids, promotion = await node._persist_suite_and_promote(
            _state(causal_path_id="cp_real_000000001"),
            _suite(GateDecision.PROCEED),
            structure_verdict="clear",
        )

        assert promotion["new_status"] == "validated"
        assert validation_repo.save_suite.await_args.kwargs[
            "estimate_id"
        ] == derive_causal_path_estimate_id("cp_real_000000001")
