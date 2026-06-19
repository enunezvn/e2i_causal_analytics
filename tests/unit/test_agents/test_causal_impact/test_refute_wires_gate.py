"""R6-F2 C2 — refute_causal_estimate wires a repo-backed ExpertReviewGate.

Before C2, ``refute_causal_estimate`` built ``RefutationNode(config=...,
validation_repo=...)`` WITHOUT an ``expert_review_gate``, so a REVIEW band only
ever consulted the lazily-built no-repository gate (bypass-to-PROCEED). No
``expert_reviews`` row was ever created — the human-in-the-loop loop had no
producer.

C2 adds a module-level ``_build_expert_review_gate()`` helper (repo-backed,
``auto_create_review=True``, graceful-degrade to None on missing Supabase) and
passes its result into the node. These tests pin:

1. ``refute_causal_estimate`` constructs the node WITH the built gate.
2. The repo-backed gate, on a REVIEW band + a NEW dag_hash, calls
   ``create_review`` once with ``review_type="dag_approval"`` (the C1 enum) and
   surfaces ``needs_review=True`` + ``expert_review_decision="pending_review"``.
3. Supabase absent (factory raises) -> ``_build_expert_review_gate`` returns
   None and the node still flags ``needs_review=True`` (no crash, no orphan row).

NO live PostgREST insert: the gate is backed by a kwargs-capturing fake repo, so
the live ``expert_reviews`` table is never touched.
"""

from __future__ import annotations

import pytest

import src.agents.causal_impact.nodes.refutation as refutation_mod
from src.agents.causal_impact.nodes.refutation import (
    RefutationNode,
    _build_expert_review_gate,
    refute_causal_estimate,
)
from src.causal_engine.expert_review_gate import ExpertReviewGate
from src.causal_engine.refutation_runner import GateDecision, RefutationSuite
from src.memory.services.factories import ServiceConnectionError


def _review_suite(confidence: float = 0.6) -> RefutationSuite:
    return RefutationSuite(
        passed=True,
        confidence_score=confidence,
        tests=[],
        gate_decision=GateDecision.REVIEW,
    )


class _CapturingRepo:
    """Fake repo recording create_review kwargs (no live DB)."""

    def __init__(self) -> None:
        self.create_kwargs: dict | None = None

    async def get_dag_approval(self, dag_hash, brand=None):
        return None

    async def get_reviews_for_dag(self, dag_hash, include_expired=False, brand=None):
        return []

    async def create_review(self, **kwargs):
        self.create_kwargs = kwargs
        return "rev-captured"


class TestRefuteBuildsRepoBackedGate:
    @pytest.mark.asyncio
    async def test_refute_passes_built_gate_into_node(self, monkeypatch):
        """refute_causal_estimate must construct the node WITH the built gate."""
        sentinel_gate = object()

        async def _fake_build():
            return sentinel_gate

        captured: dict = {}

        real_init = RefutationNode.__init__

        def _capturing_init(self, *args, **kwargs):
            captured["expert_review_gate"] = kwargs.get("expert_review_gate")
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(refutation_mod, "_build_expert_review_gate", _fake_build)
        monkeypatch.setattr(RefutationNode, "__init__", _capturing_init)

        # No estimation_result -> execute fail-closes early, but the node is still
        # constructed first, which is the wiring under test.
        await refute_causal_estimate({"query_id": "q1"})

        assert captured.get("expert_review_gate") is sentinel_gate

    @pytest.mark.asyncio
    async def test_repo_backed_gate_creates_dag_approval_row_on_review(self):
        """A REVIEW band + new dag_hash -> create_review('dag_approval') fires once."""
        repo = _CapturingRepo()
        gate = ExpertReviewGate(repository=repo, auto_create_review=True)
        node = RefutationNode(expert_review_gate=gate)

        state = {
            "treatment_var": "email_frequency",
            "outcome_var": "trx",
            "brand": "Remibrutinib",
            # The graph builder writes `dag_version_hash` (graph_builder.py); the old
            # `dag_hash` key was never populated, so the gate must key on the real one.
            "dag_version_hash": "deadbeefcafebabe",
            "query_id": "causal_impact_agent",
        }
        fields = await node._consult_review_gate(state, _review_suite())

        assert repo.create_kwargs is not None, "create_review was not called"
        assert repo.create_kwargs["review_type"] == "dag_approval"
        assert repo.create_kwargs["dag_version_hash"] == "deadbeefcafebabe"
        # needs_review is set by the caller (execute) from suite.needs_review, NOT by
        # _consult_review_gate (now shared by REVIEW + BLOCK bands).
        assert "needs_review" not in fields
        assert fields["expert_review_decision"] == "pending_review"

    @pytest.mark.asyncio
    async def test_build_gate_degrades_to_none_on_service_connection_error(self, monkeypatch):
        """Missing-config (ServiceConnectionError) -> returns None (graceful degrade).

        This is the ONLY error class the helper is allowed to swallow: it is what
        ``get_async_supabase_client`` raises when the Supabase env is absent
        (dev/test) or the connection cannot be established. Any OTHER error must
        propagate (see test_unexpected_error_propagates_not_bypass) so a real prod
        failure on a REVIEW-band estimate is never silently bypassed as approved.
        """

        async def _raise(*args, **kwargs):
            raise ServiceConnectionError("Supabase", "SUPABASE_URL environment variable is not set")

        # Patch the factory at its source so the helper's lazy import picks it up.
        monkeypatch.setattr("src.memory.services.factories.get_async_supabase_client", _raise)

        gate = await _build_expert_review_gate()
        assert gate is None

    @pytest.mark.asyncio
    async def test_unexpected_error_propagates_not_bypass(self, monkeypatch):
        """An UNEXPECTED factory error must PROPAGATE (fail-loud), not degrade.

        FIX A (codex HIGH): the old broad ``except Exception`` collapsed any error
        — including a transient/unexpected PROD Supabase failure — into the same
        ``return None`` -> bare-gate bypass (is_approved=True) as a dev/test
        missing-config. That silently self-bypasses the review gate in prod on an
        unexpected error. A non-ServiceConnectionError must surface.
        """

        async def _raise(*args, **kwargs):
            raise ValueError("boom — unexpected, not a missing-config signal")

        monkeypatch.setattr("src.memory.services.factories.get_async_supabase_client", _raise)

        # Direct helper call: the unexpected error must escape (not return None).
        with pytest.raises(ValueError, match="boom"):
            await _build_expert_review_gate()

        # And it must propagate all the way out of refute_causal_estimate, NOT be
        # swallowed into a silent bypass that proceeds as approved.
        with pytest.raises(ValueError, match="boom"):
            await refute_causal_estimate({"query_id": "q1"})

    @pytest.mark.asyncio
    async def test_supabase_absent_still_flags_needs_review(self, monkeypatch):
        """With the gate degraded to None, a REVIEW band still flags needs_review.

        needs_review is now owned by the caller (execute) and sourced from
        ``suite.needs_review`` — True for a REVIEW band REGARDLESS of whether the
        gate is present. So ``_consult_review_gate`` degrades gracefully (no
        ``needs_review`` key, no crash) and the REVIEW suite still drives the flag.
        """

        async def _build_none():
            return None

        monkeypatch.setattr(refutation_mod, "_build_expert_review_gate", _build_none)

        node = RefutationNode(expert_review_gate=None)
        fields = await node._consult_review_gate(
            {"treatment_var": "t", "outcome_var": "y", "dag_version_hash": "x"},
            _review_suite(),
        )
        # The helper no longer owns needs_review; it degrades gracefully.
        assert "needs_review" not in fields
        # The REVIEW band is the (gate-independent) source of needs_review.
        assert _review_suite().needs_review is True
