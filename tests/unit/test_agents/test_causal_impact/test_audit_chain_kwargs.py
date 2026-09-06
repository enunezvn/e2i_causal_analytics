"""Regression tests for issue #355 (S9): causal_impact audit-chain kwarg mismatch.

Pre-fix, ``src/agents/causal_impact/graph.py:173-187`` called
``AuditChainService.add_entry(...)`` with kwargs that did not match the method
signature: ``input_hash``/``output_hash``/``user_id``/``session_id``/``brand``
and passed ``agent_tier`` as a string (``.value``) instead of the
:class:`AgentTier` enum. The mismatch was silenced by
``# type: ignore[call-arg]`` at static-check time, and by a
``try/except Exception`` that turned the runtime ``TypeError`` into a debug
warning. The net effect was that no audit-chain entries were written for any
causal_impact node execution — the chain's ``previous_hash`` invariant was
broken and expert_review repository lookups keyed on audit lineage could not
trace causal_impact ancestry.

These tests assert the *call shape* (kwargs by name) instead of inner state.
The kwargs that survive are exactly those declared by
:meth:`AuditChainService.add_entry` (see ``src/utils/audit_chain.py:288-301``).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.agents.base.audit_chain_mixin import set_audit_chain_service
from src.agents.causal_impact.graph import traced_node
from src.utils.audit_chain import AgentTier, AuditChainService, RefutationResults


@pytest.fixture(autouse=True)
def _reset_audit_service():
    set_audit_chain_service(None)
    yield
    set_audit_chain_service(None)


@pytest.fixture
def mock_audit_service():
    svc = MagicMock(spec=AuditChainService)
    set_audit_chain_service(svc)
    return svc


@pytest.fixture
def mock_opik():
    """Patch get_opik_connector to a no-op async context manager."""
    mock = MagicMock()
    mock.is_enabled = True
    span = MagicMock()
    span.span_id = "span_test"
    span.set_output = MagicMock()
    span.set_attribute = MagicMock()
    mock.trace_agent = MagicMock()
    mock.trace_agent.return_value.__aenter__ = AsyncMock(return_value=span)
    mock.trace_agent.return_value.__aexit__ = AsyncMock(return_value=None)
    with patch("src.agents.causal_impact.graph.get_opik_connector", return_value=mock):
        yield mock


@pytest.fixture
def base_state():
    return {
        "audit_workflow_id": uuid4(),
        "query": "Did dispatch X cause uplift?",
        "treatment_var": "spend",
        "outcome_var": "uplift",
        "current_phase": "estimation",
        "session_id": "sess-1",
        "user_id": "user-1",
        "brand": "BrandX",
        "query_id": "qid-1",
        "span_id": None,
        "dispatch_id": "dispatch-1",
    }


# ---------------------------------------------------------------------------
# Regression cases — call-shape assertions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traced_node_passes_input_data_not_input_hash(
    mock_audit_service, mock_opik, base_state
):
    """add_entry must receive ``input_data`` (in signature), not ``input_hash``."""

    @traced_node("estimation")
    async def _node(state):
        return {"status": "ok", "current_phase": "done"}

    await _node(base_state)

    assert mock_audit_service.add_entry.called, (
        "add_entry should be called when workflow_id and audit_service present"
    )
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert "input_data" in kwargs, (
        f"add_entry must use input_data (matches signature); got kwargs={sorted(kwargs)}"
    )
    assert "input_hash" not in kwargs, "input_hash is not in add_entry signature; remove it"


@pytest.mark.asyncio
async def test_traced_node_passes_output_data_not_output_hash(
    mock_audit_service, mock_opik, base_state
):
    """add_entry must receive ``output_data`` (in signature), not ``output_hash``."""

    @traced_node("estimation")
    async def _node(state):
        return {"status": "ok", "current_phase": "done"}

    await _node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert "output_data" in kwargs, (
        f"add_entry must use output_data (matches signature); got kwargs={sorted(kwargs)}"
    )
    assert "output_hash" not in kwargs, "output_hash is not in add_entry signature; remove it"


@pytest.mark.asyncio
async def test_traced_node_passes_agent_tier_as_enum_not_string(
    mock_audit_service, mock_opik, base_state
):
    """add_entry's ``agent_tier`` parameter is typed AgentTier, not its .value."""

    @traced_node("estimation")
    async def _node(state):
        return {"status": "ok", "current_phase": "done"}

    await _node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert "agent_tier" in kwargs
    assert isinstance(kwargs["agent_tier"], AgentTier), (
        f"agent_tier must be AgentTier enum, not {type(kwargs['agent_tier']).__name__}; "
        f"got value={kwargs['agent_tier']!r}"
    )
    assert kwargs["agent_tier"] is AgentTier.CAUSAL_ANALYTICS


@pytest.mark.asyncio
async def test_traced_node_does_not_pass_user_id_session_id_brand(
    mock_audit_service, mock_opik, base_state
):
    """user_id/session_id/brand are NOT in add_entry's signature — they are
    inherited from the workflow's genesis entry. Passing them caused TypeError
    pre-fix (silenced by ``# type: ignore[call-arg]`` and try/except)."""

    @traced_node("estimation")
    async def _node(state):
        return {"status": "ok", "current_phase": "done"}

    await _node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    for forbidden in ("user_id", "session_id", "brand"):
        assert forbidden not in kwargs, (
            f"{forbidden!r} is not in add_entry signature; remove it "
            f"(inherited from genesis entry via previous.{forbidden})"
        )


@pytest.mark.asyncio
async def test_traced_node_refutation_wraps_dict_in_refutation_results_dataclass(
    mock_audit_service, mock_opik, base_state
):
    """The refutation node persists ``refutation_results`` as a *dict* (via
    :py:meth:`RefutationSuite.to_legacy_format`). But
    :py:meth:`AuditChainService.add_entry` calls
    ``refutation_results.to_dict()`` on the kwarg unconditionally
    (``audit_chain.py:345``). Passing a raw dict would raise
    ``AttributeError: 'dict' object has no attribute 'to_dict'`` — caught by
    the same ``try/except Exception`` that hid the kwarg-mismatch bug, so the
    refutation node's audit entry would still be silently dropped post the
    kwarg fix unless we construct a :class:`RefutationResults` dataclass here.

    This pre-existing dict-vs-dataclass mismatch is exposed by the kwarg fix:
    pre-fix the call failed with TypeError on ``input_hash`` *before* Python
    entered ``add_entry``'s body and reached ``.to_dict()``, so the
    AttributeError was unreachable.
    """

    @traced_node("refutation")
    async def _node(state):
        return {
            "status": "ok",
            "current_phase": "done",
            "refutation_results": {
                "tests_passed": 3,
                "tests_failed": 1,
                "total_tests": 4,
                "overall_robust": False,
                "gate_decision": "BLOCK",
                "individual_tests": {
                    "placebo_treatment": {"passed": True},
                    "random_common_cause": {"passed": True},
                    "data_subset": {"passed": True},
                    "unobserved_common_cause": {"passed": False},
                },
            },
        }

    await _node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert "refutation_results" in kwargs
    rr = kwargs["refutation_results"]
    assert isinstance(rr, RefutationResults), (
        "refutation_results must be a RefutationResults dataclass — "
        "add_entry calls .to_dict() on it. "
        f"Got {type(rr).__name__}={rr!r}"
    )
    # Field mapping spot-check (mirrors audit_chain_mixin.audited_traced_node)
    assert rr.placebo_treatment is True
    assert rr.random_common_cause is True
    assert rr.data_subset is True
    # unobserved_common_cause (dict key) maps to unobserved_confound (dataclass field)
    assert rr.unobserved_confound is False


@pytest.mark.asyncio
async def test_traced_node_refutation_wraps_bootstrap_test_in_dataclass(
    mock_audit_service, mock_opik, base_state
):
    """Issue #368 regression: ``individual_tests["bootstrap"]["passed"]`` must
    round-trip into the ``RefutationResults`` dataclass as ``bootstrap``.

    Pre-#368: ``RefutationSuite.to_legacy_format()`` emits 5 ``individual_tests``
    keys (placebo_treatment, random_common_cause, data_subset,
    unobserved_common_cause, bootstrap), but the receiving dataclass only
    declared 4 fields — bootstrap was silently dropped at the
    ``RefutationResults(...)`` call site in ``graph.py:155-166``. Bootstrap
    is the only refutation that runs in degraded DoWhy mode (causal_model
    None) — dropping it left tamper-evident logging with no record of the
    only test that actually ran.

    Post-#368: a 5th ``bootstrap`` field exists and the wrap site forwards
    ``individual.get("bootstrap", {}).get("passed")`` into it.
    """

    @traced_node("refutation")
    async def _node(state):
        return {
            "status": "ok",
            "current_phase": "done",
            "refutation_results": {
                "tests_passed": 4,
                "tests_failed": 1,
                "total_tests": 5,
                "overall_robust": False,
                "gate_decision": "BLOCK",
                "individual_tests": {
                    "placebo_treatment": {"passed": True},
                    "random_common_cause": {"passed": True},
                    "data_subset": {"passed": True},
                    "unobserved_common_cause": {"passed": False},
                    "bootstrap": {"passed": True},
                },
            },
        }

    await _node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert "refutation_results" in kwargs
    rr = kwargs["refutation_results"]
    assert isinstance(rr, RefutationResults), (
        f"refutation_results must be a RefutationResults dataclass. Got {type(rr).__name__}={rr!r}"
    )
    assert rr.placebo_treatment is True
    assert rr.random_common_cause is True
    assert rr.data_subset is True
    assert rr.unobserved_confound is False
    # Issue #368: bootstrap key from individual_tests must populate the
    # dataclass field — pre-fix, this would be silently dropped.
    assert rr.bootstrap is True, (
        f"Issue #368: bootstrap field was silently dropped (got rr.bootstrap="
        f"{rr.bootstrap!r}). The to_legacy_format() dict has a 'bootstrap' "
        f"key, but the wrap site in graph.py:155-166 did not forward it to "
        f"the dataclass. Add bootstrap=individual.get('bootstrap', {{}}).get('passed')."
    )
    # Round-trip via to_dict() must preserve bootstrap (this is what
    # AuditChainService.add_entry persists to the database).
    persisted = rr.to_dict()
    assert "bootstrap" in persisted, (
        f"Issue #368: to_dict() must include bootstrap key. Got keys={sorted(persisted)}."
    )
    assert persisted["bootstrap"] is True


@pytest.mark.asyncio
async def test_traced_node_refutation_empty_ref_leaves_refutation_results_none(
    mock_audit_service, mock_opik, base_state
):
    """When the refutation node fails early (no ``individual_tests`` in
    state), we leave ``refutation_results=None`` instead of constructing
    ``RefutationResults(None, None, None, None, None)``. This preserves the
    semantic difference between "no refutation ran" and "all tests
    returned null" in the audit chain.

    Both ralph-iter-2 and codex-iter-2 independently surfaced this guard.
    """

    @traced_node("refutation")
    async def _node(state):
        # Refutation failed early — no refutation_results field at all
        return {
            "status": "failed",
            "current_phase": "failed",
            "refutation_error": "early failure",
        }

    await _node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert "refutation_results" in kwargs
    assert kwargs["refutation_results"] is None, (
        "Empty refutation result must propagate as None — "
        "not RefutationResults(None,None,None,None,None). "
        f"Got {kwargs['refutation_results']!r}"
    )


@pytest.mark.asyncio
async def test_traced_node_audit_entry_does_not_warning_log_failure(mock_opik, base_state, caplog):
    """End-to-end shape check: with a real :class:`AuditChainService` autospec,
    the add_entry call must succeed without exception. Pre-fix, the TypeError
    from bad kwargs was caught by ``except Exception`` and surfaced as
    ``logger.warning('Failed to record audit entry: ...')``. After fix, that
    warning must NOT appear."""
    from unittest.mock import create_autospec

    svc = create_autospec(AuditChainService, instance=True)
    set_audit_chain_service(svc)

    @traced_node("estimation")
    async def _node(state):
        return {"status": "ok", "current_phase": "done"}

    with caplog.at_level(logging.WARNING, logger="src.agents.causal_impact.graph"):
        await _node(base_state)

    failure_warnings = [
        r for r in caplog.records if "Failed to record audit entry" in r.getMessage()
    ]
    assert not failure_warnings, (
        "add_entry raised an exception that was caught & warning-logged — "
        "indicates kwarg/signature mismatch is still present. "
        f"Warnings: {[r.getMessage() for r in failure_warnings]}"
    )


# ---------------------------------------------------------------------------
# Fail-closed outcomes must be readable from the audit row (codex iter-3,
# 2026-09-06): this local wrapper is what the PRODUCTION causal_impact graph
# uses, and it always recorded ``action_type=node_name`` — a fail-closed
# estimation (``estimation_error`` + status failed) or a raising node left no
# ``<node>_error`` row, the one marker the /system-health and /analytics
# readers count (``src.api.utils.audit_outcomes``).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traced_node_fail_closed_result_records_error_action(
    mock_audit_service, mock_opik, base_state
):
    @traced_node("estimation")
    async def _node(state):
        return {
            **state,
            "estimation_error": "no estimator",
            "status": "failed",
            "errors": [{"phase": "estimation", "message": "no estimator"}],
        }

    result = await _node(base_state)

    assert result["estimation_error"] == "no estimator"
    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "estimation_error"
    assert kwargs["validation_passed"] is False


@pytest.mark.asyncio
async def test_traced_node_raising_node_records_error_action_then_reraises(
    mock_audit_service, mock_opik, base_state
):
    @traced_node("sensitivity")
    async def _node(state):
        raise RuntimeError("dowhy exploded")

    with pytest.raises(RuntimeError, match="dowhy exploded"):
        await _node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "sensitivity_error"
    assert kwargs["validation_passed"] is False
    assert kwargs["agent_name"] == "causal_impact"
    assert kwargs["agent_tier"] is AgentTier.CAUSAL_ANALYTICS


@pytest.mark.asyncio
async def test_traced_node_non_robust_refutation_is_not_an_error_action(
    mock_audit_service, mock_opik, base_state
):
    """The refutation VERDICT (overall_robust False) is validation_passed=False
    on a normal ``refutation`` row, never an error action."""

    @traced_node("refutation")
    async def _node(state):
        return {
            **state,
            "status": "completed",
            "refutation_results": {"overall_robust": False, "individual_tests": {}},
        }

    await _node(base_state)

    kwargs = mock_audit_service.add_entry.call_args.kwargs
    assert kwargs["action_type"] == "refutation"
    assert kwargs["validation_passed"] is False
