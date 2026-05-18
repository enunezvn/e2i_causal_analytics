"""Forcing tests for Phase 2 collider/mediator exclusion policy.

Plan: ``.claude/plans/causal_role_propagation_FINAL.md`` §2.4 (cases 1-12).

Each case has an explicit falsifiability target — revert the named
function body and the listed case trips. The case set covers:

* STRICT vs ADVISORY vs OFF policy paths (cases 1-4).
* C1 trust-gate for LLM verdicts (case 5).
* Manifest-source unconditional drop (case 6).
* StateGraph edge wiring (case 7).
* Audit-chain ``output_data=`` kwarg regression (case 8).
* Separated ``adjustment_set_hash`` refresh, NEVER touching
  ``dag_version_hash`` which is keyed by ``repositories/expert_review``
  (cases 9 + 10 + 11 — codex-2 B1 fix).
* Log cap keeps LAST N entries (codex-2 fix; case 12).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

from src.agents.causal_impact.nodes.adjustment_set_policy import (
    POLICY_LOG_CAP_DEFAULT,
    apply_adjustment_set_policy,
    apply_role_attributions,
)
from src.agents.causal_impact.state import CausalGraph
from src.causal_engine.dag_hash import (
    compute_adjustment_set_hash,
    compute_dag_hash,
)
from src.data.role_attribution import RoleAttribution
from src.utils.audit_chain import AuditChainService

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _graph(adjustment_sets: list[list[str]]) -> CausalGraph:
    """Construct a minimal CausalGraph with a pre-computed dag_version_hash.

    The dag_version_hash is computed via the unchanged
    ``compute_dag_hash`` (nodes + edges + treatment/outcome) and must
    remain stable across the policy node — that hash is the primary key
    used by ``src/repositories/expert_review.py`` (15 lookup sites).
    """

    nodes = sorted({n for s in adjustment_sets for n in s} | {"T", "Y"})
    edges = [("T", "Y")]
    graph: CausalGraph = {  # type: ignore[typeddict-item]
        "nodes": nodes,
        "edges": edges,
        "treatment_nodes": ["T"],
        "outcome_nodes": ["Y"],
        "adjustment_sets": adjustment_sets,
        "dag_dot": "",
        "confidence": 1.0,
    }
    graph["dag_version_hash"] = compute_dag_hash(causal_graph=dict(graph))
    return graph


def _llm_attr(feature: str, causal_role: str, *, satisfied: bool = True) -> RoleAttribution:
    return RoleAttribution(
        feature=feature,
        causal_role=causal_role,
        source="llm",
        evaluator_satisfied=satisfied,
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )


def _manifest_attr(feature: str, causal_role: str) -> RoleAttribution:
    return RoleAttribution(
        feature=feature,
        causal_role=causal_role,
        source="manifest",
        evaluator_satisfied=True,
        evaluator_model="n/a",
    )


# --------------------------------------------------------------------------
# Case 1: STRICT + LLM collider satisfied → dropped
# --------------------------------------------------------------------------


def test_case1_strict_llm_collider_satisfied_dropped() -> None:
    graph = _graph([["X", "Y_conf", "C"]])
    attrs = [_llm_attr("C", "collider", satisfied=True)]

    out_graph, log, mutated = apply_role_attributions(graph, attrs, policy="STRICT")

    assert mutated is True
    assert out_graph["adjustment_sets"] == [["X", "Y_conf"]]
    assert len(log) == 1
    assert log[0]["kind"] == "dropped_collider"
    assert log[0]["feature"] == "C"
    assert log[0]["source"] == "llm"


# --------------------------------------------------------------------------
# Case 2: STRICT + LLM mediator satisfied → dropped
# --------------------------------------------------------------------------


def test_case2_strict_llm_mediator_satisfied_dropped() -> None:
    graph = _graph([["X", "Y_conf", "M"]])
    attrs = [_llm_attr("M", "mediator", satisfied=True)]

    out_graph, log, mutated = apply_role_attributions(graph, attrs, policy="STRICT")

    assert mutated is True
    assert out_graph["adjustment_sets"] == [["X", "Y_conf"]]
    assert len(log) == 1
    assert log[0]["kind"] == "dropped_mediator"
    assert log[0]["feature"] == "M"


# --------------------------------------------------------------------------
# Case 3: ADVISORY + LLM collider satisfied → kept, warning logged
# --------------------------------------------------------------------------


def test_case3_advisory_llm_collider_warns_not_dropped() -> None:
    graph = _graph([["X", "Y_conf", "C"]])
    pre_sets = [list(s) for s in graph["adjustment_sets"]]
    attrs = [_llm_attr("C", "collider", satisfied=True)]

    out_graph, log, mutated = apply_role_attributions(graph, attrs, policy="ADVISORY")

    assert mutated is False
    assert out_graph["adjustment_sets"] == pre_sets
    assert len(log) == 1
    assert log[0]["kind"] == "warning_collider"


# --------------------------------------------------------------------------
# Case 4: OFF + any input → no-op, empty log
# --------------------------------------------------------------------------


def test_case4_off_policy_is_noop_empty_log() -> None:
    graph = _graph([["X", "C"]])
    pre_sets = [list(s) for s in graph["adjustment_sets"]]
    attrs = [
        _llm_attr("C", "collider", satisfied=True),
        _manifest_attr("X", "mediator"),  # would normally drop in STRICT
    ]

    out_graph, log, mutated = apply_role_attributions(graph, attrs, policy="OFF")

    assert mutated is False
    assert out_graph["adjustment_sets"] == pre_sets
    assert log == []


# --------------------------------------------------------------------------
# Case 5: STRICT + LLM collider UNsatisfied → kept (C1 trust gate)
# --------------------------------------------------------------------------


def test_case5_strict_llm_unsatisfied_collider_kept() -> None:
    graph = _graph([["X", "C"]])
    pre_sets = [list(s) for s in graph["adjustment_sets"]]
    attrs = [_llm_attr("C", "collider", satisfied=False)]

    out_graph, log, mutated = apply_role_attributions(graph, attrs, policy="STRICT")

    # C1: LLM source with satisfied=False does NOT act — collider stays.
    assert mutated is False
    assert out_graph["adjustment_sets"] == pre_sets
    assert log == []


# --------------------------------------------------------------------------
# Case 6: STRICT + manifest-source collider → dropped (manifest always acts)
# --------------------------------------------------------------------------


def test_case6_strict_manifest_collider_dropped() -> None:
    graph = _graph([["X", "Y_conf", "C"]])
    attrs = [_manifest_attr("C", "collider")]

    out_graph, log, mutated = apply_role_attributions(graph, attrs, policy="STRICT")

    assert mutated is True
    assert out_graph["adjustment_sets"] == [["X", "Y_conf"]]
    assert len(log) == 1
    assert log[0]["kind"] == "dropped_collider"
    assert log[0]["source"] == "manifest"


# --------------------------------------------------------------------------
# Case 7: StateGraph edge wiring — graph_builder → adjustment_set_policy → estimation
# --------------------------------------------------------------------------


def test_case7_stategraph_edge_wiring() -> None:
    from src.agents.causal_impact.graph import create_causal_impact_graph

    compiled = create_causal_impact_graph()
    graph_repr = compiled.get_graph()

    node_ids = {n.id for n in graph_repr.nodes.values()}
    assert "adjustment_set_policy" in node_ids

    edge_pairs = {(e.source, e.target) for e in graph_repr.edges}
    assert ("graph_builder", "adjustment_set_policy") in edge_pairs
    assert ("adjustment_set_policy", "estimation") in edge_pairs


# --------------------------------------------------------------------------
# Case 8: audit-chain regression — output_data= kwarg used, not output_hash
# --------------------------------------------------------------------------


async def test_case8_audit_chain_uses_output_data_kwarg() -> None:
    """The traced node MUST pass ``output_data=`` to ``add_entry`` so the
    audit service does its own hashing. The pre-existing
    ``output_hash=...`` call at ``graph.py:173-180`` is a bug — the
    audit service signature does NOT accept ``output_hash``. The Phase 2
    node uses the correct kwarg from day one.
    """

    mock_service = MagicMock(spec=AuditChainService)
    workflow_id = uuid4()

    from src.agents.causal_impact import graph as graph_mod
    from src.agents.causal_impact.nodes import adjustment_set_policy as mod

    state: dict[str, Any] = {
        "query": "q",
        "query_id": "qid",
        "audit_workflow_id": workflow_id,
        "causal_graph": _graph([["X", "C"]]),
        "role_attributions": [_llm_attr("C", "collider", satisfied=True)],
        "errors": [],
        "warnings": [],
    }

    # Monkeypatch the audit service getter at the call site used by the
    # traced wrapper. Both the wrapper module and the node module must
    # see the mock — the traced wrapper is in graph.py and may call
    # get_audit_chain_service before delegating into the node.
    original_getter = graph_mod.get_audit_chain_service
    graph_mod.get_audit_chain_service = lambda: mock_service  # type: ignore[assignment]
    try:
        # Also patch the node-module copy if present.
        node_original = getattr(mod, "get_audit_chain_service", None)
        if node_original is not None:
            mod.get_audit_chain_service = lambda: mock_service  # type: ignore[assignment]

        traced = graph_mod.traced_apply_adjustment_policy
        await traced(state)  # type: ignore[arg-type]
    finally:
        graph_mod.get_audit_chain_service = original_getter  # type: ignore[assignment]
        if node_original is not None:  # type: ignore[possibly-undefined]
            mod.get_audit_chain_service = node_original  # type: ignore[assignment]

    assert mock_service.add_entry.called
    call_kwargs = mock_service.add_entry.call_args.kwargs
    assert "output_data" in call_kwargs, (
        "audit add_entry must be invoked with output_data= (not output_hash=)"
    )
    # output_data is a dict; sanity-check it carries the policy summary.
    payload = call_kwargs["output_data"]
    assert isinstance(payload, dict)
    assert "policy" in payload
    assert "mutated" in payload


# --------------------------------------------------------------------------
# Case 9: adjustment_set_hash refresh (codex-2 B1 fix)
# --------------------------------------------------------------------------


def test_case9_adjustment_set_hash_refresh_on_strict_drop() -> None:
    graph = _graph([["X", "Y_drop", "Z"]])
    attrs = [_llm_attr("Y_drop", "collider", satisfied=True)]

    out_graph, _log, mutated = apply_role_attributions(graph, attrs, policy="STRICT")

    assert mutated is True
    # adjustment_set_hash and adjustment_set_hash_pre_policy MUST diverge
    # when a drop occurs.
    pre = out_graph["adjustment_set_hash_pre_policy"]
    post = out_graph["adjustment_set_hash"]
    assert pre != post
    # The post hash equals the canonical hash of the post-drop set.
    assert post == compute_adjustment_set_hash([["X", "Z"]])
    assert pre == compute_adjustment_set_hash([["X", "Y_drop", "Z"]])


# --------------------------------------------------------------------------
# Case 10: adjustment_set_hash NO-OP under OFF policy
# --------------------------------------------------------------------------


def test_case10_adjustment_set_hash_stable_under_off() -> None:
    graph = _graph([["X", "Y_keep", "Z"]])
    attrs = [_llm_attr("Y_keep", "collider", satisfied=True)]

    out_graph, _log, mutated = apply_role_attributions(graph, attrs, policy="OFF")

    assert mutated is False
    assert out_graph["adjustment_set_hash"] == out_graph["adjustment_set_hash_pre_policy"]


# --------------------------------------------------------------------------
# Case 11: dag_version_hash stability (NEVER mutated by policy node)
# --------------------------------------------------------------------------


def test_case11_dag_version_hash_unchanged_after_policy() -> None:
    """expert_review repository (15 sites) keys lookups on
    ``dag_version_hash``. Mutating it mid-pipeline breaks every site.
    """

    graph = _graph([["X", "Y_drop", "Z"]])
    pre_dag_hash = graph["dag_version_hash"]
    attrs = [_llm_attr("Y_drop", "collider", satisfied=True)]

    out_graph, _log, mutated = apply_role_attributions(graph, attrs, policy="STRICT")

    assert mutated is True
    assert out_graph["dag_version_hash"] == pre_dag_hash


# --------------------------------------------------------------------------
# Case 12: log cap keeps LAST N entries (codex-2 fix)
# --------------------------------------------------------------------------


def test_case12_log_cap_keeps_last_n_drops() -> None:
    # 150 colliders in one giant adjustment set.
    features = [f"C{i}" for i in range(150)]
    graph = _graph([features + ["X"]])
    attrs = [_llm_attr(f, "collider", satisfied=True) for f in features]

    out_graph, log, mutated = apply_role_attributions(graph, attrs, policy="STRICT", log_cap=10)

    assert mutated is True
    assert len(log) == 10
    assert out_graph.get("policy_log_was_truncated") is True

    # Must contain LAST 10, not FIRST 10. The producer iterates features
    # in input order, so the last 10 features written to the log are
    # C140..C149.
    logged_features = [entry["feature"] for entry in log]
    assert logged_features == [f"C{i}" for i in range(140, 150)]


# --------------------------------------------------------------------------
# Bonus: POLICY_LOG_CAP_DEFAULT sanity
# --------------------------------------------------------------------------


def test_default_log_cap_is_100() -> None:
    assert POLICY_LOG_CAP_DEFAULT == 100


# --------------------------------------------------------------------------
# Bonus: apply_adjustment_set_policy node wrapper returns sanitized update
# --------------------------------------------------------------------------


async def test_node_wrapper_returns_state_update_with_policy_summary() -> None:
    """The state-update node (non-traced; what the traced wrapper
    invokes) must return a partial state dict suitable for LangGraph
    merge — ``causal_graph`` with new hashes, plus ``policy_log`` and
    ``policy_log_was_truncated``.
    """

    state: dict[str, Any] = {
        "causal_graph": _graph([["X", "C"]]),
        "role_attributions": [_llm_attr("C", "collider", satisfied=True)],
        "errors": [],
        "warnings": [],
    }

    update = await apply_adjustment_set_policy(state)  # type: ignore[arg-type]

    assert "causal_graph" in update
    assert "adjustment_set_hash" in update["causal_graph"]
    assert "adjustment_set_hash_pre_policy" in update["causal_graph"]
    assert "policy_log" in update
    assert "policy_log_was_truncated" in update
