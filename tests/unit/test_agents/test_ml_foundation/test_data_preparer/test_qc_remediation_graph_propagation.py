"""Graph-level regression for #632: a ``drop_column`` remediation must
reach the revalidation pass through LangGraph state.

The node-level tests in ``test_qc_remediation.py`` assert that
``review_and_remediate_qc`` *returns* the remediated frames. This file
goes one level deeper: it wires the REAL two-node seam the bug lives on —
``qc_remediation`` -> (retry edge) -> ``run_quality_checks`` — inside a
minimal ``StateGraph(DataPreparerState)`` and invokes it end-to-end.

Why a graph-level test matters here (and not just the node test): the
failure mode is a LangGraph *state hand-off*. LangGraph applies a node's
returned dict onto the channel state, then passes the merged state to the
next node. If a node omits a key it mutated only by rebinding a local
(``train_df = train_df.drop(...)``), that mutation never lands in the
channel state, so the downstream node reads the stale frame. Additionally,
LangGraph silently DROPS keys not declared on the state schema — so this
test also proves ``train_df`` is a declared channel that survives the hop.

We assert the OBSERVABLE downstream effect: ``run_quality_checks``
recomputes ``column_count`` from ``state["train_df"]``; after a successful
``drop_column``, that count must reflect the dropped column.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
from langgraph.graph import END, StateGraph

from src.agents.ml_foundation.data_preparer.nodes.qc_remediation import (
    review_and_remediate_qc,
)
from src.agents.ml_foundation.data_preparer.nodes.quality_checker import (
    run_quality_checks,
)
from src.agents.ml_foundation.data_preparer.state import DataPreparerState

_ANALYZE_LLM = (
    "src.agents.ml_foundation.data_preparer.nodes.qc_remediation._analyze_qc_failures_with_llm"
)


def _build_two_node_graph():
    """Compile the minimal ``qc_remediation`` -> ``run_quality_checks``
    seam using the real nodes and the real retry edge from the production
    graph (``graph.py`` L370-376)."""
    graph = StateGraph(DataPreparerState)
    graph.add_node("qc_remediation", review_and_remediate_qc)  # type: ignore[arg-type]
    graph.add_node("run_quality_checks", run_quality_checks)  # type: ignore[arg-type]
    graph.set_entry_point("qc_remediation")
    graph.add_edge("qc_remediation", "run_quality_checks")
    graph.add_edge("run_quality_checks", END)
    return graph.compile()


@pytest.mark.asyncio
async def test_drop_column_reaches_revalidation_pass() -> None:
    """RED pre-fix: ``qc_remediation`` omits ``train_df`` from its return,
    so the channel still holds the ORIGINAL 2-column frame; the
    revalidation ``column_count`` is 2. Post-fix the dropped column
    propagates and ``column_count`` is 1."""
    train_df = pd.DataFrame({"keep": [1, 2, 3], "drop_me": [4, 5, 6]})
    state: dict = {
        "audit_workflow_id": "00000000-0000-0000-0000-000000000632",
        "experiment_id": "exp-632-graph",
        "qc_status": "failed",
        "gate_passed": False,
        "overall_score": 0.5,
        "remediation_attempts": 0,
        "train_df": train_df,
        "validation_df": None,
        "test_df": None,
        # No required/unique columns so run_quality_checks doesn't block
        # for unrelated reasons; we only care about the column hand-off.
        "scope_spec": {},
    }
    analysis = {
        "can_auto_remediate": True,
        "remediation_actions": [{"type": "drop_column", "column": "drop_me", "params": {}}],
        "root_cause_summary": "test-injected",
    }

    app = _build_two_node_graph()
    with patch(_ANALYZE_LLM, new=AsyncMock(return_value=analysis)):
        final_state = await app.ainvoke(state)

    # The remediation ran.
    assert final_state["remediation_status"] == "applied"
    assert "Dropped column: drop_me" in final_state["remediation_actions_taken"]
    # The revalidation pass saw the REMEDIATED frame (1 column, not 2).
    assert final_state["column_count"] == 1, (
        "run_quality_checks revalidated the stale frame — drop_column did "
        f"not propagate through graph state (#632). column_count="
        f"{final_state['column_count']}"
    )
    assert "drop_me" not in final_state["train_df"].columns


@pytest.mark.asyncio
async def test_deduplicate_reaches_revalidation_pass() -> None:
    """The row reduction from ``deduplicate`` must be visible to the
    revalidation pass via ``run_quality_checks``' recomputed
    ``row_count``."""
    train_df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [9, 9, 8, 8]})
    state: dict = {
        "audit_workflow_id": "00000000-0000-0000-0000-0000006320d2",
        "experiment_id": "exp-632-graph-dedup",
        "qc_status": "failed",
        "gate_passed": False,
        "overall_score": 0.5,
        "remediation_attempts": 0,
        "train_df": train_df,
        "validation_df": None,
        "test_df": None,
        "scope_spec": {},
    }
    analysis = {
        "can_auto_remediate": True,
        "remediation_actions": [{"type": "deduplicate", "column": None, "params": {}}],
        "root_cause_summary": "test-injected",
    }

    app = _build_two_node_graph()
    with patch(_ANALYZE_LLM, new=AsyncMock(return_value=analysis)):
        final_state = await app.ainvoke(state)

    assert final_state["remediation_status"] == "applied"
    assert final_state["row_count"] == 2, (
        "run_quality_checks revalidated the stale frame — deduplicate did "
        f"not propagate through graph state (#632). row_count="
        f"{final_state['row_count']}"
    )
