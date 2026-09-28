"""Graph-level regressions for #2294: blocking conditions that never reached the
``blocking_issues`` channel, so ``finalize_output`` passed the QC gate.

#2283 / PR #2285 fixed the DESTRUCTION of entries already in the channel. These
tests pin the complementary producer-side gaps: a condition that should block
and was never written. Each one drives the REAL node(s) through a compiled
``StateGraph(DataPreparerState)`` that ends at the REAL ``finalize_output`` and
asserts ``gate_passed`` — asserting on a node's return dict alone cannot see
the gate (see ``test_blocking_issues_channel_2283.py``'s docstring).

Upstream QC is seeded on the initial state (``qc_status="passed"``,
``overall_score=0.95``) rather than by running ``run_quality_checks``, so the
only thing that can fail the gate in these graphs is the channel under test.
Where a crash path has to be triggered, it is either a real data shape that
crashes the check (preferred) or a ``monkeypatch`` that raises inside the
check's own dependency. Neither the node under test nor the gate is stubbed.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import END, StateGraph

from src.agents.ml_foundation.data_preparer.graph import finalize_output
from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import detect_leakage
from src.agents.ml_foundation.data_preparer.state import DataPreparerState

# Upstream QC verdict every graph below starts from: a clean pass, so a
# ``gate_passed=False`` can only come from ``blocking_issues``.
_CLEAN_UPSTREAM_QC: Dict[str, Any] = {
    "audit_workflow_id": "00000000-0000-0000-0000-000000002294",
    "qc_status": "passed",
    "overall_score": 0.95,
}


def _leakage_gate_graph():
    """``detect_leakage -> finalize_output`` over the real state schema."""
    graph = StateGraph(DataPreparerState)
    graph.add_node("detect_leakage", detect_leakage)  # type: ignore[arg-type]
    graph.add_node("finalize_output", finalize_output)  # type: ignore[arg-type]
    graph.set_entry_point("detect_leakage")
    graph.add_edge("detect_leakage", "finalize_output")
    graph.add_edge("finalize_output", END)
    return graph.compile()


def _moderate_only_frame(n: int = 400) -> pd.DataFrame:
    """A frame whose ONLY structured finding is a MODERATE target correlation.

    MODERATE findings are hard to produce alone: a feature correlated
    0.70-0.85 with a binary target normally also has single-feature AUC > 0.80
    (HIGH). Here 75% of positives carry a large spike and the rest sit slightly
    BELOW the negatives, which keeps Pearson r ~0.76 (MODERATE, p < 0.001) while
    rank AUC stays under 0.80 and the class ranges still overlap. Measured
    against every structural check: the only finding is
    ``target_correlation/moderate``.
    """
    rng = np.random.default_rng(2294)
    target = np.array([0, 1] * (n // 2))
    spend_spike = rng.standard_normal(n)
    positives = np.where(target == 1)[0]
    k = int(0.75 * len(positives))
    spend_spike[positives[:k]] += 60.0
    spend_spike[positives[k:]] += -1.5
    return pd.DataFrame({"spend_spike": spend_spike, "target": target})


@pytest.mark.asyncio
async def test_legacy_temporal_leak_blocks_alongside_a_moderate_finding() -> None:
    """Item 1. A legacy temporal leak must block even when a MODERATE
    structured finding coexists with it.

    ``f953304ea`` replaced "every leakage issue blocks" with a severity filter
    for STRUCTURED findings and kept legacy issues blocking through the
    predicate ``blocking_findings or (leakage_detected and not findings)`` —
    "leakage detected but no findings means it came from a legacy check". One
    MODERATE finding breaks that approximation and the temporal leak blocks
    nothing.
    """
    frame = _moderate_only_frame()
    n = len(frame)
    event = pd.date_range("2024-01-01", periods=n, freq="D")
    target_date = event + pd.Timedelta(days=30)
    # 40 rows whose event happens AFTER the label date: a real temporal leak.
    target_date = target_date.where(np.arange(n) >= 40, event - pd.Timedelta(days=1))
    frame["event_date"] = event.strftime("%Y-%m-%d")
    frame["target_date"] = target_date.strftime("%Y-%m-%d")

    state: Dict[str, Any] = {
        "experiment_id": "exp-2294-legacy-moderate",
        "train_df": frame,
        "scope_spec": {
            "prediction_target": "target",
            # ``check_target_leakage`` only scans ``required_features``.
            "required_features": ["spend_spike"],
            "event_date_column": "event_date",
            "target_date_column": "target_date",
        },
        **_CLEAN_UPSTREAM_QC,
    }

    final_state = await _leakage_gate_graph().ainvoke(state)

    # Preconditions: the temporal leak was found, and the structured findings
    # are exactly one MODERATE target correlation — the shape that failed open.
    assert any(i.startswith("Temporal leakage:") for i in final_state["leakage_issues"])
    findings = final_state["leakage_findings"]
    assert [(f["check_name"], f["severity"]) for f in findings] == [
        ("target_correlation", "moderate")
    ], findings

    blocking = final_state["blocking_issues"] or []
    assert any(i.startswith("leakage: Temporal leakage:") for i in blocking), (
        f"legacy temporal leak never reached the channel: {blocking!r}"
    )
    # The MODERATE finding itself is review-only and must NOT block.
    assert not any("target_correlation" in i for i in blocking), blocking
    assert final_state["gate_passed"] is False
