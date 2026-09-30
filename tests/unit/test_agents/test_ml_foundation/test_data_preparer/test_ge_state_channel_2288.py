"""Regression for #2288: Great Expectations' verdict must reach graph state.

``run_ge_validation`` returns ``ge_validation_status``, ``ge_validation_results``,
``ge_expectations_evaluated``, ``ge_expectations_passed`` and ``ge_success_rate``
(plus ``ge_validation_note`` / ``ge_validation_reason`` / ``ge_validation_error``
on its side paths). ``DataPreparerState`` is ``extra="ignore"`` (#2298), so a key
a node RETURNS that is not declared on the state is dropped at the channel
boundary. None of the ``ge_*`` keys was declared, so the node ran, logged its
verdict, and the verdict was discarded.

These tests drive the COMPILED graph over the REAL ``DataPreparerState``. A
node-level assertion on ``run_ge_validation``'s return dict is a false green
here: the return value was always correct; it is the channel that lost it.

Scope: this pins that the verdict SURVIVES. It deliberately does not make the
QC gate read ``ge_validation_status`` — GE's blocking verdict already reaches
the gate through ``blocking_issues`` (kind ``ge_validation``), which #2283
made durable. Data is read from a local CSV through ``load_data``'s file path;
nothing is mocked and nothing touches Supabase, Redis or MLflow.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
from langgraph.graph import END, StateGraph

from src.agents.ml_foundation.data_preparer.graph import finalize_output
from src.agents.ml_foundation.data_preparer.nodes import ge_validator
from src.agents.ml_foundation.data_preparer.nodes.data_loader import load_data
from src.agents.ml_foundation.data_preparer.nodes.ge_validator import run_ge_validation
from src.agents.ml_foundation.data_preparer.nodes.quality_checker import run_quality_checks
from src.agents.ml_foundation.data_preparer.state import DataPreparerState

_AUDIT_WORKFLOW_ID = "00000000-0000-0000-0000-000000002288"


def _write_csv(tmp_path: Path) -> Path:
    n = 60
    dates = pd.date_range("2099-01-01", periods=n, freq="D")
    frame = pd.DataFrame(
        {
            "patient_id": [f"pat-{i:04d}" for i in range(n)],
            "event_type": ["prescription"] * n,
            "event_date": dates.strftime("%Y-%m-%d"),
            "created_at": dates.strftime("%Y-%m-%d"),
            "days_on_therapy": [30 + (i % 15) for i in range(n)],
            "data_split": (["train"] * 36 + ["validation"] * 12 + ["test"] * 6 + ["holdout"] * 6),
        }
    )
    path = tmp_path / "patient_journeys.csv"
    frame.to_csv(path, index=False)
    return path


def _base_state(csv_path: Path) -> Dict[str, Any]:
    return {
        "audit_workflow_id": _AUDIT_WORKFLOW_ID,
        "experiment_id": "exp-2288",
        "data_source": {"type": "files", "paths": {"patient_journeys": str(csv_path)}},
        "scope_spec": {"date_column": "event_date", "data_source": "patient_journeys"},
    }


def _production_seam():
    """``load_data -> run_quality_checks -> run_ge_validation -> finalize_output``."""
    graph = StateGraph(DataPreparerState)
    graph.add_node("load_data", load_data)  # type: ignore[arg-type]
    graph.add_node("run_quality_checks", run_quality_checks)  # type: ignore[arg-type]
    graph.add_node("run_ge_validation", run_ge_validation)  # type: ignore[arg-type]
    graph.add_node("finalize_output", finalize_output)  # type: ignore[arg-type]
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "run_quality_checks")
    graph.add_edge("run_quality_checks", "run_ge_validation")
    graph.add_edge("run_ge_validation", "finalize_output")
    graph.add_edge("finalize_output", END)
    return graph.compile()


def _ge_only():
    graph = StateGraph(DataPreparerState)
    graph.add_node("run_ge_validation", run_ge_validation)  # type: ignore[arg-type]
    graph.set_entry_point("run_ge_validation")
    graph.add_edge("run_ge_validation", END)
    return graph.compile()


@pytest.mark.asyncio
async def test_ge_verdict_survives_to_final_state(tmp_path: Path) -> None:
    """RED pre-fix: every ``ge_*`` key is absent from the final state."""
    final_state = await _production_seam().ainvoke(_base_state(_write_csv(tmp_path)))

    assert final_state.get("ge_validation_status") == "passed", (
        f"GE verdict dropped at the channel boundary; "
        f"ge keys present={sorted(k for k in final_state if k.startswith('ge_'))}"
    )
    evaluated = final_state["ge_expectations_evaluated"]
    passed = final_state["ge_expectations_passed"]
    assert isinstance(evaluated, int) and evaluated > 0
    assert passed == evaluated
    assert final_state["ge_success_rate"] == pytest.approx(1.0)
    # One result dict per validated split (train / validation / test).
    results = final_state["ge_validation_results"]
    assert len(results) == 3
    assert all(isinstance(r, dict) for r in results)


@pytest.mark.asyncio
async def test_ge_skipped_reason_survives(tmp_path: Path) -> None:
    """The no-training-data side path returns ``ge_validation_reason``."""
    final_state = await _ge_only().ainvoke(
        {"audit_workflow_id": _AUDIT_WORKFLOW_ID, "experiment_id": "exp-2288-skip"}
    )
    assert final_state.get("ge_validation_status") == "skipped"
    assert final_state.get("ge_validation_reason") == "No training data available"


@pytest.mark.asyncio
async def test_ge_error_detail_survives(tmp_path: Path) -> None:
    """The unknown-``data_source``-shape path returns ``ge_validation_error``."""
    frame = pd.DataFrame({"patient_id": ["p1", "p2"], "event_type": ["rx", "rx"]})
    final_state = await _ge_only().ainvoke(
        {
            "audit_workflow_id": _AUDIT_WORKFLOW_ID,
            "experiment_id": "exp-2288-error",
            "train_df": frame,
            "data_source": {"type": "mystery"},
        }
    )
    assert final_state.get("ge_validation_status") == "error"
    assert "Unknown data_source dict shape" in (final_state.get("ge_validation_error") or "")
    assert any(i.startswith("ge_validation: ") for i in final_state["blocking_issues"])


def test_every_ge_key_the_node_returns_is_declared_on_the_state() -> None:
    """Static guard: a ``ge_*`` key added to the node later must be declared too,
    or it is silently dropped exactly as all of them were (#2288)."""
    source = Path(ge_validator.__file__).read_text()
    returned = set(re.findall(r'"(ge_[a-z_]+)"', source))
    assert returned, "found no ge_* keys in ge_validator.py — the guard is vacuous"
    undeclared = sorted(returned - set(DataPreparerState.model_fields))
    assert not undeclared, f"ge_* keys returned but undeclared on DataPreparerState: {undeclared}"
