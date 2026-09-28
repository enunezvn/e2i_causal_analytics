"""Regression for #2292: a required column with nulls must be BLOCKED by the QC
gate, not crash the graph.

``run_quality_checks`` records each blocking completeness failure by appending
the structured expectation result (a dict carrying ``expectation_type``,
``column``, ``severity`` and ``result``) to ``failed_expectations``.
``DataPreparerState`` declared that channel ``Optional[List[str]]``, so the
dict was rejected with a pydantic ``ValidationError`` when LangGraph built the
NEXT node's input — the run died on exactly the data the gate exists to block.

The same value is then handed across agents: ``agent.py`` copies it into the
``qc_report`` output, and ``model_trainer`` / ``model_selector`` validate that
report against ``QCReportSchema``, which carried the same ``List[str]``
declaration. Fixing only the state would have moved the crash one agent
downstream, so both boundaries are pinned here.

The chain is the REAL production seam from ``graph.py``
(``load_data -> run_quality_checks -> run_ge_validation -> finalize_output``)
compiled over the REAL ``DataPreparerState``. Data is read from a local CSV
through ``load_data``'s file path; nothing is mocked and nothing touches
Supabase, Redis or MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
from langgraph.graph import END, StateGraph

from src.agents.ml_foundation.data_preparer.graph import finalize_output
from src.agents.ml_foundation.data_preparer.nodes.data_loader import load_data
from src.agents.ml_foundation.data_preparer.nodes.ge_validator import run_ge_validation
from src.agents.ml_foundation.data_preparer.nodes.quality_checker import run_quality_checks
from src.agents.ml_foundation.data_preparer.schemas import QCReportSchema
from src.agents.ml_foundation.data_preparer.state import DataPreparerState
from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

_AUDIT_WORKFLOW_ID = "00000000-0000-0000-0000-000000002292"
_REQUIRED_COLUMN = "days_on_therapy"


def _build_qc_chain():
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


def _write_csv_with_required_nulls(tmp_path: Path) -> Path:
    """Every 4th row of the REQUIRED column is null, which takes the
    ``severity == "blocking"`` branch of ``_check_completeness``."""
    n = 60
    dates = pd.date_range("2099-01-01", periods=n, freq="D")
    frame = pd.DataFrame(
        {
            "patient_id": [f"pat-{i:04d}" for i in range(n)],
            "event_type": ["prescription"] * n,
            "event_date": dates.strftime("%Y-%m-%d"),
            "created_at": dates.strftime("%Y-%m-%d"),
            _REQUIRED_COLUMN: [None if i % 4 == 0 else 30 + (i % 15) for i in range(n)],
            "data_split": (["train"] * 36 + ["validation"] * 12 + ["test"] * 6 + ["holdout"] * 6),
        }
    )
    path = tmp_path / "patient_journeys.csv"
    frame.to_csv(path, index=False)
    return path


def _base_state(csv_path: Path) -> Dict[str, Any]:
    return {
        "audit_workflow_id": _AUDIT_WORKFLOW_ID,
        "experiment_id": "exp-2292",
        "data_source": {"type": "files", "paths": {"patient_journeys": str(csv_path)}},
        "scope_spec": {
            "date_column": "event_date",
            "data_source": "patient_journeys",
            "required_columns": [_REQUIRED_COLUMN],
        },
    }


@pytest.mark.asyncio
async def test_required_column_nulls_are_blocked_not_crashed(tmp_path: Path) -> None:
    """RED pre-fix: ``ValidationError: failed_expectations.0 Input should be a
    valid string`` raised by ``ainvoke``. Post-fix the run reaches the gate and
    the gate blocks, naming the column."""
    final_state = await _build_qc_chain().ainvoke(
        _base_state(_write_csv_with_required_nulls(tmp_path))
    )

    failed = final_state["failed_expectations"]
    assert len(failed) == 1
    entry = failed[0]
    # The structured result survives the channel intact — the column and
    # severity are what the remediation prompt and the persisted report use.
    assert entry["expectation_type"] == "expect_column_values_to_not_be_null"
    assert entry["column"] == _REQUIRED_COLUMN
    assert entry["severity"] == "blocking"
    assert entry["result"]["null_count"] == 9  # rows 0, 4, ..., 32 of the 36 train rows

    assert final_state["qc_status"] == "failed"
    assert final_state["gate_passed"] is False
    assert any(
        f"Critical missing values in column: {_REQUIRED_COLUMN}" in issue
        for issue in final_state["blocking_issues"]
    )


@pytest.mark.asyncio
async def test_failed_expectations_cross_the_qc_report_contract(tmp_path: Path) -> None:
    """The value ``agent.py`` copies into ``qc_report`` must validate at the
    downstream agents' boundary. RED pre-fix (after a state-only fix):
    ``ModelTrainerState`` rejects ``qc_report.failed_expectations.0``."""
    final_state = await _build_qc_chain().ainvoke(
        _base_state(_write_csv_with_required_nulls(tmp_path))
    )
    failed = final_state["failed_expectations"]

    report = QCReportSchema.model_validate(
        {"status": final_state["qc_status"], "failed_expectations": failed}
    )
    assert report.failed_expectations == failed

    trainer_state = ModelTrainerState.model_validate(
        {
            "audit_workflow_id": _AUDIT_WORKFLOW_ID,
            "qc_report": {"status": final_state["qc_status"], "failed_expectations": failed},
        }
    )
    assert trainer_state.qc_report is not None
    assert trainer_state.qc_report.failed_expectations == failed


@pytest.mark.asyncio
async def test_failed_expectations_are_json_encodable_for_persistence(tmp_path: Path) -> None:
    """Once the channel stops crashing, ``agent.py`` forwards these entries to
    the ``ml_data_quality_reports.failed_expectations`` JSONB column. The
    Supabase client sends the row through httpx, which encodes with
    ``json.dumps(..., allow_nan=False)`` — the exact call below. A
    ``numpy.bool_`` (``null_count == 0`` on a numpy scalar) is rejected there,
    and the repository swallows the error, so the row would be lost silently.
    RED pre-fix: ``TypeError: Object of type bool is not JSON serializable``."""
    final_state = await _build_qc_chain().ainvoke(
        _base_state(_write_csv_with_required_nulls(tmp_path))
    )
    failed = final_state["failed_expectations"]

    encoded = json.dumps(failed, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded[0]["success"] is False
    assert decoded[0]["result"]["completeness_ratio"] == pytest.approx(0.75)
