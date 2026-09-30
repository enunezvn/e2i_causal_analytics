"""Regression for #2290: the ``warnings`` channel is ``List[Dict]`` everywhere,
and ``run_quality_checks`` no longer overwrites other producers' entries.

Two defects, both measured on the compiled graph over the real
``DataPreparerState`` before the fix:

1. **Crash, not just a type wobble.** ``run_schema_validation``'s
   no-DataFrames path appended a bare ``str`` to a channel declared
   ``Optional[List[Dict[str, Any]]]``. The write itself lands, but the next
   node's input is validated, so the run died with ``ValidationError:
   warnings.0 Input should be a valid dictionary``. That path is reachable on
   the production edge order: a ``load_data`` failure (e.g. a missing input
   file) leaves no DataFrames, and the pydantic crash then MASKED
   ``load_data``'s real error — the same shape as #2292.
2. **Overwrite.** ``warnings`` has no reducer (LastValue) and
   ``run_quality_checks`` returned a fresh local list, so any upstream entry
   was lost — the #2283 defect on a second channel.

Element type is ``Dict`` because that is what every consumer reads:
``QCReportSchema.warnings`` / ``qc_warnings`` are ``List[Dict[str, Any]]``,
``model_trainer``'s ``qc_gate_checker`` summarises ``w.get("expectation_type")``
and ``quality_checker``'s five producers already emit dicts.

The merge follows #2283's ownership contract (own-kind replace, never
``operator.add``): each producer tags its entries with a ``kind`` and replaces
only those on re-entry, because the QC retry edge re-runs
``run_quality_checks``.

Data comes from a local CSV through ``load_data``'s file path; nothing is
mocked and nothing touches Supabase, Redis or MLflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest
from langgraph.graph import END, StateGraph

from src.agents.ml_foundation.data_preparer.nodes.data_loader import load_data
from src.agents.ml_foundation.data_preparer.nodes.quality_checker import run_quality_checks
from src.agents.ml_foundation.data_preparer.nodes.sampling_frame_audit import (
    audit_sampling_frame,
)
from src.agents.ml_foundation.data_preparer.nodes.schema_validator import (
    run_schema_validation,
)
from src.agents.ml_foundation.data_preparer.schemas import QCReportSchema
from src.agents.ml_foundation.data_preparer.state import DataPreparerState

_AUDIT_WORKFLOW_ID = "00000000-0000-0000-0000-000000002290"


def _chain(*nodes: Tuple[str, Any]):
    graph = StateGraph(DataPreparerState)
    for name, fn in nodes:
        graph.add_node(name, fn)  # type: ignore[arg-type]
    graph.set_entry_point(nodes[0][0])
    for (a, _), (b, _) in zip(nodes[:-1], nodes[1:], strict=True):
        graph.add_edge(a, b)
    graph.add_edge(nodes[-1][0], END)
    return graph.compile()


def _write_csv(tmp_path: Path) -> Path:
    """``sparse_flag`` is 1/3 null and not required, so ``_check_completeness``
    emits a non-blocking ``expect_column_null_percentage`` WARNING for it —
    ``run_quality_checks`` then has an entry of its own in the channel."""
    n = 60
    dates = pd.date_range("2099-01-01", periods=n, freq="D")
    frame = pd.DataFrame(
        {
            "patient_id": [f"pat-{i:04d}" for i in range(n)],
            "event_type": ["prescription"] * n,
            "event_date": dates.strftime("%Y-%m-%d"),
            "created_at": dates.strftime("%Y-%m-%d"),
            "days_on_therapy": [30 + (i % 15) for i in range(n)],
            "sparse_flag": [None if i % 3 == 0 else 1 for i in range(n)],
            "data_split": (["train"] * 36 + ["validation"] * 12 + ["test"] * 6 + ["holdout"] * 6),
        }
    )
    path = tmp_path / "patient_journeys.csv"
    frame.to_csv(path, index=False)
    return path


def _base_state(csv_path: Path | str) -> Dict[str, Any]:
    return {
        "audit_workflow_id": _AUDIT_WORKFLOW_ID,
        "experiment_id": "exp-2290",
        "data_source": {"type": "files", "paths": {"patient_journeys": str(csv_path)}},
        "scope_spec": {"date_column": "event_date", "data_source": "patient_journeys"},
    }


def _qc_entries(warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [w for w in warnings if w.get("kind") == "quality_check"]


@pytest.mark.asyncio
async def test_load_failure_reaches_quality_check_instead_of_crashing(tmp_path: Path) -> None:
    """RED pre-fix: ``ainvoke`` raises ``ValidationError: warnings.0 Input
    should be a valid dictionary`` — the production edge order
    ``load_data -> audit_sampling_frame -> run_schema_validation ->
    run_quality_checks`` on a missing input file."""
    final_state = await _chain(
        ("load_data", load_data),
        ("audit_sampling_frame", audit_sampling_frame),
        ("run_schema_validation", run_schema_validation),
        ("run_quality_checks", run_quality_checks),
    ).ainvoke(_base_state(tmp_path / "does_not_exist.csv"))

    warnings = final_state["warnings"]
    assert all(isinstance(w, dict) for w in warnings), warnings
    schema_entries = [w for w in warnings if w.get("kind") == "schema"]
    assert len(schema_entries) == 1, warnings
    assert "No DataFrames available for schema validation" in schema_entries[0]["message"]
    # The run is reported as failed, not crashed.
    assert final_state["qc_status"] == "failed"
    assert final_state["blocking_issues"]


@pytest.mark.asyncio
async def test_quality_checker_preserves_foreign_warnings(tmp_path: Path) -> None:
    """RED pre-fix: the seeded upstream entry is gone after ``run_quality_checks``."""
    foreign = {"kind": "schema", "severity": "warning", "message": "upstream note"}
    state = _base_state(_write_csv(tmp_path))
    state["warnings"] = [foreign]

    final_state = await _chain(
        ("load_data", load_data), ("run_quality_checks", run_quality_checks)
    ).ainvoke(state)

    warnings = final_state["warnings"]
    assert foreign in warnings, f"run_quality_checks overwrote the channel: {warnings!r}"
    own = _qc_entries(warnings)
    assert [w["column"] for w in own] == ["sparse_flag"], warnings


@pytest.mark.asyncio
async def test_re_entry_replaces_own_warnings_instead_of_appending(tmp_path: Path) -> None:
    """The QC retry edge re-runs ``run_quality_checks``; a second pass must
    replace its own entries, not duplicate them (rules out ``operator.add``
    and a plain ``incoming + own`` merge)."""
    state = _base_state(_write_csv(tmp_path))
    final_state = await _chain(
        ("load_data", load_data),
        ("run_quality_checks", run_quality_checks),
        ("run_quality_checks_retry", run_quality_checks),
    ).ainvoke(state)

    own = _qc_entries(final_state["warnings"])
    assert len(own) == 1, f"own warnings duplicated on re-entry: {final_state['warnings']!r}"


@pytest.mark.asyncio
async def test_stale_own_warning_is_retracted_on_a_clean_pass(tmp_path: Path) -> None:
    """A previous pass's quality_check warning must disappear once the condition
    clears; a foreign entry must not."""
    state = _base_state(_write_csv(tmp_path))
    stale = {"kind": "quality_check", "column": "long_gone", "severity": "warning"}
    foreign = {"kind": "schema", "severity": "warning", "message": "keep me"}
    state["warnings"] = [stale, foreign]

    final_state = await _chain(
        ("load_data", load_data), ("run_quality_checks", run_quality_checks)
    ).ainvoke(state)

    warnings = final_state["warnings"]
    assert stale not in warnings
    assert foreign in warnings


@pytest.mark.asyncio
async def test_tagged_warnings_still_satisfy_the_qc_report_contract(tmp_path: Path) -> None:
    """``agent.py`` copies the channel into ``qc_report.warnings`` and
    ``qc_warnings``; the typed report contract must still accept it, and the
    tag must not leak into ``expectation_results`` (the warning entries are
    tagged copies, not the same dict objects mutated in place)."""
    final_state = await _chain(
        ("load_data", load_data), ("run_quality_checks", run_quality_checks)
    ).ainvoke(_base_state(_write_csv(tmp_path)))

    warnings = final_state["warnings"]
    report = QCReportSchema.model_validate({"warnings": warnings, "qc_warnings": warnings})
    assert report.qc_warnings == warnings
    assert not any("kind" in r for r in final_state["expectation_results"])
