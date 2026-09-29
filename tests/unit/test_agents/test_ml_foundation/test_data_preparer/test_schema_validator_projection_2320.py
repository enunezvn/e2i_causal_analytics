"""#2320: the Pandera schema gate validates a table cohort contract against its
table's schema, restricted to the contract's column PROJECTION.

A table cohort contract (``{"type": "table", "table": ..., "columns": [...]}``,
migration 151) loads ONLY its ``columns``. Two defects met here:

* In the production state shape the contract dict reaches ``run_schema_validation``
  on the top-level ``data_source`` only — ``ScopeSpecSchema.data_source`` is
  ``Optional[str]`` and ``scope_definer`` never sets it — so the node fell through to
  ``""``, found no schema, and reported ``skipped``: every goldstd retrain ran with
  NO Pandera check at all (fail-open; measured on main 2026-09-29).
* Resolving the dict to its table alone is not enough: the whole-table models require
  their id columns (``PatientJourneysSchema``: ``patient_journey_id`` + ``patient_id``),
  which no goldstd contract projects — every split then fails
  ``column_in_dataframe`` and, since #2294, the QC gate can never pass.

The rule mirrors the GE contract suite (``ge_validator._register_contract_suite``,
owner decision 2026-09-23): a schema column outside the projection is NOT
APPLICABLE and is skipped; a schema column inside it keeps every check it declares;
every projected column must exist in the frame. A non-projected load (string
``data_source``, or a table dict with no ``columns``) keeps full-schema behaviour.

Each gate test drives the REAL ``run_schema_validation`` node and the REAL
``finalize_output`` gate through a compiled ``StateGraph(DataPreparerState)`` — whose
state validation is what rejects a dict in ``scope_spec.data_source`` — with the REAL
registry schemas; nothing under test is stubbed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from langgraph.graph import END, StateGraph

from src.agents.ml_foundation.data_preparer.graph import finalize_output
from src.agents.ml_foundation.data_preparer.nodes.schema_validator import (
    run_schema_validation,
)
from src.agents.ml_foundation.data_preparer.state import DataPreparerState
from src.mlops.gold_standard_eval.cohort_spec import _PATIENT_COVARIATES, _PATIENT_LABELS
from src.mlops.pandera_schemas import validate_dataframe
from src.services.cohort_contract import decode_data_source

_REPO = Path(__file__).resolve().parents[5]
_MIGRATION = _REPO / "database" / "migrations" / "151_registry_cohort_contracts_goldstd.sql"

# Upstream QC verdict every graph starts from: a clean pass, so ``gate_passed=False``
# can only come from ``blocking_issues``.
_CLEAN_UPSTREAM_QC: Dict[str, Any] = {
    "audit_workflow_id": "00000000-0000-0000-0000-000000002320",
    "qc_status": "passed",
    "overall_score": 0.95,
}


def _schema_gate_graph():
    """``run_schema_validation -> finalize_output`` over the real state schema."""
    graph = StateGraph(DataPreparerState)
    graph.add_node("run_schema_validation", run_schema_validation)  # type: ignore[arg-type]
    graph.add_node("finalize_output", finalize_output)  # type: ignore[arg-type]
    graph.set_entry_point("run_schema_validation")
    graph.add_edge("run_schema_validation", "finalize_output")
    graph.add_edge("finalize_output", END)
    return graph.compile()


def _migration_151_contracts() -> Dict[str, Dict[str, Any]]:
    """Every table contract migration 151 seeds, decoded, keyed by model name."""
    sql = _MIGRATION.read_text()
    found = re.findall(
        r"SET cohort_data_source = '(\{[^']*\})',.*?WHERE model_name = '([a-z_0-9]+)'",
        sql,
        flags=re.DOTALL,
    )
    contracts = {name: decode_data_source(literal) for literal, name in found}
    assert contracts, "no contract UPDATE found in migration 151"
    return contracts


def _patient_contract(cohort: str, brand: str = "Kisqali") -> Dict[str, Any]:
    """The goldstd contract shape the retrain path loads (cohort_spec projection)."""
    return {
        "type": "table",
        "table": "patient_journeys",
        "filters": {"brand": brand, "is_synthetic": True},
        "columns": list(_PATIENT_COVARIATES[cohort]) + [_PATIENT_LABELS[cohort]],
    }


def _frame(columns: List[str], n: int = 40, seed: int = 2320) -> pd.DataFrame:
    """Values shaped like the real Kisqali cohort (lower-case region enum, 0/1 flags).

    ``data_split`` is added the way ``_load_precomputed_split`` SELECTs it.
    """
    rng = np.random.default_rng(seed)
    data: Dict[str, Any] = {}
    for c in columns:
        if c == "geographic_region":
            data[c] = rng.choice(["northeast", "south", "midwest", "west"], n)
        elif c == "insurance_type":
            data[c] = rng.choice(["commercial", "medicare", "medicaid"], n)
        elif c == "age_at_diagnosis":
            data[c] = rng.normal(55.0, 10.0, n)
        elif c == "patient_id":
            data[c] = [f"pat-{i:04d}" for i in range(n)]
        elif c == "patient_journey_id":
            data[c] = [f"pj-{i:04d}" for i in range(n)]
        else:
            data[c] = rng.integers(0, 2, n)
    data["data_split"] = ["train"] * n
    return pd.DataFrame(data)


def _state(
    data_source: Any,
    columns: List[str],
    frame: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    # Production shape: ``scope_definer`` never sets ``scope_spec.data_source`` and the
    # typed state only admits a string there, so a contract dict lives on the top level.
    scope_spec: Dict[str, Any] = {"experiment_id": "exp-2320"}
    if isinstance(data_source, str):
        scope_spec["data_source"] = data_source
    return {
        "experiment_id": "exp-2320",
        "data_source": data_source,
        "scope_spec": scope_spec,
        "train_df": frame if frame is not None else _frame(columns),
        "validation_df": _frame(columns, 20, seed=1),
        "test_df": _frame(columns, 10, seed=2),
        "blocking_issues": [],
        **_CLEAN_UPSTREAM_QC,
    }


def _schema_entries(final_state: Dict[str, Any]) -> List[str]:
    return [i for i in final_state.get("blocking_issues") or [] if i.startswith("schema: ")]


# =============================================================================
# The defect: a projected goldstd patient cohort never passed the schema gate
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("cohort", sorted(_PATIENT_LABELS))
async def test_projected_goldstd_patient_cohort_passes_the_schema_gate(cohort: str) -> None:
    contract = _patient_contract(cohort)
    final_state = await _schema_gate_graph().ainvoke(_state(contract, contract["columns"]))

    assert final_state["schema_validation_status"] == "passed", final_state.get(
        "schema_validation_errors"
    )
    assert final_state["schema_splits_validated"] == 3
    assert _schema_entries(final_state) == []
    assert final_state["gate_passed"] is True


@pytest.mark.asyncio
async def test_every_migration_151_contract_passes_the_schema_gate() -> None:
    """The literals actually seeded in prod, not a reconstruction of them."""
    for name, contract in _migration_151_contracts().items():
        assert isinstance(contract, dict) and contract.get("type") == "table", name
        final_state = await _schema_gate_graph().ainvoke(_state(contract, contract["columns"]))
        assert final_state["schema_validation_status"] == "passed", (
            name,
            final_state.get("schema_validation_errors"),
        )
        assert final_state["gate_passed"] is True, (name, final_state.get("blocking_issues"))


# =============================================================================
# Teeth: the projection narrows the schema, it does not silence it
# =============================================================================


@pytest.mark.asyncio
async def test_projected_patient_id_with_nulls_still_fails() -> None:
    contract = _patient_contract("initiation")
    contract["columns"] = ["patient_id"] + contract["columns"]
    frame = _frame(contract["columns"])
    frame.loc[[3, 7], "patient_id"] = None

    final_state = await _schema_gate_graph().ainvoke(_state(contract, contract["columns"], frame))

    assert final_state["schema_validation_status"] == "failed"
    errors = [e for e in final_state["schema_validation_errors"] if e.get("split") == "train"]
    assert {e.get("column") for e in errors} == {"patient_id"}, errors
    assert any("not_nullable" in str(e.get("check")) for e in errors), errors
    assert len(_schema_entries(final_state)) == 1
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_projected_journey_id_duplicates_still_fail_uniqueness() -> None:
    contract = _patient_contract("initiation")
    contract["columns"] = ["patient_journey_id"] + contract["columns"]
    frame = _frame(contract["columns"])
    frame.loc[5, "patient_journey_id"] = frame.loc[4, "patient_journey_id"]

    final_state = await _schema_gate_graph().ainvoke(_state(contract, contract["columns"], frame))

    assert final_state["schema_validation_status"] == "failed"
    checks = {
        str(e.get("check"))
        for e in final_state["schema_validation_errors"]
        if e.get("column") == "patient_journey_id"
    }
    assert "field_uniqueness" in checks, final_state["schema_validation_errors"]
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_projected_enum_column_with_bad_values_still_fails() -> None:
    """``geographic_region`` IS in every goldstd projection: its ``isin`` check stays."""
    contract = _patient_contract("persistence")
    frame = _frame(contract["columns"])
    frame.loc[0, "geographic_region"] = "atlantis"

    final_state = await _schema_gate_graph().ainvoke(_state(contract, contract["columns"], frame))

    assert final_state["schema_validation_status"] == "failed"
    assert {e.get("column") for e in final_state["schema_validation_errors"]} == {
        "geographic_region"
    }
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dropped",
    [
        # A projected column the schema does not know about.
        "disease_severity",
        # A projected column the schema declares Optional (required=False).
        "geographic_region",
    ],
)
async def test_projected_column_missing_from_the_frame_fails(dropped: str) -> None:
    contract = _patient_contract("initiation")
    frame = _frame(contract["columns"]).drop(columns=[dropped])

    final_state = await _schema_gate_graph().ainvoke(_state(contract, contract["columns"], frame))

    assert final_state["schema_validation_status"] == "failed"
    train_errors = [e for e in final_state["schema_validation_errors"] if e["split"] == "train"]
    assert train_errors, final_state["schema_validation_errors"]
    assert all(e.get("check") == "column_in_dataframe" for e in train_errors), train_errors
    assert {str(e.get("failure_case")) for e in train_errors} == {dropped}, train_errors
    assert final_state["gate_passed"] is False


# =============================================================================
# Non-projected loads keep full-schema behaviour
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data_source",
    [
        "patient_journeys",
        {"type": "table", "table": "patient_journeys", "filters": {}},
        {"type": "table", "table": "patient_journeys", "filters": {}, "columns": None},
        # ``load_data`` SELECTs every column for an empty list too.
        {"type": "table", "table": "patient_journeys", "filters": {}, "columns": []},
    ],
    ids=["string", "dict-no-columns", "dict-columns-none", "dict-columns-empty"],
)
async def test_full_table_load_still_requires_the_id_columns(data_source: Any) -> None:
    columns = list(_PATIENT_COVARIATES["initiation"]) + [_PATIENT_LABELS["initiation"]]
    final_state = await _schema_gate_graph().ainvoke(_state(data_source, columns))

    assert final_state["schema_validation_status"] == "failed"
    missing = {
        str(e.get("failure_case"))
        for e in final_state["schema_validation_errors"]
        if e.get("check") == "column_in_dataframe"
    }
    assert missing == {"patient_journey_id", "patient_id"}, final_state["schema_validation_errors"]
    assert final_state["gate_passed"] is False


@pytest.mark.asyncio
async def test_full_table_frame_with_ids_passes() -> None:
    columns = ["patient_journey_id", "patient_id", "geographic_region"]
    final_state = await _schema_gate_graph().ainvoke(_state("patient_journeys", columns))

    assert final_state["schema_validation_status"] == "passed"
    assert final_state["gate_passed"] is True


# =============================================================================
# Registry-wide: a table contract resolves ITS table's schema
# =============================================================================


@pytest.mark.asyncio
async def test_table_contract_on_another_registered_table_uses_that_tables_schema() -> None:
    """A ``triggers`` contract used to fall back to ``patient_journeys``."""
    contract = {"type": "table", "table": "triggers", "filters": {}, "columns": ["priority"]}
    frame = pd.DataFrame({"priority": ["high", "low"] * 20})
    state = _state(contract, ["priority"], frame)
    state["validation_df"] = frame.copy()
    state["test_df"] = frame.copy()

    passed = await _schema_gate_graph().ainvoke(state)
    assert passed["schema_validation_status"] == "passed", passed.get("schema_validation_errors")
    assert passed["gate_passed"] is True

    bad = frame.copy()
    bad.loc[0, "priority"] = "urgent"
    state = _state(contract, ["priority"], bad)
    state["validation_df"] = frame.copy()
    state["test_df"] = frame.copy()
    failed = await _schema_gate_graph().ainvoke(state)
    assert failed["schema_validation_status"] == "failed"
    assert {e.get("column") for e in failed["schema_validation_errors"]} == {"priority"}
    assert {e.get("data_source") for e in failed["schema_validation_errors"]} == {"triggers"}
    assert failed["gate_passed"] is False


@pytest.mark.asyncio
async def test_table_contract_on_an_unregistered_table_is_skipped_like_its_string() -> None:
    """``hcp_brand_adoption`` has no Pandera model: same verdict as the string source."""
    contract = {
        "type": "table",
        "table": "hcp_brand_adoption",
        "filters": {},
        "columns": ["adopted"],
    }
    final_state = await _schema_gate_graph().ainvoke(_state(contract, ["adopted"]))

    assert final_state["schema_validation_status"] == "skipped"
    assert _schema_entries(final_state) == []


@pytest.mark.parametrize(
    ("table", "id_columns"),
    [
        ("business_metrics", ["metric_id", "metric_date"]),
        ("predictions", ["prediction_id"]),
        ("ml_predictions", ["prediction_id"]),
        ("triggers", ["trigger_id", "patient_id"]),
        ("patient_journeys", ["patient_journey_id", "patient_id"]),
        ("causal_paths", ["path_id"]),
        ("agent_activities", ["activity_id"]),
    ],
)
def test_every_registered_schema_honours_a_projection(table: str, id_columns: List[str]) -> None:
    """Every registry entry has required id columns; none may block a projection that
    omits them, and each still requires them on a full-table load."""
    frame = pd.DataFrame({"feature_a": [1, 2, 3]})

    full = validate_dataframe(frame, table)
    assert full["status"] == "failed"
    assert {e["failure_case"] for e in full["errors"]} >= set(id_columns), full["errors"]

    projected = validate_dataframe(frame, table, columns=["feature_a"])
    assert projected["status"] == "passed", projected["errors"]

    missing = validate_dataframe(frame, table, columns=["feature_a", "feature_b"])
    assert missing["status"] == "failed"
    assert [(e["check"], e["failure_case"]) for e in missing["errors"]] == [
        ("column_in_dataframe", "feature_b")
    ]


# =============================================================================
# Codex r1 MED: the node resolves the source exactly as ``load_data`` does
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (pd.DataFrame({"value": [1.0, 2.0]}), "failed"),
        (
            pd.DataFrame({"metric_id": ["m1", "m2"], "metric_date": ["2026-01-01", "2026-01-02"]}),
            "passed",
        ),
    ],
    ids=["missing-ids", "valid"],
)
async def test_no_source_anywhere_validates_the_business_metrics_default(
    frame: pd.DataFrame, expected: str
) -> None:
    """``load_data`` loads ``business_metrics`` when neither ``state.data_source`` nor
    ``scope_spec.data_source`` is given; the frame it loaded is held to that schema."""
    state = {
        "experiment_id": "exp-2320-default",
        "scope_spec": {"experiment_id": "exp-2320-default"},
        "train_df": frame,
        "blocking_issues": [],
        **_CLEAN_UPSTREAM_QC,
    }
    final_state = await _schema_gate_graph().ainvoke(state)

    assert final_state["schema_validation_status"] == expected, final_state.get(
        "schema_validation_errors"
    )
    assert final_state["gate_passed"] is (expected == "passed")
