"""#2207 split contract: the GE validation node accepts a table cohort dict.

Codex r1 HIGH on PR #2241: ``run_ge_validation`` accepted only ``file_dir`` / ``files``
dicts and failed CLOSED on any other dict — and it runs AFTER ``run_quality_checks``, so
its blocking entry survives to the QC gate. Every table contract seeded by migration 151
therefore loaded fine and then never reached training. Measured on the live Kisqali
initiation contract (2026-09-23): status=error, blocking
"unrecognised data_source dict shape (type='table')"; merely mapping the dict to its
table name scored 3/18 against the whole-table ``patient_journeys`` suite (it expects
``event_type`` / ``event_date`` / ``patient_id``, which a column-scoped projection cannot
carry).

Owner decision (2026-09-23): a table dict resolves to its table's suite (auto-detect
kept); because the contract's ``columns`` is a deliberate PROJECTION, suite expectations
on columns outside it are NOT APPLICABLE — they are skipped and logged at INFO, never
failed and never silently dropped as a whole — while table-level expectations and those
on projected columns still run and still block; the contract adds its own checks (every
declared column exists, the prediction target is non-null). Zero applicable expectations
is "warning", never "passed".

Hermetic: real ``DataQualityValidator`` on in-memory frames, no DB.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes import ge_validator
from src.services.cohort_contract import decode_data_source

_REPO = Path(__file__).resolve().parents[5]
_MIGRATION = _REPO / "database" / "migrations" / "151_registry_cohort_contracts_goldstd.sql"


def _kisqali_initiation_contract() -> Dict[str, Any]:
    """Migration 151's literal for initiation_kisqali_goldstd_lr_v1, decoded."""
    sql = _MIGRATION.read_text()
    m = re.search(
        r"SET cohort_data_source = '(\{[^']*\})',\s*cohort_target_outcome = 'treatment_initiated',"
        r"\s*cohort_feature_manifest_source = '[a-z_]+'"
        r"\s*WHERE model_name = 'initiation_kisqali_goldstd_lr_v1'",
        sql,
    )
    assert m, "Kisqali initiation UPDATE not found in migration 151"
    decoded = decode_data_source(m.group(1))
    assert isinstance(decoded, dict)
    return decoded


def _frame(columns: list[str], n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data: Dict[str, Any] = {}
    for c in columns:
        if c == "geographic_region":
            data[c] = rng.choice(["Northeast", "South"], n)
        elif c == "insurance_type":
            data[c] = rng.choice(["commercial", "medicare"], n)
        else:
            data[c] = rng.integers(0, 3, n)
    data["data_split"] = ["train"] * n
    return pd.DataFrame(data)


def _state(data_source: Any, columns: list[str], **frames: pd.DataFrame) -> Dict[str, Any]:
    target = "treatment_initiated"
    return {
        "experiment_id": "exp-2207",
        "data_source": data_source,
        "scope_spec": {"prediction_target": target, "data_source": data_source},
        "train_df": frames.get("train", _frame(columns)),
        "validation_df": frames.get("validation", _frame(columns, 20)),
        "test_df": frames.get("test", _frame(columns, 10)),
        "blocking_issues": [],
    }


@pytest.mark.asyncio
async def test_migration_151_kisqali_contract_passes_and_logs_skipped_expectations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    contract = _kisqali_initiation_contract()
    cols = contract["columns"]
    with caplog.at_level(logging.INFO, logger=ge_validator.__name__):
        out = await ge_validator.run_ge_validation(_state(contract, cols))
    assert out["ge_validation_status"] == "passed", out
    assert not out.get("blocking_issues"), out
    suites = {r["expectation_suite_name"] for r in out["ge_validation_results"]}
    assert suites == {"patient_journeys__contract"}
    # per split: row count (table-level, kept from the patient_journeys suite)
    # + 11 declared columns exist + target non-null = 13
    assert out["ge_expectations_evaluated"] == 13 * 3
    assert out["ge_expectations_passed"] == 13 * 3
    skipped = [
        r.getMessage() for r in caplog.records if "not in contract projection" in r.getMessage()
    ]
    assert skipped, "skipped suite expectations were not logged"
    for name in ("patient_id", "event_type", "event_date"):
        assert any(name in m for m in skipped), (name, skipped)


@pytest.mark.asyncio
async def test_contract_suite_has_teeth_on_a_missing_declared_column() -> None:
    contract = _kisqali_initiation_contract()
    cols = contract["columns"]
    train = _frame(cols).drop(columns=["academic_hcp"])
    out = await ge_validator.run_ge_validation(_state(contract, cols, train=train))
    assert out["ge_validation_status"] == "failed", out
    assert any("academic_hcp" in issue for issue in out["blocking_issues"]), out["blocking_issues"]


@pytest.mark.asyncio
async def test_contract_suite_has_teeth_on_a_null_target() -> None:
    contract = _kisqali_initiation_contract()
    cols = contract["columns"]
    train = _frame(cols).astype({"treatment_initiated": "float"})
    train.loc[train.index[:20], "treatment_initiated"] = np.nan
    out = await ge_validator.run_ge_validation(_state(contract, cols, train=train))
    assert out["ge_validation_status"] == "failed", out
    assert any("treatment_initiated" in issue for issue in out["blocking_issues"]), out


@pytest.mark.asyncio
async def test_projected_column_expectation_from_table_suite_still_runs_and_blocks() -> None:
    """A suite expectation whose column IS in the projection is applicable and blocking:
    here the table suite demands patient_id non-null and the contract projects it."""
    contract = {
        "type": "table",
        "table": "patient_journeys",
        "columns": ["patient_id", "disease_severity", "treatment_initiated"],
    }
    cols = contract["columns"]
    train = _frame(cols).astype({"patient_id": "object"})
    train.loc[train.index[:5], "patient_id"] = None
    out = await ge_validator.run_ge_validation(_state(contract, cols, train=train))
    assert out["ge_validation_status"] == "failed", out
    assert any(
        "expect_column_values_to_not_be_null on patient_id" in i for i in out["blocking_issues"]
    ), out["blocking_issues"]


@pytest.mark.asyncio
async def test_bare_table_dict_uses_the_tables_own_suite() -> None:
    validator = MagicMock()
    validator.SUITES = {"patient_journeys": [], "business_metrics": []}
    result = MagicMock(
        blocking=False, expectations_evaluated=1, expectations_passed=1, expectations_failed=0
    )
    result.to_dict.return_value = {"expectation_suite_name": "patient_journeys"}
    validator.validate_splits = AsyncMock(return_value={"train": result})
    contract = {"type": "table", "table": "patient_journeys"}  # no columns
    cols = ["patient_id", "event_type", "event_date"]
    with patch.object(ge_validator, "get_data_quality_validator", return_value=validator):
        out = await ge_validator.run_ge_validation(_state(contract, cols))
    assert out["ge_validation_status"] == "passed"
    kwargs = validator.validate_splits.await_args.kwargs
    assert kwargs["suite_name"] == "patient_journeys"
    assert kwargs["table_name"] == "patient_journeys"
    validator.register_suite.assert_not_called()


@pytest.mark.asyncio
async def test_projection_keeping_a_list_valued_suite_expectation_does_not_crash() -> None:
    """codex r2 MED: the dedupe key hashed raw kwargs; ``ml_patients`` carries
    ``expect_column_values_to_be_in_set(discontinuation_flag, [0, 1, True, False])``, a
    list-valued kwarg, so a contract projecting ``discontinuation_flag`` raised
    ``TypeError: unhashable type: 'list'`` and every such run returned a blocking GE error."""
    cols = [
        "patient_journey_id",
        "patient_id",
        "brand",
        "discontinuation_flag",
        "treatment_initiated",
    ]
    contract = {"type": "table", "table": "patient_journeys", "columns": cols}
    n = 40
    train = pd.DataFrame(
        {
            "patient_journey_id": [f"PJ{i}" for i in range(n)],
            "patient_id": [f"P{i}" for i in range(n)],
            "brand": ["Kisqali"] * n,
            "discontinuation_flag": [i % 2 for i in range(n)],
            "treatment_initiated": [i % 2 for i in range(n)],
        }
    )
    out = await ge_validator.run_ge_validation(
        _state(contract, cols, train=train, validation=train.head(20), test=train.head(10))
    )
    assert out["ge_validation_status"] == "passed", out
    assert not out.get("blocking_issues"), out
    # The auto-detected ml_patients suite's in-set expectation on the projected column
    # is applicable and kept (alongside the contract checks).
    assert out["ge_expectations_evaluated"] == 3 * (1 + 5 + 1 + 3), (
        out
    )  # row + exists*5 + not-null target + kept ml_patients (2 not-null + in-set)


def test_contract_suite_construction_never_yields_zero_expectations() -> None:
    """codex r2 LOW: the "zero applicable expectations" status guard is defensive — real
    construction always adds the row-count check plus one existence check per declared
    column, so it is unreachable for a non-empty projection. Pin that, so the warning path
    below is understood as an aggregation guard over an impossible validator result."""
    validator = MagicMock()
    validator.SUITES = {"patient_journeys": []}
    name = ge_validator._register_contract_suite(
        validator, "patient_journeys", "patient_journeys", ["x"], None
    )
    registered = validator.register_suite.call_args.args[1]
    assert name == "patient_journeys__contract"
    assert len(registered) == 2, registered
    name2 = ge_validator._register_contract_suite(
        validator, "patient_journeys", None, ["x", "y"], "y"
    )
    registered2 = validator.register_suite.call_args.args[1]
    assert name2 == name and len(registered2) == 4, registered2


@pytest.mark.asyncio
async def test_zero_applicable_expectations_is_warning_not_passed() -> None:
    """Aggregation guard only (see the construction pin above): a validator result with
    zero evaluations must never aggregate to "passed"."""
    validator = MagicMock()
    validator.SUITES = {"patient_journeys": [], "business_metrics": []}
    result = MagicMock(
        blocking=False, expectations_evaluated=0, expectations_passed=0, expectations_failed=0
    )
    result.to_dict.return_value = {"expectation_suite_name": "patient_journeys__contract"}
    validator.validate_splits = AsyncMock(return_value={"train": result})
    contract = {"type": "table", "table": "patient_journeys", "columns": ["x"]}
    with patch.object(ge_validator, "get_data_quality_validator", return_value=validator):
        out = await ge_validator.run_ge_validation(_state(contract, ["x"]))
    assert out["ge_validation_status"] == "warning", out
    assert out["ge_expectations_evaluated"] == 0
    assert "zero" in (out.get("ge_validation_note") or "").lower()


@pytest.mark.asyncio
async def test_unknown_dict_shape_still_fails_closed() -> None:
    out = await ge_validator.run_ge_validation(_state({"type": "s3", "bucket": "x"}, ["x"]))
    assert out["ge_validation_status"] == "error"
    assert any("unrecognised data_source dict shape" in i for i in out["blocking_issues"])


@pytest.mark.asyncio
async def test_string_source_unchanged() -> None:
    validator = MagicMock()
    validator.SUITES = {"business_metrics": []}
    result = MagicMock(
        blocking=False, expectations_evaluated=1, expectations_passed=1, expectations_failed=0
    )
    result.to_dict.return_value = {}
    validator.validate_splits = AsyncMock(return_value={"train": result})
    with patch.object(ge_validator, "get_data_quality_validator", return_value=validator):
        out = await ge_validator.run_ge_validation(_state("business_metrics", ["x"]))
    assert out["ge_validation_status"] == "passed"
    assert validator.validate_splits.await_args.kwargs["suite_name"] == "business_metrics"
    validator.register_suite.assert_not_called()
