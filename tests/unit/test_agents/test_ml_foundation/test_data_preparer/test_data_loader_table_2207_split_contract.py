"""#2207 split contract: the Supabase-table route of ``data_loader`` honours ``data_split``.

Owner decision (2026-09-23, decision 4): a cohort loaded from a Supabase table declares
its split by carrying a ``data_split`` column and ``data_loader`` honours it verbatim,
exactly as ``_load_from_files`` does via ``_split_from_column``. Cohort identity for a
table is a dict so a brand-partitioned, provenance-scoped, column-scoped cohort is
reloadable exactly::

    {"type": "table", "table": "patient_journeys",
     "filters": {"brand": "Kisqali", "is_synthetic": true},
     "columns": ["disease_severity", ..., "treatment_initiated"]}

Before this lane the string route could NEVER pass the trainer's ``split_enforcer``:
``_load_from_supabase`` hard-coded ``holdout=None`` (``MLDataset`` has no holdout and
``combined_split`` yields none either) while the enforcer requires a non-empty holdout
and 60/20/10/10 +-2 %. Hermetic: every loader is a fake; nothing touches the live DB.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes import data_loader
from src.agents.ml_foundation.model_trainer.nodes.split_enforcer import enforce_splits

_COVARIATES = ["disease_severity", "academic_hcp", "geographic_region"]
_TARGET = "treatment_initiated"
_CONTRACT: Dict[str, Any] = {
    "type": "table",
    "table": "patient_journeys",
    "filters": {"brand": "Kisqali", "is_synthetic": True},
    "columns": _COVARIATES + [_TARGET],
}


def _labelled_frame(n_train: int, n_val: int, n_test: int, n_holdout: int) -> pd.DataFrame:
    labels = (
        ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test + ["holdout"] * n_holdout
    )
    n = len(labels)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "patient_journey_id": [f"PJ_{i:06d}" for i in range(n)],
            "patient_id": [f"PAT_{i:06d}" for i in range(n)],
            "brand": ["Kisqali"] * n,
            "is_synthetic": [True] * n,
            "disease_severity": rng.integers(1, 4, n),
            "academic_hcp": rng.integers(0, 2, n),
            "geographic_region": rng.choice(["Northeast", "South"], n),
            "days_to_treatment": rng.integers(0, 90, n),
            _TARGET: rng.integers(0, 2, n),
            "data_split": labels,
        }
    )


def _fake_loader(
    full_df: pd.DataFrame, *, has_data_split: bool = True
) -> tuple[MagicMock, List[Dict[str, Any]]]:
    """A loader whose ``load_table_sample`` mirrors the real one: ``columns`` is the
    projection, ``limit`` caps the rows, a missing column yields an EMPTY frame
    (the real loader logs and returns ``pd.DataFrame()`` on a PostgREST 42703)."""
    calls: List[Dict[str, Any]] = []
    source = full_df if has_data_split else full_df.drop(columns=["data_split"])

    async def _sample(
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
        columns: Optional[List[str]] = None,
        include_synthetic: bool = False,
    ) -> pd.DataFrame:
        calls.append(
            {
                "table": table,
                "filters": filters,
                "limit": limit,
                "columns": columns,
                "include_synthetic": include_synthetic,
            }
        )
        if columns and any(c not in source.columns for c in columns):
            return pd.DataFrame()
        frame = source[list(columns)] if columns else source
        return frame.head(limit).reset_index(drop=True)

    loader = MagicMock()
    loader.load_table_sample = AsyncMock(side_effect=_sample)
    loader.has_column = AsyncMock(return_value=has_data_split)
    loader.load_for_training = AsyncMock(
        return_value=MagicMock(train=source.iloc[:3], val=source.iloc[3:5], test=source.iloc[5:6])
    )
    return loader, calls


def _state(data_source: Any, **scope: Any) -> Dict[str, Any]:
    scope_spec = {"prediction_target": _TARGET, "filters": {}, **scope}
    return {"experiment_id": "exp-2207", "scope_spec": scope_spec, "data_source": data_source}


# --------------------------------------------------------------------------- #
# (i) dict table source -> precomputed split, non-None holdout, 4 partitions
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_dict_table_source_uses_data_split_column() -> None:
    df = _labelled_frame(60, 20, 10, 10)
    loader, calls = _fake_loader(df)
    with patch.object(data_loader, "get_ml_data_loader", return_value=loader):
        out = await data_loader.load_data(_state(dict(_CONTRACT)))

    assert "error" not in out, out
    assert len(out["train_df"]) == 60
    assert len(out["validation_df"]) == 20
    assert len(out["test_df"]) == 10
    assert out["holdout_df"] is not None and len(out["holdout_df"]) == 10
    # The temporal path must not have run.
    loader.load_for_training.assert_not_awaited()
    # The full load is column-scoped: the contract's columns plus the split column,
    # never the leaky whole table (days_to_treatment is NOT in the frame).
    full = [c for c in calls if c["limit"] == data_loader._TABLE_ROW_LIMIT]
    assert len(full) == 1
    assert full[0]["columns"] == _COVARIATES + [_TARGET, "data_split"]
    assert full[0]["table"] == "patient_journeys"
    assert "days_to_treatment" not in out["train_df"].columns
    assert set(out["train_df"].columns) == set(_COVARIATES + [_TARGET, "data_split"])


# --------------------------------------------------------------------------- #
# (ii) explicit is_synthetic filter -> include_synthetic=True reaches the loader
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_explicit_is_synthetic_filter_forces_include_synthetic() -> None:
    df = _labelled_frame(6, 2, 1, 1)
    loader, calls = _fake_loader(df)
    with patch.object(data_loader, "get_ml_data_loader", return_value=loader):
        out = await data_loader.load_data(_state(dict(_CONTRACT), filters={"region": "NE"}))

    assert "error" not in out, out
    assert calls, "loader never called"
    for call in calls:
        assert call["include_synthetic"] is True, call
        # Contract filters win over scope filters and the merge keeps both.
        assert call["filters"] == {"region": "NE", "brand": "Kisqali", "is_synthetic": True}


@pytest.mark.asyncio
async def test_no_is_synthetic_filter_keeps_default_exclude() -> None:
    df = _labelled_frame(6, 2, 1, 1)
    loader, calls = _fake_loader(df)
    contract = {**_CONTRACT, "filters": {"brand": "Kisqali"}}
    with patch.object(data_loader, "get_ml_data_loader", return_value=loader):
        out = await data_loader.load_data(_state(contract))

    assert "error" not in out, out
    assert all(call["include_synthetic"] is False for call in calls), calls


# --------------------------------------------------------------------------- #
# (iii) string source on a table WITHOUT data_split -> temporal path unchanged
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_string_source_without_data_split_keeps_temporal_path() -> None:
    df = _labelled_frame(6, 2, 1, 1)
    loader, calls = _fake_loader(df, has_data_split=False)
    with patch.object(data_loader, "get_ml_data_loader", return_value=loader):
        out = await data_loader.load_data(
            _state("patient_journeys", split_date="2026-01-01", val_days=7, test_days=3)
        )

    assert "error" not in out, out
    loader.load_for_training.assert_awaited_once_with(
        table="patient_journeys",
        filters={},
        date_column="created_at",
        split_date="2026-01-01",
        val_days=7,
        test_days=3,
        columns=None,
        include_synthetic=False,
    )
    # The presence probe asked the repository, and load_table_sample never ran.
    loader.has_column.assert_awaited_once_with("patient_journeys", "data_split")
    assert calls == []
    assert out["holdout_df"] is None  # today's behaviour: temporal path has no holdout


@pytest.mark.asyncio
async def test_string_source_on_table_with_data_split_now_uses_it() -> None:
    df = _labelled_frame(6, 2, 1, 1)
    loader, _calls = _fake_loader(df)
    with patch.object(data_loader, "get_ml_data_loader", return_value=loader):
        out = await data_loader.load_data(_state("patient_journeys"))

    assert "error" not in out, out
    loader.load_for_training.assert_not_awaited()
    assert out["holdout_df"] is not None and len(out["holdout_df"]) == 1
    assert len(out["train_df"]) == 6


# --------------------------------------------------------------------------- #
# (iv) truncation -> ValueError (a silently truncated cohort is a plausible-wrong model)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_truncated_load_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_loader, "_TABLE_ROW_LIMIT", 50)
    df = _labelled_frame(60, 20, 10, 10)  # 100 rows > the 50-row cap
    loader, _calls = _fake_loader(df)
    with patch.object(data_loader, "get_ml_data_loader", return_value=loader):
        with pytest.raises(ValueError, match="truncat"):
            await data_loader._load_from_supabase(
                data_source="patient_journeys",
                filters=_CONTRACT["filters"],
                date_column="created_at",
                entity_column=None,
                split_date=None,
                val_days=30,
                test_days=30,
                columns=_CONTRACT["columns"],
                include_synthetic=True,
            )
        out = await data_loader.load_data(_state(dict(_CONTRACT)))
    assert out["error_type"] == "data_loading_error"
    assert "truncat" in out["error"]


@pytest.mark.asyncio
async def test_empty_cohort_fails_loud() -> None:
    df = _labelled_frame(6, 2, 1, 1)
    loader, _calls = _fake_loader(df)
    contract = {**_CONTRACT, "columns": _COVARIATES + [_TARGET, "no_such_column"]}
    with patch.object(data_loader, "get_ml_data_loader", return_value=loader):
        out = await data_loader.load_data(_state(contract))
    assert out["error_type"] == "data_loading_error"
    assert "empty" in out["error"].lower()


# --------------------------------------------------------------------------- #
# (v) target missing from the contract's columns -> ValueError (catches will_adopt)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_target_missing_from_columns_fails_loud() -> None:
    df = _labelled_frame(6, 2, 1, 1)
    loader, calls = _fake_loader(df)
    contract = {**_CONTRACT, "columns": _COVARIATES}  # no treatment_initiated
    with patch.object(data_loader, "get_ml_data_loader", return_value=loader):
        out = await data_loader.load_data(_state(contract))
    assert out["error_type"] == "data_loading_error"
    assert "contract columns omit the target column" in out["error"]
    assert _TARGET in out["error"]
    # Fail before any query is spent.
    assert calls == []


# --------------------------------------------------------------------------- #
# (vi) unlabelled rows -> dropped, with a warning naming the count
# --------------------------------------------------------------------------- #
def test_split_from_column_warns_on_dropped_unlabelled_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = _labelled_frame(6, 2, 1, 1)
    df.loc[[0, 1, 2], "data_split"] = ["", None, "bogus"]
    with caplog.at_level(logging.WARNING, logger=data_loader.__name__):
        result = data_loader._split_from_column(df)
    assert len(result["train"]) == 3
    assert len(result["val"]) + len(result["test"]) + len(result["holdout"]) == 4
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "no warning emitted for dropped unlabelled rows"
    assert "3" in warnings[0].getMessage()


def test_split_from_column_no_warning_when_all_rows_labelled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=data_loader.__name__):
        data_loader._split_from_column(_labelled_frame(6, 2, 1, 1))
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


# --------------------------------------------------------------------------- #
# (vii) a 60/20/10/10-labelled table passes the REAL split_enforcer end to end
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_precomputed_table_split_passes_real_split_enforcer() -> None:
    df = _labelled_frame(3600, 1200, 600, 600)  # 6000 rows, 60/20/10/10
    loader, _calls = _fake_loader(df)
    with patch.object(data_loader, "get_ml_data_loader", return_value=loader):
        out = await data_loader.load_data(_state(dict(_CONTRACT)))
    assert "error" not in out, out

    # Exactly the split_loader math (model_trainer/nodes/split_loader.py):
    counts = {
        "train": len(out["train_df"]),
        "validation": len(out["validation_df"]),
        "test": len(out["test_df"]),
        "holdout": len(out["holdout_df"]),
    }
    total = sum(counts.values())
    state = {
        **{f"{k}_samples": v for k, v in counts.items()},
        **{f"{k}_ratio": v / total for k, v in counts.items()},
        "total_samples": total,
    }
    verdict = await enforce_splits(state)
    assert verdict["split_ratios_valid"] is True, verdict


# --------------------------------------------------------------------------- #
# unknown dict types keep failing (the file route already did)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_unknown_dict_type_still_fails() -> None:
    loader, calls = _fake_loader(_labelled_frame(6, 2, 1, 1))
    with patch.object(data_loader, "get_ml_data_loader", return_value=loader):
        out = await data_loader.load_data(_state({"type": "s3", "bucket": "x"}))
    assert out["error_type"] == "data_loading_error"
    assert calls == []


# --------------------------------------------------------------------------- #
# codex r1 MED: a probe ERROR must not read as "no data_split column"
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_probe_error_fails_loud_instead_of_silently_taking_temporal_path() -> None:
    """Only a real 42703 means "absent" (``MLDataLoader.has_column``); any other failure
    propagates, so a transient probe failure can never route a table that DOES carry
    data_split down the holdout-less temporal path."""
    df = _labelled_frame(6, 2, 1, 1)
    loader, _calls = _fake_loader(df)
    loader.has_column = AsyncMock(side_effect=ConnectionError("PostgREST unreachable"))
    with patch.object(data_loader, "get_ml_data_loader", return_value=loader):
        out = await data_loader.load_data(_state("patient_journeys"))
    assert out.get("error_type") == "data_loading_error", out
    assert "unreachable" in out["error"]
    loader.load_for_training.assert_not_awaited()
