"""scripts/load_optum_causal_cohort.py — parquet -> optum_biologic_persistence_causal.

No DB here: the client is a recording fake. The real-DB arm-split probe lives in
tests/integration/test_optum_causal_cohort_realdb.py (E2I_DB_INTEGRATION=1).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.load_optum_causal_cohort import (  # noqa: E402
    BATCH_SIZE,
    ON_CONFLICT,
    OUTCOME_COLUMNS,
    TABLE,
    arm_split,
    fetch_live_split,
    load_frame,
    main,
    to_records,
    upsert,
    verify,
)


def _frame(n_x: int = 3, n_d: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_x + n_d):
        dup = int(i >= n_x)
        rows.append(
            {
                "patient_journey_id": f"PJ_{i}",
                "patient_id": f"PAT_{i}",
                "patient_hash": f"{i:020x}",
                "index_date": pd.Timestamp("2020-01-01"),
                "journey_start_date": pd.Timestamp("2020-01-01"),
                "journey_status": "active",
                "discontinuation_flag": 0,
                "data_quality_score": 0.98,
                "data_split": "train",
                "index_biologic_brand": "DUPIXENT" if dup else "XOLAIR",
                "treatment_dupixent": dup,
                "treatment_start_date": pd.Timestamp("2020-01-01"),
                "persistent_at_180d_g28": 1 if i % 2 == 0 else 0,
                "discontinued_180d": 0,
                "biologic_switch_180d_flag": 0,
                "persistent_at_180d": 1 if i % 3 == 0 else 0,
                "age_at_index": 50.0 + i,
                "geographic_region": None if i == 0 else "south",
                "charlson_score": np.int64(i),
                "is_synthetic": False,
            }
        )
    return pd.DataFrame(rows)


def _write(tmp_path: Path, df: pd.DataFrame) -> Path:
    p = tmp_path / "cohort.parquet"
    df.to_parquet(p)
    return p


class _FakeQuery:
    def __init__(self, table: "_FakeTable", op: str):
        self._t, self._op, self._filters, self._count = table, op, [], None

    def select(self, cols, count=None):
        self._count = count
        return self

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def execute(self):
        rows = [r for r in self._t.rows.values() if all(r.get(c) == v for c, v in self._filters)]
        return type("R", (), {"data": rows, "count": len(rows)})()


class _FakeTable:
    def __init__(self, missing: bool = False):
        self.rows: dict = {}
        self.upserts: list = []
        self.missing = missing

    def select(self, cols, count=None):
        if self.missing:
            raise RuntimeError('relation "optum_biologic_persistence_causal" does not exist')
        return _FakeQuery(self, "select").select(cols, count=count)

    def upsert(self, batch, on_conflict=None):
        self.upserts.append((list(batch), on_conflict))
        for rec in batch:
            self.rows[rec["patient_id"]] = rec
        return _FakeQuery(self, "upsert")


class _FakeClient:
    def __init__(self, missing: bool = False):
        self.t = _FakeTable(missing=missing)
        self.tables: list = []

    def table(self, name):
        self.tables.append(name)
        return self.t


def test_constants():
    assert TABLE == "optum_biologic_persistence_causal"
    assert ON_CONFLICT == "patient_id"
    assert BATCH_SIZE == 500
    assert OUTCOME_COLUMNS == (
        "persistent_at_180d_g28",
        "discontinued_180d",
        "biologic_switch_180d_flag",
        "persistent_at_180d",
    )


def test_load_frame_accepts_the_export(tmp_path):
    df = load_frame(_write(tmp_path, _frame()))
    assert len(df) == 5


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d.assign(is_synthetic=True), "is_synthetic"),
        (lambda d: pd.concat([d, d.iloc[[0]]]), "patient_id"),
        (lambda d: d.drop(columns=["persistent_at_180d_g28"]), "persistent_at_180d_g28"),
        (
            lambda d: d.assign(treatment_dupixent=1 - d["treatment_dupixent"]),
            "index_biologic_brand",
        ),
        (lambda d: d.assign(index_biologic_brand="RHAPSIDO"), "index_biologic_brand"),
        (lambda d: d.assign(persistent_at_180d_g28=2), "persistent_at_180d_g28"),
    ],
)
def test_load_frame_fails_loud(tmp_path, mutate, match):
    with pytest.raises(ValueError, match=match):
        load_frame(_write(tmp_path, mutate(_frame())))


def test_arm_split_counts_and_rates():
    split = arm_split(_frame(n_x=3, n_d=2))
    assert split["n"] == 5
    assert split["arms"] == {"XOLAIR": 3, "DUPIXENT": 2}
    assert split["outcome_positives"]["persistent_at_180d_g28"] == {"XOLAIR": 2, "DUPIXENT": 1}
    assert set(split["outcome_positives"]) == set(OUTCOME_COLUMNS)


def test_to_records_is_json_safe_and_deterministic():
    r1, r2 = to_records(_frame()), to_records(_frame())
    assert r1 == r2
    rec = r1[0]
    assert rec["index_date"] == "2020-01-01" and date.fromisoformat(rec["treatment_start_date"])
    assert rec["geographic_region"] is None  # NaN/None -> null, never the string 'nan'
    assert isinstance(rec["charlson_score"], int) and not isinstance(
        rec["charlson_score"], np.generic
    )
    assert isinstance(rec["treatment_dupixent"], int)
    assert rec["is_synthetic"] is False
    assert isinstance(rec["age_at_index"], float)


def test_upsert_batches_on_patient_id():
    client = _FakeClient()
    n = upsert(client, to_records(_frame(n_x=600, n_d=100)), batch_size=500)
    assert n == 700
    assert [len(b) for b, _ in client.t.upserts] == [500, 200]
    assert {oc for _, oc in client.t.upserts} == {ON_CONFLICT}
    assert client.tables and set(client.tables) == {TABLE}
    # idempotent: a second run rewrites the same keys, no growth
    upsert(client, to_records(_frame(n_x=600, n_d=100)), batch_size=500)
    assert len(client.t.rows) == 700


def test_fetch_live_split_counts_by_arm_and_outcome():
    client = _FakeClient()
    upsert(client, to_records(_frame(n_x=3, n_d=2)))
    live = fetch_live_split(client)
    assert live["n"] == 5 and live["arms"] == {"XOLAIR": 3, "DUPIXENT": 2}
    assert live["outcome_positives"]["persistent_at_180d_g28"] == {"XOLAIR": 2, "DUPIXENT": 1}


def test_fetch_live_split_reports_a_missing_table_as_none():
    assert fetch_live_split(_FakeClient(missing=True)) is None


def test_verify_verdicts():
    a = arm_split(_frame(n_x=3, n_d=2))
    assert verify(a, a) == []
    b = arm_split(_frame(n_x=3, n_d=1))
    problems = verify(a, b)
    assert any("DUPIXENT" in p for p in problems) and any("n" in p for p in problems)


def test_main_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, _frame())
    client = _FakeClient()
    import scripts.load_optum_causal_cohort as mod

    monkeypatch.setattr(mod, "_client", lambda: client)
    assert main(["--input", str(path)]) == 0
    assert client.t.upserts == []
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "XOLAIR" in out


def test_main_execute_loads_then_verifies(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, _frame())
    client = _FakeClient()
    import scripts.load_optum_causal_cohort as mod

    monkeypatch.setattr(mod, "_client", lambda: client)
    assert main(["--input", str(path), "--execute"]) == 0
    assert len(client.t.rows) == 5
    assert "VERIFIED" in capsys.readouterr().out


def test_main_execute_returns_nonzero_when_live_split_disagrees(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, _frame())
    client = _FakeClient()
    import scripts.load_optum_causal_cohort as mod

    monkeypatch.setattr(mod, "_client", lambda: client)
    # a stale extra row the parquet does not carry
    client.t.rows["PAT_stale"] = {
        "patient_id": "PAT_stale",
        "index_biologic_brand": "XOLAIR",
        "treatment_dupixent": 0,
        **dict.fromkeys(OUTCOME_COLUMNS, 0),
    }
    assert main(["--input", str(path), "--execute"]) == 1
    assert "MISMATCH" in capsys.readouterr().out
