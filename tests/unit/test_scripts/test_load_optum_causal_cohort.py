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
    fetch_live_rows,
    fetch_live_split,
    load_frame,
    main,
    to_records,
    upsert,
    verify,
    verify_rows,
)
from src.data.manifests import MART_SAFE_FEATURES  # noqa: E402


def _frame(n_x: int = 3, n_d: int = 2) -> pd.DataFrame:
    """All 81 causal-export columns: the 64 MART_SAFE_FEATURES default to 0, then
    the journey-metadata/treatment/outcome keys (and a few interesting per-row
    baseline overrides already exercised by other tests) are layered on top."""
    rows = []
    for i in range(n_x + n_d):
        dup = int(i >= n_x)
        row: dict = dict.fromkeys(MART_SAFE_FEATURES, 0)
        row.update(
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
        rows.append(row)
    return pd.DataFrame(rows)


def _write(tmp_path: Path, df: pd.DataFrame) -> Path:
    p = tmp_path / "cohort.parquet"
    df.to_parquet(p)
    return p


class _FakeQuery:
    """A faithful-enough PostgREST query stub: nothing happens until ``.execute()``
    is called, and ``.execute().count`` is only a real integer when the select
    actually asked for ``count="exact"`` — otherwise it is ``None``, exactly like a
    real client, so ``int(...)`` on a dropped ``count="exact"`` fails loudly instead
    of silently returning a filtered row count."""

    def __init__(self, table: "_FakeTable", op: str, batch=None, on_conflict=None):
        self._t, self._op, self._filters, self._count = table, op, [], None
        self._batch, self._on_conflict = batch, on_conflict
        self._range = None

    def select(self, cols, count=None):
        self._count = count
        return self

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def execute(self):
        if self._op == "upsert":
            for rec in self._batch:
                self._t.rows[rec["patient_id"]] = rec
            self._t.upserts.append((list(self._batch), self._on_conflict))
            return type("R", (), {"data": list(self._batch), "count": None})()
        # rows.values() preserves insertion order (dict semantics), so paging via
        # .range() below sees a stable, real-order slice like a live table would.
        rows = [r for r in self._t.rows.values() if all(r.get(c) == v for c, v in self._filters)]
        total = len(rows)
        if self._range is not None:
            start, end = self._range
            rows = rows[start : end + 1]
        count = total if self._count == "exact" else None
        return type("R", (), {"data": rows, "count": count})()


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
        # Building the query must not write anything -- only .execute() on the
        # returned object may (mutation-proof for a dropped .execute() call).
        return _FakeQuery(self, "upsert", batch=batch, on_conflict=on_conflict)


class _FakeTableIgnoresExactCount(_FakeTable):
    """Simulates a live select where ``count="exact"`` has no effect -- the shape
    fetch_live_split would see if it (or a client regression) stopped actually
    requesting exact counts. Upserts behave normally so the table can be populated."""

    def select(self, cols, count=None):
        if self.missing:
            raise RuntimeError('relation "optum_biologic_persistence_causal" does not exist')
        return _FakeQuery(self, "select").select(cols, count=None)


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
        # a dropped baseline confounder must refuse loud, never upsert as a silent NULL
        (lambda d: d.drop(columns=["cci_mi"]), "cci_mi"),
        (lambda d: d.assign(unexpected_extra_column="oops"), "unexpected_extra_column"),
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
    assert split["treatment"] == {"0": 3, "1": 2}


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


def test_upsert_only_writes_when_the_query_is_executed():
    """Mutation-proof for dropping .execute() from the upsert chain: building the
    query alone must not write; only calling .execute() on it may."""
    client = _FakeClient()
    batch = to_records(_frame(n_x=1, n_d=0))
    client.table(TABLE).upsert(batch, on_conflict=ON_CONFLICT)  # note: no .execute()
    assert client.t.rows == {}
    assert client.t.upserts == []


def test_fetch_live_split_counts_by_arm_and_outcome():
    client = _FakeClient()
    upsert(client, to_records(_frame(n_x=3, n_d=2)))
    live = fetch_live_split(client)
    assert live["n"] == 5 and live["arms"] == {"XOLAIR": 3, "DUPIXENT": 2}
    assert live["outcome_positives"]["persistent_at_180d_g28"] == {"XOLAIR": 2, "DUPIXENT": 1}
    assert live["treatment"] == {"0": 3, "1": 2}


def test_fetch_live_split_reports_a_missing_table_as_none():
    assert fetch_live_split(_FakeClient(missing=True)) is None


def test_fetch_live_split_degrades_to_none_when_exact_counts_are_unavailable():
    """Mutation-proof for dropping count="exact" from the live selects: when an exact
    count is unavailable, execute().count comes back None and int(None) is caught --
    the honest "unreachable" verdict, never a silently wrong split."""
    client = _FakeClient()
    client.t = _FakeTableIgnoresExactCount()
    upsert(client, to_records(_frame(n_x=3, n_d=2)))
    assert fetch_live_split(client) is None


def test_fetch_live_rows_pages_the_whole_table():
    client = _FakeClient()
    upsert(client, to_records(_frame(n_x=3, n_d=2)))
    rows = fetch_live_rows(client)
    assert rows is not None
    assert {r["patient_id"] for r in rows} == {f"PAT_{i}" for i in range(5)}


def test_fetch_live_rows_reports_a_missing_table_as_none():
    assert fetch_live_rows(_FakeClient(missing=True)) is None


def test_verify_rows_catches_field_corruption_and_id_drift():
    """Mutation-proof for the gap aggregate verification cannot see: wrong
    patient_ids and corrupted covariates must surface even when arm/outcome
    totals still match."""
    records = to_records(_frame(n_x=3, n_d=2))
    live = [dict(r) for r in records]
    live[0]["age_at_index"] = (live[0]["age_at_index"] or 0) + 999  # corrupted covariate
    live[1]["patient_id"] = "PAT_changed"  # id drift: PAT_1 vanishes, PAT_changed appears
    live.append({**records[0], "patient_id": "PAT_extra"})  # an extra live row

    problems = verify_rows(records, live)
    assert any("age_at_index" in p for p in problems)
    assert any("PAT_1" in p for p in problems)
    assert any("PAT_changed" in p for p in problems)
    assert any("PAT_extra" in p for p in problems)


def test_verify_rows_agrees_on_identical_data():
    records = to_records(_frame(n_x=3, n_d=2))
    assert verify_rows(records, [dict(r) for r in records]) == []


def test_verify_verdicts():
    a = arm_split(_frame(n_x=3, n_d=2))
    assert verify(a, a) == []
    b = arm_split(_frame(n_x=3, n_d=1))
    problems = verify(a, b)
    assert any("DUPIXENT" in p for p in problems) and any("n" in p for p in problems)
    assert any("treatment=1" in p for p in problems)


def test_main_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, _frame())
    client = _FakeClient()
    import scripts.load_optum_causal_cohort as mod

    monkeypatch.setattr(mod, "_client", lambda: client)
    assert main(["--input", str(path)]) == 0
    assert client.t.upserts == []
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "XOLAIR" in out


def test_main_dry_run_flag_is_an_explicit_alias(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, _frame())
    client = _FakeClient()
    import scripts.load_optum_causal_cohort as mod

    monkeypatch.setattr(mod, "_client", lambda: client)
    assert main(["--input", str(path), "--dry-run"]) == 0
    assert client.t.upserts == []
    assert "DRY RUN" in capsys.readouterr().out


def test_main_rejects_dry_run_and_execute_together(tmp_path):
    path = _write(tmp_path, _frame())
    with pytest.raises(SystemExit) as exc_info:
        main(["--input", str(path), "--dry-run", "--execute"])
    assert exc_info.value.code == 2


def test_main_execute_loads_then_verifies(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, _frame())
    client = _FakeClient()
    import scripts.load_optum_causal_cohort as mod

    monkeypatch.setattr(mod, "_client", lambda: client)
    assert main(["--input", str(path), "--execute"]) == 0
    assert len(client.t.rows) == 5
    out = capsys.readouterr().out
    assert "VERIFIED" in out
    assert "compared 5 exported rows" in out


def test_main_execute_returns_nonzero_when_a_live_row_is_corrupted(tmp_path, monkeypatch, capsys):
    """Mutation-proof for verify() alone: the aggregate arm/outcome/treatment
    margins still match (same rows, same brand/outcome values) but a covariate on
    one live row is wrong -- row-level verification must still fail the run."""
    path = _write(tmp_path, _frame())
    client = _FakeClient()
    import scripts.load_optum_causal_cohort as mod

    monkeypatch.setattr(mod, "_client", lambda: client)
    real_fetch_live_rows = mod.fetch_live_rows

    def _corrupting_fetch_live_rows(c):
        rows = real_fetch_live_rows(c)
        if rows:
            rows[0] = {**rows[0], "age_at_index": (rows[0]["age_at_index"] or 0) + 999}
        return rows

    monkeypatch.setattr(mod, "fetch_live_rows", _corrupting_fetch_live_rows)
    assert main(["--input", str(path), "--execute"]) == 1
    assert "MISMATCH" in capsys.readouterr().out


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
