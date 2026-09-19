"""#1833 reseed script — offline (pure) parts.

The DB-facing parts (reading the live aggregate rows, the upsert) are
exercised by running ``--dry-run`` against the real local DB; the frame
assembly and the diff summary are pure and pinned here.
"""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.reseed_business_metrics_aggregate as reseed
import src.ml.synthetic.frontier_append as fa
from scripts.reseed_business_metrics_aggregate import (
    AGGREGATE_METRIC_NAMES,
    build_reseed_frame,
    diff_summary,
    execute_refusal,
)


def _summary(**overrides):
    base = {
        "target_changed": 0,
        "ids_only_in_db": [],
        "ids_only_in_regen": [],
        "new_series_ids": [],
    }
    base.update(overrides)
    return base


class TestExecuteGuard:
    """``--execute`` must FAIL CLOSED on id drift in EITHER direction (codex
    iter-1 BLOCKER: the first cut refused only stale DB ids, so running after
    a month boundary but before the Mon-3AM cron would silently INSERT the
    not-yet-appended cohort rows instead of doing an in-place reseed)."""

    @staticmethod
    def _summary(**over):
        base = {"ids_only_in_regen": [], "ids_only_in_db": [], "target_changed": 0}
        base.update(over)
        return base

    def test_clean_summary_is_allowed(self):
        assert execute_refusal(self._summary()) is None

    def test_regen_ids_absent_from_db_refuse_by_default(self):
        # the codex scenario: --frontier 2026-09-01 before the cron appended m2609_*
        reason = execute_refusal(self._summary(ids_only_in_regen=["m2609_0000", "m2609_0001"]))
        assert reason is not None
        assert "2 regenerated ids" in reason and "cron" in reason
        assert "--allow-new-cohorts" in reason

    def test_regen_ids_absent_from_db_need_the_explicit_opt_in(self):
        s = self._summary(ids_only_in_regen=["m2609_0000"])
        assert execute_refusal(s, allow_id_drift=True) is not None  # wrong flag
        assert execute_refusal(s, allow_new_cohorts=True) is None

    def test_stale_db_ids_refuse_by_default(self):
        reason = execute_refusal(self._summary(ids_only_in_db=["metric_deadbeef0000"]))
        assert reason is not None
        assert "1 aggregate ids in the DB" in reason and "--allow-id-drift" in reason

    def test_stale_db_ids_need_the_explicit_opt_in(self):
        s = self._summary(ids_only_in_db=["metric_deadbeef0000"])
        assert execute_refusal(s, allow_new_cohorts=True) is not None  # wrong flag
        assert execute_refusal(s, allow_id_drift=True) is None

    def test_target_drift_is_never_allowed(self):
        s = self._summary(target_changed=3)
        assert execute_refusal(s, allow_id_drift=True, allow_new_cohorts=True) is not None
        assert "3 targets" in execute_refusal(s)


class TestBuildReseedFrame:
    def test_covers_base_and_cohort_months_through_frontier(self):
        frame = build_reseed_frame(frontier=date(2026, 8, 30))
        base = frame[frame["metric_id"].str.startswith("metric_")]
        cohort = frame[frame["metric_id"].str.startswith("m2608_")]
        nbrx = frame[frame["metric_id"].str.startswith("nbrx_")]
        assert len(base) == 9780
        assert base["metric_date"].max() == "2026-07-01"
        assert len(cohort) == 60
        # 163 base months + the 2026-08 cohort month, 12 cells each
        assert len(nbrx) == 164 * 12
        assert len(frame) == 9780 + 60 + 164 * 12
        assert frame["metric_id"].is_unique
        assert frame["is_synthetic"].all()
        assert frame["value"].notna().all()
        assert set(frame["metric_name"]) == set(AGGREGATE_METRIC_NAMES)

    def test_cohort_rows_are_the_cron_cohort(self):
        frame = build_reseed_frame(frontier=date(2026, 9, 15))
        cohort = fa.generate_month_cohort(date(2026, 9, 1))["business_metrics"]
        for prefix in ("m2609_", "nbrx_202609_"):
            got = frame[frame["metric_id"].str.startswith(prefix)].reset_index(drop=True)
            want = cohort[cohort["metric_id"].str.startswith(prefix)].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                got[["metric_id", "metric_date", "value", "target"]],
                want[["metric_id", "metric_date", "value", "target"]],
            )

    def test_frontier_before_epoch_is_base_only(self):
        frame = build_reseed_frame(frontier=date(2026, 7, 31))
        assert len(frame) == 9780 + 1956


class TestNewSeries:
    def test_nbrx_is_an_aggregate_metric_name(self):
        assert "nbrx" in AGGREGATE_METRIC_NAMES

    def test_nbrx_ids_absent_from_the_db_are_a_new_series_not_a_cohort(self):
        db = _rows(("a", "2026-07-01", "Kisqali", "midwest", "trx", 100.0, 110.0))
        regen = _rows(
            ("a", "2026-07-01", "Kisqali", "midwest", "trx", 100.0, 110.0),
            ("nbrx_202607_kisqali_midwest", "2026-07-01", "Kisqali", "midwest", "nbrx", 9.0, 10.0),
        )
        s = diff_summary(db, regen, scale_month="2026-07-01")
        assert s["ids_only_in_regen"] == []
        assert s["new_series_ids"] == ["nbrx_202607_kisqali_midwest"]

    def test_new_series_refuse_without_the_opt_in(self):
        s = _summary(new_series_ids=["nbrx_202607_kisqali_midwest"])
        refusal = execute_refusal(s)
        assert refusal is not None and "--allow-new-series" in refusal
        assert execute_refusal(s, allow_new_series=True) is None

    def test_the_cohort_opt_in_does_not_admit_a_new_series(self):
        s = _summary(new_series_ids=["nbrx_202607_kisqali_midwest"])
        assert execute_refusal(s, allow_new_cohorts=True, allow_id_drift=True) is not None


def _rows(*specs):
    cols = ["metric_id", "metric_date", "brand", "region", "metric_name", "value", "target"]
    return pd.DataFrame([dict(zip(cols, s, strict=True)) for s in specs])


class TestDiffSummary:
    def test_counts_and_scale(self):
        db = _rows(
            ("a", "2026-07-01", "Kisqali", "midwest", "trx", 100.0, 110.0),
            ("b", "2026-07-01", "Kisqali", "west", "trx", 100.0, 110.0),
            ("c", "2026-07-01", "Kisqali", "west", "market_share", 0.5, 0.6),
            ("stale", "2026-07-01", "Fabhalta", "west", "trx", 5.0, 6.0),
        )
        regen = _rows(
            ("a", "2026-07-01", "Kisqali", "midwest", "trx", 88.0, 110.0),
            ("b", "2026-07-01", "Kisqali", "west", "trx", 100.0, 110.0),
            ("c", "2026-07-01", "Kisqali", "west", "market_share", 0.5, 0.6),
            ("new", "2026-08-01", "Kisqali", "west", "trx", 120.0, 130.0),
        )
        s = diff_summary(db, regen, scale_month="2026-07-01")
        assert s["rows_to_upsert"] == 4
        assert s["ids_only_in_regen"] == ["new"]
        assert s["ids_only_in_db"] == ["stale"]
        assert s["value_changed"] == 1
        assert s["value_unchanged"] == 2
        assert s["target_changed"] == 0
        # per-brand national trx scale for the month, before -> after
        assert s["national_trx"]["Kisqali"] == {"before": 200.0, "after": 188.0, "ratio": 0.94}
        # a brand only present on one side is reported, not dropped
        assert s["national_trx"]["Fabhalta"]["before"] == 5.0
        assert s["national_trx"]["Fabhalta"]["after"] == 0.0

    def test_identical_frames_are_a_no_op(self):
        db = _rows(("a", "2026-07-01", "Kisqali", "midwest", "trx", 100.0, 110.0))
        s = diff_summary(db, db.copy(), scale_month="2026-07-01")
        assert s["value_changed"] == 0 and s["target_changed"] == 0
        assert s["ids_only_in_db"] == [] and s["ids_only_in_regen"] == []


# ---------------------------------------------------------------------------
# The --execute path end to end (codex Tasks 4+5 review): the real main(),
# fetch, diff_summary, refusals and verification over an in-memory table. Only
# the Supabase client and the BatchLoader are replaced; build_reseed_frame is
# patched to a tiny frame because it only supplies the regenerated rows.
# ---------------------------------------------------------------------------
_LEGACY = ("metric_aaaaaaaaaaaa", "2026-07-01", "Kisqali", "midwest", "trx", 100.0, 110.0)
_COHORT = ("m2608_0000", "2026-08-01", "Kisqali", "midwest", "trx", 101.0, 111.0)
_NBRX = ("nbrx_202607_kisqali_midwest", "2026-07-01", "Kisqali", "midwest", "nbrx", 9.0, 10.0)


class _FakeQuery:
    """The PostgREST subset the script uses: select / in_ / order / range / execute.
    ``in_`` never matches NULL, like SQL ``IN``. ``cap`` truncates every response
    to that many rows, like a server-side max-rows setting."""

    def __init__(self, rows, cap=None):
        self._rows = rows
        self._cap = cap

    def select(self, _columns):
        return self

    def in_(self, column, values):
        wanted = set(values)
        return _FakeQuery([r for r in self._rows if r.get(column) in wanted], self._cap)

    def order(self, column):
        return _FakeQuery(sorted(self._rows, key=lambda r: r[column]), self._cap)

    def range(self, start, end):
        return _FakeQuery(self._rows[start : end + 1], self._cap)

    def execute(self):
        rows = self._rows if self._cap is None else self._rows[: self._cap]
        return SimpleNamespace(data=[dict(r) for r in rows])


class _FakeClient:
    def __init__(self, rows, cap=None):
        self.rows = [dict(r) for r in rows]
        self.lookups = 0
        self._cap = cap

    def table(self, name):
        assert name == "business_metrics"
        self.lookups += 1
        return _FakeQuery(self.rows, self._cap)


class _FakeLoader:
    """Upserts on metric_id into the fake table; ``drop_ids`` are silently not
    written (an incomplete upsert the re-read must catch) and ``target_shift_ids``
    are written with target + 1 (a target the re-read must catch)."""

    def __init__(self, client, drop_ids=(), target_shift_ids=()):
        self.client = client
        self.calls = []
        self._drop = set(drop_ids)
        self._shift = set(target_shift_ids)

    def load_table(self, table, df):
        self.calls.append((table, len(df)))
        by_id = {r["metric_id"]: r for r in self.client.rows}
        for rec in df.to_dict("records"):
            if rec["metric_id"] in self._drop:
                continue
            row = {c: rec[c] for c in reseed.DIFF_COLUMNS}
            if rec["metric_id"] in self._shift:
                row["target"] = row["target"] + 1.0
            by_id[rec["metric_id"]] = row
        self.client.rows = list(by_id.values())
        return SimpleNamespace(
            records_loaded=len(df) - len(self._drop), records_failed=0, total_batches=1, errors=[]
        )


def _run_main(
    monkeypatch, db_specs, regen_specs, flags, drop_ids=(), raw_db_rows=(), target_shift_ids=()
):
    regen = _rows(*regen_specs)
    regen["is_synthetic"] = True
    monkeypatch.setattr(reseed, "build_reseed_frame", lambda frontier: regen)
    client = _FakeClient(_rows(*db_specs).to_dict("records") + list(raw_db_rows))
    loader = _FakeLoader(client, drop_ids, target_shift_ids)
    code = reseed.main(["--execute", "--frontier", "2026-08-30", *flags], loader=loader)
    return code, loader


class TestExecutePath:
    def test_an_existing_nbrx_row_with_a_changed_target_refuses_even_with_the_opt_in(
        self, monkeypatch
    ):
        moved = (*_NBRX[:6], 12.0)
        s = diff_summary(_rows(_LEGACY, _NBRX), _rows(_LEGACY, moved), scale_month="2026-07-01")
        assert s["target_changed"] == 1 and s["new_series_ids"] == []
        assert execute_refusal(s, allow_new_series=True) is not None
        code, loader = _run_main(
            monkeypatch, [_LEGACY, _NBRX], [_LEGACY, moved], ["--allow-new-series"]
        )
        assert code == 3 and loader.calls == []

    def test_every_opt_in_together_still_refuses_a_target_change(self, monkeypatch):
        moved = (*_LEGACY[:6], 115.0)
        code, loader = _run_main(
            monkeypatch,
            [_LEGACY],
            [moved, _COHORT, _NBRX],
            ["--allow-id-drift", "--allow-new-cohorts", "--allow-new-series"],
        )
        assert code == 3 and loader.calls == []

    @pytest.mark.parametrize(
        ("flags", "code", "calls"),
        [([], 3, 0), (["--allow-new-series"], 0, 1)],
    )
    def test_a_refusal_happens_before_load_table(self, monkeypatch, flags, code, calls):
        got, loader = _run_main(monkeypatch, [_LEGACY], [_LEGACY, _NBRX], flags)
        assert got == code
        assert len(loader.calls) == calls

    @pytest.mark.parametrize(
        ("dropped", "line"),
        [
            (_COHORT[0], "missing ids=1 missing new-series ids=0"),
            (_NBRX[0], "missing ids=0 missing new-series ids=1"),
        ],
    )
    def test_an_incomplete_upsert_exits_1_and_names_both_missing_counts(
        self, monkeypatch, capsys, dropped, line
    ):
        code, loader = _run_main(
            monkeypatch,
            [_LEGACY],
            [_LEGACY, _COHORT, _NBRX],
            ["--allow-new-cohorts", "--allow-new-series"],
            drop_ids=[dropped],
        )
        assert len(loader.calls) == 1
        assert code == 1
        assert line in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("spec", "metric_name"),
        [(_NBRX, None), (_COHORT, "per_hcp_rollup")],
    )
    def test_an_insert_that_collides_with_a_non_aggregate_row_refuses_before_load(
        self, monkeypatch, spec, metric_name
    ):
        # The aggregate read filters on metric_name, so this row is invisible to
        # diff_summary and would be classed as a new series / new cohort.
        raw = {**_rows(spec).to_dict("records")[0], "metric_name": metric_name}
        code, loader = _run_main(
            monkeypatch,
            [_LEGACY],
            [_LEGACY, spec],
            ["--allow-new-cohorts", "--allow-new-series"],
            raw_db_rows=[raw],
        )
        assert code == 4 and loader.calls == []

    def test_the_id_lookup_ignores_metric_name_and_pages_in_chunks(self):
        rows = [
            {"metric_id": f"nbrx_20260{m}_kisqali_west", "metric_name": None} for m in range(1, 6)
        ]
        client = _FakeClient(rows)
        ids = [r["metric_id"] for r in rows] + ["nbrx_202612_kisqali_west"]
        found = reseed.existing_metric_ids(client, ids, chunk_size=2)
        assert found == sorted(r["metric_id"] for r in rows)
        # one page plus the terminating empty page per chunk (cap-agnostic paging)
        assert client.lookups == 6

    def test_the_id_lookup_pages_past_a_per_response_row_cap(self):
        rows = [
            {"metric_id": f"nbrx_20260{m}_kisqali_west", "metric_name": None} for m in range(1, 8)
        ]
        client = _FakeClient(rows, cap=3)
        ids = [r["metric_id"] for r in rows]
        found = reseed.existing_metric_ids(client, ids)  # a single 100-id chunk
        assert found == sorted(ids)
        assert "nbrx_202607_kisqali_west" in found  # beyond the first 3-row response

    def test_an_existing_nbrx_row_with_a_null_db_target_refuses_before_load(self, monkeypatch):
        null_target = (*_NBRX[:6], None)
        code, loader = _run_main(
            monkeypatch, [_LEGACY, null_target], [_LEGACY, _NBRX], ["--allow-new-series"]
        )
        assert code == 3 and loader.calls == []

    def test_a_target_altered_on_re_read_exits_1(self, monkeypatch, capsys):
        code, loader = _run_main(
            monkeypatch,
            [_LEGACY],
            [_LEGACY, _NBRX],
            ["--allow-new-series"],
            target_shift_ids=[_NBRX[0]],
        )
        assert len(loader.calls) == 1
        assert code == 1
        assert "target mismatches=1" in capsys.readouterr().out


class TestNullTransitions:
    """NaN arithmetic compares False, so a null on exactly one side must be
    counted explicitly (codex r2)."""

    @staticmethod
    def _frame(column, value):
        spec = list(_LEGACY)
        spec[5 if column == "value" else 6] = value
        frame = _rows(tuple(spec))
        # the numeric coercion fetch_db_aggregate_rows applies to live rows
        frame[["value", "target"]] = frame[["value", "target"]].astype(float)
        return frame

    @pytest.mark.parametrize("column", ["value", "target"])
    @pytest.mark.parametrize(
        ("db_value", "regen_value", "changed"),
        [(None, 10.0, 1), (10.0, None, 1), (None, None, 0)],
    )
    def test_a_null_on_exactly_one_side_is_a_change(self, column, db_value, regen_value, changed):
        s = diff_summary(
            self._frame(column, db_value),
            self._frame(column, regen_value),
            scale_month="2026-07-01",
        )
        assert s[f"{column}_changed"] == changed
