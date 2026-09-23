"""R5 outcome feed — pure aggregation logic for ExperimentOutcomeRepository.

These tests exercise the REAL aggregation logic (no mocks) over real-shaped
business_metrics ``per_hcp_rollup`` rows: map a primary_metric to its typed
column, collapse multiple metric_date rows per HCP to one scalar (SUM for
counts, MEAN for rates), then split per-unit values by assignment variant into
the (control, treatment) arrays that ResultsAnalysisService._compute_results
consumes. DB I/O is integration-tested separately (gated, real Supabase).
"""

from __future__ import annotations

import numpy as np
import pytest


def _repo():
    from src.repositories.experiment_outcome import ExperimentOutcomeRepository

    # resolve_column/aggregate_to_arrays are @staticmethod — exercise them on the
    # CLASS so no Supabase client is resolved (key-less CI would otherwise raise
    # ServiceConnectionError at construction).
    return ExperimentOutcomeRepository


class TestResolveColumn:
    # test_count_metrics_map_to_sum was removed by the canonical TRx lane
    # (migration 144): it asserted that trx / nrx / total_rx and the three legacy
    # *_count names RESOLVE. That contract is inverted now — all six of those
    # values are cases of TestHonestTriggerColumns's refusal test, and the
    # counts-SUM behaviour it covered is carried by
    # TestHonestTriggerColumns.test_trigger_counts_resolve_and_sum.

    def test_rate_metrics_map_to_mean(self):
        repo = _repo()
        for metric in ("market_share", "conversion_rate", "engagement_score", "call_frequency"):
            column, reducer = repo.resolve_column(metric)
            assert reducer == "mean", f"{metric} should mean"
            assert column == metric

    def test_unknown_metric_fails_closed(self):
        repo = _repo()
        with pytest.raises(ValueError):
            repo.resolve_column("adoption_propensity")

    def test_the_twin_cohort_outcome_resolves_to_its_own_column_and_means(self):
        """#2206 item C: a draft experiment created from a twin proposal carries
        prediction_target = the outcome the twin predicted an effect ON
        (business_metrics.cohort_conversion_outcome, migration 147). The final
        results feed must measure the SAME quantity, else the loop terminates in
        an honest "Unsupported primary_metric" skip. It is a per-HCP numeric
        outcome (NOT a count), so the per-HCP window collapses by MEAN."""
        from src.data.per_hcp_cohort_columns import COHORT_OUTCOME_COLUMN

        repo = _repo()
        column, reducer = repo.resolve_column(COHORT_OUTCOME_COLUMN)
        assert column == COHORT_OUTCOME_COLUMN == "cohort_conversion_outcome"
        assert reducer == "mean"


class TestAggregateToArrays:
    def test_sums_count_values_per_hcp_then_splits_by_variant(self):
        repo = _repo()
        assignments = [
            ("HCP_1", "control"),
            ("HCP_2", "control"),
            ("HCP_3", "treatment"),
        ]
        # Two date rows for HCP_1 -> summed to 3; HCP_2 -> 5; HCP_3 -> 9.
        rows = [
            {"hcp_id": "HCP_1", "triggers_delivered_count": 1},
            {"hcp_id": "HCP_1", "triggers_delivered_count": 2},
            {"hcp_id": "HCP_2", "triggers_delivered_count": 5},
            {"hcp_id": "HCP_3", "triggers_delivered_count": 4},
            {"hcp_id": "HCP_3", "triggers_delivered_count": 5},
        ]
        control, treatment = repo.aggregate_to_arrays(
            assignments, rows, column="triggers_delivered_count", reducer="sum"
        )
        assert sorted(control.tolist()) == [3.0, 5.0]
        assert treatment.tolist() == [9.0]

    def test_means_rate_values_per_hcp(self):
        repo = _repo()
        assignments = [("HCP_1", "control"), ("HCP_2", "treatment")]
        rows = [
            {"hcp_id": "HCP_1", "market_share": 0.2},
            {"hcp_id": "HCP_1", "market_share": 0.4},  # mean 0.3
            {"hcp_id": "HCP_2", "market_share": 0.8},
        ]
        control, treatment = repo.aggregate_to_arrays(
            assignments, rows, column="market_share", reducer="mean"
        )
        assert control.tolist() == pytest.approx([0.3])
        assert treatment.tolist() == pytest.approx([0.8])

    def test_skips_null_outcome_values(self):
        repo = _repo()
        assignments = [("HCP_1", "control"), ("HCP_2", "treatment")]
        # HCP_2 has only NULLs -> excluded entirely (no NaN in the array).
        rows = [
            {"hcp_id": "HCP_1", "triggers_delivered_count": 7},
            {"hcp_id": "HCP_2", "triggers_delivered_count": None},
        ]
        control, treatment = repo.aggregate_to_arrays(
            assignments, rows, column="triggers_delivered_count", reducer="sum"
        )
        assert control.tolist() == [7.0]
        assert treatment.size == 0
        assert not np.isnan(control).any()

    def test_empty_when_no_assignments(self):
        repo = _repo()
        control, treatment = repo.aggregate_to_arrays(
            [],
            [{"hcp_id": "HCP_1", "triggers_delivered_count": 1}],
            column="triggers_delivered_count",
            reducer="sum",
        )
        assert control.size == 0 and treatment.size == 0

    def test_unassigned_hcp_metrics_ignored(self):
        repo = _repo()
        assignments = [("HCP_1", "control"), ("HCP_2", "treatment")]
        rows = [
            {"hcp_id": "HCP_1", "triggers_delivered_count": 1},
            {"hcp_id": "HCP_2", "triggers_delivered_count": 2},
            {"hcp_id": "HCP_999", "triggers_delivered_count": 100},  # not assigned -> ignored
        ]
        control, treatment = repo.aggregate_to_arrays(
            assignments, rows, column="triggers_delivered_count", reducer="sum"
        )
        assert control.tolist() == [1.0]
        assert treatment.tolist() == [2.0]


# ----------------------------------------------------------- provenance (R6)
class _FakeQuery:
    """Records ``.eq`` calls against the business_metrics query builder."""

    def __init__(self, data, eq_log):
        self._data = data
        self._eq_log = eq_log

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        self._eq_log.append(a)
        return self

    def in_(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def execute(self):
        class _R:
            pass

        r = _R()
        rng = getattr(self, "_range", None)
        r.data = self._data if rng is None else self._data[rng[0] : rng[1] + 1]
        r.count = len(self._data)
        return r


class _FakeClient:
    """Three-table fake: assignments table yields one assignment, the unit
    outcome feed (migration 155) is EMPTY so the legacy path runs, and
    business_metrics records its ``.eq`` calls so the provenance predicate can
    be asserted."""

    def __init__(self, bm_eq_log):
        self._bm_eq_log = bm_eq_log

    def table(self, name):
        if name == "ab_experiment_assignments":
            return _FakeQuery([{"unit_id": "HCP_1", "variant": "control"}], [])
        if name == "ab_experiment_unit_outcomes":
            return _FakeQuery([], [])
        # business_metrics
        return _FakeQuery(
            [
                {
                    "hcp_id": "HCP_1",
                    "triggers_delivered_count": 5,
                    "metric_date": "2025-01-01",
                    "brand": "Kisqali",
                }
            ],
            self._bm_eq_log,
        )


class TestLoadArraysProvenance:
    """R6: business_metrics join default-excludes synthetic per-HCP rollups."""

    def test_load_arrays_query_default_excludes_synthetic(self):
        import asyncio
        from uuid import uuid4

        from src.repositories.experiment_outcome import ExperimentOutcomeRepository

        bm_eq_log: list = []
        repo = ExperimentOutcomeRepository(supabase_client=_FakeClient(bm_eq_log))
        asyncio.run(repo.load_arrays(uuid4(), "triggers_delivered"))

        assert ("is_synthetic", False) in bm_eq_log

    def test_load_arrays_query_includes_synthetic_when_opted_in(self):
        import asyncio
        from uuid import uuid4

        from src.repositories.experiment_outcome import ExperimentOutcomeRepository

        bm_eq_log: list = []
        repo = ExperimentOutcomeRepository(supabase_client=_FakeClient(bm_eq_log))
        asyncio.run(repo.load_arrays(uuid4(), "triggers_delivered", include_synthetic=True))

        assert ("is_synthetic", False) not in bm_eq_log


class TestHonestTriggerColumns:
    """Canonical TRx lane: per-HCP rows carry trigger funnel counts, not prescriptions.

    Measured 2026-09-17 on prod: ml_experiments holds 1,068 rows and NONE has a
    prediction_target among the eight prescription shorthands refused below, so
    refusing them breaks no stored experiment. That
    is a SNAPSHOT of today's data, not an invariant — a future row could use one,
    and the refusal below is what makes that case loud instead of silently
    measuring trigger deliveries as if they were prescriptions.
    """

    @pytest.mark.parametrize(
        "metric,column",
        [
            ("triggers_delivered", "triggers_delivered_count"),
            ("triggers_delivered_count", "triggers_delivered_count"),
            ("triggers_accepted", "triggers_accepted_count"),
            ("triggers_accepted_count", "triggers_accepted_count"),
            ("triggers_total", "triggers_total_count"),
            ("TRIGGERS_TOTAL", "triggers_total_count"),
        ],
    )
    def test_trigger_counts_resolve_and_sum(self, metric, column):
        from src.repositories.experiment_outcome import ExperimentOutcomeRepository

        assert ExperimentOutcomeRepository.resolve_column(metric) == (column, "sum")

    @pytest.mark.parametrize(
        "metric",
        [
            "trx",
            "nrx",
            "rx",
            "total_rx",
            "TRx",
            "trx_count",
            "nrx_count",
            "total_rx_count",
        ],
    )
    def test_prescription_shorthands_are_refused(self, metric):
        from src.repositories.experiment_outcome import ExperimentOutcomeRepository

        with pytest.raises(ValueError, match="TRIGGER funnel counts"):
            ExperimentOutcomeRepository.resolve_column(metric)

    def test_rates_are_unchanged(self):
        from src.repositories.experiment_outcome import ExperimentOutcomeRepository

        assert ExperimentOutcomeRepository.resolve_column("conversion") == (
            "conversion_rate",
            "mean",
        )


# ------------------------------------------- per-experiment unit outcome feed
# Option d1 (owner decision 2026-09-23, Part of #2207): load_arrays reads
# ab_experiment_unit_outcomes FIRST (one observed outcome per (experiment, unit,
# metric), time-indexed by observed_at) and falls back to the business_metrics
# per_hcp_rollup join byte-for-byte when the experiment has no unit outcomes.
class _Page:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


_POSTGREST_MAX_ROWS = 1000


class _FeedQuery:
    """PostgREST-shaped fake for ONE table: records every filter, serves
    ``.range()`` pages of its rows; an UN-ranged read is capped at PostgREST's
    max-rows (1,000 on Supabase) exactly like the server (codex r1 HIGH: an
    unbounded assignments read silently drops units 1,001+)."""

    def __init__(self, table, rows, log):
        self._table = table
        self._rows = rows
        self._log = log
        self._range = None
        self._order = None

    def _rec(self, op, *a):
        self._log.append((self._table, op, a))
        return self

    def select(self, *a, **k):
        return self._rec("select", *a)

    def eq(self, *a, **k):
        return self._rec("eq", *a)

    def in_(self, *a, **k):
        return self._rec("in_", *a)

    def order(self, *a, **k):
        self._order = a[0] if a else None
        return self._rec("order", *a)

    def range(self, start, end):
        self._range = (start, end)
        return self._rec("range", start, end)

    def execute(self):
        self._log.append((self._table, "execute", ()))
        if self._range is None:
            return _Page(list(self._rows[:_POSTGREST_MAX_ROWS]), len(self._rows))
        s, e = self._range
        # Offset paging is only stable over a total order (codex r2 LOW): a
        # ranged read without .order(<unique column>) is served in a SHUFFLED
        # order per call, like a heap scan may, so a reader that drops the
        # order duplicates/omits rows with the total count unchanged.
        if self._order in ("unit_id", "id"):
            rows = sorted(self._rows, key=lambda r: str(r.get(self._order)))
        else:
            import random

            rows = list(self._rows)
            random.Random(len(self._log)).shuffle(rows)
        return _Page(rows[s : e + 1], len(self._rows))


class _FeedClient:
    def __init__(self, tables):
        self._tables = tables
        self.log = []

    def table(self, name):
        return _FeedQuery(name, self._tables.get(name, []), self.log)

    def tables_queried(self):
        return [t for t, op, _ in self.log if op == "execute"]


def _uo(unit, value, observed):
    return {"unit_id": unit, "outcome_value": value, "observed_at": observed}


class TestLoadArraysUnitOutcomeFeed:
    _ASSIGN = [
        {"unit_id": "scvhcp_00001", "variant": "control"},
        {"unit_id": "scvhcp_00002", "variant": "control"},
        {"unit_id": "scvhcp_00003", "variant": "treatment"},
        {"unit_id": "scvhcp_00004", "variant": "treatment"},
    ]

    def _run(self, client, metric="pnh_persistence", **kw):
        import asyncio
        from uuid import uuid4

        from src.repositories.experiment_outcome import ExperimentOutcomeRepository

        repo = ExperimentOutcomeRepository(supabase_client=client)
        return asyncio.run(repo.load_arrays(uuid4(), metric, brand="Fabhalta", **kw))

    def test_unit_outcomes_present_feed_the_arrays_and_business_metrics_is_not_queried(
        self, monkeypatch
    ):
        """(vi) rows exist -> arrays come from them (mean reducer), the
        business_metrics join is NOT issued, the provenance predicate and the
        (experiment, metric) filters are applied to the unit-outcome query."""
        monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
        client = _FeedClient(
            {
                "ab_experiment_assignments": self._ASSIGN,
                "ab_experiment_unit_outcomes": [
                    _uo("scvhcp_00001", 0.0, "2026-09-01T00:00:00+00:00"),
                    _uo("scvhcp_00002", 1.0, "2026-09-02T00:00:00+00:00"),
                    _uo("scvhcp_00003", 1.0, "2026-09-03T00:00:00+00:00"),
                    _uo("scvhcp_00004", 1.0, "2026-09-04T00:00:00+00:00"),
                ],
                # a decoy: if the fallback ran it would produce a DIFFERENT answer
                "business_metrics": [
                    {"hcp_id": "scvhcp_00001", "pnh_persistence": 9.0, "metric_date": "2026-09-01"}
                ],
            }
        )
        control, treatment = self._run(client)
        assert sorted(control.tolist()) == [0.0, 1.0]
        assert treatment.tolist() == [1.0, 1.0]
        assert "business_metrics" not in client.tables_queried()
        uo_filters = [(op, a) for t, op, a in client.log if t == "ab_experiment_unit_outcomes"]
        assert ("eq", ("metric_name", "pnh_persistence")) in uo_filters
        assert ("eq", ("is_synthetic", False)) in uo_filters
        assert any(op == "eq" and a[0] == "experiment_id" for op, a in uo_filters)

    def test_unit_outcomes_absent_falls_back_to_the_business_metrics_join(self, monkeypatch):
        """(vii) zero unit-outcome rows -> today's path: resolve_column + the
        per_hcp_rollup join on business_metrics with the same filters as before."""
        monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
        client = _FeedClient(
            {
                "ab_experiment_assignments": self._ASSIGN,
                "ab_experiment_unit_outcomes": [],
                "business_metrics": [
                    {"hcp_id": "scvhcp_00001", "conversion_rate": 0.2, "metric_date": "2026-09-01"},
                    {"hcp_id": "scvhcp_00001", "conversion_rate": 0.4, "metric_date": "2026-09-02"},
                    {"hcp_id": "scvhcp_00003", "conversion_rate": 0.8, "metric_date": "2026-09-01"},
                ],
            }
        )
        control, treatment = self._run(client, metric="conversion_rate")
        assert control.tolist() == pytest.approx([0.3])
        assert treatment.tolist() == pytest.approx([0.8])
        assert "business_metrics" in client.tables_queried()
        bm = [(op, a) for t, op, a in client.log if t == "business_metrics"]
        assert ("eq", ("metric_type", "per_hcp_rollup")) in bm
        assert ("eq", ("is_synthetic", False)) in bm
        assert ("eq", ("brand", "Fabhalta")) in bm
        assert any(op == "in_" and a[0] == "hcp_id" for op, a in bm)

    def test_unknown_metric_without_unit_outcomes_still_fails_closed(self):
        """(viii) the fail-closed contract of the fallback is preserved."""
        client = _FeedClient(
            {"ab_experiment_assignments": self._ASSIGN, "ab_experiment_unit_outcomes": []}
        )
        with pytest.raises(ValueError, match="Unsupported primary_metric"):
            self._run(client, metric="pnh_persistence")
        assert "business_metrics" not in client.tables_queried()

    def test_unit_outcomes_are_paged_to_exhaustion(self):
        """(ix) 1,400 units over two 1,000-row pages -> 1,400 values, not 1,000 —
        on BOTH legs: the assignments read is paged too (codex r1 HIGH: an
        un-ranged assignments read is capped at 1,000 by PostgREST, and the
        1,000-unit variant map silently dropped the other 400 units' outcomes)."""
        n = 1400
        assign = [
            {"unit_id": f"scvhcp_{i:05d}", "variant": "control" if i % 2 else "treatment"}
            for i in range(n)
        ]
        rows = [_uo(f"scvhcp_{i:05d}", float(i % 2), "2026-09-01T00:00:00+00:00") for i in range(n)]
        client = _FeedClient(
            {"ab_experiment_assignments": assign, "ab_experiment_unit_outcomes": rows}
        )
        control, treatment = self._run(client)
        assert control.size + treatment.size == n
        assert control.size == 700 and treatment.size == 700
        for table in ("ab_experiment_unit_outcomes", "ab_experiment_assignments"):
            ranges = [a for t, op, a in client.log if t == table and op == "range"]
            assert ranges and ranges[0] == (0, 999), table
            assert len(ranges) >= 2, table
            orders = [a for t, op, a in client.log if t == table and op == "order"]
            assert orders and all(a[0] == "unit_id" for a in orders), table
        # the per-unit values survived paging intact (no duplicate / omitted unit)
        assert sorted(control.tolist()) == [1.0] * 700
        assert sorted(treatment.tolist()) == [0.0] * 700

    def test_short_assignments_read_fails_loud_instead_of_a_truncated_atE(self):
        """A server that reports 1,400 assignments but serves fewer must not feed
        a partial variant map to the pooled test."""
        n = 1400
        assign = [
            {"unit_id": f"scvhcp_{i:05d}", "variant": "control" if i % 2 else "treatment"}
            for i in range(n)
        ]
        rows = [_uo(f"scvhcp_{i:05d}", float(i % 2), "2026-09-01T00:00:00+00:00") for i in range(n)]
        client = _FeedClient(
            {"ab_experiment_assignments": assign, "ab_experiment_unit_outcomes": rows}
        )

        class _Lying(_FeedQuery):
            def execute(self):
                page = super().execute()
                if self._table == "ab_experiment_assignments":
                    page.count = n + 1  # server claims one more than it serves
                return page

        client.table = lambda name: _Lying(name, client._tables.get(name, []), client.log)
        with pytest.raises(RuntimeError, match="ab_experiment_assignments"):
            self._run(client)

    def test_window_days_is_a_per_unit_post_assignment_window_on_observed_at(self):
        """(x) window_days on the time-indexed feed is PER UNIT: keep an outcome
        iff assigned_at <= observed_at <= assigned_at + window_days, at timestamp
        precision (codex r1 MED: anchoring on the experiment's newest outcome
        would select recent ENROLLEES under rolling enrollment, not outcomes
        observed within each unit's window)."""

        def _asn(unit, variant, assigned):
            return {"unit_id": unit, "variant": variant, "assigned_at": assigned}

        client = _FeedClient(
            {
                "ab_experiment_assignments": [
                    _asn("scvhcp_00001", "control", "2026-09-01T00:00:00+00:00"),
                    _asn("scvhcp_00002", "control", "2026-09-10T00:00:00+00:00"),
                    _asn("scvhcp_00003", "treatment", "2026-09-15T00:00:00+00:00"),
                    _asn("scvhcp_00004", "treatment", "2026-09-15T12:00:00+00:00"),
                ],
                "ab_experiment_unit_outcomes": [
                    # 20 d after assignment -> outside a 7 d window (though recent)
                    _uo("scvhcp_00001", 1.0, "2026-09-21T00:00:00+00:00"),
                    # 2 d after assignment -> inside
                    _uo("scvhcp_00002", 0.0, "2026-09-12T00:00:00+00:00"),
                    # exactly 7 d after -> inside (inclusive bound)
                    _uo("scvhcp_00003", 1.0, "2026-09-22T00:00:00+00:00"),
                    # 7 d + 1 s after a NOON assignment -> outside at timestamp precision
                    _uo("scvhcp_00004", 0.0, "2026-09-22T12:00:01+00:00"),
                ],
            }
        )
        control, treatment = self._run(client, window_days=7)
        assert control.tolist() == [0.0]  # 00001 outside, 00002 inside
        assert treatment.tolist() == [1.0]  # 00003 inside, 00004 outside by 1 s
        # no window -> every unit
        control, treatment = self._run(client)
        assert control.size == 2 and treatment.size == 2

    def test_empty_metric_never_queries_unit_outcomes(self):
        """A blank primary_metric cannot name a unit-outcome metric: go straight to
        the fallback, which fails closed."""
        client = _FeedClient({"ab_experiment_assignments": self._ASSIGN})
        with pytest.raises(ValueError):
            self._run(client, metric="")
        assert "ab_experiment_unit_outcomes" not in client.tables_queried()
