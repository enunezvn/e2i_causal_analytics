"""Batch-2 KPI history backfill: as-of monthly recompute handlers.

Covers the brief's locked behaviors:

- month grouping edge cases (partial leading/trailing months are DROPPED)
- WS2-TR-004 denominator lock: delivered/viewed ONLY (migration 092 — never
  regress to all-non-null acceptance_status)
- replace semantics: (kpi_id, source) rows are deleted BEFORE the fresh upsert
- lower-is-better status wiring (TR-005/006/007/008, BR-001/004)
- registry-coherence smoke (YAML-only; the live-DB comparison stays a manual
  post-deploy probe because kpi_query_registry rows live in the database)

All handler tests run against an in-memory fake Supabase client — no DB.
"""

import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.kpi import history_backfill as hb
from src.kpi.models import KPIThreshold

# ---------------------------------------------------------------------------
# Fake async Supabase client (applies eq filters + range pagination)
# ---------------------------------------------------------------------------


class _FakeTableQuery:
    def __init__(self, rows):
        self._rows = rows
        self._filters = []
        self._range = None

    def select(self, *a, **k):
        return self

    def eq(self, column, value):
        self._filters.append((column, value))
        return self

    def order(self, *a, **k):
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def limit(self, *a, **k):
        return self

    @property
    def not_(self):
        return self

    def is_(self, *a, **k):
        return self

    async def execute(self):
        rows = [r for r in self._rows if all(r.get(c) == v for c, v in self._filters)]
        if self._range is not None:
            start, end = self._range
            rows = rows[start : end + 1]
        return SimpleNamespace(data=rows)


class _FakeClient:
    def __init__(self, tables):
        self._tables = tables
        self.table_calls = 0

    def table(self, name):
        self.table_calls += 1
        return _FakeTableQuery(self._tables.get(name, []))


def _meta(kpi_id, threshold=None):
    return SimpleNamespace(id=kpi_id, threshold=threshold)


def _run(coro):
    return asyncio.run(coro)


def _rx(patient, brand, day, seq=2):
    return {
        "patient_id": patient,
        "brand": brand,
        "event_type": "prescription",
        "event_date": day,
        "sequence_number": seq,
    }


# ---------------------------------------------------------------------------
# Month grouping edge cases
# ---------------------------------------------------------------------------


class TestCompleteMonths:
    def test_partial_edge_months_dropped(self):
        months = hb._complete_months([date(2026, 1, 15), date(2026, 3, 10)])
        assert months == [date(2026, 2, 1)]

    def test_first_month_kept_when_data_starts_on_the_1st(self):
        months = hb._complete_months([date(2026, 1, 1), date(2026, 3, 10)])
        assert months == [date(2026, 1, 1), date(2026, 2, 1)]

    def test_last_month_kept_when_data_reaches_month_end(self):
        months = hb._complete_months([date(2026, 1, 5), date(2026, 2, 28)])
        assert months == [date(2026, 2, 1)]

    def test_single_partial_month_yields_nothing(self):
        assert hb._complete_months([date(2026, 1, 5), date(2026, 1, 20)]) == []
        assert hb._complete_months([]) == []


# ---------------------------------------------------------------------------
# Pagination + per-run cache
# ---------------------------------------------------------------------------


class TestFetchAll:
    def test_paginates_and_caches(self):
        rows = [{"trigger_id": i} for i in range(5)]
        client = _FakeClient({"triggers": rows})
        cache = {}
        with patch.object(hb, "_PAGE_SIZE", 2):
            got = _run(
                hb._fetch_all(
                    client, "triggers", "trigger_id", "trigger_id", cache=cache, cache_key="t"
                )
            )
            assert got == rows
            assert client.table_calls == 3  # ceil(5/2) pages
            again = _run(
                hb._fetch_all(
                    client, "triggers", "trigger_id", "trigger_id", cache=cache, cache_key="t"
                )
            )
        assert again is got
        assert client.table_calls == 3  # served from cache, no new queries


# ---------------------------------------------------------------------------
# Prescription-volume family
# ---------------------------------------------------------------------------


class TestRxFamily:
    def _client(self):
        rows = [
            # Dec 2025 is partial (starts on the 15th) -> dropped.
            _rx("p1", "Kisqali", "2025-12-15", seq=1),
            # January 2026 (complete).
            _rx("p1", "Kisqali", "2026-01-05", seq=2),
            _rx("p2", "Kisqali", "2026-01-10", seq=1),
            _rx("p3", "Fabhalta", "2026-01-20", seq=1),
            # February 2026 (complete).
            _rx("p4", "Fabhalta", "2026-02-10", seq=1),
            # March 2026 is partial (frontier mid-month) -> dropped.
            _rx("p5", "Kisqali", "2026-03-02", seq=1),
        ]
        return _FakeClient({"treatment_events": rows})

    def test_trx_counts_global_and_brand_with_genuine_zero(self):
        points = _run(hb._backfill_trx(self._client(), _meta("WS3-BI-005")))
        glob = {p["metric_date"]: p["value"] for p in points if p["brand"] == ""}
        assert glob == {"2026-01-01": 3.0, "2026-02-01": 1.0}
        kis = {p["metric_date"]: p["value"] for p in points if p["brand"] == "Kisqali"}
        # Kisqali wrote nothing in Feb: genuine zero inside the covered span.
        assert kis == {"2026-01-01": 2.0, "2026-02-01": 0.0}
        # No threshold (volume metric) -> INFORMATIONAL, never UNKNOWN.
        assert {p["status"] for p in points} == {"informational"}
        assert all(p["is_synthetic"] and p["region"] == "" for p in points)
        assert all(p["source"] == "treatment_events.event_date" for p in points)

    def test_nrx_counts_only_sequence_number_1(self):
        points = _run(hb._backfill_nrx(self._client(), _meta("WS3-BI-006")))
        glob = {p["metric_date"]: p["value"] for p in points if p["brand"] == ""}
        # Jan: p2 + p3 are seq 1; p1's Jan script is seq 2.
        assert glob == {"2026-01-01": 2.0, "2026-02-01": 1.0}

    def test_nbrx_first_brand_prescription_month_and_no_global_series(self):
        points = _run(hb._backfill_nbrx(self._client(), _meta("WS3-BI-007")))
        # NBRx is undefined without a brand (live calculator fails loud) ->
        # no fabricated global rows.
        assert all(p["brand"] != "" for p in points)
        kis = {p["metric_date"]: p["value"] for p in points if p["brand"] == "Kisqali"}
        # p1's first Kisqali script was Dec 15 (outside the complete months);
        # p2's first is Jan. p5's March first-Rx falls in a dropped month.
        assert kis == {"2026-01-01": 1.0, "2026-02-01": 0.0}
        fab = {p["metric_date"]: p["value"] for p in points if p["brand"] == "Fabhalta"}
        assert fab == {"2026-01-01": 1.0, "2026-02-01": 1.0}

    def test_trx_share_is_brand_over_category(self):
        points = _run(hb._backfill_trx_share(self._client(), _meta("WS3-BI-008")))
        assert all(p["brand"] != "" for p in points)
        kis = {p["metric_date"]: p["value"] for p in points if p["brand"] == "Kisqali"}
        assert kis["2026-01-01"] == pytest.approx(2 / 3)
        assert kis["2026-02-01"] == pytest.approx(0.0)


class TestConversionRate:
    def test_30_day_window_and_right_censoring(self):
        def trig(tid, patient, ts):
            return {"trigger_id": tid, "patient_id": patient, "trigger_timestamp": ts}

        triggers = [
            # January 2026 completes (trigger data spans 1/1 .. 2/28).
            trig("t1", "p1", "2026-01-01T08:00:00+00:00"),  # rx exactly +30d -> converted
            trig("t2", "p2", "2026-01-05T08:00:00+00:00"),  # rx +31d -> NOT converted
            trig("t3", "p3", "2026-01-10T08:00:00+00:00"),  # rx BEFORE trigger -> NOT converted
            trig("t4", "p4", "2026-01-20T08:00:00+00:00"),  # no rx at all (censored/unconverted)
            trig("t5", "p5", "2026-02-28T08:00:00+00:00"),
        ]
        rx = [
            _rx("p1", "Kisqali", "2026-01-31"),
            _rx("p2", "Kisqali", "2026-02-05"),
            _rx("p3", "Kisqali", "2026-01-02"),
        ]
        client = _FakeClient({"triggers": triggers, "treatment_events": rx})
        points = _run(hb._backfill_conversion_rate(client, _meta("WS3-BI-009")))
        by_month = {p["metric_date"]: p["value"] for p in points}
        assert by_month["2026-01-01"] == pytest.approx(1 / 4)
        assert all(p["brand"] == "" for p in points)


# ---------------------------------------------------------------------------
# Trigger performance family
# ---------------------------------------------------------------------------


def _trigger_row(ts, **overrides):
    row = {
        "trigger_id": overrides.pop("trigger_id", ts),
        "patient_id": "p",
        "trigger_timestamp": ts,
        "delivery_status": "delivered",
        "acceptance_status": None,
        "false_positive_flag": False,
        "lead_time_days": None,
        "outcome_tracked": False,
        "outcome_value": None,
        "previous_trigger_id": None,
        "change_failed": None,
    }
    row.update(overrides)
    return row


def _pad_month(rows):
    """Ensure January 2026 is a complete month (data on the 1st and 31st)."""
    return (
        [_trigger_row("2026-01-01T00:10:00+00:00", trigger_id="pad-lo")]
        + rows
        + [_trigger_row("2026-01-31T23:00:00+00:00", trigger_id="pad-hi")]
    )


class TestTriggerRatios:
    def test_tr004_denominator_is_delivered_or_viewed_only(self):
        """The migration-092 lock: pending/failed triggers NEVER enter the
        denominator; the numerator counts accepted rows unrestricted, exactly
        like the registry SQL."""
        rows = _pad_month(
            [
                _trigger_row(
                    "2026-01-05T08:00:00+00:00",
                    trigger_id="a",
                    delivery_status="delivered",
                    acceptance_status="accepted",
                ),
                _trigger_row(
                    "2026-01-06T08:00:00+00:00",
                    trigger_id="b",
                    delivery_status="viewed",
                    acceptance_status="rejected",
                ),
                _trigger_row(
                    "2026-01-07T08:00:00+00:00",
                    trigger_id="c",
                    delivery_status="pending",
                    acceptance_status="accepted",
                ),
                _trigger_row(
                    "2026-01-08T08:00:00+00:00",
                    trigger_id="d",
                    delivery_status="failed",
                    acceptance_status=None,
                ),
            ]
        )
        client = _FakeClient({"triggers": rows})
        points = _run(hb._backfill_tr004_acceptance(client, _meta("WS2-TR-004")))
        assert len(points) == 1
        # 2 accepted (a + pending c) / 4 delivered-or-viewed (a, b, pads).
        assert points[0]["value"] == pytest.approx(2 / 4)
        # The all-non-null regression would divide by 3 (a, b, c) -> 2/3.
        assert points[0]["value"] != pytest.approx(2 / 3)

    def test_tr005_false_alert_rate_uses_lower_is_better_status(self):
        rows = _pad_month(
            [_trigger_row("2026-01-05T08:00:00+00:00", trigger_id="fp", false_positive_flag=True)]
        )
        threshold = KPIThreshold(target=0.10, warning=0.20, critical=0.30)
        client = _FakeClient({"triggers": rows})
        points = _run(hb._backfill_tr005_false_alert(client, _meta("WS2-TR-005", threshold)))
        assert points[0]["value"] == pytest.approx(1 / 3)
        # 0.333 > warning 0.20 must read CRITICAL (lower-is-better). The
        # higher-is-better misread would call 0.333 >= target 0.10 "good".
        assert points[0]["status"] == "critical"

    def test_tr006_override_rate_delivered_denominator(self):
        rows = _pad_month(
            [
                _trigger_row(
                    "2026-01-05T08:00:00+00:00",
                    trigger_id="o",
                    delivery_status="viewed",
                    acceptance_status="overridden",
                ),
                _trigger_row(
                    "2026-01-06T08:00:00+00:00", trigger_id="x", delivery_status="pending"
                ),
            ]
        )
        client = _FakeClient({"triggers": rows})
        points = _run(hb._backfill_tr006_override(client, _meta("WS2-TR-006")))
        assert points[0]["value"] == pytest.approx(1 / 3)  # pads + o; pending excluded

    def test_tr007_median_lead_time_and_good_status(self):
        rows = [
            _trigger_row("2026-01-01T00:10:00+00:00", trigger_id="w", lead_time_days=8),
            _trigger_row("2026-01-10T08:00:00+00:00", trigger_id="x", lead_time_days=10),
            _trigger_row("2026-01-20T08:00:00+00:00", trigger_id="y", lead_time_days=20),
            _trigger_row("2026-01-31T23:00:00+00:00", trigger_id="z", lead_time_days=None),
        ]
        threshold = KPIThreshold(target=14, warning=21, critical=30)
        client = _FakeClient({"triggers": rows})
        points = _run(hb._backfill_tr007_lead_time(client, _meta("WS2-TR-007", threshold)))
        assert points[0]["value"] == pytest.approx(10.0)  # median of [8, 10, 20]
        assert points[0]["status"] == "good"  # 10 <= target 14, lower-is-better

    def test_tr001_precision_and_tr008_cfr(self):
        rows = _pad_month(
            [
                _trigger_row(
                    "2026-01-05T08:00:00+00:00",
                    trigger_id="hit",
                    outcome_tracked=True,
                    outcome_value=2.0,
                ),
                _trigger_row(
                    "2026-01-06T08:00:00+00:00",
                    trigger_id="miss",
                    outcome_tracked=True,
                    outcome_value=0.0,
                ),
                _trigger_row(
                    "2026-01-07T08:00:00+00:00",
                    trigger_id="chg-ok",
                    previous_trigger_id="prev1",
                    change_failed=False,
                ),
                _trigger_row(
                    "2026-01-08T08:00:00+00:00",
                    trigger_id="chg-bad",
                    previous_trigger_id="prev2",
                    change_failed=True,
                ),
            ]
        )
        client = _FakeClient({"triggers": rows})
        precision = _run(hb._backfill_tr001_precision(client, _meta("WS2-TR-001")))
        assert precision[0]["value"] == pytest.approx(1 / 2)
        cfr = _run(hb._backfill_tr008_cfr(client, _meta("WS2-TR-008")))
        assert cfr[0]["value"] == pytest.approx(1 / 2)


# ---------------------------------------------------------------------------
# Active users
# ---------------------------------------------------------------------------


class TestActiveUsers:
    def _sessions(self):
        rows = []
        # Full June 2026 coverage: user u1 every Monday, u2 twice in week 1,
        # u3 only in week 2. Data touches 6/1 and 6/30 so June is complete.
        for day, user in [
            (1, "u1"),
            (2, "u2"),
            (3, "u2"),
            (8, "u1"),
            (9, "u3"),
            (15, "u1"),
            (22, "u1"),
            (30, "u1"),
        ]:
            rows.append(
                {
                    "session_id": f"s{day}-{user}",
                    "user_id": user,
                    "session_start": f"2026-06-{day:02d}T09:00:00+00:00",
                }
            )
        return rows

    def test_mau_counts_distinct_users_per_complete_month(self):
        client = _FakeClient({"user_sessions": self._sessions()})
        points = _run(hb._backfill_mau(client, _meta("WS3-BI-001")))
        assert len(points) == 1
        assert points[0]["metric_date"] == "2026-06-01"
        assert points[0]["value"] == 3.0  # u1, u2, u3 — deduped

    def test_wau_is_mean_of_weeks_fully_inside_the_month(self):
        client = _FakeClient({"user_sessions": self._sessions()})
        points = _run(hb._backfill_wau(client, _meta("WS3-BI-002")))
        assert len(points) == 1
        # Weeks starting 6/1 (u1,u2), 6/8 (u1,u3), 6/15 (u1), 6/22 (u1) are
        # fully inside June; the 6/29 week spans July and is excluded.
        assert points[0]["value"] == pytest.approx((2 + 2 + 1 + 1) / 4)


# ---------------------------------------------------------------------------
# Brand-specific
# ---------------------------------------------------------------------------


class TestBrandSpecific:
    def test_br001_monthly_uas7_cohort(self):
        def ev(patient, day, assay, value):
            return {
                "patient_id": patient,
                "brand": "Remibrutinib",
                "event_subtype": "baseline_antihistamine",
                "drug_class": "R06A",
                "event_date": day,
                "lab_values": {"assay": assay, "value": value},
            }

        rows = [
            ev("p1", "2026-01-01", "UAS7", 9),  # uncontrolled
            ev("p2", "2026-01-10", "UAS7", 5),  # controlled
            ev("p3", "2026-01-15", "UAS7", 7),  # boundary: >= 7 is uncontrolled
            ev("p4", "2026-01-20", "OTHER", 40),  # non-UAS7 assay ignored
            ev("p5", "2026-01-31", "UAS7", 2),  # controlled (also completes Jan)
        ]
        client = _FakeClient({"treatment_events": rows})
        points = _run(hb._backfill_br001_ah_uncontrolled(client, _meta("BR-001")))
        assert len(points) == 1
        assert points[0]["value"] == pytest.approx(2 / 4)  # p1 + p3 of p1,p2,p3,p5
        assert points[0]["brand"] == ""  # single-brand KPI lives in the global scope

    def test_br003_asof_cumulative_rate(self):
        journeys = [
            {
                "patient_id": "p1",
                "brand": "Fabhalta",
                "primary_diagnosis_code": "D59.5",
                "journey_start_date": "2026-01-10",
            },
            {
                "patient_id": "p2",
                "brand": "Fabhalta",
                "primary_diagnosis_code": "D59.5",
                "journey_start_date": "2026-02-10",
            },
            {
                "patient_id": "p3",
                "brand": "Fabhalta",
                "primary_diagnosis_code": "D59.5",
                "journey_start_date": "2026-03-20",
            },
            # Wrong brand / dx never enter the denominator.
            {
                "patient_id": "px",
                "brand": "Kisqali",
                "primary_diagnosis_code": "D59.5",
                "journey_start_date": "2026-01-02",
            },
            {
                "patient_id": "py",
                "brand": "Fabhalta",
                "primary_diagnosis_code": "C50.9",
                "journey_start_date": "2026-01-02",
            },
        ]
        pnh = [
            {
                "patient_id": "p1",
                "event_subtype": "pnh_flow_cytometry",
                "event_date": "2026-01-15",
                "loinc_codes": ["55164-8"],
            },
            {
                "patient_id": "p2",
                "event_subtype": "pnh_flow_cytometry",
                "event_date": "2026-03-05",
                "loinc_codes": ["35468-8"],
            },
            # Non-PNH LOINC never enters the numerator.
            {
                "patient_id": "p1",
                "event_subtype": "pnh_flow_cytometry",
                "event_date": "2026-01-02",
                "loinc_codes": ["9999-9"],
            },
        ]
        client = _FakeClient({"patient_journeys": journeys, "treatment_events": pnh})
        points = _run(hb._backfill_br003_pnh_tested(client, _meta("BR-003")))
        by_month = {p["metric_date"]: p["value"] for p in points}
        # Frontier = min(max journey start 3/20, max PNH test 3/5) = 3/5, so
        # month-ends Jan 31 and Feb 28 qualify; Mar 31 does not.
        assert by_month == {
            "2026-01-01": pytest.approx(1 / 1),  # p1 eligible + tested
            "2026-02-01": pytest.approx(1 / 2),  # p2 eligible, tested only in March
        }

    def test_br004_median_days_preserves_journey_join_multiplicity(self):
        rx = [
            _rx("pad", "Fabhalta", "2026-01-01"),  # completes January
            _rx("p1", "Kisqali", "2026-01-21"),
            _rx("p1", "Kisqali", "2026-01-25"),  # later scripts don't move first-Rx
            _rx("p2", "Kisqali", "2026-01-26"),
            _rx("pad2", "Fabhalta", "2026-01-31"),
        ]
        journeys = [
            {
                "patient_id": "p1",
                "brand": "Kisqali",
                "primary_diagnosis_code": "C50.9",
                "journey_start_date": "2026-01-01",
            },
            {
                "patient_id": "p1",
                "brand": "Kisqali",
                "primary_diagnosis_code": "C50.9",
                "journey_start_date": "2026-01-11",
            },
            {
                "patient_id": "p2",
                "brand": "Kisqali",
                "primary_diagnosis_code": "C50.9",
                "journey_start_date": "2026-01-06",
            },
            # Journey starting AFTER the first Rx contributes nothing.
            {
                "patient_id": "p2",
                "brand": "Kisqali",
                "primary_diagnosis_code": "C50.9",
                "journey_start_date": "2026-01-30",
            },
        ]
        client = _FakeClient({"treatment_events": rx, "patient_journeys": journeys})
        points = _run(hb._backfill_br004_dx_adoption(client, _meta("BR-004")))
        assert len(points) == 1
        # Deltas: p1 -> 20 and 10 (both journeys), p2 -> 20. Median = 20.
        assert points[0]["value"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Replace semantics (delete before upsert) + repository delete
# ---------------------------------------------------------------------------


class TestReplaceSemantics:
    def test_run_backfill_deletes_source_rows_before_upserting(self):
        calls = []

        def _record_delete(kpi_id, source):
            calls.append(("delete", kpi_id, source))
            return 7

        def _record_upsert(points):
            calls.append(("upsert", len(points)))
            return len(points)

        repo = SimpleNamespace(
            delete_source=AsyncMock(side_effect=_record_delete),
            upsert_points=AsyncMock(side_effect=_record_upsert),
        )
        client = _FakeClient(
            {
                "treatment_events": [
                    _rx("p1", "Kisqali", "2026-01-01"),
                    _rx("p2", "Kisqali", "2026-02-28"),
                ]
            }
        )
        registry = SimpleNamespace(get=lambda kpi_id: _meta(kpi_id))
        with (
            patch(
                "src.memory.services.factories.get_async_supabase_client",
                new=AsyncMock(return_value=client),
            ),
            patch(
                "src.repositories.kpi_history.get_kpi_history_repository",
                new=AsyncMock(return_value=repo),
            ),
            patch("src.kpi.registry.KPIRegistry", new=lambda: registry),
        ):
            summary = _run(hb.run_backfill(["WS3-BI-005"]))

        assert calls[0] == ("delete", "WS3-BI-005", "treatment_events.event_date")
        assert calls[1][0] == "upsert"
        assert summary["deleted"]["WS3-BI-005"] == 7
        assert summary["written"]["WS3-BI-005"] == calls[1][1] > 0
        assert summary["errors"] == {}

    def test_delete_source_filters_on_kpi_and_source(self):
        from src.repositories.kpi_history import KPIHistoryRepository

        captured = {}

        class _DeleteQuery:
            def delete(self):
                captured["deleted"] = True
                return self

            def eq(self, column, value):
                captured[column] = value
                return self

            async def execute(self):
                return SimpleNamespace(data=[{"id": 1}, {"id": 2}])

        class _Client:
            def table(self, name):
                captured["table"] = name
                return _DeleteQuery()

        repo = KPIHistoryRepository(supabase_client=_Client())
        deleted = _run(repo.delete_source("WS2-TR-004", "triggers.trigger_timestamp"))
        assert deleted == 2
        assert captured == {
            "table": "kpi_history",
            "deleted": True,
            "kpi_id": "WS2-TR-004",
            "source": "triggers.trigger_timestamp",
        }


# ---------------------------------------------------------------------------
# Registry coherence smoke (YAML-only — no live DB in unit tests; the
# registry-SQL comparison stays a manual post-deploy probe)
# ---------------------------------------------------------------------------


class TestRegistryCoherence:
    def test_every_handler_kpi_exists_with_expected_threshold_shape(self):
        from src.kpi.registry import KPIRegistry

        registry = KPIRegistry()
        assert set(hb.HANDLER_SOURCES) == set(hb.HANDLERS)
        for kpi_id in hb.HANDLERS:
            meta = registry.get(kpi_id)
            assert meta is not None, f"{kpi_id} missing from config/kpi_definitions.yaml"
            if meta.threshold is None:
                # Volume metrics tracked without a target BY DESIGN.
                assert kpi_id in {"WS3-BI-005", "WS3-BI-006", "WS3-BI-007"}
                continue
            target, warning = meta.threshold.target, meta.threshold.warning
            if target is None or warning is None:
                continue
            if kpi_id in hb.LOWER_IS_BETTER:
                # lower-is-better thresholds ascend (target < warning).
                assert target < warning, f"{kpi_id} direction mismatch vs YAML"
            else:
                assert target > warning, f"{kpi_id} direction mismatch vs YAML"

    def test_lower_is_better_mirrors_live_calculators(self):
        # Locked to trigger_performance.py / brand_specific.py inline sets.
        assert hb.LOWER_IS_BETTER == {
            "WS2-TR-005",
            "WS2-TR-006",
            "WS2-TR-007",
            "WS2-TR-008",
            "BR-001",
            "BR-004",
        }
