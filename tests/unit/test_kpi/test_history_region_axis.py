"""#1536: kpi_history region axis — backfill handlers emit region-scoped series.

Locked behaviors (each mirrors a vetted live registry variant — the backfill
must never invent a region reading the live platform cannot produce):

- WS3-BI-010 ROI: ``business_metrics.region`` DIRECT (migration 125 idiom) —
  region-only + brand×region monthly means; global/per-brand series unchanged.
- WS3-BI-005/006 TRx/NRx: region via the event's OWN ``patient_journey_id`` →
  ``patient_journeys.geographic_region`` (migration 077 join). Events with a
  NULL journey link or a region-less journey are dropped from region series
  ONLY (they still count globally) — exactly what the live
  ``patient_journey_id IN (...)`` predicate does.
- WS3-BI-007 NBRx / WS3-BI-008 TRx Share: brand×region ONLY (the live
  calculators fail loud without a brand; 077's share category = the REGION's
  prescriptions).
- WS3-BI-009 Conversion + WS2-TR-*: region via ``patient_id`` MEMBERSHIP
  (077/078: ``patient_id IN region_patients``) — a patient with journeys in
  two regions counts in BOTH; empty-denominator months are skipped.
- WS2-TR-001 maturation cutoff stays anchored to the GLOBAL trigger frontier
  (migration 113's unscoped ``MAX(trigger_timestamp)``), never a per-region
  frontier.
- ``REGION_AXIS_KPI_IDS`` lockstep: every region-axis KPI maps to a live
  region-capable registry variant defined in migrations 077/078/113/125.

All handler tests run against an in-memory fake Supabase client — no DB.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.kpi import history_backfill as hb

# ---------------------------------------------------------------------------
# Fake async Supabase client (applies eq filters + range pagination) — same
# idiom as test_history_backfill_batch2.
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

    def table(self, name):
        return _FakeTableQuery(self._tables.get(name, []))


def _meta(kpi_id="WS3-BI-005"):
    return SimpleNamespace(id=kpi_id, threshold=None)


def _by_scope(points):
    """Index points as {(brand, region): {metric_date: value}}."""
    out = {}
    for p in points:
        out.setdefault((p["brand"], p["region"]), {})[p["metric_date"]] = p["value"]
    return out


# Journeys: P2 spans TWO regions (south + west); J5 has no region.
_JOURNEYS = [
    {
        "patient_journey_id": "J1",
        "patient_id": "P1",
        "geographic_region": "northeast",
        "journey_start_date": "2025-01-01",
        "brand": "Kisqali",
        "primary_diagnosis_code": "C50",
    },
    {
        "patient_journey_id": "J2",
        "patient_id": "P2",
        "geographic_region": "south",
        "journey_start_date": "2025-01-01",
        "brand": "Kisqali",
        "primary_diagnosis_code": "C50",
    },
    {
        "patient_journey_id": "J3",
        "patient_id": "P3",
        "geographic_region": "northeast",
        "journey_start_date": "2025-01-01",
        "brand": "Kisqali",
        "primary_diagnosis_code": "C50",
    },
    {
        "patient_journey_id": "J4",
        "patient_id": "P2",
        "geographic_region": "west",
        "journey_start_date": "2025-01-01",
        "brand": "Fabhalta",
        "primary_diagnosis_code": "D59.5",
    },
    {
        "patient_journey_id": "J5",
        "patient_id": "P4",
        "geographic_region": None,
        "journey_start_date": "2025-01-01",
        "brand": "Fabhalta",
        "primary_diagnosis_code": "D59.5",
    },
]

# Prescriptions spanning exactly the complete months Mar+Apr 2025.
# e5 has NO journey link; e6's journey (J5) has NO region.
_PRESCRIPTIONS = [
    {
        "patient_id": "P1",
        "brand": "Kisqali",
        "event_date": "2025-03-01",
        "sequence_number": 1,
        "patient_journey_id": "J1",
        "event_type": "prescription",
    },
    {
        "patient_id": "P2",
        "brand": "Kisqali",
        "event_date": "2025-03-10",
        "sequence_number": 1,
        "patient_journey_id": "J2",
        "event_type": "prescription",
    },
    {
        "patient_id": "P2",
        "brand": "Fabhalta",
        "event_date": "2025-03-15",
        "sequence_number": 2,
        "patient_journey_id": "J4",
        "event_type": "prescription",
    },
    {
        "patient_id": "P3",
        "brand": "Kisqali",
        "event_date": "2025-04-05",
        "sequence_number": 1,
        "patient_journey_id": "J3",
        "event_type": "prescription",
    },
    {
        "patient_id": "P1",
        "brand": "Kisqali",
        "event_date": "2025-04-30",
        "sequence_number": 2,
        "patient_journey_id": None,
        "event_type": "prescription",
    },
    {
        "patient_id": "P4",
        "brand": "Fabhalta",
        "event_date": "2025-04-10",
        "sequence_number": 1,
        "patient_journey_id": "J5",
        "event_type": "prescription",
    },
]


def _rx_client():
    return _FakeClient({"treatment_events": _PRESCRIPTIONS, "patient_journeys": _JOURNEYS})


# ---------------------------------------------------------------------------
# WS3-BI-010 ROI — direct business_metrics.region
# ---------------------------------------------------------------------------


class TestRoiRegionAxis:
    _ROWS = [
        {"metric_date": "2025-03-01", "brand": "Kisqali", "roi": 2.0, "region": "northeast"},
        {"metric_date": "2025-03-01", "brand": "Kisqali", "roi": 4.0, "region": "south"},
        {"metric_date": "2025-03-01", "brand": "Fabhalta", "roi": 3.0, "region": "northeast"},
        {"metric_date": "2025-03-01", "brand": None, "roi": 5.0, "region": "west"},
        {"metric_date": "2025-03-01", "brand": "Remibrutinib", "roi": 6.0, "region": None},
    ]

    def _points(self):
        client = _FakeClient({"business_metrics": self._ROWS})
        return asyncio.run(hb._backfill_roi(client, _meta("WS3-BI-010")))

    def test_global_and_brand_series_unchanged(self):
        scopes = _by_scope(self._points())
        assert scopes[("", "")]["2025-03-01"] == (2.0 + 4.0 + 3.0 + 5.0 + 6.0) / 5
        assert scopes[("Kisqali", "")]["2025-03-01"] == 3.0
        assert scopes[("Fabhalta", "")]["2025-03-01"] == 3.0
        assert scopes[("Remibrutinib", "")]["2025-03-01"] == 6.0

    def test_region_series_mean_over_region_rows(self):
        scopes = _by_scope(self._points())
        assert scopes[("", "northeast")]["2025-03-01"] == 2.5  # mean(2.0, 3.0)
        assert scopes[("", "south")]["2025-03-01"] == 4.0
        assert scopes[("", "west")]["2025-03-01"] == 5.0

    def test_brand_region_series(self):
        scopes = _by_scope(self._points())
        assert scopes[("Kisqali", "northeast")]["2025-03-01"] == 2.0
        assert scopes[("Kisqali", "south")]["2025-03-01"] == 4.0
        assert scopes[("Fabhalta", "northeast")]["2025-03-01"] == 3.0
        # brand-less west row and region-less Remibrutinib row produce no combo
        assert ("Remibrutinib", "west") not in scopes

    def test_regionless_rows_stay_out_of_region_series(self):
        scopes = _by_scope(self._points())
        # Remibrutinib's row has region None: global + brand only.
        region_scopes_with_remi = [s for s in scopes if s[0] == "Remibrutinib" and s[1] != ""]
        assert region_scopes_with_remi == []


# ---------------------------------------------------------------------------
# WS3-BI-005/006 TRx / NRx — journey-link attribution
# ---------------------------------------------------------------------------


class TestTrxRegionAxis:
    def _points(self):
        return asyncio.run(hb._backfill_trx(_rx_client(), _meta("WS3-BI-005")))

    def test_global_series_unchanged_by_region_rows(self):
        scopes = _by_scope(self._points())
        assert scopes[("", "")] == {"2025-03-01": 3.0, "2025-04-01": 3.0}
        assert scopes[("Kisqali", "")] == {"2025-03-01": 2.0, "2025-04-01": 2.0}
        assert scopes[("Fabhalta", "")] == {"2025-03-01": 1.0, "2025-04-01": 1.0}

    def test_region_counts_follow_journey_link(self):
        scopes = _by_scope(self._points())
        assert scopes[("", "northeast")] == {"2025-03-01": 1.0, "2025-04-01": 1.0}
        assert scopes[("", "south")] == {"2025-03-01": 1.0, "2025-04-01": 0.0}
        assert scopes[("", "west")] == {"2025-03-01": 1.0, "2025-04-01": 0.0}

    def test_unlinked_and_regionless_events_drop_from_region_series_only(self):
        # April global = 3 (e4 + unlinked e5 + region-less e6) but the April
        # region series sum = 1 (only e4 carries an honest region link).
        scopes = _by_scope(self._points())
        april_region_sum = sum(
            series.get("2025-04-01", 0.0)
            for (brand, region), series in scopes.items()
            if brand == "" and region != ""
        )
        assert april_region_sum == 1.0
        assert scopes[("", "")]["2025-04-01"] == 3.0

    def test_brand_region_product_zero_filled(self):
        scopes = _by_scope(self._points())
        assert scopes[("Kisqali", "northeast")] == {"2025-03-01": 1.0, "2025-04-01": 1.0}
        assert scopes[("Kisqali", "south")] == {"2025-03-01": 1.0, "2025-04-01": 0.0}
        assert scopes[("Fabhalta", "west")] == {"2025-03-01": 1.0, "2025-04-01": 0.0}
        # Genuine zeros inside the complete span for never-seen combos.
        assert scopes[("Fabhalta", "northeast")] == {"2025-03-01": 0.0, "2025-04-01": 0.0}


class TestNrxRegionAxis:
    def test_region_counts_first_fills_only(self):
        points = asyncio.run(hb._backfill_nrx(_rx_client(), _meta("WS3-BI-006")))
        scopes = _by_scope(points)
        assert scopes[("", "")] == {"2025-03-01": 2.0, "2025-04-01": 2.0}
        assert scopes[("", "northeast")] == {"2025-03-01": 1.0, "2025-04-01": 1.0}
        assert scopes[("", "south")] == {"2025-03-01": 1.0, "2025-04-01": 0.0}
        # e3 (west) is sequence 2 — west has no NRx.
        assert scopes[("", "west")] == {"2025-03-01": 0.0, "2025-04-01": 0.0}


# ---------------------------------------------------------------------------
# WS3-BI-007 NBRx / WS3-BI-008 TRx Share — brand×region only
# ---------------------------------------------------------------------------


class TestNbrxRegionAxis:
    def _scopes(self):
        points = asyncio.run(hb._backfill_nbrx(_rx_client(), _meta("WS3-BI-007")))
        return _by_scope(points)

    def test_no_region_only_rows(self):
        # "new-to-brand" is undefined without a brand — mirror the live
        # fail-loud: no (brand='', region!='') rows may exist.
        assert [s for s in self._scopes() if s[0] == "" and s[1] != ""] == []

    def test_brand_region_first_rx_in_region(self):
        scopes = self._scopes()
        # P1 first Kisqali-in-northeast 2025-03-01; P3 2025-04-05.
        assert scopes[("Kisqali", "northeast")] == {"2025-03-01": 1.0, "2025-04-01": 1.0}
        assert scopes[("Kisqali", "south")] == {"2025-03-01": 1.0, "2025-04-01": 0.0}
        assert scopes[("Fabhalta", "west")] == {"2025-03-01": 1.0, "2025-04-01": 0.0}


class TestTrxShareRegionAxis:
    def _scopes(self):
        points = asyncio.run(hb._backfill_trx_share(_rx_client(), _meta("WS3-BI-008")))
        return _by_scope(points)

    def test_no_region_only_rows(self):
        assert [s for s in self._scopes() if s[0] == "" and s[1] != ""] == []

    def test_share_denominator_is_region_category(self):
        scopes = self._scopes()
        # March northeast category = 1 (e1); Kisqali took it all.
        assert scopes[("Kisqali", "northeast")]["2025-03-01"] == 1.0
        # March west category = 1 (e3, Fabhalta) → Kisqali west share = 0.0.
        assert scopes[("Kisqali", "west")]["2025-03-01"] == 0.0
        assert scopes[("Fabhalta", "west")]["2025-03-01"] == 1.0

    def test_empty_region_category_months_skipped(self):
        scopes = self._scopes()
        # April south/west have NO prescriptions: NULLIF semantics — no point.
        assert "2025-04-01" not in scopes[("Kisqali", "south")]
        assert "2025-04-01" not in scopes[("Fabhalta", "west")]


# ---------------------------------------------------------------------------
# WS3-BI-009 Conversion — patient-membership attribution
# ---------------------------------------------------------------------------


class TestConversionRegionAxis:
    _TRIGGERS = [
        {"trigger_id": "t0", "patient_id": "P1", "trigger_timestamp": "2025-03-01"},
        {"trigger_id": "t2", "patient_id": "P2", "trigger_timestamp": "2025-03-05"},
        {"trigger_id": "t3", "patient_id": "P3", "trigger_timestamp": "2025-03-20"},
        {"trigger_id": "t1", "patient_id": "P1", "trigger_timestamp": "2025-04-10"},
        {"trigger_id": "t4", "patient_id": "P5", "trigger_timestamp": "2025-04-25"},
        {"trigger_id": "t5", "patient_id": "P5", "trigger_timestamp": "2025-04-30"},
    ]

    def _scopes(self):
        client = _FakeClient(
            {
                "treatment_events": _PRESCRIPTIONS,
                "patient_journeys": _JOURNEYS,
                "triggers": self._TRIGGERS,
            }
        )
        points = asyncio.run(hb._backfill_conversion_rate(client, _meta("WS3-BI-009")))
        return _by_scope(points)

    def test_global_series_unchanged(self):
        scopes = self._scopes()
        assert scopes[("", "")]["2025-03-01"] == 1.0  # t0, t2, t3 all convert
        assert scopes[("", "")]["2025-04-01"] == 1.0 / 3.0  # only t1 converts

    def test_multi_region_patient_counts_in_both(self):
        scopes = self._scopes()
        # P2's March trigger converts; P2 has journeys in south AND west.
        assert scopes[("", "south")]["2025-03-01"] == 1.0
        assert scopes[("", "west")]["2025-03-01"] == 1.0

    def test_empty_region_months_skipped(self):
        scopes = self._scopes()
        # No south/west triggers in April → no point (never a fabricated 0.0).
        assert "2025-04-01" not in scopes[("", "south")]
        assert "2025-04-01" not in scopes[("", "west")]

    def test_region_series_only_for_journeyed_patients(self):
        scopes = self._scopes()
        # P5 has no journeys: its triggers exist globally, in no region.
        assert scopes[("", "northeast")]["2025-04-01"] == 1.0  # t1 (P1) only


# ---------------------------------------------------------------------------
# WS2-TR-* — trigger family (membership + global-frontier maturation)
# ---------------------------------------------------------------------------


def _trigger_row(tid, pid, ts, **kw):
    row = {
        "trigger_id": tid,
        "patient_id": pid,
        "trigger_timestamp": ts,
        "delivery_status": None,
        "acceptance_status": None,
        "false_positive_flag": False,
        "lead_time_days": None,
        "outcome_tracked": False,
        "outcome_value": None,
        "previous_trigger_id": None,
        "change_failed": False,
    }
    row.update(kw)
    return row


class TestTriggerRatioRegionAxis:
    _TRIGGERS = [
        _trigger_row(
            "a1", "P1", "2025-03-01", delivery_status="delivered", acceptance_status="accepted"
        ),
        _trigger_row(
            "a2", "P1", "2025-03-10", delivery_status="delivered", acceptance_status="rejected"
        ),
        _trigger_row(
            "a3", "P2", "2025-03-05", delivery_status="delivered", acceptance_status="accepted"
        ),
        _trigger_row(
            "a4", "P5", "2025-03-20", delivery_status="delivered", acceptance_status="accepted"
        ),
        _trigger_row(
            "a5", "P3", "2025-04-30", delivery_status="delivered", acceptance_status="accepted"
        ),
        _trigger_row(
            "a6", "P1", "2025-04-05", delivery_status="viewed", acceptance_status="overridden"
        ),
    ]

    def _scopes(self):
        client = _FakeClient({"triggers": self._TRIGGERS, "patient_journeys": _JOURNEYS})
        points = asyncio.run(hb._backfill_tr004_acceptance(client, _meta("WS2-TR-004")))
        return _by_scope(points)

    def test_global_series_unchanged(self):
        scopes = self._scopes()
        assert scopes[("", "")]["2025-03-01"] == 3.0 / 4.0
        assert scopes[("", "")]["2025-04-01"] == 1.0 / 2.0

    def test_region_membership_series(self):
        scopes = self._scopes()
        assert scopes[("", "northeast")]["2025-03-01"] == 1.0 / 2.0  # a1 acc, a2 rej
        assert scopes[("", "northeast")]["2025-04-01"] == 1.0 / 2.0  # a5 acc, a6 over

    def test_multi_region_patient_duplicates_series(self):
        scopes = self._scopes()
        # P2 (south+west): identical 1/1 March reading in both regions.
        assert scopes[("", "south")] == {"2025-03-01": 1.0}
        assert scopes[("", "west")] == {"2025-03-01": 1.0}


class TestTr001MaturationRegionAxis:
    _TRIGGERS = [
        # Boundary triggers pin the complete-month span (untracked: no den).
        _trigger_row("m0", "P5", "2025-03-01"),
        _trigger_row("m9", "P5", "2025-04-30"),
        _trigger_row(
            "m1",
            "P1",
            "2025-03-15",
            acceptance_status="accepted",
            outcome_tracked=True,
            outcome_value=1.0,
        ),
        _trigger_row(
            "m2",
            "P1",
            "2025-04-10",
            acceptance_status="accepted",
            outcome_tracked=True,
            outcome_value=1.0,
        ),
    ]

    def test_cutoff_is_global_frontier_not_per_region(self):
        client = _FakeClient({"triggers": self._TRIGGERS, "patient_journeys": _JOURNEYS})
        points = asyncio.run(hb._backfill_tr001_precision(client, _meta("WS2-TR-001")))
        scopes = _by_scope(points)
        # Global frontier 2025-04-30 → cutoff 03-31: m1 (03-15) matured,
        # m2 (04-10) not. A per-region (northeast) frontier would be 04-10 →
        # cutoff 03-11, wrongly excluding m1 and erasing the March point.
        assert scopes[("", "northeast")] == {"2025-03-01": 1.0}


class TestTr007LeadTimeRegionAxis:
    _TRIGGERS = [
        _trigger_row("l0", "P5", "2025-03-01", lead_time_days=20.0),
        _trigger_row("l1", "P1", "2025-03-05", lead_time_days=5.0),
        _trigger_row("l2", "P3", "2025-03-10", lead_time_days=7.0),
        _trigger_row("l3", "P2", "2025-03-15", lead_time_days=10.0),
        _trigger_row("l4", "P5", "2025-03-31"),
    ]

    def test_median_per_region(self):
        client = _FakeClient({"triggers": self._TRIGGERS, "patient_journeys": _JOURNEYS})
        points = asyncio.run(hb._backfill_tr007_lead_time(client, _meta("WS2-TR-007")))
        scopes = _by_scope(points)
        assert scopes[("", "")]["2025-03-01"] == 8.5  # median(5, 7, 10, 20)
        assert scopes[("", "northeast")]["2025-03-01"] == 6.0  # median(5, 7)
        assert scopes[("", "south")]["2025-03-01"] == 10.0
        assert scopes[("", "west")]["2025-03-01"] == 10.0


# ---------------------------------------------------------------------------
# Region-axis declaration + migration lockstep
# ---------------------------------------------------------------------------


class TestRegionAxisLockstep:
    _EXPECTED = {
        "WS3-BI-010": (
            "125_kpi_roi_headline_scoping.sql",
            "business_impact_roi_business_metrics_scoped",
        ),
        "WS3-BI-005": ("077_kpi_region_variants.sql", "business_impact_trx_region"),
        "WS3-BI-006": ("077_kpi_region_variants.sql", "business_impact_nrx_region"),
        "WS3-BI-007": ("077_kpi_region_variants.sql", "business_impact_nbrx_region"),
        "WS3-BI-008": ("077_kpi_region_variants.sql", "business_impact_trx_share_region"),
        "WS3-BI-009": ("077_kpi_region_variants.sql", "business_impact_conversion_rate_region"),
        "WS2-TR-001": (
            "113_kpi_ws2_truth_metrics_brand_variants.sql",
            "trigger_performance_precision_region",
        ),
        "WS2-TR-004": (
            "078_kpi_region_variants_trigger_dataquality.sql",
            "trigger_performance_acceptance_rate_region",
        ),
        "WS2-TR-005": (
            "078_kpi_region_variants_trigger_dataquality.sql",
            "trigger_performance_false_alert_rate_region",
        ),
        "WS2-TR-006": (
            "078_kpi_region_variants_trigger_dataquality.sql",
            "trigger_performance_override_rate_region",
        ),
        "WS2-TR-007": (
            "078_kpi_region_variants_trigger_dataquality.sql",
            "trigger_performance_lead_time_region",
        ),
        "WS2-TR-008": (
            "078_kpi_region_variants_trigger_dataquality.sql",
            "trigger_performance_cfr_region",
        ),
    }

    def test_declared_set_matches_expected(self):
        assert hb.REGION_AXIS_KPI_IDS == frozenset(self._EXPECTED)

    def test_every_region_axis_kpi_is_backfilled(self):
        assert hb.REGION_AXIS_KPI_IDS <= set(hb.HANDLERS)

    def test_every_region_axis_kpi_has_live_region_variant(self):
        migrations_dir = Path(__file__).resolve().parents[3] / "database" / "migrations"
        for kpi_id, (migration, query_id) in self._EXPECTED.items():
            sql = (migrations_dir / migration).read_text()
            assert f"'{query_id}'" in sql, (
                f"{kpi_id}: live region variant {query_id!r} not found in {migration}"
            )

    def test_non_region_axis_handlers_stay_out(self):
        # MAU/WAU (no region substrate) and BR-* (no live region routing) must
        # never grow a region axis without a live variant to mirror.
        for kpi_id in ("WS3-BI-001", "WS3-BI-002", "BR-001", "BR-002", "BR-003", "BR-004"):
            assert kpi_id not in hb.REGION_AXIS_KPI_IDS


# ---------------------------------------------------------------------------
# Region label case canon (codex iter-1 finding 2)
# ---------------------------------------------------------------------------


class TestRegionCaseCanon:
    """Every live region variant matches ``LOWER(region) = LOWER($n)``
    (077/078/125). The mirror applies LOWER at the substrate-read seams, so a
    mixed-case label merges into ONE canonical lowercase series instead of
    forking a duplicate scope. No trim — live ``LOWER()`` does not trim either.
    """

    def test_journey_regions_lowercase_canon(self):
        rows = [
            {"patient_journey_id": "J1", "patient_id": "P1", "geographic_region": "Northeast"},
            {"patient_journey_id": "J2", "patient_id": "P2", "geographic_region": "northeast"},
        ]
        assert hb._journey_regions(rows) == {"J1": "northeast", "J2": "northeast"}

    def test_patient_regions_lowercase_canon(self):
        rows = [
            {"patient_journey_id": "J1", "patient_id": "P1", "geographic_region": "WEST"},
            {"patient_journey_id": "J2", "patient_id": "P1", "geographic_region": "west"},
        ]
        assert hb._patient_regions(rows)["P1"] == {"west"}

    def test_roi_mixed_case_regions_merge_into_one_series(self):
        rows = [
            {"metric_date": "2025-03-01", "brand": "Kisqali", "roi": 1.0, "region": "Northeast"},
            {"metric_date": "2025-03-01", "brand": "Kisqali", "roi": 3.0, "region": "northeast"},
        ]
        client = _FakeClient({"business_metrics": rows})
        points = asyncio.run(hb._backfill_roi(client, _meta("WS3-BI-010")))
        scopes = _by_scope(points)
        assert ("", "Northeast") not in scopes
        assert scopes[("", "northeast")]["2025-03-01"] == 2.0  # merged mean, not a forked scope


class TestHistoryReadSeamCaseInsensitive:
    """``/api/kpis/{id}/history?region=`` mirrors the live variants'
    ``LOWER(region) = LOWER($n)``: stored canon is lowercase (write seam
    lowercases), so the read seam lowercases the filter input. Without this a
    caller sending ``region=Northeast`` gets a live current value but an empty
    history series (codex iter-1 finding 2).
    """

    def test_get_history_matches_mixed_case_region_input(self):
        from src.repositories.kpi_history import KPIHistoryRepository

        rows = [
            {
                "kpi_id": "WS3-BI-005",
                "brand": "",
                "region": "northeast",
                "metric_date": "2025-03-01",
                "value": 249.0,
            },
        ]
        repo = KPIHistoryRepository(supabase_client=_FakeClient({"kpi_history": rows}))
        out = asyncio.run(repo.get_history("WS3-BI-005", brand="", region="Northeast"))
        assert [r["metric_date"] for r in out] == ["2025-03-01"]
