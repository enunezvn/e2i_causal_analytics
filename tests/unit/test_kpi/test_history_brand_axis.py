"""kpi_history brand axis — backfill handlers emit per-brand series.

The Time-Series page offers its brand selector / "Compare Brands" overlay only
for KPIs whose coverage carries ≥1 / ≥2 named brand scopes. Before this change
only the Rx family + ROI wrote brand rows; the trigger family and conversion
rate were global-only although their LIVE calculators serve brand-scoped
readings (migration 113 ``_brand`` variants; 111/128 for conversion).

Locked behaviors (each mirrors a vetted live brand variant — the backfill must
never invent a brand reading the live platform cannot produce):

- WS2-TR-* ratio KPIs: brand = ``triggers.brand_id`` (113: ``brand_id::text =
  $1`` — exact label, NOT lowercased, unlike regions); brand×region composes
  brand with patient MEMBERSHIP; empty-denominator months are skipped.
- Brand-less triggers (NULL ``brand_id``) count in the global series only —
  exactly what the live equality predicate does.
- WS2-TR-001 maturation cutoff stays anchored to the GLOBAL trigger frontier
  for every brand (113's unscoped ``MAX(trigger_timestamp)``), never a
  per-brand frontier.
- WS2-TR-007: median lead time over the brand's own triggers (× region).
- WS3-BI-009: a brand's trigger converts ONLY on a SAME-brand prescription
  (111 ``te.brand = $1``); the global series keeps the any-brand rule.
- Global + region series are byte-identical to the pre-brand-axis output.
- ``BRAND_AXIS_KPI_IDS`` lockstep: every brand-axis KPI maps to a live
  brand-capable registry variant defined in migrations 089/111/113/125.

All handler tests run against an in-memory fake Supabase client — no DB.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.kpi import history_backfill as hb

# ---------------------------------------------------------------------------
# Fake async Supabase client — same idiom as test_history_region_axis.
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


def _meta(kpi_id):
    return SimpleNamespace(id=kpi_id, threshold=None)


def _by_scope(points):
    """Index points as {(brand, region): {metric_date: value}}."""
    out = {}
    for p in points:
        out.setdefault((p["brand"], p["region"]), {})[p["metric_date"]] = p["value"]
    return out


# Journeys: P1/P3 northeast; P2 spans TWO regions (south + west); P4's journey
# has no region; P5 has no journey at all.
_JOURNEYS = [
    {"patient_journey_id": "J1", "patient_id": "P1", "geographic_region": "northeast"},
    {"patient_journey_id": "J2", "patient_id": "P2", "geographic_region": "south"},
    {"patient_journey_id": "J3", "patient_id": "P3", "geographic_region": "northeast"},
    {"patient_journey_id": "J4", "patient_id": "P2", "geographic_region": "west"},
    {"patient_journey_id": "J5", "patient_id": "P4", "geographic_region": None},
]


def _trigger_row(tid, pid, ts, brand_id=None, **kw):
    row = {
        "trigger_id": tid,
        "patient_id": pid,
        "brand_id": brand_id,
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


# ---------------------------------------------------------------------------
# _trigger_brand — exact label canon (no LOWER(), unlike regions)
# ---------------------------------------------------------------------------


class TestTriggerBrandLabel:
    def test_null_or_empty_brand_is_global_only(self):
        assert hb._trigger_brand({"brand_id": None}) == ""
        assert hb._trigger_brand({"brand_id": ""}) == ""
        assert hb._trigger_brand({}) == ""

    def test_label_kept_verbatim(self):
        # 113: ``brand_id::text = $1`` is case-sensitive — the stored canonical
        # label is what the FE sends back as ?brand=, so it must not be folded.
        assert hb._trigger_brand({"brand_id": "Kisqali"}) == "Kisqali"
        assert hb._trigger_brand({"brand_id": "Remibrutinib"}) == "Remibrutinib"


# ---------------------------------------------------------------------------
# WS2-TR-004 acceptance — shared ratio helper (brand + brand×region)
# ---------------------------------------------------------------------------


class TestTriggerRatioBrandAxis:
    # Same rows as the region-axis fixture, now carrying brand ids; a6 is
    # brand-less.
    _TRIGGERS = [
        _trigger_row(
            "a1",
            "P1",
            "2025-03-01",
            "Kisqali",
            delivery_status="delivered",
            acceptance_status="accepted",
        ),
        _trigger_row(
            "a2",
            "P1",
            "2025-03-10",
            "Kisqali",
            delivery_status="delivered",
            acceptance_status="rejected",
        ),
        _trigger_row(
            "a3",
            "P2",
            "2025-03-05",
            "Fabhalta",
            delivery_status="delivered",
            acceptance_status="accepted",
        ),
        _trigger_row(
            "a4",
            "P5",
            "2025-03-20",
            "Fabhalta",
            delivery_status="delivered",
            acceptance_status="accepted",
        ),
        _trigger_row(
            "a5",
            "P3",
            "2025-04-30",
            "Kisqali",
            delivery_status="delivered",
            acceptance_status="accepted",
        ),
        _trigger_row(
            "a6",
            "P1",
            "2025-04-05",
            None,
            delivery_status="viewed",
            acceptance_status="overridden",
        ),
    ]

    def _scopes(self):
        client = _FakeClient({"triggers": self._TRIGGERS, "patient_journeys": _JOURNEYS})
        points = asyncio.run(hb._backfill_tr004_acceptance(client, _meta("WS2-TR-004")))
        return _by_scope(points)

    def test_global_and_region_series_unchanged(self):
        scopes = self._scopes()
        # Identical to the pre-brand-axis region test expectations.
        assert scopes[("", "")]["2025-03-01"] == 3.0 / 4.0
        assert scopes[("", "")]["2025-04-01"] == 1.0 / 2.0
        assert scopes[("", "northeast")]["2025-03-01"] == 1.0 / 2.0
        assert scopes[("", "northeast")]["2025-04-01"] == 1.0 / 2.0
        assert scopes[("", "south")] == {"2025-03-01": 1.0}
        assert scopes[("", "west")] == {"2025-03-01": 1.0}

    def test_brand_series_scope_to_brand_id(self):
        scopes = self._scopes()
        assert scopes[("Kisqali", "")]["2025-03-01"] == 1.0 / 2.0  # a1 acc, a2 rej
        assert scopes[("Kisqali", "")]["2025-04-01"] == 1.0  # a5 only
        assert scopes[("Fabhalta", "")] == {"2025-03-01": 1.0}  # a3, a4

    def test_empty_brand_months_skipped(self):
        scopes = self._scopes()
        # No Fabhalta triggers in April → no point (never a fabricated 0.0).
        assert "2025-04-01" not in scopes[("Fabhalta", "")]

    def test_brandless_trigger_counts_globally_only(self):
        scopes = self._scopes()
        # a6 (NULL brand_id) is in the April global denominator (1/2) but in
        # no brand series — Kisqali April is a5 alone (1/1).
        assert scopes[("", "")]["2025-04-01"] == 1.0 / 2.0
        assert scopes[("Kisqali", "")]["2025-04-01"] == 1.0
        assert {b for b, _ in scopes} == {"", "Kisqali", "Fabhalta"}

    def test_brand_region_series_compose_membership(self):
        scopes = self._scopes()
        assert scopes[("Kisqali", "northeast")]["2025-03-01"] == 1.0 / 2.0  # a1, a2 (P1)
        assert scopes[("Kisqali", "northeast")]["2025-04-01"] == 1.0  # a5 (P3)
        # P2 spans south + west: a3 lands in both Fabhalta×region series.
        assert scopes[("Fabhalta", "south")] == {"2025-03-01": 1.0}
        assert scopes[("Fabhalta", "west")] == {"2025-03-01": 1.0}
        # a4 (P5, no journey) is in Fabhalta global only — no Fabhalta northeast.
        assert ("Fabhalta", "northeast") not in scopes

    def test_brand_labels_not_lowercased(self):
        scopes = self._scopes()
        assert ("kisqali", "") not in scopes
        assert ("Kisqali", "") in scopes


# ---------------------------------------------------------------------------
# WS2-TR-001 — maturation cutoff is the GLOBAL frontier for every brand
# ---------------------------------------------------------------------------


class TestTr001MaturationBrandAxis:
    _TRIGGERS = [
        # Fabhalta boundary triggers pin the complete-month span (untracked).
        _trigger_row("m0", "P5", "2025-03-01", "Fabhalta"),
        _trigger_row("m9", "P5", "2025-04-30", "Fabhalta"),
        _trigger_row(
            "m1",
            "P1",
            "2025-03-15",
            "Kisqali",
            acceptance_status="accepted",
            outcome_tracked=True,
            outcome_value=1.0,
        ),
        _trigger_row(
            "m2",
            "P1",
            "2025-04-10",
            "Kisqali",
            acceptance_status="accepted",
            outcome_tracked=True,
            outcome_value=1.0,
        ),
    ]

    def test_cutoff_is_global_frontier_not_per_brand(self):
        client = _FakeClient({"triggers": self._TRIGGERS, "patient_journeys": _JOURNEYS})
        points = asyncio.run(hb._backfill_tr001_precision(client, _meta("WS2-TR-001")))
        scopes = _by_scope(points)
        # Global frontier 2025-04-30 → cutoff 03-31: m1 (03-15) matured, m2
        # (04-10) not. A per-brand (Kisqali) frontier would be 04-10 → cutoff
        # 03-11, wrongly excluding m1 and erasing the Kisqali March point.
        assert scopes[("Kisqali", "")] == {"2025-03-01": 1.0}
        assert scopes[("Kisqali", "northeast")] == {"2025-03-01": 1.0}
        # Fabhalta has no tracked-accepted triggers → no Fabhalta series.
        assert ("Fabhalta", "") not in scopes


# ---------------------------------------------------------------------------
# WS2-TR-007 — median lead time per brand (× region)
# ---------------------------------------------------------------------------


class TestTr007LeadTimeBrandAxis:
    _TRIGGERS = [
        _trigger_row("l0", "P5", "2025-03-01", "Fabhalta", lead_time_days=20.0),
        _trigger_row("l1", "P1", "2025-03-05", "Kisqali", lead_time_days=5.0),
        _trigger_row("l2", "P3", "2025-03-10", "Kisqali", lead_time_days=7.0),
        _trigger_row("l3", "P2", "2025-03-15", "Fabhalta", lead_time_days=10.0),
        _trigger_row("l4", "P5", "2025-03-31", None),
    ]

    def _scopes(self):
        client = _FakeClient({"triggers": self._TRIGGERS, "patient_journeys": _JOURNEYS})
        points = asyncio.run(hb._backfill_tr007_lead_time(client, _meta("WS2-TR-007")))
        return _by_scope(points)

    def test_global_and_region_medians_unchanged(self):
        scopes = self._scopes()
        assert scopes[("", "")]["2025-03-01"] == 8.5  # median(5, 7, 10, 20)
        assert scopes[("", "northeast")]["2025-03-01"] == 6.0  # median(5, 7)
        assert scopes[("", "south")]["2025-03-01"] == 10.0
        assert scopes[("", "west")]["2025-03-01"] == 10.0

    def test_median_per_brand(self):
        scopes = self._scopes()
        assert scopes[("Kisqali", "")]["2025-03-01"] == 6.0  # median(5, 7)
        assert scopes[("Fabhalta", "")]["2025-03-01"] == 15.0  # median(20, 10)

    def test_median_per_brand_region(self):
        scopes = self._scopes()
        assert scopes[("Kisqali", "northeast")]["2025-03-01"] == 6.0  # P1 + P3
        assert scopes[("Fabhalta", "south")]["2025-03-01"] == 10.0  # P2 only
        assert scopes[("Fabhalta", "west")]["2025-03-01"] == 10.0
        assert ("Fabhalta", "northeast") not in scopes
        assert ("Kisqali", "south") not in scopes


# ---------------------------------------------------------------------------
# WS3-BI-009 conversion — same-brand conversion rule
# ---------------------------------------------------------------------------


class TestConversionBrandAxis:
    _PRESCRIPTIONS = [
        {
            "event_type": "prescription",
            "patient_id": "P1",
            "brand": "Kisqali",
            "event_date": "2025-03-05",
            "sequence_number": 1,
            "patient_journey_id": "J1",
        },
        {
            "event_type": "prescription",
            "patient_id": "P2",
            "brand": "Fabhalta",
            "event_date": "2025-03-10",
            "sequence_number": 1,
            "patient_journey_id": "J2",
        },
        {
            "event_type": "prescription",
            "patient_id": "P3",
            "brand": "Kisqali",
            "event_date": "2025-04-15",
            "sequence_number": 1,
            "patient_journey_id": "J3",
        },
        {
            "event_type": "prescription",
            "patient_id": "P1",
            "brand": "Fabhalta",
            "event_date": "2025-04-20",
            "sequence_number": 2,
            "patient_journey_id": "J1",
        },
    ]
    _TRIGGERS = [
        _trigger_row("t0", "P1", "2025-03-01", "Kisqali"),  # → Kisqali rx 03-05 (same brand)
        _trigger_row("t2", "P2", "2025-03-05", "Kisqali"),  # → Fabhalta rx 03-10 (cross-brand)
        _trigger_row("t3", "P3", "2025-03-20", "Kisqali"),  # → Kisqali rx 04-15 (26 days)
        _trigger_row("t1", "P1", "2025-04-10", "Fabhalta"),  # → Fabhalta rx 04-20 (same brand)
        _trigger_row("t4", "P5", "2025-04-25", "Fabhalta"),  # no rx
        _trigger_row("t5", "P5", "2025-04-30", None),  # brand-less, no rx
    ]

    def _scopes(self):
        client = _FakeClient(
            {
                "treatment_events": self._PRESCRIPTIONS,
                "patient_journeys": _JOURNEYS,
                "triggers": self._TRIGGERS,
            }
        )
        points = asyncio.run(hb._backfill_conversion_rate(client, _meta("WS3-BI-009")))
        return _by_scope(points)

    def test_global_series_keeps_any_brand_rule(self):
        scopes = self._scopes()
        assert scopes[("", "")]["2025-03-01"] == 1.0  # t0, t2, t3 all convert
        assert scopes[("", "")]["2025-04-01"] == 1.0 / 3.0  # only t1 converts

    def test_cross_brand_prescription_does_not_convert_in_brand_series(self):
        scopes = self._scopes()
        # Kisqali March: t0 ✔, t2 ✘ (Fabhalta script), t3 ✔ → 2/3 — while the
        # global March reading counts t2 as converted (1.0 above).
        assert scopes[("Kisqali", "")]["2025-03-01"] == 2.0 / 3.0

    def test_brand_series_scope_to_brand_triggers(self):
        scopes = self._scopes()
        assert scopes[("Fabhalta", "")] == {"2025-04-01": 1.0 / 2.0}  # t1 ✔, t4 ✘
        assert "2025-04-01" not in scopes[("Kisqali", "")]  # no Kisqali April triggers
        assert {b for b, _ in scopes} == {"", "Kisqali", "Fabhalta"}  # t5 global only

    def test_brand_region_series(self):
        scopes = self._scopes()
        assert scopes[("Kisqali", "northeast")]["2025-03-01"] == 1.0  # t0 (P1), t3 (P3)
        # P2's Kisqali trigger converted on a Fabhalta script: a genuine 0.0 in
        # both of P2's regions (cohort exists, none converted) — not skipped.
        assert scopes[("Kisqali", "south")] == {"2025-03-01": 0.0}
        assert scopes[("Kisqali", "west")] == {"2025-03-01": 0.0}
        assert scopes[("Fabhalta", "northeast")] == {"2025-04-01": 1.0}  # t1 (P1)


# ---------------------------------------------------------------------------
# Brand-axis declaration + migration lockstep
# ---------------------------------------------------------------------------


class TestBrandAxisLockstep:
    # kpi_id -> (migration defining the live brand-capable variant, query_id).
    # The Rx family's BASE statements take the brand as ``$1`` (089); ROI reads
    # business_metrics.brand directly (125); conversion + the trigger family
    # have dedicated ``_brand`` variants (111 / 113).
    _EXPECTED = {
        "WS3-BI-010": (
            "125_kpi_roi_headline_scoping.sql",
            "business_impact_roi_business_metrics_scoped",
        ),
        "WS3-BI-005": ("089_kpi_data_frontier_anchoring.sql", "business_impact_trx"),
        "WS3-BI-006": ("089_kpi_data_frontier_anchoring.sql", "business_impact_nrx"),
        "WS3-BI-007": ("089_kpi_data_frontier_anchoring.sql", "business_impact_nbrx"),
        "WS3-BI-008": ("089_kpi_data_frontier_anchoring.sql", "business_impact_trx_share"),
        "WS3-BI-009": (
            "111_kpi_conversion_share_axis_window.sql",
            "business_impact_conversion_rate_brand",
        ),
        "WS2-TR-001": (
            "113_kpi_ws2_truth_metrics_brand_variants.sql",
            "trigger_performance_precision_brand",
        ),
        "WS2-TR-004": (
            "113_kpi_ws2_truth_metrics_brand_variants.sql",
            "trigger_performance_acceptance_rate_brand",
        ),
        "WS2-TR-005": (
            "113_kpi_ws2_truth_metrics_brand_variants.sql",
            "trigger_performance_false_alert_rate_brand",
        ),
        "WS2-TR-006": (
            "113_kpi_ws2_truth_metrics_brand_variants.sql",
            "trigger_performance_override_rate_brand",
        ),
        "WS2-TR-007": (
            "113_kpi_ws2_truth_metrics_brand_variants.sql",
            "trigger_performance_lead_time_brand",
        ),
        "WS2-TR-008": (
            "113_kpi_ws2_truth_metrics_brand_variants.sql",
            "trigger_performance_cfr_brand",
        ),
    }

    def test_declared_set_matches_expected(self):
        assert hb.BRAND_AXIS_KPI_IDS == frozenset(self._EXPECTED)

    def test_every_brand_axis_kpi_is_backfilled(self):
        assert hb.BRAND_AXIS_KPI_IDS <= set(hb.HANDLERS)

    def test_every_brand_axis_kpi_has_live_brand_variant(self):
        migrations_dir = Path(__file__).resolve().parents[3] / "database" / "migrations"
        for kpi_id, (migration, query_id) in self._EXPECTED.items():
            sql = (migrations_dir / migration).read_text()
            assert f"'{query_id}'" in sql, (
                f"{kpi_id}: live brand variant {query_id!r} not found in {migration}"
            )

    def test_trigger_family_variants_filter_brand_id(self):
        # The mirrored predicate: 113's ``_brand`` variants scope on
        # triggers.brand_id (exact text match) — the column the backfill reads.
        migrations_dir = Path(__file__).resolve().parents[3] / "database" / "migrations"
        sql = (migrations_dir / "113_kpi_ws2_truth_metrics_brand_variants.sql").read_text()
        assert "brand_id::text = $1" in sql

    def test_non_brand_axis_handlers_stay_out(self):
        # MAU/WAU (user_sessions has no brand column) and BR-* (single-brand by
        # definition) must never grow a brand axis.
        for kpi_id in ("WS3-BI-001", "WS3-BI-002", "BR-001", "BR-002", "BR-003", "BR-004"):
            assert kpi_id not in hb.BRAND_AXIS_KPI_IDS

    def test_trigger_fetch_selects_brand_id(self, monkeypatch):
        # The brand axis is only as honest as the column it reads: the shared
        # trigger fetch must request brand_id (silently absent → every
        # trigger would look brand-less and the axis would vanish).
        seen = {}

        async def fake_fetch_all(client, table, columns, order_col, **kw):
            seen[table] = columns
            return []

        monkeypatch.setattr(hb, "_fetch_all", fake_fetch_all)
        asyncio.run(hb._fetch_triggers(object(), None))
        assert "brand_id" in seen["triggers"].split(",")
