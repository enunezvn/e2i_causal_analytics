"""Tests for the KPI-aware data resolution service (issue #810).

Discipline: NO mocking of the resolution LOGIC. The pure functions
(``_compute_conversion_outcome`` / ``_assemble_conversion_frame``) are exercised
on REAL-shaped DataFrames so the real logic runs; the DB integration is verified
by a FAITHFUL test against the real Supabase (reachability-gated), never a mocked
DB. Nothing is hardcoded to a specific brand or region — brand/region are
parameters matched against the actual data values.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.services import kpi_resolution as kr

# ---------------------------------------------------------------------------
# recognize_kpi — registry-driven (all defined KPIs), dynamic, not keyed to "conversion"
# ---------------------------------------------------------------------------


def test_recognize_conversion_kpi():
    kpi = kr.recognize_kpi("What drove Kisqali conversion in the Northeast?")
    assert kpi is not None
    assert kpi.id == kr.CONVERSION_KPI_ID  # WS3-BI-009
    assert kpi.name == "Conversion Rate"


def test_recognize_is_brand_region_agnostic():
    # Same KPI recognized regardless of brand/region in the query.
    for q in (
        "what drove Fabhalta conversion in the west",
        "Remibrutinib conversion rate south",
        "conversion drivers",
    ):
        assert kr.recognize_kpi(q).id == kr.CONVERSION_KPI_ID


def test_recognize_other_kpis_dynamically():
    # Recognition generalizes across the registry, not just conversion.
    assert kr.recognize_kpi("show me the TRx trend").id == "WS3-BI-005"
    assert kr.recognize_kpi("what is the ROI for this campaign").id == "WS3-BI-010"


def test_recognize_none_when_no_kpi():
    assert kr.recognize_kpi("hello, how are you today?") is None


# ---------------------------------------------------------------------------
# _compute_conversion_outcome — pure real logic (prescription within window)
# ---------------------------------------------------------------------------


def _triggers(rows):
    return pd.DataFrame(rows)


def test_conversion_within_window_is_1():
    trig = _triggers(
        [{"trigger_id": "t1", "patient_id": "p1", "trigger_timestamp": "2026-01-01T00:00:00Z"}]
    )
    ev = pd.DataFrame([{"patient_id": "p1", "event_date": "2026-01-10"}])  # 9 days later
    out = kr._compute_conversion_outcome(trig, ev, window_days=30)
    assert list(out) == [1]


def test_conversion_outside_window_is_0():
    trig = _triggers(
        [{"trigger_id": "t1", "patient_id": "p1", "trigger_timestamp": "2026-01-01T00:00:00Z"}]
    )
    ev = pd.DataFrame([{"patient_id": "p1", "event_date": "2026-03-01"}])  # ~59 days later
    out = kr._compute_conversion_outcome(trig, ev, window_days=30)
    assert list(out) == [0]


def test_conversion_before_trigger_is_0():
    # A prescription BEFORE the trigger does not count as conversion.
    trig = _triggers(
        [{"trigger_id": "t1", "patient_id": "p1", "trigger_timestamp": "2026-02-01T00:00:00Z"}]
    )
    ev = pd.DataFrame([{"patient_id": "p1", "event_date": "2026-01-01"}])
    out = kr._compute_conversion_outcome(trig, ev, window_days=30)
    assert list(out) == [0]


def test_conversion_no_event_is_0():
    trig = _triggers(
        [{"trigger_id": "t1", "patient_id": "p1", "trigger_timestamp": "2026-01-01T00:00:00Z"}]
    )
    ev = pd.DataFrame([{"patient_id": "pX", "event_date": "2026-01-10"}])
    out = kr._compute_conversion_outcome(trig, ev, window_days=30)
    assert list(out) == [0]


def test_conversion_mixed_population():
    trig = _triggers(
        [
            {"trigger_id": "t1", "patient_id": "p1", "trigger_timestamp": "2026-01-01T00:00:00Z"},
            {"trigger_id": "t2", "patient_id": "p2", "trigger_timestamp": "2026-01-01T00:00:00Z"},
            {"trigger_id": "t3", "patient_id": "p3", "trigger_timestamp": "2026-01-01T00:00:00Z"},
        ]
    )
    ev = pd.DataFrame(
        [
            {"patient_id": "p1", "event_date": "2026-01-15"},  # in window -> 1
            {"patient_id": "p2", "event_date": "2026-06-01"},  # out of window -> 0
        ]
    )
    out = kr._compute_conversion_outcome(trig, ev, window_days=30)
    assert list(out) == [1, 0, 0]


# ---------------------------------------------------------------------------
# _assemble_conversion_frame — pure: region filter + drivers + fail-closed
# ---------------------------------------------------------------------------


def _real_shaped_inputs():
    triggers = pd.DataFrame(
        [
            {
                "trigger_id": f"t{i}",
                "patient_id": f"p{i}",
                "hcp_id": "HCP_NE" if i % 2 == 0 else "HCP_W",
                "trigger_timestamp": "2026-01-01T00:00:00Z",
                "trigger_type": "adherence_risk",
                "delivery_channel": "email",
                "priority": "high",
                "confidence_score": 0.7 + 0.01 * i,
                "lead_time_days": 5 + i,
                "acceptance_status": "accepted" if i % 3 == 0 else "overridden",
            }
            for i in range(6)
        ]
    )
    hcp_regions = pd.DataFrame(
        [
            {"hcp_id": "HCP_NE", "geographic_region": "northeast"},
            {"hcp_id": "HCP_W", "geographic_region": "west"},
        ]
    )
    # prescriptions for some patients within window
    events = pd.DataFrame(
        [
            {"patient_id": "p0", "event_date": "2026-01-10"},
            {"patient_id": "p2", "event_date": "2026-01-20"},
        ]
    )
    return triggers, hcp_regions, events


def test_assemble_filters_region_and_builds_outcome():
    triggers, hcp_regions, events = _real_shaped_inputs()
    kf = kr._assemble_conversion_frame(
        triggers, hcp_regions, events, region_canonical="northeast", window_days=30
    )
    assert kf is not None
    # Only the 3 NE triggers (even-index hcp HCP_NE) survive the region filter.
    assert len(kf.frame) == 3
    assert kf.outcome_column == "converted"
    assert set(kf.frame[kf.outcome_column].unique()) <= {0, 1}
    # Driver columns are present and real.
    for col in ("trigger_type", "delivery_channel", "confidence_score", "lead_time_days"):
        assert col in kf.frame.columns
        assert col in kf.driver_columns
    assert kf.kpi_id == kr.CONVERSION_KPI_ID


def test_assemble_no_region_filter_keeps_all():
    triggers, hcp_regions, events = _real_shaped_inputs()
    kf = kr._assemble_conversion_frame(
        triggers, hcp_regions, events, region_canonical=None, window_days=30
    )
    assert kf is not None and len(kf.frame) == 6


def test_assemble_unrecognized_region_fails_closed():
    triggers, hcp_regions, events = _real_shaped_inputs()
    kf = kr._assemble_conversion_frame(
        triggers, hcp_regions, events, region_canonical="antarctica", window_days=30
    )
    assert kf is None


def test_assemble_empty_triggers_fails_closed():
    _, hcp_regions, events = _real_shaped_inputs()
    kf = kr._assemble_conversion_frame(
        pd.DataFrame(), hcp_regions, events, region_canonical=None, window_days=30
    )
    assert kf is None


# ---------------------------------------------------------------------------
# resolve_kpi_frame dispatch — honest "no builder yet" (never fabricates)
# ---------------------------------------------------------------------------


def test_resolve_unbuilt_kpi_returns_none():
    # A KPI with no substrate builder must return None (honest), not fabricate.
    from src.kpi.registry import get_registry

    trx = get_registry().get("WS3-BI-005")  # Total Prescriptions — no builder yet
    assert trx is not None
    assert kr.resolve_kpi_frame(trx, "Kisqali", "northeast") is None


# ---------------------------------------------------------------------------
# Brand resolution — case-insensitive distinct scan over a PG-enum column, with a
# NON-SILENT truncation signal (regression guard for the codex MED: a capped scan
# that misses the brand must report truncation, not a silent "unrecognized brand").
# Uses a tiny query stub at the DB boundary, never mocking the resolution logic.
# ---------------------------------------------------------------------------


class _BrandScanStub:
    """Minimal Supabase-query stub returning a fixed set of brand rows from a plain
    scan. ``treatment_events.brand`` is a PG enum, so resolution cannot use ILIKE;
    it scans distinct values and matches case-insensitively in Python."""

    def __init__(self, brand_rows):
        self._rows = list(brand_rows)

    def table(self, _name):
        return self

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    @property
    def not_(self):
        return self

    def is_(self, *_a, **_k):
        return self

    def limit(self, _n):
        return self

    def execute(self):
        from types import SimpleNamespace

        return SimpleNamespace(data=list(self._rows))


def test_brand_resolution_is_case_insensitive():
    stub = _BrandScanStub([{"brand": "Kisqali"}, {"brand": "Fabhalta"}])
    canon, truncated = kr._resolve_brand_canonical(stub, "kisqali")
    assert canon == "Kisqali" and truncated is False
    canon, truncated = kr._resolve_brand_canonical(stub, "  FABHALTA ")
    assert canon == "Fabhalta" and truncated is False


def test_brand_resolution_unknown_brand_fails_closed():
    stub = _BrandScanStub([{"brand": "Kisqali"}])
    canon, truncated = kr._resolve_brand_canonical(stub, "nonexistent_brand")
    assert canon is None and truncated is False
    canon, _ = kr._resolve_brand_canonical(stub, "")
    assert canon is None


def test_brand_scan_truncation_is_not_silent(monkeypatch):
    # codex MED regression guard: when the distinct scan hits the row cap AND the
    # requested brand is absent from the (truncated) sample, resolution must return
    # truncated=True so the caller can distinguish "absent" from "beyond the cap" —
    # never a silent fail-closed.
    monkeypatch.setattr(kr, "_MAX_ROWS", 2)
    stub = _BrandScanStub([{"brand": "Kisqali"}, {"brand": "Kisqali"}])  # len == cap
    canon, truncated = kr._resolve_brand_canonical(stub, "Remibrutinib")
    assert canon is None
    assert truncated is True


# ---------------------------------------------------------------------------
# FAITHFUL integration — real Supabase, dynamic across brands (reachability-gated)
# ---------------------------------------------------------------------------


def _supabase_reachable() -> bool:
    try:
        from src.repositories import get_supabase_client

        get_supabase_client().table("triggers").select("trigger_id").limit(1).execute()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _supabase_reachable(),
    reason="real Supabase not reachable (triggers table) — faithful KPI-frame test skipped",
)
@pytest.mark.parametrize("brand", ["Kisqali", "Fabhalta", "Remibrutinib"])
def test_resolve_conversion_frame_real_supabase(brand):
    kpi = kr.recognize_kpi(f"what drove {brand} conversion in the northeast")
    kf = kr.resolve_kpi_frame(kpi, brand, "northeast")
    assert kf is not None, f"no KPI frame resolved for {brand}/northeast"
    assert kf.outcome_column == "converted"
    assert len(kf.frame) > 0
    rate = kf.frame["converted"].mean()
    # Non-degenerate, real conversion outcome (not all-0 / all-1).
    assert 0.0 < rate < 1.0, f"{brand}: degenerate conversion rate {rate}"
    assert kf.driver_columns, "no driver columns exposed"
    # Real substrate is well under the row cap -> not a truncated sample.
    assert kf.is_truncated is False
