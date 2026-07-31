"""Unit tests for ``scripts.check_kpi_coverage`` id resolution (#1389).

The coverage probe used to HARD-CODE each KPI's ``*_include_synthetic`` twin id
in its ``PROBES`` map. When migrations 085 (patient_touch_rate) and 095 (the four
view-backed WS1 data-quality ids) added twins for ids the map still pointed at
their BASE (synthetic-excluding) form, five rows were stranded on base ids that
read honest-null on the synthetic-gold instance — the script reported
``TOTAL 45 MAPPED 40 EMPTY 5`` after the 2026-07-30 provenance reseed even though
the platform was healthy (chat resolves the twins via
``resolve_kpi_query_id`` / ``trigger_effectiveness_query_id``).

The fix stores BASE registry ids in ``PROBES`` and routes every id through the
SAME production resolver the deployed KPI path uses, so a future twin migration
is picked up automatically instead of silently stranding another row. These
tests pin that contract; they need no DB and always run.
"""

from __future__ import annotations

from scripts.check_kpi_coverage import PROBES
from src.kpi.synthetic_mode import SYNTHETIC_TWINNED_QUERY_IDS

_SUFFIX = "_include_synthetic"
# The five rows #1389 stranded on base ids, with the JSON key each twin populates.
_AFFECTED = {
    "WS1-DQ-003": "data_quality_cross_source_match",
    "WS1-DQ-004": "data_quality_stacking_lift",
    "WS1-DQ-007": "data_quality_data_lag",
    "WS1-DQ-009": "data_quality_time_to_release",
    "WS3-BI-003": "business_impact_patient_touch_rate",
}
# WS2-TR-009: an ADDITIVE trigger-effectiveness id (migration 118, #1360) that is
# deliberately ABSENT from SYNTHETIC_TWINNED_QUERY_IDS and resolved by its own
# helper, not resolve_kpi_query_id.
_ADDITIVE_BASE = "trigger_effectiveness_funnel_conversion"


def test_no_probes_row_hardcodes_a_twin_suffix():
    """Every ``PROBES`` id must be stored as a BASE registry id.

    A hard-coded ``_include_synthetic`` literal is exactly the drift that
    stranded the five #1389 rows: migrations 085/095 added their twins but the
    hand-maintained map kept probing base. Storing base ids and resolving
    dynamically makes a future twin migration a zero-line change here.
    """
    offenders = {kid: qid for kid, (qid, _p, _k) in PROBES.items() if qid.endswith(_SUFFIX)}
    assert not offenders, f"PROBES rows hard-code twin ids (store base instead): {offenders}"


def test_five_affected_kpis_still_present_as_base():
    """Regression guard: the five #1389 ids stay mapped to their base registry ids."""
    for kid, base in _AFFECTED.items():
        assert kid in PROBES, f"{kid} dropped from PROBES"
        assert PROBES[kid][0] == base, f"{kid} maps to {PROBES[kid][0]!r}, expected base {base!r}"
        assert base in SYNTHETIC_TWINNED_QUERY_IDS, f"{base} lost its twin in the frozenset"


def test_five_affected_kpis_resolve_to_twins_under_synthetic_mode(monkeypatch):
    """Under the deployed synthetic flag the five ids resolve to their twins
    (the ones DB-verified to return live values on the synthetic-gold instance)."""
    from scripts.check_kpi_coverage import resolved_probe_id

    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "true")
    for base in _AFFECTED.values():
        assert resolved_probe_id(base) == f"{base}{_SUFFIX}"


def test_five_affected_kpis_stay_base_when_flag_off(monkeypatch):
    """With the flag off the production strict-exclusion gate is preserved — the
    resolver returns the base id untouched (mirrors the sibling resolver tests)."""
    from scripts.check_kpi_coverage import resolved_probe_id

    monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    for base in _AFFECTED.values():
        assert resolved_probe_id(base) == base


def test_additive_trigger_effectiveness_resolves_via_its_own_helper(monkeypatch):
    """WS2-TR-009 is an additive #1360 id OUTSIDE the resolve_kpi_query_id twin
    family; it must resolve through trigger_effectiveness_query_id (base when the
    flag is off, twin when on) — DB-verified: base funnel_conversion is null,
    twin returns 0.2219 on the synthetic-gold instance."""
    from scripts.check_kpi_coverage import resolved_probe_id

    assert _ADDITIVE_BASE not in SYNTHETIC_TWINNED_QUERY_IDS
    assert PROBES["WS2-TR-009"][0] == _ADDITIVE_BASE

    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "true")
    assert resolved_probe_id(_ADDITIVE_BASE) == f"{_ADDITIVE_BASE}{_SUFFIX}"

    monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    assert resolved_probe_id(_ADDITIVE_BASE) == _ADDITIVE_BASE


def test_every_probes_id_goes_through_the_resolver(monkeypatch):
    """Construction contract: EVERY row's served id is produced by the resolver.

    Under synthetic mode each id resolves to its twin iff it is synthetic-gated
    (in the 066/085/095 frozenset OR the additive #1360 family); a twinless id
    (e.g. model_performance_feature_drift) passes through unchanged. This fails on
    the pre-fix hand-maintained map, which never called the resolver at all.
    """
    from scripts.check_kpi_coverage import resolved_probe_id

    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "true")
    for kid, (base, _p, _k) in PROBES.items():
        resolved = resolved_probe_id(base)
        gated = base in SYNTHETIC_TWINNED_QUERY_IDS or base == _ADDITIVE_BASE
        if gated:
            assert resolved == f"{base}{_SUFFIX}", f"{kid}: {base} should resolve to its twin"
        else:
            assert resolved == base, f"{kid}: twinless {base} must pass through unchanged"


def test_twinless_feature_drift_passes_through(monkeypatch):
    """WS1-MP-009 reads model_performance_feature_drift, which is NOT synthetic-
    gated (no twin); the resolver must leave it as-is even under synthetic mode so
    the probe never asks for a non-existent twin."""
    from scripts.check_kpi_coverage import resolved_probe_id

    base = "model_performance_feature_drift"
    assert base not in SYNTHETIC_TWINNED_QUERY_IDS
    assert PROBES["WS1-MP-009"][0] == base
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "true")
    assert resolved_probe_id(base) == base
