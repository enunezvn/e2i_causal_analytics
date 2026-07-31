"""Unit tests for the KPI synthetic-visibility mode (demo / review).

Covers the env flag, the query_id resolver (twin swap / pass-through /
idempotence), and a DRIFT lock that parses migration 066 and asserts the
hard-coded twin set equals the migration's ``*_include_synthetic`` family — so a
twin added by a later migration cannot silently desync the resolver.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.kpi.synthetic_mode import (
    SYNTHETIC_TWINNED_QUERY_IDS,
    kpi_include_synthetic,
    resolve_kpi_query_id,
    trigger_effectiveness_query_id,
)

_FLAG = "E2I_KPI_INCLUDE_SYNTHETIC"
# tests/unit/test_kpi/test_synthetic_mode.py -> repo root is parents[3].
_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "database/migrations"
# Migrations that register `*_include_synthetic` twins. 066 is the original M-family
# bulk; 085 adds the view-backed WS3-BI-003 patient_touch_rate twin (#1064 — the
# touch-rate KPI reads a view, so it was absent from 066's table-wrapping pass);
# 095 adds the four view-backed WS1 data-quality twins (cross_source_match /
# stacking_lift / data_lag / time_to_release — the same 066 gap as #1064).
_TWIN_MIGRATIONS = (
    _MIGRATIONS_DIR / "066_kpi_query_synthetic_exclusion.sql",
    _MIGRATIONS_DIR / "085_kpi_patient_touch_rate_include_synthetic.sql",
    _MIGRATIONS_DIR / "095_kpi_dq_view_include_synthetic_twins.sql",
)


def _twin_bases_from_migration() -> set[str]:
    """Base query_ids that the twin-registering migrations expose an `_include_synthetic` twin for."""
    bases: set[str] = set()
    for path in _TWIN_MIGRATIONS:
        bases |= set(re.findall(r"\('([a-z0-9_]+)_include_synthetic'", path.read_text()))
    return bases


# --- the env flag --------------------------------------------------------------


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    assert kpi_include_synthetic() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "Yes", " yes "])
def test_flag_truthy_spellings(monkeypatch, val):
    monkeypatch.setenv(_FLAG, val)
    assert kpi_include_synthetic() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "", "off", "2"])
def test_flag_falsy_spellings(monkeypatch, val):
    monkeypatch.setenv(_FLAG, val)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    assert kpi_include_synthetic() is False


def test_unified_deployment_flag_flips_kpi_reads(monkeypatch):
    """WS-SYNTH: the deployment-wide ``E2I_INCLUDE_SYNTHETIC`` showcase switch
    flips KPI reads too (ONE env for the whole synthetic-gold instance), even
    with the KPI-specific flag off. Generalizes the demo-mode idiom."""
    monkeypatch.delenv(_FLAG, raising=False)
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    assert kpi_include_synthetic() is True


# --- the resolver --------------------------------------------------------------


def test_resolver_passthrough_when_flag_off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    # Hermetic: the deployment-wide showcase switch also flips the resolver, and
    # it IS set on the synthetic-gold droplet — delete it so this test asserts
    # the flag-off gate everywhere, not just in CI's unset environment.
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    # Even a twinned id is untouched while the flag is off (production gate intact).
    assert resolve_kpi_query_id("business_impact_trx") == "business_impact_trx"
    assert resolve_kpi_query_id("model_performance_roc_auc") == "model_performance_roc_auc"


def test_resolver_swaps_twinned_when_flag_on(monkeypatch):
    monkeypatch.setenv(_FLAG, "true")
    assert resolve_kpi_query_id("business_impact_trx") == "business_impact_trx_include_synthetic"
    assert (
        resolve_kpi_query_id("model_performance_roc_auc")
        == "model_performance_roc_auc_include_synthetic"
    )


def test_resolver_swaps_patient_touch_rate_when_flag_on(monkeypatch):
    """#1064: WS3-BI-003 patient_touch_rate is view-backed (v_patient_eligibility,
    which migration 067 made synthetic-excluding) and was absent from the 066
    table-wrapping pass. Its twin (migration 085) must be reachable so the demo
    cohort (100% is_synthetic) computes instead of returning null."""
    monkeypatch.setenv(_FLAG, "true")
    assert (
        resolve_kpi_query_id("business_impact_patient_touch_rate")
        == "business_impact_patient_touch_rate_include_synthetic"
    )


def test_resolver_passes_through_twinless_when_flag_on(monkeypatch):
    """A registry id that touches no synthetic-taggable table has no twin and is
    not synthetic-gated -> base id is returned even in demo mode (no 404 twin).
    (data_quality_data_lag left this list when migration 095 twinned it.)"""
    monkeypatch.setenv(_FLAG, "1")
    for twinless in (
        "active_experiments_count",
        "data_quality_label_quality",
        "model_performance_feature_drift",
        "business_impact_mau_view",
    ):
        assert twinless not in SYNTHETIC_TWINNED_QUERY_IDS
        assert resolve_kpi_query_id(twinless) == twinless


def test_resolver_swaps_dq_view_backed_ids_when_flag_on(monkeypatch):
    """Migration 095: the four view-backed WS1 data-quality KPIs read views that
    migration 067 made synthetic-excluding and were absent from the 066
    table-wrapping pass (same gap as #1064's patient_touch_rate). Their twins
    must be reachable so a synthetic-gold instance computes real values instead
    of rendering /data-quality rows as "No data"."""
    monkeypatch.setenv(_FLAG, "true")
    for base in (
        "data_quality_cross_source_match",
        "data_quality_stacking_lift",
        "data_quality_data_lag",
        "data_quality_time_to_release",
    ):
        assert resolve_kpi_query_id(base) == f"{base}_include_synthetic"


def test_resolver_is_idempotent_on_a_twin_id(monkeypatch):
    """Calling the resolver on an already-resolved twin must not double-suffix."""
    monkeypatch.setenv(_FLAG, "yes")
    assert (
        resolve_kpi_query_id("business_impact_trx_include_synthetic")
        == "business_impact_trx_include_synthetic"
    )


# --- drift lock: the hard-coded set must equal migration 066's twin family -----


# --- trigger_effectiveness_query_id regioned+windowed variant (#1388) ---------


def test_trigger_effectiveness_id_windowed_regioned(monkeypatch):
    """#1388: windowed=True + regioned=True -> the migration-120
    _windowed_region id (region can co-bind now the RPC binds 6 params)."""
    monkeypatch.delenv(_FLAG, raising=False)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    assert (
        trigger_effectiveness_query_id("precision", windowed=True, regioned=True)
        == "trigger_effectiveness_precision_windowed_region"
    )


def test_trigger_effectiveness_id_windowed_regioned_self_suffixes(monkeypatch):
    """The _windowed_region id is additive (absent from the twinned set), so it
    self-suffixes _include_synthetic under the showcase flag."""
    monkeypatch.setenv(_FLAG, "1")
    assert (
        trigger_effectiveness_query_id("funnel_conversion", windowed=True, regioned=True)
        == "trigger_effectiveness_funnel_conversion_windowed_region_include_synthetic"
    )
    assert (
        "trigger_effectiveness_funnel_conversion_windowed_region" not in SYNTHETIC_TWINNED_QUERY_IDS
    )


def test_trigger_effectiveness_id_regioned_requires_windowed(monkeypatch):
    """regioned=True without windowed=True is a programming error: the
    non-windowed form already binds region as a nullable param (no id suffix)."""
    monkeypatch.delenv(_FLAG, raising=False)
    with pytest.raises(ValueError, match="windowed"):
        trigger_effectiveness_query_id("precision", windowed=False, regioned=True)


def test_twinned_set_matches_migrations():
    parsed = _twin_bases_from_migration()
    # Non-vacuous guard on the PARSED set first: a partial/empty regex match
    # would make the equality below pass for the wrong reason, so assert the
    # migrations actually yielded the expected twin count before comparing.
    # 36 from 066 + 1 (business_impact_patient_touch_rate) from 085 + 4
    # (the view-backed WS1 data-quality ids) from 095 = 41.
    assert len(parsed) == 41, f"twin-migration parse found {len(parsed)} twins, expected 41"
    assert parsed == SYNTHETIC_TWINNED_QUERY_IDS
