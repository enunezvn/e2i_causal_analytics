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
)

_FLAG = "E2I_KPI_INCLUDE_SYNTHETIC"
# tests/unit/test_kpi/test_synthetic_mode.py -> repo root is parents[3].
_MIGRATION_066 = (
    Path(__file__).resolve().parents[3]
    / "database/migrations/066_kpi_query_synthetic_exclusion.sql"
)


def _twin_bases_from_migration() -> set[str]:
    """Base query_ids that migration 066 registers an `_include_synthetic` twin for."""
    text = _MIGRATION_066.read_text()
    return set(re.findall(r"\('([a-z0-9_]+)_include_synthetic'", text))


# --- the env flag --------------------------------------------------------------


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert kpi_include_synthetic() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "Yes", " yes "])
def test_flag_truthy_spellings(monkeypatch, val):
    monkeypatch.setenv(_FLAG, val)
    assert kpi_include_synthetic() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "", "off", "2"])
def test_flag_falsy_spellings(monkeypatch, val):
    monkeypatch.setenv(_FLAG, val)
    assert kpi_include_synthetic() is False


# --- the resolver --------------------------------------------------------------


def test_resolver_passthrough_when_flag_off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
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


def test_resolver_passes_through_twinless_when_flag_on(monkeypatch):
    """A registry id that touches no synthetic-taggable table has no twin and is
    not synthetic-gated -> base id is returned even in demo mode (no 404 twin)."""
    monkeypatch.setenv(_FLAG, "1")
    for twinless in (
        "active_experiments_count",
        "data_quality_data_lag",
        "model_performance_feature_drift",
        "business_impact_mau_view",
    ):
        assert twinless not in SYNTHETIC_TWINNED_QUERY_IDS
        assert resolve_kpi_query_id(twinless) == twinless


def test_resolver_is_idempotent_on_a_twin_id(monkeypatch):
    """Calling the resolver on an already-resolved twin must not double-suffix."""
    monkeypatch.setenv(_FLAG, "yes")
    assert (
        resolve_kpi_query_id("business_impact_trx_include_synthetic")
        == "business_impact_trx_include_synthetic"
    )


# --- drift lock: the hard-coded set must equal migration 066's twin family -----


def test_twinned_set_matches_migration_066():
    assert SYNTHETIC_TWINNED_QUERY_IDS == _twin_bases_from_migration()


def test_migration_066_defines_expected_twin_count():
    # Guards against a partial/empty parse silently passing the equality test.
    assert len(SYNTHETIC_TWINNED_QUERY_IDS) == 36
