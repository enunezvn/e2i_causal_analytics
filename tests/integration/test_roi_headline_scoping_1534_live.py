"""#1534 faithful e2e: the brand/region-scoped WS3-BI-010 headline against the
LIVE DB — no mocks.

These assert MEANING (the #574 lesson):

- Equivalence invariant: the 2-nullable-param scoped registry query called
  with ``[None, None]`` returns the IDENTICAL value to the legacy 0-param
  headline query (same 089 frontier-anchored 30-day window, same M4 synthetic
  gating) — migration 125 adds scope, it must not move the unscoped number.
- The full ``calculate()`` path under ``{"brand": "Kisqali"}`` returns exactly
  what the scoped registry query returns for ``["Kisqali", None]`` — the
  calculator adds no arithmetic of its own.
- Region matching is case-insensitive (the 124/business_impact_trx idiom).
- An unknown brand fails LOUD through the public path (result.error set) —
  never a silently relabeled portfolio number, and never the agent_activities
  fallback (which has no brand/region dimension).

CAPABILITY-GATED: skips if SUPABASE_* unset or migration 125 isn't applied.
"""

import os

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.registry import get_registry

HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_SUPABASE, reason="SUPABASE_* not set"),
]

SCOPED_TWIN = "business_impact_roi_business_metrics_scoped_include_synthetic"
LEGACY_TWIN = "business_impact_roi_business_metrics_include_synthetic"


@pytest.fixture
def calc():
    c = BusinessImpactCalculator()
    if c.db_client is None:
        pytest.skip("no Supabase client")
    try:
        c.db_client.rpc(
            "kpi_query",
            {"query_id": "business_impact_roi_business_metrics_scoped", "params": [None, None]},
        ).execute()
    except Exception as e:
        pytest.skip(f"#1534 scoped headline query unavailable (migration 125 not applied?): {e}")
    return c


def _one(calc, query_id, params):
    resp = calc.db_client.rpc("kpi_query", {"query_id": query_id, "params": params}).execute()
    assert resp.data, f"{query_id} returned no rows"
    return resp.data[0]


def test_null_scope_is_value_identical_to_legacy_unscoped_query(calc):
    scoped = _one(calc, SCOPED_TWIN, [None, None])
    legacy = _one(calc, LEGACY_TWIN, [])
    assert scoped["avg_roi"] == pytest.approx(legacy["avg_roi"])
    assert scoped["data_through"] == legacy["data_through"]


def test_base_variants_agree_on_all_synthetic_substrate(calc):
    """Real-mode (synthetic-excluded) equivalence: on this all-synthetic
    substrate both base variants must agree — currently honest-NULL avg."""
    scoped = _one(calc, "business_impact_roi_business_metrics_scoped", [None, None])
    legacy = _one(calc, "business_impact_roi_business_metrics", [])
    assert scoped["avg_roi"] == legacy["avg_roi"]


def test_brand_scope_narrows_the_headline_and_differs_from_portfolio(calc):
    portfolio = _one(calc, SCOPED_TWIN, [None, None])
    per_brand = {
        brand: _one(calc, SCOPED_TWIN, [brand, None])["avg_roi"]
        for brand in ("Kisqali", "Fabhalta", "Remibrutinib")
    }
    for brand, value in per_brand.items():
        assert value is not None, f"{brand} has ROI rows in the window (measured 2026-08-10)"
    # The portfolio average must sit within the per-brand envelope — it is a
    # weighted mean of exactly these slices (structural, reseed-proof).
    lo, hi = min(per_brand.values()), max(per_brand.values())
    assert lo <= portfolio["avg_roi"] <= hi


def test_region_matching_is_case_insensitive(calc):
    lower = _one(calc, SCOPED_TWIN, ["Kisqali", "northeast"])
    mixed = _one(calc, SCOPED_TWIN, ["Kisqali", "NorthEast"])
    assert lower["avg_roi"] is not None
    assert lower["avg_roi"] == pytest.approx(mixed["avg_roi"])


def test_full_calculator_path_returns_the_scoped_value(calc, monkeypatch):
    """calculate() under brand context == the scoped registry value, exactly.

    E2I_KPI_INCLUDE_SYNTHETIC is a real deployment flag (SSOT showcase gate) —
    setting the env var IS the faithful showcase configuration, not a mock.
    """
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
    kpi = get_registry().get("WS3-BI-010")
    assert kpi is not None

    expected = _one(calc, SCOPED_TWIN, ["Kisqali", None])["avg_roi"]
    result = calc.calculate(kpi, context={"brand": "Kisqali"})
    assert result.error is None
    assert result.value == pytest.approx(float(expected))


def test_unknown_brand_fails_loud_through_public_path(calc, monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
    kpi = get_registry().get("WS3-BI-010")
    assert kpi is not None

    result = calc.calculate(kpi, context={"brand": "NotABrand"})
    assert result.error is not None
    assert result.value is None
