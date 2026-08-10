"""#1532 faithful e2e: the WS3-BI-010 temporal-variability band computes REAL
per-slice trailing-12-month ROI ranges against the LIVE DB — no mocks.

These assert MEANING (the #574 lesson):
- The ``business_impact_roi_temporal_band_include_synthetic`` registry query
  returns per-(metric_name, brand, region) slices with 1 <= n <= 12 (monthly
  data, 12-month window — n can never exceed 12) and internally-consistent
  aggregates (roi_min <= roi_mean <= roi_max).
- The base (synthetic-excluded) variant is HONEST-EMPTY on an all-synthetic
  substrate (measured 2026-08-10: business_metrics has zero
  is_synthetic=false rows) — a shape assertion, never a fabricated band.
- The full calculator path stashes the band into KPIResult metadata with the
  #1532 suppression contract (n >= 6), and the band always covers the SAME
  population as the headline (codex iter-2 invariant): since #1534/migration
  125 the headline honors brand/region scope, so the band follows it — an
  unscoped request keeps the portfolio-wide band.
- #1527 regression: nothing in the calculated ROI result speaks
  confidence-interval language.

CAPABILITY-GATED: skips if SUPABASE_* unset or migration 124 isn't applied.
"""

import json
import os

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.registry import get_registry

HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_SUPABASE, reason="SUPABASE_* not set"),
]

_BAND_COLUMNS = {
    "metric_name",
    "brand",
    "region",
    "n",
    "roi_mean",
    "roi_stddev",
    "roi_min",
    "roi_max",
    "data_through",
}


@pytest.fixture
def calc():
    c = BusinessImpactCalculator()
    if c.db_client is None:
        pytest.skip("no Supabase client")
    try:
        c.db_client.rpc(
            "kpi_query",
            {"query_id": "business_impact_roi_temporal_band", "params": [None, None]},
        ).execute()
    except Exception as e:
        pytest.skip(f"#1532 band query unavailable (migration 124 not applied?): {e}")
    return c


def _run(calc, query_id, params):
    resp = calc.db_client.rpc("kpi_query", {"query_id": query_id, "params": params}).execute()
    return resp.data


def test_band_twin_returns_real_monthly_slices(calc):
    rows = _run(calc, "business_impact_roi_temporal_band_include_synthetic", [None, None])
    assert isinstance(rows, list) and len(rows) > 0, "synthetic-gold substrate must have slices"
    for row in rows:
        assert _BAND_COLUMNS.issubset(row.keys())
        n = int(row["n"])
        # Monthly substrate: a 12-month window holds at most 12 observations
        # per slice — n > 12 would mean the GROUP BY is pooling rows the #1527
        # analysis says must never pool.
        assert 1 <= n <= 12, f"slice n={n} outside monthly-window bounds"
        if row["roi_min"] is not None and row["roi_max"] is not None:
            lo, hi = float(row["roi_min"]), float(row["roi_max"])
            mean = float(row["roi_mean"])
            assert lo <= mean <= hi, f"inconsistent aggregates: {lo}/{mean}/{hi}"
        assert row["data_through"] is not None


def test_band_base_variant_is_honest_empty_on_all_synthetic_substrate(calc):
    rows = _run(calc, "business_impact_roi_temporal_band", [None, None])
    assert isinstance(rows, list)
    # This prod substrate is 100% synthetic (measured 2026-08-10); if real rows
    # ever appear the shape contract still holds — never a hard count.
    for row in rows:
        assert _BAND_COLUMNS.issubset(row.keys())


def test_band_brand_filter_narrows_slices(calc):
    all_rows = _run(calc, "business_impact_roi_temporal_band_include_synthetic", [None, None])
    kisqali = _run(calc, "business_impact_roi_temporal_band_include_synthetic", ["Kisqali", None])
    assert len(kisqali) > 0
    assert len(kisqali) < len(all_rows)
    assert {row["brand"] for row in kisqali} == {"Kisqali"}


def test_full_calculator_path_stashes_band_with_suppression_contract(calc, monkeypatch):
    """Real calculate() against the live DB: headline value + band metadata.

    E2I_KPI_INCLUDE_SYNTHETIC is a real deployment flag (SSOT showcase gate) —
    setting the env var IS the faithful showcase configuration, not a mock.
    """
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
    kpi = get_registry().get("WS3-BI-010")
    assert kpi is not None

    result = calc.calculate(kpi, context={})
    assert result.error is None
    assert isinstance(result.value, float)

    band = (result.metadata.get("context") or {}).get("temporal_variability_band")
    assert band is not None, "showcase mode with 12 months of data must carry the band"
    assert band["min_n"] == 6
    assert len(band["slices"]) > 0
    for s in band["slices"]:
        assert isinstance(s["n"], int) and s["n"] >= 1
        if s["band_suppressed"]:
            assert s["band"] is None
        else:
            assert s["n"] >= band["min_n"]
            assert s["band"]["roi_min"] <= s["band"]["roi_mean"] <= s["band"]["roi_max"]


def test_full_calculator_path_band_matches_headline_population(calc, monkeypatch):
    """Codex iter-2 invariant: band and headline must describe the same
    population. Since #1534 (migration 125) the headline honors brand scope,
    so a Kisqali request now yields a Kisqali-scoped headline AND a band whose
    slices are Kisqali-only — while an unscoped request keeps the full
    portfolio band (both directions exercised against the live DB)."""
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
    kpi = get_registry().get("WS3-BI-010")
    assert kpi is not None

    result = calc.calculate(kpi, context={"brand": "Kisqali"})
    assert result.error is None
    band = (result.metadata.get("context") or {}).get("temporal_variability_band")
    assert band is not None
    assert {s["brand"] for s in band["slices"]} == {"Kisqali"}

    unscoped = calc.calculate(kpi, context={})
    assert unscoped.error is None
    full_band = (unscoped.metadata.get("context") or {}).get("temporal_variability_band")
    assert full_band is not None
    # The substrate has multiple brands (test_band_brand_filter_narrows_slices
    # measures kisqali < all): the unscoped band must still show more than one.
    assert len({s["brand"] for s in full_band["slices"]}) > 1


def test_calculated_roi_result_speaks_no_interval_language_regression_1527(calc, monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
    kpi = get_registry().get("WS3-BI-010")
    assert kpi is not None

    result = calc.calculate(kpi, context={})
    assert result.error is None
    band = (result.metadata.get("context") or {}).get("temporal_variability_band")
    dumped = json.dumps(band).lower()
    for forbidden in ("confidence_interval", "ci_lower", "ci_upper", "95%", "confidence"):
        assert forbidden not in dumped, f"band payload contains forbidden term {forbidden!r}"


def test_shared_context_batch_does_not_leak_band_across_kpis_live(calc, monkeypatch):
    """Codex iter-1 finding 1, run against the LIVE DB (no mocks): two ROI-
    family calculations sharing one caller context — the band must appear only
    on the WS3-BI-010 result's metadata, the other KPI's metadata must not
    inherit it, and the caller's own dict must stay unmutated (it previously
    received the band BY REFERENCE and leaked it into every later result and
    cache key).

    The shared context must be NON-empty — ``calculate()``'s ``context or {}``
    accidentally isolates an empty dict, so the faithful repro is the
    insights_strategic shape: a brand-scoped batch context."""
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
    registry = get_registry()
    roi_kpi = registry.get("WS3-BI-010")
    trx_kpi = registry.get("WS3-BI-005")
    assert roi_kpi is not None and trx_kpi is not None

    from src.kpi.calculator import KPICalculator

    engine = KPICalculator(registry=registry)
    engine.register_calculator(roi_kpi.workstream, calc)

    shared_context: dict = {"brand": "Kisqali"}
    batch = engine.calculate_batch(
        kpi_ids=["WS3-BI-010", "WS3-BI-005"], use_cache=False, context=shared_context
    )
    by_id = {r.kpi_id: r for r in batch.results}
    roi_result, trx_result = by_id["WS3-BI-010"], by_id["WS3-BI-005"]
    assert roi_result.error is None

    roi_ctx = roi_result.metadata.get("context") or {}
    trx_ctx = trx_result.metadata.get("context") or {}
    assert roi_ctx.get("temporal_variability_band") is not None
    assert "temporal_variability_band" not in trx_ctx
    assert "temporal_variability_band" not in shared_context
