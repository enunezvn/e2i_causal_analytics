"""#1532: WS3-BI-010 per-slice trailing-12-month temporal-variability band.

Pure unit coverage of the band-assembly helper (no DB, no mocks — real row
dicts shaped exactly like the ``business_impact_roi_temporal_band`` registry
rows). The live query + full calculator path are covered by
``tests/integration/test_roi_temporal_band_1532_live.py``.

The naming tests are the #1532 acceptance criterion made executable: the band
measures recent TEMPORAL VARIABILITY of a slice's monthly ROI — it is NOT a
confidence interval, and no payload key or wording may imply inferential
coverage (the #1526 ``sensitivity_band`` naming discipline).
"""

import json

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator

assemble = BusinessImpactCalculator._assemble_roi_temporal_band


def _row(
    metric_name="market_share",
    brand="Kisqali",
    region="northeast",
    n=12,
    roi_mean=1.5,
    roi_stddev=0.2,
    roi_min=1.1,
    roi_max=1.9,
):
    return {
        "metric_name": metric_name,
        "brand": brand,
        "region": region,
        "n": n,
        "roi_mean": roi_mean,
        "roi_stddev": roi_stddev,
        "roi_min": roi_min,
        "roi_max": roi_max,
    }


@pytest.mark.unit
def test_full_slice_carries_band_with_n():
    band = assemble([_row()])
    assert band is not None
    assert band["min_n"] == 6
    (s,) = band["slices"]
    assert s["metric_name"] == "market_share"
    assert s["brand"] == "Kisqali"
    assert s["region"] == "northeast"
    assert s["n"] == 12
    assert s["band_suppressed"] is False
    assert s["band"] == {
        "roi_min": 1.1,
        "roi_max": 1.9,
        "roi_mean": 1.5,
        "roi_stddev": 0.2,
    }


@pytest.mark.unit
@pytest.mark.parametrize("n", [1, 2, 5])
def test_young_slice_suppresses_band_but_reports_n(n):
    """#1532: slices younger than 12 months show their actual n; the band is
    suppressed below min_n — n is reported, band values are NOT fabricated."""
    band = assemble([_row(n=n, roi_stddev=None)])
    assert band is not None
    (s,) = band["slices"]
    assert s["n"] == n
    assert s["band_suppressed"] is True
    assert s["band"] is None


@pytest.mark.unit
def test_min_n_boundary_six_gets_band_five_does_not():
    band = assemble([_row(n=6), _row(brand="Fabhalta", n=5)])
    assert band is not None
    by_brand = {s["brand"]: s for s in band["slices"]}
    assert by_brand["Kisqali"]["band"] is not None
    assert by_brand["Fabhalta"]["band"] is None
    assert by_brand["Fabhalta"]["band_suppressed"] is True


@pytest.mark.unit
def test_empty_rows_mean_no_band_at_all():
    """Real-mode on an all-synthetic substrate returns zero slices (measured
    2026-08-10: business_metrics has NO is_synthetic=false rows). The band is
    then ABSENT — honest absence, never an empty shell that renders as data."""
    assert assemble([]) is None
    assert assemble(None) is None


@pytest.mark.unit
def test_band_payload_never_speaks_confidence_interval_language():
    """#1532 acceptance criterion 2, executable: the payload must be
    audit-proof — no key or wording implying inferential coverage."""
    band = assemble([_row(), _row(brand="Fabhalta", n=3, roi_stddev=None)])
    dumped = json.dumps(band).lower()
    for forbidden in ("confidence_interval", "ci_lower", "ci_upper", "95%", "confidence"):
        assert forbidden not in dumped, f"band payload contains forbidden term {forbidden!r}"


@pytest.mark.unit
def test_band_semantics_state_temporal_variability_not_uncertainty():
    """The chat synthesizer sees this payload verbatim (tool results are
    json-dumped into the synthesis prompt), so the semantics ride the data."""
    band = assemble([_row()])
    semantics = band["semantics"].lower()
    assert "12 months" in semantics
    assert "temporal variability" in semantics
    assert "not" in semantics  # ...NOT uncertainty about the current value
    assert "trailing 12 months" in band["window"]


@pytest.mark.unit
def test_null_stat_fields_suppress_rather_than_fabricate():
    """A slice at/above min_n but with NULL aggregates (defensive: should not
    happen with roi IS NOT NULL in the query) must suppress, not emit nulls
    inside a band that downstream would render as zeros."""
    band = assemble([_row(n=8, roi_min=None, roi_max=None, roi_mean=None, roi_stddev=None)])
    (s,) = band["slices"]
    assert s["band"] is None
    assert s["band_suppressed"] is True
    assert s["n"] == 8
