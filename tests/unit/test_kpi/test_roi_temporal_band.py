"""#1532: WS3-BI-010 per-slice trailing-12-month temporal-variability band.

Band-assembly coverage is pure (no DB, no mocks — real row dicts shaped
exactly like the ``business_impact_roi_temporal_band`` registry rows); the
context-isolation section below uses the ``test_calculator.py`` double
pattern to pin the KPICalculator seam. The live query + full calculator path
are covered by ``tests/integration/test_roi_temporal_band_1532_live.py``.

The naming tests are the #1532 acceptance criterion made executable: the band
measures recent TEMPORAL VARIABILITY of a slice's monthly ROI — it is NOT a
confidence interval, and no payload key or wording may imply inferential
coverage (the #1526 ``sensitivity_band`` naming discipline).
"""

import json
from unittest.mock import MagicMock, Mock

import pytest

from src.kpi.calculator import KPICalculator, KPICalculatorBase
from src.kpi.calculators.business_impact import BusinessImpactCalculator
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)
from src.kpi.registry import KPIRegistry

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


# ---------------------------------------------------------------------------
# Context isolation (codex iter-1 HIGH findings, 2026-08-10, made executable):
# calculate_batch passes ONE context dict to every KPI, and calculators embed
# it in metadata BY REFERENCE — so ROI's stashed band leaked into every other
# KPI's result metadata (even results created BEFORE ROI ran) and into their
# cache keys. Each calculate() call must own a private copy, and a stale band
# must never survive a reused context.
# ---------------------------------------------------------------------------


class _RoiLikeStashingCalculator(KPICalculatorBase):
    """Mimics the real seam: stashes the band into context ONLY for ROI."""

    def calculate(self, kpi: KPIMetadata, context: dict | None = None) -> KPIResult:
        context = context if context is not None else {}
        if kpi.id == "WS3-BI-010":
            context["temporal_variability_band"] = {"min_n": 6, "slices": []}
        return KPIResult(
            kpi_id=kpi.id,
            value=1.0,
            status=KPIStatus.INFORMATIONAL,
            metadata={"context": context},
        )

    def supports(self, kpi: KPIMetadata) -> bool:
        return True


def _ws3_kpi(kpi_id: str) -> KPIMetadata:
    return KPIMetadata(
        id=kpi_id,
        name=kpi_id,
        definition="test",
        formula="test",
        calculation_type=CalculationType.DERIVED,
        workstream=Workstream.WS3_BUSINESS,
        threshold=None,
    )


@pytest.fixture
def isolated_calculator():
    registry = Mock(spec=KPIRegistry)
    registry.get.side_effect = _ws3_kpi
    cache = Mock()
    cache.enabled = False
    cache.get.return_value = None
    calc = KPICalculator(registry=registry, cache=cache, router=Mock())
    calc.register_calculator(Workstream.WS3_BUSINESS, _RoiLikeStashingCalculator())
    return calc


@pytest.mark.unit
def test_batch_shared_context_does_not_leak_band_into_other_kpis(isolated_calculator):
    """POST /api/kpis/batch with ROI + TRx: the TRx result's serialized
    metadata must not carry the ROI band (codex iter-1 finding 1).

    The context must be NON-empty: ``calculate()``'s ``context or {}``
    accidentally isolates an empty dict (falsy), so the leak only manifests
    on the real callers' shape — e.g. insights_strategic passes
    ``{"brand": ...}`` when a brand is selected."""
    batch = isolated_calculator.calculate_batch(
        kpi_ids=["WS3-BI-010", "WS3-BI-005"], use_cache=False, context={"brand": "Kisqali"}
    )
    by_id = {r.kpi_id: r for r in batch.results}
    assert "temporal_variability_band" in by_id["WS3-BI-010"].metadata["context"]
    assert "temporal_variability_band" not in by_id["WS3-BI-005"].metadata["context"]


@pytest.mark.unit
def test_calculate_does_not_mutate_callers_context(isolated_calculator):
    caller_ctx = {"brand": "Kisqali"}
    result = isolated_calculator.calculate("WS3-BI-010", use_cache=False, context=caller_ctx)
    assert caller_ctx == {"brand": "Kisqali"}, "caller's dict must stay untouched"
    embedded = result.metadata["context"]
    assert embedded["temporal_variability_band"] is not None
    assert embedded["brand"] == "Kisqali", "caller values must still flow through"


@pytest.mark.unit
def test_earlier_result_metadata_not_rewritten_by_later_calculation(isolated_calculator):
    """By-reference embedding made later mutations retroactively visible in
    results created earlier; each result must own its context snapshot.
    (Non-empty shared context: an empty dict is accidentally isolated by
    ``context or {}``.)"""
    shared: dict = {"region": "northeast"}
    roi = isolated_calculator.calculate("WS3-BI-010", use_cache=False, context=shared)
    trx = isolated_calculator.calculate("WS3-BI-005", use_cache=False, context=shared)
    assert "temporal_variability_band" in roi.metadata["context"]
    assert "temporal_variability_band" not in trx.metadata["context"]


@pytest.mark.unit
def test_stale_band_cleared_when_band_query_fails():
    """A reused context must never carry a previous calculation's band past a
    failed band query — stale range beside a fresh headline is the same
    plausible-but-wrong shape #1527 rejected (codex iter-1 finding 1b)."""
    client = MagicMock()
    client.rpc.return_value.execute.side_effect = RuntimeError("rpc down")
    calc = BusinessImpactCalculator(db_client=client)
    context = {"temporal_variability_band": {"min_n": 6, "slices": [{"stale": True}]}}
    calc._stash_roi_temporal_band(context)
    assert "temporal_variability_band" not in context


@pytest.mark.unit
def test_stale_band_cleared_when_band_query_returns_empty():
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=[])
    calc = BusinessImpactCalculator(db_client=client)
    context = {"temporal_variability_band": {"min_n": 6, "slices": [{"stale": True}]}}
    calc._stash_roi_temporal_band(context)
    assert "temporal_variability_band" not in context
