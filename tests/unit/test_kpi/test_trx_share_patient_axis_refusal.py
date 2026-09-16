"""TRx Share is undefined on a patient axis; the calculator refuses before any query.

Incident (chat session_1789548670222_fcscf3u, 2026-09-16): "Remibrutinib TRx Share
by severity tier and biologic status" served low 38.7% / medium 33.5% / high 29.8%
(high flagged ``warning``) and biologic-naive = biologic-experienced = 1.000.

Both were wrong by construction. Every patient in ``patient_journeys`` is on
exactly ONE tracked brand (measured live: every prescription's brand equals its
patient's brand), so "the brand's share of the portfolio's Rx among patients in
bucket B" divides a CSU brand's Rx by the Rx of breast-cancer and PNH patients who
happen to carry the same label. The high-severity figure was lowest only because
Fabhalta's high-tier patients script heavily. Biologic status and IgE tier exist
only on Remibrutinib rows, so their denominator IS the brand -- always 100%.

The meaningful per-bucket answer is the brand's own TRx by bucket (a volume axis,
whose buckets sum to the brand total). The refusal says so.
"""

from typing import Any, Dict, List

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator

_WINDOW = {"start": "S", "end": "E"}

# Every patient axis the business-impact calculator knows, with a probe value.
# therapy_line 0 is included on purpose: it is a real bucket and a truthiness
# check would let it slip through.
_AXES: List[Dict[str, Any]] = [
    {"segment": "high_severity"},
    {"therapy_line": 0},
    {"therapy_line": 3},
    {"biologic": "naive"},
    {"biologic": "experienced"},
    {"ige_tier": "high"},
]


class _NoQueries:
    """Any attribute access means a query was attempted before the refusal."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"TRx share on a patient axis must not reach the DB ({name})")


@pytest.fixture(autouse=True)
def _no_synthetic(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("axis", _AXES, ids=lambda a: "-".join(f"{k}={v}" for k, v in a.items()))
def test_trx_share_on_a_patient_axis_is_refused_before_any_query(axis, windowed):
    calc = BusinessImpactCalculator(db_client=_NoQueries())
    context: Dict[str, Any] = {"brand": "Remibrutinib", **axis}
    if windowed:
        context["window"] = dict(_WINDOW)
    with pytest.raises(RuntimeError) as exc:
        calc._calc_trx_share(context)
    msg = str(exc.value)
    assert "WS3-BI-008" in msg
    assert "one tracked brand" in msg
    assert "TRx by" in msg  # the next step: the brand's own volume by bucket


def test_refusal_names_the_tautology_for_brand_only_axes():
    calc = BusinessImpactCalculator(db_client=_NoQueries())
    with pytest.raises(RuntimeError, match="always 100%"):
        calc._calc_trx_share({"brand": "Remibrutinib", "biologic": "naive"})


def test_calculate_surfaces_the_refusal_as_an_error_not_a_value():
    """/api/kpis path: calculate() turns the raise into an error result."""
    from src.kpi.registry import get_registry

    kpi = get_registry().get("WS3-BI-008")
    assert kpi is not None
    result = BusinessImpactCalculator(db_client=_NoQueries()).calculate(
        kpi, {"brand": "Remibrutinib", "segment": "high_severity"}
    )
    assert result.value is None
    assert result.error and "one tracked brand" in result.error


def test_brand_level_share_is_untouched():
    """Positive control: no axis -> the plain frontier-anchored share still runs."""

    class _Client:
        def __init__(self):
            self.calls: List[Dict[str, Any]] = []

        def rpc(self, name, payload):
            self.calls.append(payload)
            return type(
                "E", (), {"execute": lambda s: type("R", (), {"data": [{"share": 0.33}]})()}
            )()

    client = _Client()
    calc = BusinessImpactCalculator(db_client=client)
    assert calc._calc_trx_share({"brand": "Remibrutinib"}) == 0.33
    assert client.calls[0]["query_id"] == "business_impact_trx_share"
    assert calc._calc_trx_share({"brand": "Remibrutinib", "region": "northeast"}) == 0.33
    assert client.calls[1]["query_id"] == "business_impact_trx_share_region"
