"""#1564: region variants for the brand_specific KPI family (BR-001..BR-005).

The region axis shipped in #1536/#1538 covered 3 of the 6 calculator families;
``brand_specific.py`` had zero region references, so every region+brand ask
("Kisqali oncologist reach in the northeast") answered portfolio-level with the
honest ``not_applicable`` hedge. Migration 127 adds vetted ``*_region``
registry variants for all five BR KPIs (region exists in every source:
``patient_journeys.geographic_region`` for the patient-based KPIs,
``hcp_profiles.geographic_region`` for the HCP-based ones), and the calculator
routes on ``context["region"]``, setting ``context["_region_routed"] = True``
at the exact decision point per the #1538 provenance contract.

Test instrument: a stub kpi_query client that answers PER QUERY ID — the
region-scoped id returns the REGION row, the base id the PORTFOLIO row — so a
passing test proves the calculator genuinely selected the region SQL (the two
values differ), not that a global value was relabeled.
"""

from typing import Any

import pytest

from src.kpi.calculator import KPICalculator
from src.kpi.calculators.brand_specific import BrandSpecificCalculator
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)


@pytest.fixture(autouse=True)
def _no_synthetic(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


class _Resp:
    def __init__(self, data):
        self.data = data


class _Exec:
    def __init__(self, data):
        self._d = data

    def execute(self):
        return _Resp(self._d)


class _RoutingClient:
    """kpi_query stub answering per query_id (region id -> region row, base id
    -> portfolio row). A missing id KeyErrors — routing to an unregistered
    variant fails the test instead of silently serving the wrong row."""

    def __init__(self, rows_by_query_id: dict[str, list[dict[str, Any]]]):
        self.rows = rows_by_query_id
        self.calls: list[dict[str, Any]] = []

    def rpc(self, name, payload):
        self.calls.append(payload)
        return _Exec(self.rows[payload["query_id"]])


# ---- BR-001 Remi AH Uncontrolled --------------------------------------------


@pytest.mark.unit
def test_br001_region_routes_appends_region_and_stamps():
    client = _RoutingClient(
        {
            "brand_specific_remi_ah_uncontrolled": [{"uncontrolled_rate": 0.41}],
            "brand_specific_remi_ah_uncontrolled_region": [{"uncontrolled_rate": 0.62}],
        }
    )
    calc = BrandSpecificCalculator(db_client=client)
    ctx: dict[str, Any] = {"region": "northeast"}
    value = calc._calc_remi_ah_uncontrolled(ctx)
    assert value == 0.62  # the REGION row, not the portfolio 0.41
    assert client.calls[0]["query_id"] == "brand_specific_remi_ah_uncontrolled_region"
    # Base params stay first ($1 = UAS7 threshold), region appended as $2.
    assert client.calls[0]["params"] == [7, "northeast"]
    assert ctx.get("_region_routed") is True


@pytest.mark.unit
def test_br001_no_region_base_untouched():
    client = _RoutingClient({"brand_specific_remi_ah_uncontrolled": [{"uncontrolled_rate": 0.41}]})
    calc = BrandSpecificCalculator(db_client=client)
    ctx: dict[str, Any] = {}
    value = calc._calc_remi_ah_uncontrolled(ctx)
    assert value == 0.41
    assert client.calls[0]["query_id"] == "brand_specific_remi_ah_uncontrolled"
    assert client.calls[0]["params"] == [7]
    assert ctx.get("_region_routed") is None


@pytest.mark.unit
def test_br001_region_empty_fails_loud_naming_region():
    client = _RoutingClient({"brand_specific_remi_ah_uncontrolled_region": []})
    calc = BrandSpecificCalculator(db_client=client)
    with pytest.raises(RuntimeError, match="northeast"):
        calc._calc_remi_ah_uncontrolled({"region": "northeast"})


# ---- BR-002 Remi Intent-to-Prescribe Delta (region primary -> region fallback) ----


@pytest.mark.unit
def test_br002_region_primary_leg_routes_and_stamps():
    client = _RoutingClient(
        {
            "brand_specific_remi_intent_delta_primary_region": [{"intent_delta": 0.28}],
        }
    )
    calc = BrandSpecificCalculator(db_client=client)
    ctx: dict[str, Any] = {"region": "west"}
    value = calc._calc_remi_intent_delta(ctx)
    assert value == pytest.approx(0.28)
    assert client.calls[0]["query_id"] == "brand_specific_remi_intent_delta_primary_region"
    assert client.calls[0]["params"] == ["west"]
    assert ctx.get("_region_routed") is True


@pytest.mark.unit
def test_br002_region_primary_null_falls_back_to_region_fallback():
    client = _RoutingClient(
        {
            "brand_specific_remi_intent_delta_primary_region": [{"intent_delta": None}],
            "brand_specific_remi_intent_delta_fallback_region": [{"intent_delta": -0.07}],
        }
    )
    calc = BrandSpecificCalculator(db_client=client)
    ctx: dict[str, Any] = {"region": "south"}
    value = calc._calc_remi_intent_delta(ctx)
    assert value == pytest.approx(-0.07)
    assert [c["query_id"] for c in client.calls] == [
        "brand_specific_remi_intent_delta_primary_region",
        "brand_specific_remi_intent_delta_fallback_region",
    ]
    assert all(c["params"] == ["south"] for c in client.calls)
    assert ctx.get("_region_routed") is True


@pytest.mark.unit
def test_br002_region_both_legs_empty_fails_loud_naming_region():
    client = _RoutingClient(
        {
            "brand_specific_remi_intent_delta_primary_region": [],
            "brand_specific_remi_intent_delta_fallback_region": [],
        }
    )
    calc = BrandSpecificCalculator(db_client=client)
    with pytest.raises(RuntimeError, match="south"):
        calc._calc_remi_intent_delta({"region": "south"})


@pytest.mark.unit
def test_br002_no_region_chain_untouched():
    """region=None keeps the certified primary(view) -> fallback chain, no marker."""
    client = _RoutingClient({"brand_specific_remi_intent_delta_primary": [{"intent_delta": 0.32}]})
    calc = BrandSpecificCalculator(db_client=client)
    ctx: dict[str, Any] = {"brand": "Remibrutinib"}
    assert calc._calc_remi_intent_delta(ctx) == pytest.approx(0.32)
    assert client.calls[0]["query_id"] == "brand_specific_remi_intent_delta_primary"
    assert ctx.get("_region_routed") is None


# ---- BR-003 Fabhalta PNH Tested ---------------------------------------------


@pytest.mark.unit
def test_br003_region_routes_and_value_differs_from_portfolio():
    client = _RoutingClient(
        {
            "brand_specific_fabhalta_pnh_tested": [{"tested_rate": 0.55, "pnh_events_total": 40}],
            "brand_specific_fabhalta_pnh_tested_region": [
                {"tested_rate": 0.71, "pnh_events_total": 40}
            ],
        }
    )
    calc = BrandSpecificCalculator(db_client=client)
    ctx: dict[str, Any] = {"region": "midwest"}
    value = calc._calc_fabhalta_pnh_tested(ctx)
    assert value == 0.71
    assert client.calls[0]["query_id"] == "brand_specific_fabhalta_pnh_tested_region"
    assert client.calls[0]["params"] == ["midwest"]
    assert ctx.get("_region_routed") is True


@pytest.mark.unit
def test_br003_region_structural_zero_guard_still_raises():
    """#1116 guard survives the region path: pnh_events_total stays TABLE-WIDE
    in the region variant (substrate coverage is not a per-region fact), so a
    region cohort with zero events anywhere still fails loud."""
    client = _RoutingClient(
        {
            "brand_specific_fabhalta_pnh_tested_region": [
                {"tested_rate": 0.0, "pnh_events_total": 0}
            ],
        }
    )
    calc = BrandSpecificCalculator(db_client=client)
    with pytest.raises(RuntimeError, match="structurally-zero"):
        calc._calc_fabhalta_pnh_tested({"region": "midwest"})


@pytest.mark.unit
def test_br003_no_region_base_untouched():
    client = _RoutingClient(
        {"brand_specific_fabhalta_pnh_tested": [{"tested_rate": 0.55, "pnh_events_total": 40}]}
    )
    calc = BrandSpecificCalculator(db_client=client)
    ctx: dict[str, Any] = {}
    assert calc._calc_fabhalta_pnh_tested(ctx) == 0.55
    assert client.calls[0]["query_id"] == "brand_specific_fabhalta_pnh_tested"
    assert client.calls[0]["params"] == []
    assert ctx.get("_region_routed") is None


# ---- BR-004 Kisqali Dx Adoption ---------------------------------------------


@pytest.mark.unit
def test_br004_region_routes_and_value_differs_from_portfolio():
    client = _RoutingClient(
        {
            "brand_specific_kisqali_dx_adoption": [{"median_days": 42.0}],
            "brand_specific_kisqali_dx_adoption_region": [{"median_days": 33.0}],
        }
    )
    calc = BrandSpecificCalculator(db_client=client)
    ctx: dict[str, Any] = {"region": "northeast"}
    value = calc._calc_kisqali_dx_adoption(ctx)
    assert value == 33.0
    assert client.calls[0]["query_id"] == "brand_specific_kisqali_dx_adoption_region"
    assert client.calls[0]["params"] == ["northeast"]
    assert ctx.get("_region_routed") is True


@pytest.mark.unit
def test_br004_no_region_base_untouched():
    client = _RoutingClient({"brand_specific_kisqali_dx_adoption": [{"median_days": 42.0}]})
    calc = BrandSpecificCalculator(db_client=client)
    ctx: dict[str, Any] = {}
    assert calc._calc_kisqali_dx_adoption(ctx) == 42.0
    assert client.calls[0]["params"] == []
    assert ctx.get("_region_routed") is None


# ---- BR-005 Kisqali Oncologist Reach ----------------------------------------


@pytest.mark.unit
def test_br005_region_routes_and_value_differs_from_portfolio():
    client = _RoutingClient(
        {
            "brand_specific_kisqali_oncologist_reach": [{"reach": 0.44}],
            "brand_specific_kisqali_oncologist_reach_region": [{"reach": 0.58}],
        }
    )
    calc = BrandSpecificCalculator(db_client=client)
    ctx: dict[str, Any] = {"region": "west"}
    value = calc._calc_kisqali_oncologist_reach(ctx)
    assert value == 0.58
    assert client.calls[0]["query_id"] == "brand_specific_kisqali_oncologist_reach_region"
    assert client.calls[0]["params"] == ["west"]
    assert ctx.get("_region_routed") is True


@pytest.mark.unit
def test_br005_no_region_base_untouched():
    client = _RoutingClient({"brand_specific_kisqali_oncologist_reach": [{"reach": 0.44}]})
    calc = BrandSpecificCalculator(db_client=client)
    ctx: dict[str, Any] = {}
    assert calc._calc_kisqali_oncologist_reach(ctx) == 0.44
    assert client.calls[0]["params"] == []
    assert ctx.get("_region_routed") is None


# ---- synthetic-mode twin honor ----------------------------------------------


@pytest.mark.unit
def test_region_id_honors_synthetic_showcase_flag(monkeypatch):
    """Under E2I_KPI_INCLUDE_SYNTHETIC the region read swaps to the
    ``*_region_include_synthetic`` twin (region_query_id self-suffixes; the
    _execute_query resolve pass is a no-op on the already-suffixed id)."""
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
    client = _RoutingClient(
        {
            "brand_specific_kisqali_oncologist_reach_region_include_synthetic": [{"reach": 0.51}],
        }
    )
    calc = BrandSpecificCalculator(db_client=client)
    ctx: dict[str, Any] = {"region": "south"}
    assert calc._calc_kisqali_oncologist_reach(ctx) == 0.51
    assert (
        client.calls[0]["query_id"]
        == "brand_specific_kisqali_oncologist_reach_region_include_synthetic"
    )
    assert ctx.get("_region_routed") is True


# ---- end-to-end: KPICalculator stamps region provenance ---------------------


class _AlwaysMissCache:
    enabled = True

    def get(self, kpi_id, **context):
        return None

    def set(self, result, ttl=None, **context):
        return True


class _StubRegistry:
    def __init__(self, kpi: KPIMetadata):
        self._kpi = kpi

    def get(self, kpi_id: str) -> KPIMetadata | None:
        return self._kpi if kpi_id == self._kpi.id else None


def _br_kpi(kpi_id: str, name: str) -> KPIMetadata:
    return KPIMetadata(
        id=kpi_id,
        name=name,
        definition="d",
        formula="f",
        calculation_type=CalculationType.DERIVED,
        workstream=Workstream.BRAND_SPECIFIC,
    )


def _wired(kpi: KPIMetadata, client) -> KPICalculator:
    calc = KPICalculator(registry=_StubRegistry(kpi), cache=_AlwaysMissCache())
    calc.register_calculator(Workstream.BRAND_SPECIFIC, BrandSpecificCalculator(db_client=client))
    return calc


@pytest.mark.unit
def test_e2e_br005_region_ask_stamps_applied_with_region_value():
    kpi = _br_kpi("BR-005", "Kisqali - Oncologist Reach")
    client = _RoutingClient(
        {
            "brand_specific_kisqali_oncologist_reach": [{"reach": 0.44}],
            "brand_specific_kisqali_oncologist_reach_region": [{"reach": 0.58}],
        }
    )
    res = _wired(kpi, client).calculate("BR-005", context={"region": "west"})
    assert res.error is None
    assert res.value == 0.58  # region-filtered value, differs from portfolio 0.44
    assert res.region_status == "applied"
    assert res.region_requested == "west"
    assert res.region_applied == "west"


@pytest.mark.unit
def test_e2e_br005_no_region_stays_default():
    kpi = _br_kpi("BR-005", "Kisqali - Oncologist Reach")
    client = _RoutingClient({"brand_specific_kisqali_oncologist_reach": [{"reach": 0.44}]})
    res = _wired(kpi, client).calculate("BR-005")
    assert res.value == 0.44
    assert res.region_status == "default"
    assert res.region_requested is None


@pytest.mark.unit
def test_e2e_br001_region_ask_stamps_applied():
    kpi = _br_kpi("BR-001", "Remi - AH Uncontrolled %")
    client = _RoutingClient(
        {
            "brand_specific_remi_ah_uncontrolled": [{"uncontrolled_rate": 0.41}],
            "brand_specific_remi_ah_uncontrolled_region": [{"uncontrolled_rate": 0.62}],
        }
    )
    res = _wired(kpi, client).calculate("BR-001", context={"region": "northeast"})
    assert res.value == 0.62
    assert res.region_status == "applied"


@pytest.mark.unit
def test_e2e_region_error_still_not_applicable_provenance():
    """A region-scoped read that fails loud surfaces error-first; the stamp
    echoes the requested region (consumers read ``error`` before provenance)."""
    kpi = _br_kpi("BR-004", "Kisqali - Dx Adoption")
    client = _RoutingClient({"brand_specific_kisqali_dx_adoption_region": []})
    res = _wired(kpi, client).calculate("BR-004", context={"region": "south"})
    assert res.error is not None and "south" in res.error
    assert res.value is None
    assert res.region_requested == "south"


# ---- unchanged families keep the honest hedge -------------------------------


@pytest.mark.unit
def test_unrouted_family_still_not_applicable():
    """causal_metrics / model_performance ship no region variants in #1564 —
    a region ask against them must keep the truthful not_applicable hedge
    (the #1538 stamp handles it because no seam sets the marker)."""
    r = KPIResult(kpi_id="WS1-MP-001", value=0.87, status=KPIStatus.GOOD)
    out = KPICalculator._stamp_region(r, {"region": "northeast"})
    assert out.region_status == "not_applicable"
    assert out.value == 0.87
