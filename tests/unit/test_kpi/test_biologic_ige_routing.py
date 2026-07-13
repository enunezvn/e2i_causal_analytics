"""_resolve_windowed_call routing + brand-gate guard for the biologic / IgE-tier
axes (migration 108), plus the VOLUME calculators threading them.

Mirrors test_segment_line_windowed_call.py and test_segment_line_volume_routing.py
one axis level up. The biologic/ige_tier axes are brand-gated: the columns are
real ONLY for _BIOLOGIC_AXIS_BRANDS (Remibrutinib), so a request for any other
brand (or no brand) must FAIL CLOSED with an explicit RuntimeError -- never a
silent 0 that reads as a fabricated split.
"""

import pytest

from src.kpi.calculators.business_impact import (
    _BIOLOGIC_AXIS_BRANDS,
    BusinessImpactCalculator,
)
from src.ml.synthetic.clinical_codes import BRAND_ELIGIBILITY_FIELDS


def _calc():
    return BusinessImpactCalculator(db_client=None)  # helper is pure; no DB needed


@pytest.fixture(autouse=True)
def _no_synthetic(monkeypatch):
    """Pin synthetic flags off so the resolved ids are the bare (non-twin) forms."""
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "0")
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "0")


_WINDOW = {"start": "2025-01-01T00:00:00+00:00", "end": "2025-04-01T00:00:00+00:00"}


# --- SSOT consistency ------------------------------------------------------


def test_biologic_axis_brands_match_dgp_ssot():
    """_BIOLOGIC_AXIS_BRANDS (serving) must equal the DGP SSOT set of brands whose
    eligibility fields include the biologic columns -- drift-locked, mirroring
    causal._BRAND_CLINICAL_COVARIATES <-> BRAND_ELIGIBILITY_FIELDS."""
    by_bio = {b for b, f in BRAND_ELIGIBILITY_FIELDS.items() if "biologic_experienced" in f}
    by_ige = {b for b, f in BRAND_ELIGIBILITY_FIELDS.items() if "ige_level" in f}
    assert _BIOLOGIC_AXIS_BRANDS == by_bio
    assert _BIOLOGIC_AXIS_BRANDS == by_ige  # both columns ride the same brands


# --- biologic routing ------------------------------------------------------


def test_biologic_base():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_nrx",
        brand="Remibrutinib",
        region=None,
        window=None,
        biologic="experienced",
    )
    assert qid == "business_impact_nrx_biologic"
    assert params == ["Remibrutinib", "experienced"]


def test_biologic_windowed():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_nrx",
        brand="Remibrutinib",
        region=None,
        window=_WINDOW,
        biologic="naive",
    )
    assert qid == "business_impact_nrx_biologic_windowed"
    assert params == ["Remibrutinib", "naive", _WINDOW["start"], _WINDOW["end"]]


# --- ige_tier routing ------------------------------------------------------


def test_ige_tier_base():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx", brand="Remibrutinib", region=None, window=None, ige_tier="high"
    )
    assert qid == "business_impact_trx_ige_tier"
    assert params == ["Remibrutinib", "high"]


def test_ige_tier_windowed():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx",
        brand="Remibrutinib",
        region=None,
        window=_WINDOW,
        ige_tier="low",
    )
    assert qid == "business_impact_trx_ige_tier_windowed"
    assert params == ["Remibrutinib", "low", _WINDOW["start"], _WINDOW["end"]]


# --- brand-gate: fail closed for non-eligible brands -----------------------


@pytest.mark.parametrize("brand", ["Kisqali", "Fabhalta"])
def test_biologic_fails_closed_for_offbrand(brand):
    with pytest.raises(RuntimeError, match="not available for"):
        _calc()._resolve_windowed_call(
            "business_impact_nrx", brand=brand, region=None, window=None, biologic="experienced"
        )


@pytest.mark.parametrize("brand", ["Kisqali", "Fabhalta"])
def test_ige_tier_fails_closed_for_offbrand(brand):
    with pytest.raises(RuntimeError, match="not available for"):
        _calc()._resolve_windowed_call(
            "business_impact_trx", brand=brand, region=None, window=None, ige_tier="high"
        )


def test_biologic_requires_brand():
    with pytest.raises(RuntimeError, match="requires a brand"):
        _calc()._resolve_windowed_call(
            "business_impact_nrx", brand=None, region=None, window=None, biologic="naive"
        )


def test_biologic_guard_is_case_insensitive_membership():
    """A lower-cased eligible brand passes the guard (the SQL brand predicate is
    case-sensitive separately; the guard only decides availability)."""
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_nrx", brand="remibrutinib", region=None, window=None, biologic="naive"
    )
    assert qid == "business_impact_nrx_biologic"
    assert params == ["remibrutinib", "naive"]


# --- precedence ------------------------------------------------------------


def test_segment_takes_precedence_over_biologic():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx",
        brand="Remibrutinib",
        region=None,
        window=None,
        segment="high_severity",
        biologic="experienced",
    )
    assert qid == "business_impact_trx_segment"
    assert params == ["Remibrutinib", "high_severity"]


def test_therapy_line_takes_precedence_over_biologic():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx",
        brand="Remibrutinib",
        region=None,
        window=None,
        therapy_line=1,
        biologic="experienced",
    )
    assert qid == "business_impact_trx_line"
    assert params == ["Remibrutinib", 1]


def test_biologic_takes_precedence_over_ige_tier():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx",
        brand="Remibrutinib",
        region=None,
        window=None,
        biologic="naive",
        ige_tier="low",
    )
    assert qid == "business_impact_trx_biologic"
    assert params == ["Remibrutinib", "naive"]


def test_biologic_takes_precedence_over_region():
    qid, params = _calc()._resolve_windowed_call(
        "business_impact_trx",
        brand="Remibrutinib",
        region="northeast",
        window=None,
        biologic="experienced",
    )
    assert qid == "business_impact_trx_biologic"
    assert params == ["Remibrutinib", "experienced"]


# --- volume calculators thread the axes (stub client) ----------------------


class _Resp:
    def __init__(self, data):
        self.data = data


class _Exec:
    def __init__(self, data):
        self._d = data

    def execute(self):
        return _Resp(self._d)


class _StubClient:
    def __init__(self, row):
        self.row = row
        self.calls = []

    def rpc(self, name, payload):
        self.calls.append(payload)
        return _Exec([self.row])


def test_nrx_threads_biologic():
    client = _StubClient({"nrx": 1258})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_nrx({"brand": "Remibrutinib", "biologic": "experienced"})
    assert value == 1258.0
    assert client.calls[0]["query_id"] == "business_impact_nrx_biologic"
    assert client.calls[0]["params"] == ["Remibrutinib", "experienced"]


def test_trx_threads_ige_tier():
    client = _StubClient({"trx": 4321})
    calc = BusinessImpactCalculator(db_client=client)
    value = calc._calc_trx({"brand": "Remibrutinib", "ige_tier": "low"})
    assert value == 4321.0
    assert client.calls[0]["query_id"] == "business_impact_trx_ige_tier"
    assert client.calls[0]["params"] == ["Remibrutinib", "low"]


def test_nbrx_biologic_offbrand_fails_closed():
    """The guard raises before any query is sent (calls stays empty)."""
    client = _StubClient({"nbrx": 1})
    calc = BusinessImpactCalculator(db_client=client)
    with pytest.raises(RuntimeError, match="not available for Kisqali"):
        calc._calc_nbrx({"brand": "Kisqali", "biologic": "experienced"})
    assert client.calls == []


def test_trx_share_threads_biologic_never_windowed():
    """TRx Share has no windowed variant: even with a window key in context, the
    resolved id stays the BASE `_biologic` id."""
    client = _StubClient({"share": 0.42})
    calc = BusinessImpactCalculator(db_client=client)
    context = {
        "brand": "Remibrutinib",
        "biologic": "experienced",
        "window": {"start": "S", "end": "E"},
    }
    value = calc._calc_trx_share(context)
    assert value == 0.42
    assert client.calls[0]["query_id"] == "business_impact_trx_share_biologic"
    assert client.calls[0]["params"] == ["Remibrutinib", "experienced"]
