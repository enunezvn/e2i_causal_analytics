"""WS2 trigger-KPI brand/region variant dispatch (migration 113).

The calculator must route each of the 8 WS2 KPIs to the right registry variant
from the request context, binding params in the variant's declared order:

  * no scope            -> certified base id, []          (byte-identical gates)
  * region only         -> {base}_region, [region]        (migration 078, unchanged)
  * brand only          -> {base}_brand, [brand]          (migration 113)
  * brand + region      -> {base}_brand_region, [brand, region]

Under E2I_KPI_INCLUDE_SYNTHETIC the scoped ids self-suffix _include_synthetic
via the synthetic_mode helpers (the scoped ids are deliberately ABSENT from
SYNTHETIC_TWINNED_QUERY_IDS; resolve_kpi_query_id is a no-op on them).
"""

from unittest.mock import MagicMock

import pytest

from src.kpi.calculators.trigger_performance import TriggerPerformanceCalculator


@pytest.fixture(autouse=True)
def _synthetic_flag_off(monkeypatch):
    """Hermetic env isolation (#1289 lesson): this droplet's .env sets
    E2I_KPI_INCLUDE_SYNTHETIC (synthetic-gold demo instance), which would
    suffix every id and turn these routing assertions env-dependent. Clears the
    deployment-wide E2I_INCLUDE_SYNTHETIC showcase switch too
    (kpi_include_synthetic honors both). The two flag-ON tests re-set one
    explicitly."""
    monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


# (method, result key, base query id) — all 8 WS2 calculators.
DISPATCHED = [
    ("_calc_trigger_precision", "precision", "trigger_performance_precision"),
    ("_calc_trigger_recall", "recall", "trigger_performance_recall"),
    ("_calc_action_rate_uplift", "action_rate_uplift", "trigger_performance_action_rate_uplift"),
    ("_calc_acceptance_rate", "acceptance_rate", "trigger_performance_acceptance_rate"),
    ("_calc_false_alert_rate", "false_alert_rate", "trigger_performance_false_alert_rate"),
    ("_calc_override_rate", "override_rate", "trigger_performance_override_rate"),
    ("_calc_lead_time", "median_lead_time", "trigger_performance_lead_time"),
    ("_calc_change_fail_rate", "cfr", "trigger_performance_cfr"),
]
_IDS = [m for m, _k, _q in DISPATCHED]


def _calc_and_client(key):
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=[{key: 0.5}])
    return TriggerPerformanceCalculator(db_client=client), client


def _rpc_payload(client):
    args, kwargs = client.rpc.call_args
    assert args[0] == "kpi_query"
    return args[1]


@pytest.mark.parametrize("method,key,base", DISPATCHED, ids=_IDS)
def test_no_scope_routes_to_certified_base(method, key, base):
    calc, client = _calc_and_client(key)
    getattr(calc, method)({})
    payload = _rpc_payload(client)
    assert payload == {"query_id": base, "params": []}


@pytest.mark.parametrize("method,key,base", DISPATCHED, ids=_IDS)
def test_region_scope_routes_to_region_variant(method, key, base):
    calc, client = _calc_and_client(key)
    getattr(calc, method)({"region": "northeast"})
    payload = _rpc_payload(client)
    assert payload == {"query_id": f"{base}_region", "params": ["northeast"]}


@pytest.mark.parametrize("method,key,base", DISPATCHED, ids=_IDS)
def test_brand_scope_routes_to_brand_variant(method, key, base):
    calc, client = _calc_and_client(key)
    getattr(calc, method)({"brand": "Remibrutinib"})
    payload = _rpc_payload(client)
    assert payload == {"query_id": f"{base}_brand", "params": ["Remibrutinib"]}


@pytest.mark.parametrize("method,key,base", DISPATCHED, ids=_IDS)
def test_brand_and_region_route_to_brand_region_variant(method, key, base):
    """brand binds $1, region $2 — the migration-113 declared order."""
    calc, client = _calc_and_client(key)
    getattr(calc, method)({"brand": "Kisqali", "region": "south"})
    payload = _rpc_payload(client)
    assert payload == {"query_id": f"{base}_brand_region", "params": ["Kisqali", "south"]}


def test_brand_scope_suffixes_include_synthetic_under_flag(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
    calc, client = _calc_and_client("precision")
    calc._calc_trigger_precision({"brand": "Fabhalta"})
    payload = _rpc_payload(client)
    assert payload["query_id"] == "trigger_performance_precision_brand_include_synthetic"
    assert payload["params"] == ["Fabhalta"]


def test_brand_region_scope_suffixes_include_synthetic_under_flag(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
    calc, client = _calc_and_client("recall")
    calc._calc_trigger_recall({"brand": "Fabhalta", "region": "west"})
    payload = _rpc_payload(client)
    assert payload["query_id"] == "trigger_performance_recall_brand_region_include_synthetic"
    assert payload["params"] == ["Fabhalta", "west"]
