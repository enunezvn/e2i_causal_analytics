"""WS2 trigger-effectiveness ask-bound dispatch (#1360, migration 118).

The #1360 ruling: trigger precision / acceptance rate / override rate / trigger
funnel conversion are chat-KPI-path KPIs. The calculator must bind the ask's
axes through the migration-118 statement families:

  * no trigger_type, no window  -> UNCHANGED legacy routing (certified base /
    _region / _brand / _brand_region ids stay byte-identical — mig 078/113).
  * trigger_type (no window)    -> trigger_effectiveness_<metric>,
                                   [brand, region, trigger_type] (nullables).
  * window (no region)          -> trigger_effectiveness_<metric>_windowed,
                                   [brand, trigger_type, start, end].
  * region + window             -> trigger_effectiveness_<metric>_windowed_region,
                                   [brand, region, trigger_type, start, end]
                                   (#1388: the kpi_query RPC now binds 6 positional
                                   params — migration 120 — so region no longer has
                                   to be dropped when a window is asked for).

The funnel KPI (WS2-TR-009) is new — it ALWAYS routes to the effectiveness
family and surfaces the stage counts via context["funnel_stages"].
"""

from unittest.mock import MagicMock

import pytest

from src.kpi.calculators.trigger_performance import TriggerPerformanceCalculator
from src.kpi.registry import get_registry


@pytest.fixture(autouse=True)
def _synthetic_flag_off(monkeypatch):
    """Hermetic env isolation (#1289 lesson): this droplet's .env sets
    E2I_KPI_INCLUDE_SYNTHETIC, which would suffix every id and turn these
    routing assertions env-dependent."""
    monkeypatch.delenv("E2I_KPI_INCLUDE_SYNTHETIC", raising=False)
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)


# (method, result key, effectiveness metric, legacy base id) — the three ruled
# KPIs that already had legacy statements.
RULED_EXISTING = [
    ("_calc_trigger_precision", "precision", "precision", "trigger_performance_precision"),
    (
        "_calc_acceptance_rate",
        "acceptance_rate",
        "acceptance_rate",
        "trigger_performance_acceptance_rate",
    ),
    (
        "_calc_override_rate",
        "override_rate",
        "override_rate",
        "trigger_performance_override_rate",
    ),
]
_IDS = [m for m, _k, _e, _b in RULED_EXISTING]

WINDOW = {"start": "2026-06-01", "end": "2026-07-01"}


def _calc_and_client(key, value=0.5):
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=[{key: value}])
    return TriggerPerformanceCalculator(db_client=client), client


def _rpc_payload(client):
    args, kwargs = client.rpc.call_args
    assert args[0] == "kpi_query"
    return args[1]


# ---------------------------------------------------------------------------
# Legacy routing stays byte-identical when no new axis is asked for
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method,key,metric,base", RULED_EXISTING, ids=_IDS)
def test_no_new_axis_keeps_certified_legacy_routing(method, key, metric, base):
    calc, client = _calc_and_client(key)
    getattr(calc, method)({"brand": "Kisqali"})
    payload = _rpc_payload(client)
    assert payload == {"query_id": f"{base}_brand", "params": ["Kisqali"]}


# ---------------------------------------------------------------------------
# trigger_type axis -> effectiveness family (3 nullable params)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method,key,metric,base", RULED_EXISTING, ids=_IDS)
def test_trigger_type_routes_to_effectiveness_family(method, key, metric, base):
    calc, client = _calc_and_client(key)
    getattr(calc, method)({"trigger_type": "adherence_risk"})
    payload = _rpc_payload(client)
    assert payload == {
        "query_id": f"trigger_effectiveness_{metric}",
        "params": [None, None, "adherence_risk"],
    }


@pytest.mark.parametrize("method,key,metric,base", RULED_EXISTING, ids=_IDS)
def test_brand_region_trigger_type_bind_in_declared_order(method, key, metric, base):
    calc, client = _calc_and_client(key)
    getattr(calc, method)({"brand": "Fabhalta", "region": "west", "trigger_type": "engagement_gap"})
    payload = _rpc_payload(client)
    assert payload == {
        "query_id": f"trigger_effectiveness_{metric}",
        "params": ["Fabhalta", "west", "engagement_gap"],
    }


# ---------------------------------------------------------------------------
# window axis -> _windowed family (4 params, half-open)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method,key,metric,base", RULED_EXISTING, ids=_IDS)
def test_window_routes_to_windowed_family(method, key, metric, base):
    calc, client = _calc_and_client(key)
    getattr(calc, method)({"window": dict(WINDOW)})
    payload = _rpc_payload(client)
    assert payload == {
        "query_id": f"trigger_effectiveness_{metric}_windowed",
        "params": [None, None, WINDOW["start"], WINDOW["end"]],
    }


@pytest.mark.parametrize("method,key,metric,base", RULED_EXISTING, ids=_IDS)
def test_brand_trigger_type_window_bind_in_declared_order(method, key, metric, base):
    calc, client = _calc_and_client(key)
    getattr(calc, method)(
        {"brand": "Remibrutinib", "trigger_type": "cross_sell", "window": dict(WINDOW)}
    )
    payload = _rpc_payload(client)
    assert payload == {
        "query_id": f"trigger_effectiveness_{metric}_windowed",
        "params": ["Remibrutinib", "cross_sell", WINDOW["start"], WINDOW["end"]],
    }


@pytest.mark.parametrize("method,key,metric,base", RULED_EXISTING, ids=_IDS)
def test_region_plus_window_routes_to_windowed_regioned(method, key, metric, base):
    """#1388: region+window now CO-BIND. The kpi_query RPC binds 6 positional
    params (migration 120), so region no longer has to be dropped when a window
    is asked for — the ask routes to the ``_windowed_region`` variant that binds
    brand ($1), region ($2, via the patient_journeys join), trigger_type ($3)
    and the half-open window ($4/$5), never a silent region drop (the
    dead-'territory'-key lesson)."""
    calc, client = _calc_and_client(key)
    getattr(calc, method)(
        {"region": "east", "trigger_type": "adherence_risk", "window": dict(WINDOW)}
    )
    payload = _rpc_payload(client)
    assert payload == {
        "query_id": f"trigger_effectiveness_{metric}_windowed_region",
        "params": [None, "east", "adherence_risk", WINDOW["start"], WINDOW["end"]],
    }


@pytest.mark.parametrize("method,key,metric,base", RULED_EXISTING, ids=_IDS)
def test_brand_region_trigger_type_window_bind_in_declared_order(method, key, metric, base):
    """#1388: brand + region + trigger_type + window all bind, in the migration
    120 declared order [brand, region, trigger_type, start, end]."""
    calc, client = _calc_and_client(key)
    getattr(calc, method)(
        {
            "brand": "Kisqali",
            "region": "west",
            "trigger_type": "engagement_gap",
            "window": dict(WINDOW),
        }
    )
    payload = _rpc_payload(client)
    assert payload == {
        "query_id": f"trigger_effectiveness_{metric}_windowed_region",
        "params": ["Kisqali", "west", "engagement_gap", WINDOW["start"], WINDOW["end"]],
    }


def test_windowed_id_self_suffixes_include_synthetic_under_flag(monkeypatch):
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
    calc, client = _calc_and_client("precision")
    calc._calc_trigger_precision({"trigger_type": "reactivation", "window": dict(WINDOW)})
    payload = _rpc_payload(client)
    assert payload["query_id"] == "trigger_effectiveness_precision_windowed_include_synthetic"
    assert payload["params"] == [None, "reactivation", WINDOW["start"], WINDOW["end"]]


def test_windowed_regioned_id_self_suffixes_include_synthetic_under_flag(monkeypatch):
    """#1388: the regioned+windowed id also self-suffixes _include_synthetic
    under the showcase flag (same additive idiom as the plain _windowed id)."""
    monkeypatch.setenv("E2I_KPI_INCLUDE_SYNTHETIC", "1")
    calc, client = _calc_and_client("precision")
    calc._calc_trigger_precision(
        {"region": "north", "trigger_type": "reactivation", "window": dict(WINDOW)}
    )
    payload = _rpc_payload(client)
    assert (
        payload["query_id"] == "trigger_effectiveness_precision_windowed_region_include_synthetic"
    )
    assert payload["params"] == [None, "north", "reactivation", WINDOW["start"], WINDOW["end"]]


# ---------------------------------------------------------------------------
# WS2-TR-009 Trigger Funnel Conversion (new)
# ---------------------------------------------------------------------------
FUNNEL_ROW = {
    "funnel_conversion": 0.2262,
    "n_delivered": 33023,
    "n_viewed": 9351,
    "n_accepted": 18120,
    "n_actioned": 7471,
    "n_outcome": 1830,
    "data_through": "2026-07-30",
}


def test_funnel_no_scope_routes_to_effectiveness_family():
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=[dict(FUNNEL_ROW)])
    calc = TriggerPerformanceCalculator(db_client=client)
    value = calc._calc_funnel_conversion({})
    payload = _rpc_payload(client)
    assert payload == {
        "query_id": "trigger_effectiveness_funnel_conversion",
        "params": [None, None, None],
    }
    assert value == pytest.approx(0.2262)


def test_funnel_stashes_stage_counts_and_data_through_into_context():
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=[dict(FUNNEL_ROW)])
    calc = TriggerPerformanceCalculator(db_client=client)
    context: dict = {}
    calc._calc_funnel_conversion(context)
    assert context["funnel_stages"] == {
        "delivered": 33023,
        "viewed": 9351,
        "accepted": 18120,
        "actioned": 7471,
        "outcome": 1830,
    }
    assert context["data_through"] == "2026-07-30"


def test_funnel_windowed_routing():
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=[dict(FUNNEL_ROW)])
    calc = TriggerPerformanceCalculator(db_client=client)
    calc._calc_funnel_conversion({"brand": "Kisqali", "window": dict(WINDOW)})
    payload = _rpc_payload(client)
    assert payload == {
        "query_id": "trigger_effectiveness_funnel_conversion_windowed",
        "params": ["Kisqali", None, WINDOW["start"], WINDOW["end"]],
    }


def test_funnel_windowed_regioned_routing():
    """#1388: the funnel KPI also co-binds region+window via the migration 120
    _windowed_region variant, params [brand, region, trigger_type, start, end]."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=[dict(FUNNEL_ROW)])
    calc = TriggerPerformanceCalculator(db_client=client)
    calc._calc_funnel_conversion({"brand": "Kisqali", "region": "west", "window": dict(WINDOW)})
    payload = _rpc_payload(client)
    assert payload == {
        "query_id": "trigger_effectiveness_funnel_conversion_windowed_region",
        "params": ["Kisqali", "west", None, WINDOW["start"], WINDOW["end"]],
    }


def test_funnel_fails_loud_when_no_delivered_rows():
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(
        data=[{"funnel_conversion": None, "n_delivered": 0}]
    )
    calc = TriggerPerformanceCalculator(db_client=client)
    with pytest.raises(RuntimeError, match="WS2-TR-009"):
        calc._calc_funnel_conversion({})


def test_funnel_wired_into_calculate_dispatch():
    """calculate() must dispatch WS2-TR-009 (registry-loaded) to the funnel calc
    and return an INFORMATIONAL result (no ratified threshold yet)."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=[dict(FUNNEL_ROW)])
    calc = TriggerPerformanceCalculator(db_client=client)
    kpi = get_registry().get("WS2-TR-009")
    assert kpi is not None, "WS2-TR-009 must be defined in config/kpi_definitions.yaml"
    result = calc.calculate(kpi, context={})
    assert result.error is None
    assert result.value == pytest.approx(0.2262)
    assert result.status == "informational"
    assert result.metadata["context"]["funnel_stages"]["actioned"] == 7471


# ---------------------------------------------------------------------------
# Registry windowable metadata (window provenance stamping depends on it)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kpi_id", ["WS2-TR-001", "WS2-TR-004", "WS2-TR-006", "WS2-TR-009"])
def test_ruled_kpis_are_windowable_needs_care(kpi_id):
    """All four ruled KPIs accept explicit windows (needs_care: statuses/outcomes
    mature after delivery, so recent windows under-count — disclosed, not
    blocked). not_applicable would stamp window_status='not_applicable' and the
    chat answer would silently ignore the asked period."""
    kpi = get_registry().get(kpi_id)
    assert kpi is not None
    assert kpi.windowable == "needs_care", f"{kpi_id} must be windowable=needs_care"
    assert kpi.window is not None and kpi.window["column"] == "trigger_timestamp"
