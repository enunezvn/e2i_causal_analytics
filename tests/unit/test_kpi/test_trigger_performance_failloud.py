"""Anti-fabrication fail-loud contract for WS2 TriggerPerformanceCalculator (#421/#439/#574/#577).

The seven remaining `_calc_*` methods used to end with a fabricating `return 0.0`,
inventing a zero KPI whenever the kpi_query RPC returned no rows or a NULL value on a
dead/empty backend. They are now hardened to mirror the WS2-TR-003 precedent: the
empty/NULL FALLBACK raises ``RuntimeError("KPI <ID> unavailable: ...")``, while a
GENUINE value returned BY the query — INCLUDING a real ``0.0`` rate (a legitimately-zero
acceptance/override/false-alert/etc. rate) — is STILL returned, never raised.

Each hardened method is asserted against three cases:
  1. empty result (``data=[]``) and NULL (``data=[{<key>: None}]``) -> raise "unavailable".
  2. a real non-zero value is returned unchanged.
  3. a GENUINE ``0.0`` (``data=[{<key>: 0.0}]``) is RETURNED, not raised — the key
     anti-regression for rate metrics where zero is meaningful.
"""

from unittest.mock import MagicMock

import pytest

from src.kpi.calculators.trigger_performance import TriggerPerformanceCalculator


def _calc_returning(rows):
    """A TriggerPerformanceCalculator whose kpi_query RPC returns `rows`."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=rows)
    return TriggerPerformanceCalculator(db_client=client)


# (method name, result key, KPI id, a real non-zero value to round-trip)
# WS2-TR-003 (_calc_action_rate_uplift) is intentionally EXCLUDED — it was already
# hardened and is covered by test_kpi_query_forwarding.py.
HARDENED_METHODS = [
    ("_calc_trigger_precision", "precision", "WS2-TR-001", 0.87),
    ("_calc_trigger_recall", "recall", "WS2-TR-002", 0.73),
    ("_calc_acceptance_rate", "acceptance_rate", "WS2-TR-004", 0.42),
    ("_calc_false_alert_rate", "false_alert_rate", "WS2-TR-005", 0.08),
    ("_calc_override_rate", "override_rate", "WS2-TR-006", 0.15),
    ("_calc_lead_time", "median_lead_time", "WS2-TR-007", 12.5),
    ("_calc_change_fail_rate", "cfr", "WS2-TR-008", 0.05),
]

_IDS = [f"{method}[{kpi_id}]" for method, _key, kpi_id, _val in HARDENED_METHODS]


@pytest.mark.parametrize("method,key,kpi_id,real_value", HARDENED_METHODS, ids=_IDS)
def test_fails_loud_on_empty_result(method, key, kpi_id, real_value):
    """No rows (``data=[]``) -> raise "unavailable", NOT a fabricated 0.0. The `result`
    falsiness guard must fire before `result[0]` (no IndexError)."""
    calc = _calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        getattr(calc, method)({})


@pytest.mark.parametrize("method,key,kpi_id,real_value", HARDENED_METHODS, ids=_IDS)
def test_fails_loud_on_null_value(method, key, kpi_id, real_value):
    """A row with a NULL metric (``data=[{<key>: None}]``) -> raise "unavailable",
    NOT a fabricated 0.0."""
    calc = _calc_returning([{key: None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        getattr(calc, method)({})


@pytest.mark.parametrize("method,key,kpi_id,real_value", HARDENED_METHODS, ids=_IDS)
def test_real_value_is_returned(method, key, kpi_id, real_value):
    """A genuine non-zero value returned by the query is passed through unchanged."""
    calc = _calc_returning([{key: real_value}])
    assert getattr(calc, method)({}) == pytest.approx(real_value)


@pytest.mark.parametrize("method,key,kpi_id,real_value", HARDENED_METHODS, ids=_IDS)
def test_genuine_zero_rate_is_returned_not_raised(method, key, kpi_id, real_value):
    """THE KEY ANTI-REGRESSION: a genuine ``0.0`` returned BY the query (a legitimately
    zero rate — e.g. acceptance/override/false-alert rate of 0, or a real zero
    precision/recall/lead-time/CFR) is a meaningful value and MUST be returned, not
    raised. Only the empty/NULL no-data FALLBACK raises."""
    calc = _calc_returning([{key: 0.0}])
    assert getattr(calc, method)({}) == 0.0


@pytest.mark.parametrize("method,key,kpi_id,real_value", HARDENED_METHODS, ids=_IDS)
def test_raise_message_names_the_correct_kpi_id(method, key, kpi_id, real_value):
    """The fail-loud message must name the method's own KPI id (from the calculator_map),
    so an operator can tell WHICH metric had no data."""
    calc = _calc_returning([])
    with pytest.raises(RuntimeError, match=kpi_id):
        getattr(calc, method)({})
