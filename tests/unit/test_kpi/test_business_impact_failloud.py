"""Anti-fabrication fail-loud contract for BusinessImpactCalculator (#421/#439/#574).

Every WS3-BI _calc_* method that previously ended with a fabricating ``return 0.0``
on the empty/NULL path must now RAISE ``RuntimeError("KPI <id> unavailable: ...")``
— mirroring the existing _calc_patient_touch_rate precedent in the same file.

Two facets per hardened method:
  1. an empty result (``data=[]``) and a NULL value (``data=[{<key>: None}]``) FAIL LOUD;
  2. a genuine real value (``data=[{<key>: <number>}]``) is still returned — INCLUDING a
     real ``0.0`` returned BY the query, which is a legitimate measured value, not the
     fabricated empty-path zero.

Uses the same MagicMock db_client idiom as test_kpi_query_forwarding.py: a mock whose
``.rpc(...).execute()`` returns ``MagicMock(data=<rows>)``.
"""

from unittest.mock import MagicMock

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator


def _bi_calc_returning(rows):
    """A BusinessImpactCalculator whose kpi_query RPC returns ``rows``."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=rows)
    return BusinessImpactCalculator(db_client=client), client


# (method, result-key, KPI id, a real non-zero value, genuine-zero-is-valid?)
# genuine_zero_valid is True for rate/count metrics where a query-returned 0.0 is a real
# measured value (no MAU, zero TRx, 0% conversion). The empty/NULL path is the ONLY
# fabrication being removed; a query that genuinely returns 0.0 must still pass through.
HARDENED = [
    ("_calc_mau", "mau", "WS3-BI-001", 1234.0, True),
    ("_calc_wau", "wau", "WS3-BI-002", 567.0, True),
    ("_calc_hcp_coverage", "coverage", "WS3-BI-004", 0.73, True),
    ("_calc_trx", "trx", "WS3-BI-005", 4200.0, True),
    ("_calc_nrx", "nrx", "WS3-BI-006", 880.0, True),
    ("_calc_conversion_rate", "conversion_rate", "WS3-BI-009", 0.18, True),
    ("_calc_roi", "avg_roi", "WS3-BI-010", 3.4, True),
]

# Brand-required metrics carry an extra no-brand fail-loud guard (tested separately).
HARDENED_BRAND_REQUIRED = [
    ("_calc_nbrx", "nbrx", "WS3-BI-007", 312.0, True),
    ("_calc_trx_share", "share", "WS3-BI-008", 0.27, True),
]

# A context that supplies a brand so the no-brand guard never fires for these tests.
CTX = {"brand": "Fabhalta"}


@pytest.mark.parametrize(
    "method,key,kpi_id,real_value,_zero_ok",
    HARDENED + HARDENED_BRAND_REQUIRED,
    ids=[m for m, *_ in HARDENED + HARDENED_BRAND_REQUIRED],
)
def test_empty_result_fails_loud(method, key, kpi_id, real_value, _zero_ok):
    """An empty result (no rows) must RAISE, never fabricate a 0.0."""
    calc, _ = _bi_calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        getattr(calc, method)(CTX)


@pytest.mark.parametrize(
    "method,key,kpi_id,real_value,_zero_ok",
    HARDENED + HARDENED_BRAND_REQUIRED,
    ids=[m for m, *_ in HARDENED + HARDENED_BRAND_REQUIRED],
)
def test_null_value_fails_loud(method, key, kpi_id, real_value, _zero_ok):
    """A NULL metric value (key present, value None) must RAISE, never fabricate a 0.0."""
    calc, _ = _bi_calc_returning([{key: None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        getattr(calc, method)(CTX)


@pytest.mark.parametrize(
    "method,key,kpi_id,real_value,_zero_ok",
    HARDENED + HARDENED_BRAND_REQUIRED,
    ids=[m for m, *_ in HARDENED + HARDENED_BRAND_REQUIRED],
)
def test_error_message_names_the_right_kpi_id(method, key, kpi_id, real_value, _zero_ok):
    """The fail-loud message must name the correct KPI id for the method."""
    calc, _ = _bi_calc_returning([])
    with pytest.raises(RuntimeError, match=kpi_id):
        getattr(calc, method)(CTX)


@pytest.mark.parametrize(
    "method,key,kpi_id,real_value,_zero_ok",
    HARDENED + HARDENED_BRAND_REQUIRED,
    ids=[m for m, *_ in HARDENED + HARDENED_BRAND_REQUIRED],
)
def test_real_value_is_returned(method, key, kpi_id, real_value, _zero_ok):
    """A genuine real value from the query is returned unchanged (value-present branch
    is untouched by the hardening)."""
    calc, _ = _bi_calc_returning([{key: real_value}])
    assert getattr(calc, method)(CTX) == real_value


@pytest.mark.parametrize(
    "method,key,kpi_id,real_value,zero_ok",
    HARDENED + HARDENED_BRAND_REQUIRED,
    ids=[m for m, *_ in HARDENED + HARDENED_BRAND_REQUIRED],
)
def test_genuine_zero_from_query_is_returned_not_raised(method, key, kpi_id, real_value, zero_ok):
    """A genuine 0.0 RETURNED BY the query (key present, value 0.0) is a legitimate measured
    value and must be returned, NOT raised. Only the empty/NULL fallback was a fabrication."""
    if not zero_ok:
        pytest.skip(f"{method}: query-returned 0.0 not asserted (metric-specific)")
    calc, _ = _bi_calc_returning([{key: 0.0}])
    assert getattr(calc, method)(CTX) == 0.0


def test_mau_falls_back_to_direct_then_fails_loud():
    """MAU tries the view, then the fallback; when BOTH are empty it fails loud
    (the trailing fabricated 0.0 is gone). The view leg returning a real value is returned."""
    # Both legs empty -> raise.
    calc, _ = _bi_calc_returning([])
    with pytest.raises(RuntimeError, match="WS3-BI-001"):
        calc._calc_mau({})
    # View leg has a genuine value -> returned (no fallback consulted).
    calc, _ = _bi_calc_returning([{"mau": 999.0}])
    assert calc._calc_mau({}) == 999.0


def test_roi_falls_back_to_agent_activities_then_fails_loud():
    """ROI tries business_metrics then agent_activities; both empty -> fail loud."""
    calc, _ = _bi_calc_returning([])
    with pytest.raises(RuntimeError, match="WS3-BI-010"):
        calc._calc_roi({})


# --- Brand-required no-brand fail-loud guard (NBRx, TRx Share) ---------------------------


@pytest.mark.parametrize(
    "method,kpi_id",
    [("_calc_nbrx", "WS3-BI-007"), ("_calc_trx_share", "WS3-BI-008")],
)
def test_brand_required_metric_fails_loud_without_brand(method, kpi_id):
    """NBRx and TRx Share are brand-specific by definition: with no brand in context the
    metric is undefined. The old ``not brand -> return 0.0`` fabricated a plausible
    0 prescriptions / 0% share; it must now fail loud naming the KPI id."""
    calc, _ = _bi_calc_returning(
        [{"nbrx": 1.0, "share": 1.0}]
    )  # data irrelevant; guard fires first
    with pytest.raises(RuntimeError, match=kpi_id):
        getattr(calc, method)({})  # no brand
    # An empty-string brand is also "no brand" -> still fails loud.
    calc, _ = _bi_calc_returning([{"nbrx": 1.0, "share": 1.0}])
    with pytest.raises(RuntimeError, match="unavailable"):
        getattr(calc, method)({"brand": ""})
