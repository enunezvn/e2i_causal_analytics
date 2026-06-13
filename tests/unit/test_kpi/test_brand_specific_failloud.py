"""Anti-fabrication fail-loud tests for BrandSpecificCalculator (#421/#439/#574/#577).

The remaining ``_calc_*`` methods that ended with a fabricating ``return 0.0`` on the
empty/NULL path are hardened to raise ``RuntimeError("KPI BR-... unavailable: ...")``,
mirroring the existing BR-001 (``_calc_remi_ah_uncontrolled``) and BR-003
(``_calc_fabhalta_pnh_tested``) precedents in the same file.

For each hardened method:
  1. empty result (``data=[]``) -> raises RuntimeError matching "unavailable"
  2. NULL key   (``data=[{<key>: None}]``) -> raises RuntimeError matching "unavailable"
  3. a real value (including a genuine 0.0 from the query) is still RETURNED, never raised.

BR-002 (``_calc_remi_intent_delta``) additionally tries a PRIMARY query then a FALLBACK
query before raising — so a NULL primary with a valid fallback must return the fallback
value (the primary->fallback chain is preserved; only the FINAL empty path raises).
"""

from unittest.mock import MagicMock

import pytest

from src.kpi.calculators.brand_specific import BrandSpecificCalculator


def _brand_calc_returning(rows):
    """A BrandSpecificCalculator whose kpi_query RPC returns ``rows`` on every call."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=rows)
    return BrandSpecificCalculator(db_client=client), client


def _brand_calc_side_effect(rows_per_call):
    """A BrandSpecificCalculator whose successive ``.rpc().execute()`` calls return
    each entry in ``rows_per_call`` in order (for the primary->fallback chain)."""
    client = MagicMock()
    client.rpc.return_value.execute.side_effect = [MagicMock(data=rows) for rows in rows_per_call]
    return BrandSpecificCalculator(db_client=client), client


# --- BR-002 Remi Intent-to-Prescribe Delta (primary -> fallback -> fail loud) ----------


def test_br002_intent_delta_primary_value_returned():
    """A valid PRIMARY intent_delta is returned (the fallback is never consulted)."""
    calc, _ = _brand_calc_returning([{"intent_delta": 0.32}])
    assert calc._calc_remi_intent_delta({"brand": "Remibrutinib"}) == pytest.approx(0.32)


def test_br002_intent_delta_primary_null_falls_back_to_valid_fallback():
    """NULL primary but a VALID fallback -> the fallback value is returned (chain intact)."""
    calc, client = _brand_calc_side_effect([[{"intent_delta": None}], [{"intent_delta": 0.21}]])
    assert calc._calc_remi_intent_delta({"brand": "Remibrutinib"}) == pytest.approx(0.21)
    # The chain made two RPC calls: the primary view, then the fallback.
    assert client.rpc.return_value.execute.call_count == 2


def test_br002_intent_delta_primary_empty_falls_back_to_valid_fallback():
    """Empty primary (``[]``) but a VALID fallback -> the fallback value is returned."""
    calc, client = _brand_calc_side_effect([[], [{"intent_delta": -0.05}]])
    assert calc._calc_remi_intent_delta({"brand": "Remibrutinib"}) == pytest.approx(-0.05)
    assert client.rpc.return_value.execute.call_count == 2


def test_br002_intent_delta_fails_loud_when_both_null():
    """NULL primary AND NULL fallback -> fail loud (NOT a fabricated 0.0 delta)."""
    calc, _ = _brand_calc_side_effect([[{"intent_delta": None}], [{"intent_delta": None}]])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_remi_intent_delta({"brand": "Remibrutinib"})


def test_br002_intent_delta_fails_loud_when_both_empty():
    """Empty primary AND empty fallback (``[]``) -> fail loud, never IndexError."""
    calc, _ = _brand_calc_side_effect([[], []])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_remi_intent_delta({"brand": "Remibrutinib"})


def test_br002_intent_delta_genuine_zero_primary_returned_not_raised():
    """A genuine 0.0 delta from the PRIMARY (HCPs scored, no net change) is a legitimate
    value -> returned, never raised."""
    calc, _ = _brand_calc_returning([{"intent_delta": 0.0}])
    assert calc._calc_remi_intent_delta({"brand": "Remibrutinib"}) == 0.0


def test_br002_intent_delta_genuine_zero_fallback_returned_not_raised():
    """A genuine 0.0 delta arriving via the FALLBACK (NULL primary) is returned, not raised."""
    calc, _ = _brand_calc_side_effect([[{"intent_delta": None}], [{"intent_delta": 0.0}]])
    assert calc._calc_remi_intent_delta({"brand": "Remibrutinib"}) == 0.0


# --- BR-004 Kisqali Dx Adoption (median days; lower-is-better) --------------------------


def test_br004_dx_adoption_value_returned():
    """A real median_days is returned."""
    calc, _ = _brand_calc_returning([{"median_days": 42.0}])
    assert calc._calc_kisqali_dx_adoption({"brand": "Kisqali"}) == pytest.approx(42.0)


def test_br004_dx_adoption_fails_loud_on_empty():
    """No dx->first-Rx pairs (empty result) -> fail loud (NOT a fabricated 0 days)."""
    calc, _ = _brand_calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_kisqali_dx_adoption({"brand": "Kisqali"})


def test_br004_dx_adoption_fails_loud_on_null():
    """NULL median_days -> fail loud (NOT a fabricated 0 days under lower-is-better)."""
    calc, _ = _brand_calc_returning([{"median_days": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_kisqali_dx_adoption({"brand": "Kisqali"})


def test_br004_dx_adoption_genuine_zero_returned_not_raised():
    """A genuine median of 0.0 days from the query is returned by the value-present
    branch, never raised."""
    calc, _ = _brand_calc_returning([{"median_days": 0.0}])
    assert calc._calc_kisqali_dx_adoption({"brand": "Kisqali"}) == 0.0


# --- BR-005 Kisqali Oncologist Reach (% engaged) ----------------------------------------


def test_br005_oncologist_reach_value_returned():
    """A real reach fraction is returned."""
    calc, _ = _brand_calc_returning([{"reach": 0.73}])
    assert calc._calc_kisqali_oncologist_reach({"brand": "Kisqali"}) == pytest.approx(0.73)


def test_br005_oncologist_reach_fails_loud_on_empty():
    """No oncologist universe (empty result) -> fail loud (NOT a fabricated 0% reach)."""
    calc, _ = _brand_calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_kisqali_oncologist_reach({"brand": "Kisqali"})


def test_br005_oncologist_reach_fails_loud_on_null():
    """NULL reach -> fail loud (NOT a fabricated 0% reach)."""
    calc, _ = _brand_calc_returning([{"reach": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_kisqali_oncologist_reach({"brand": "Kisqali"})


def test_br005_oncologist_reach_genuine_zero_returned_not_raised():
    """A genuine 0.0 reach (universe exists, none engaged) is a LEGITIMATE value returned by
    the value-present branch, never raised."""
    calc, _ = _brand_calc_returning([{"reach": 0.0}])
    assert calc._calc_kisqali_oncologist_reach({"brand": "Kisqali"}) == 0.0
