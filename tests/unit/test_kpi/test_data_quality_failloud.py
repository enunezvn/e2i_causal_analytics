"""Anti-fabrication fail-loud contract for DataQualityCalculator (#421/#439/#574/#577).

Every WS1 data-quality `_calc_*` method that previously ended with a fabricating
`return 0.0`/`1.0` on the empty/NULL path now RAISES instead — a missing/NULL query
result must surface a real error, never a plausible-but-fake KPI value. A GENUINE real
value (including a real 0.0 from the query) must STILL be returned.

Mirrors the MagicMock RPC pattern in test_kpi_query_forwarding.py:
`DataQualityCalculator(db_client=<mock>)` whose `kpi_query` RPC returns `rows`.

NOTE: WS1-DQ-008 (_calc_label_quality) already raises structurally (covered in the
forwarding suite) and the tuple-returning fail-CLOSED feature_drift (WS1-MP-009, a
DIFFERENT calculator) are intentionally NOT exercised here — they were already hardened.
"""

from unittest.mock import MagicMock

import pytest

from src.kpi.calculators.data_quality import DataQualityCalculator


def _calc_returning(rows):
    """A DataQualityCalculator whose kpi_query RPC returns `rows`."""
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data=rows)
    return DataQualityCalculator(db_client=client), client


# --- WS1-DQ-001 source_coverage_patients (covered/total; total<=0 == no universe) ---------


def test_dq001_fails_loud_on_empty():
    calc, _ = _calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_source_coverage_patients({"brand": "Fabhalta"})


def test_dq001_fails_loud_on_null_total():
    calc, _ = _calc_returning([{"covered": 0, "total": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_source_coverage_patients({"brand": "Fabhalta"})


def test_dq001_fails_loud_on_zero_reference_universe():
    """total == 0 is an UNDEFINED ratio (no reference universe), NOT 0.0 coverage."""
    calc, _ = _calc_returning([{"covered": 0, "total": 0}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_source_coverage_patients({"brand": "Fabhalta"})


def test_dq001_real_value_returned():
    calc, _ = _calc_returning([{"covered": 75, "total": 100}])
    assert abs(calc._calc_source_coverage_patients({"brand": "Fabhalta"}) - 0.75) < 1e-9


def test_dq001_genuine_zero_coverage_returned():
    """0 covered over a REAL reference universe is a legitimate 0.0 coverage, returned."""
    calc, _ = _calc_returning([{"covered": 0, "total": 100}])
    assert calc._calc_source_coverage_patients({"brand": "Fabhalta"}) == 0.0


# --- WS1-DQ-002 source_coverage_hcps (covered/total; total<=0 == no universe) -------------


def test_dq002_fails_loud_on_empty():
    calc, _ = _calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_source_coverage_hcps({"brand": None})


def test_dq002_fails_loud_on_null_total():
    calc, _ = _calc_returning([{"covered": 0, "total": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_source_coverage_hcps({"brand": None})


def test_dq002_fails_loud_on_zero_reference_universe():
    calc, _ = _calc_returning([{"covered": 0, "total": 0}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_source_coverage_hcps({"brand": None})


def test_dq002_real_value_returned():
    calc, _ = _calc_returning([{"covered": 546, "total": 21240}])
    assert abs(calc._calc_source_coverage_hcps({"brand": None}) - 546 / 21240) < 1e-9


def test_dq002_genuine_zero_coverage_returned():
    calc, _ = _calc_returning([{"covered": 0, "total": 21240}])
    assert calc._calc_source_coverage_hcps({"brand": None}) == 0.0


# --- WS1-DQ-003 cross_source_match -------------------------------------------------------


def test_dq003_fails_loud_on_empty():
    calc, _ = _calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_cross_source_match({})


def test_dq003_fails_loud_on_null():
    calc, _ = _calc_returning([{"match_rate": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_cross_source_match({})


def test_dq003_real_value_returned():
    calc, _ = _calc_returning([{"match_rate": 0.87}])
    assert abs(calc._calc_cross_source_match({}) - 0.87) < 1e-9


def test_dq003_genuine_zero_returned():
    """Sources exist but none matched -> a legitimate 0.0 match rate, returned."""
    calc, _ = _calc_returning([{"match_rate": 0.0}])
    assert calc._calc_cross_source_match({}) == 0.0


# --- WS1-DQ-004 stacking_lift (old fabricated fallback was 1.0 "neutral") ----------------


def test_dq004_fails_loud_on_empty():
    calc, _ = _calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_stacking_lift({})


def test_dq004_fails_loud_on_null():
    calc, _ = _calc_returning([{"lift_score": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_stacking_lift({})


def test_dq004_real_value_returned():
    calc, _ = _calc_returning([{"lift_score": 1.42}])
    assert abs(calc._calc_stacking_lift({}) - 1.42) < 1e-9


def test_dq004_genuine_neutral_lift_returned_not_fabricated():
    """A REAL 1.0 lift (measured neutral) must return — distinct from the old fabricated
    1.0 emitted on empty/no-data."""
    calc, _ = _calc_returning([{"lift_score": 1.0}])
    assert calc._calc_stacking_lift({}) == 1.0


# --- WS1-DQ-005 completeness_pass_rate (old inline `or 0.0` swallowed NULL) ---------------


def test_dq005_fails_loud_on_empty():
    calc, _ = _calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_completeness_pass_rate({})


def test_dq005_fails_loud_on_null():
    calc, _ = _calc_returning([{"pass_rate": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_completeness_pass_rate({})


def test_dq005_real_value_returned():
    calc, _ = _calc_returning([{"pass_rate": 0.93}])
    assert abs(calc._calc_completeness_pass_rate({}) - 0.93) < 1e-9


def test_dq005_genuine_zero_returned():
    """Records exist but none passed completeness -> a legitimate 0.0, returned."""
    calc, _ = _calc_returning([{"pass_rate": 0.0}])
    assert calc._calc_completeness_pass_rate({}) == 0.0


# --- WS1-DQ-006 geographic_consistency (gap; 0.0 == perfect match, best case) -------------


def test_dq006_fails_loud_on_empty():
    calc, _ = _calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_geographic_consistency({"brand": "Fabhalta"})


def test_dq006_fails_loud_on_null():
    calc, _ = _calc_returning([{"max_gap": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_geographic_consistency({"brand": "Fabhalta"})


def test_dq006_real_value_returned():
    calc, _ = _calc_returning([{"max_gap": 0.1049}])
    assert abs(calc._calc_geographic_consistency({"brand": "Fabhalta"}) - 0.1049) < 1e-9


def test_dq006_genuine_zero_gap_returned():
    """A real 0.0 max_gap (source distribution perfectly matches the universe) is the
    legitimate best case, returned not raised."""
    calc, _ = _calc_returning([{"max_gap": 0.0}])
    assert calc._calc_geographic_consistency({"brand": "Fabhalta"}) == 0.0


# --- WS1-DQ-007 data_lag (days; 0.0 == same-day, best case) -------------------------------


def test_dq007_fails_loud_on_empty():
    calc, _ = _calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_data_lag({})


def test_dq007_fails_loud_on_null():
    calc, _ = _calc_returning([{"median_lag_days": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_data_lag({})


def test_dq007_real_value_returned():
    calc, _ = _calc_returning([{"median_lag_days": 3.5}])
    assert abs(calc._calc_data_lag({}) - 3.5) < 1e-9


def test_dq007_genuine_zero_lag_returned():
    """A real 0.0 median lag (data lands same-day) is the legitimate best case, returned."""
    calc, _ = _calc_returning([{"median_lag_days": 0.0}])
    assert calc._calc_data_lag({}) == 0.0


# --- WS1-DQ-009 time_to_release (hours; 0.0 == instantaneous, best case) ------------------


def test_dq009_fails_loud_on_empty():
    calc, _ = _calc_returning([])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_time_to_release({})


def test_dq009_fails_loud_on_null():
    calc, _ = _calc_returning([{"avg_ttr_hours": None}])
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_time_to_release({})


def test_dq009_real_value_returned():
    calc, _ = _calc_returning([{"avg_ttr_hours": 36.0}])
    assert abs(calc._calc_time_to_release({}) - 36.0) < 1e-9


def test_dq009_genuine_zero_returned():
    """A real 0.0 hours (instantaneous release) is the legitimate best case, returned."""
    calc, _ = _calc_returning([{"avg_ttr_hours": 0.0}])
    assert calc._calc_time_to_release({}) == 0.0


# --- Correct KPI id appears in each fail-loud message (guards the calculator_map mapping) --


@pytest.mark.parametrize(
    "method,kpi_id",
    [
        ("_calc_source_coverage_patients", "WS1-DQ-001"),
        ("_calc_source_coverage_hcps", "WS1-DQ-002"),
        ("_calc_cross_source_match", "WS1-DQ-003"),
        ("_calc_stacking_lift", "WS1-DQ-004"),
        ("_calc_completeness_pass_rate", "WS1-DQ-005"),
        ("_calc_geographic_consistency", "WS1-DQ-006"),
        ("_calc_data_lag", "WS1-DQ-007"),
        ("_calc_time_to_release", "WS1-DQ-009"),
    ],
)
def test_failloud_message_names_correct_kpi_id(method, kpi_id):
    calc, _ = _calc_returning([])
    with pytest.raises(RuntimeError, match=kpi_id):
        getattr(calc, method)({"brand": "Fabhalta"})
