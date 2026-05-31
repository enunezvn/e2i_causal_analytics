"""#577 Tier A faithful e2e: the two pure-wire KPI metrics compute the RIGHT value
against the LIVE DB through the kpi_query allowlist — not just "run without raising".

The #574 lesson: a query that executes can still measure the wrong thing. So these
assert MEANING — value in range, data-driven (responds to the brand param rather than
returning a constant), and (DQ-002) internally consistent covered<=total.

CAPABILITY-GATED: skips unless SUPABASE_* is set AND kpi_query exists (migration 044/045).
"""

import os

import pytest

from src.kpi.calculators.data_quality import DataQualityCalculator

HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_SUPABASE, reason="SUPABASE_* not set"),
]


@pytest.fixture
def calc():
    c = DataQualityCalculator()
    if c.db_client is None:
        pytest.skip("no Supabase client")
    try:
        # DQ-002's own registry id must exist (migration 045 applied), else skip.
        c.db_client.rpc(
            "kpi_query", {"query_id": "data_quality_source_coverage_hcps", "params": [None]}
        ).execute()
    except Exception as e:
        pytest.skip(
            f"data_quality_source_coverage_hcps unavailable (migration 045 not applied?): {e}"
        )
    return c


def test_dq002_source_coverage_hcps_is_real_ratio(calc):
    """WS1-DQ-002: covered HCPs / reference universe — a real fraction in (0, 1]."""
    val = calc._calc_source_coverage_hcps({"brand": None})
    assert val is not None
    assert 0.0 < val <= 1.0, f"coverage ratio out of range: {val}"


def test_dq006_geographic_consistency_is_data_driven(calc):
    """WS1-DQ-006: max regional |share_source - share_universe| in [0,1], and the value
    MOVES with the brand param (proves it reads data, not a hardcoded constant)."""
    v_all = calc._calc_geographic_consistency({"brand": None})
    v_fab = calc._calc_geographic_consistency({"brand": "Fabhalta"})
    assert v_all is not None and v_fab is not None
    assert 0.0 <= v_all <= 1.0, f"all-brand gap out of range: {v_all}"
    assert 0.0 <= v_fab <= 1.0, f"Fabhalta gap out of range: {v_fab}"
    # A brand-banded distributional gap must differ from the all-brand gap; a constant
    # return (the fabrication failure mode) would make these identical.
    assert v_all != v_fab, "geographic gap is constant across brand filter — not data-driven"
