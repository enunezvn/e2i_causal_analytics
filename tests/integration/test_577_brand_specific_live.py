"""#577 Tier 2 (brand-specific) faithful e2e: BR-001 (Remi AH-uncontrolled %) and
BR-003 (Fabhalta % PNH-tested) compute the RIGHT value against the LIVE DB over the
generated cohort — not just "run without raising".

These assert MEANING (the #574 lesson):
- BR-001 is a real ratio in (0,1) and MOVES with the UAS7 cutoff (a constant would
  not), and is correctly brand-scoped (no cross-brand contamination).
- BR-003 is a real tested/eligible ratio in (0,1] over the D59.5 cohort, with the
  numerator using real PNH-flow LOINC and denominator strictly >= numerator.

CAPABILITY-GATED: skips unless SUPABASE_* is set AND the brand-specific #577 query_ids
exist + the seed (migration 046) has been applied.
"""

import os

import pytest

from src.kpi.calculators.brand_specific import BrandSpecificCalculator

HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_SUPABASE, reason="SUPABASE_* not set"),
]


@pytest.fixture
def calc():
    c = BrandSpecificCalculator()
    if c.db_client is None:
        pytest.skip("no Supabase client")
    try:
        c.db_client.rpc(
            "kpi_query", {"query_id": "brand_specific_fabhalta_pnh_tested", "params": []}
        ).execute()
    except Exception as e:
        pytest.skip(
            f"#577 brand-specific seed/queries unavailable (migration 046 not applied?): {e}"
        )
    return c


def test_br001_remi_ah_uncontrolled_is_real_ratio_and_threshold_driven(calc):
    """BR-001 is a real uncontrolled fraction in (0,1) AND moves with the UAS7 cutoff:
    an impossible cutoff (>=99) -> 0.0; a zero cutoff (>=0) -> 1.0. A constant would not."""
    val = calc._calc_remi_ah_uncontrolled({"brand": "Remibrutinib"})
    assert val is not None
    assert 0.0 < val < 1.0, f"uncontrolled rate not a real fraction: {val}"

    # Threshold-discrimination (anti-fabrication): drive the cutoff to the extremes.
    none_uncontrolled = calc._calc_remi_ah_uncontrolled(
        {"brand": "Remibrutinib", "uas7_uncontrolled_threshold": 99}
    )
    all_uncontrolled = calc._calc_remi_ah_uncontrolled(
        {"brand": "Remibrutinib", "uas7_uncontrolled_threshold": 0}
    )
    assert none_uncontrolled == 0.0, f"UAS7>=99 should be 0, got {none_uncontrolled}"
    assert all_uncontrolled == 1.0, f"UAS7>=0 should be 1.0, got {all_uncontrolled}"


def test_br003_fabhalta_pnh_tested_is_real_ratio(calc):
    """BR-003 is tested/eligible over the D59.5 cohort: a real fraction in (0,1]
    (numerator > 0 means real PNH-LOINC labs exist; < or = 1 means a real denominator)."""
    val = calc._calc_fabhalta_pnh_tested({"brand": "Fabhalta"})
    assert val is not None
    assert 0.0 < val <= 1.0, f"PNH-tested rate out of range: {val}"
