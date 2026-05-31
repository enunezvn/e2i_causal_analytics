"""#577 WS3-BI-003 (patient_touch_rate) faithful e2e: the metric computes the RIGHT
value against the LIVE DB over the real cohort — not just "runs without raising".

These assert MEANING (the #574 lesson):
- The touch rate is a real fraction in (0,1) and is NON-DEGENERATE: it lands in the
  honest ~0.91 band, NOT the degenerate ~0.995 that the rejected "any trigger"
  definition would produce. The honesty lever is delivery_status IN
  ('delivered','viewed') — a trigger that actually reached the patient. Verified
  live over the SAME code-anchored eligible cohort: delivered=0.9074 vs
  any-trigger=0.9948 (a real 236-patient reach gap). The upper-bound assertion
  regression-locks against anyone reverting the metric to the any-trigger relabel.
- The optional brand param genuinely MOVES the value (a constant would not), and a
  brand with no code-anchored eligible cohort (e.g. 'competitor') FAILS LOUD rather
  than fabricating a 0.0.

CAPABILITY-GATED: skips unless SUPABASE_* is set AND the patient-touch query_id exists
(migration 050 applied).
"""

import os

import pytest

from src.kpi.calculators.business_impact import BusinessImpactCalculator

HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_SUPABASE, reason="SUPABASE_* not set"),
]


@pytest.fixture
def calc():
    c = BusinessImpactCalculator()
    if c.db_client is None:
        pytest.skip("no Supabase client")
    try:
        c.db_client.rpc(
            "kpi_query", {"query_id": "business_impact_patient_touch_rate", "params": [""]}
        ).execute()
    except Exception as e:
        pytest.skip(f"#577 patient-touch view/query unavailable (migration 050 not applied?): {e}")
    return c


def test_patient_touch_rate_is_real_nondegenerate_fraction(calc):
    """The all-brands touch rate is a real fraction in (0,1) and is NON-DEGENERATE.

    The honest delivered-touch value (~0.907) lands well below the degenerate
    ~0.995 the rejected any-trigger definition would give. The < 0.96 upper bound
    is the anti-fabrication guard: reverting the metric to "any trigger" (counting
    pending/failed/expired as a touchpoint) would push it above this bound and fail."""
    val = calc._calc_patient_touch_rate({})
    assert val is not None
    assert 0.0 < val <= 1.0, f"touch rate out of range: {val}"
    assert 0.85 < val < 0.96, (
        f"touch rate {val} is outside the honest delivered-touch band; ~0.995 would mean "
        "the degenerate any-trigger relabel #574 forbids"
    )


def test_patient_touch_rate_brand_param_moves_the_value(calc):
    """The optional brand filter genuinely changes the cohort (a constant would not),
    and per-brand values reflect the real per-brand reach gaps (Fabhalta < Kisqali live)."""
    all_brands = calc._calc_patient_touch_rate({})
    fabhalta = calc._calc_patient_touch_rate({"brand": "Fabhalta"})
    kisqali = calc._calc_patient_touch_rate({"brand": "Kisqali"})
    remi = calc._calc_patient_touch_rate({"brand": "Remibrutinib"})

    # The param actually filters: brand-scoped values differ from the all-brands aggregate.
    assert fabhalta != all_brands, "brand param had no effect — filter not applied"
    # Each per-brand value is a real, non-degenerate fraction.
    for b, v in [("Fabhalta", fabhalta), ("Kisqali", kisqali), ("Remibrutinib", remi)]:
        assert 0.85 < v < 0.96, f"{b} touch rate {v} outside the honest band"
    # Real per-brand ordering: Fabhalta has the largest reach gap (lowest rate) live.
    assert fabhalta < kisqali, f"expected Fabhalta ({fabhalta}) < Kisqali ({kisqali}) per live data"


def test_patient_touch_rate_fails_loud_for_brand_with_no_eligible_cohort(calc):
    """A brand with no code-anchored eligible cohort (no qualifying dx) -> empty
    denominator -> NULL -> fail loud, NEVER a fabricated 0.0."""
    with pytest.raises(RuntimeError, match="unavailable"):
        calc._calc_patient_touch_rate({"brand": "competitor"})
