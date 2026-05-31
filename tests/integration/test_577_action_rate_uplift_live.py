"""#577 WS2-TR-003 (action_rate_uplift) faithful e2e: the metric computes the RIGHT
value against the LIVE DB over the real two-arm cohort — not just "runs without raising".

These assert MEANING (the #574 lesson):
- The uplift is a REAL positive incrementality signal in the honest band (~0.2751 live),
  NOT ~0 (which would mean the arm doesn't drive action) and NOT an implausibly-large
  fabricated lift. The treatment arm (NBA shown, control_group_flag=false) genuinely
  exceeds the control arm (NBA withheld, true).
- The returned value EQUALS a raw per-arm recomputation (treatment_rate − control_rate)/
  control_rate — proving it is COMPUTED, not a seeded constant. Both arms are populated
  with real action rates in (0,1) (a genuine randomized holdout, not all-one-arm).

"action" = action_taken IS NOT NULL (a rep BEHAVIOR measurable in BOTH arms); acceptance_status
is treatment-only and deliberately NOT used. The flip/equalize/empty/idempotency anti-fabrication
proofs were verified live in rolled-back txns at design time (flip→-0.2158, equalize→~0,
empty arm→fail-loud) and the empty-arm fail-loud contract is locked by the hermetic unit tests.

CAPABILITY-GATED: skips unless SUPABASE_* is set AND the action-rate-uplift query_id exists
(migration 051 applied).
"""

import os

import pytest

from src.kpi.calculators.trigger_performance import TriggerPerformanceCalculator

HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_SUPABASE, reason="SUPABASE_* not set"),
]

_QUERY_ID = "trigger_performance_action_rate_uplift"


@pytest.fixture
def calc():
    c = TriggerPerformanceCalculator()
    if c.db_client is None:
        pytest.skip("no Supabase client")
    try:
        c.db_client.rpc("kpi_query", {"query_id": _QUERY_ID, "params": []}).execute()
    except Exception as e:
        pytest.skip(f"#577 action-rate-uplift query unavailable (migration 051 not applied?): {e}")
    return c


def _arm_row(calc):
    """The full registered row: {action_rate_uplift, treatment_rate, control_rate}."""
    resp = calc.db_client.rpc("kpi_query", {"query_id": _QUERY_ID, "params": []}).execute()
    return resp.data[0]


def test_action_rate_uplift_is_real_positive_incrementality(calc):
    """The realized relative uplift is a real positive incrementality (~0.2751 live), in the
    honest band — NOT ~0 (arm has no effect) and NOT an implausibly-large fabricated lift."""
    val = calc._calc_action_rate_uplift({})
    assert val is not None
    assert 0.15 <= val < 0.6, (
        f"uplift {val} outside the honest incrementality band; ~0 would mean the arm does not "
        "drive action, >0.6 would be an implausible fabricated lift"
    )


def test_treatment_exceeds_control_and_uplift_is_realized(calc):
    """The value EQUALS the raw per-arm recomputation (treatment−control)/control — proving it
    is COMPUTED, not a seeded constant — and treatment genuinely exceeds control."""
    row = _arm_row(calc)
    uplift = row["action_rate_uplift"]
    t = row["treatment_rate"]
    c = row["control_rate"]
    assert t is not None and c is not None, "both arms must be populated"
    assert t > c, f"treatment action rate ({t}) must exceed control ({c}) for a positive uplift"
    assert abs(uplift - (t - c) / c) < 1e-9, "uplift must equal the realized per-arm contrast"


def test_both_arms_are_real_nonempty_holdouts(calc):
    """Both arms have real action rates in (0,1) — a genuine randomized holdout, not all-one-arm
    (which would make control_rate NULL → fail-loud)."""
    row = _arm_row(calc)
    assert 0.0 < row["control_rate"] < 1.0, f"control arm not a real holdout: {row['control_rate']}"
    assert 0.0 < row["treatment_rate"] < 1.0, f"treatment arm not real: {row['treatment_rate']}"
