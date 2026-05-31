"""#577 causal trio faithful e2e (PR1): CM-003 (causal_impact) computes the RIGHT
value against the LIVE DB over the discovered causal_paths cohort — not just "runs
without raising".

These assert MEANING (the #574 lesson):
- CM-003 is the path-level mean causal_effect_size over discovered paths: a real
  fraction in (0,1) with a per-start_node descriptive breakdown and the anti-relabel
  code-anchor (start_node is a discovered path SOURCE, NOT an intervention target).
- The validation_status filter actually narrows the cohort and MOVES the value (a
  constant/fabricated metric would not), and an impossible filter fails loud (None +
  error) without mutating any data.

CM-004 (counterfactual) and CM-005 (mediation) are intentionally NOT covered here:
their source columns are independent uniform noise, so they remain fail-loud pending
a generator-coherence rework (PR2/PR3 of the causal trio).

CAPABILITY-GATED: skips unless SUPABASE_* is set AND the CM-003 query_id exists
(migration 047 applied).
"""

import os

import pytest

from src.kpi.calculators.causal_metrics import CausalMetricsCalculator

HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_SUPABASE, reason="SUPABASE_* not set"),
]


@pytest.fixture
def calc():
    c = CausalMetricsCalculator()
    if c.db_client is None:
        pytest.skip("no Supabase client")
    try:
        c.db_client.rpc(
            "kpi_query", {"query_id": "causal_metrics_causal_impact", "params": [""]}
        ).execute()
    except Exception as e:
        pytest.skip(f"#577 causal_impact query unavailable (migration 047 not applied?): {e}")
    return c


def test_cm003_causal_impact_is_real_discovered_effect_aggregate(calc):
    """CM-003 = path-level mean causal_effect_size over discovered paths: a real
    fraction in (0,1) with a per-start_node breakdown + the anti-relabel note."""
    out = calc._calc_causal_impact({})
    assert out["value"] is not None
    assert 0.0 < out["value"] < 1.0, f"causal impact out of range: {out['value']}"
    md = out["metadata"]
    assert md["n_paths"] > 0
    assert md["breakdown"], "expected a start_node breakdown"
    # start_node is the discovered path SOURCE, NOT an intervention target (#574).
    assert "intervention target" in md.get("note", "").lower()


def test_cm003_validation_filter_discriminates(calc):
    """The validation_status filter narrows the cohort and CHANGES the value — a
    constant/fabricated metric would not move. all-paths != validated-only."""
    all_paths = calc._calc_causal_impact({})
    validated = calc._calc_causal_impact({"validation_status": "validated"})
    assert all_paths["value"] is not None
    assert validated["value"] is not None
    assert validated["metadata"]["n_paths"] < all_paths["metadata"]["n_paths"]
    assert all_paths["value"] != validated["value"], "validation filter did not move the value"


def test_cm003_returns_none_on_empty_cohort(calc):
    """An impossible validation_status yields no paths -> value None + error (fail-loud,
    never a fabricated 0.0) — and mutates nothing."""
    out = calc._calc_causal_impact({"validation_status": "__no_such_status__"})
    assert out["value"] is None
    assert "error" in out["metadata"]
