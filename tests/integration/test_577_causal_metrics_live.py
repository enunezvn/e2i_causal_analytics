"""#577 causal trio faithful e2e (PR1 + PR2): CM-003 (causal_impact) and CM-004
(counterfactual) compute the RIGHT value against the LIVE DB — not just "run without
raising".

These assert MEANING (the #574 lesson):
- CM-003 is the path-level mean causal_effect_size over discovered paths: a real
  fraction in (0,1) with a per-start_node descriptive breakdown and the anti-relabel
  code-anchor (start_node is a discovered path SOURCE, NOT an intervention target). The
  validation_status filter narrows the cohort and MOVES the value; an impossible filter
  fails loud (None + error) without mutating data.
- CM-004 is the counterfactual outcome LEVEL E[Y(a')] over the coherent ml_predictions
  subset, where counterfactual_outcome = max(0, factual − treatment effect). A per-row
  coherence proof shows the do-contrast is real (the prior independent uniform noise
  would fail it); mean_realized_contrast is the true floor-attenuated contrast; the
  prediction_type filter discriminates; an impossible filter fails loud.

CM-005 (mediation) is intentionally NOT covered here: causal_chain edges still don't
reconcile with causal_effect_size, so it remains fail-loud pending a further
generator-coherence rework (PR3 of the causal trio).

CAPABILITY-GATED: each metric's tests gate on their OWN query_id (CM-003 = migration
047, CM-004 = migration 048); skips if SUPABASE_* unset or the migration isn't applied.
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


# --- #577 PR2: CM-004 counterfactual (coherent do-contrast) ------------------------------


@pytest.fixture
def cf_calc():
    c = CausalMetricsCalculator()
    if c.db_client is None:
        pytest.skip("no Supabase client")
    try:
        c.db_client.rpc(
            "kpi_query", {"query_id": "causal_metrics_counterfactual", "params": [""]}
        ).execute()
    except Exception as e:
        pytest.skip(f"#577 counterfactual query unavailable (migration 048 not applied?): {e}")
    return c


def test_cm004_counterfactual_level_below_factual(cf_calc):
    """CM-004 = counterfactual outcome LEVEL E[Y(a')]: a real fraction in (0,1) strictly
    below the factual mean — because counterfactual = factual − a positive treatment
    effect. Independent-noise counterfactual would sit at ~the factual mean."""
    out = cf_calc._calc_counterfactual({})
    assert out["value"] is not None
    assert 0.0 < out["value"] < 1.0, f"counterfactual level out of range: {out['value']}"
    md = out["metadata"]
    assert md["n"] > 0
    assert md["mean_effect"] > 0
    assert out["value"] < md["mean_factual"], "counterfactual level should sit below factual"
    # mean_realized_contrast is exactly the aggregate factual − counterfactual ...
    assert md["mean_realized_contrast"] == pytest.approx(
        md["mean_factual"] - out["value"], abs=1e-6
    )
    # ... it is a real positive contrast, floor-attenuated so it does NOT exceed the
    # nominal mean treatment effect, yet stays close to it (only ~16% of rows clamp).
    assert md["mean_realized_contrast"] > 0
    assert md["mean_realized_contrast"] <= md["mean_effect"] + 1e-6
    assert md["mean_realized_contrast"] == pytest.approx(md["mean_effect"], abs=0.05)


def test_cm004_counterfactual_is_coherent_do_contrast(cf_calc):
    """Decisive anti-fabrication proof: for EVERY coherent row, counterfactual_outcome is
    the floored factual − treatment-effect contrast (within rounding). The prior
    independent uniform(0.2,0.8) noise would miss this by ~0.3 on ~every row."""
    resp = (
        cf_calc.db_client.table("ml_predictions")
        .select("prediction_value,treatment_effect_estimate,counterfactual_outcome")
        .not_.is_("counterfactual_outcome", "null")
        .limit(2000)
        .execute()
    )
    rows = resp.data
    assert rows, "expected coherent counterfactual rows"
    checked = 0
    for r in rows:
        if r["treatment_effect_estimate"] is None or r["prediction_value"] is None:
            continue
        pred = float(r["prediction_value"])
        tee = float(r["treatment_effect_estimate"])
        cf = float(r["counterfactual_outcome"])
        expected = max(0.0, round(pred - tee, 3))
        # tolerance covers the Python/Postgres half-rounding convention diff (<=1 unit at
        # scale 3); decisively rejects the old noise (which was ~0.3 off per row).
        assert abs(cf - expected) <= 0.0016, f"incoherent counterfactual row: {r}"
        checked += 1
    assert checked > 0


def test_cm004_prediction_type_discriminates(cf_calc):
    """The prediction_type filter narrows the cohort and CHANGES the value (a constant
    would not). churn-only != all-types, and the churn cohort is smaller."""
    all_types = cf_calc._calc_counterfactual({})
    churn = cf_calc._calc_counterfactual({"prediction_type": "churn"})
    assert all_types["value"] is not None
    assert churn["value"] is not None
    assert churn["metadata"]["n"] < all_types["metadata"]["n"]
    assert churn["value"] != all_types["value"]


def test_cm004_returns_none_on_empty_cohort(cf_calc):
    """An impossible prediction_type yields no rows -> value None + error (fail-loud)."""
    out = cf_calc._calc_counterfactual({"prediction_type": "__no_such_type__"})
    assert out["value"] is None
    assert "error" in out["metadata"]
