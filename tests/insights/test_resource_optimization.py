from src.insights.resource_optimization import _fallback, build_grounding, generate_insight


def _grounding(**overrides):
    kwargs = {
        "objective": "maximize_roi",
        "brand": "Remibrutinib",
        "resource_type": "budget",
        "solver_status": "optimal",
        "entity_count": 40,
        "total_budget": 8_500_000.0,
        "projected_lift_pct": 2.4,
        "top_increases": [
            {"entity_id": "northeast-T07", "change_percentage": 45.3, "change": 78131.0},
            {"entity_id": "south-T01", "change_percentage": 8.1, "change": 22268.0},
        ],
        "top_decreases": [
            {"entity_id": "northeast-T08", "change_percentage": -18.2, "change": -30833.0},
        ],
        "synthetic": True,
        "optimization_summary": "Optimization complete. Projected outcome: 4095 "
        "(outcome lift vs current: +0.2%).",
        "recommendations": [
            "Increase northeast-T07 by 78,131",
            "Reduce northeast-T08 by 30,833",
        ],
    }
    kwargs.update(overrides)
    return build_grounding(**kwargs)


def test_build_grounding_derives_moves_outcome_and_chips():
    g = _grounding()
    assert "northeast-T07 +45.3%" in g["moves"]
    assert "northeast-T08 -18.2%" in g["moves"]
    assert "+2.4%" in g["outcome"]
    assert "$8,500,000" in g["outcome"]
    assert "SYNTHETIC" in g["caveats"]
    assert any(c["label"] == "Brand" and c["value"] == "Remibrutinib" for c in g["grounding"])
    assert any(c["label"] == "Projected lift" and c["value"] == "+2.4%" for c in g["grounding"])


def test_build_grounding_all_brands_and_caller_targets():
    g = _grounding(brand=None, synthetic=False)
    assert "All brands" in g["scope"]
    assert "SYNTHETIC" not in g["caveats"]
    assert any(c["label"] == "Brand" and c["value"] == "All brands" for c in g["grounding"])


def test_generate_insight_fallback_surfaces_agent_output():
    # No LM configured in the test env -> deterministic factual fallback that
    # surfaces the agent's own summary + recommendations, never fabrication.
    g = _grounding()
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "Optimization complete" in out["insight"]
    assert "+2.4%" in out["insight"]
    assert out["key_takeaways"][0].startswith("Increase northeast-T07")


def test_fallback_without_summary_asks_for_a_run():
    g = _grounding(optimization_summary="", recommendations=[])
    out = _fallback(g)
    assert out["is_fallback"] is True
    assert "run an optimization" in out["insight"].lower()


def test_underspend_is_narrated_not_hidden():
    # maximize_roi can intentionally deploy less than the budget (marginal
    # return below the hurdle); the outcome must say so instead of claiming
    # the full budget is "under optimization".
    g = _grounding(total_budget=300.0, total_spend=250.0)
    assert "deploying $250" in g["outcome"]
    assert "$50 intentionally unallocated" in g["outcome"]
    assert "total budget under optimization" not in g["outcome"]
    assert any(c["label"] == "Deployed" and c["value"] == "$250" for c in g["grounding"])


def test_zero_spend_is_narrated_not_treated_as_missing():
    # A genuine $0 recommendation over a nonzero budget is the most extreme
    # underspend — it must not be confused with "no spend info sent".
    g = _grounding(total_budget=300.0, total_spend=0.0)
    assert "deploying $0 of the $300 budget" in g["outcome"]
    assert "total budget under optimization" not in g["outcome"]
    assert any(c["label"] == "Deployed" and c["value"] == "$0" for c in g["grounding"])


def test_minimize_cost_underspend_is_savings_not_hurdle():
    # minimize_cost underspends BY DESIGN — narrating it with maximize_roi's
    # hurdle-rate rationale would be the wrong business story (codex round 2).
    g = _grounding(objective="minimize_cost", total_budget=300.0, total_spend=250.0)
    assert "deploying $250" in g["outcome"]
    assert "$50 in savings while preserving the current outcome level" in g["outcome"]
    assert "hurdle" not in g["outcome"]
    assert "intentionally unallocated" not in g["outcome"]
    assert any(c["label"] == "Deployed" and c["value"] == "$250" for c in g["grounding"])


def test_other_objective_underspend_is_neutral():
    # No invented rationale for objectives without a designed underspend story.
    g = _grounding(objective="maximize_outcome", total_budget=300.0, total_spend=250.0)
    assert "deploying $250" in g["outcome"]
    assert "$50 left unallocated" in g["outcome"]
    assert "hurdle" not in g["outcome"]
    assert "savings" not in g["outcome"]


def test_full_deployment_keeps_budget_phrase():
    g = _grounding(total_budget=300.0, total_spend=300.0)
    assert "total budget under optimization: $300" in g["outcome"]
    assert not any(c["label"] == "Deployed" for c in g["grounding"])


def test_no_spend_info_keeps_budget_phrase():
    # Callers that don't send total_spend (older clients) keep the old text.
    g = _grounding()
    assert "total budget under optimization" in g["outcome"]
    assert not any(c["label"] == "Deployed" for c in g["grounding"])


# ---- Causal-registry context (commercial grain, 2026-07-07) ---------------------
CAUSAL_DRIVERS = [
    {
        "start": "rep_detailing_frequency",
        "end": "trx_volume",
        "effect": 0.2977,
        "confidence": 0.87,
        "synthetic": True,
    },
    {
        "start": "copay_support_program",
        "end": "roi",
        "effect": 0.11,
        "confidence": 0.8,
        "synthetic": True,
    },
]


def test_build_grounding_carries_causal_context_and_chip():
    """Registry-modeled drivers ground the WHY behind allocation moves: the
    causal_context string carries humanized driver chains + provenance, and a
    grounding chip discloses how many modeled chains inform the read."""
    g = _grounding(causal_drivers=CAUSAL_DRIVERS)
    assert "rep detailing frequency" in g["causal_context"]
    assert "TRx volume" in g["causal_context"]
    assert "synthetic" in g["causal_context"].lower()
    assert any(c["label"] == "Modeled drivers" and "2" in c["value"] for c in g["grounding"])


def test_build_grounding_without_drivers_says_none():
    g = _grounding()  # no causal_drivers argument at all
    assert "no modeled causal drivers" in g["causal_context"].lower()
    assert not any(c["label"] == "Modeled drivers" for c in g["grounding"])


def test_fallback_appends_causal_context_when_present():
    g = _grounding(causal_drivers=CAUSAL_DRIVERS)
    out = generate_insight(g)  # LM off in tests -> deterministic fallback
    assert out["is_fallback"] is True
    assert "rep detailing frequency" in out["insight"]


def test_fallback_stays_silent_without_drivers():
    g = _grounding()
    out = generate_insight(g)
    assert "causal" not in out["insight"].lower()
