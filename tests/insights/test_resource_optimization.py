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
