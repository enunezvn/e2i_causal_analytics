from src.insights.resource_optimization import to_insight


def test_to_insight_surfaces_existing_summary():
    out = to_insight(
        optimization_summary="Reallocating to high-ROI HCPs lifts projected outcome 6%.",
        recommendations=["Shift 12% budget to segment A", "Hold segment C"],
        projected_lift_pct=6.0,
        solver_status="optimal",
    )
    assert out["is_fallback"] is False
    assert "high-ROI" in out["insight"]
    assert out["key_takeaways"][0].startswith("Shift 12%")
    assert any(c["label"] == "Projected lift" for c in out["grounding"])


def test_to_insight_empty_summary_is_fallback():
    out = to_insight(
        optimization_summary="", recommendations=[], projected_lift_pct=None,
        solver_status="infeasible",
    )
    assert out["is_fallback"] is True
