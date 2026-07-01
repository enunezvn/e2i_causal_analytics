from src.insights.causal_discovery import build_grounding, generate_insight

EFFECTS = [
    {
        "treatment": "copay_card",
        "outcome": "adherence_180d",
        "ate": 0.043,
        "ate_ci_lower": 0.02,
        "ate_ci_upper": 0.066,
        "status": "proceed",
        "selected_estimator": "CausalForestDML",
    },
    {
        "treatment": "nurse_call",
        "outcome": "adherence_180d",
        "ate": 0.011,
        "ate_ci_lower": -0.01,
        "ate_ci_upper": 0.03,
        "status": "review",
        "selected_estimator": "LinearDML",
    },
]


def test_build_grounding_ranks_and_counts_gates():
    g = build_grounding("Kisqali", "patient", EFFECTS)
    assert "proceed" in g["gate_summary"] and "review" in g["gate_summary"]
    assert any(c["label"] == "Effects" and c["value"] == "2" for c in g["grounding"])


def test_generate_insight_fallback_grounded():
    g = build_grounding("Kisqali", "patient", EFFECTS)
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "copay_card" in out["insight"]
