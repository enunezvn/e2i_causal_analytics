"""Digital-twin strategic insight: grounded fallback path (no live LLM).

The conftest forces ``ensure_dspy_configured -> False`` so ``generate_insight``
exercises the deterministic factual fallback computed from the REAL grounding —
never fabricated content.
"""

from src.digital_twin.effect.provider import INTERVENTION_CATALOG
from src.insights.digital_twin import build_grounding, generate_insight

_MODELS = [
    {"model_name": "hcp_Remibrutinib_twin", "model_id": "m-1"},
]

_SIMULATIONS = [
    {
        "simulation_id": "s-1",
        "simulation_status": "completed",
        "recommendation": "deploy",
        "intervention_type": "digital_engagement",
        "simulated_ate": 0.2312,
        "simulated_ci_lower": 0.1978,
        "simulated_ci_upper": 0.2647,
        "data_provenance": "cohort_estimated_synthetic_gold_v1",
    },
    {
        "simulation_id": "s-2",
        "simulation_status": "completed",
        "recommendation": "refine",
        "intervention_type": "email_campaign",
        "simulated_ate": 0.031,
        "simulated_ci_lower": -0.002,
        "simulated_ci_upper": 0.064,
        "data_provenance": "cohort_estimated_synthetic_gold_v1",
    },
    {"simulation_id": "s-3", "simulation_status": "running", "recommendation": None},
]

_ALL_AVAILABLE = {value: True for value, _ in INTERVENTION_CATALOG}


def test_build_grounding_summarizes_real_rows():
    g = build_grounding("Remibrutinib", _MODELS, _SIMULATIONS, _ALL_AVAILABLE, INTERVENTION_CATALOG)
    assert g["scope"] == "Remibrutinib"
    assert "1 active twin model(s)" in g["model_summary"]
    assert "hcp_Remibrutinib_twin" in g["model_summary"]
    # 3 recorded, 2 completed, deploy rate 50%.
    assert "3 simulation(s)" in g["simulation_summary"]
    assert "2 completed" in g["simulation_summary"]
    assert "deploy rate 50%" in g["simulation_summary"]
    # Latest completed = first row (repo returns newest-first).
    assert "digital_engagement" in g["latest_result"]
    assert "+0.231" in g["latest_result"]
    assert "cohort_estimated_synthetic_gold_v1" in g["latest_result"]
    # Full coverage: 8/8 identified.
    assert (
        f"{len(INTERVENTION_CATALOG)} of {len(INTERVENTION_CATALOG)}"
        in (g["intervention_coverage"])
    )
    chips = {c["label"]: c["value"] for c in g["grounding"]}
    assert chips["Twin models"] == "1"
    assert chips["Simulations"] == "3"
    assert chips["Deploy rate"] == "50%"
    assert chips["Identified interventions"] == "8/8"


def test_build_grounding_partial_coverage_names_missing_channels():
    partial = dict(_ALL_AVAILABLE)
    partial["rep_training_quality"] = False
    g = build_grounding("Kisqali", _MODELS, [], partial, INTERVENTION_CATALOG)
    assert "7 of 8" in g["intervention_coverage"]
    assert "Not yet identified: Rep Training Quality." in g["intervention_coverage"]
    assert g["latest_result"] == "No completed simulation yet for this brand."


def test_build_grounding_empty_program_is_honest():
    g = build_grounding("Fabhalta", [], [], {}, INTERVENTION_CATALOG)
    assert "No active twin model" in g["model_summary"]
    assert "0 simulation(s)" in g["simulation_summary"]
    chips = {c["label"]: c["value"] for c in g["grounding"]}
    assert chips["Identified interventions"] == "0/8"


def test_generate_insight_fallback_is_grounded_and_flagged():
    g = build_grounding("Remibrutinib", _MODELS, _SIMULATIONS, _ALL_AVAILABLE, INTERVENTION_CATALOG)
    result = generate_insight(g)
    assert result["is_fallback"] is True
    # The fallback narrative is composed of the real grounded strings.
    assert "Remibrutinib" in result["insight"]
    assert "deploy rate 50%" in result["insight"]
    # Honesty-critical: the synthetic substrate is disclosed.
    assert "synthetic-gold" in result["insight"]
    assert result["grounding"] == g["grounding"]
    assert result["key_takeaways"]
