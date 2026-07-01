from src.insights.predictive_cohort import build_grounding, generate_insight


def test_build_grounding_summarizes_distribution_and_drivers():
    g = build_grounding(
        model_version="csu_adherence_v3",
        n_scored=250,
        mean_prob=0.34,
        top_targets=[
            {"entity_id": "HCP7", "probability": 0.91},
            {"entity_id": "HCP3", "probability": 0.88},
        ],
        top_drivers=[
            {"feature": "prior_adherence", "importance": 0.4},
            {"feature": "copay", "importance": 0.25},
        ],
    )
    assert any(c["label"] == "Scored" and c["value"] == "250" for c in g["grounding"])
    assert "prior_adherence" in g["drivers_summary"]


def test_generate_insight_fallback_grounded():
    g = build_grounding(
        "m1",
        250,
        0.34,
        [{"entity_id": "HCP7", "probability": 0.91}],
        [{"feature": "prior_adherence", "importance": 0.4}],
    )
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "HCP7" in out["insight"] and "250" in out["insight"]
