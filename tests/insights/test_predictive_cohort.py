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


# ---------------------------------------------------------------------------
# Registry context (2026-07-07 follow-up): outcome-matched drivers derived
# from the gold-standard model name (<cohort>_<brand>_goldstd_lr_v1).
# ---------------------------------------------------------------------------

from src.insights.predictive_cohort import outcome_terms_for_model

DRIVERS = [
    {
        "start": "copay_support_program",
        "end": "persistent_180d",
        "effect": 0.18,
        "confidence": 0.82,
        "synthetic": True,
    },
]


def test_outcome_terms_for_model_parses_cohort_and_brand():
    assert outcome_terms_for_model("persistence_kisqali_goldstd_lr_v1") == (
        "Kisqali",
        ("persistence",),
    )
    assert outcome_terms_for_model("initiation_remibrutinib_goldstd_lr_v1") == (
        "Remibrutinib",
        ("initiation",),
    )
    assert outcome_terms_for_model("discontinuation_fabhalta_goldstd_lr_v1") == (
        "Fabhalta",
        ("discontinuation",),
    )
    # HCP adoption feeds prescriber-side outcomes: intent + new-to-brand volume.
    assert outcome_terms_for_model("hcp_adoption_fabhalta_goldstd_lr_v1") == (
        "Fabhalta",
        ("intent to prescribe", "NBRx"),
    )


def test_outcome_terms_for_model_unknown_is_empty_not_guessy():
    # An unrecognizable model name must yield NO terms (honest empty context),
    # never a generic commercial fetch pretending relevance.
    assert outcome_terms_for_model("csu_adherence_v3") == (None, ())


def test_build_grounding_carries_digit_free_registry_context_and_chip():
    g = build_grounding(
        "persistence_kisqali_goldstd_lr_v1",
        250,
        0.34,
        [{"entity_id": "HCP7", "probability": 0.91}],
        [{"feature": "prior_adherence", "importance": 0.4}],
        causal_drivers=DRIVERS,
    )
    assert "copay support program → patient persistence" in g["registry_context"]
    assert not any(ch.isnumeric() for ch in g["registry_context"])
    assert any(c["label"] == "Registry chains" and c["value"] == "1" for c in g["grounding"])


def test_fallback_appends_registry_line_only_when_present():
    args = (
        "persistence_kisqali_goldstd_lr_v1",
        250,
        0.34,
        [{"entity_id": "HCP7", "probability": 0.91}],
        [{"feature": "prior_adherence", "importance": 0.4}],
    )
    out = generate_insight(build_grounding(*args, causal_drivers=DRIVERS))
    assert out["is_fallback"] is True
    assert "Registry-modeled causal chains" in out["insight"]
    bare = generate_insight(build_grounding(*args))
    assert "Registry-modeled" not in bare["insight"]
