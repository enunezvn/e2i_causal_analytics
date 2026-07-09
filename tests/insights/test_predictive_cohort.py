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


# ---------------------------------------------------------------------------
# Entity-kind facets + what-if insight (2026-07-09 predictive-page clarity):
# targets labeled patients/prescribers, honest absent-drivers wording, and a
# per-row what-if grounding with an explicit predictive-not-causal fallback.
# ---------------------------------------------------------------------------

from src.insights.predictive_cohort import (  # noqa: E402
    build_whatif_grounding,
    cohort_facets_for_model,
    generate_whatif_insight,
)


def test_cohort_facets_patient_hcp_and_unknown():
    assert cohort_facets_for_model("persistence_remibrutinib_goldstd_lr_v1") == (
        "patients",
        "staying on therapy at 180 days",
    )
    assert cohort_facets_for_model("hcp_adoption_kisqali_goldstd_lr_v1") == (
        "prescribers (HCPs)",
        "adopting the brand (intent to prescribe)",
    )
    assert cohort_facets_for_model("mystery_model") == (None, None)


def test_build_grounding_names_entity_kind_and_outcome():
    g = build_grounding(
        "initiation_fabhalta_goldstd_lr_v1",
        4239,
        0.47,
        [{"entity_id": "scvpt_1", "probability": 0.98}],
        [],
    )
    assert "4239 patients scored" in g["distribution_summary"]
    assert "of starting treatment" in g["distribution_summary"]


def test_build_grounding_absent_drivers_points_to_drill_down_not_none():
    g = build_grounding("initiation_fabhalta_goldstd_lr_v1", 10, 0.5, [], [])
    # The LM must never read an empty driver list as "the model has no feature
    # importances" — per-target SHAP always exists in the drill-down.
    assert g["drivers_summary"] != "none"
    assert "not computed at cohort level" in g["drivers_summary"]
    assert "drill-down" in g["drivers_summary"]


def test_build_whatif_grounding_summarizes_profile_result_and_drivers():
    g = build_whatif_grounding(
        "persistence_remibrutinib_goldstd_lr_v1",
        {"disease_severity": 5.6, "academic_hcp": 0},
        probability=0.87,
        confidence=0.87,
        cohort_mean=0.45,
        n_scored=4847,
        top_drivers=[{"feature": "disease_severity", "importance": -1.21}],
    )
    assert g["profile_summary"].startswith("hypothetical patient:")
    # Inputs are sorted for cache-key stability.
    assert "academic_hcp=0; disease_severity=5.6" in g["profile_summary"]
    assert (
        g["result_summary"] == "predicted probability 0.87 of staying on therapy at 180 days "
        "vs cohort mean 0.45 across 4847 scored patients; model confidence 0.87"
    )
    assert "disease_severity (-1.21)" in g["drivers_summary"]
    assert any(c["label"] == "Cohort mean" and c["value"] == "0.45" for c in g["grounding"])


def test_build_whatif_grounding_hcp_singular_and_missing_bits_honest():
    g = build_whatif_grounding(
        "hcp_adoption_kisqali_goldstd_lr_v1",
        {"years_experience": 20},
        probability=0.6,
        confidence=None,
        cohort_mean=None,
        n_scored=None,
        top_drivers=[],
    )
    assert g["profile_summary"].startswith("hypothetical prescriber (HCP):")
    assert "cohort mean" not in g["result_summary"]
    assert "SHAP unavailable" in g["drivers_summary"]


def test_generate_whatif_insight_fallback_is_grounded_and_not_causal():
    g = build_whatif_grounding(
        "initiation_fabhalta_goldstd_lr_v1",
        {"disease_severity": 8},
        probability=0.91,
        confidence=0.9,
        cohort_mean=0.6,
        n_scored=1234,
        top_drivers=[{"feature": "disease_severity", "importance": 0.4}],
    )
    out = generate_whatif_insight(g)
    assert out["is_fallback"] is True
    assert "0.91" in out["insight"] and "disease_severity" in out["insight"]
    assert "not a causal estimate" in out["insight"]
