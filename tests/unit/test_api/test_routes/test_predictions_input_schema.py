"""Curated, brand/cohort-appropriate model input schema (predictive-analytics).

Regression: the page derived its Input Features form from each model's raw
encoded `feature_columns`, so the Kisqali model (breast cancer / oncology)
offered CSU specialties (dermatology, allergy_immunology, ...). The backend must
expose a curated `input_fields` schema with the cohort's real covariates and
BRAND-appropriate categorical choices (per hcp_generator.BRAND_SPECIALTY_DIST).
"""

from src.api.routes.predictions import build_curated_input_fields


def _by_name(fields):
    return {f["name"]: f for f in fields}


def test_kisqali_hcp_adoption_specialty_is_oncology_only():
    fields = build_curated_input_fields("hcp_adoption_kisqali_goldstd_lr_v1")
    assert fields is not None
    # Features = the canonical HCP-adoption covariates, in order.
    assert [f["name"] for f in fields] == [
        "peer_influence_score",
        "influence_network_size",
        "years_experience",
        "specialty",
        "geographic_region",
    ]
    by = _by_name(fields)
    # Kisqali is HR+/HER2- breast cancer -> ONCOLOGY, not CSU specialties.
    assert by["specialty"]["type"] == "category"
    assert by["specialty"]["choices"] == ["oncology"]
    assert "dermatology" not in by["specialty"]["choices"]
    assert "allergy_immunology" not in by["specialty"]["choices"]
    # geographic_region is a brand-agnostic categorical with the 4 US regions.
    assert by["geographic_region"]["type"] == "category"
    assert set(by["geographic_region"]["choices"]) == {
        "northeast",
        "south",
        "midwest",
        "west",
    }
    # Continuous covariates stay numeric.
    assert by["peer_influence_score"]["type"] == "number"
    assert by["years_experience"]["type"] == "number"


def test_remibrutinib_hcp_adoption_specialty_is_csu():
    fields = build_curated_input_fields("hcp_adoption_remibrutinib_goldstd_lr_v1")
    by = _by_name(fields)
    assert set(by["specialty"]["choices"]) == {
        "dermatology",
        "allergy_immunology",
        "rheumatology",
    }


def test_fabhalta_hcp_adoption_specialty_is_pnh():
    fields = build_curated_input_fields("hcp_adoption_fabhalta_goldstd_lr_v1")
    by = _by_name(fields)
    assert set(by["specialty"]["choices"]) == {
        "hematology",
        "internal_medicine",
        "neurology",
    }


def test_patient_cohort_features_have_no_specialty():
    fields = build_curated_input_fields("persistence_kisqali_goldstd_lr_v1")
    assert [f["name"] for f in fields] == [
        "disease_severity",
        "academic_hcp",
        "geographic_region",
    ]
    by = _by_name(fields)
    assert "specialty" not in by
    assert by["disease_severity"]["type"] == "number"
    assert by["geographic_region"]["type"] == "category"


def test_unknown_model_returns_none():
    assert build_curated_input_fields("totally_unknown_model") is None
    assert (
        build_curated_input_fields("hcp_adoption_notabrand_goldstd_lr_v1") is None
    )
