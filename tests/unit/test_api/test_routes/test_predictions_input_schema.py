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
    # T9: persistence/discontinuation cohorts carry 7 leakage-safe covariates, plus
    # copay_support as of COMM-ARMS Phase 1. Order is asserted, not just membership:
    # this list drives the RENDERED field order on the what-if form.
    assert [f["name"] for f in fields] == [
        "disease_severity",
        "academic_hcp",
        "geographic_region",
        "insurance_type",
        "age_at_diagnosis",
        "comorbidity_burden",
        "prior_therapy_lines",
        "copay_support",
        "psp_enrolled",
    ]
    by = _by_name(fields)
    assert "specialty" not in by
    assert by["disease_severity"]["type"] == "number"
    assert by["geographic_region"]["type"] == "category"
    # insurance_type is a categorical access gradient → dropdown, not a number input.
    assert by["insurance_type"]["type"] == "category"
    assert set(by["insurance_type"]["choices"]) == {"commercial", "medicare", "medicaid"}
    assert by["comorbidity_burden"]["type"] == "number"
    assert by["prior_therapy_lines"]["type"] == "number"
    # copay_support / psp_enrolled are 0/1 intervention flags: they must be BOUNDED, or
    # the form offers a free numeric input on which a user can enter a value the model
    # never saw.
    assert by["copay_support"]["type"] == "number"
    assert (by["copay_support"]["min"], by["copay_support"]["max"]) == (0, 1)
    assert by["psp_enrolled"]["type"] == "number"
    assert (by["psp_enrolled"]["min"], by["psp_enrolled"]["max"]) == (0, 1)


def test_unknown_model_returns_none():
    assert build_curated_input_fields("totally_unknown_model") is None
    assert build_curated_input_fields("hcp_adoption_notabrand_goldstd_lr_v1") is None


def test_patient_numeric_fields_carry_dgp_grounded_guidance():
    """Every numeric input tells the user what values make sense (min/max/step/hint).

    Bounds mirror the DGP draws: severity Normal(5,2) clip 0-10, age 18-85,
    comorbidity Poisson clip 0-5, prior lines 0-3, academic_hcp binary.
    """
    by = _by_name(build_curated_input_fields("persistence_kisqali_goldstd_lr_v1"))
    assert (by["disease_severity"]["min"], by["disease_severity"]["max"]) == (0, 10)
    assert by["disease_severity"]["step"] == 0.1
    assert (by["age_at_diagnosis"]["min"], by["age_at_diagnosis"]["max"]) == (18, 85)
    assert (by["comorbidity_burden"]["min"], by["comorbidity_burden"]["max"]) == (0, 5)
    assert (by["prior_therapy_lines"]["min"], by["prior_therapy_lines"]["max"]) == (0, 3)
    assert (by["academic_hcp"]["min"], by["academic_hcp"]["max"]) == (0, 1)
    assert "academic" in by["academic_hcp"]["hint"]
    for name in (
        "disease_severity",
        "academic_hcp",
        "age_at_diagnosis",
        "comorbidity_burden",
        "prior_therapy_lines",
    ):
        assert by[name]["hint"], f"{name} must carry a user-facing hint"


def test_hcp_numeric_fields_carry_guidance_lognormals_unbounded_above():
    """years_experience is hard-bounded (2-40); the two influence covariates are
    log-normal in the DGP so they get a floor + observed-range hint but NO hard
    max (a what-if may deliberately extrapolate)."""
    by = _by_name(build_curated_input_fields("hcp_adoption_kisqali_goldstd_lr_v1"))
    assert (by["years_experience"]["min"], by["years_experience"]["max"]) == (2, 40)
    assert by["peer_influence_score"]["min"] == 0
    assert "max" not in by["peer_influence_score"]
    assert by["influence_network_size"]["min"] == 0
    assert "max" not in by["influence_network_size"]
    for name in ("peer_influence_score", "influence_network_size", "years_experience"):
        assert by[name]["hint"], f"{name} must carry a user-facing hint"
