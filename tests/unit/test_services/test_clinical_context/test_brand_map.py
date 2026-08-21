"""The brand->drug/disease/fallback map joins the clinical_codes SSOT with the
enrichment-only static fallbacks. Verifies it stays consistent with the SSOT."""

import pytest

from src.services.clinical_context.brand_map import (
    BRAND_CLINICAL_MAP,
    BrandClinicalProfile,
    analysis_framing_sentence,
    compose_rwe_search_term,
    endpoint_mapping_for_outcome,
    resolve_brand_profile,
    treatment_context_for,
)


@pytest.mark.unit
def test_three_brands_resolved_from_ssot():
    assert set(BRAND_CLINICAL_MAP) == {"Kisqali", "Remibrutinib", "Fabhalta"}
    kis = resolve_brand_profile("Kisqali")
    assert isinstance(kis, BrandClinicalProfile)
    assert kis.drug_name == "ribociclib"
    assert kis.disease == "Malignant neoplasm of breast"
    # Static MoA fallback the spec pins (used when ChEMBL is down).
    assert kis.moa_fallback == "CDK4/6 inhibitor"


@pytest.mark.unit
def test_each_brand_carries_static_endpoint_and_rwe_fallbacks():
    fab = resolve_brand_profile("Fabhalta")
    assert fab.drug_name == "iptacopan"
    assert fab.disease == "Paroxysmal nocturnal hemoglobinuria"
    assert fab.moa_fallback == "complement Factor B inhibitor"
    # PNH pivotal endpoints (verified live 2026-06-19 on ClinicalTrials.gov v2).
    assert any("transfusion" in e.lower() for e in fab.pivotal_endpoints_fallback)
    assert any("LDH" in e or "hemoglobin" in e.lower() for e in fab.pivotal_endpoints_fallback)
    # A non-empty PubMed search term so the RWE provider has something to query.
    assert fab.rwe_search_term


@pytest.mark.unit
def test_remibrutinib_btk_csu():
    rem = resolve_brand_profile("Remibrutinib")
    assert rem.drug_name == "remibrutinib"
    assert rem.disease == "Chronic spontaneous urticaria"
    assert rem.moa_fallback == "BTK inhibitor"
    assert any("UAS7" in e for e in rem.pivotal_endpoints_fallback)


@pytest.mark.unit
def test_outcome_to_real_endpoint_mapping_is_brand_aware():
    # Our synthetic 'persistent_180d' maps to a real retention/persistence framing.
    m = endpoint_mapping_for_outcome("Kisqali", "persistent_180d")
    assert m is not None
    assert "persist" in m.lower() or "treatment-free" in m.lower() or "duration" in m.lower()
    # A synthetic outcome with no curated mapping returns None (honest — not faked).
    assert endpoint_mapping_for_outcome("Kisqali", "made_up_outcome") is None


@pytest.mark.unit
def test_unknown_brand_raises_keyerror():
    with pytest.raises(KeyError):
        resolve_brand_profile("NotABrand")


# --- Task 2: therapy-label + curated-competitor fields ---


@pytest.mark.unit
def test_competitor_map_resolves_by_disease_key():
    """competitor_map[profile.disease.lower()] must be a non-empty list for each brand."""
    for brand in ("Kisqali", "Fabhalta", "Remibrutinib"):
        profile = resolve_brand_profile(brand)
        key = profile.disease.lower()
        competitors = profile.competitor_map.get(key)
        assert competitors, (
            f"{brand}: competitor_map[{key!r}] is empty or missing; "
            f"available keys: {list(profile.competitor_map)}"
        )


@pytest.mark.unit
def test_kisqali_indications_fallback_non_empty():
    profile = resolve_brand_profile("Kisqali")
    assert len(profile.indications_fallback) >= 1
    assert any("breast cancer" in ind.lower() for ind in profile.indications_fallback)


@pytest.mark.unit
def test_remibrutinib_limitations_fallback():
    profile = resolve_brand_profile("Remibrutinib")
    assert profile.limitations_fallback == "Not indicated for other forms of urticaria."


@pytest.mark.unit
def test_fabhalta_boxed_warning_non_empty():
    profile = resolve_brand_profile("Fabhalta")
    assert profile.boxed_warning_fallback
    assert "encapsulated" in profile.boxed_warning_fallback.lower()


@pytest.mark.unit
def test_new_fields_default_empty_for_unknown_pattern():
    """Verify the dataclass fields have sensible defaults (no construction breakage)."""
    # We cannot call the private constructor directly for a frozen dataclass with
    # required positional fields, so we verify via the existing resolved profiles that
    # Kisqali (no LoU / no boxed warning) has None for those optional fields.
    kis = resolve_brand_profile("Kisqali")
    assert kis.limitations_fallback is None
    assert kis.boxed_warning_fallback is None


# --- #1763: analysis-aware framing (treatment side + composed literature term) ---


@pytest.mark.unit
def test_each_profile_carries_a_plain_language_disease_search_term():
    """The SSOT disease description is a clinical-coding string ('Malignant neoplasm
    of breast'); literature search needs the plain-language term clinicians publish
    under."""
    assert resolve_brand_profile("Kisqali").disease_search_term == "breast cancer"
    assert (
        resolve_brand_profile("Remibrutinib").disease_search_term == "chronic spontaneous urticaria"
    )
    assert (
        resolve_brand_profile("Fabhalta").disease_search_term
        == "paroxysmal nocturnal hemoglobinuria"
    )


@pytest.mark.unit
def test_treatment_context_frames_the_drug_therapy_arm_per_brand():
    tc = treatment_context_for("Kisqali", "treatment_arm")
    assert tc is not None
    assert tc.column == "treatment_arm"
    assert tc.kind == "drug_therapy"
    # Brand-faithful: the framing names THIS brand's molecule, not a generic "treatment".
    assert "ribociclib" in tc.framing.lower()
    assert "iptacopan" in treatment_context_for("Fabhalta", "treatment_arm").framing.lower()


@pytest.mark.unit
def test_commercial_levers_are_labelled_commercial_not_clinical():
    """copay/PSP/detailing/sampling/NBA are access-and-promotion levers. Labelling
    them honestly is what lets the evidence layer refuse to invent clinical
    evidence for the treatment side."""
    for column in (
        "copay_support",
        "psp_enrolled",
        "rep_detailing_high",
        "sample_dropped",
        "trigger_accepted",
    ):
        tc = treatment_context_for("Kisqali", column)
        assert tc is not None, column
        assert tc.kind == "commercial", column


@pytest.mark.unit
def test_brand_distinct_treatments_resolve_only_on_their_own_brand():
    # #1321 per-brand axes: each is a treatment ONLY for the brand it was planted for.
    assert treatment_context_for("Fabhalta", "complement_inhibitor_status") is not None
    assert treatment_context_for("Kisqali", "complement_inhibitor_status") is None
    assert treatment_context_for("Kisqali", "disease_stage") is not None
    assert treatment_context_for("Remibrutinib", "disease_stage") is None
    assert treatment_context_for("Remibrutinib", "urticaria_severity_uas7") is not None
    assert treatment_context_for("Fabhalta", "urticaria_severity_uas7") is None


@pytest.mark.unit
def test_unmapped_treatment_or_brand_returns_none_never_fabricated():
    assert treatment_context_for("Kisqali", "made_up_treatment") is None
    assert treatment_context_for("NotABrand", "treatment_arm") is None


@pytest.mark.unit
def test_composed_rwe_term_carries_drug_disease_outcome_and_treatment_themes():
    profile = resolve_brand_profile("Kisqali")
    term = compose_rwe_search_term(profile, "persistent_180d", "copay_support").lower()
    assert "ribociclib" in term
    assert "breast cancer" in term
    assert "persistence" in term
    assert "copay" in term


@pytest.mark.unit
def test_composed_rwe_term_varies_with_the_analysis():
    profile = resolve_brand_profile("Kisqali")
    persistence = compose_rwe_search_term(profile, "persistent_180d", "treatment_arm")
    discontinuation = compose_rwe_search_term(profile, "discontinued_180d", "treatment_arm")
    assert persistence != discontinuation


@pytest.mark.unit
def test_composed_rwe_term_falls_back_to_the_curated_brand_term():
    """Nothing to compose from => the curated brand term, NOT a half-built query."""
    profile = resolve_brand_profile("Fabhalta")
    assert compose_rwe_search_term(profile, "made_up_outcome", None) == profile.rwe_search_term
    # A lever with no clinical-literature theme (rep detailing) adds no term of its own.
    assert (
        compose_rwe_search_term(profile, "made_up_outcome", "rep_detailing_high")
        == profile.rwe_search_term
    )


@pytest.mark.unit
def test_analysis_framing_sentence_names_treatment_outcome_drug_and_disease():
    profile = resolve_brand_profile("Kisqali")
    sentence = analysis_framing_sentence(profile, "persistent_180d", "treatment_arm")
    assert sentence is not None
    assert sentence.startswith("This analysis estimates the effect of ")
    low = sentence.lower()
    assert "ribociclib" in low
    assert "180-day treatment persistence" in low
    assert "malignant neoplasm of breast" in low


@pytest.mark.unit
def test_analysis_framing_is_none_without_a_curated_treatment():
    """No treatment (brand-level view) or an unmapped one => no sentence at all,
    rather than a sentence with a hole in it."""
    profile = resolve_brand_profile("Kisqali")
    assert analysis_framing_sentence(profile, "persistent_180d", None) is None
    assert analysis_framing_sentence(profile, "persistent_180d", "made_up_treatment") is None


@pytest.mark.unit
def test_fabhalta_indications_fallback_covers_every_labelled_indication():
    """codex LOW, confirmed against the LIVE openFDA label 2026-08-21: FABHALTA is
    indicated for PNH, IgAN and C3G. The curated fallback is what an analyst sees
    when openFDA is unreachable, so a missing indication is a silently incomplete
    label in exactly the moment we cannot check."""
    profile = resolve_brand_profile("Fabhalta")
    joined = " | ".join(profile.indications_fallback).lower()
    assert "paroxysmal nocturnal hemoglobinuria" in joined
    assert "iga nephropathy" in joined
    assert "c3g" in joined or "complement 3 glomerulopathy" in joined
