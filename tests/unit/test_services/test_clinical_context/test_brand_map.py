"""The brand->drug/disease/fallback map joins the clinical_codes SSOT with the
enrichment-only static fallbacks. Verifies it stays consistent with the SSOT."""

import pytest

from src.services.clinical_context.brand_map import (
    BRAND_CLINICAL_MAP,
    BrandClinicalProfile,
    resolve_brand_profile,
    endpoint_mapping_for_outcome,
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
