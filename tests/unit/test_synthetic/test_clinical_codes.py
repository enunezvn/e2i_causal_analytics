from src.ml.synthetic.clinical_codes import (
    BRAND_NDC,
    brand_codes,
)

EXACT_BRANDS = {"Remibrutinib", "Kisqali", "Fabhalta"}  # INDEX §SHARED CONTRACTS


def test_brand_ndc_covers_all_three_brands_with_drug_name():
    assert set(BRAND_NDC) == EXACT_BRANDS
    for brand, entry in BRAND_NDC.items():
        assert entry["drug_name"], f"{brand} missing drug_name"
        assert entry["ndc"].count("-") == 2, f"{brand} NDC not 5-4-2 format"


def test_brand_codes_resolver_is_indication_correct():
    remi = brand_codes("Remibrutinib")
    assert remi["icd10"][0] == "L50.1"  # CSU primary
    assert remi["drug_class"] == "BTK Inhibitor"
    assert remi["drug_name"] == "remibrutinib"

    kis = brand_codes("Kisqali")
    assert kis["icd10"][0] == "C50.1"  # breast
    assert kis["drug_class"] == "CDK4/6 Inhibitor"
    assert kis["ndc"].startswith("00078-0903")  # ribociclib labeler

    fab = brand_codes("Fabhalta")
    assert fab["icd10"] == ["D59.5"]  # PNH
    assert fab["drug_class"] == "Complement Inhibitor"
    assert fab["drug_name"] == "iptacopan"


def test_brand_codes_rejects_unknown_brand():
    import pytest

    with pytest.raises(KeyError):
        brand_codes("Tylenol")
