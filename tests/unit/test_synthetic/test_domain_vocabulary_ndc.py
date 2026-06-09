from pathlib import Path

import yaml


def _load_ndc():
    text = Path("config/domain_vocabulary.yaml").read_text()
    return yaml.safe_load(text)["brand_ndc_codes"]["mappings"]


def test_all_three_brands_have_ndc_entries():
    m = _load_ndc()
    assert {"Kisqali", "Remibrutinib", "Fabhalta"} <= set(m)
    assert m["Remibrutinib"]["drug_name"] == "remibrutinib"
    assert m["Fabhalta"]["drug_name"] == "iptacopan"
    for brand in ("Remibrutinib", "Fabhalta"):
        codes = m[brand]["ndc_codes"]
        assert codes, f"{brand} has no ndc_codes"
        assert all(c.count("-") == 2 for c in codes), f"{brand} NDC not 5-4-2"
