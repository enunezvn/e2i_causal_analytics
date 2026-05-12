"""Unit tests for issue #154 NPPES rewires in scripts/convert_optum_rwd.py.

Covers:
  * §7.7 provider-mix sharpening — full-taxonomy-code matching replaces
    the legacy 4-char prefix matching.
  * §3 HCP-profile NPPES enrichment — gated on cache-loader registration;
    obfuscated-NPI cohorts (default Optum case) yield unchanged behavior.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from scripts import rwd_common as rwdc
from scripts.convert_optum_rwd import OptumDataConverter
from scripts.rwd_common import (
    NppesAddress,
    NppesRecord,
    NppesTaxonomy,
)


@pytest.fixture(autouse=True)
def _clear_loader():
    rwdc.set_npi_cache_loader(None)
    yield
    rwdc.set_npi_cache_loader(None)


def _make_converter() -> OptumDataConverter:
    return OptumDataConverter(parquet_dir=".", output_dir=".", cohorts=("initiation",))


# --------------------------------------------------------------------------- #
# §7.7 — taxonomy_in primitive                                                 #
# --------------------------------------------------------------------------- #


def test_taxonomy_in_matches_exact_code_not_4char_prefix():
    """The legacy code used `tax.startswith("207K")` which would match any
    future "207K1xxxxX" subspecialty. taxonomy_in must NOT match unless
    the code is explicitly in the allowed list."""
    # "207K00000X" IS in the allergy list → True
    assert rwdc.taxonomy_in("207K00000X", rwdc.NUCC_ALLERGY_IMMUNOLOGY_CODES) is True
    # Hypothetical future code with same 4-char prefix but not in list:
    assert rwdc.taxonomy_in("207KX9999X", rwdc.NUCC_ALLERGY_IMMUNOLOGY_CODES) is False


def test_taxonomy_in_dermatology_includes_known_subspecialties():
    """Pediatric dermatology, dermatopathology, MOHS should all classify as
    dermatology under the new exact-match scheme."""
    for code in ("207N00000X", "207NP0225X", "207ND0900X", "207ND0101X"):
        assert rwdc.taxonomy_in(code, rwdc.NUCC_DERMATOLOGY_CODES) is True


def test_taxonomy_in_handles_case_and_whitespace():
    assert rwdc.taxonomy_in("  207k00000x  ", rwdc.NUCC_ALLERGY_IMMUNOLOGY_CODES) is True
    assert rwdc.taxonomy_in("207K00000X", rwdc.NUCC_ALLERGY_IMMUNOLOGY_CODES) is True


def test_taxonomy_in_rejects_none_and_non_strings():
    assert rwdc.taxonomy_in(None, rwdc.NUCC_ALLERGY_IMMUNOLOGY_CODES) is False
    assert rwdc.taxonomy_in("", rwdc.NUCC_ALLERGY_IMMUNOLOGY_CODES) is False
    assert rwdc.taxonomy_in(12345, rwdc.NUCC_ALLERGY_IMMUNOLOGY_CODES) is False  # type: ignore[arg-type]


def test_taxonomy_in_returns_false_on_unknown_code():
    assert rwdc.taxonomy_in("999X00000X", rwdc.NUCC_ALLERGY_IMMUNOLOGY_CODES) is False


# --------------------------------------------------------------------------- #
# §3 — _build_hcp_profiles without NPPES enrichment (obfuscated cohort)        #
# --------------------------------------------------------------------------- #


def _seed_provider(c: OptumDataConverter, obf_to_tax: dict[str, str]) -> None:
    """Wire the obfuscated-NPI → taxonomy mapping directly (mirrors what
    _prepare_data does after reading the parquet)."""
    c._provider_by_npi = dict(obf_to_tax)


def _seed_claims(c: OptumDataConverter, npi_to_rx: dict[str, list[int]]) -> None:
    """Build minimal med + proc frames so _build_hcp_profiles has rows.

    npi_to_rx: {obfuscated_npi: [patid_1, patid_2, ...]} — each list entry
    becomes one medication row attributed to that NPI.
    """
    med_rows = []
    for npi, patids in npi_to_rx.items():
        for pid in patids:
            med_rows.append(
                {"npi": npi, "patid": pid, "medication_date": pd.Timestamp("2023-01-01")}
            )
    c.med = pd.DataFrame(med_rows)
    c.proc = pd.DataFrame(columns=["npi", "patid", "proc_date"])
    c.now_iso = "2024-01-01T00:00:00"


def test_build_hcp_profiles_uses_exact_taxonomy_for_specialty_bucketing():
    """With NO cache loader registered, the specialty field still upgrades
    from 4-char prefix to full-taxonomy matching."""
    c = _make_converter()
    _seed_provider(c, {"OBF_A": "207K00000X", "OBF_B": "207NP0225X", "OBF_C": "999X00000X"})
    _seed_claims(c, {"OBF_A": [1, 2, 3], "OBF_B": [4, 5], "OBF_C": [6]})

    profiles = c._build_hcp_profiles(kept_patids={1, 2, 3, 4, 5, 6})
    by_specialty = {p["specialty"] for p in profiles}
    assert "Allergy/Immunology" in by_specialty
    assert "Dermatology" in by_specialty  # Pediatric Dermatology subspec → Dermatology
    assert "Other" in by_specialty


def test_build_hcp_profiles_returns_none_fields_when_lookup_misses():
    """Without a cache loader, lookup_npi returns None and all 8 enriched
    fields stay None (the structural reality on obfuscated-NPI cohorts)."""
    c = _make_converter()
    _seed_provider(c, {"OBF_A": "207K00000X"})
    _seed_claims(c, {"OBF_A": [1, 2, 3]})

    profiles = c._build_hcp_profiles(kept_patids={1, 2, 3})
    assert len(profiles) == 1
    p = profiles[0]
    assert p["sub_specialty"] is None
    assert p["practice_size"] is None
    assert p["geographic_region"] is None
    assert p["state"] is None
    assert p["city"] is None
    assert p["zip_code"] is None
    assert p["years_experience"] is None
    assert p["affiliation_primary"] is None


# --------------------------------------------------------------------------- #
# §3 — _build_hcp_profiles WITH NPPES enrichment                               #
# --------------------------------------------------------------------------- #


def test_build_hcp_profiles_populates_eight_fields_when_lookup_hits():
    """When the cache loader returns an NppesRecord, all eight currently-
    None fields populate from it. This is the acceptance criterion §3."""
    c = _make_converter()
    _seed_provider(c, {"OBF_A": "207K00000X"})
    _seed_claims(c, {"OBF_A": [1, 2, 3]})

    # Pre-compute the generated Luhn NPI so the loader matches on the
    # converter's lookup key (the converter looks up by generated NPI,
    # not obfuscated key).
    generated = rwdc.generate_luhn_npi("OBF_A")
    rec = NppesRecord(
        npi=generated,
        entity_type="1",
        enumeration_date=date(2010, 1, 1),
        taxonomies=(NppesTaxonomy(code="207K00000X", desc="Allergy & Immunology", primary=True),),
        practice_address=NppesAddress(city="Boston", state="MA", postal_code="02101"),
        parent_organization_legal_name="Big Health System",
        sole_proprietor=False,
        first_name="Jane",
        last_name="Doe",
    )
    rwdc.set_npi_cache_loader(lambda npi: rec if npi == generated else None)

    profiles = c._build_hcp_profiles(kept_patids={1, 2, 3})
    assert len(profiles) == 1
    p = profiles[0]
    assert p["sub_specialty"] == "Allergy & Immunology"
    assert p["state"] == "MA"
    assert p["city"] == "Boston"
    assert p["zip_code"] == "02101"
    assert p["geographic_region"] == "northeast"
    assert p["affiliation_primary"] == "Big Health System"
    assert p["first_name"] == "Jane"
    assert p["last_name"] == "Doe"
    # years_experience derived from enumeration_date=2010 → at least ~13
    assert p["years_experience"] is not None and p["years_experience"] >= 13


def test_build_hcp_profiles_practice_size_solo_for_sole_proprietor():
    c = _make_converter()
    _seed_provider(c, {"OBF_A": "207N00000X"})
    _seed_claims(c, {"OBF_A": [1]})

    generated = rwdc.generate_luhn_npi("OBF_A")
    rec = NppesRecord(npi=generated, entity_type="1", sole_proprietor=True)
    rwdc.set_npi_cache_loader(lambda npi: rec if npi == generated else None)

    profiles = c._build_hcp_profiles(kept_patids={1})
    assert profiles[0]["practice_size"] == "Solo"


def test_build_hcp_profiles_practice_size_group_for_organization():
    c = _make_converter()
    _seed_provider(c, {"OBF_A": "282N00000X"})
    # Need >50 patients so practice heuristic also returns "Group"; the
    # NPPES entity_type=2 then refines practice_size to "Group" too.
    _seed_claims(c, {"OBF_A": list(range(1, 60))})

    generated = rwdc.generate_luhn_npi("OBF_A")
    rec = NppesRecord(npi=generated, entity_type="2", sole_proprietor=False)
    rwdc.set_npi_cache_loader(lambda npi: rec if npi == generated else None)

    profiles = c._build_hcp_profiles(kept_patids=set(range(1, 60)))
    assert profiles[0]["practice_size"] == "Group"


def test_build_hcp_profiles_does_not_hit_live_api(monkeypatch):
    """Acceptance: converter must use cache-only lookup. If the loader
    misses, the converter must NOT trigger an HTTP request even when
    the env-var-driven default would otherwise enable it."""
    monkeypatch.setenv("NPPES_API_FALLBACK", "1")
    # Loader that always misses.
    rwdc.set_npi_cache_loader(lambda _n: None)

    # Sentinel: if anyone calls _fetch_nppes_via_api_sync, we explode.
    def explode(*_a, **_kw):
        raise AssertionError("Converter must not hit the live API")

    monkeypatch.setattr(rwdc, "_fetch_nppes_via_api_sync", explode)

    c = _make_converter()
    _seed_provider(c, {"OBF_A": "207K00000X"})
    _seed_claims(c, {"OBF_A": [1, 2, 3]})

    # Must complete without exploding.
    profiles = c._build_hcp_profiles(kept_patids={1, 2, 3})
    assert len(profiles) == 1
