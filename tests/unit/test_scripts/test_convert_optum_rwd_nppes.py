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
    """When the cache loader returns an NppesRecord, the eight non-PII
    enrichment fields populate from it. ``first_name`` / ``last_name`` are
    intentionally NOT exported even when the cache has them — codex PR #162
    post-merge MEDIUM-3 (PII scope discipline)."""
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
    # PII intentionally NOT propagated to the cohort output (MEDIUM-3).
    assert p["first_name"] is None
    assert p["last_name"] is None
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


# --------------------------------------------------------------------------- #
# PR #162 codex post-merge — remediation regressions                           #
# --------------------------------------------------------------------------- #


def test_build_hcp_profiles_does_not_export_first_or_last_name_even_when_cache_has_them():
    """Regression for codex PR #162 post-merge MEDIUM-3 (PII scope).

    The documented 8-field NPPES enrichment contract does NOT include
    named provider PII. Even when the NPPES cache record carries
    first_name / last_name (as it does for real Type-1 providers), the
    converter MUST keep those fields None at the cohort output boundary.
    """
    c = _make_converter()
    _seed_provider(c, {"OBF_A": "207K00000X"})
    _seed_claims(c, {"OBF_A": [1]})

    def loader(npi):
        return NppesRecord(
            npi=npi,
            entity_type="1",
            first_name="Jane",
            last_name="Doe",
            taxonomies=(NppesTaxonomy(code="207K00000X", primary=True),),
        )

    rwdc.set_npi_cache_loader(loader)
    profiles = c._build_hcp_profiles(kept_patids={1})
    assert len(profiles) == 1
    # Even though the loader returned "Jane" / "Doe", the output stays None.
    assert profiles[0]["first_name"] is None
    assert profiles[0]["last_name"] is None


def test_build_hcp_profiles_real_npi_input_used_as_cache_key_unmodified():
    """Regression for codex PR #162 post-merge MEDIUM-2 (real-NPI lookup).

    When the input is already a valid 10-digit NPI (real-NPI cohort), the
    cache loader must be queried with that exact value — NOT a re-hashed
    Luhn NPI from `generate_luhn_npi(real_npi)`. Without this, real-NPI
    cohorts silently fail to enrich even when the loader is registered.
    """
    real_npi = "1234567893"  # known Luhn-valid
    c = _make_converter()
    _seed_provider(c, {real_npi: "207K00000X"})
    _seed_claims(c, {real_npi: [1]})

    queried_keys: list[str] = []

    def loader(npi):
        queried_keys.append(npi)
        return NppesRecord(
            npi=npi,
            entity_type="1",
            parent_organization_legal_name="Acme Health",
            taxonomies=(NppesTaxonomy(code="207K00000X", primary=True),),
        )

    rwdc.set_npi_cache_loader(loader)
    profiles = c._build_hcp_profiles(kept_patids={1})
    assert len(profiles) == 1
    assert real_npi in queried_keys, (
        f"loader was queried with {queried_keys!r} instead of the real NPI {real_npi!r}"
    )
    # The output NPI must ALSO preserve the real value (not re-hashed).
    assert profiles[0]["npi"] == real_npi
    # Enrichment fields populated from the loader hit.
    assert profiles[0]["affiliation_primary"] == "Acme Health"


def test_build_hcp_profiles_obfuscated_npi_still_uses_generated_for_lookup():
    """Regression for MEDIUM-2 — symmetric case: obfuscated cohort input
    (not a 10-digit number) gets `generate_luhn_npi()` applied and the
    cache lookup uses the GENERATED Luhn NPI, preserving the contract for
    obfuscated cohorts (which is what current Optum + CSU look like)."""
    c = _make_converter()
    _seed_provider(c, {"OBF_FOO": "207K00000X"})
    _seed_claims(c, {"OBF_FOO": [1]})

    queried_keys: list[str] = []

    def loader(npi):
        queried_keys.append(npi)
        return None  # cache miss is fine for this assertion

    rwdc.set_npi_cache_loader(loader)
    profiles = c._build_hcp_profiles(kept_patids={1})
    expected = rwdc.generate_luhn_npi("OBF_FOO")
    assert len(profiles) == 1
    assert queried_keys == [expected]
    assert profiles[0]["npi"] == expected


def test_nucc_pcp_codes_include_active_family_medicine_subcodes():
    """Regression for codex PR #162 post-merge MEDIUM-1 — PCP set must
    include all active 207Q* Family Medicine sub-codes per NUCC 24.0 +
    CMS PCP definition (Internal Medicine + General Practice + Pediatrics
    + Primary Care Clinic)."""
    required_family_medicine = {
        "207Q00000X",  # Family Medicine (parent)
        "207QA0000X",  # Adolescent Medicine
        "207QA0401X",  # Addiction Medicine
        "207QA0505X",  # Adult Medicine
        "207QB0002X",  # Obesity Medicine
        "207QG0300X",  # Geriatric Medicine
        "207QH0002X",  # Hospice and Palliative Care
        "207QS0010X",  # Sports Medicine
        "207QS1201X",  # Sleep Medicine
    }
    missing = required_family_medicine - set(rwdc.NUCC_PCP_CODES)
    assert missing == set(), f"NUCC_PCP_CODES missing active 207Q sub-codes: {missing}"
    # And CMS PCP categories beyond Family Medicine
    assert "207R00000X" in rwdc.NUCC_PCP_CODES  # Internal Medicine
    assert "208000000X" in rwdc.NUCC_PCP_CODES  # Pediatrics
    assert "208D00000X" in rwdc.NUCC_PCP_CODES  # General Practice


def test_nppes_entity_type_constants_defined():
    """Regression for codex PR #162 post-merge LOW-1 — named constants
    replace `"1"` / `"2"` magic strings."""
    assert rwdc.NPPES_ENTITY_TYPE_INDIVIDUAL == "1"
    assert rwdc.NPPES_ENTITY_TYPE_ORGANIZATION == "2"


def test_is_valid_npi_recognizes_real_npi_and_rejects_obfuscated():
    """Regression for MEDIUM-2 — the public is_valid_npi helper that
    `_build_hcp_profiles` uses to gate the lookup-key branch."""
    assert rwdc.is_valid_npi("1234567893") is True  # 10-digit
    assert rwdc.is_valid_npi("OBF_FOO") is False
    assert rwdc.is_valid_npi("") is False
    assert rwdc.is_valid_npi(None) is False
    assert rwdc.is_valid_npi("12345") is False  # too short
    assert rwdc.is_valid_npi("12345678901") is False  # too long
    # Backwards-compat alias still works
    assert rwdc._is_valid_npi("1234567893") is True


def test_is_real_cms_npi_passes_real_npi_fails_generated_obfuscated():
    """Regression for codex PR #165 pass-1 MEDIUM: the strict CMS-NPI Luhn
    (80840-prefix variant) must DISTINGUISH real CMS NPIs from generated
    obfuscated ones, even when both are 10-digit Luhn-numeric.

    Real NPI ``1234567893`` is CMS-Luhn-valid (verified manually). Any
    output of ``generate_luhn_npi`` is plain-Luhn valid but FAILS CMS-NPI
    Luhn because plain Luhn doesn't include the 80840 prefix → the check
    digit lands on a different residue."""
    # Real CMS-style NPI (Luhn-valid against 80840 prefix)
    assert rwdc.is_real_cms_npi("1234567893") is True

    # Generated obfuscated NPIs (deterministic Luhn-valid w/o 80840 prefix)
    # MUST fail the strict check. Try 10 obfuscated inputs — none should
    # accidentally pass.
    for obf in [f"OBF_{i:04d}" for i in range(10)]:
        generated = rwdc.generate_luhn_npi(obf)
        assert rwdc.is_valid_npi(generated) is True, (
            f"generated NPI {generated!r} should pass syntactic check"
        )
        assert rwdc.is_real_cms_npi(generated) is False, (
            f"generated obfuscated NPI {generated!r} accidentally passed "
            f"CMS-Luhn — the 80840-prefix variant is not discriminating "
            f"correctly"
        )


def test_is_real_cms_npi_rejects_malformed():
    """Strict Luhn-NPI rejects anything not 10 digits or not Luhn-valid."""
    assert rwdc.is_real_cms_npi("") is False
    assert rwdc.is_real_cms_npi(None) is False
    assert rwdc.is_real_cms_npi("12345") is False  # too short
    assert rwdc.is_real_cms_npi("12345678901") is False  # too long
    assert rwdc.is_real_cms_npi("123456789A") is False  # non-digit
    assert rwdc.is_real_cms_npi("1234567894") is False  # wrong check digit


def test_build_hcp_profiles_coincidental_10_digit_obfuscated_routes_to_hashing():
    """Regression for codex PR #165 pass-1 MEDIUM: an obfuscated key that
    happens to be 10 digits but fails CMS-NPI Luhn MUST route to the
    hashing branch — not the real-NPI lookup branch."""
    # 10-digit numeric string with WRONG CMS-Luhn checksum.
    obf_10digit = "1234567894"  # last digit deliberately wrong vs 1234567893
    assert rwdc.is_valid_npi(obf_10digit) is True  # syntactic passes
    assert rwdc.is_real_cms_npi(obf_10digit) is False  # strict fails

    c = _make_converter()
    _seed_provider(c, {obf_10digit: "207K00000X"})
    _seed_claims(c, {obf_10digit: [1]})

    queried_keys: list[str] = []

    def loader(npi):
        queried_keys.append(npi)
        return None

    rwdc.set_npi_cache_loader(loader)
    profiles = c._build_hcp_profiles(kept_patids={1})
    expected_generated = rwdc.generate_luhn_npi(obf_10digit)
    # Must have been hashed: lookup_key == generated, not raw obf_10digit.
    assert queried_keys == [expected_generated]
    assert profiles[0]["npi"] == expected_generated
