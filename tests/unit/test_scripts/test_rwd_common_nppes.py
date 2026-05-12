"""Unit tests for the NPPES NPI taxonomy helper in scripts.rwd_common.

Issue #154 PR-0 scope: helper API + cache loader hook + live-API fallback
parser. Production DB writes are tested in tests/unit/test_tasks/.
"""

from __future__ import annotations

from datetime import date

import pytest

from scripts import rwd_common as rwdc
from scripts.rwd_common import (
    NppesRecord,
    bulk_lookup_npis,
    get_npi_cache_loader,
    lookup_npi,
    parse_nppes_api_result,
    set_npi_cache_loader,
)


@pytest.fixture(autouse=True)
def _clear_loader():
    """Each test starts with no registered loader so they can't pollute
    each other."""
    set_npi_cache_loader(None)
    yield
    set_npi_cache_loader(None)


# --------------------------------------------------------------------------- #
# parse_nppes_api_result                                                       #
# --------------------------------------------------------------------------- #


def _api_fixture(*, npi: str = "1234567893") -> dict:
    """Minimal but realistic NPPES API response shape (one result)."""
    return {
        "enumeration_type": "NPI-1",
        "basic": {
            "first_name": "Jane",
            "last_name": "Doe",
            "enumeration_date": "06/15/2010",
            "last_updated": "01/02/2024",
            "sole_proprietor": "NO",
            "parent_organization_legal_name": "Big Health System",
            "organization_name": "",
        },
        "taxonomies": [
            {
                "code": "207K00000X",
                "desc": "Allergy & Immunology",
                "primary": True,
                "license": "MD123",
                "state": "CA",
            },
            {
                "code": "207N00000X",
                "desc": "Dermatology",
                "primary": False,
                "license": "",
                "state": "CA",
            },
        ],
        "addresses": [
            {
                "address_purpose": "MAILING",
                "address_1": "PO Box 1",
                "city": "Mailtown",
                "state": "CA",
                "postal_code": "90000",
                "country_code": "US",
            },
            {
                "address_purpose": "LOCATION",
                "address_1": "100 Clinic Way",
                "city": "Clinictown",
                "state": "CA",
                "postal_code": "90210",
                "country_code": "US",
            },
        ],
    }


def test_parse_api_result_extracts_core_fields():
    rec = parse_nppes_api_result(_api_fixture(), "1234567893")
    assert rec is not None
    assert rec.npi == "1234567893"
    assert rec.entity_type == "1"  # NPI-1 → "1"
    assert rec.first_name == "Jane"
    assert rec.last_name == "Doe"
    assert rec.sole_proprietor is False
    assert rec.parent_organization_legal_name == "Big Health System"
    assert rec.organization_legal_name is None  # empty string normalized
    assert rec.enumeration_date == date(2010, 6, 15)
    assert rec.last_updated_nppes == date(2024, 1, 2)
    assert rec.source == "api_fallback"


def test_parse_api_result_picks_location_address_over_mailing():
    rec = parse_nppes_api_result(_api_fixture(), "1234567893")
    assert rec is not None
    assert rec.practice_address is not None
    assert rec.practice_address.address_1 == "100 Clinic Way"
    assert rec.practice_address.city == "Clinictown"


def test_parse_api_result_taxonomies_marked_primary():
    rec = parse_nppes_api_result(_api_fixture(), "1234567893")
    assert rec is not None
    assert len(rec.taxonomies) == 2
    assert rec.taxonomies[0].code == "207K00000X"
    assert rec.taxonomies[0].primary is True
    assert rec.taxonomies[1].primary is False
    assert rec.primary_taxonomy is not None
    assert rec.primary_taxonomy.code == "207K00000X"


def test_parse_api_result_falls_back_to_first_taxonomy_if_none_primary():
    payload = _api_fixture()
    for t in payload["taxonomies"]:
        t["primary"] = False
    rec = parse_nppes_api_result(payload, "1234567893")
    assert rec is not None
    assert rec.primary_taxonomy is not None
    assert rec.primary_taxonomy.code == "207K00000X"


def test_parse_api_result_handles_missing_taxonomies_and_addresses():
    payload = {"enumeration_type": "NPI-2", "basic": {}, "taxonomies": [], "addresses": []}
    rec = parse_nppes_api_result(payload, "9876543210")
    assert rec is not None
    assert rec.entity_type == "2"
    assert rec.taxonomies == ()
    assert rec.practice_address is None
    assert rec.primary_taxonomy is None


def test_parse_api_result_drops_taxonomies_with_empty_code():
    payload = _api_fixture()
    payload["taxonomies"].append({"code": "", "primary": False, "desc": "ghost"})
    rec = parse_nppes_api_result(payload, "1234567893")
    assert rec is not None
    # The 3rd dummy entry with empty code is dropped.
    assert len(rec.taxonomies) == 2


def test_parse_api_result_returns_none_for_non_mapping_input():
    assert parse_nppes_api_result("garbage", "1234567893") is None  # type: ignore[arg-type]


def test_parse_api_result_sole_proprietor_yes_variants():
    payload = _api_fixture()
    payload["basic"]["sole_proprietor"] = "YES"
    rec = parse_nppes_api_result(payload, "1234567893")
    assert rec is not None
    assert rec.sole_proprietor is True

    payload["basic"]["sole_proprietor"] = "Y"
    rec = parse_nppes_api_result(payload, "1234567893")
    assert rec is not None
    assert rec.sole_proprietor is True

    payload["basic"]["sole_proprietor"] = "NO"
    rec = parse_nppes_api_result(payload, "1234567893")
    assert rec is not None
    assert rec.sole_proprietor is False


def test_parse_api_result_sole_proprietor_unknown_maps_to_none():
    """Empty / whitespace / unrecognized sole-proprietor value must map to
    None (unknown), not False. Otherwise downstream `practice_size` /
    `academic_hcp` derivation conflates 'unknown' with 'not sole prop'."""
    for sentinel in ("", "   ", "UNKNOWN", "?", None):
        payload = _api_fixture()
        payload["basic"]["sole_proprietor"] = sentinel
        rec = parse_nppes_api_result(payload, "1234567893")
        assert rec is not None
        assert rec.sole_proprietor is None, f"expected None for {sentinel!r}"


# --------------------------------------------------------------------------- #
# NppesRecord helpers                                                          #
# --------------------------------------------------------------------------- #


def test_years_since_enumeration_handles_missing_date():
    rec = NppesRecord(npi="1234567893")
    assert rec.years_since_enumeration() is None


def test_years_since_enumeration_basic_arithmetic():
    rec = NppesRecord(npi="1234567893", enumeration_date=date(2010, 1, 1))
    assert rec.years_since_enumeration(today=date(2025, 1, 1)) == 15


def test_years_since_enumeration_returns_none_for_future_date():
    rec = NppesRecord(npi="1234567893", enumeration_date=date(3000, 1, 1))
    assert rec.years_since_enumeration(today=date(2025, 1, 1)) is None


# --------------------------------------------------------------------------- #
# lookup_npi + cache loader                                                    #
# --------------------------------------------------------------------------- #


def test_lookup_npi_rejects_malformed_input():
    assert lookup_npi("not-an-npi") is None
    assert lookup_npi("123") is None
    assert lookup_npi("") is None
    assert lookup_npi("12345abcde") is None


def test_lookup_npi_returns_cached_record_without_hitting_api():
    rec = NppesRecord(npi="1234567893", source="bulk_dump")

    calls: list[str] = []

    def loader(npi: str):
        calls.append(npi)
        return rec

    set_npi_cache_loader(loader)
    out = lookup_npi("1234567893", use_api_fallback=False)
    assert out is rec
    assert calls == ["1234567893"]


def test_lookup_npi_falls_back_to_none_when_cache_misses_and_api_disabled():
    set_npi_cache_loader(lambda _npi: None)
    assert lookup_npi("1234567893", use_api_fallback=False) is None


def test_set_and_get_loader_roundtrip():
    sentinel = object()
    set_npi_cache_loader(sentinel)
    assert get_npi_cache_loader() is sentinel


def test_lookup_npi_swallows_loader_exceptions(caplog):
    def bad_loader(_npi):
        raise RuntimeError("db down")

    set_npi_cache_loader(bad_loader)
    # API disabled → returns None; key is "doesn't propagate"
    assert lookup_npi("1234567893", use_api_fallback=False) is None


# --------------------------------------------------------------------------- #
# bulk_lookup_npis                                                             #
# --------------------------------------------------------------------------- #


def test_bulk_lookup_dedupes_and_skips_invalid():
    cache = {
        "1234567893": NppesRecord(npi="1234567893"),
        "9999999999": NppesRecord(npi="9999999999"),
    }
    set_npi_cache_loader(lambda n: cache.get(n))
    out = bulk_lookup_npis(
        ["1234567893", "1234567893", "bad", None, "9999999999"],
        use_api_fallback=False,
    )
    assert set(out.keys()) == {"1234567893", "9999999999"}


def test_bulk_lookup_omits_unresolved_npis():
    set_npi_cache_loader(lambda _n: None)
    out = bulk_lookup_npis(["1234567893"], use_api_fallback=False)
    assert out == {}


# --------------------------------------------------------------------------- #
# Module-level constants (sanity)                                              #
# --------------------------------------------------------------------------- #


def test_pharmacy_channel_constants_are_full_taxonomy_codes():
    # All NUCC taxonomy codes are 10 chars (9 digits + trailing X).
    assert rwdc.NUCC_SPECIALTY_PHARMACY == "3336S0011X"
    assert rwdc.NUCC_MAIL_ORDER_PHARMACY == "3336M0002X"
    assert rwdc.NUCC_HOME_INFUSION_PHARMACY == "3336H0001X"
    assert all(len(c) == 10 for c in rwdc.PHARMACY_CHANNEL_CODES)
