"""Shared deterministic query-text entity extraction (#1351).

The 2026-07-29 empirical pass proved the chat dispatch path never populates
``parsed_query`` (no producer exists), so a brand/region named ONLY in the ask
text was invisible to every dispatcher input resolver — the exact q11 failure
mode #1356 fixed privately inside cohort_profiler's ``ask.py``. This module
lifts that proven extraction into a shared service so ALL resolvers ground the
same way (owner ruling on #1351: resolvers everywhere).

Semantics pinned by these tests (identical to ask.py's originals):
* brand binds only when the text pins down EXACTLY ONE brand (two named ⇒ None
  — never guess);
* brand name match wins over indication inference;
* region binds only on exactly one of the four canonical region tokens.
"""

from __future__ import annotations

import pytest

from src.services.query_entities import (
    SUPPORTED_BRANDS,
    brand_from_text,
    canonical_brand,
    region_from_text,
)


class TestCanonicalBrand:
    def test_exact_casing_passthrough(self) -> None:
        assert canonical_brand("Kisqali") == "Kisqali"

    def test_case_insensitive_normalization(self) -> None:
        assert canonical_brand("kisqali") == "Kisqali"
        assert canonical_brand("FABHALTA") == "Fabhalta"

    def test_unknown_and_empty_return_none(self) -> None:
        assert canonical_brand("Humira") is None
        assert canonical_brand("") is None
        assert canonical_brand(None) is None


class TestBrandFromText:
    def test_single_brand_name_binds(self) -> None:
        assert brand_from_text("Why did Kisqali TRx drop in Q1?") == "Kisqali"

    def test_brand_binds_case_insensitively_with_canonical_casing(self) -> None:
        assert brand_from_text("what is driving remibrutinib NRx?") == "Remibrutinib"

    def test_two_brands_named_is_ambiguous(self) -> None:
        assert brand_from_text("Compare Kisqali and Fabhalta conversion") is None

    def test_indication_grounds_brand_when_no_name(self) -> None:
        assert brand_from_text("profile CSU patients on therapy") == "Remibrutinib"
        assert brand_from_text("PNH persistence drivers") == "Fabhalta"
        assert brand_from_text("HR+ breast cancer starts") == "Kisqali"

    def test_brand_name_wins_over_indication(self) -> None:
        # A named brand is authoritative; the indication scan must not run.
        assert brand_from_text("Kisqali uptake in urticaria clinics") == "Kisqali"

    def test_no_brand_returns_none(self) -> None:
        assert brand_from_text("Did rep actions lift prescriptions?") is None
        assert brand_from_text("") is None


class TestRegionFromText:
    def test_single_region_binds(self) -> None:
        assert region_from_text("Why did TRx drop in the northeast region?") == "northeast"

    def test_case_insensitive(self) -> None:
        assert region_from_text("Midwest gap analysis") == "midwest"

    def test_two_regions_is_ambiguous(self) -> None:
        assert region_from_text("compare northeast and west conversion") is None

    def test_no_region_returns_none(self) -> None:
        assert region_from_text("Why did Kisqali TRx drop?") is None

    def test_substring_tokens_do_not_false_match(self) -> None:
        # "southern" is not the canonical token "south" — word-boundary anchored.
        assert region_from_text("the southernmost territory") is None


class TestSupportedBrands:
    def test_matches_cohort_profiler_contract(self) -> None:
        # The canonical casing set the KPI brand predicate depends on.
        assert SUPPORTED_BRANDS == ("Remibrutinib", "Fabhalta", "Kisqali")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
