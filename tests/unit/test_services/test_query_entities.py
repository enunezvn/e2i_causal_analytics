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

from src.services.enum_labels import REGION_ENUM_LABELS
from src.services.query_entities import (
    SUPPORTED_BRANDS,
    SUPPORTED_REGIONS,
    brand_from_text,
    brand_scan,
    canonical_brand,
    region_from_text,
    region_scan,
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


class TestBrandScan:
    """#2114 codex r1/r2 HIGH: ``brand_from_text`` collapses "named several" and
    "named none" into the same ``None``, so a caller that can speak to the user
    cannot tell an AMBIGUOUS ask from an intentional portfolio ask. The scan
    keeps both facts, exactly as :func:`region_scan` does for regions (#1572).

    ``grounded_brands`` is the ambiguity signal and runs BOTH routes — a brand
    named and an indication naming a different brand are two distinct scopes
    ("NBRx for Kisqali and PNH"). ``brand`` keeps the legacy #1356 precedence
    (a named brand suppresses the indication pass) because cohort_profiler's
    ask.py and the #1574 gap rule depend on it byte-for-byte."""

    def test_two_named_brands_are_reported_in_order_and_bind_nothing(self) -> None:
        scan = brand_scan("What is NBRx for Kisqali and Fabhalta?")
        assert scan.brand is None
        assert scan.grounded_brands == ("Fabhalta", "Kisqali")
        assert scan.is_ambiguous is True

    def test_three_named_brands_are_all_reported(self) -> None:
        scan = brand_scan("TRx for Remibrutinib, Fabhalta and Kisqali")
        assert scan.brand is None
        assert scan.grounded_brands == ("Remibrutinib", "Fabhalta", "Kisqali")
        assert scan.is_ambiguous is True

    def test_a_repeated_brand_is_not_ambiguous(self) -> None:
        scan = brand_scan("Kisqali NBRx — is Kisqali growing?")
        assert scan.brand == "Kisqali"
        assert scan.grounded_brands == ("Kisqali",)
        assert scan.is_ambiguous is False

    def test_two_indications_are_ambiguous_too(self) -> None:
        # The indication pass grounds brands just as a name does (#1356), so
        # "CSU and PNH" names two brands as surely as "Remibrutinib and Fabhalta".
        scan = brand_scan("What is NBRx for CSU and PNH?")
        assert scan.brand is None
        assert scan.grounded_brands == ("Remibrutinib", "Fabhalta")
        assert scan.is_ambiguous is True

    def test_no_brand_named_is_not_ambiguous(self) -> None:
        # The intentional-portfolio case: nothing named, nothing to clarify.
        for query in ("What is NBRx?", "Did rep actions lift prescriptions?", "", None):
            scan = brand_scan(query)
            assert scan.brand is None, query
            assert scan.grounded_brands == (), query
            assert scan.is_ambiguous is False, query

    def test_a_named_brand_still_wins_the_brand_field_over_an_indication(self) -> None:
        """The legacy precedence is untouched where it matters: ``brand`` is
        Kisqali, so every consumer of ``brand_from_text`` is unmoved. The
        ambiguity SIGNAL sees both scopes — that is the r2 fix."""
        scan = brand_scan("Kisqali uptake in urticaria clinics")
        assert scan.brand == "Kisqali"
        assert scan.grounded_brands == ("Kisqali", "Remibrutinib")
        assert scan.is_ambiguous is True

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("What is NBRx for Kisqali and PNH?", ("Kisqali", "Fabhalta")),
            ("What is NBRx for PNH and Kisqali?", ("Kisqali", "Fabhalta")),
            ("What is TRx for Kisqali and CSU?", ("Kisqali", "Remibrutinib")),
            ("What is NRx for urticaria and Kisqali?", ("Kisqali", "Remibrutinib")),
        ],
    )
    def test_a_brand_name_plus_another_brands_indication_is_ambiguous(
        self, query, expected
    ) -> None:
        """The r2 defect: PNH grounds Fabhalta (INDICATION_TO_BRAND), so
        "Kisqali and PNH" asks about TWO brands. Before this fix the indication
        pass never ran once a brand was named, ``brand`` bound Kisqali, and the
        ask was answered with a Kisqali-only figure. Order is deterministic:
        named brands first (SUPPORTED_BRANDS order), then indication-only ones
        (INDICATION_TO_BRAND order) -- never the order they appear in the text."""
        scan = brand_scan(query)
        assert scan.brand == "Kisqali", "the legacy field must not move"
        assert scan.grounded_brands == expected
        assert scan.is_ambiguous is True

    @pytest.mark.parametrize(
        "query",
        [
            "What is TRx for Kisqali in HR+ breast cancer?",
            "What is TRx for Kisqali in breast cancer?",
            "What is NBRx for Fabhalta in PNH?",
        ],
    )
    def test_a_brand_beside_its_OWN_indication_is_one_scope(self, query) -> None:
        """Measured decision (2026-09-15): when both routes ground the SAME
        brand the indication CORROBORATES the name, it does not add a scope --
        so the signal de-duplicates and stays unambiguous. This is the same
        rule as ``test_a_repeated_brand_is_not_ambiguous``, not a special case,
        and it keeps the commonest clinical phrasing answerable."""
        scan = brand_scan(query)
        assert scan.is_ambiguous is False
        assert len(scan.grounded_brands) == 1
        assert scan.brand == scan.grounded_brands[0]

    def test_an_indication_alone_still_grounds_its_brand(self) -> None:
        scan = brand_scan("What is TRx for PNH?")
        assert scan.brand == "Fabhalta"
        assert scan.grounded_brands == ("Fabhalta",)
        assert scan.is_ambiguous is False

    @pytest.mark.parametrize(
        "query,expected",
        [
            # MEASURED 2026-09-15, not copied from the implementation.
            ("Why did Kisqali TRx drop in Q1?", "Kisqali"),
            ("what is driving remibrutinib NRx?", "Remibrutinib"),
            ("Compare Kisqali and Fabhalta conversion", None),
            ("profile CSU patients on therapy", "Remibrutinib"),
            ("PNH persistence drivers", "Fabhalta"),
            ("HR+ breast cancer starts", "Kisqali"),
            ("Kisqali uptake in urticaria clinics", "Kisqali"),
            ("Did rep actions lift prescriptions?", None),
            ("What is NBRx for Kisqali and PNH?", "Kisqali"),
            ("What is NBRx for PNH and Kisqali?", "Kisqali"),
            ("What is TRx for Kisqali in HR+ breast cancer?", "Kisqali"),
            # The r3 alias fix moved these two deliberately (see the 0c0a73dd2
            # differential): HR+ alone now grounds Kisqali, and HR+ beside
            # another indication is two brands, so nothing binds.
            ("HR+ TRx trend", "Kisqali"),
            ("What is NBRx for PNH and HR+?", None),
            ("HR+breast uptake", "Kisqali"),
            ("chr+ something", None),
            ("", None),
            (None, None),
        ],
    )
    def test_brand_from_text_returns_the_expected_value(self, query, expected) -> None:
        """The #1356 EXACTLY-ONE-or-None contract, asserted against literal
        expected values.

        This replaces an equality against ``brand_scan(query).brand``, which
        became a TAUTOLOGY the moment ``brand_from_text`` was defined as exactly
        that expression — it read as proof across two review rounds while
        asserting ``x == x``. An equality test is evidence only while its two
        sides have independent definitions; a literal table survives the very
        refactor that hollowed the old assertion out.
        """
        assert brand_from_text(query) == expected


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


class TestRegionScan:
    """#1572: the free-text scan runs through the shared #1565 alias
    vocabulary, and a region-like phrase that genuinely spans multiple census
    regions ("East Coast", bare "East") raises the needs-clarify signal
    instead of silently binding nothing (which read as "national" downstream).
    """

    # (a) unambiguous natural phrasings resolve via the shared vocabulary.
    def test_natural_phrasings_resolve(self) -> None:
        assert region_from_text("What is TRx for Kisqali in the Northeast region?") == "northeast"
        assert region_from_text("What is TRx for Kisqali on the West Coast?") == "west"
        assert region_from_text("New England TRx for Kisqali") == "northeast"
        assert region_from_text("north-east TRx") == "northeast"

    def test_separated_compound_directionals_resolve_as_units(self) -> None:
        # codex iter-2 HIGH: "south-east" bound south on the pre-#1572 token
        # scan; masking only "south" would spuriously clarify on the leftover
        # "east". The phrase resolves as ONE unit via the shared vocabulary.
        scan = region_scan("What is the TRx for Kisqali in the south-east?")
        assert scan.region == "south"
        assert not scan.needs_clarification
        assert region_from_text("TRx for Kisqali in the south west") == "south"
        assert region_from_text("north west market share") == "west"

    def test_fused_compound_directionals_stay_unbound(self) -> None:
        # "southeast Michigan" / "northwest Indiana" are census-MIDWEST: the
        # fused form is the metro-area modifier idiom, so binding it would
        # produce a silently WRONG region. Unchanged-from-main coverage bound:
        # no region, no clarify.
        for query in ("Southeast TRx for Fabhalta", "southwest performance"):
            scan = region_scan(query)
            assert scan.region is None, query
            assert not scan.needs_clarification, query

    def test_allowlist_phrases_all_resolve_through_the_shared_vocabulary(self) -> None:
        # Drift pin: every free-text phrase must mean the same thing the chat
        # KPI tool would resolve it to — one vocabulary, two surfaces.
        from src.services.enum_labels import resolve_region_label
        from src.services.query_entities import _FREE_TEXT_REGION_PHRASES

        for phrase in _FREE_TEXT_REGION_PHRASES:
            assert resolve_region_label(phrase, allow_synonyms=True) is not None, phrase

    def test_resolved_scan_carries_no_clarify_signal(self) -> None:
        scan = region_scan("What is TRx for Kisqali on the West Coast?")
        assert scan.region == "west"
        assert scan.ambiguous_phrase is None
        assert not scan.needs_clarification

    # (b) multi-region phrases produce the needs-clarify signal, never a
    # silent no-region result.
    def test_east_coast_needs_clarification(self) -> None:
        scan = region_scan("What is the TRx for Kisqali on the East Coast?")
        assert scan.region is None
        assert scan.needs_clarification
        assert scan.ambiguous_phrase is not None
        assert scan.ambiguous_phrase.lower() == "east coast"

    def test_bare_east_needs_clarification(self) -> None:
        scan = region_scan("Kisqali TRx in the East")
        assert scan.region is None
        assert scan.needs_clarification
        assert scan.ambiguous_phrase.lower() == "east"

    def test_resolvable_plus_ambiguous_scope_clarifies_and_binds_nothing(self) -> None:
        # "Northeast vs the East Coast" is ambiguous AS A WHOLE: binding
        # northeast would silently drop the unresolvable half of the ask.
        scan = region_scan("compare Northeast and East Coast TRx")
        assert scan.region is None
        assert scan.needs_clarification

    # (c) guard phrases produce NEITHER a region NOR a clarify.
    def test_central_coast_guard_produces_neither(self) -> None:
        # #1565 guard: California's central coast is not "central" (midwest),
        # and a locality mention is not evidence a census region was meant.
        scan = region_scan("Kisqali TRx on the central coast")
        assert scan.region is None
        assert not scan.needs_clarification

    def test_middle_and_far_east_are_not_region_like(self) -> None:
        for query in (
            "Middle East supply impact on Kisqali TRx",
            "Middle  East supply impact",  # any separator run, not one char
            "far-east distribution for Fabhalta",
        ):
            scan = region_scan(query)
            assert scan.region is None, query
            assert not scan.needs_clarification, query

    def test_locality_modifier_aliases_never_bind_a_wrong_region(self) -> None:
        # The wrong-region traps that force the free-text ALLOWLIST: each of
        # these contains a tool-surface alias ("southern" -> south, "western"
        # -> west, "northwest" -> west, "pacific" -> west) whose census region
        # is WRONG (or nonexistent) for the named locality. They must bind
        # nothing — exactly the pre-#1572 behaviour.
        for query in (
            "southern california TRx",  # census-west
            "western pennsylvania TRx",  # census-northeast
            "northwest Indiana TRx",  # census-midwest
            "Asia-Pacific TRx strategy",  # not a US region
        ):
            scan = region_scan(query)
            assert scan.region is None, query
            assert not scan.needs_clarification, query

    def test_new_england_journal_is_not_a_region_scope(self) -> None:
        scan = region_scan("New England Journal of Medicine study on Kisqali TRx")
        assert scan.region is None
        assert not scan.needs_clarification

    def test_two_letter_abbreviations_do_not_bind_in_free_text(self) -> None:
        # "per se" contains the token "se"; the abbreviation aliases resolve
        # only at the tool surface, where the argument arrives pre-segmented.
        scan = region_scan("Did Kisqali TRx drop per se?")
        assert scan.region is None
        assert not scan.needs_clarification

    # (d) plain no-region questions are unchanged.
    def test_no_region_produces_neither(self) -> None:
        scan = region_scan("Why did Kisqali TRx drop?")
        assert scan.region is None
        assert scan.ambiguous_phrase is None

    def test_two_canonical_regions_are_retained_for_clarification(self) -> None:
        # #2191: exactly-one still means no region binds, but the scan must not
        # collapse "named several" into the same result as "named none". A
        # scalar consumer can now clarify/fail closed instead of serving its
        # national default.
        scan = region_scan("compare northeast and west conversion")
        assert scan.region is None
        assert scan.grounded_regions == ("northeast", "west")
        assert scan.needs_clarification

    def test_empty_and_none_produce_neither(self) -> None:
        for query in (None, "", "   "):
            scan = region_scan(query)
            assert scan.region is None
            assert scan.ambiguous_phrase is None


class TestSupportedBrands:
    def test_matches_cohort_profiler_contract(self) -> None:
        # The canonical casing set the KPI brand predicate depends on.
        assert SUPPORTED_BRANDS == ("Remibrutinib", "Fabhalta", "Kisqali")


class TestSupportedRegions:
    def test_matches_the_enum_ssot(self) -> None:
        # The scan resolves through enum_labels (#1572); the local constant
        # must stay in step with the database enum contract.
        assert set(SUPPORTED_REGIONS) == set(REGION_ENUM_LABELS)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])


class TestHrPlusIndicationAlias:
    """#2114 codex r3 HIGH-1: ``\\bhr\\+\\b`` could not match a standalone "HR+".

    ``\\b`` after ``+`` demands a word-character transition, so the alias fired
    ONLY when a word followed immediately ("HR+breast", "HR+2") — never before a
    space or punctuation, which is how people actually write it. Kisqali's
    indication therefore grounded nothing in "HR+ and PNH", and the ask was
    answered with Fabhalta alone. The fix drops the trailing ``\\b``; the LEADING
    ``\\b`` still rejects "chr+". Codex proposed ``\\bhr\\+(?!\\w)``, which was
    measured to REGRESS the two shapes that work today.
    """

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("HR+ and PNH", ("Kisqali", "Fabhalta")),
            ("PNH and HR+", ("Kisqali", "Fabhalta")),
            ("What is NBRx for HR+ and PNH?", ("Kisqali", "Fabhalta")),
            ("HR+, PNH", ("Kisqali", "Fabhalta")),
            ("HR+ and CSU", ("Remibrutinib", "Kisqali")),
        ],
    )
    def test_standalone_hr_plus_grounds_kisqali_and_is_ambiguous(self, query, expected) -> None:
        scan = brand_scan(query)
        assert set(scan.grounded_brands) == set(expected), scan.grounded_brands
        assert scan.is_ambiguous is True

    @pytest.mark.parametrize("query", ["HR+breast", "HR+2 positive", "HR+ breast cancer"])
    def test_the_shapes_that_already_worked_still_work(self, query) -> None:
        """Pins the regression codex's `(?!\\w)` would have caused: these match
        today and must keep matching."""
        scan = brand_scan(query)
        assert scan.grounded_brands == ("Kisqali",), scan.grounded_brands
        assert scan.brand == "Kisqali"

    @pytest.mark.parametrize("query", ["chr+ something", "the chr+ marker"])
    def test_a_leading_word_character_still_rejects(self, query) -> None:
        """The LEADING ``\\b`` is what stops "chr+"; dropping the trailing one
        does not weaken it."""
        assert brand_scan(query).grounded_brands == ()
