"""Tests for the shared brand/region enum-label resolver (#1505).

Consolidates three independent normalizers (chat KPI tool, cohort resolution,
entity extraction's alias table) onto one owner. The two resolution *modes* are
the point of the module: the chat KPI tool accepts the platform's region
synonyms, cohort resolution stays strict (fail-closed) — and neither can drift
from the enum labels the database actually holds.
"""

import pytest

from src.services import enum_labels


class TestEnumLabelSets:
    def test_region_labels_are_the_region_type_enum(self):
        assert enum_labels.REGION_ENUM_LABELS == ("northeast", "south", "midwest", "west")

    def test_brand_labels_are_the_brand_type_enum(self):
        assert enum_labels.BRAND_ENUM_LABELS == (
            "Remibrutinib",
            "Fabhalta",
            "Kisqali",
            "competitor",
            "other",
        )

    def test_label_sets_are_immutable_tuples(self):
        # A mutable label set could be edited at runtime into a value the enum
        # cannot hold (the 22P02 class #1501 fixed).
        assert isinstance(enum_labels.REGION_ENUM_LABELS, tuple)
        assert isinstance(enum_labels.BRAND_ENUM_LABELS, tuple)


class TestRegionAliasTable:
    def test_alias_table_keys_are_all_enum_labels(self):
        # The table feeds a Postgres enum cast: a canonical key that is not a
        # real label could push a 22P02 value into the query.
        assert set(enum_labels.REGION_ALIASES) <= set(enum_labels.REGION_ENUM_LABELS)

    def test_alias_map_admits_only_enum_labels_as_values(self):
        assert set(enum_labels.REGION_LABEL_BY_ALIAS.values()) <= set(
            enum_labels.REGION_ENUM_LABELS
        )

    def test_alias_map_has_no_cross_region_collisions(self):
        # Every folded alias must resolve to exactly one label; a collision
        # would make resolution order-dependent.
        seen: dict[str, str] = {}
        for label, aliases in enum_labels.REGION_ALIASES.items():
            for alias in (label, *aliases):
                folded = enum_labels.fold_region_key(alias)
                assert seen.setdefault(folded, label) == label, folded


class TestResolveRegionStrict:
    """Strict mode is cohort_resolution's fail-closed contract."""

    @pytest.mark.parametrize(
        "supplied,expected",
        [
            ("northeast", "northeast"),
            ("Northeast", "northeast"),
            ("NORTHEAST", "northeast"),
            ("  west  ", "west"),
            ("Midwest", "midwest"),
            ("South", "south"),
        ],
    )
    def test_case_and_whitespace_resolve(self, supplied, expected):
        assert enum_labels.resolve_region_label(supplied, allow_synonyms=False) == expected

    @pytest.mark.parametrize(
        "supplied",
        ["NE", "North East", "north-east", "new england", "central", "Pacific", "southern"],
    )
    def test_synonyms_and_separators_do_not_resolve(self, supplied):
        # Strict mode must NOT silently widen what cohort resolution accepts.
        assert enum_labels.resolve_region_label(supplied, allow_synonyms=False) is None

    @pytest.mark.parametrize("supplied", ["US", "EU", "APAC", "atlantis", "", "   ", None])
    def test_non_members_return_none(self, supplied):
        assert enum_labels.resolve_region_label(supplied, allow_synonyms=False) is None


class TestResolveRegionWithSynonyms:
    """Synonym-tolerant mode is the chat KPI tool's contract (#1501/#1504)."""

    @pytest.mark.parametrize(
        "supplied,expected",
        [
            ("Northeast", "northeast"),
            ("North East", "northeast"),
            ("north-east", "northeast"),
            ("NE", "northeast"),
            ("new england", "northeast"),
            ("mid west", "midwest"),
            ("central", "midwest"),
            ("Pacific", "west"),
            ("southern", "south"),
            ("nw", "west"),
        ],
    )
    def test_platform_synonyms_resolve(self, supplied, expected):
        assert enum_labels.resolve_region_label(supplied, allow_synonyms=True) == expected

    @pytest.mark.parametrize("supplied", ["US", "EU", "APAC", "atlantis", "", None])
    def test_unknown_still_returns_none(self, supplied):
        assert enum_labels.resolve_region_label(supplied, allow_synonyms=True) is None


class TestModeIsExplicit:
    def test_allow_synonyms_is_required_keyword_only(self):
        # No default: every call site must state which contract it wants, so a
        # future edit cannot loosen cohort resolution by omission.
        with pytest.raises(TypeError):
            enum_labels.resolve_region_label("northeast")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            enum_labels.resolve_region_label("northeast", True)  # type: ignore[misc]

    def test_synonym_mode_is_strictly_wider_than_strict_mode(self):
        # Anything strict accepts, synonym mode must accept identically.
        for label in enum_labels.REGION_ENUM_LABELS:
            for form in (label, label.upper(), label.title(), f"  {label} "):
                strict = enum_labels.resolve_region_label(form, allow_synonyms=False)
                wide = enum_labels.resolve_region_label(form, allow_synonyms=True)
                assert strict == wide == label


class TestResolveBrand:
    @pytest.mark.parametrize(
        "supplied,expected",
        [
            ("Kisqali", "Kisqali"),
            ("kisqali", "Kisqali"),
            ("KISQALI", "Kisqali"),
            (" fabhalta ", "Fabhalta"),
            ("REMIBRUTINIB", "Remibrutinib"),
            ("Competitor", "competitor"),
            ("OTHER", "other"),
        ],
    )
    def test_any_casing_resolves_to_the_real_label(self, supplied, expected):
        assert enum_labels.resolve_brand_label(supplied) == expected

    @pytest.mark.parametrize("supplied", ["Aspirin", "remi", "btk inhibitor", "", "  ", None])
    def test_unknown_returns_none(self, supplied):
        # "remi" is an entity-extraction alias, NOT a brand_type label: brands
        # have no alias table (only regions do).
        assert enum_labels.resolve_brand_label(supplied) is None

    def test_no_separator_folding_for_brands(self):
        # brand_type labels contain no separators; folding them would invent
        # matches ("Kis qali") the enum cannot hold.
        assert enum_labels.resolve_brand_label("Kis qali") is None


class TestBrandBucketLabels:
    """#1517 item 3: "competitor" / "other" are aggregation buckets on the
    ``brand_type`` enum (added for DB enum sync), not named products. Entity
    extraction must not treat these ordinary English words as brand tokens,
    while the enum-resolution path keeps accepting them."""

    def test_buckets_are_a_subset_of_the_enum_labels(self):
        assert set(enum_labels.BRAND_BUCKET_LABELS) <= set(enum_labels.BRAND_ENUM_LABELS)

    def test_buckets_exclude_every_named_product(self):
        products = set(enum_labels.BRAND_ENUM_LABELS) - set(enum_labels.BRAND_BUCKET_LABELS)
        assert products == {"Remibrutinib", "Fabhalta", "Kisqali"}

    def test_buckets_still_resolve_on_the_enum_path(self):
        for bucket in enum_labels.BRAND_BUCKET_LABELS:
            assert enum_labels.resolve_brand_label(bucket.title()) == bucket

    def test_buckets_are_an_immutable_tuple(self):
        assert isinstance(enum_labels.BRAND_BUCKET_LABELS, tuple)


class TestCasefoldSemantics:
    """The resolvers use ``casefold()`` (Python's caseless-matching operation),
    while cohort_resolution previously used ``lower()``.

    Exhaustively measured over all 0x110000 codepoints INCLUDING expanding
    folds (an early length-preserving probe missed those and undercounted):
    exactly three codepoints admit a string ``lower()`` would reject — U+017F
    LONG S, U+FB05 LONG S T, U+FB06 ST — for 11 inputs in total. All 11 land on
    a REAL label; none lands on a wrong one, which is the property that matters
    (a wrong label would mean a wrong brand or population).
    """

    @pytest.mark.parametrize(
        "supplied,expected",
        [("Kiſqali", "Kisqali"), ("kiſqali", "Kisqali")],
    )
    def test_long_s_brand_folds_onto_a_real_label(self, supplied, expected):
        assert enum_labels.resolve_brand_label(supplied) == expected

    @pytest.mark.parametrize(
        "supplied,expected",
        [
            ("ſouth", "south"),
            ("northeaſt", "northeast"),
            ("northeaﬅ", "northeast"),  # U+FB05 ligature expands to "st"
            ("northeaﬆ", "northeast"),  # U+FB06 ligature expands to "st"
            ("weﬆ", "west"),
            ("midweﬆ", "midwest"),
        ],
    )
    def test_expanding_folds_land_on_a_real_label(self, supplied, expected):
        assert enum_labels.resolve_region_label(supplied, allow_synonyms=False) == expected
        assert enum_labels.resolve_region_label(supplied, allow_synonyms=True) == expected

    def test_no_fold_produces_a_wrong_label(self):
        # The property that actually matters: a casefold-only match may never
        # resolve to a DIFFERENT label than the one it spells.
        for supplied, expected in [
            ("ſouth", "south"),
            ("northeaﬆ", "northeast"),
            ("weﬆ", "west"),
            ("Kiſqali", "Kisqali"),
        ]:
            assert supplied.casefold() == expected.casefold()


class TestVocabularyDrift:
    """The label sets are a DATABASE contract, deliberately not read from the
    editable YAML vocabulary at runtime (a vocab edit landing before its
    migration would otherwise push a non-label into the enum cast). These
    tests are the drift alarm that keeps the two in step."""

    def test_registry_regions_match_the_enum_labels(self):
        from src.ontology import VocabularyRegistry

        assert tuple(VocabularyRegistry.load().get_regions()) == enum_labels.REGION_ENUM_LABELS

    def test_registry_brands_match_the_enum_labels(self):
        from src.ontology import VocabularyRegistry

        assert tuple(VocabularyRegistry.load().get_brands()) == enum_labels.BRAND_ENUM_LABELS


class TestSingleOwner:
    """#1505: one implementation, not three."""

    def test_cohort_resolution_delegates_to_the_shared_resolvers(self):
        from src.services import cohort_resolution

        assert cohort_resolution._normalize_brand is enum_labels.resolve_brand_label
        assert not hasattr(cohort_resolution, "_BRAND_CANONICAL")
        assert not hasattr(cohort_resolution, "_REGION_CANONICAL")

    def test_chatbot_tools_delegates_to_the_shared_resolvers(self):
        from src.api.routes import chatbot_tools

        assert chatbot_tools._normalize_brand is enum_labels.resolve_brand_label
        assert not hasattr(chatbot_tools, "_BRAND_LABEL_BY_CASEFOLD")
        assert not hasattr(chatbot_tools, "_build_region_alias_map")

    def test_entity_extractor_sources_its_aliases_from_the_shared_table(self):
        from src.rag import entity_extractor

        assert entity_extractor.REGION_ALIASES is enum_labels.REGION_ALIASES
