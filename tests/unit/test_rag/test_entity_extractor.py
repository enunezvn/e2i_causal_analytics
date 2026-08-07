"""
Unit tests for E2I RAG Entity Extractor.

Tests entity extraction from natural language queries using
fixed E2I domain vocabularies.

All tests use vocabulary-based matching with no medical NER.
"""

import pytest

from src.rag.entity_extractor import REGION_ALIASES, EntityExtractor, EntityVocabulary

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def extractor():
    """Create EntityExtractor with default vocabulary."""
    return EntityExtractor()


@pytest.fixture
def custom_vocabulary():
    """Create custom vocabulary for testing."""
    return EntityVocabulary(
        brands={
            "TestBrand": ["testbrand", "tb"],
            "AnotherBrand": ["anotherbrand", "ab"],
        },
        regions={
            "testregion": ["testregion", "tr"],
        },
        kpis={
            "test_kpi": ["test kpi", "tkpi"],
        },
        agents={},
        journey_stages={},
        time_references={},
        hcp_segments={},
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestEntityExtractorInit:
    """Tests for EntityExtractor initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default vocabulary."""
        extractor = EntityExtractor()
        assert extractor.vocabulary is not None
        assert len(extractor.vocabulary.brands) > 0
        assert "Remibrutinib" in extractor.vocabulary.brands
        assert "Kisqali" in extractor.vocabulary.brands
        assert "Fabhalta" in extractor.vocabulary.brands

    def test_init_with_custom_vocabulary(self, custom_vocabulary):
        """Test initialization with custom vocabulary."""
        extractor = EntityExtractor(vocabulary=custom_vocabulary)
        assert len(extractor.vocabulary.brands) == 2
        assert "TestBrand" in extractor.vocabulary.brands

    def test_repr(self, extractor):
        """Test string representation."""
        repr_str = repr(extractor)
        assert "EntityExtractor" in repr_str
        assert "brands=" in repr_str
        assert "kpis=" in repr_str


# ============================================================================
# Brand Extraction Tests
# ============================================================================


class TestBrandExtraction:
    """Tests for brand name extraction."""

    def test_extract_kisqali(self, extractor):
        """Test extracting Kisqali brand."""
        entities = extractor.extract("Why did Kisqali adoption drop?")
        assert "Kisqali" in entities.brands

    def test_extract_remibrutinib(self, extractor):
        """Test extracting Remibrutinib brand."""
        entities = extractor.extract("Show Remibrutinib conversion rates")
        assert "Remibrutinib" in entities.brands

    def test_extract_fabhalta(self, extractor):
        """Test extracting Fabhalta brand."""
        entities = extractor.extract("What is Fabhalta market share?")
        assert "Fabhalta" in entities.brands

    def test_extract_brand_alias(self, extractor):
        """Test extracting brand using alias."""
        entities = extractor.extract("Show me Remi TRx trends")
        assert "Remibrutinib" in entities.brands

    def test_extract_multiple_brands(self, extractor):
        """Test extracting multiple brands."""
        entities = extractor.extract("Compare Kisqali and Fabhalta in Q3")
        assert "Kisqali" in entities.brands
        assert "Fabhalta" in entities.brands
        assert len(entities.brands) == 2

    def test_case_insensitive_brand(self, extractor):
        """Test case-insensitive brand matching."""
        entities = extractor.extract("What happened to KISQALI?")
        assert "Kisqali" in entities.brands

    def test_no_brand_found(self, extractor):
        """Test when no brand is found."""
        entities = extractor.extract("Show overall market trends")
        assert len(entities.brands) == 0


# ============================================================================
# Region Extraction Tests
# ============================================================================


class TestRegionExtraction:
    """Tests for region extraction."""

    def test_extract_west(self, extractor):
        """Test extracting West region."""
        entities = extractor.extract("What happened in the West region?")
        assert "west" in entities.regions

    def test_extract_northeast(self, extractor):
        """Test extracting Northeast region."""
        entities = extractor.extract("Northeast TRx is declining")
        assert "northeast" in entities.regions

    def test_extract_multiple_regions(self, extractor):
        """Test extracting multiple regions."""
        entities = extractor.extract("Compare West and Midwest performance")
        assert "west" in entities.regions
        assert "midwest" in entities.regions

    def test_no_region_found(self, extractor):
        """Test when no region is found."""
        entities = extractor.extract("Show national TRx trends")
        assert len(entities.regions) == 0


# ============================================================================
# KPI Extraction Tests
# ============================================================================


class TestKPIExtraction:
    """Tests for KPI extraction."""

    def test_extract_trx(self, extractor):
        """Test extracting TRx KPI."""
        entities = extractor.extract("Show TRx for Kisqali")
        assert "trx" in entities.kpis

    def test_extract_nrx(self, extractor):
        """Test extracting NRx KPI."""
        entities = extractor.extract("What is the NRx trend?")
        assert "nrx" in entities.kpis

    def test_extract_conversion_rate(self, extractor):
        """Test extracting conversion rate KPI."""
        entities = extractor.extract("What is the conversion rate?")
        assert "conversion_rate" in entities.kpis

    def test_extract_market_share(self, extractor):
        """Test extracting market share KPI."""
        entities = extractor.extract("Show market share by region")
        assert "trx_share" in entities.kpis

    def test_extract_multiple_kpis(self, extractor):
        """Test extracting multiple KPIs."""
        entities = extractor.extract("Compare TRx and NRx for Q3")
        assert "trx" in entities.kpis
        assert "nrx" in entities.kpis

    def test_no_kpi_found(self, extractor):
        """Test when no KPI is found."""
        entities = extractor.extract("What happened last quarter?")
        assert len(entities.kpis) == 0


# ============================================================================
# Time Reference Extraction Tests
# ============================================================================


class TestTimeReferenceExtraction:
    """Tests for time reference extraction."""

    def test_extract_quarter(self, extractor):
        """Test extracting quarter reference."""
        entities = extractor.extract("What happened in Q3?")
        assert "Q3" in entities.time_references

    def test_extract_ytd(self, extractor):
        """Test extracting YTD reference."""
        entities = extractor.extract("Show YTD performance")
        assert "YTD" in entities.time_references

    def test_extract_year(self, extractor):
        """Test extracting year reference."""
        entities = extractor.extract("Compare 2024 vs 2023")
        assert "2024" in entities.time_references
        assert "2023" in entities.time_references

    def test_extract_last_month(self, extractor):
        """Test extracting last month reference."""
        entities = extractor.extract("What happened last month?")
        assert "last_month" in entities.time_references

    def test_extract_multiple_time_refs(self, extractor):
        """Test extracting multiple time references."""
        entities = extractor.extract("Compare Q1 and Q2 performance")
        assert "Q1" in entities.time_references
        assert "Q2" in entities.time_references


# ============================================================================
# HCP Segment Extraction Tests
# ============================================================================


class TestHCPSegmentExtraction:
    """Tests for HCP segment extraction."""

    def test_extract_high_volume(self, extractor):
        """Test extracting high volume segment."""
        entities = extractor.extract("Show high volume prescribers")
        assert "high_volume" in entities.hcp_segments

    def test_extract_kol(self, extractor):
        """Test extracting KOL segment."""
        entities = extractor.extract("Which KOLs are prescribing?")
        assert "key_opinion_leader" in entities.hcp_segments

    def test_extract_academic(self, extractor):
        """Test extracting academic segment."""
        entities = extractor.extract("Focus on academic centers")
        assert "academic" in entities.hcp_segments


# ============================================================================
# Journey Stage Extraction Tests
# ============================================================================


class TestJourneyStageExtraction:
    """Tests for patient journey stage extraction."""

    def test_extract_first_line(self, extractor):
        """Test extracting first line stage."""
        entities = extractor.extract("Show first line patients")
        assert "first_line" in entities.journey_stages

    def test_extract_switch(self, extractor):
        """Test extracting switch stage."""
        entities = extractor.extract("What about switching patients?")
        assert "switch" in entities.journey_stages

    def test_extract_treatment_naive(self, extractor):
        """Test extracting treatment naive stage."""
        entities = extractor.extract("Focus on treatment naive")
        assert "treatment_naive" in entities.journey_stages


# ============================================================================
# Agent Extraction Tests
# ============================================================================


class TestAgentExtraction:
    """Tests for agent name extraction."""

    def test_extract_causal_impact(self, extractor):
        """Test extracting causal impact agent."""
        entities = extractor.extract("Use causal impact analysis")
        assert "causal_impact" in entities.agents

    def test_extract_drift_monitor(self, extractor):
        """Test extracting drift monitor agent."""
        entities = extractor.extract("Check drift monitoring")
        assert "drift_monitor" in entities.agents


# ============================================================================
# Complex Query Tests
# ============================================================================


class TestComplexQueries:
    """Tests for complex queries with multiple entity types."""

    def test_complex_query_brand_kpi_region_time(self, extractor):
        """Test extracting from complex query."""
        query = "Why did Kisqali TRx drop in the West during Q3?"
        entities = extractor.extract(query)

        assert "Kisqali" in entities.brands
        assert "trx" in entities.kpis
        assert "west" in entities.regions
        assert "Q3" in entities.time_references

    def test_complex_query_multiple_brands(self, extractor):
        """Test extracting multiple brands with context."""
        query = "Compare Remibrutinib conversion to Fabhalta in Northeast"
        entities = extractor.extract(query)

        assert "Remibrutinib" in entities.brands
        assert "Fabhalta" in entities.brands
        assert "conversion_rate" in entities.kpis
        assert "northeast" in entities.regions

    def test_is_empty(self, extractor):
        """Test is_empty method on extracted entities."""
        entities = extractor.extract("Hello world")
        assert entities.is_empty()

        entities = extractor.extract("Show Kisqali TRx")
        assert not entities.is_empty()


# ============================================================================
# Word Boundary Tests
# ============================================================================


class TestWordBoundaries:
    """Tests for word boundary matching."""

    def test_no_partial_match_trx(self, extractor):
        """Test that 'trx' doesn't match in 'matrix'."""
        entities = extractor.extract("Show the matrix data")
        assert "trx" not in entities.kpis

    def test_no_partial_match_west(self, extractor):
        """Test that 'west' doesn't match in 'investment'."""
        entities = extractor.extract("Check the investment")
        # 'west' should not be found as it's not in 'investment'
        assert "west" not in entities.regions

    def test_exact_word_match(self, extractor):
        """Test exact word matching."""
        entities = extractor.extract("Focus on West region")
        assert "west" in entities.regions


# ============================================================================
# Confidence Score Tests
# ============================================================================


class TestConfidenceScores:
    """Tests for extraction with confidence scores."""

    def test_extract_with_confidence(self, extractor):
        """Test extracting with confidence scores."""
        result = extractor.extract_with_confidence("Show Kisqali TRx in West")

        assert "brands" in result
        assert len(result["brands"]) == 1
        assert result["brands"][0]["entity"] == "Kisqali"
        assert result["brands"][0]["confidence"] == 0.95
        assert result["brands"][0]["source"] == "vocabulary"

        assert "kpis" in result
        assert "regions" in result

    def test_confidence_empty_query(self, extractor):
        """Test confidence scores for empty results."""
        result = extractor.extract_with_confidence("Hello world")
        # No entity types should be in result
        assert len(result) == 0


# ============================================================================
# Vocabulary Tests
# ============================================================================


class TestEntityVocabulary:
    """Tests for EntityVocabulary class."""

    def test_default_vocabulary(self):
        """Test creating default vocabulary."""
        vocab = EntityVocabulary.from_default()

        assert "Remibrutinib" in vocab.brands
        assert "Kisqali" in vocab.brands
        assert "Fabhalta" in vocab.brands
        assert len(vocab.regions) > 0
        assert len(vocab.kpis) > 0

    def test_custom_vocabulary(self, custom_vocabulary):
        """Test custom vocabulary."""
        assert "TestBrand" in custom_vocabulary.brands
        assert "AnotherBrand" in custom_vocabulary.brands
        assert len(custom_vocabulary.brands) == 2

    def test_default_regions_are_built_from_the_shared_alias_table(self):
        """from_default() must source region aliases from REGION_ALIASES (#1505).

        The chat KPI tool's region_type normalization resolves exactly the
        phrasings this vocabulary recognizes (#1501/#1504). Before this pin the
        coupling was only observable from the consumer's tests, so a local
        re-literalisation here would have decoupled the two surfaces silently.
        """
        vocab = EntityVocabulary.from_default()

        for region, aliases in vocab.regions.items():
            assert aliases == REGION_ALIASES[region], region
            # list() copies, not the shared object: each vocabulary instance
            # keeps independently mutable alias lists.
            assert aliases is not REGION_ALIASES[region], region

    def test_default_regions_cover_every_alias_table_entry(self):
        """No alias-table region may be dropped on the way into the vocabulary."""
        vocab = EntityVocabulary.from_default()
        assert set(REGION_ALIASES) <= set(vocab.regions)

    def test_alias_table_is_the_shared_owner(self):
        """REGION_ALIASES is re-exported from the shared enum-label module."""
        from src.services import enum_labels

        assert REGION_ALIASES is enum_labels.REGION_ALIASES

    def test_registry_unavailable_fallback_follows_the_shared_label_set(self, monkeypatch):
        """The VocabularyRegistry-unavailable fallback must not hand-copy the
        region labels (#1505) — a copy here would silently stop tracking the
        region_type enum the rest of the platform resolves against."""
        from src.ontology import VocabularyRegistry
        from src.rag import entity_extractor as ee

        def _unavailable():
            raise RuntimeError("VocabularyRegistry unavailable")

        monkeypatch.setattr(VocabularyRegistry, "load", staticmethod(_unavailable))
        monkeypatch.setattr(ee, "REGION_ENUM_LABELS", ("northeast", "sentinelregion"))

        vocab = EntityVocabulary.from_default()

        assert set(vocab.regions) == {"northeast", "sentinelregion"}
        # A label with no alias-table entry still gets a usable single alias.
        assert vocab.regions["sentinelregion"] == ["sentinelregion"]
        assert vocab.regions["northeast"] == REGION_ALIASES["northeast"]


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_empty_query(self, extractor):
        """Test handling empty query."""
        entities = extractor.extract("")
        assert entities.is_empty()

    def test_none_safe(self, extractor):
        """Test that None doesn't crash."""
        # This should raise an error or handle gracefully
        try:
            extractor.extract(None)  # type: ignore
        except (TypeError, AttributeError):
            pass  # Expected behavior


# ============================================================================
# Brand bucket labels (#1517 item 3)
# ============================================================================


class TestBrandBucketLabelsNotExtracted:
    """``brand_type`` carries two aggregation buckets — "competitor" and
    "other" — added to the vocabulary for DB enum sync (get_enum_values), not
    for NLP. They are ordinary English words: extracting them as brand entities
    scopes graph search / analytics to a bucket the user never named (the
    knowledge graph HAS a ``competitor`` Brand node, matched case-insensitively
    by the graph backend). The registry-unavailable fallback already excluded
    them ON PURPOSE (#1509); the primary VocabularyRegistry path must agree.
    """

    def test_default_vocabulary_excludes_bucket_labels(self):
        from src.services.enum_labels import BRAND_BUCKET_LABELS

        vocab = EntityVocabulary.from_default()
        for bucket in BRAND_BUCKET_LABELS:
            assert bucket not in vocab.brands, bucket

    def test_common_words_are_not_extracted_as_brands(self, extractor):
        entities = extractor.extract("How does Kisqali compare to competitor brands?")
        assert entities.brands == ["Kisqali"]

        entities = extractor.extract("What other factors drove TRx in the West?")
        assert entities.brands == []

    def test_product_brands_still_extracted(self, extractor):
        entities = extractor.extract("Compare Kisqali, Fabhalta and Remibrutinib")
        assert entities.brands == ["Fabhalta", "Kisqali", "Remibrutinib"]

    def test_bucket_labels_are_real_enum_labels(self):
        # The buckets stay valid brand_type labels (the enum-resolution path is
        # untouched); they are only excluded from EXTRACTION.
        from src.services.enum_labels import (
            BRAND_BUCKET_LABELS,
            BRAND_ENUM_LABELS,
            resolve_brand_label,
        )

        assert set(BRAND_BUCKET_LABELS) <= set(BRAND_ENUM_LABELS)
        for bucket in BRAND_BUCKET_LABELS:
            assert resolve_brand_label(bucket.upper()) == bucket

    def test_registry_fallback_excludes_buckets_via_shared_constant(self, monkeypatch):
        from src.ontology import VocabularyRegistry
        from src.services.enum_labels import BRAND_BUCKET_LABELS

        def _unavailable():
            raise RuntimeError("VocabularyRegistry unavailable")

        monkeypatch.setattr(VocabularyRegistry, "load", staticmethod(_unavailable))
        vocab = EntityVocabulary.from_default()
        assert set(vocab.brands) == {"Remibrutinib", "Fabhalta", "Kisqali"}
        for bucket in BRAND_BUCKET_LABELS:
            assert bucket not in vocab.brands, bucket


# ============================================================================
# YAML config path keeps the shared region alias table (#1517 item 5)
# ============================================================================


class TestParseConfigRegionAliases:
    """``_parse_config`` (the YAML config path) previously attached only
    ``[region.lower()]`` as each region's alias list, silently dropping
    REGION_ALIASES — a config-loaded extractor rejected "New England", "NE",
    "central" that the default-path extractor accepts. For canonical
    ``region_type`` labels the YAML path must yield the same alias set as the
    default path; unknown custom regions keep the single-alias behavior.
    """

    _CONFIG = {
        "regions": {
            "description": "US geographic regions",
            "values": ["northeast", "south", "midwest", "west"],
        }
    }

    def test_canonical_regions_get_the_shared_alias_table(self, extractor):
        vocab = extractor._parse_config(self._CONFIG)
        for region in self._CONFIG["regions"]["values"]:
            assert vocab.regions[region] == REGION_ALIASES[region], region
            # A copy, not the shared list object.
            assert vocab.regions[region] is not REGION_ALIASES[region], region

    def test_config_loaded_extractor_matches_alias_phrasings(self, extractor):
        vocab = extractor._parse_config(self._CONFIG)
        config_extractor = EntityExtractor(vocabulary=vocab)
        entities = config_extractor.extract("How is TRx doing in New England?")
        assert entities.regions == ["northeast"]

    def test_unknown_custom_region_keeps_single_alias(self, extractor):
        vocab = extractor._parse_config({"regions": {"values": ["EMEA"]}})
        assert vocab.regions["EMEA"] == ["emea"]

    def test_yaml_brands_skip_bucket_labels(self, extractor):
        # The standard domain_vocabulary.yaml lists the buckets in
        # brands.values (DB enum sync); the YAML path must not turn them into
        # extraction tokens any more than the default path does (#1517).
        vocab = extractor._parse_config(
            {"brands": {"values": ["Remibrutinib", "Fabhalta", "Kisqali", "competitor", "other"]}}
        )
        assert "competitor" not in vocab.brands
        assert "other" not in vocab.brands
        assert "Kisqali" in vocab.brands
