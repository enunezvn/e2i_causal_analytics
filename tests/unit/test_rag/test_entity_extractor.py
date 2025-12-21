"""
Unit tests for E2I RAG Entity Extractor.

Tests entity extraction from natural language queries using
fixed E2I domain vocabularies.

All tests use vocabulary-based matching with no medical NER.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.rag.entity_extractor import EntityExtractor, EntityVocabulary
from src.rag.types import ExtractedEntities
from src.rag.exceptions import EntityExtractionError


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
        result = extractor.extract_with_confidence(
            "Show Kisqali TRx in West"
        )

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
            entities = extractor.extract(None)  # type: ignore
        except (TypeError, AttributeError):
            pass  # Expected behavior
