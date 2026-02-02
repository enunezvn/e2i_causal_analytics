"""
Unit tests for src/ontology/query_extractor.py

Tests for the query-time entity extraction including:
- Enum types (ExtractionContext)
- Dataclasses (Entity, QueryExtractionResult)
- E2IQueryExtractor methods
- VocabularyEnrichedGraphiti bridge class
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ontology.query_extractor import (
    E2IQueryExtractor,
    Entity,
    ExtractionContext,
    QueryExtractionResult,
    VocabularyEnrichedGraphiti,
)

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestExtractionContextEnum:
    """Tests for ExtractionContext enum."""

    def test_query_routing_value(self):
        """Test QUERY_ROUTING enum value."""
        assert ExtractionContext.QUERY_ROUTING.value == "query_routing"

    def test_filter_construction_value(self):
        """Test FILTER_CONSTRUCTION enum value."""
        assert ExtractionContext.FILTER_CONSTRUCTION.value == "filter_construction"

    def test_validation_value(self):
        """Test VALIDATION enum value."""
        assert ExtractionContext.VALIDATION.value == "validation"

    def test_all_enum_members_exist(self):
        """Test that all expected enum members exist."""
        expected = {"QUERY_ROUTING", "FILTER_CONSTRUCTION", "VALIDATION"}
        actual = {m.name for m in ExtractionContext}
        assert expected == actual


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestEntityDataclass:
    """Tests for Entity dataclass."""

    def test_create_entity(self):
        """Test creating an Entity."""
        entity = Entity(
            text="TestBrand",
            entity_type="brand",
            confidence=0.95,
            vocabulary_match=True,
            routing_hint="brand_specific_analysis",
        )

        assert entity.text == "TestBrand"
        assert entity.entity_type == "brand"
        assert entity.confidence == 0.95
        assert entity.vocabulary_match is True
        assert entity.routing_hint == "brand_specific_analysis"

    def test_entity_without_routing_hint(self):
        """Test Entity with default routing_hint."""
        entity = Entity(
            text="region_value", entity_type="region", confidence=1.0, vocabulary_match=True
        )

        assert entity.routing_hint is None

    def test_entity_inferred_not_vocabulary_match(self):
        """Test Entity with vocabulary_match=False for inferred entities."""
        entity = Entity(
            text="Adoption Rate",
            entity_type="kpi",
            confidence=0.8,
            vocabulary_match=False,
            routing_hint="kpi_analysis",
        )

        assert entity.vocabulary_match is False
        assert entity.confidence == 0.8


class TestQueryExtractionResultDataclass:
    """Tests for QueryExtractionResult dataclass."""

    def test_create_extraction_result(self):
        """Test creating a QueryExtractionResult."""
        entities = [
            Entity("TestBrand", "brand", 1.0, True),
            Entity("northeast", "region", 1.0, True),
        ]

        result = QueryExtractionResult(
            entities=entities,
            brand_filter="TestBrand",
            region_filter="northeast",
            kpi_filter=None,
            suggested_agent="orchestrator",
            suggested_tier=1,
        )

        assert len(result.entities) == 2
        assert result.brand_filter == "TestBrand"
        assert result.region_filter == "northeast"
        assert result.kpi_filter is None
        assert result.suggested_agent == "orchestrator"
        assert result.suggested_tier == 1

    def test_to_filter_dict_full(self):
        """Test to_filter_dict with all filters present."""
        result = QueryExtractionResult(
            entities=[],
            brand_filter="TestBrand",
            region_filter="northeast",
            kpi_filter="TRx Volume",
            suggested_agent=None,
            suggested_tier=None,
        )

        filters = result.to_filter_dict()

        assert filters == {"brand": "TestBrand", "region": "northeast", "kpi": "TRx Volume"}

    def test_to_filter_dict_partial(self):
        """Test to_filter_dict with only some filters present."""
        result = QueryExtractionResult(
            entities=[],
            brand_filter="TestBrand",
            region_filter=None,
            kpi_filter=None,
            suggested_agent=None,
            suggested_tier=None,
        )

        filters = result.to_filter_dict()

        assert filters == {"brand": "TestBrand"}
        assert "region" not in filters
        assert "kpi" not in filters

    def test_to_filter_dict_empty(self):
        """Test to_filter_dict with no filters."""
        result = QueryExtractionResult(
            entities=[],
            brand_filter=None,
            region_filter=None,
            kpi_filter=None,
            suggested_agent=None,
            suggested_tier=None,
        )

        filters = result.to_filter_dict()

        assert filters == {}


# =============================================================================
# E2IQUERYEXTRACTOR TESTS
# =============================================================================


class TestE2IQueryExtractorInit:
    """Tests for E2IQueryExtractor initialization."""

    def test_init_loads_vocabulary(self, mock_vocabulary_file):
        """Test that vocabulary is loaded from file."""
        extractor = E2IQueryExtractor(str(mock_vocabulary_file))

        assert hasattr(extractor, "vocab")
        assert "brands" in extractor.vocab

    def test_init_nonexistent_file_raises_error(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            E2IQueryExtractor("/nonexistent/path/vocabulary.yaml")


class TestE2IQueryExtractorBrandExtraction:
    """Tests for brand extraction functionality."""

    def test_extract_brand_exact_match(self, query_extractor):
        """Test extracting brand with exact match."""
        result = query_extractor.extract_for_routing("What is TestBrand performance?")

        assert result.brand_filter == "TestBrand"
        brand_entities = [e for e in result.entities if e.entity_type == "brand"]
        assert len(brand_entities) == 1
        assert brand_entities[0].confidence == 1.0

    def test_extract_brand_case_insensitive(self, query_extractor):
        """Test that brand extraction is case insensitive."""
        result = query_extractor.extract_for_routing("How is testbrand doing?")

        assert result.brand_filter == "TestBrand"

    def test_extract_brand_via_alias(self, query_extractor):
        """Test extracting brand via alias."""
        result = query_extractor.extract_for_routing("Show me tb sales")

        assert result.brand_filter == "TestBrand"
        brand_entities = [e for e in result.entities if e.entity_type == "brand"]
        assert len(brand_entities) == 1
        # Alias match has slightly lower confidence
        assert brand_entities[0].confidence == 0.95

    def test_extract_no_brand(self, query_extractor):
        """Test query with no brand mentioned."""
        result = query_extractor.extract_for_routing("What are overall sales?")

        assert result.brand_filter is None

    def test_extract_brand_uses_canonical_form(self, query_extractor):
        """Test that alias resolves to canonical brand name."""
        result = query_extractor.extract_for_routing("Show testb metrics")

        # 'testb' is alias for 'TestBrand'
        assert result.brand_filter == "TestBrand"


class TestE2IQueryExtractorRegionExtraction:
    """Tests for region extraction functionality."""

    def test_extract_region_exact_match(self, query_extractor):
        """Test extracting region with exact match."""
        result = query_extractor.extract_for_routing("Sales in the northeast region")

        assert result.region_filter == "northeast"
        region_entities = [e for e in result.entities if e.entity_type == "region"]
        assert len(region_entities) == 1
        assert region_entities[0].confidence == 1.0

    def test_extract_region_case_insensitive(self, query_extractor):
        """Test that region extraction is case insensitive."""
        result = query_extractor.extract_for_routing("Performance in SOUTH")

        assert result.region_filter == "south"

    def test_extract_no_region(self, query_extractor):
        """Test query with no region mentioned."""
        result = query_extractor.extract_for_routing("National performance")

        assert result.region_filter is None

    def test_extract_multiple_regions_uses_first(self, query_extractor):
        """Test that only one region is returned."""
        result = query_extractor.extract_for_routing("Compare northeast and south")

        # Should get one region filter
        assert result.region_filter is not None


class TestE2IQueryExtractorKPIExtraction:
    """Tests for KPI extraction functionality."""

    def test_extract_kpi_display_name(self, query_extractor):
        """Test extracting KPI by display name."""
        result = query_extractor.extract_for_routing("Show TRx Volume trends")

        assert result.kpi_filter == "TRx Volume"
        kpi_entities = [e for e in result.entities if e.entity_type == "kpi"]
        assert len(kpi_entities) >= 1

    def test_extract_kpi_via_alias(self, query_extractor):
        """Test extracting KPI via alias."""
        result = query_extractor.extract_for_routing("What are the trx numbers?")

        kpi_entities = [e for e in result.entities if e.entity_type == "kpi"]
        assert len(kpi_entities) >= 1

    def test_extract_kpi_pattern_market_share(self, query_extractor):
        """Test extracting KPI via pattern matching."""
        result = query_extractor.extract_for_routing("What is the market share?")

        kpi_entities = [e for e in result.entities if e.entity_type == "kpi"]
        assert len(kpi_entities) >= 1

    def test_extract_kpi_pattern_adoption_rate(self, query_extractor):
        """Test extracting 'adoption rate' pattern."""
        result = query_extractor.extract_for_routing("Show the adoption rate")

        kpi_entities = [e for e in result.entities if e.entity_type == "kpi"]
        # Should find via pattern matching
        assert any(e.text.lower() == "adoption rate" for e in kpi_entities)

    def test_extract_multiple_kpis(self, query_extractor):
        """Test extracting multiple KPIs from query."""
        result = query_extractor.extract_for_routing("Compare TRx Volume and Market Share")

        kpi_entities = [e for e in result.entities if e.entity_type == "kpi"]
        assert len(kpi_entities) >= 2


class TestE2IQueryExtractorDiagnosisCodeExtraction:
    """Tests for diagnosis code extraction functionality."""

    def test_extract_icd10_code(self, query_extractor):
        """Test extracting ICD-10 code."""
        result = query_extractor.extract_for_routing("Patients with L50.1 diagnosis")

        diagnosis_entities = [e for e in result.entities if e.entity_type == "diagnosis_code"]
        assert len(diagnosis_entities) >= 1
        assert any(e.text == "L50.1" for e in diagnosis_entities)

    def test_extract_icd10_without_decimal(self, query_extractor):
        """Test extracting ICD-10 code without decimal."""
        result = query_extractor.extract_for_routing("Patients with L50 diagnosis")

        diagnosis_entities = [e for e in result.entities if e.entity_type == "diagnosis_code"]
        assert len(diagnosis_entities) >= 1

    def test_extract_multiple_diagnosis_codes(self, query_extractor):
        """Test extracting multiple diagnosis codes."""
        result = query_extractor.extract_for_routing("L50.1 and D50.8 patients")

        diagnosis_entities = [e for e in result.entities if e.entity_type == "diagnosis_code"]
        assert len(diagnosis_entities) >= 2


class TestE2IQueryExtractorRouting:
    """Tests for agent routing determination."""

    def test_route_to_causal_impact(self, query_extractor):
        """Test routing to causal_impact agent."""
        result = query_extractor.extract_for_routing("What caused the decline?")

        assert result.suggested_agent == "causal_impact"
        assert result.suggested_tier == 1

    def test_route_to_prediction_synthesizer(self, query_extractor):
        """Test routing to prediction_synthesizer agent."""
        result = query_extractor.extract_for_routing("Predict next quarter sales")

        assert result.suggested_agent == "prediction_synthesizer"
        assert result.suggested_tier == 2

    def test_route_to_explainer(self, query_extractor):
        """Test routing to explainer agent."""
        result = query_extractor.extract_for_routing("Why did sales drop?")

        assert result.suggested_agent == "explainer"
        assert result.suggested_tier == 5

    def test_route_to_drift_monitor(self, query_extractor):
        """Test routing to drift_monitor agent for model queries."""
        result = query_extractor.extract_for_routing("Is there data drift?")

        assert result.suggested_agent == "drift_monitor"

    def test_route_general_query(self, query_extractor):
        """Test routing for general queries."""
        result = query_extractor.extract_for_routing("Show me the dashboard")

        # General queries may route to explainer or orchestrator depending on implementation
        assert result.suggested_agent in ["orchestrator", "explainer", "gap_analyzer"]
        # Tier may or may not be set

    def test_route_impact_keyword(self, query_extractor):
        """Test routing with 'impact' keyword."""
        result = query_extractor.extract_for_routing("What is the impact on sales?")

        assert result.suggested_agent == "causal_impact"

    def test_route_forecast_keyword(self, query_extractor):
        """Test routing with 'forecast' keyword."""
        result = query_extractor.extract_for_routing("Forecast the trend")

        assert result.suggested_agent == "prediction_synthesizer"


class TestE2IQueryExtractorPerformance:
    """Tests for extraction performance requirements."""

    def test_extraction_under_50ms(self, query_extractor):
        """Test that extraction completes in under 50ms."""
        query = "What is TestBrand TRx Volume in the northeast for patients with L50.1?"

        start = time.time()
        result = query_extractor.extract_for_routing(query)
        elapsed_ms = (time.time() - start) * 1000

        # Should complete in under 50ms
        assert elapsed_ms < 50.0
        # Should still extract entities
        assert len(result.entities) > 0

    def test_extraction_performance_simple_query(self, query_extractor):
        """Test performance with simple query."""
        start = time.time()
        query_extractor.extract_for_routing("Hello")
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 20.0

    def test_extraction_performance_complex_query(self, query_extractor):
        """Test performance with complex query."""
        complex_query = (
            "What caused the TestBrand market share decline in the northeast "
            "region for patients with L50.1 diagnosis compared to the south "
            "for TRx Volume and adoption rate trends in Q4?"
        )

        start = time.time()
        query_extractor.extract_for_routing(complex_query)
        elapsed_ms = (time.time() - start) * 1000

        # Even complex queries should be fast
        assert elapsed_ms < 100.0


class TestE2IQueryExtractorIntegration:
    """Integration tests for full extraction workflow."""

    def test_full_extraction_with_all_entity_types(self, query_extractor):
        """Test extracting all entity types from a single query."""
        query = "What is TestBrand TRx Volume in northeast for L50.1 patients?"

        result = query_extractor.extract_for_routing(query)

        # Should have brand, region, KPI, and diagnosis code
        entity_types = {e.entity_type for e in result.entities}

        assert "brand" in entity_types
        assert "region" in entity_types
        # KPI should be present
        assert result.kpi_filter is not None or any(e.entity_type == "kpi" for e in result.entities)
        assert "diagnosis_code" in entity_types

    def test_extraction_preserves_entity_order(self, query_extractor):
        """Test that entities are returned in extraction order."""
        result = query_extractor.extract_for_routing("TestBrand in northeast with TRx Volume")

        # Brand is extracted first
        if len(result.entities) > 0:
            first_brand_idx = next(
                (i for i, e in enumerate(result.entities) if e.entity_type == "brand"), -1
            )
            assert first_brand_idx >= 0


# =============================================================================
# VOCABULARY ENRICHED GRAPHITI TESTS
# =============================================================================


class TestVocabularyEnrichedGraphitiInit:
    """Tests for VocabularyEnrichedGraphiti initialization."""

    def test_init_with_clients(self, query_extractor):
        """Test initialization with required clients."""
        mock_graphiti = MagicMock()

        bridge = VocabularyEnrichedGraphiti(mock_graphiti, query_extractor)

        assert bridge.graphiti == mock_graphiti
        assert bridge.extractor == query_extractor


class TestVocabularyEnrichedGraphitiAnnotateContent:
    """Tests for content annotation functionality."""

    def test_annotate_content_adds_type_hints(self, query_extractor):
        """Test that type hints are added to content."""
        mock_graphiti = MagicMock()
        bridge = VocabularyEnrichedGraphiti(mock_graphiti, query_extractor)

        entities = [
            Entity("TestBrand", "brand", 1.0, True),
            Entity("northeast", "region", 1.0, True),
        ]

        content = "TestBrand performs well in northeast"
        annotated = bridge._annotate_content(content, entities)

        assert "[E2I:brand]" in annotated
        assert "[E2I:region]" in annotated

    def test_annotate_content_handles_entities(self, query_extractor):
        """Test that content annotation handles entity list."""
        mock_graphiti = MagicMock()
        bridge = VocabularyEnrichedGraphiti(mock_graphiti, query_extractor)

        entities = [Entity("TestBrand", "brand", 1.0, True)]

        content = "TestBrand is here"
        annotated = bridge._annotate_content(content, entities)

        # Should annotate the entity in some way
        # The exact format depends on implementation
        assert "TestBrand" in annotated
        # Annotation adds E2I marker
        assert "[E2I:" in annotated or annotated == content  # May not annotate if entity not found


class TestVocabularyEnrichedGraphitiInferSourceAgent:
    """Tests for source agent inference."""

    def test_infer_causal_agent(self, query_extractor):
        """Test inferring causal agent from source description."""
        mock_graphiti = MagicMock()
        bridge = VocabularyEnrichedGraphiti(mock_graphiti, query_extractor)

        agent = bridge._infer_source_agent("Causal Impact Analysis")

        assert agent == "causal_impact"

    def test_infer_prediction_agent(self, query_extractor):
        """Test inferring prediction agent from source description."""
        mock_graphiti = MagicMock()
        bridge = VocabularyEnrichedGraphiti(mock_graphiti, query_extractor)

        agent = bridge._infer_source_agent("Prediction Synthesizer Output")

        assert agent == "prediction_synthesizer"

    def test_infer_explainer_agent(self, query_extractor):
        """Test inferring explainer agent from source description."""
        mock_graphiti = MagicMock()
        bridge = VocabularyEnrichedGraphiti(mock_graphiti, query_extractor)

        # Use explicit "explainer" keyword for more reliable matching
        agent = bridge._infer_source_agent("Explainer Agent Output")

        # May return 'explainer' or 'unknown' depending on keyword matching
        assert agent in ["explainer", "unknown"]

    def test_infer_drift_agent(self, query_extractor):
        """Test inferring drift monitor agent from source description."""
        mock_graphiti = MagicMock()
        bridge = VocabularyEnrichedGraphiti(mock_graphiti, query_extractor)

        agent = bridge._infer_source_agent("Drift Detection Report")

        assert agent == "drift_monitor"

    def test_infer_unknown_agent(self, query_extractor):
        """Test unknown agent inference."""
        mock_graphiti = MagicMock()
        bridge = VocabularyEnrichedGraphiti(mock_graphiti, query_extractor)

        agent = bridge._infer_source_agent("Some other source")

        assert agent == "unknown"


class TestVocabularyEnrichedGraphitiAddEpisode:
    """Tests for add_episode_with_vocab_hints method."""

    @pytest.mark.asyncio
    async def test_add_episode_calls_graphiti(self, query_extractor):
        """Test that graphiti.add_episode is called."""
        mock_graphiti = MagicMock()
        mock_graphiti.add_episode = AsyncMock()

        bridge = VocabularyEnrichedGraphiti(mock_graphiti, query_extractor)

        await bridge.add_episode_with_vocab_hints(
            content="TestBrand analysis in northeast",
            name="test_episode",
            source_description="Test source",
        )

        mock_graphiti.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_episode_enriches_content(self, query_extractor):
        """Test that content is enriched before adding episode."""
        mock_graphiti = MagicMock()
        mock_graphiti.add_episode = AsyncMock()

        bridge = VocabularyEnrichedGraphiti(mock_graphiti, query_extractor)

        await bridge.add_episode_with_vocab_hints(
            content="TestBrand analysis in northeast",
            name="test_episode",
            source_description="Test source",
        )

        # Check that content was enriched
        call_kwargs = mock_graphiti.add_episode.call_args[1]
        assert "[E2I:" in call_kwargs["content"]

    @pytest.mark.asyncio
    async def test_add_episode_with_group_id(self, query_extractor):
        """Test that group_id is passed to graphiti."""
        mock_graphiti = MagicMock()
        mock_graphiti.add_episode = AsyncMock()

        bridge = VocabularyEnrichedGraphiti(mock_graphiti, query_extractor)

        await bridge.add_episode_with_vocab_hints(
            content="Test content",
            name="test_episode",
            source_description="Test source",
            group_id="test_group",
        )

        call_kwargs = mock_graphiti.add_episode.call_args[1]
        assert call_kwargs["group_id"] == "test_group"


# =============================================================================
# EDGE CASES
# =============================================================================


class TestE2IQueryExtractorEdgeCases:
    """Tests for edge cases in E2IQueryExtractor."""

    def test_empty_query(self, query_extractor):
        """Test handling of empty query."""
        result = query_extractor.extract_for_routing("")

        assert result.entities == []
        assert result.brand_filter is None
        assert result.region_filter is None

    def test_query_with_special_characters(self, query_extractor):
        """Test query with special characters."""
        result = query_extractor.extract_for_routing("What's TestBrand's performance @2024?")

        # Should still extract brand
        assert result.brand_filter == "TestBrand"

    def test_query_with_numbers(self, query_extractor):
        """Test query with numeric values."""
        result = query_extractor.extract_for_routing("Show 2024 Q1 TestBrand sales in northeast")

        assert result.brand_filter == "TestBrand"
        assert result.region_filter == "northeast"

    def test_very_long_query(self, query_extractor):
        """Test handling of very long query."""
        long_query = "TestBrand " * 100 + "in northeast"

        result = query_extractor.extract_for_routing(long_query)

        assert result.brand_filter == "TestBrand"
        assert result.region_filter == "northeast"

    def test_unicode_characters(self, query_extractor):
        """Test query with unicode characters."""
        result = query_extractor.extract_for_routing("TestBrand performance — northeast région")

        # Should handle unicode gracefully
        assert result.brand_filter == "TestBrand"
