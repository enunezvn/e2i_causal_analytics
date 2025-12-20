"""
Unit tests for InsightEnricher.

Tests cover:
- LLM response generation
- JSON response parsing
- Caching behavior
- Error handling
- Edge cases
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import json

from src.rag.insight_enricher import InsightEnricher, _RESPONSE_CACHE
from src.rag.models.insight_models import EnrichedInsight
from src.rag.models.retrieval_models import RetrievalResult


@pytest.fixture
def sample_results():
    """Create sample retrieval results for testing."""
    return [
        RetrievalResult(
            content="Kisqali TRx increased 15% in Q4 2024 vs Q3",
            source="agent_activities",
            source_id="act_001",
            score=0.9,
            retrieval_method="dense",
            metadata={"brand": "Kisqali", "timestamp": "2024-12-15T10:00:00Z"},
        ),
        RetrievalResult(
            content="Northeast region shows highest conversion rate at 32%",
            source="business_metrics",
            source_id="met_002",
            score=0.8,
            retrieval_method="sparse",
            metadata={"region": "Northeast"},
        ),
        RetrievalResult(
            content="Top HCPs in oncology segment driving adoption",
            source="triggers",
            source_id="trg_003",
            score=0.7,
            retrieval_method="graph",
            metadata={"category": "oncology"},
        ),
    ]


@pytest.fixture
def mock_anthropic_response():
    """Create mock Anthropic API response."""
    mock_message = MagicMock()
    mock_message.content = [
        MagicMock(text=json.dumps({
            "summary": "Kisqali TRx grew 15% in Q4 2024. Northeast region leads with 32% conversion.",
            "key_findings": [
                "TRx increased 15% quarter-over-quarter",
                "Northeast region has highest conversion at 32%",
                "Top oncology HCPs are key adoption drivers"
            ],
            "confidence": 0.85
        }))
    ]
    mock_message.usage = MagicMock(output_tokens=150)
    return mock_message


@pytest.fixture
def enricher():
    """Create InsightEnricher instance for testing."""
    return InsightEnricher(cache_enabled=False)


class TestInsightEnricherInit:
    """Tests for InsightEnricher initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        enricher = InsightEnricher()
        assert enricher.model == "claude-sonnet-4-20250514"
        assert enricher.max_tokens == 1024
        assert enricher.temperature == 0.3
        assert enricher.cache_enabled is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        enricher = InsightEnricher(
            model="claude-3-haiku-20240307",
            max_tokens=512,
            temperature=0.5,
            cache_enabled=False,
        )
        assert enricher.model == "claude-3-haiku-20240307"
        assert enricher.max_tokens == 512
        assert enricher.temperature == 0.5
        assert enricher.cache_enabled is False


class TestInsightEnricherEnrich:
    """Tests for the enrich method."""

    def setup_method(self):
        """Clear cache before each test."""
        _RESPONSE_CACHE.clear()

    @pytest.mark.asyncio
    async def test_enrich_empty_results(self, enricher):
        """Test enrichment with empty input returns appropriate response."""
        result = await enricher.enrich([], "test query")

        assert result.summary == "No relevant insights found for this query."
        assert result.key_findings == []
        assert result.confidence == 0.0
        assert result.supporting_evidence == []

    @pytest.mark.asyncio
    async def test_enrich_basic(self, sample_results, mock_anthropic_response):
        """Test basic enrichment flow."""
        enricher = InsightEnricher(cache_enabled=False)

        with patch.object(enricher, '_client', create=True) as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_anthropic_response)
            enricher._client = mock_client

            result = await enricher.enrich(sample_results, "What is TRx trend for Kisqali?")

            assert "Kisqali" in result.summary
            assert len(result.key_findings) == 3
            assert result.confidence == 0.85
            assert len(result.supporting_evidence) <= 5

    @pytest.mark.asyncio
    async def test_enrich_with_parsed_query_object(self, sample_results, mock_anthropic_response):
        """Test enrichment with ParsedQuery-like object."""
        enricher = InsightEnricher(cache_enabled=False)

        query = Mock()
        query.text = "parsed query text"

        with patch.object(enricher, '_client', create=True) as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_anthropic_response)
            enricher._client = mock_client

            result = await enricher.enrich(sample_results, query)

            assert result.summary != ""

    @pytest.mark.asyncio
    async def test_enrich_caching(self, sample_results, mock_anthropic_response):
        """Test that results are cached when cache_enabled is True."""
        enricher = InsightEnricher(cache_enabled=True)

        with patch.object(enricher, '_client', create=True) as mock_client:
            mock_client.messages.create = AsyncMock(return_value=mock_anthropic_response)
            enricher._client = mock_client

            # First call
            result1 = await enricher.enrich(sample_results, "test query")

            # Second call with same query
            result2 = await enricher.enrich(sample_results, "test query")

            # API should only be called once
            assert mock_client.messages.create.call_count == 1
            assert result1.summary == result2.summary

    @pytest.mark.asyncio
    async def test_enrich_api_error_fallback(self, sample_results):
        """Test graceful fallback on API error."""
        enricher = InsightEnricher(cache_enabled=False)

        with patch.object(enricher, '_generate', AsyncMock(side_effect=Exception("API Error"))):
            result = await enricher.enrich(sample_results, "test query")

            assert "Unable to synthesize" in result.summary
            assert result.confidence == 0.0
            assert len(result.supporting_evidence) == 3


class TestInsightEnricherParseResponse:
    """Tests for response parsing."""

    def test_parse_valid_json(self, sample_results):
        """Test parsing valid JSON response."""
        enricher = InsightEnricher()

        response = json.dumps({
            "summary": "Test summary",
            "key_findings": ["Finding 1", "Finding 2"],
            "confidence": 0.9
        })

        result = enricher._parse_response(response, sample_results)

        assert result.summary == "Test summary"
        assert result.key_findings == ["Finding 1", "Finding 2"]
        assert result.confidence == 0.9

    def test_parse_json_with_surrounding_text(self, sample_results):
        """Test parsing JSON embedded in surrounding text."""
        enricher = InsightEnricher()

        response = """Here is the analysis:
        {"summary": "Embedded summary", "key_findings": ["F1"], "confidence": 0.8}
        End of analysis."""

        result = enricher._parse_response(response, sample_results)

        assert result.summary == "Embedded summary"
        assert result.key_findings == ["F1"]
        assert result.confidence == 0.8

    def test_parse_invalid_json_fallback(self, sample_results):
        """Test fallback for invalid JSON."""
        enricher = InsightEnricher()

        response = "This is not valid JSON but still useful information."

        result = enricher._parse_response(response, sample_results)

        assert response[:100] in result.summary
        assert result.key_findings == []
        assert result.confidence == 0.3  # Fallback confidence

    def test_parse_empty_response(self, sample_results):
        """Test handling of empty response."""
        enricher = InsightEnricher()

        result = enricher._parse_response("", sample_results)

        assert "No response" in result.summary
        assert result.confidence == 0.0

    def test_parse_confidence_clamping(self, sample_results):
        """Test that confidence is clamped to [0, 1]."""
        enricher = InsightEnricher()

        # Confidence > 1
        response = json.dumps({
            "summary": "Test",
            "key_findings": [],
            "confidence": 1.5
        })
        result = enricher._parse_response(response, sample_results)
        assert result.confidence == 1.0

        # Confidence < 0
        response = json.dumps({
            "summary": "Test",
            "key_findings": [],
            "confidence": -0.5
        })
        result = enricher._parse_response(response, sample_results)
        assert result.confidence == 0.0

    def test_parse_non_list_findings(self, sample_results):
        """Test handling of non-list key_findings."""
        enricher = InsightEnricher()

        response = json.dumps({
            "summary": "Test",
            "key_findings": "Single finding as string",
            "confidence": 0.7
        })

        result = enricher._parse_response(response, sample_results)

        assert isinstance(result.key_findings, list)
        assert "Single finding" in result.key_findings[0]


class TestInsightEnricherBuildPrompt:
    """Tests for prompt building."""

    def test_build_prompt_includes_query(self):
        """Test that prompt includes the query."""
        enricher = InsightEnricher()

        prompt = enricher._build_prompt(
            "What is TRx for Kisqali?",
            "Context here",
            5
        )

        assert "What is TRx for Kisqali?" in prompt

    def test_build_prompt_includes_context(self):
        """Test that prompt includes the context."""
        enricher = InsightEnricher()

        prompt = enricher._build_prompt(
            "Query",
            "This is the retrieved context",
            5
        )

        assert "This is the retrieved context" in prompt

    def test_build_prompt_includes_max_findings(self):
        """Test that prompt includes max_findings."""
        enricher = InsightEnricher()

        prompt = enricher._build_prompt("Query", "Context", 3)

        assert "3" in prompt

    def test_build_prompt_includes_json_format(self):
        """Test that prompt specifies JSON format."""
        enricher = InsightEnricher()

        prompt = enricher._build_prompt("Query", "Context", 5)

        assert "JSON" in prompt
        assert "summary" in prompt
        assert "key_findings" in prompt
        assert "confidence" in prompt


class TestInsightEnricherExtractFreshness:
    """Tests for timestamp extraction."""

    def test_extract_freshness_from_metadata(self):
        """Test extracting timestamp from metadata."""
        enricher = InsightEnricher()

        results = [
            RetrievalResult(
                content="Test",
                source="test",
                source_id="1",
                score=0.5,
                retrieval_method="dense",
                metadata={"timestamp": "2024-12-15T10:00:00Z"},
            )
        ]

        freshness = enricher._extract_freshness(results)

        assert freshness.year == 2024
        assert freshness.month == 12
        assert freshness.day == 15

    def test_extract_freshness_latest(self):
        """Test that latest timestamp is returned."""
        enricher = InsightEnricher()

        results = [
            RetrievalResult(
                content="Older",
                source="test",
                source_id="1",
                score=0.5,
                retrieval_method="dense",
                metadata={"timestamp": "2024-12-10T10:00:00Z"},
            ),
            RetrievalResult(
                content="Newer",
                source="test",
                source_id="2",
                score=0.5,
                retrieval_method="dense",
                metadata={"timestamp": "2024-12-20T10:00:00Z"},
            ),
        ]

        freshness = enricher._extract_freshness(results)

        assert freshness.day == 20

    def test_extract_freshness_no_timestamp(self):
        """Test fallback when no timestamp available."""
        enricher = InsightEnricher()

        results = [
            RetrievalResult(
                content="No timestamp",
                source="test",
                source_id="1",
                score=0.5,
                retrieval_method="dense",
                metadata={},
            )
        ]

        freshness = enricher._extract_freshness(results)

        # Should return current time as fallback
        assert freshness is not None
        assert isinstance(freshness, datetime)


class TestInsightEnricherCaching:
    """Tests for caching behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        _RESPONSE_CACHE.clear()

    def test_cache_key_generation(self):
        """Test cache key generation."""
        enricher = InsightEnricher()

        results = [
            RetrievalResult(
                content="Test",
                source="test",
                source_id="unique_id_123",
                score=0.5,
                retrieval_method="dense",
            )
        ]

        key1 = enricher._build_cache_key("query1", results)
        key2 = enricher._build_cache_key("query1", results)
        key3 = enricher._build_cache_key("query2", results)

        assert key1 == key2  # Same query, same results
        assert key1 != key3  # Different query

    def test_cache_eviction(self):
        """Test cache eviction when max size reached."""
        enricher = InsightEnricher()

        # Fill cache
        for i in range(105):  # Exceed max of 100
            insight = EnrichedInsight(
                summary=f"Test {i}",
                key_findings=[],
                supporting_evidence=[],
                confidence=0.5,
                data_freshness=None,
            )
            enricher._cache_response(f"key_{i}", insight)

        # Cache should be at max size
        assert len(_RESPONSE_CACHE) <= 100
