"""
Unit tests for QueryOptimizer LLM-enhanced expansion.

Tests cover:
- Rule-based expansion (backwards compatibility)
- LLM-based query expansion
- HyDE document generation
- Caching behavior with TTL
- Fallback on API failure
- Full optimization pipeline
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.rag.query_optimizer import _EXPANSION_CACHE, QueryOptimizer


@pytest.fixture
def optimizer():
    """Create QueryOptimizer instance with caching disabled for isolated tests."""
    return QueryOptimizer(cache_enabled=False)


@pytest.fixture
def cached_optimizer():
    """Create QueryOptimizer instance with caching enabled."""
    return QueryOptimizer(cache_enabled=True)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear cache before each test."""
    _EXPANSION_CACHE.clear()
    yield
    _EXPANSION_CACHE.clear()


@pytest.fixture
def mock_anthropic_response():
    """Create mock Anthropic API response."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Expanded query with additional terms")]
    return mock_response


class TestQueryOptimizerInit:
    """Tests for QueryOptimizer initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        optimizer = QueryOptimizer()
        assert optimizer.model == "claude-sonnet-4-20250514"
        assert optimizer.max_tokens == 256
        assert optimizer.temperature == 0.3
        assert optimizer.cache_enabled is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        optimizer = QueryOptimizer(
            model="claude-3-haiku-20240307",
            max_tokens=128,
            temperature=0.5,
            cache_enabled=False,
        )
        assert optimizer.model == "claude-3-haiku-20240307"
        assert optimizer.max_tokens == 128
        assert optimizer.temperature == 0.5
        assert optimizer.cache_enabled is False

    def test_synonyms_loaded(self):
        """Test that domain synonyms are loaded."""
        optimizer = QueryOptimizer()
        assert "trx" in optimizer._synonyms
        assert "kisqali" in optimizer._synonyms
        assert "hcp" in optimizer._synonyms

    def test_kpi_relations_loaded(self):
        """Test that KPI relations are loaded."""
        optimizer = QueryOptimizer()
        assert "trx" in optimizer._kpi_relations
        assert "nrx" in optimizer._kpi_relations


class TestRuleBasedExpansion:
    """Tests for rule-based query expansion."""

    def test_expand_with_synonyms(self, optimizer):
        """Test expansion adds synonyms for recognized terms."""
        result = optimizer.expand("TRx for Kisqali")

        assert "TRx for Kisqali" in result
        assert "total prescriptions" in result or "prescription volume" in result

    def test_expand_with_kpi_relations(self, optimizer):
        """Test expansion adds related KPIs."""
        result = optimizer.expand("What is the TRx trend?")

        assert "TRx" in result
        # Should include related KPIs
        assert "nrx" in result.lower() or "conversion" in result.lower()

    def test_expand_multiple_terms(self, optimizer):
        """Test expansion with multiple recognized terms."""
        result = optimizer.expand("TRx and NRx for HCP targeting")

        assert "total prescriptions" in result.lower() or "total rx" in result.lower()
        assert "new prescriptions" in result.lower() or "new rx" in result.lower()

    def test_expand_no_match_returns_original(self, optimizer):
        """Test that unrecognized queries return unchanged."""
        query = "What is the weather today?"
        result = optimizer.expand(query)

        assert result == query

    def test_expand_with_parsed_query_object(self, optimizer):
        """Test expansion with ParsedQuery-like object."""
        query = Mock()
        query.text = "Kisqali market share"

        result = optimizer.expand(query)

        assert "Kisqali" in result
        assert "share" in result.lower()

    def test_add_temporal_context(self, optimizer):
        """Test adding temporal context to query."""
        result = optimizer.add_temporal_context("TRx trend", "last 30 days")

        assert "TRx trend" in result
        assert "last 30 days" in result

    def test_add_temporal_context_none(self, optimizer):
        """Test temporal context with None time range."""
        result = optimizer.add_temporal_context("TRx trend", None)

        assert result == "TRx trend"


class TestLLMExpansion:
    """Tests for LLM-based query expansion."""

    def test_expand_with_llm_success(self, optimizer, mock_anthropic_response):
        """Test successful LLM expansion."""
        with patch.object(optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_anthropic_response

            result = optimizer.expand_with_llm("TRx for Kisqali")

            assert result == "Expanded query with additional terms"
            mock_client.return_value.messages.create.assert_called_once()

    def test_expand_with_llm_fallback_on_error(self, optimizer):
        """Test fallback to rule-based on API error."""
        with patch.object(optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = Exception("API Error")

            result = optimizer.expand_with_llm("TRx for Kisqali")

            # Should fall back to rule-based expansion
            assert "TRx for Kisqali" in result
            assert "total prescriptions" in result.lower() or "prescription" in result.lower()

    def test_expand_with_llm_strips_quotes(self, optimizer):
        """Test that quotes are stripped from LLM response."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='"Quoted expanded query"')]

        with patch.object(optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_response

            result = optimizer.expand_with_llm("Test query")

            assert result == "Quoted expanded query"
            assert '"' not in result

    def test_expand_with_llm_with_context(self, optimizer, mock_anthropic_response):
        """Test LLM expansion with conversation context."""
        with patch.object(optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_anthropic_response

            optimizer.expand_with_llm(
                "What about Q4?", context="Previous discussion about Kisqali TRx trends"
            )

            # Verify context was included in prompt
            call_args = mock_client.return_value.messages.create.call_args
            prompt = call_args[1]["messages"][0]["content"]
            assert "Previous discussion" in prompt

    def test_expand_with_llm_missing_api_key(self):
        """Test error when API key is missing."""
        import os

        original_key = os.environ.get("ANTHROPIC_API_KEY")

        try:
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

            optimizer = QueryOptimizer(cache_enabled=False)

            # Should fall back to rule-based expansion
            result = optimizer.expand_with_llm("TRx for Kisqali")

            # Fallback should work
            assert "TRx" in result

        finally:
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    @pytest.mark.asyncio
    async def test_expand_with_llm_async(self, optimizer, mock_anthropic_response):
        """Test async LLM expansion."""
        with patch.object(optimizer, "_call_llm", return_value="Async expanded query"):
            result = await optimizer.expand_with_llm_async("TRx for Kisqali")

            assert result == "Async expanded query"

    @pytest.mark.asyncio
    async def test_expand_with_llm_async_fallback(self, optimizer):
        """Test async fallback to rule-based on error."""
        with patch.object(optimizer, "_call_llm", side_effect=Exception("API Error")):
            result = await optimizer.expand_with_llm_async("TRx for Kisqali")

            # Should fall back to rule-based
            assert "TRx" in result


class TestHyDEGeneration:
    """Tests for HyDE (Hypothetical Document Embeddings) generation."""

    def test_generate_hyde_document_success(self, optimizer):
        """Test successful HyDE document generation."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text="Kisqali TRx increased 15% in Q4 2024 across the Northeast region.")
        ]

        with patch.object(optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_response

            result = optimizer.generate_hyde_document("What is Kisqali TRx?")

            assert "Kisqali" in result
            assert "TRx" in result or "15%" in result

    def test_generate_hyde_document_types(self, optimizer):
        """Test HyDE with different document types."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated document")]

        with patch.object(optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_response

            # Test insight type
            optimizer.generate_hyde_document("Test", document_type="insight")
            prompt = mock_client.return_value.messages.create.call_args[1]["messages"][0]["content"]
            assert "insight" in prompt.lower()

            # Test report type
            optimizer.generate_hyde_document("Test", document_type="report")
            prompt = mock_client.return_value.messages.create.call_args[1]["messages"][0]["content"]
            assert "report" in prompt.lower() or "summary" in prompt.lower()

    def test_generate_hyde_fallback_on_error(self, optimizer):
        """Test HyDE fallback to expanded query on error."""
        with patch.object(optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.side_effect = Exception("API Error")

            result = optimizer.generate_hyde_document("TRx for Kisqali")

            # Should fall back to rule-based expansion
            assert "TRx" in result

    @pytest.mark.asyncio
    async def test_generate_hyde_async(self, optimizer):
        """Test async HyDE generation."""
        with patch.object(optimizer, "_call_llm", return_value="Async HyDE document"):
            result = await optimizer.generate_hyde_document_async("TRx for Kisqali")

            assert result == "Async HyDE document"


class TestCaching:
    """Tests for query expansion caching."""

    def test_cache_hit_returns_cached_result(self, cached_optimizer, mock_anthropic_response):
        """Test that cache hits return cached result without API call."""
        with patch.object(cached_optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_anthropic_response

            # First call - should hit API
            result1 = cached_optimizer.expand_with_llm("TRx for Kisqali")
            assert mock_client.return_value.messages.create.call_count == 1

            # Second call - should use cache
            result2 = cached_optimizer.expand_with_llm("TRx for Kisqali")
            assert mock_client.return_value.messages.create.call_count == 1  # Still 1

            assert result1 == result2

    def test_cache_disabled_always_calls_api(self, optimizer, mock_anthropic_response):
        """Test that cache disabled always calls API."""
        with patch.object(optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_anthropic_response

            optimizer.expand_with_llm("TRx for Kisqali")
            optimizer.expand_with_llm("TRx for Kisqali")

            assert mock_client.return_value.messages.create.call_count == 2

    def test_cache_different_queries(self, cached_optimizer, mock_anthropic_response):
        """Test that different queries have separate cache entries."""
        with patch.object(cached_optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_anthropic_response

            cached_optimizer.expand_with_llm("Query A")
            cached_optimizer.expand_with_llm("Query B")

            assert mock_client.return_value.messages.create.call_count == 2

    def test_cache_key_generation(self, cached_optimizer):
        """Test cache key generation is consistent."""
        key1 = cached_optimizer._build_cache_key("Test query", "llm_expand")
        key2 = cached_optimizer._build_cache_key("Test query", "llm_expand")
        key3 = cached_optimizer._build_cache_key("Different query", "llm_expand")
        key4 = cached_optimizer._build_cache_key("Test query", "hyde")

        assert key1 == key2  # Same query, same type
        assert key1 != key3  # Different query
        assert key1 != key4  # Same query, different type

    def test_cache_ttl_expiration(self, cached_optimizer, mock_anthropic_response):
        """Test that cached entries expire after TTL."""
        with patch.object(cached_optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_anthropic_response

            # First call
            cached_optimizer.expand_with_llm("TRx for Kisqali")

            # Manually expire the cache entry
            cache_key = cached_optimizer._build_cache_key("TRx for Kisqali", "llm_expand")
            _EXPANSION_CACHE[cache_key]["timestamp"] = time.time() - 4000  # Expired

            # Second call should hit API due to expiration
            cached_optimizer.expand_with_llm("TRx for Kisqali")

            assert mock_client.return_value.messages.create.call_count == 2

    def test_cache_eviction_on_max_size(self, cached_optimizer):
        """Test cache eviction when max size is reached."""
        # Fill cache beyond max size
        from src.rag.query_optimizer import _CACHE_MAX_SIZE

        for i in range(_CACHE_MAX_SIZE + 10):
            cached_optimizer._cache_result(f"key_{i}", f"result_{i}")

        # Cache should be at max size
        assert len(_EXPANSION_CACHE) <= _CACHE_MAX_SIZE

    def test_clear_cache(self, cached_optimizer):
        """Test cache clearing."""
        cached_optimizer._cache_result("key1", "result1")
        cached_optimizer._cache_result("key2", "result2")

        assert len(_EXPANSION_CACHE) == 2

        count = cached_optimizer.clear_cache()

        assert count == 2
        assert len(_EXPANSION_CACHE) == 0


class TestOptimizeQuery:
    """Tests for full optimization pipeline."""

    @pytest.mark.asyncio
    async def test_optimize_query_rule_only(self, optimizer):
        """Test optimization with rule-based only."""
        result = await optimizer.optimize_query("TRx for Kisqali", use_llm=False, use_hyde=False)

        assert result["original"] == "TRx for Kisqali"
        assert (
            "total prescriptions" in result["rule_expanded"].lower()
            or "prescription" in result["rule_expanded"].lower()
        )
        assert result["llm_expanded"] is None
        assert result["hyde_document"] is None
        assert result["recommended"] == result["rule_expanded"]

    @pytest.mark.asyncio
    async def test_optimize_query_with_llm(self, optimizer):
        """Test optimization with LLM expansion."""
        with patch.object(optimizer, "_call_llm", return_value="LLM expanded result"):
            result = await optimizer.optimize_query("TRx for Kisqali", use_llm=True, use_hyde=False)

            assert result["llm_expanded"] == "LLM expanded result"
            assert result["hyde_document"] is None
            assert result["recommended"] == "LLM expanded result"

    @pytest.mark.asyncio
    async def test_optimize_query_with_hyde(self, optimizer):
        """Test optimization with HyDE."""
        with patch.object(optimizer, "_call_llm", return_value="HyDE document"):
            result = await optimizer.optimize_query("TRx for Kisqali", use_llm=False, use_hyde=True)

            assert result["llm_expanded"] is None
            assert result["hyde_document"] == "HyDE document"
            assert result["recommended"] == "HyDE document"

    @pytest.mark.asyncio
    async def test_optimize_query_hyde_preferred_over_llm(self, optimizer):
        """Test that HyDE is preferred over LLM when both enabled."""
        call_count = 0

        def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            if "Hypothetical" in prompt or "insight" in prompt.lower():
                return "HyDE document"
            return "LLM expanded"

        with patch.object(optimizer, "_call_llm", side_effect=mock_llm):
            result = await optimizer.optimize_query("TRx for Kisqali", use_llm=True, use_hyde=True)

            assert result["hyde_document"] is not None
            assert result["recommended"] == result["hyde_document"]

    @pytest.mark.asyncio
    async def test_optimize_query_with_context(self, optimizer):
        """Test optimization passes context to LLM."""
        with patch.object(optimizer, "_call_llm") as mock_llm:
            mock_llm.return_value = "Contextual expansion"

            await optimizer.optimize_query(
                "What about Q4?", use_llm=True, context="Previous discussion about Kisqali"
            )

            # Verify context was included
            prompt = mock_llm.call_args[0][0]
            assert "Previous discussion" in prompt


class TestPromptBuilding:
    """Tests for prompt building methods."""

    def test_expansion_prompt_includes_query(self, optimizer):
        """Test expansion prompt includes the query."""
        prompt = optimizer._build_expansion_prompt("TRx for Kisqali")

        assert "TRx for Kisqali" in prompt
        assert "pharmaceutical" in prompt.lower()

    def test_expansion_prompt_includes_context(self, optimizer):
        """Test expansion prompt includes context when provided."""
        prompt = optimizer._build_expansion_prompt(
            "What about Q4?", context="Previous discussion about Kisqali"
        )

        assert "Previous discussion" in prompt

    def test_hyde_prompt_includes_query(self, optimizer):
        """Test HyDE prompt includes the query."""
        prompt = optimizer._build_hyde_prompt("What is Kisqali TRx?", "insight")

        assert "Kisqali TRx" in prompt
        assert "insight" in prompt.lower()

    def test_hyde_prompt_document_types(self, optimizer):
        """Test HyDE prompt varies by document type."""
        insight_prompt = optimizer._build_hyde_prompt("Test", "insight")
        report_prompt = optimizer._build_hyde_prompt("Test", "report")
        analysis_prompt = optimizer._build_hyde_prompt("Test", "analysis")

        assert "insight" in insight_prompt.lower()
        assert "summary" in report_prompt.lower() or "report" in report_prompt.lower()
        assert "analysis" in analysis_prompt.lower()


class TestParsedQueryObject:
    """Tests for handling ParsedQuery objects."""

    def test_expand_with_parsed_query(self, optimizer):
        """Test rule-based expansion with ParsedQuery object."""
        query = Mock()
        query.text = "Kisqali TRx trend"

        result = optimizer.expand(query)

        assert "Kisqali" in result

    def test_llm_expand_with_parsed_query(self, optimizer, mock_anthropic_response):
        """Test LLM expansion with ParsedQuery object."""
        query = Mock()
        query.text = "Kisqali TRx trend"

        with patch.object(optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_anthropic_response

            result = optimizer.expand_with_llm(query)

            assert result == "Expanded query with additional terms"

    def test_hyde_with_parsed_query(self, optimizer):
        """Test HyDE with ParsedQuery object."""
        query = Mock()
        query.text = "Kisqali TRx trend"

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="HyDE document")]

        with patch.object(optimizer, "_get_client") as mock_client:
            mock_client.return_value.messages.create.return_value = mock_response

            result = optimizer.generate_hyde_document(query)

            assert result == "HyDE document"


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_query(self, optimizer):
        """Test handling of empty query."""
        result = optimizer.expand("")
        assert result == ""

    def test_whitespace_query(self, optimizer):
        """Test handling of whitespace-only query."""
        result = optimizer.expand("   ")
        assert result.strip() == ""

    def test_special_characters(self, optimizer):
        """Test handling of special characters in query."""
        result = optimizer.expand("What is TRx for 'Kisqali'?")

        assert "Kisqali" in result

    def test_unicode_query(self, optimizer):
        """Test handling of unicode characters."""
        result = optimizer.expand("TRx für Kisqali")

        assert "Kisqali" in result

    def test_very_long_query(self, optimizer):
        """Test handling of very long query."""
        long_query = "What is " + "TRx " * 100 + "for Kisqali?"
        result = optimizer.expand(long_query)

        assert "Kisqali" in result
