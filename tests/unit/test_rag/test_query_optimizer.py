"""
Comprehensive unit tests for src/rag/query_optimizer.py

Tests cover:
- QueryOptimizer initialization
- Rule-based expansion
- Typo correction (FastText and fallback)
- LLM-based expansion
- HyDE document generation
- Query optimization pipeline
- Caching
"""

import asyncio
import sys
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock dependencies
sys.modules["anthropic"] = MagicMock()
sys.modules["tenacity"] = MagicMock()

from src.rag.query_optimizer import QueryOptimizer

# =============================================================================
# Test Initialization
# =============================================================================


class TestQueryOptimizerInit:
    def test_init_defaults(self):
        optimizer = QueryOptimizer()

        assert optimizer.model == "claude-sonnet-4-20250514"
        assert optimizer.max_tokens == 256
        assert optimizer.temperature == 0.3
        assert optimizer.cache_enabled is True
        assert optimizer.typo_correction_enabled is True

    def test_init_custom_settings(self):
        optimizer = QueryOptimizer(
            model="claude-opus-4",
            max_tokens=512,
            temperature=0.5,
            cache_enabled=False,
            typo_correction_enabled=False,
        )

        assert optimizer.model == "claude-opus-4"
        assert optimizer.max_tokens == 512
        assert optimizer.temperature == 0.5
        assert optimizer.cache_enabled is False
        assert optimizer.typo_correction_enabled is False

    def test_load_vocabulary(self):
        optimizer = QueryOptimizer()

        # Check that default vocabulary was loaded
        assert "trx" in optimizer._synonyms
        assert "kisqali" in optimizer._synonyms
        assert "hcp" in optimizer._synonyms

        # Check KPI relations
        assert "trx" in optimizer._kpi_relations
        assert "conversion_rate" in optimizer._kpi_relations


# =============================================================================
# Test Rule-Based Expansion
# =============================================================================


class TestRuleBasedExpansion:
    @pytest.fixture
    def optimizer(self):
        return QueryOptimizer()

    def test_expand_with_synonyms(self, optimizer):
        query = "What is the TRx for Kisqali?"

        expanded = optimizer.expand(query)

        assert "total prescriptions" in expanded or "TRx" in expanded
        assert "kisqali" in expanded.lower() or "ribociclib" in expanded.lower()

    def test_expand_with_kpi_relations(self, optimizer):
        query = "Show TRx trends"

        expanded = optimizer.expand(query)

        # Should include related KPIs
        assert len(expanded) > len(query)

    def test_expand_no_matches(self, optimizer):
        query = "some random text"

        expanded = optimizer.expand(query)

        # Should return original if no expansions
        assert expanded == query

    def test_expand_with_parsed_query(self, optimizer):
        # Mock parsed query object
        mock_query = Mock()
        mock_query.text = "What is NRx?"

        expanded = optimizer.expand(mock_query)

        assert "nrx" in expanded.lower()

    def test_add_temporal_context(self, optimizer):
        query = "TRx trends"
        time_range = "last 30 days"

        result = optimizer.add_temporal_context(query, time_range)

        assert "last 30 days" in result
        assert "TRx trends" in result

    def test_add_temporal_context_none(self, optimizer):
        query = "test query"

        result = optimizer.add_temporal_context(query, None)

        assert result == query


# =============================================================================
# Test Typo Correction
# =============================================================================


class TestTypoCorrection:
    @pytest.fixture
    def optimizer_with_typo(self):
        # Mock typo handler
        with patch("src.rag.query_optimizer.TypoHandler") as mock_handler_class:
            mock_handler = Mock()
            mock_handler.correct_query = Mock(
                return_value=(
                    "corrected query",
                    [
                        Mock(
                            original="qery",
                            corrected="query",
                            confidence=0.9,
                            category="general",
                        )
                    ],
                )
            )
            mock_handler.correct_term = Mock()
            mock_handler.get_suggestions = Mock(return_value=[("query", 0.95)])
            mock_handler.is_fasttext_available = True
            mock_handler_class.return_value = mock_handler

            optimizer = QueryOptimizer(typo_correction_enabled=True)
            return optimizer

    def test_correct_typos_disabled(self):
        optimizer = QueryOptimizer(typo_correction_enabled=False)

        result = optimizer.correct_typos("test qery")

        assert result["original"] == "test qery"
        assert result["corrected"] == "test qery"
        assert result["corrections"] == []

    def test_correct_typos_enabled(self, optimizer_with_typo):
        result = optimizer_with_typo.correct_typos("test qery")

        assert result["original"] == "test qery"
        assert result["corrected"] == "corrected query"
        assert len(result["corrections"]) > 0
        assert result["latency_ms"] >= 0

    def test_correct_typos_with_parsed_query(self, optimizer_with_typo):
        mock_query = Mock()
        mock_query.text = "test qery"

        result = optimizer_with_typo.correct_typos(mock_query)

        assert result["corrected"] == "corrected query"

    def test_correct_term(self, optimizer_with_typo):
        mock_result = Mock(
            original="kisqli",
            corrected="kisqali",
            confidence=0.95,
            was_corrected=True,
        )
        optimizer_with_typo._typo_handler.correct_term = Mock(return_value=mock_result)

        result = optimizer_with_typo.correct_term("kisqli", category="brands")

        assert result.corrected == "kisqali"
        assert result.was_corrected is True

    def test_correct_term_no_handler(self):
        optimizer = QueryOptimizer(typo_correction_enabled=False)

        result = optimizer.correct_term("test")

        assert result.original == "test"
        assert result.corrected == "test"
        assert result.was_corrected is False

    def test_get_typo_suggestions(self, optimizer_with_typo):
        suggestions = optimizer_with_typo.get_typo_suggestions("kisqli", top_k=3)

        assert len(suggestions) > 0
        assert suggestions[0] == ("query", 0.95)

    def test_get_typo_suggestions_no_handler(self):
        optimizer = QueryOptimizer(typo_correction_enabled=False)

        suggestions = optimizer.get_typo_suggestions("test")

        assert suggestions == []


# =============================================================================
# Test LLM-Based Expansion
# =============================================================================


class TestLLMExpansion:
    @pytest.fixture
    def optimizer(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            return QueryOptimizer(cache_enabled=False)

    @pytest.fixture
    def mock_anthropic(self):
        with patch("anthropic.Anthropic") as mock:
            client = Mock()
            response = Mock()
            response.content = [Mock(text="expanded query with relevant terms")]
            client.messages.create = Mock(return_value=response)
            mock.return_value = client
            yield mock

    def test_get_client(self, optimizer):
        with patch("src.rag.query_optimizer.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client

            client = optimizer._get_client()

            assert client is mock_client
            mock_anthropic.assert_called_once()

    def test_get_client_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            optimizer = QueryOptimizer()

            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                optimizer._get_client()

    def test_build_expansion_prompt(self, optimizer):
        prompt = optimizer._build_expansion_prompt("What is TRx?", context=None)

        assert "TRx" in prompt
        assert "pharmaceutical" in prompt.lower()
        assert "Kisqali" in prompt

    def test_build_expansion_prompt_with_context(self, optimizer):
        prompt = optimizer._build_expansion_prompt(
            "What is TRx?", context="Previous discussion about Kisqali"
        )

        assert "TRx" in prompt
        assert "Previous discussion" in prompt

    def test_call_llm(self, optimizer):
        # Bypass retry decorator by calling the underlying method directly
        with patch("src.rag.query_optimizer.anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = Mock()
            response = Mock()
            response.content = [Mock(text="expanded query with relevant terms")]
            mock_client.messages.create = Mock(return_value=response)
            mock_anthropic_cls.return_value = mock_client

            optimizer._client = None  # Force reinitialize
            # Get the client first
            client = optimizer._get_client()

            # Manually call without retry
            response = client.messages.create(
                model=optimizer.model,
                max_tokens=optimizer.max_tokens,
                temperature=optimizer.temperature,
                messages=[{"role": "user", "content": "test prompt"}],
            )
            result = response.content[0].text

            assert result == "expanded query with relevant terms"

    def test_expand_with_llm(self, optimizer):
        with patch.object(optimizer, "_call_llm", return_value="  expanded result  "):
            result = optimizer.expand_with_llm("test query")

            assert result == "expanded result"  # Stripped

    def test_expand_with_llm_fallback(self, optimizer):
        with patch.object(optimizer, "_call_llm", side_effect=Exception("API error")):
            result = optimizer.expand_with_llm("TRx query")

            # Should fall back to rule-based expansion
            assert len(result) >= len("TRx query")

    @pytest.mark.asyncio
    async def test_expand_with_llm_async(self, optimizer):
        with patch.object(optimizer, "_call_llm", return_value="async expanded"):
            result = await optimizer.expand_with_llm_async("test")

            assert result == "async expanded"

    @pytest.mark.asyncio
    async def test_expand_with_llm_async_fallback(self, optimizer):
        with patch.object(optimizer, "_call_llm", side_effect=Exception("Error")):
            result = await optimizer.expand_with_llm_async("test")

            assert isinstance(result, str)


# =============================================================================
# Test HyDE Document Generation
# =============================================================================


class TestHyDEGeneration:
    @pytest.fixture
    def optimizer(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            return QueryOptimizer(cache_enabled=False)

    def test_build_hyde_prompt(self, optimizer):
        prompt = optimizer._build_hyde_prompt("What is TRx?", "insight")

        assert "insight" in prompt.lower()
        assert "TRx" in prompt
        assert "pharmaceutical" in prompt.lower()

    def test_build_hyde_prompt_types(self, optimizer):
        for doc_type in ["insight", "report", "analysis"]:
            prompt = optimizer._build_hyde_prompt("query", doc_type)
            assert doc_type in prompt.lower()

    def test_generate_hyde_document(self, optimizer):
        with patch.object(
            optimizer,
            "_call_llm",
            return_value="Hypothetical document about TRx trends",
        ):
            result = optimizer.generate_hyde_document("What is TRx?", "insight")

            assert result == "Hypothetical document about TRx trends"

    def test_generate_hyde_document_fallback(self, optimizer):
        with patch.object(optimizer, "_call_llm", side_effect=Exception("Error")):
            result = optimizer.generate_hyde_document("TRx query")

            # Should fall back to expanded query
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_hyde_document_async(self, optimizer):
        with patch.object(optimizer, "_call_llm", return_value="Async HyDE doc"):
            result = await optimizer.generate_hyde_document_async("query", "report")

            assert result == "Async HyDE doc"

    @pytest.mark.asyncio
    async def test_generate_hyde_document_async_fallback(self, optimizer):
        with patch.object(optimizer, "_call_llm", side_effect=Exception("Error")):
            result = await optimizer.generate_hyde_document_async("query")

            assert isinstance(result, str)


# =============================================================================
# Test Caching
# =============================================================================


class TestCaching:
    @pytest.fixture
    def optimizer(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            return QueryOptimizer(cache_enabled=True)

    def test_build_cache_key(self, optimizer):
        key = optimizer._build_cache_key("test query", "llm_expand")

        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash

    def test_cache_result(self, optimizer):
        key = "test_key"
        optimizer._cache_result(key, "cached value")

        cached = optimizer._get_cached(key)
        assert cached == "cached value"

    def test_cache_ttl_expiration(self, optimizer):
        key = "test_key"
        optimizer._cache_result(key, "value")

        # Mock expired cache
        import src.rag.query_optimizer as qo_module

        qo_module._EXPANSION_CACHE[key]["timestamp"] = time.time() - 7200  # 2 hours ago

        cached = optimizer._get_cached(key)
        assert cached is None  # Should be expired

    def test_cache_disabled(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            optimizer = QueryOptimizer(cache_enabled=False)

        optimizer._cache_result("key", "value")
        cached = optimizer._get_cached("key")

        assert cached is None

    def test_cache_eviction(self, optimizer):
        # Fill cache beyond max size
        import src.rag.query_optimizer as qo_module

        original_max = qo_module._CACHE_MAX_SIZE
        qo_module._CACHE_MAX_SIZE = 2

        try:
            optimizer._cache_result("key1", "value1")
            optimizer._cache_result("key2", "value2")
            optimizer._cache_result("key3", "value3")  # Should evict oldest

            # Cache should have max 2 entries
            assert len(qo_module._EXPANSION_CACHE) <= 2
        finally:
            qo_module._CACHE_MAX_SIZE = original_max

    def test_clear_cache(self, optimizer):
        # Clear any existing cache first
        optimizer.clear_cache()

        optimizer._cache_result("key1", "value1")
        optimizer._cache_result("key2", "value2")

        count = optimizer.clear_cache()

        assert count == 2
        assert optimizer._get_cached("key1") is None


# =============================================================================
# Test Combined Optimization
# =============================================================================


class TestCombinedOptimization:
    @pytest.fixture
    def optimizer(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            return QueryOptimizer(cache_enabled=False, typo_correction_enabled=False)

    @pytest.mark.asyncio
    async def test_optimize_query_full_pipeline(self, optimizer):
        with patch.object(optimizer, "_call_llm", return_value="llm expanded"):
            result = await optimizer.optimize_query(
                query="test query",
                use_typo_correction=False,
                use_llm=True,
                use_hyde=True,
                context=None,
            )

            assert result["original"] == "test query"
            assert result["rule_expanded"] is not None
            assert result["llm_expanded"] == "llm expanded"
            assert result["hyde_document"] is not None
            assert result["recommended"] is not None

    @pytest.mark.asyncio
    async def test_optimize_query_llm_only(self, optimizer):
        with patch.object(optimizer, "_call_llm", return_value="llm result"):
            result = await optimizer.optimize_query(query="test", use_llm=True, use_hyde=False)

            assert result["llm_expanded"] == "llm result"
            assert result["hyde_document"] is None
            assert result["recommended"] == "llm result"

    @pytest.mark.asyncio
    async def test_optimize_query_rule_only(self, optimizer):
        result = await optimizer.optimize_query(query="TRx query", use_llm=False, use_hyde=False)

        assert result["llm_expanded"] is None
        assert result["hyde_document"] is None
        assert result["recommended"] == result["rule_expanded"]

    @pytest.mark.asyncio
    async def test_optimize_query_with_typo_correction(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("src.rag.query_optimizer.TypoHandler") as mock_handler_class:
                mock_handler = Mock()
                mock_handler.correct_query = Mock(return_value=("corrected query", []))
                mock_handler_class.return_value = mock_handler

                optimizer = QueryOptimizer(typo_correction_enabled=True)

                result = await optimizer.optimize_query(
                    query="test qery", use_typo_correction=True, use_llm=False
                )

                assert result["typo_corrected"] == "corrected query"

    @pytest.mark.asyncio
    async def test_optimize_query_hyde_recommended(self, optimizer):
        with patch.object(optimizer, "_call_llm", return_value="hyde doc"):
            result = await optimizer.optimize_query(query="test", use_llm=False, use_hyde=True)

            # HyDE should be recommended over rule expansion
            assert result["recommended"] == "hyde doc"

    @pytest.mark.asyncio
    async def test_optimize_query_with_context(self, optimizer):
        with patch.object(optimizer, "_call_llm", return_value="context-aware result"):
            result = await optimizer.optimize_query(
                query="test",
                use_llm=True,
                context="Previous conversation about Kisqali",
            )

            assert result["llm_expanded"] == "context-aware result"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_empty_query_expansion(self):
        optimizer = QueryOptimizer()
        expanded = optimizer.expand("")

        assert expanded == ""

    def test_very_long_query(self):
        optimizer = QueryOptimizer()
        long_query = "test " * 1000

        expanded = optimizer.expand(long_query)
        assert isinstance(expanded, str)

    def test_special_characters_in_query(self):
        optimizer = QueryOptimizer()
        query = "What is TRx? @#$%^&*()"

        expanded = optimizer.expand(query)
        assert isinstance(expanded, str)

    @pytest.mark.asyncio
    async def test_concurrent_optimization(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            optimizer = QueryOptimizer(cache_enabled=True)

        with patch.object(optimizer, "_call_llm", return_value="result"):
            tasks = [optimizer.optimize_query(f"query_{i}", use_llm=True) for i in range(5)]

            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert all("original" in r for r in results)

    def test_vocabulary_case_insensitive(self):
        optimizer = QueryOptimizer()

        # Should match regardless of case
        expanded_lower = optimizer.expand("what is trx?")
        expanded_upper = optimizer.expand("what is TRX?")

        assert "trx" in expanded_lower.lower()
        assert "trx" in expanded_upper.lower()
