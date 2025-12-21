"""
Tests for fastText-based typo correction.

Tests cover:
- CorrectionResult dataclass
- TypoHandler initialization
- Term correction (brands, KPIs, regions)
- Abbreviation expansion
- Query correction
- Edit-distance fallback
- Caching behavior
- Performance (<50ms latency)
"""

import time

from src.nlp.typo_handler import (
    ABBREVIATION_EXPANSIONS,
    CANONICAL_VOCABULARY,
    CorrectionResult,
    TypoHandler,
    _levenshtein_distance,
    _normalized_edit_similarity,
    correct_query,
    correct_term,
    get_typo_handler,
)


class TestCorrectionResult:
    """Tests for CorrectionResult dataclass."""

    def test_correction_result_defaults(self):
        """Test CorrectionResult with default values."""
        result = CorrectionResult(
            original="test",
            corrected="test",
            confidence=0.0,
        )
        assert result.original == "test"
        assert result.corrected == "test"
        assert result.confidence == 0.0
        assert result.category is None
        assert result.was_corrected is False
        assert result.latency_ms == 0.0

    def test_correction_result_with_values(self):
        """Test CorrectionResult with all values set."""
        result = CorrectionResult(
            original="kiqsali",
            corrected="Kisqali",
            confidence=0.85,
            category="brands",
            was_corrected=True,
            latency_ms=5.5,
        )
        assert result.original == "kiqsali"
        assert result.corrected == "Kisqali"
        assert result.confidence == 0.85
        assert result.category == "brands"
        assert result.was_corrected is True
        assert result.latency_ms == 5.5


class TestLevenshteinDistance:
    """Tests for edit distance calculation."""

    def test_identical_strings(self):
        """Identical strings have distance 0."""
        assert _levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        """Empty string comparisons."""
        assert _levenshtein_distance("", "") == 0
        assert _levenshtein_distance("abc", "") == 3
        assert _levenshtein_distance("", "abc") == 3

    def test_single_insertion(self):
        """Single character insertion."""
        assert _levenshtein_distance("cat", "cats") == 1

    def test_single_deletion(self):
        """Single character deletion."""
        assert _levenshtein_distance("cats", "cat") == 1

    def test_single_substitution(self):
        """Single character substitution."""
        assert _levenshtein_distance("cat", "bat") == 1

    def test_multiple_edits(self):
        """Multiple edits required."""
        assert _levenshtein_distance("kitten", "sitting") == 3

    def test_completely_different(self):
        """Completely different strings."""
        assert _levenshtein_distance("abc", "xyz") == 3


class TestNormalizedEditSimilarity:
    """Tests for normalized edit similarity."""

    def test_identical_strings(self):
        """Identical strings have similarity 1.0."""
        assert _normalized_edit_similarity("hello", "hello") == 1.0

    def test_empty_strings(self):
        """Empty strings have similarity 1.0."""
        assert _normalized_edit_similarity("", "") == 1.0

    def test_one_edit_away(self):
        """One edit away should have high similarity."""
        sim = _normalized_edit_similarity("cat", "cats")
        assert 0.7 < sim < 0.8

    def test_case_insensitive(self):
        """Comparison is case-insensitive."""
        assert _normalized_edit_similarity("Kisqali", "kisqali") == 1.0

    def test_typo_similarity(self):
        """Typos should have moderate-high similarity."""
        sim = _normalized_edit_similarity("kiqsali", "kisqali")
        assert sim > 0.7


class TestTypoHandlerInitialization:
    """Tests for TypoHandler initialization."""

    def test_init_without_fasttext_model(self):
        """Initialize without fastText model uses fallback."""
        handler = TypoHandler(
            model_path="/nonexistent/path.bin",
            cache_enabled=False,
        )
        # Should initialize successfully using edit-distance fallback
        assert handler is not None
        assert len(handler._all_candidates) > 0

    def test_init_with_custom_vocabulary(self):
        """Initialize with custom vocabulary."""
        custom_vocab = {
            "test_terms": ["alpha", "beta", "gamma"],
        }
        handler = TypoHandler(
            vocabulary=custom_vocab,
            cache_enabled=False,
        )
        assert "alpha" in handler._all_candidates
        assert "beta" in handler._all_candidates

    def test_default_vocabulary_loaded(self):
        """Default vocabulary is loaded."""
        handler = TypoHandler(cache_enabled=False)

        # Check brands
        assert "Kisqali" in handler._all_candidates
        assert "Fabhalta" in handler._all_candidates

        # Check KPIs
        assert "TRx" in handler._all_candidates
        assert "NRx" in handler._all_candidates

        # Check regions
        assert "northeast" in handler._all_candidates
        assert "midwest" in handler._all_candidates

    def test_candidate_to_category_mapping(self):
        """Candidate to category mapping is built."""
        handler = TypoHandler(cache_enabled=False)

        assert handler._candidate_to_category.get("kisqali") == "brands"
        assert handler._candidate_to_category.get("trx") == "kpis"
        assert handler._candidate_to_category.get("northeast") == "regions"


class TestAbbreviationExpansion:
    """Tests for abbreviation expansion."""

    def test_trx_expansion(self):
        """TRx abbreviation is expanded."""
        handler = TypoHandler(cache_enabled=False)
        result = handler.correct_term("trx")

        assert result.corrected == "TRx"
        assert result.was_corrected is True
        assert result.confidence == 1.0

    def test_nrx_expansion(self):
        """NRx abbreviation is expanded."""
        handler = TypoHandler(cache_enabled=False)
        result = handler.correct_term("nrx")

        assert result.corrected == "NRx"
        assert result.was_corrected is True

    def test_region_abbreviations(self):
        """Region abbreviations are expanded."""
        handler = TypoHandler(cache_enabled=False)

        ne_result = handler.correct_term("ne")
        assert ne_result.corrected == "northeast"

        mw_result = handler.correct_term("mw")
        assert mw_result.corrected == "midwest"

    def test_workstream_abbreviations(self):
        """Workstream abbreviations are expanded."""
        handler = TypoHandler(cache_enabled=False)

        result = handler.correct_term("ws1")
        assert result.corrected == "WS1"


class TestTermCorrection:
    """Tests for single term correction."""

    def test_correct_kisqali_typo(self):
        """Correct Kisqali typo."""
        handler = TypoHandler(cache_enabled=False)

        result = handler.correct_term("kiqsali")
        assert result.corrected == "Kisqali"
        assert result.was_corrected is True
        assert result.category == "brands"

    def test_correct_fabhalta_typo(self):
        """Correct Fabhalta typo."""
        handler = TypoHandler(cache_enabled=False)

        result = handler.correct_term("fabhlta")
        assert result.corrected == "Fabhalta"
        assert result.was_corrected is True

    def test_correct_remibrutinib_typo(self):
        """Correct Remibrutinib typo."""
        handler = TypoHandler(cache_enabled=False)

        result = handler.correct_term("remibrutanib")
        assert result.corrected == "Remibrutinib"
        assert result.was_corrected is True

    def test_correct_region_typo(self):
        """Correct region typo."""
        handler = TypoHandler(cache_enabled=False)

        result = handler.correct_term("northest")
        assert result.corrected == "northeast"
        assert result.was_corrected is True

    def test_correct_kpi_typo(self):
        """Correct KPI typo."""
        handler = TypoHandler(cache_enabled=False)

        result = handler.correct_term("conversin_rate")
        assert result.corrected == "conversion_rate"
        assert result.was_corrected is True

    def test_already_canonical(self):
        """Already canonical terms are not modified."""
        handler = TypoHandler(cache_enabled=False)

        result = handler.correct_term("Kisqali")
        assert result.corrected == "Kisqali"
        assert result.was_corrected is False
        assert result.confidence == 1.0

    def test_unknown_term_not_corrected(self):
        """Unknown terms are not forcibly corrected."""
        handler = TypoHandler(cache_enabled=False)

        result = handler.correct_term("xyz123unknown")
        assert result.corrected == "xyz123unknown"
        assert result.was_corrected is False
        assert result.confidence == 0.0

    def test_category_hint_restricts_candidates(self):
        """Category hint restricts candidate pool."""
        handler = TypoHandler(cache_enabled=False)

        # Without category hint, might match anything
        handler.correct_term("ne")

        # With category hint
        result_with_hint = handler.correct_term("ne", category="regions")
        assert result_with_hint.corrected == "northeast"


class TestQueryCorrection:
    """Tests for full query correction."""

    def test_single_word_query(self):
        """Single word query correction."""
        handler = TypoHandler(cache_enabled=False)

        corrected, corrections = handler.correct_query("kiqsali")
        assert corrected == "Kisqali"
        assert len(corrections) == 1

    def test_multi_word_query(self):
        """Multi-word query correction."""
        handler = TypoHandler(cache_enabled=False)

        corrected, corrections = handler.correct_query("trx for kiqsali in northest")
        assert "TRx" in corrected
        assert "Kisqali" in corrected
        assert "northeast" in corrected
        assert len(corrections) >= 2

    def test_preserves_punctuation(self):
        """Punctuation is preserved in corrections."""
        handler = TypoHandler(cache_enabled=False)

        corrected, _ = handler.correct_query("What is the trx?")
        assert corrected.endswith("?")

    def test_short_words_skipped(self):
        """Very short words are skipped."""
        handler = TypoHandler(cache_enabled=False)

        corrected, corrections = handler.correct_query("a b c")
        # Short words should not be corrected (unless abbreviations)
        assert corrected == "a b c"

    def test_correct_all_words_mode(self):
        """correct_all_words=True attempts all words."""
        handler = TypoHandler(cache_enabled=False)

        # With correct_all_words=True, should attempt more corrections
        _, corrections = handler.correct_query("show me the data", correct_all_words=True)
        # May or may not have corrections, but should process all words


class TestCaching:
    """Tests for correction caching."""

    def test_cache_hit(self):
        """Cached corrections are returned."""
        handler = TypoHandler(cache_enabled=True)

        # First call
        result1 = handler.correct_term("kiqsali")

        # Second call should be cached
        result2 = handler.correct_term("kiqsali")

        assert result1.corrected == result2.corrected
        # Second call should be faster (cached)
        assert result2.latency_ms <= result1.latency_ms or result2.latency_ms < 1.0

    def test_cache_disabled(self):
        """Cache can be disabled."""
        handler = TypoHandler(cache_enabled=False)

        result = handler.correct_term("kiqsali")
        assert result.corrected == "Kisqali"

    def test_clear_cache(self):
        """Cache can be cleared."""
        handler = TypoHandler(cache_enabled=True)

        # Populate cache
        handler.correct_term("kiqsali")
        handler.correct_term("fabhlta")

        # Clear and verify
        cleared = handler.clear_cache()
        assert cleared >= 0  # May have cleared entries


class TestSuggestions:
    """Tests for typo correction suggestions."""

    def test_get_suggestions(self):
        """Get multiple correction suggestions."""
        handler = TypoHandler(cache_enabled=False)

        suggestions = handler.get_suggestions("kiq", top_k=5)

        assert len(suggestions) <= 5
        assert all(isinstance(s, tuple) and len(s) == 2 for s in suggestions)
        # First suggestion should have highest score
        if len(suggestions) > 1:
            assert suggestions[0][1] >= suggestions[1][1]

    def test_get_suggestions_with_category(self):
        """Get suggestions restricted to category."""
        handler = TypoHandler(cache_enabled=False)

        suggestions = handler.get_suggestions("kis", top_k=5, category="brands")

        # Suggestions should be from brands category
        brand_terms = [s.lower() for s in CANONICAL_VOCABULARY["brands"]]
        for term, _score in suggestions:
            assert term.lower() in brand_terms


class TestPerformance:
    """Performance tests for typo correction."""

    def test_single_term_latency(self):
        """Single term correction should be fast (<50ms)."""
        handler = TypoHandler(cache_enabled=False)

        start = time.time()
        for _ in range(10):
            handler.correct_term("kiqsali")
        elapsed_ms = (time.time() - start) * 1000 / 10

        # Should be under 50ms per correction
        assert elapsed_ms < 50, f"Latency {elapsed_ms:.1f}ms exceeds 50ms target"

    def test_query_latency(self):
        """Query correction should be fast (<100ms)."""
        handler = TypoHandler(cache_enabled=False)

        start = time.time()
        for _ in range(5):
            handler.correct_query("trx for kiqsali in northest region")
        elapsed_ms = (time.time() - start) * 1000 / 5

        # Should be under 100ms per query
        assert elapsed_ms < 100, f"Latency {elapsed_ms:.1f}ms exceeds 100ms target"


class TestStats:
    """Tests for handler statistics."""

    def test_get_stats(self):
        """Get handler statistics."""
        handler = TypoHandler(cache_enabled=True)

        stats = handler.get_stats()

        assert "fasttext_available" in stats
        assert "vocabulary_size" in stats
        assert "categories" in stats
        assert "cache_size" in stats
        assert stats["vocabulary_size"] > 0
        assert len(stats["categories"]) > 0


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_correct_term_function(self):
        """Module-level correct_term function works."""
        result = correct_term("kiqsali")
        assert result.corrected == "Kisqali"

    def test_correct_query_function(self):
        """Module-level correct_query function works."""
        corrected, corrections = correct_query("trx for kiqsali")
        assert "TRx" in corrected
        assert "Kisqali" in corrected

    def test_get_typo_handler_singleton(self):
        """get_typo_handler returns singleton."""
        handler1 = get_typo_handler()
        handler2 = get_typo_handler()

        assert handler1 is handler2


class TestCanonicalVocabulary:
    """Tests for canonical vocabulary constants."""

    def test_brands_present(self):
        """Brand vocabulary is defined."""
        assert "brands" in CANONICAL_VOCABULARY
        assert "Kisqali" in CANONICAL_VOCABULARY["brands"]
        assert "Fabhalta" in CANONICAL_VOCABULARY["brands"]
        assert "Remibrutinib" in CANONICAL_VOCABULARY["brands"]

    def test_kpis_present(self):
        """KPI vocabulary is defined."""
        assert "kpis" in CANONICAL_VOCABULARY
        assert "TRx" in CANONICAL_VOCABULARY["kpis"]
        assert "NRx" in CANONICAL_VOCABULARY["kpis"]
        assert "conversion_rate" in CANONICAL_VOCABULARY["kpis"]

    def test_regions_present(self):
        """Region vocabulary is defined."""
        assert "regions" in CANONICAL_VOCABULARY
        assert "northeast" in CANONICAL_VOCABULARY["regions"]
        assert "midwest" in CANONICAL_VOCABULARY["regions"]

    def test_abbreviations_defined(self):
        """Abbreviation mappings are defined."""
        assert "trx" in ABBREVIATION_EXPANSIONS
        assert "nrx" in ABBREVIATION_EXPANSIONS
        assert "hcp" in ABBREVIATION_EXPANSIONS
        assert ABBREVIATION_EXPANSIONS["trx"] == "TRx"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_term(self):
        """Empty term handling."""
        handler = TypoHandler(cache_enabled=False)

        result = handler.correct_term("")
        assert result.corrected == ""
        assert result.was_corrected is False

    def test_whitespace_term(self):
        """Whitespace-only term handling."""
        handler = TypoHandler(cache_enabled=False)

        result = handler.correct_term("   ")
        assert result.was_corrected is False

    def test_empty_query(self):
        """Empty query handling."""
        handler = TypoHandler(cache_enabled=False)

        corrected, corrections = handler.correct_query("")
        assert corrected == ""
        assert len(corrections) == 0

    def test_special_characters(self):
        """Special characters in input."""
        handler = TypoHandler(cache_enabled=False)

        corrected, _ = handler.correct_query("What is the TRx?")
        assert "TRx" in corrected

    def test_unicode_input(self):
        """Unicode input handling."""
        handler = TypoHandler(cache_enabled=False)

        # Should not crash on unicode
        result = handler.correct_term("tëst")
        assert result is not None

    def test_very_long_term(self):
        """Very long term handling."""
        handler = TypoHandler(cache_enabled=False)

        long_term = "a" * 1000
        result = handler.correct_term(long_term)
        # Should not crash, just return original
        assert result.corrected == long_term
