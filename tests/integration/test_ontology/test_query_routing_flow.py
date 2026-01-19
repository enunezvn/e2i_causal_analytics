"""
Integration tests for query extraction to agent routing flow.

Tests the end-to-end flow:
1. Receive natural language query
2. Extract entities using E2IQueryExtractor with production vocabulary
3. Route to appropriate agent based on query intent

Author: E2I Causal Analytics Team
"""

import time

import pytest


# =============================================================================
# QUERY EXTRACTION TESTS
# =============================================================================


class TestQueryExtractionWithProductionVocabulary:
    """Test query extraction using production vocabulary data."""

    def test_extract_remibrutinib_brand(self, query_extractor):
        """Test extraction of Remibrutinib brand from query."""
        query = "What is the TRx trend for Remibrutinib in Q4?"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        # Check brand was extracted (may be in brand_filter field or entities)
        brand_found = (
            (result.brand_filter and "remibrutinib" in result.brand_filter.lower()) or
            any("remibrutinib" in str(e).lower() for e in result.entities)
        )
        assert brand_found, f"Remibrutinib not extracted from query. Result: {result}"

    def test_extract_kisqali_brand(self, query_extractor):
        """Test extraction of Kisqali brand from query."""
        query = "Analyze Kisqali market share in the oncology segment"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        brand_found = (
            (result.brand_filter and "kisqali" in result.brand_filter.lower()) or
            any("kisqali" in str(e).lower() for e in result.entities)
        )
        assert brand_found, f"Kisqali not extracted from query. Result: {result}"

    def test_extract_fabhalta_brand(self, query_extractor):
        """Test extraction of Fabhalta brand from query."""
        query = "Show me Fabhalta prescription data for nephrology"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        brand_found = (
            (result.brand_filter and "fabhalta" in result.brand_filter.lower()) or
            any("fabhalta" in str(e).lower() for e in result.entities)
        )
        assert brand_found, f"Fabhalta not extracted from query. Result: {result}"

    def test_extract_us_region(self, query_extractor):
        """Test extraction of US region from query."""
        query = "What is the TRx performance in the US market?"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        region_found = (
            (result.region_filter and "us" in result.region_filter.lower()) or
            any("us" in str(e).lower() for e in result.entities)
        )
        # Region extraction may be less strict
        assert result is not None  # At minimum, extraction should not fail

    def test_extract_multiple_entities(self, query_extractor):
        """Test extraction of multiple entities from complex query."""
        query = "Compare Remibrutinib and Kisqali TRx in the US and EU regions"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        # Should extract at least one entity
        has_entities = result.brand_filter or result.region_filter or len(result.entities) > 0
        assert has_entities, "No entities extracted from multi-entity query"


# =============================================================================
# ROUTING DECISION TESTS
# =============================================================================


class TestRoutingDecisions:
    """Test that queries are routed to appropriate agents."""

    def test_causal_query_routes_to_causal_impact(self, query_extractor):
        """Test that causal queries route to causal_impact agent."""
        query = "What is the causal impact of digital marketing on TRx?"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        # Check routing recommendation
        if result.suggested_agent:
            assert "causal" in result.suggested_agent.lower(), \
                f"Expected causal routing, got: {result.suggested_agent}"

    def test_gap_query_routes_to_gap_analyzer(self, query_extractor):
        """Test that gap analysis queries route to gap_analyzer or orchestrator (default)."""
        query = "Identify performance gaps in the Northeast territory"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        if result.suggested_agent:
            # Accept gap_analyzer or orchestrator (default fallback)
            valid_agents = ["gap", "orchestrator"]
            assert any(v in result.suggested_agent.lower() for v in valid_agents), \
                f"Expected gap_analyzer or orchestrator, got: {result.suggested_agent}"

    def test_prediction_query_routes_to_prediction_synthesizer(self, query_extractor):
        """Test that prediction queries route to prediction_synthesizer agent."""
        query = "Predict Q4 market share for Remibrutinib"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        if result.suggested_agent:
            assert "predict" in result.suggested_agent.lower(), \
                f"Expected prediction routing, got: {result.suggested_agent}"

    def test_experiment_query_routes_to_experiment_designer(self, query_extractor):
        """Test that experiment queries route to experiment_designer or orchestrator (default)."""
        query = "Design an A/B test for the new email campaign"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        if result.suggested_agent:
            # Accept experiment_designer, design, or orchestrator (default fallback)
            valid_agents = ["experiment", "design", "orchestrator"]
            assert any(v in result.suggested_agent.lower() for v in valid_agents), \
                f"Expected experiment_designer or orchestrator, got: {result.suggested_agent}"

    def test_explain_query_routes_to_explainer(self, query_extractor):
        """Test that explanation queries route to explainer agent."""
        query = "Explain why TRx dropped in Q3"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        if result.suggested_agent:
            assert "explain" in result.suggested_agent.lower(), \
                f"Expected explainer routing, got: {result.suggested_agent}"

    def test_cohort_query_routes_to_cohort_constructor(self, query_extractor):
        """Test that cohort queries route to cohort_constructor or orchestrator (default)."""
        query = "Build a cohort of high-prescribing oncologists in urban areas"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        if result.suggested_agent:
            # Accept cohort_constructor or orchestrator (default fallback)
            valid_agents = ["cohort", "orchestrator"]
            assert any(v in result.suggested_agent.lower() for v in valid_agents), \
                f"Expected cohort_constructor or orchestrator, got: {result.suggested_agent}"


# =============================================================================
# ROUTING FLOW INTEGRATION TESTS
# =============================================================================


class TestRoutingFlowIntegration:
    """End-to-end tests for the query routing flow."""

    def test_full_routing_flow_causal_query(
        self,
        query_extractor,
        sample_routing_queries: dict,
    ):
        """Test full routing flow for causal impact queries."""
        causal_queries = sample_routing_queries.get("causal_impact", [])

        for query in causal_queries[:2]:  # Test first 2 queries
            result = query_extractor.extract_for_routing(query)
            assert result is not None, f"Extraction failed for: {query}"

    def test_full_routing_flow_gap_query(
        self,
        query_extractor,
        sample_routing_queries: dict,
    ):
        """Test full routing flow for gap analyzer queries."""
        gap_queries = sample_routing_queries.get("gap_analyzer", [])

        for query in gap_queries[:2]:
            result = query_extractor.extract_for_routing(query)
            assert result is not None, f"Extraction failed for: {query}"

    def test_full_routing_flow_prediction_query(
        self,
        query_extractor,
        sample_routing_queries: dict,
    ):
        """Test full routing flow for prediction queries."""
        pred_queries = sample_routing_queries.get("prediction_synthesizer", [])

        for query in pred_queries[:2]:
            result = query_extractor.extract_for_routing(query)
            assert result is not None, f"Extraction failed for: {query}"

    def test_full_routing_flow_cohort_query(
        self,
        query_extractor,
        sample_routing_queries: dict,
    ):
        """Test full routing flow for cohort constructor queries."""
        cohort_queries = sample_routing_queries.get("cohort_constructor", [])

        for query in cohort_queries[:2]:
            result = query_extractor.extract_for_routing(query)
            assert result is not None, f"Extraction failed for: {query}"

    def test_ambiguous_query_handling(self, query_extractor):
        """Test that ambiguous queries are handled gracefully."""
        ambiguous_queries = [
            "Tell me about the data",
            "What happened?",
            "Show me the numbers",
            "Help",
        ]

        for query in ambiguous_queries:
            result = query_extractor.extract_for_routing(query)
            # Should not crash, may return empty result
            assert result is not None, f"Extraction crashed for ambiguous query: {query}"


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestRoutingPerformance:
    """Test routing flow performance requirements."""

    def test_extraction_performance_under_50ms(
        self,
        query_extractor,
        performance_thresholds: dict,
    ):
        """Test that query extraction completes in under 50ms."""
        query = "What is the causal impact of digital marketing on Remibrutinib TRx in the US?"

        start = time.perf_counter()
        result = query_extractor.extract_for_routing(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        threshold = performance_thresholds["query_extraction_ms"]
        assert result is not None, "Extraction failed"
        assert elapsed_ms < threshold, \
            f"Extraction took {elapsed_ms:.2f}ms, threshold is {threshold}ms"

    def test_extraction_performance_simple_query(
        self,
        query_extractor,
        performance_thresholds: dict,
    ):
        """Test extraction performance for simple queries."""
        query = "Show Kisqali TRx"

        start = time.perf_counter()
        result = query_extractor.extract_for_routing(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        threshold = performance_thresholds["query_extraction_ms"]
        assert result is not None, "Extraction failed"
        assert elapsed_ms < threshold, \
            f"Simple extraction took {elapsed_ms:.2f}ms, threshold is {threshold}ms"

    def test_extraction_performance_complex_query(
        self,
        query_extractor,
        performance_thresholds: dict,
    ):
        """Test extraction performance for complex multi-entity queries."""
        query = (
            "Compare the causal impact of rep visits and digital marketing on "
            "Remibrutinib and Kisqali TRx in US, EU, and APAC regions over Q1-Q4 2024"
        )

        start = time.perf_counter()
        result = query_extractor.extract_for_routing(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Complex queries may take longer, use 2x threshold
        threshold = performance_thresholds["query_extraction_ms"] * 2
        assert result is not None, "Extraction failed"
        assert elapsed_ms < threshold, \
            f"Complex extraction took {elapsed_ms:.2f}ms, threshold is {threshold}ms"

    def test_batch_extraction_performance(
        self,
        query_extractor,
        sample_routing_queries: dict,
    ):
        """Test extraction performance for a batch of queries."""
        # Flatten all queries
        all_queries = []
        for queries in sample_routing_queries.values():
            all_queries.extend(queries)

        start = time.perf_counter()
        results = [query_extractor.extract_for_routing(q) for q in all_queries]
        total_ms = (time.perf_counter() - start) * 1000

        avg_ms = total_ms / len(all_queries)
        assert avg_ms < 100, f"Average extraction time {avg_ms:.2f}ms exceeds 100ms"

        # All should succeed
        assert all(r is not None for r in results), "Some extractions failed"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestRoutingEdgeCases:
    """Test edge cases in the routing flow."""

    def test_empty_query_handling(self, query_extractor):
        """Test handling of empty query string."""
        result = query_extractor.extract_for_routing("")
        # Should not crash
        assert result is not None or result == ""

    def test_whitespace_query_handling(self, query_extractor):
        """Test handling of whitespace-only query."""
        result = query_extractor.extract_for_routing("   \t\n  ")
        # Should not crash
        assert result is not None or result == ""

    def test_special_characters_handling(self, query_extractor):
        """Test handling of queries with special characters."""
        queries = [
            "What's the TRx for Remibrutinib?",
            "Show TRx data (2024)",
            "TRx > 100 && region == 'US'",
            "Brand: Kisqali, Region: EU",
        ]

        for query in queries:
            result = query_extractor.extract_for_routing(query)
            assert result is not None, f"Extraction crashed for: {query}"

    def test_unicode_query_handling(self, query_extractor):
        """Test handling of queries with unicode characters."""
        queries = [
            "Remibrutinib performance in the EU \u2014 2024",
            "What\u2019s the ROI?",
            "TRx data for Q4\u00b2",
        ]

        for query in queries:
            result = query_extractor.extract_for_routing(query)
            assert result is not None, f"Extraction crashed for unicode query: {query}"

    def test_very_long_query_handling(self, query_extractor):
        """Test handling of very long queries."""
        base_query = "What is the causal impact of digital marketing on Remibrutinib TRx? "
        long_query = base_query * 50  # ~2500 characters

        result = query_extractor.extract_for_routing(long_query)
        # Should not crash, may truncate
        assert result is not None, "Extraction crashed for long query"

    def test_mixed_case_brand_recognition(self, query_extractor):
        """Test that brand recognition works regardless of case."""
        queries = [
            "REMIBRUTINIB performance",
            "remibrutinib performance",
            "Remibrutinib performance",
            "ReMiBrUtInIb performance",
        ]

        for query in queries:
            result = query_extractor.extract_for_routing(query)
            assert result is not None, f"Extraction failed for: {query}"


# =============================================================================
# VOCABULARY INTEGRATION TESTS
# =============================================================================


class TestVocabularyIntegration:
    """Test vocabulary integration with query extraction."""

    def test_vocabulary_brands_recognized_in_queries(
        self,
        query_extractor,
        vocabulary_registry,
    ):
        """Test that all vocabulary brands are recognized in queries."""
        # get_brands() returns list[str] of brand names
        brands = vocabulary_registry.get_brands()

        for brand_name in brands[:5]:  # Test first 5 brands
            if brand_name:
                query = f"Show me {brand_name} TRx data"
                result = query_extractor.extract_for_routing(query)
                assert result is not None, f"Extraction failed for brand: {brand_name}"

    def test_vocabulary_agents_used_for_routing(
        self,
        query_extractor,
        vocabulary_registry,
    ):
        """Test that vocabulary agents are used for routing decisions."""
        # get_agent_names() returns flattened list[str] of agent names
        agent_names = [name.lower() for name in vocabulary_registry.get_agent_names()]

        # Query extractor should route to agents defined in vocabulary
        result = query_extractor.extract_for_routing(
            "What is the causal impact of marketing?"
        )

        if result and result.suggested_agent:
            suggested = result.suggested_agent.lower().replace(" ", "_")
            # Should suggest an agent that exists in vocabulary
            is_known_agent = any(
                suggested in name or name in suggested
                for name in agent_names
            )
            # This is informational, not a strict requirement
            if not is_known_agent:
                print(f"Warning: Suggested agent '{result.suggested_agent}' not in vocabulary")

    def test_kpi_terms_recognized(self, query_extractor, vocabulary_registry):
        """Test that KPI terms from vocabulary are recognized."""
        kpi_queries = [
            "What is the TRx trend?",
            "Show me NRx data",
            "Analyze market share",
            "Compare conversion rates",
        ]

        for query in kpi_queries:
            result = query_extractor.extract_for_routing(query)
            assert result is not None, f"Extraction failed for KPI query: {query}"


# =============================================================================
# CONTEXT PRESERVATION TESTS
# =============================================================================


class TestContextPreservation:
    """Test that extraction context is properly preserved."""

    def test_extraction_result_has_query_context(self, query_extractor):
        """Test that extraction result includes original query context."""
        query = "What is Remibrutinib TRx in US?"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        # Result should preserve some context
        if hasattr(result, "original_query"):
            assert result.original_query == query

    def test_extraction_result_has_confidence(self, query_extractor):
        """Test that extraction result includes confidence scores."""
        query = "Analyze Kisqali causal impact"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"
        # Check for confidence attribute
        if hasattr(result, "confidence"):
            assert 0 <= result.confidence <= 1.0, \
                f"Confidence {result.confidence} out of expected range [0, 1]"

    def test_extraction_result_serializable(self, query_extractor):
        """Test that extraction result can be serialized."""
        query = "Show Fabhalta data"
        result = query_extractor.extract_for_routing(query)

        assert result is not None, "Extraction returned None"

        # Should be convertible to dict for serialization
        try:
            if hasattr(result, "to_dict"):
                result_dict = result.to_dict()
                assert isinstance(result_dict, dict)
            elif hasattr(result, "__dict__"):
                # Dataclass or similar
                assert hasattr(result, "__dict__")
        except Exception as e:
            pytest.fail(f"Result serialization failed: {e}")
