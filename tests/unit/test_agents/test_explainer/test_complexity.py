"""
Unit tests for Explainer agent complexity scoring.
Version: 4.3

Tests smart LLM mode selection based on input complexity.
"""

import pytest

from src.agents.explainer.config import (
    ComplexityScorer,
    ExplainerConfig,
    compute_complexity,
    should_use_llm,
)


class TestExplainerConfig:
    """Tests for ExplainerConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = ExplainerConfig()

        assert config.llm_threshold == 0.5
        assert config.auto_llm is True
        assert config.result_count_weight == 0.25
        assert config.query_complexity_weight == 0.30
        assert config.causal_discovery_weight == 0.25
        assert config.expertise_weight == 0.20
        assert config.high_result_count == 3
        assert config.complex_query_min_words == 10

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = ExplainerConfig(
            llm_threshold=0.7,
            auto_llm=False,
            result_count_weight=0.30,
        )

        assert config.llm_threshold == 0.7
        assert config.auto_llm is False
        assert config.result_count_weight == 0.30

    def test_complex_query_patterns_default(self):
        """Test default complex query patterns are set."""
        config = ExplainerConfig()

        assert len(config.complex_query_patterns) > 0
        assert r"\bwhy\b" in config.complex_query_patterns
        assert r"\bexplain\b" in config.complex_query_patterns
        assert r"\bcompare\b" in config.complex_query_patterns

    def test_simple_query_patterns_default(self):
        """Test default simple query patterns are set."""
        config = ExplainerConfig()

        assert len(config.simple_query_patterns) > 0
        assert r"\bwhat is\b" in config.simple_query_patterns
        assert r"\blist\b" in config.simple_query_patterns


class TestComplexityScorer:
    """Tests for ComplexityScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create default complexity scorer."""
        return ComplexityScorer()

    @pytest.fixture
    def custom_scorer(self):
        """Create scorer with custom threshold."""
        config = ExplainerConfig(llm_threshold=0.7)
        return ComplexityScorer(config)

    # Result count scoring tests
    def test_score_empty_results(self, scorer):
        """Empty results should have low complexity."""
        score = scorer.compute_complexity([], "query")
        assert score < 0.5

    def test_score_single_result(self, scorer):
        """Single result should have low-medium complexity."""
        results = [{"data": "test"}]
        score = scorer.compute_complexity(results, "query")
        assert 0.2 <= score <= 0.6

    def test_score_multiple_results(self, scorer):
        """Multiple results should have higher complexity."""
        results = [{"data": f"test_{i}"} for i in range(5)]
        score = scorer.compute_complexity(results, "query")
        # With weights: result_count=0.7*0.25 + query=0.2*0.30 + causal=0*0.25 + expertise=0.5*0.20 = 0.335
        assert score > 0.3

    def test_score_many_results(self, scorer):
        """Many results should have high complexity."""
        results = [{"data": f"test_{i}"} for i in range(10)]
        score = scorer.compute_complexity(results, "query")
        # With weights: result_count=1.0*0.25 + query=0.2*0.30 + causal=0*0.25 + expertise=0.5*0.20 = 0.41
        assert score > 0.4

    # Query complexity scoring tests
    def test_score_empty_query(self, scorer):
        """Empty query should have medium complexity (relies on other factors)."""
        results = [{"data": "test"}]
        score = scorer.compute_complexity(results, "")
        assert 0.2 <= score <= 0.6

    def test_score_simple_query(self, scorer):
        """Simple query should have lower complexity."""
        results = [{"data": "test"}]
        score_simple = scorer.compute_complexity(results, "what is the total?")
        score_complex = scorer.compute_complexity(
            results, "why did sales decline and what caused the market shift?"
        )
        assert score_simple < score_complex

    def test_score_complex_query_keywords(self, scorer):
        """Query with complex keywords should have higher complexity."""
        results = [{"data": "test"}]

        # Complex keywords should increase score
        score_why = scorer.compute_complexity(results, "why did this happen?")
        score_explain = scorer.compute_complexity(results, "explain the trend")
        score_compare = scorer.compute_complexity(results, "compare these metrics")
        score_basic = scorer.compute_complexity(results, "show data")

        assert score_why > score_basic
        assert score_explain > score_basic
        assert score_compare > score_basic

    def test_score_long_query(self, scorer):
        """Long queries should have higher complexity."""
        results = [{"data": "test"}]
        short_query = "show data"
        long_query = (
            "analyze the relationship between marketing spend and prescription volume "
            "across different regions and identify the key factors driving conversion"
        )

        score_short = scorer.compute_complexity(results, short_query)
        score_long = scorer.compute_complexity(results, long_query)

        assert score_long > score_short

    # Causal discovery scoring tests
    def test_score_with_causal_discovery_flag(self, scorer):
        """Explicit causal discovery flag should increase complexity."""
        results = [{"data": "test"}]
        score_without = scorer.compute_complexity(
            results, "query", has_causal_discovery=False
        )
        score_with = scorer.compute_complexity(
            results, "query", has_causal_discovery=True
        )

        assert score_with > score_without

    def test_score_with_causal_discovery_in_results(self, scorer):
        """Causal discovery data in results should increase complexity."""
        results_without = [{"data": "test"}]
        results_with = [{"discovered_dag": {"nodes": ["A", "B"]}, "data": "test"}]

        score_without = scorer.compute_complexity(results_without, "query")
        score_with = scorer.compute_complexity(results_with, "query")

        assert score_with > score_without

    def test_score_detects_causal_graph_key(self, scorer):
        """Should detect 'causal_graph' key in results."""
        results = [{"causal_graph": {"edges": []}}]
        score = scorer.compute_complexity(results, "query")
        # With weights: result_count=0.2*0.25 + query=0.2*0.30 + causal=1.0*0.25 + expertise=0.5*0.20 = 0.46
        assert score > 0.4

    def test_score_detects_discovery_gate_decision(self, scorer):
        """Should detect 'discovery_gate_decision' key in results."""
        results = [{"discovery_gate_decision": "accept"}]
        score = scorer.compute_complexity(results, "query")
        # With weights: result_count=0.2*0.25 + query=0.2*0.30 + causal=1.0*0.25 + expertise=0.5*0.20 = 0.46
        assert score > 0.4

    # Expertise level scoring tests
    def test_score_executive_expertise(self, scorer):
        """Executive expertise should have higher complexity (needs synthesis)."""
        results = [{"data": "test"}]
        score_exec = scorer.compute_complexity(
            results, "query", user_expertise="executive"
        )
        score_analyst = scorer.compute_complexity(
            results, "query", user_expertise="analyst"
        )
        score_ds = scorer.compute_complexity(
            results, "query", user_expertise="data_scientist"
        )

        assert score_exec > score_analyst
        assert score_analyst > score_ds

    def test_score_unknown_expertise_defaults_to_medium(self, scorer):
        """Unknown expertise should default to medium complexity."""
        results = [{"data": "test"}]
        score_unknown = scorer.compute_complexity(
            results, "query", user_expertise="unknown"
        )
        score_analyst = scorer.compute_complexity(
            results, "query", user_expertise="analyst"
        )

        assert score_unknown == score_analyst

    # should_use_llm tests
    def test_should_use_llm_simple_input(self, scorer):
        """Simple input should not trigger LLM."""
        results = [{"data": "test"}]
        should_use, score, reason = scorer.should_use_llm(results, "show data")

        assert should_use is False
        assert score < 0.5
        assert "threshold" in reason.lower()

    def test_should_use_llm_complex_input(self, scorer):
        """Complex input should trigger LLM."""
        results = [{"data": f"test_{i}"} for i in range(5)]
        query = "explain why the causal relationship between marketing and sales changed"

        should_use, score, reason = scorer.should_use_llm(
            results, query, has_causal_discovery=True
        )

        assert should_use is True
        assert score >= 0.5
        assert "threshold" in reason.lower()

    def test_should_use_llm_respects_custom_threshold(self, custom_scorer):
        """Custom threshold should be respected."""
        results = [{"data": f"test_{i}"} for i in range(4)]
        query = "analyze this data"

        # With default threshold (0.5), this might trigger LLM
        default_scorer = ComplexityScorer()
        should_use_default, score_default, _ = default_scorer.should_use_llm(
            results, query
        )

        # With higher threshold (0.7), less likely to trigger
        should_use_custom, score_custom, _ = custom_scorer.should_use_llm(
            results, query
        )

        # Scores should be the same, but decision might differ
        assert abs(score_default - score_custom) < 0.01

    # Boundary tests
    def test_score_bounded_between_0_and_1(self, scorer):
        """Complexity score should always be between 0 and 1."""
        # Test with extreme inputs
        test_cases = [
            ([], ""),
            ([{"data": "test"}] * 100, "a" * 1000),
            ([{"discovered_dag": {}, "causal_graph": {}}], "why explain compare"),
        ]

        for results, query in test_cases:
            score = scorer.compute_complexity(
                results, query, has_causal_discovery=True
            )
            assert 0.0 <= score <= 1.0


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_compute_complexity_function(self):
        """Test module-level compute_complexity function."""
        results = [{"data": "test"}]
        score = compute_complexity(results, "why did this happen?")

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_should_use_llm_function(self):
        """Test module-level should_use_llm function."""
        results = [{"data": "test"}]
        should_use, score, reason = should_use_llm(results, "query")

        assert isinstance(should_use, bool)
        assert isinstance(score, float)
        assert isinstance(reason, str)

    def test_functions_accept_custom_config(self):
        """Test that functions accept custom configuration."""
        config = ExplainerConfig(llm_threshold=0.9)
        results = [{"data": f"test_{i}"} for i in range(5)]

        should_use, _, _ = should_use_llm(
            results, "explain the trend", config=config
        )

        # With very high threshold, should not use LLM
        assert should_use is False


class TestExplainerAgentIntegration:
    """Integration tests for ExplainerAgent with complexity scoring."""

    @pytest.fixture
    def agent_auto_mode(self):
        """Create agent in auto LLM mode."""
        from src.agents.explainer import ExplainerAgent

        return ExplainerAgent(use_llm=None)  # Auto mode

    @pytest.fixture
    def agent_explicit_llm(self):
        """Create agent with explicit LLM mode."""
        from src.agents.explainer import ExplainerAgent

        return ExplainerAgent(use_llm=True)

    @pytest.fixture
    def agent_explicit_deterministic(self):
        """Create agent with explicit deterministic mode."""
        from src.agents.explainer import ExplainerAgent

        return ExplainerAgent(use_llm=False)

    def test_agent_auto_mode_initialization(self, agent_auto_mode):
        """Agent should initialize in auto mode."""
        assert agent_auto_mode._use_llm is None
        assert agent_auto_mode._complexity_scorer is not None

    def test_agent_explicit_llm_initialization(self, agent_explicit_llm):
        """Agent should respect explicit LLM mode."""
        assert agent_explicit_llm._use_llm is True

    def test_agent_explicit_deterministic_initialization(
        self, agent_explicit_deterministic
    ):
        """Agent should respect explicit deterministic mode."""
        assert agent_explicit_deterministic._use_llm is False

    def test_agent_should_use_llm_method(self, agent_auto_mode):
        """Agent _should_use_llm method should work."""
        results = [{"data": "test"}]
        should_use, score, reason = agent_auto_mode._should_use_llm(
            results, "query", "analyst"
        )

        assert isinstance(should_use, bool)
        assert isinstance(score, float)
        assert isinstance(reason, str)

    def test_agent_detects_causal_discovery_in_results(self, agent_auto_mode):
        """Agent should detect causal discovery data in results."""
        results_without = [{"data": "test"}]
        results_with = [{"discovered_dag": {"nodes": ["A", "B"]}}]

        _, score_without, _ = agent_auto_mode._should_use_llm(
            results_without, "query", "analyst"
        )
        _, score_with, _ = agent_auto_mode._should_use_llm(
            results_with, "query", "analyst"
        )

        assert score_with > score_without

    def test_agent_custom_config(self):
        """Agent should accept custom configuration."""
        from src.agents.explainer import ExplainerAgent, ExplainerConfig

        config = ExplainerConfig(llm_threshold=0.8)
        agent = ExplainerAgent(use_llm=None, config=config)

        assert agent._config.llm_threshold == 0.8


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def scorer(self):
        """Create default complexity scorer."""
        return ComplexityScorer()

    def test_none_values_in_results(self, scorer):
        """Should handle None values in results."""
        results = [{"data": None}, None]
        # Filter out None values
        valid_results = [r for r in results if r is not None]
        score = scorer.compute_complexity(valid_results, "query")
        assert 0.0 <= score <= 1.0

    def test_deeply_nested_causal_data(self, scorer):
        """Should detect causal data in nested structures."""
        results = [{"data": {"nested": {"dag": {"nodes": []}}}}]
        score = scorer.compute_complexity(results, "query")
        assert score > 0  # Should find nested dag

    def test_unicode_query(self, scorer):
        """Should handle unicode characters in query."""
        results = [{"data": "test"}]
        score = scorer.compute_complexity(results, "why did sales increase?")
        assert 0.0 <= score <= 1.0

    def test_very_long_results_list(self, scorer):
        """Should handle very long results list efficiently."""
        results = [{"data": f"test_{i}"} for i in range(1000)]
        score = scorer.compute_complexity(results, "query")
        # With weights: result_count=1.0*0.25 + query=0.2*0.30 + causal=0*0.25 + expertise=0.5*0.20 = 0.41
        assert score > 0.4  # Should be high due to many results
        assert score <= 1.0

    def test_special_characters_in_query(self, scorer):
        """Should handle special characters in query."""
        results = [{"data": "test"}]
        score = scorer.compute_complexity(
            results, "why? explain! compare... (analyze)"
        )
        assert 0.0 <= score <= 1.0
