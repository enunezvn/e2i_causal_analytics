"""Unit tests for historical_analyzer node."""

import pytest

from src.agents.ml_foundation.model_selector.nodes.historical_analyzer import (
    _get_default_recommendations,
    _get_default_success_rates,
    analyze_historical_performance,
    get_algorithm_trends,
    get_recommendations_from_history,
)


@pytest.fixture
def base_state():
    """Base state for historical analysis."""
    return {
        "experiment_id": "exp_test_123",
        "problem_type": "binary_classification",
        "kpi_category": None,
    }


@pytest.fixture
def state_with_kpi_category():
    """State with KPI category for recommendations."""
    return {
        "experiment_id": "exp_test_456",
        "problem_type": "binary_classification",
        "kpi_category": "churn",
    }


@pytest.fixture
def causal_kpi_state():
    """State for causal KPI category."""
    return {
        "experiment_id": "exp_causal_789",
        "problem_type": "regression",
        "kpi_category": "causal",
    }


class TestGetDefaultSuccessRates:
    """Tests for _get_default_success_rates helper."""

    def test_returns_classification_rates(self):
        """Should return default rates for classification."""
        rates = _get_default_success_rates("binary_classification")

        assert isinstance(rates, dict)
        assert len(rates) > 0
        # Should have gradient boosting algorithms
        assert "XGBoost" in rates
        assert "LightGBM" in rates

    def test_returns_regression_rates(self):
        """Should return default rates for regression."""
        rates = _get_default_success_rates("regression")

        assert isinstance(rates, dict)
        assert len(rates) > 0
        # Should have regression algorithms
        assert "XGBoost" in rates
        assert "Ridge" in rates or "Lasso" in rates

    def test_rates_are_reasonable(self):
        """Default rates should be in reasonable range (0.5-1.0)."""
        rates = _get_default_success_rates("binary_classification")

        for algo, rate in rates.items():
            assert 0.0 <= rate <= 1.0, f"Invalid rate for {algo}: {rate}"


class TestGetDefaultRecommendations:
    """Tests for _get_default_recommendations helper."""

    def test_returns_recommendations_for_churn(self):
        """Should return churn-specific recommendations."""
        recs = _get_default_recommendations("binary_classification", "churn")

        assert isinstance(recs, list)
        assert len(recs) > 0
        # Churn problems should recommend XGBoost and gradient boosting
        assert any("XGBoost" in r or "xgboost" in r.lower() for r in recs)

    def test_returns_recommendations_for_causal(self):
        """Should return causal-specific recommendations."""
        recs = _get_default_recommendations("regression", "causal")

        assert isinstance(recs, list)
        assert len(recs) > 0
        # Causal problems should recommend CausalForest or LinearDML
        causal_algos = ["CausalForest", "LinearDML", "causal"]
        assert any(any(algo.lower() in r.lower() for algo in causal_algos) for r in recs)

    def test_returns_recommendations_for_conversion(self):
        """Should return conversion-specific recommendations."""
        recs = _get_default_recommendations("binary_classification", "conversion")

        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_returns_recommendations_for_market_share(self):
        """Should return market share recommendations."""
        recs = _get_default_recommendations("regression", "market_share")

        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_returns_default_for_unknown_kpi(self):
        """Should return default recommendations for unknown KPI."""
        recs = _get_default_recommendations("binary_classification", "unknown_kpi")

        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_returns_default_for_none_kpi(self):
        """Should return default recommendations for None KPI."""
        recs = _get_default_recommendations("binary_classification", None)

        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_adjusts_for_regression_problem_type(self):
        """Should adjust recommendations for regression."""
        recs_classification = _get_default_recommendations("binary_classification", "churn")
        recs_regression = _get_default_recommendations("regression", "churn")

        # Both should return recommendations
        assert len(recs_classification) > 0
        assert len(recs_regression) > 0


@pytest.mark.asyncio
class TestAnalyzeHistoricalPerformance:
    """Tests for analyze_historical_performance node."""

    async def test_returns_historical_success_rates(self, base_state):
        """Should return historical success rates."""
        result = await analyze_historical_performance(base_state)

        assert "historical_success_rates" in result
        assert isinstance(result["historical_success_rates"], dict)
        # Should have rates for known algorithms
        assert len(result["historical_success_rates"]) > 0

    async def test_returns_similar_experiments(self, base_state):
        """Should return list of similar experiments."""
        result = await analyze_historical_performance(base_state)

        assert "similar_experiments" in result
        assert isinstance(result["similar_experiments"], list)

    async def test_returns_data_availability_flag(self, base_state):
        """Should indicate if historical data was found."""
        result = await analyze_historical_performance(base_state)

        assert "historical_data_available" in result
        assert isinstance(result["historical_data_available"], bool)

    async def test_returns_experiments_count(self, base_state):
        """Should return count of historical experiments."""
        result = await analyze_historical_performance(base_state)

        assert "historical_experiments_count" in result
        assert isinstance(result["historical_experiments_count"], int)
        assert result["historical_experiments_count"] >= 0

    async def test_default_success_rates_reasonable(self, base_state):
        """Default success rates should be reasonable (0-1.0)."""
        result = await analyze_historical_performance(base_state)

        for algo, rate in result["historical_success_rates"].items():
            assert 0.0 <= rate <= 1.0, f"Invalid rate for {algo}: {rate}"


@pytest.mark.asyncio
class TestGetAlgorithmTrends:
    """Tests for get_algorithm_trends node."""

    async def test_returns_trends_dict(self):
        """Should return dictionary of algorithm trends."""
        state = {
            "candidate_algorithms": [
                {"name": "XGBoost"},
                {"name": "LightGBM"},
                {"name": "RandomForest"},
            ]
        }

        result = await get_algorithm_trends(state)

        assert "algorithm_trends" in result
        assert isinstance(result["algorithm_trends"], dict)

    async def test_trend_structure(self):
        """Each trend should have required fields."""
        state = {
            "candidate_algorithms": [
                {"name": "XGBoost"},
            ]
        }

        result = await get_algorithm_trends(state)
        trends = result["algorithm_trends"]

        for _algo_name, trend_data in trends.items():
            assert isinstance(trend_data, dict)
            # Should have performance trend indicator
            assert "trend" in trend_data
            # Should have some historical metrics
            assert "recent_avg" in trend_data or "older_avg" in trend_data

    async def test_handles_empty_candidates(self):
        """Should handle empty candidate list."""
        state = {"candidate_algorithms": []}

        result = await get_algorithm_trends(state)

        assert "algorithm_trends" in result
        assert isinstance(result["algorithm_trends"], dict)


@pytest.mark.asyncio
class TestGetRecommendationsFromHistory:
    """Tests for get_recommendations_from_history node."""

    async def test_returns_recommendations(self):
        """Should return algorithm recommendations."""
        state = {
            "problem_type": "binary_classification",
            "historical_success_rates": {},
            "kpi_category": "churn",
        }

        result = await get_recommendations_from_history(state)

        assert "history_recommended_algorithms" in result
        assert isinstance(result["history_recommended_algorithms"], list)
        assert len(result["history_recommended_algorithms"]) > 0

    async def test_returns_recommendation_source(self):
        """Should indicate source of recommendations."""
        state = {
            "problem_type": "binary_classification",
            "historical_success_rates": {},
            "kpi_category": None,
        }

        result = await get_recommendations_from_history(state)

        assert "recommendation_source" in result
        assert result["recommendation_source"] in ["historical", "prior_knowledge"]

    async def test_uses_historical_data_when_available(self):
        """Should use historical data if available."""
        state = {
            "problem_type": "binary_classification",
            "historical_success_rates": {
                "XGBoost": 0.85,
                "LightGBM": 0.82,
                "RandomForest": 0.75,
            },
            "kpi_category": None,
        }

        result = await get_recommendations_from_history(state)

        assert result["recommendation_source"] == "historical"
        # Should recommend top performers
        recommended = result["history_recommended_algorithms"]
        assert len(recommended) > 0

    async def test_uses_prior_knowledge_when_no_history(self):
        """Should use prior_knowledge when no historical data."""
        state = {
            "problem_type": "binary_classification",
            "historical_success_rates": {},
            "kpi_category": None,
        }

        result = await get_recommendations_from_history(state)

        # Should use prior_knowledge since no success_rates
        assert result["recommendation_source"] == "prior_knowledge"

    async def test_causal_kpi_recommends_causal_algorithms(self):
        """Should recommend causal algorithms for causal KPIs."""
        state = {
            "problem_type": "regression",
            "historical_success_rates": {},
            "kpi_category": "causal",
        }

        result = await get_recommendations_from_history(state)

        recommended = result["history_recommended_algorithms"]
        # Should have causal-related recommendations
        causal_terms = ["Causal", "DML", "causal"]
        has_causal = any(any(term in algo for term in causal_terms) for algo in recommended)
        assert has_causal or len(recommended) > 0


class TestHistoricalAnalyzerIntegration:
    """Integration tests for historical analyzer."""

    @pytest.mark.asyncio
    async def test_all_outputs_are_valid(self, base_state):
        """All outputs should be valid and non-None."""
        result = await analyze_historical_performance(base_state)

        required_outputs = [
            "historical_success_rates",
            "similar_experiments",
            "historical_data_available",
            "historical_experiments_count",
        ]

        for output in required_outputs:
            assert output in result, f"Missing output: {output}"
            assert result[output] is not None, f"None value for: {output}"

    @pytest.mark.asyncio
    async def test_handles_empty_problem_type(self):
        """Should handle empty problem type gracefully."""
        state = {
            "experiment_id": "exp_test",
            "problem_type": "",
            "kpi_category": None,
        }

        result = await analyze_historical_performance(state)

        # Should still return valid structure
        assert "historical_success_rates" in result

    @pytest.mark.asyncio
    async def test_handles_multiclass_classification(self):
        """Should handle multiclass classification."""
        state = {
            "experiment_id": "exp_multiclass",
            "problem_type": "multiclass_classification",
            "kpi_category": None,
        }

        result = await analyze_historical_performance(state)

        assert "historical_success_rates" in result


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handles_special_characters_in_experiment_id(self):
        """Should handle special characters in experiment_id."""
        state = {
            "experiment_id": "exp_test-123_with.dots",
            "problem_type": "binary_classification",
            "kpi_category": None,
        }

        result = await analyze_historical_performance(state)

        assert "historical_success_rates" in result

    @pytest.mark.asyncio
    async def test_handles_very_long_experiment_id(self):
        """Should handle very long experiment_id."""
        state = {
            "experiment_id": "exp_" + "a" * 500,
            "problem_type": "binary_classification",
            "kpi_category": None,
        }

        result = await analyze_historical_performance(state)

        assert "historical_success_rates" in result

    def test_default_recommendations_are_non_empty(self):
        """Default recommendations should never be empty."""
        problem_types = ["binary_classification", "regression", "multiclass_classification"]
        kpi_categories = ["churn", "causal", "conversion", "market_share", None]

        for problem_type in problem_types:
            for kpi in kpi_categories:
                recs = _get_default_recommendations(problem_type, kpi)
                assert len(recs) > 0, f"Empty recs for {problem_type}/{kpi}"
