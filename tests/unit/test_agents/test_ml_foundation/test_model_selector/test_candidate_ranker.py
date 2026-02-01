"""Tests for candidate_ranker.py ranking and selection logic."""

import pytest

from src.agents.ml_foundation.model_selector.nodes.candidate_ranker import (
    _compute_selection_score,
    _get_algorithm_class,
    rank_candidates,
    select_primary_candidate,
)


class TestComputeSelectionScore:
    """Test composite scoring algorithm."""

    def test_historical_success_rate_weighted_40_percent(self):
        """Historical success should contribute 40% to score."""
        candidate = {
            "name": "XGBoost",
            "inference_latency_ms": 1,  # Perfect
            "memory_gb": 0.1,  # Perfect
            "interpretability_score": 1.0,  # Perfect
            "family": "gradient_boosting",
        }
        success_rates = {"XGBoost": 1.0}  # 100% historical success

        score = _compute_selection_score(candidate, success_rates, [], 1000)

        # 1.0 * 0.4 (historical) + 1.0 * 0.2 (latency) + 1.0 * 0.15 (memory) + 1.0 * 0.15 (interp) = 0.9
        assert 0.85 <= score <= 0.95

    def test_causal_ml_gets_10_percent_bonus(self):
        """Causal ML algorithms should get 10% bonus (E2I preference)."""
        candidate_causal = {
            "name": "CausalForest",
            "inference_latency_ms": 50,
            "memory_gb": 4.0,
            "interpretability_score": 0.7,
            "family": "causal_ml",
        }
        candidate_non_causal = {
            "name": "XGBoost",
            "inference_latency_ms": 50,
            "memory_gb": 4.0,
            "interpretability_score": 0.7,
            "family": "gradient_boosting",
        }
        success_rates = {"CausalForest": 0.5, "XGBoost": 0.5}

        score_causal = _compute_selection_score(
            candidate_causal, success_rates, [], 1000, requires_causal=True
        )
        score_non_causal = _compute_selection_score(
            candidate_non_causal, success_rates, [], 1000, requires_causal=True
        )

        # Causal should have ~0.10 higher score when problem requires causal inference
        assert score_causal > score_non_causal
        assert abs(score_causal - score_non_causal - 0.10) < 0.05

    def test_user_preference_adds_10_percent_bonus(self):
        """User preferences should add 10% bonus."""
        candidate = {
            "name": "XGBoost",
            "inference_latency_ms": 50,
            "memory_gb": 2.0,
            "interpretability_score": 0.6,
            "family": "gradient_boosting",
        }
        success_rates = {"XGBoost": 0.5}

        score_without_pref = _compute_selection_score(candidate, success_rates, [], 1000)
        score_with_pref = _compute_selection_score(candidate, success_rates, ["XGBoost"], 1000)

        # Preference should add 0.10
        assert score_with_pref > score_without_pref
        assert abs(score_with_pref - score_without_pref - 0.10) < 0.01

    def test_low_latency_scores_higher(self):
        """Lower latency should score higher."""
        candidate_fast = {
            "name": "LinearDML",
            "inference_latency_ms": 10,
            "memory_gb": 1.0,
            "interpretability_score": 0.9,
            "family": "causal_ml",
        }
        candidate_slow = {
            "name": "CausalForest",
            "inference_latency_ms": 100,
            "memory_gb": 1.0,
            "interpretability_score": 0.9,
            "family": "causal_ml",
        }
        success_rates = {"LinearDML": 0.5, "CausalForest": 0.5}

        score_fast = _compute_selection_score(candidate_fast, success_rates, [], 1000)
        score_slow = _compute_selection_score(candidate_slow, success_rates, [], 1000)

        assert score_fast > score_slow

    def test_low_memory_scores_higher(self):
        """Lower memory usage should score higher."""
        candidate_low_mem = {
            "name": "LinearDML",
            "inference_latency_ms": 10,
            "memory_gb": 1.0,
            "interpretability_score": 0.9,
            "family": "causal_ml",
        }
        candidate_high_mem = {
            "name": "CausalForest",
            "inference_latency_ms": 10,
            "memory_gb": 8.0,
            "interpretability_score": 0.9,
            "family": "causal_ml",
        }
        success_rates = {"LinearDML": 0.5, "CausalForest": 0.5}

        score_low_mem = _compute_selection_score(candidate_low_mem, success_rates, [], 1000)
        score_high_mem = _compute_selection_score(candidate_high_mem, success_rates, [], 1000)

        assert score_low_mem > score_high_mem

    def test_high_interpretability_scores_higher(self):
        """Higher interpretability should score higher."""
        candidate_interpretable = {
            "name": "InterpretableModel",
            "inference_latency_ms": 10,
            "memory_gb": 1.0,
            "interpretability_score": 0.9,
            "family": "gradient_boosting",
        }
        candidate_black_box = {
            "name": "BlackBoxModel",
            "inference_latency_ms": 10,
            "memory_gb": 1.0,
            "interpretability_score": 0.3,
            "family": "gradient_boosting",
        }
        success_rates = {"InterpretableModel": 0.5, "BlackBoxModel": 0.5}

        score_interpretable = _compute_selection_score(
            candidate_interpretable, success_rates, [], 1000
        )
        score_black_box = _compute_selection_score(candidate_black_box, success_rates, [], 1000)

        assert score_interpretable > score_black_box

    def test_poor_scalability_penalized_for_large_datasets(self):
        """Poor scalability should be penalized for large datasets (>100k)."""
        candidate_poor_scalability = {
            "name": "TestAlgo",
            "inference_latency_ms": 10,
            "memory_gb": 1.0,
            "interpretability_score": 0.9,
            "family": "test",
            "scalability_score": 0.3,  # Poor scalability
        }
        success_rates = {"TestAlgo": 0.5}

        score_small_data = _compute_selection_score(
            candidate_poor_scalability, success_rates, [], 1000
        )
        score_large_data = _compute_selection_score(
            candidate_poor_scalability, success_rates, [], 200000
        )

        # Should be penalized for large dataset
        assert score_small_data > score_large_data

    def test_score_clamped_to_0_1_range(self):
        """Score should be clamped to [0, 1]."""
        candidate = {
            "name": "Perfect",
            "inference_latency_ms": 1,
            "memory_gb": 0.1,
            "interpretability_score": 1.0,
            "family": "causal_ml",
        }
        success_rates = {"Perfect": 1.0}
        preferences = ["Perfect"]

        score = _compute_selection_score(candidate, success_rates, preferences, 1000)

        assert 0.0 <= score <= 1.0

    def test_default_historical_success_rate_50_percent(self):
        """New algorithms without history should default to 50%."""
        candidate = {
            "name": "NewAlgo",
            "inference_latency_ms": 10,
            "memory_gb": 1.0,
            "interpretability_score": 0.9,
            "family": "causal_ml",
        }
        success_rates = {}  # No history

        score = _compute_selection_score(candidate, success_rates, [], 1000)

        # Should use 0.5 default for historical component
        # 0.5 * 0.4 = 0.2 from historical
        assert score > 0.0


@pytest.mark.asyncio
class TestRankCandidates:
    """Test candidate ranking node."""

    async def test_rank_candidates_sorts_by_selection_score(self):
        """Candidates should be sorted by selection score descending."""
        state = {
            "candidate_algorithms": [
                {
                    "name": "Slow",
                    "inference_latency_ms": 100,
                    "memory_gb": 8.0,
                    "interpretability_score": 0.3,
                    "family": "gradient_boosting",
                },
                {
                    "name": "Fast",
                    "inference_latency_ms": 10,
                    "memory_gb": 1.0,
                    "interpretability_score": 0.9,
                    "family": "gradient_boosting",
                },
                {
                    "name": "Medium",
                    "inference_latency_ms": 50,
                    "memory_gb": 4.0,
                    "interpretability_score": 0.7,
                    "family": "gradient_boosting",
                },
            ],
            "historical_success_rates": {"Slow": 0.5, "Fast": 0.5, "Medium": 0.5},
            "algorithm_preferences": [],
            "row_count": 1000,
        }

        result = await rank_candidates(state)

        # Verify ranked_candidates present
        assert "ranked_candidates" in result
        assert "selection_scores" in result

        # Verify sorting (Fast should be first due to better specs)
        ranked = result["ranked_candidates"]
        assert ranked[0]["name"] == "Fast"
        assert ranked[2]["name"] == "Slow"

        # Verify selection_score added to candidates
        for candidate in ranked:
            assert "selection_score" in candidate

    async def test_rank_candidates_no_candidates_error(self):
        """Should return error if no candidates."""
        state = {
            "candidate_algorithms": [],
            "historical_success_rates": {},
            "algorithm_preferences": [],
            "row_count": 1000,
        }

        result = await rank_candidates(state)

        assert "error" in result
        assert result["error_type"] == "no_candidates_error"

    async def test_selection_scores_dictionary_populated(self):
        """selection_scores dict should map algorithm name -> score."""
        state = {
            "candidate_algorithms": [
                {
                    "name": "XGBoost",
                    "inference_latency_ms": 20,
                    "memory_gb": 2.0,
                    "interpretability_score": 0.6,
                    "family": "gradient_boosting",
                },
                {
                    "name": "LinearDML",
                    "inference_latency_ms": 10,
                    "memory_gb": 1.0,
                    "interpretability_score": 0.9,
                    "family": "causal_ml",
                },
            ],
            "historical_success_rates": {"XGBoost": 0.8, "LinearDML": 0.7},
            "algorithm_preferences": [],
            "row_count": 1000,
        }

        result = await rank_candidates(state)

        scores = result["selection_scores"]
        assert "XGBoost" in scores
        assert "LinearDML" in scores
        assert isinstance(scores["XGBoost"], float)
        assert isinstance(scores["LinearDML"], float)


@pytest.mark.asyncio
class TestSelectPrimaryCandidate:
    """Test primary candidate selection."""

    async def test_select_primary_candidate_picks_top_ranked(self):
        """Should select top-ranked candidate as primary."""
        state = {
            "ranked_candidates": [
                {
                    "name": "Best",
                    "family": "causal_ml",
                    "selection_score": 0.95,
                    "inference_latency_ms": 10,
                    "memory_gb": 1.0,
                    "interpretability_score": 0.9,
                    "scalability_score": 0.9,
                    "default_hyperparameters": {"n_estimators": 500},
                    "hyperparameter_space": {},
                },
                {
                    "name": "Second",
                    "family": "gradient_boosting",
                    "selection_score": 0.85,
                    "inference_latency_ms": 20,
                    "memory_gb": 2.0,
                    "interpretability_score": 0.6,
                    "scalability_score": 0.8,
                    "default_hyperparameters": {},
                    "hyperparameter_space": {},
                },
            ]
        }

        result = await select_primary_candidate(state)

        assert result["algorithm_name"] == "Best"
        assert result["algorithm_family"] == "causal_ml"
        assert result["selection_score"] == 0.95

    async def test_select_primary_candidate_populates_all_fields(self):
        """Should populate all output fields."""
        state = {
            "ranked_candidates": [
                {
                    "name": "CausalForest",
                    "family": "causal_ml",
                    "selection_score": 0.90,
                    "inference_latency_ms": 50,
                    "memory_gb": 4.0,
                    "interpretability_score": 0.7,
                    "scalability_score": 0.8,
                    "default_hyperparameters": {"n_estimators": 500, "max_depth": 10},
                    "hyperparameter_space": {"n_estimators": {"type": "int"}},
                }
            ]
        }

        result = await select_primary_candidate(state)

        # Verify all required fields
        assert "primary_candidate" in result
        assert "alternative_candidates" in result
        assert "algorithm_name" in result
        assert "algorithm_family" in result
        assert "algorithm_class" in result
        assert "default_hyperparameters" in result
        assert "hyperparameter_search_space" in result
        assert "estimated_inference_latency_ms" in result
        assert "memory_requirement_gb" in result
        assert "interpretability_score" in result
        assert "scalability_score" in result
        assert "selection_score" in result

    async def test_alternative_candidates_includes_top_2_3(self):
        """Should include top 2-3 alternatives."""
        state = {
            "ranked_candidates": [
                {"name": "1st", "family": "gradient_boosting", "selection_score": 0.95},
                {"name": "2nd", "family": "linear", "selection_score": 0.85},
                {"name": "3rd", "family": "tree", "selection_score": 0.75},
                {"name": "4th", "family": "causal", "selection_score": 0.65},
                {"name": "5th", "family": "ensemble", "selection_score": 0.55},
            ]
        }

        result = await select_primary_candidate(state)

        alternatives = result["alternative_candidates"]
        assert len(alternatives) == 3
        assert alternatives[0]["name"] == "2nd"
        assert alternatives[1]["name"] == "3rd"
        assert alternatives[2]["name"] == "4th"

    async def test_no_alternatives_if_only_one_candidate(self):
        """Should have empty alternatives if only one candidate."""
        state = {
            "ranked_candidates": [
                {"name": "OnlyOne", "family": "gradient_boosting", "selection_score": 0.90}
            ]
        }

        result = await select_primary_candidate(state)

        assert len(result["alternative_candidates"]) == 0

    async def test_no_ranked_candidates_error(self):
        """Should return error if no ranked candidates."""
        state = {"ranked_candidates": []}

        result = await select_primary_candidate(state)

        assert "error" in result
        assert result["error_type"] == "no_ranked_candidates_error"


class TestGetAlgorithmClass:
    """Test algorithm class path mapping."""

    def test_causal_forest_class_path(self):
        """CausalForest should map to econml.dml.CausalForestDML."""
        candidate = {"name": "CausalForest", "framework": "econml"}
        assert _get_algorithm_class(candidate) == "econml.dml.CausalForestDML"

    def test_linear_dml_class_path(self):
        """LinearDML should map to econml.dml.LinearDML."""
        candidate = {"name": "LinearDML", "framework": "econml"}
        assert _get_algorithm_class(candidate) == "econml.dml.LinearDML"

    def test_xgboost_class_path(self):
        """XGBoost should map to xgboost.XGBClassifier."""
        candidate = {"name": "XGBoost", "framework": "xgboost"}
        assert _get_algorithm_class(candidate) == "xgboost.XGBClassifier"

    def test_lightgbm_class_path(self):
        """LightGBM should map to lightgbm.LGBMClassifier."""
        candidate = {"name": "LightGBM", "framework": "lightgbm"}
        assert _get_algorithm_class(candidate) == "lightgbm.LGBMClassifier"

    def test_random_forest_class_path(self):
        """RandomForest should map to sklearn.ensemble.RandomForestClassifier."""
        candidate = {"name": "RandomForest", "framework": "sklearn"}
        assert _get_algorithm_class(candidate) == "sklearn.ensemble.RandomForestClassifier"

    def test_logistic_regression_class_path(self):
        """LogisticRegression should map to sklearn.linear_model.LogisticRegression."""
        candidate = {"name": "LogisticRegression", "framework": "sklearn"}
        assert _get_algorithm_class(candidate) == "sklearn.linear_model.LogisticRegression"

    def test_ridge_class_path(self):
        """Ridge should map to sklearn.linear_model.Ridge."""
        candidate = {"name": "Ridge", "framework": "sklearn"}
        assert _get_algorithm_class(candidate) == "sklearn.linear_model.Ridge"

    def test_lasso_class_path(self):
        """Lasso should map to sklearn.linear_model.Lasso."""
        candidate = {"name": "Lasso", "framework": "sklearn"}
        assert _get_algorithm_class(candidate) == "sklearn.linear_model.Lasso"

    def test_unknown_algorithm_fallback(self):
        """Unknown algorithm should fallback to framework.AlgorithmName."""
        candidate = {"name": "UnknownAlgo", "framework": "custom_framework"}
        assert _get_algorithm_class(candidate) == "custom_framework.UnknownAlgo"
