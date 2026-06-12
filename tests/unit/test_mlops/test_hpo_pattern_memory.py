"""Tests for HPO pattern memory module.

This module tests the procedural memory integration for hyperparameter
optimization patterns, including storage, retrieval, and warm-starting.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mlops import hpo_pattern_memory as hpo_mem
from src.mlops.hpo_pattern_memory import (
    HPOPatternInput,
    HPOPatternMatch,
    WarmStartConfig,
    cleanup_old_patterns,
    find_similar_patterns,
    get_pattern_stats,
    get_warmstart_hyperparameters,
    record_warmstart_outcome,
    store_hpo_pattern,
)

# ============================================================================
# DATA CLASS TESTS
# ============================================================================


class TestHPOPatternInput:
    """Tests for HPOPatternInput dataclass."""

    def test_creates_pattern_input_with_required_fields(self):
        """Should create pattern input with required fields."""
        pattern = HPOPatternInput(
            algorithm_name="XGBoost",
            problem_type="binary_classification",
            search_space={"n_estimators": {"type": "int", "low": 50, "high": 200}},
            best_hyperparameters={"n_estimators": 150},
            best_value=0.92,
            optimization_metric="roc_auc",
            n_trials=50,
            n_completed=48,
        )

        assert pattern.algorithm_name == "XGBoost"
        assert pattern.problem_type == "binary_classification"
        assert pattern.best_value == 0.92
        assert pattern.n_pruned == 0  # default
        assert pattern.duration_seconds == 0.0  # default

    def test_creates_pattern_input_with_all_fields(self):
        """Should create pattern input with all optional fields."""
        pattern = HPOPatternInput(
            algorithm_name="LightGBM",
            problem_type="regression",
            search_space={"learning_rate": {"type": "float", "low": 0.01, "high": 0.3}},
            best_hyperparameters={"learning_rate": 0.1},
            best_value=0.05,
            optimization_metric="rmse",
            n_trials=100,
            n_completed=95,
            n_pruned=5,
            duration_seconds=120.5,
            study_name="lightgbm_hpo_study",
            n_samples=10000,
            n_features=50,
            n_classes=None,
            class_balance=None,
            feature_types={"numeric": 40, "categorical": 10},
            experiment_id="exp_123",
        )

        assert pattern.n_pruned == 5
        assert pattern.n_samples == 10000
        assert pattern.feature_types == {"numeric": 40, "categorical": 10}


class TestHPOPatternMatch:
    """Tests for HPOPatternMatch dataclass."""

    def test_creates_pattern_match(self):
        """Should create pattern match result."""
        match = HPOPatternMatch(
            pattern_id="abc-123",
            algorithm_name="XGBoost",
            problem_type="binary_classification",
            best_hyperparameters={"n_estimators": 150},
            best_value=0.92,
            optimization_metric="roc_auc",
            n_samples=10000,
            n_features=50,
            similarity_score=0.85,
            times_used=3,
        )

        assert match.similarity_score == 0.85
        assert match.times_used == 3


class TestWarmStartConfig:
    """Tests for WarmStartConfig dataclass."""

    def test_creates_warmstart_config(self):
        """Should create warm-start config."""
        config = WarmStartConfig(
            initial_hyperparameters={"n_estimators": 150, "max_depth": 6},
            pattern_id="abc-123",
            similarity_score=0.85,
            original_best_value=0.92,
            algorithm_name="XGBoost",
        )

        assert config.initial_hyperparameters == {"n_estimators": 150, "max_depth": 6}
        assert config.similarity_score == 0.85


# ============================================================================
# STORAGE FUNCTION TESTS
# ============================================================================


class TestStoreHpoPattern:
    """Tests for store_hpo_pattern function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_supabase_unavailable(self):
        """Should return None when Supabase is not available."""
        with patch("src.mlops.hpo_pattern_memory._get_supabase_client", return_value=None):
            pattern = HPOPatternInput(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
                search_space={},
                best_hyperparameters={"n_estimators": 100},
                best_value=0.9,
                optimization_metric="roc_auc",
                n_trials=10,
                n_completed=10,
            )

            result = await store_hpo_pattern(pattern)

            assert result is None

    @pytest.mark.asyncio
    async def test_stores_pattern_successfully(self):
        """Should store pattern and return pattern ID."""
        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock()

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            pattern = HPOPatternInput(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
                search_space={"n_estimators": {"type": "int", "low": 50, "high": 200}},
                best_hyperparameters={"n_estimators": 150},
                best_value=0.92,
                optimization_metric="roc_auc",
                n_trials=50,
                n_completed=48,
            )

            result = await store_hpo_pattern(pattern)

            assert result is not None
            assert len(result) == 36  # UUID length
            assert mock_client.table.call_count == 2  # procedural_memories + ml_hpo_patterns

    @pytest.mark.asyncio
    async def test_handles_storage_error_gracefully(self):
        """Should return None on storage error."""
        mock_client = MagicMock()
        mock_client.table.return_value.insert.side_effect = Exception("DB error")

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            pattern = HPOPatternInput(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
                search_space={},
                best_hyperparameters={},
                best_value=0.9,
                optimization_metric="roc_auc",
                n_trials=10,
                n_completed=10,
            )

            result = await store_hpo_pattern(pattern)

            assert result is None


# ============================================================================
# RETRIEVAL FUNCTION TESTS
# ============================================================================


class TestFindSimilarPatterns:
    """Tests for find_similar_patterns function."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_supabase_unavailable(self):
        """Should return empty list when Supabase is not available."""
        with patch("src.mlops.hpo_pattern_memory._get_supabase_client", return_value=None):
            result = await find_similar_patterns(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
            )

            assert result == []

    @pytest.mark.asyncio
    async def test_returns_pattern_matches(self):
        """Should return list of pattern matches."""
        mock_client = MagicMock()
        mock_client.rpc.return_value.execute.return_value = MagicMock(
            data=[
                {
                    "pattern_id": "abc-123",
                    "algorithm_name": "XGBoost",
                    "problem_type": "binary_classification",
                    "best_hyperparameters": json.dumps({"n_estimators": 150}),
                    "best_value": 0.92,
                    "optimization_metric": "roc_auc",
                    "n_samples": 10000,
                    "n_features": 50,
                    "similarity_score": 0.85,
                    "times_used": 2,
                }
            ]
        )

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            result = await find_similar_patterns(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
                n_samples=10000,
                n_features=50,
            )

            assert len(result) == 1
            assert result[0].pattern_id == "abc-123"
            assert result[0].similarity_score == 0.85
            assert result[0].best_hyperparameters == {"n_estimators": 150}

    @pytest.mark.asyncio
    async def test_handles_already_parsed_json(self):
        """Should handle hyperparameters that are already dicts."""
        mock_client = MagicMock()
        mock_client.rpc.return_value.execute.return_value = MagicMock(
            data=[
                {
                    "pattern_id": "abc-123",
                    "algorithm_name": "XGBoost",
                    "problem_type": "binary_classification",
                    "best_hyperparameters": {"n_estimators": 150},  # Already a dict
                    "best_value": 0.92,
                    "optimization_metric": "roc_auc",
                    "n_samples": None,
                    "n_features": None,
                    "similarity_score": 0.8,
                    "times_used": 0,
                }
            ]
        )

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            result = await find_similar_patterns(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
            )

            assert len(result) == 1
            assert result[0].best_hyperparameters == {"n_estimators": 150}

    @pytest.mark.asyncio
    async def test_handles_retrieval_error_gracefully(self):
        """Should return empty list on retrieval error."""
        mock_client = MagicMock()
        mock_client.rpc.side_effect = Exception("DB error")

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            result = await find_similar_patterns(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
            )

            assert result == []


class TestGetWarmstartHyperparameters:
    """Tests for get_warmstart_hyperparameters function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_patterns_found(self):
        """Should return None when no patterns found."""
        with patch(
            "src.mlops.hpo_pattern_memory.find_similar_patterns",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await get_warmstart_hyperparameters(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_similarity_below_threshold(self):
        """Should return None when best match is below similarity threshold."""
        low_similarity_match = HPOPatternMatch(
            pattern_id="abc-123",
            algorithm_name="XGBoost",
            problem_type="binary_classification",
            best_hyperparameters={"n_estimators": 150},
            best_value=0.92,
            optimization_metric="roc_auc",
            n_samples=10000,
            n_features=50,
            similarity_score=0.3,  # Below default 0.5 threshold
            times_used=1,
        )

        with patch(
            "src.mlops.hpo_pattern_memory.find_similar_patterns",
            new_callable=AsyncMock,
            return_value=[low_similarity_match],
        ):
            result = await get_warmstart_hyperparameters(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
                min_similarity=0.5,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_warmstart_config_when_match_found(self):
        """Should return WarmStartConfig when good match found."""
        good_match = HPOPatternMatch(
            pattern_id="abc-123",
            algorithm_name="XGBoost",
            problem_type="binary_classification",
            best_hyperparameters={"n_estimators": 150, "max_depth": 6},
            best_value=0.92,
            optimization_metric="roc_auc",
            n_samples=10000,
            n_features=50,
            similarity_score=0.85,
            times_used=5,
        )

        with patch(
            "src.mlops.hpo_pattern_memory.find_similar_patterns",
            new_callable=AsyncMock,
            return_value=[good_match],
        ):
            result = await get_warmstart_hyperparameters(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
                min_similarity=0.5,
            )

            assert result is not None
            assert isinstance(result, WarmStartConfig)
            assert result.pattern_id == "abc-123"
            assert result.similarity_score == 0.85
            assert result.initial_hyperparameters == {"n_estimators": 150, "max_depth": 6}


# ============================================================================
# OUTCOME TRACKING TESTS
# ============================================================================


class TestRecordWarmstartOutcome:
    """Tests for record_warmstart_outcome function."""

    @pytest.mark.asyncio
    async def test_does_nothing_when_supabase_unavailable(self):
        """Should silently return when Supabase is not available."""
        with patch("src.mlops.hpo_pattern_memory._get_supabase_client", return_value=None):
            # Should not raise
            await record_warmstart_outcome(
                pattern_id="abc-123",
                new_best_value=0.95,
                original_best_value=0.92,
            )

    @pytest.mark.asyncio
    async def test_records_improvement(self):
        """Should record improvement via RPC call."""
        mock_client = MagicMock()
        mock_client.rpc.return_value.execute.return_value = MagicMock()

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            await record_warmstart_outcome(
                pattern_id="abc-123",
                new_best_value=0.95,
                original_best_value=0.92,
            )

            # Verify RPC was called with correct arguments (use approx for float comparison)
            mock_client.rpc.assert_called_once()
            call_args = mock_client.rpc.call_args
            assert call_args[0][0] == "record_hpo_warmstart_usage"
            assert call_args[0][1]["p_pattern_id"] == "abc-123"
            assert abs(call_args[0][1]["p_improvement"] - 0.03) < 1e-6

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(self):
        """Should not raise on RPC error."""
        mock_client = MagicMock()
        mock_client.rpc.side_effect = Exception("DB error")

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            # Should not raise
            await record_warmstart_outcome(
                pattern_id="abc-123",
                new_best_value=0.95,
                original_best_value=0.92,
            )


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================


class TestGetPatternStats:
    """Tests for get_pattern_stats function."""

    @pytest.mark.asyncio
    async def test_returns_unavailable_when_no_client(self):
        """Should return unavailable status when Supabase not available."""
        with patch("src.mlops.hpo_pattern_memory._get_supabase_client", return_value=None):
            result = await get_pattern_stats()

            assert result["available"] is False

    @pytest.mark.asyncio
    async def test_returns_empty_stats_when_no_patterns(self):
        """Should return empty stats when no patterns exist."""
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.execute.return_value = MagicMock(data=[])

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            result = await get_pattern_stats()

            assert result["total_patterns"] == 0
            assert result["by_algorithm"] == {}

    @pytest.mark.asyncio
    async def test_aggregates_stats_by_algorithm(self):
        """Should aggregate statistics by algorithm."""
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.execute.return_value = MagicMock(
            data=[
                {
                    "algorithm_name": "XGBoost",
                    "problem_type": "binary_classification",
                    "best_value": 0.9,
                    "times_used_as_warmstart": 5,
                },
                {
                    "algorithm_name": "XGBoost",
                    "problem_type": "regression",
                    "best_value": 0.8,
                    "times_used_as_warmstart": 3,
                },
                {
                    "algorithm_name": "LightGBM",
                    "problem_type": "binary_classification",
                    "best_value": 0.85,
                    "times_used_as_warmstart": 2,
                },
            ]
        )

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            result = await get_pattern_stats()

            assert result["total_patterns"] == 3
            assert result["total_warmstarts"] == 10
            assert "XGBoost" in result["by_algorithm"]
            assert result["by_algorithm"]["XGBoost"]["count"] == 2
            assert "binary_classification" in result["by_algorithm"]["XGBoost"]["problem_types"]
            assert "regression" in result["by_algorithm"]["XGBoost"]["problem_types"]


class TestCleanupOldPatterns:
    """Tests for cleanup_old_patterns function."""

    @pytest.mark.asyncio
    async def test_returns_unavailable_when_no_client(self):
        """Should return unavailable when Supabase not available."""
        with patch("src.mlops.hpo_pattern_memory._get_supabase_client", return_value=None):
            result = await cleanup_old_patterns()

            assert result["cleaned"] == 0
            assert result["reason"] == "Supabase not available"

    @pytest.mark.asyncio
    async def test_dry_run_returns_would_clean(self):
        """Should return patterns that would be cleaned in dry run."""
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.lt.return_value.lte.return_value.execute.return_value = MagicMock(
            data=[
                {"pattern_id": "abc-123", "algorithm_name": "XGBoost"},
                {"pattern_id": "def-456", "algorithm_name": "LightGBM"},
            ]
        )

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            result = await cleanup_old_patterns(days_old=90, dry_run=True)

            assert result["would_clean"] == 2
            assert result["dry_run"] is True
            assert "abc-123" in result["patterns"]
            assert "def-456" in result["patterns"]

    @pytest.mark.asyncio
    async def test_actual_cleanup_deletes_patterns(self):
        """Should delete patterns when not in dry run."""
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.lt.return_value.lte.return_value.execute.return_value = MagicMock(
            data=[{"pattern_id": "abc-123", "algorithm_name": "XGBoost"}]
        )
        mock_client.table.return_value.delete.return_value.eq.return_value.execute.return_value = (
            MagicMock()
        )

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            result = await cleanup_old_patterns(days_old=90, dry_run=False)

            assert result["cleaned"] == 1
            assert result["dry_run"] is False


# ============================================================================
# SEARCH-SPACE SERIALIZATION TESTS (issue #758)
# ============================================================================
#
# Regression: `search_space` may hold real Optuna distribution objects
# (FloatDistribution / IntDistribution / CategoricalDistribution), which the
# stdlib ``json`` encoder cannot serialize. Previously ``store_hpo_pattern``
# did ``json.dumps(pattern.search_space)`` → ``TypeError`` → silently caught →
# the procedural-memory write was skipped (warm-start cache miss). The fix
# serializes distributions with Optuna's own serializer and deserializes
# symmetrically on read-back.


class TestSearchSpaceSerialization:
    """Round-trip Optuna distributions through the search-space (de)serializers."""

    def _optuna_search_space(self):
        from optuna.distributions import (
            CategoricalDistribution,
            FloatDistribution,
            IntDistribution,
        )

        return {
            "learning_rate": FloatDistribution(1e-4, 0.1, log=True),
            "n_estimators": IntDistribution(50, 500),
            "booster": CategoricalDistribution(["gbtree", "dart"]),
        }

    def test_roundtrips_optuna_distributions(self):
        """Optuna distribution objects survive serialize → JSON-native dict → deserialize."""
        search_space = self._optuna_search_space()

        raw = hpo_mem._serialize_search_space(search_space)

        # #883 deferred fix: must be a JSON-NATIVE structure (a pre-dumped
        # string would be double-encoded by postgrest into a JSONB string
        # scalar) with no un-encodable objects leaked.
        assert isinstance(raw, dict)
        json.dumps(raw)  # fully JSON-native, i.e. no un-encodable objects leaked

        restored = hpo_mem._deserialize_search_space(raw)
        assert restored == search_space  # Optuna distributions compare by value

    def test_roundtrips_plain_e2i_dicts_unchanged(self):
        """Plain E2I-format dicts (already JSON-safe) pass through untouched."""
        search_space = {
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        }

        restored = hpo_mem._deserialize_search_space(hpo_mem._serialize_search_space(search_space))
        assert restored == search_space

    def test_deserialize_handles_empty_and_none(self):
        """Empty / None inputs deserialize to an empty dict (no crash)."""
        assert hpo_mem._deserialize_search_space(None) == {}
        assert hpo_mem._deserialize_search_space(hpo_mem._serialize_search_space({})) == {}

    @pytest.mark.asyncio
    async def test_store_pattern_with_optuna_distributions_persists(self):
        """A search_space of real Optuna distributions stores without silent failure."""
        search_space = self._optuna_search_space()
        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock()

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            pattern = HPOPatternInput(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
                search_space=search_space,
                best_hyperparameters={"n_estimators": 150},
                best_value=0.92,
                optimization_metric="roc_auc",
                n_trials=50,
                n_completed=48,
            )

            result = await store_hpo_pattern(pattern)

        # Before the fix this is None (json.dumps raised, caught at the warning).
        assert result is not None
        assert len(result) == 36  # UUID

        # The ml_hpo_patterns insert is the 2nd insert call; its search_space
        # payload must round-trip back to the original Optuna distributions.
        insert_calls = mock_client.table.return_value.insert.call_args_list
        hpo_record = insert_calls[1].args[0]
        assert hpo_record["pattern_id"] == result
        restored = hpo_mem._deserialize_search_space(hpo_record["search_space"])
        assert restored == search_space

    @pytest.mark.asyncio
    async def test_find_similar_patterns_deserializes_search_space(self):
        """find_similar_patterns exposes search_space deserialized to Optuna objects."""
        search_space = self._optuna_search_space()
        serialized = hpo_mem._serialize_search_space(search_space)

        mock_client = MagicMock()
        mock_client.rpc.return_value.execute.return_value = MagicMock(
            data=[
                {
                    "pattern_id": "abc-123",
                    "algorithm_name": "XGBoost",
                    "problem_type": "binary_classification",
                    "best_hyperparameters": json.dumps({"n_estimators": 150}),
                    "search_space": serialized,
                    "best_value": 0.92,
                    "optimization_metric": "roc_auc",
                    "n_samples": 10000,
                    "n_features": 50,
                    "similarity_score": 0.85,
                    "times_used": 2,
                }
            ]
        )

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            result = await find_similar_patterns(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
            )

        assert len(result) == 1
        assert result[0].search_space == search_space

    @pytest.mark.asyncio
    async def test_find_similar_patterns_search_space_defaults_empty(self):
        """A row without search_space yields an empty dict, not an error."""
        mock_client = MagicMock()
        mock_client.rpc.return_value.execute.return_value = MagicMock(
            data=[
                {
                    "pattern_id": "abc-123",
                    "algorithm_name": "XGBoost",
                    "problem_type": "binary_classification",
                    "best_hyperparameters": {"n_estimators": 150},
                    "best_value": 0.92,
                    "optimization_metric": "roc_auc",
                    "n_samples": None,
                    "n_features": None,
                    "similarity_score": 0.8,
                    "times_used": 0,
                }
            ]
        )

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            result = await find_similar_patterns(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
            )

        assert len(result) == 1
        assert result[0].search_space == {}

    def test_roundtrips_mixed_distributions_and_primitives(self):
        """A search space mixing Optuna distributions with plain values round-trips."""
        from optuna.distributions import FloatDistribution

        search_space = {
            "learning_rate": FloatDistribution(1e-4, 0.1, log=True),
            "fixed_param": 0.01,  # plain JSON-native value, must pass through
            "objective": "binary:logistic",
        }

        restored = hpo_mem._deserialize_search_space(hpo_mem._serialize_search_space(search_space))
        assert restored == search_space

    @pytest.mark.asyncio
    async def test_find_similar_patterns_tolerates_malformed_search_space(self):
        """A malformed tagged distribution must not discard the whole result set.

        The deserializer adds a new raise-site (json_to_distribution) into the
        find_similar_patterns row loop, whose outer except would otherwise drop
        ALL patterns if one row's search_space is corrupt. search_space is not
        consumed by warm-start, so a bad entry should degrade, not nuke results.
        """
        # Valid outer JSON, but the tagged payload is not valid Optuna JSON.
        malformed = json.dumps({"lr": {"__optuna_distribution__": "not-valid-optuna"}})
        mock_client = MagicMock()
        mock_client.rpc.return_value.execute.return_value = MagicMock(
            data=[
                {
                    "pattern_id": "abc-123",
                    "algorithm_name": "XGBoost",
                    "problem_type": "binary_classification",
                    "best_hyperparameters": json.dumps({"n_estimators": 150}),
                    "search_space": malformed,
                    "best_value": 0.92,
                    "optimization_metric": "roc_auc",
                    "n_samples": 10000,
                    "n_features": 50,
                    "similarity_score": 0.85,
                    "times_used": 2,
                }
            ]
        )

        with patch(
            "src.mlops.hpo_pattern_memory._get_supabase_client",
            return_value=mock_client,
        ):
            result = await find_similar_patterns(
                algorithm_name="XGBoost",
                problem_type="binary_classification",
            )

        # The pattern is still returned; the un-decodable entry is left as-is.
        assert len(result) == 1
        assert result[0].pattern_id == "abc-123"
        assert result[0].search_space == {"lr": {"__optuna_distribution__": "not-valid-optuna"}}
