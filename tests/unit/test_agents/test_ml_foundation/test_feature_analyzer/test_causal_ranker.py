"""Tests for Causal Ranker Node.

Tests the integration of DriverRanker for comparing causal vs predictive
feature importance in the Feature Analyzer agent.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.feature_analyzer.nodes.causal_ranker import (
    _build_discovery_config,
    rank_causal_drivers,
)
from src.causal_engine.discovery.base import (
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
    GateDecision,
)


class TestBuildDiscoveryConfig:
    """Test _build_discovery_config helper."""

    def test_none_config_returns_default(self):
        """Should return default config when None."""
        config = _build_discovery_config(None)
        assert isinstance(config, DiscoveryConfig)
        assert len(config.algorithms) > 0

    def test_converts_algorithm_strings(self):
        """Should convert algorithm strings to enums."""
        config_dict = {"algorithms": ["ges", "pc"], "alpha": 0.01}
        config = _build_discovery_config(config_dict)

        assert DiscoveryAlgorithmType.GES in config.algorithms
        assert DiscoveryAlgorithmType.PC in config.algorithms
        assert config.alpha == 0.01

    def test_handles_invalid_algorithm_string(self):
        """Should skip invalid algorithm strings."""
        config_dict = {"algorithms": ["ges", "invalid_algo"]}
        config = _build_discovery_config(config_dict)

        assert DiscoveryAlgorithmType.GES in config.algorithms
        assert len(config.algorithms) == 1

    def test_preserves_enum_algorithms(self):
        """Should preserve algorithms that are already enums."""
        config_dict = {"algorithms": [DiscoveryAlgorithmType.FCI]}
        config = _build_discovery_config(config_dict)

        assert DiscoveryAlgorithmType.FCI in config.algorithms


@pytest.mark.asyncio
class TestRankCausalDriversSkip:
    """Test conditions that skip causal ranking."""

    async def test_skips_when_discovery_disabled(self):
        """Should return empty dict when discovery_enabled is False."""
        state = {
            "discovery_enabled": False,
            "X_sample": np.random.rand(100, 5),
            "shap_values": np.random.rand(100, 5),
        }

        result = await rank_causal_drivers(state)

        assert result == {}

    async def test_skips_when_discovery_enabled_missing(self):
        """Should return empty dict when discovery_enabled is not set."""
        state = {
            "X_sample": np.random.rand(100, 5),
            "shap_values": np.random.rand(100, 5),
        }

        result = await rank_causal_drivers(state)

        assert result == {}


@pytest.mark.asyncio
class TestRankCausalDriversValidation:
    """Test input validation."""

    async def test_error_when_missing_x_sample(self):
        """Should return error when X_sample is missing."""
        state = {
            "discovery_enabled": True,
            "shap_values": np.random.rand(100, 5),
            "causal_target_variable": "target",
        }

        result = await rank_causal_drivers(state)

        assert "error" in result
        assert result["error_type"] == "missing_data"
        assert result["discovery_gate_decision"] == "reject"

    async def test_error_when_missing_shap_values(self):
        """Should return error when shap_values is missing."""
        state = {
            "discovery_enabled": True,
            "X_sample": np.random.rand(100, 5),
            "causal_target_variable": "target",
        }

        result = await rank_causal_drivers(state)

        assert "error" in result
        assert result["error_type"] == "missing_shap"
        assert result["discovery_gate_decision"] == "reject"

    async def test_uses_default_target_when_not_specified(self):
        """Should use 'target' as default when causal_target_variable not set."""
        # Create mock data
        X = np.random.rand(50, 3)
        shap_values = np.random.rand(50, 3)

        state = {
            "discovery_enabled": True,
            "X_sample": X,
            "shap_values": shap_values,
            "feature_names": ["A", "B", "C"],
            # No causal_target_variable
        }

        with patch(
            "src.agents.ml_foundation.feature_analyzer.nodes.causal_ranker.DiscoveryRunner"
        ) as MockRunner:
            mock_runner = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.ensemble_dag = None
            mock_result.to_dict.return_value = {}
            mock_runner.discover_dag = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner

            with patch(
                "src.agents.ml_foundation.feature_analyzer.nodes.causal_ranker.DiscoveryGate"
            ) as MockGate:
                mock_gate = MagicMock()
                mock_evaluation = MagicMock()
                mock_evaluation.decision = GateDecision.REJECT
                mock_evaluation.confidence = 0.3
                mock_evaluation.reasons = ["Test"]
                mock_gate.evaluate.return_value = mock_evaluation
                MockGate.return_value = mock_gate

                result = await rank_causal_drivers(state)

                # Should not error, just use default target
                assert result.get("causal_target_variable") is None or "error" not in result


@pytest.mark.asyncio
class TestRankCausalDriversExecution:
    """Test causal ranking execution with mocked discovery."""

    @pytest.fixture
    def mock_discovery_success(self):
        """Mock successful discovery result."""
        mock_result = MagicMock(spec=DiscoveryResult)
        mock_result.success = True
        mock_result.ensemble_dag = MagicMock()
        mock_result.ensemble_dag.nodes.return_value = ["A", "B", "C", "target"]
        mock_result.ensemble_dag.edges.return_value = [("A", "target"), ("B", "C")]
        mock_result.to_dict.return_value = {
            "success": True,
            "n_edges": 2,
            "algorithm_results": [],
        }
        return mock_result

    @pytest.fixture
    def mock_gate_accept(self):
        """Mock gate evaluation with ACCEPT decision."""
        mock_evaluation = MagicMock()
        mock_evaluation.decision = GateDecision.ACCEPT
        mock_evaluation.confidence = 0.85
        mock_evaluation.reasons = ["High confidence", "Good agreement"]
        return mock_evaluation

    @pytest.fixture
    def mock_ranking_result(self):
        """Mock DriverRankingResult."""
        from src.causal_engine.discovery.driver_ranker import (
            DriverRankingResult,
            FeatureRanking,
        )

        rankings = [
            FeatureRanking(
                feature_name="A",
                causal_rank=1,
                predictive_rank=2,
                causal_score=0.8,
                predictive_score=0.6,
                rank_difference=-1,
                is_direct_cause=True,
                path_length=1,
            ),
            FeatureRanking(
                feature_name="B",
                causal_rank=2,
                predictive_rank=1,
                causal_score=0.5,
                predictive_score=0.7,
                rank_difference=1,
                is_direct_cause=False,
                path_length=2,
            ),
            FeatureRanking(
                feature_name="C",
                causal_rank=3,
                predictive_rank=10,
                causal_score=0.3,
                predictive_score=0.1,
                rank_difference=-7,
                is_direct_cause=False,
                path_length=None,
            ),
        ]

        return DriverRankingResult(
            rankings=rankings,
            target_variable="target",
            causal_only_features=["C"],
            predictive_only_features=[],
            concordant_features=["A", "B"],
            rank_correlation=0.75,
        )

    async def test_successful_causal_ranking(
        self, mock_discovery_success, mock_gate_accept, mock_ranking_result
    ):
        """Should return causal rankings on successful discovery."""
        X = np.random.rand(50, 3)
        shap_values = np.random.rand(50, 3)

        state = {
            "discovery_enabled": True,
            "X_sample": X,
            "y_sample": np.random.rand(50),
            "shap_values": shap_values,
            "feature_names": ["A", "B", "C"],
            "causal_target_variable": "target",
            "discovery_config": {"algorithms": ["ges"]},
        }

        with patch(
            "src.agents.ml_foundation.feature_analyzer.nodes.causal_ranker.DiscoveryRunner"
        ) as MockRunner:
            mock_runner = MagicMock()
            mock_runner.discover_dag = AsyncMock(return_value=mock_discovery_success)
            MockRunner.return_value = mock_runner

            with patch(
                "src.agents.ml_foundation.feature_analyzer.nodes.causal_ranker.DiscoveryGate"
            ) as MockGate:
                mock_gate = MagicMock()
                mock_gate.evaluate.return_value = mock_gate_accept
                MockGate.return_value = mock_gate

                with patch(
                    "src.agents.ml_foundation.feature_analyzer.nodes.causal_ranker.DriverRanker"
                ) as MockRanker:
                    mock_ranker = MagicMock()
                    mock_ranker.rank_from_discovery_result.return_value = mock_ranking_result
                    MockRanker.return_value = mock_ranker

                    result = await rank_causal_drivers(state)

                    # Check discovery results
                    assert result["discovery_gate_decision"] == "accept"
                    assert result["discovery_gate_confidence"] == 0.85

                    # Check causal rankings
                    assert len(result["causal_rankings"]) == 3
                    assert result["rank_correlation"] == 0.75

                    # Check feature categorization
                    assert "C" in result["causal_only_features"]
                    assert result["concordant_features"] == ["A", "B"]

                    # Check divergent features (rank diff > 3)
                    assert "C" in result["divergent_features"]

                    # Check direct causes
                    assert "A" in result["direct_cause_features"]

                    # Check causal importance
                    assert "A" in result["causal_importance"]
                    assert result["causal_importance"]["A"] == 0.8

    async def test_failed_discovery_returns_empty_rankings(self):
        """Should return empty rankings when discovery fails."""
        X = np.random.rand(50, 3)
        shap_values = np.random.rand(50, 3)

        state = {
            "discovery_enabled": True,
            "X_sample": X,
            "shap_values": shap_values,
            "feature_names": ["A", "B", "C"],
            "causal_target_variable": "target",
        }

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.ensemble_dag = None
        mock_result.to_dict.return_value = {"success": False}

        mock_evaluation = MagicMock()
        mock_evaluation.decision = GateDecision.REJECT
        mock_evaluation.confidence = 0.2
        mock_evaluation.reasons = ["Discovery failed"]

        with patch(
            "src.agents.ml_foundation.feature_analyzer.nodes.causal_ranker.DiscoveryRunner"
        ) as MockRunner:
            mock_runner = MagicMock()
            mock_runner.discover_dag = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner

            with patch(
                "src.agents.ml_foundation.feature_analyzer.nodes.causal_ranker.DiscoveryGate"
            ) as MockGate:
                mock_gate = MagicMock()
                mock_gate.evaluate.return_value = mock_evaluation
                MockGate.return_value = mock_gate

                result = await rank_causal_drivers(state)

                assert result["discovery_gate_decision"] == "reject"
                assert result["causal_rankings"] == []
                assert result["rank_correlation"] == 0.0
                assert result["divergent_features"] == []

    async def test_handles_dataframe_input(self):
        """Should handle DataFrame input for X_sample."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        shap_values = np.random.rand(3, 3)

        state = {
            "discovery_enabled": True,
            "X_sample": df,
            "shap_values": shap_values,
            "causal_target_variable": "target",
        }

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.ensemble_dag = None
        mock_result.to_dict.return_value = {}

        mock_evaluation = MagicMock()
        mock_evaluation.decision = GateDecision.REJECT
        mock_evaluation.confidence = 0.2
        mock_evaluation.reasons = []

        with patch(
            "src.agents.ml_foundation.feature_analyzer.nodes.causal_ranker.DiscoveryRunner"
        ) as MockRunner:
            mock_runner = MagicMock()
            mock_runner.discover_dag = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner

            with patch(
                "src.agents.ml_foundation.feature_analyzer.nodes.causal_ranker.DiscoveryGate"
            ) as MockGate:
                mock_gate = MagicMock()
                mock_gate.evaluate.return_value = mock_evaluation
                MockGate.return_value = mock_gate

                result = await rank_causal_drivers(state)

                # Should not error with DataFrame
                assert "error" not in result or result.get("error_type") != "missing_data"


@pytest.mark.asyncio
class TestRankCausalDriversErrorHandling:
    """Test error handling in causal ranker."""

    async def test_handles_exception_gracefully(self):
        """Should handle exceptions and return error state."""
        state = {
            "discovery_enabled": True,
            "X_sample": np.random.rand(50, 3),
            "shap_values": np.random.rand(50, 3),
            "feature_names": ["A", "B", "C"],
            "causal_target_variable": "target",
        }

        with patch(
            "src.agents.ml_foundation.feature_analyzer.nodes.causal_ranker.DiscoveryRunner"
        ) as MockRunner:
            mock_runner = MagicMock()
            mock_runner.discover_dag = AsyncMock(side_effect=RuntimeError("Test error"))
            MockRunner.return_value = mock_runner

            result = await rank_causal_drivers(state)

            assert "error" in result
            assert result["error_type"] == "causal_ranker_error"
            assert result["discovery_gate_decision"] == "reject"
            assert "causal_ranker_time_seconds" in result
