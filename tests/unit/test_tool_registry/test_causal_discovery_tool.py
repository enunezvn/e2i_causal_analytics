"""
Tests for Causal Discovery Tools.

Tests cover:
- DiscoverDagInput and DiscoverDagOutput schemas
- RankDriversInput and RankDriversOutput schemas
- CausalDiscoveryTool initialization and invocation
- DriverRankerTool initialization and invocation
- DAG discovery with various algorithms
- Driver ranking with causal vs predictive importance
- Opik tracing integration
- Error handling and validation
- Tool registration
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.tool_registry.tools.causal_discovery import (
    CausalDiscoveryTool,
    DiscoverDagInput,
    DiscoverDagOutput,
    DriverRankerTool,
    FeatureRankingItem,
    RankDriversInput,
    RankDriversOutput,
    discover_dag,
    get_discovery_tool,
    get_ranker_tool,
    rank_drivers,
    register_all_discovery_tools,
    register_discover_dag_tool,
    register_rank_drivers_tool,
)


class TestDiscoverDagInput:
    """Tests for DiscoverDagInput schema."""

    def test_discover_dag_input_defaults(self):
        """Test input schema with default values."""
        input_data = DiscoverDagInput(data={"x": [1, 2, 3], "y": [4, 5, 6]})

        assert input_data.algorithms == ["ges", "pc"]
        assert input_data.ensemble_threshold == 0.5
        assert input_data.alpha == 0.05
        assert input_data.max_k is None
        assert input_data.node_names is None

    def test_discover_dag_input_custom_values(self):
        """Test input schema with custom values."""
        input_data = DiscoverDagInput(
            data={"x": [1, 2], "y": [3, 4]},
            algorithms=["ges", "pc", "lingam"],
            ensemble_threshold=0.7,
            alpha=0.01,
            max_k=3,
            node_names=["X", "Y"],
        )

        assert input_data.algorithms == ["ges", "pc", "lingam"]
        assert input_data.ensemble_threshold == 0.7
        assert input_data.alpha == 0.01
        assert input_data.max_k == 3
        assert input_data.node_names == ["X", "Y"]

    def test_discover_dag_input_validates_threshold(self):
        """Test that ensemble_threshold is validated."""
        with pytest.raises(ValidationError):
            DiscoverDagInput(
                data={"x": [1, 2]},
                ensemble_threshold=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValidationError):
            DiscoverDagInput(
                data={"x": [1, 2]},
                ensemble_threshold=-0.1,  # Invalid: < 0.0
            )

    def test_discover_dag_input_validates_alpha(self):
        """Test that alpha is validated."""
        with pytest.raises(ValidationError):
            DiscoverDagInput(data={"x": [1, 2]}, alpha=0.0)  # Invalid: too low

        with pytest.raises(ValidationError):
            DiscoverDagInput(data={"x": [1, 2]}, alpha=0.6)  # Invalid: too high


class TestDiscoverDagOutput:
    """Tests for DiscoverDagOutput schema."""

    def test_discover_dag_output_minimal(self):
        """Test output schema with minimal fields."""
        output = DiscoverDagOutput(
            success=True,
            ensemble_threshold=0.5,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        assert output.success is True
        assert output.n_edges == 0
        assert output.n_nodes == 0
        assert output.edge_list == []
        assert output.algorithms_used == []
        assert output.errors == []

    def test_discover_dag_output_full(self):
        """Test output schema with all fields."""
        output = DiscoverDagOutput(
            success=True,
            n_edges=5,
            n_nodes=3,
            edge_list=[{"source": "A", "target": "B", "confidence": 0.9, "type": "directed"}],
            algorithms_used=["ges", "pc"],
            algorithm_results={"ges": {"n_edges": 5, "runtime_seconds": 1.2, "converged": True}},
            ensemble_threshold=0.5,
            gate_decision="accept",
            gate_confidence=0.95,
            gate_reasons=["High edge confidence"],
            total_runtime_seconds=2.5,
            timestamp=datetime.now(timezone.utc).isoformat(),
            trace_id="test-trace-123",
        )

        assert output.n_edges == 5
        assert len(output.edge_list) == 1
        assert output.gate_decision == "accept"


class TestRankDriversInput:
    """Tests for RankDriversInput schema."""

    def test_rank_drivers_input_minimal(self):
        """Test input schema with minimal fields."""
        input_data = RankDriversInput(
            dag_edge_list=[{"source": "A", "target": "B"}],
            target="B",
            shap_values=[[0.5, 0.3], [0.6, 0.4]],
            feature_names=["A", "C"],
        )

        assert input_data.target == "B"
        assert len(input_data.dag_edge_list) == 1
        assert input_data.concordance_threshold == 2
        assert input_data.importance_percentile == 0.25

    def test_rank_drivers_input_custom_values(self):
        """Test input schema with custom values."""
        input_data = RankDriversInput(
            dag_edge_list=[{"source": "X", "target": "Y"}],
            target="Y",
            shap_values=[[0.1, 0.2], [0.3, 0.4]],
            feature_names=["X", "Z"],
            concordance_threshold=3,
            importance_percentile=0.1,
        )

        assert input_data.concordance_threshold == 3
        assert input_data.importance_percentile == 0.1

    def test_rank_drivers_input_validates_percentile(self):
        """Test that importance_percentile is validated."""
        with pytest.raises(ValidationError):
            RankDriversInput(
                dag_edge_list=[],
                target="Y",
                shap_values=[[]],
                feature_names=[],
                importance_percentile=1.5,  # Invalid
            )


class TestRankDriversOutput:
    """Tests for RankDriversOutput schema."""

    def test_rank_drivers_output_minimal(self):
        """Test output schema with minimal fields."""
        output = RankDriversOutput(
            success=True,
            target_variable="Y",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        assert output.success is True
        assert output.n_features == 0
        assert output.rankings == []
        assert output.errors == []

    def test_rank_drivers_output_with_rankings(self):
        """Test output schema with rankings."""
        rankings = [
            FeatureRankingItem(
                feature_name="X",
                causal_rank=1,
                predictive_rank=1,
                rank_difference=0,
                causal_score=0.9,
                predictive_score=0.85,
                is_direct_cause=True,
                path_length=1,
            )
        ]

        output = RankDriversOutput(
            success=True,
            target_variable="Y",
            rankings=rankings,
            n_features=1,
            rank_correlation=0.95,
            causal_only_features=["X"],
            predictive_only_features=[],
            concordant_features=["X"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        assert len(output.rankings) == 1
        assert output.rankings[0].feature_name == "X"
        assert output.rank_correlation == 0.95


class TestCausalDiscoveryTool:
    """Tests for CausalDiscoveryTool."""

    def test_tool_initialization(self):
        """Test tool initialization."""
        tool = CausalDiscoveryTool()

        assert tool.opik_enabled is True
        assert tool._runner is None
        assert tool._gate is None

    def test_tool_initialization_opik_disabled(self):
        """Test tool with Opik disabled."""
        tool = CausalDiscoveryTool(opik_enabled=False)

        assert tool.opik_enabled is False

    @pytest.mark.asyncio
    async def test_ensure_initialized(self):
        """Test lazy initialization."""
        tool = CausalDiscoveryTool()

        assert tool._runner is None

        with patch("src.causal_engine.discovery.DiscoveryRunner"):
            with patch("src.causal_engine.discovery.DiscoveryGate"):
                tool._ensure_initialized()

                assert tool._runner is not None
                assert tool._gate is not None

    @pytest.mark.asyncio
    async def test_invoke_with_dict_input(self):
        """Test invocation with dict input."""
        tool = CausalDiscoveryTool(opik_enabled=False)

        # Mock the discovery components
        mock_runner = MagicMock()
        mock_gate = MagicMock()

        # Create mock result
        from src.causal_engine.discovery import (
            DiscoveryResult,
        )

        mock_result = MagicMock(spec=DiscoveryResult)
        mock_result.n_edges = 2
        mock_result.n_nodes = 3
        mock_result.edges = []
        mock_result.algorithm_results = []
        mock_result.config = MagicMock()
        mock_result.config.ensemble_threshold = 0.5

        mock_evaluation = MagicMock()
        mock_evaluation.decision.value = "accept"
        mock_evaluation.confidence = 0.9
        mock_evaluation.reasons = ["Good structure"]

        mock_runner.discover_dag = AsyncMock(return_value=mock_result)
        mock_gate.evaluate.return_value = mock_evaluation

        tool._runner = mock_runner
        tool._gate = mock_gate

        # Test invocation
        input_dict = {"data": {"x": [1, 2, 3], "y": [4, 5, 6]}}

        result = await tool.invoke(input_dict)

        assert isinstance(result, DiscoverDagOutput)
        assert result.success is True
        assert result.n_edges == 2
        assert result.gate_decision == "accept"

    @pytest.mark.asyncio
    async def test_invoke_with_pydantic_input(self):
        """Test invocation with Pydantic input."""
        tool = CausalDiscoveryTool(opik_enabled=False)

        # Mock components
        mock_runner = MagicMock()
        mock_gate = MagicMock()

        from src.causal_engine.discovery import DiscoveryResult

        mock_result = MagicMock(spec=DiscoveryResult)
        mock_result.n_edges = 0
        mock_result.n_nodes = 2
        mock_result.edges = []
        mock_result.algorithm_results = []
        mock_result.config = MagicMock()
        mock_result.config.ensemble_threshold = 0.5

        mock_evaluation = MagicMock()
        mock_evaluation.decision.value = "review"
        mock_evaluation.confidence = 0.5
        mock_evaluation.reasons = []

        mock_runner.discover_dag = AsyncMock(return_value=mock_result)
        mock_gate.evaluate.return_value = mock_evaluation

        tool._runner = mock_runner
        tool._gate = mock_gate

        input_data = DiscoverDagInput(data={"x": [1, 2], "y": [3, 4]})

        result = await tool.invoke(input_data)

        assert result.success is True
        assert result.gate_decision == "review"

    @pytest.mark.asyncio
    async def test_invoke_handles_errors(self):
        """Test error handling during invocation."""
        tool = CausalDiscoveryTool(opik_enabled=False)

        # Mock components to raise exception
        mock_runner = MagicMock()
        mock_runner.discover_dag = AsyncMock(side_effect=ValueError("Test error"))

        tool._runner = mock_runner
        tool._gate = MagicMock()

        input_dict = {"data": {"x": [1, 2], "y": [3, 4]}}

        result = await tool.invoke(input_dict)

        assert result.success is False
        assert len(result.errors) > 0
        assert "Test error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_invoke_with_unknown_algorithm(self):
        """Test handling of unknown algorithms."""
        tool = CausalDiscoveryTool(opik_enabled=False)

        # Mock components
        mock_runner = MagicMock()
        mock_gate = MagicMock()

        from src.causal_engine.discovery import DiscoveryResult

        mock_result = MagicMock(spec=DiscoveryResult)
        mock_result.n_edges = 0
        mock_result.n_nodes = 2
        mock_result.edges = []
        mock_result.algorithm_results = []
        mock_result.config = MagicMock()
        mock_result.config.ensemble_threshold = 0.5

        mock_evaluation = MagicMock()
        mock_evaluation.decision.value = "accept"
        mock_evaluation.confidence = 0.8
        mock_evaluation.reasons = []

        mock_runner.discover_dag = AsyncMock(return_value=mock_result)
        mock_gate.evaluate.return_value = mock_evaluation

        tool._runner = mock_runner
        tool._gate = mock_gate

        input_dict = {
            "data": {"x": [1, 2], "y": [3, 4]},
            "algorithms": ["unknown_algo"],
        }

        result = await tool.invoke(input_dict)

        assert result.success is True
        assert any("Unknown algorithm" in err for err in result.errors)


class TestDriverRankerTool:
    """Tests for DriverRankerTool."""

    def test_tool_initialization(self):
        """Test tool initialization."""
        tool = DriverRankerTool()

        assert tool.opik_enabled is True
        assert tool._ranker is None

    @pytest.mark.asyncio
    async def test_ensure_initialized(self):
        """Test lazy initialization."""
        tool = DriverRankerTool()

        with patch("src.causal_engine.discovery.DriverRanker"):
            tool._ensure_initialized()

            assert tool._ranker is not None

    @pytest.mark.asyncio
    async def test_invoke_with_dict_input(self):
        """Test invocation with dict input."""
        tool = DriverRankerTool(opik_enabled=False)

        # Mock the ranker
        mock_ranker = MagicMock()

        from src.causal_engine.discovery import DriverRankingResult

        mock_result = MagicMock(spec=DriverRankingResult)
        mock_result.target_variable = "Y"
        mock_result.rankings = []
        mock_result.rank_correlation = 0.85
        mock_result.causal_only_features = []
        mock_result.predictive_only_features = []
        mock_result.concordant_features = []

        mock_ranker.rank_drivers.return_value = mock_result
        tool._ranker = mock_ranker

        input_dict = {
            "dag_edge_list": [{"source": "X", "target": "Y"}],
            "target": "Y",
            "shap_values": [[0.5, 0.3]],
            "feature_names": ["X", "Z"],
        }

        result = await tool.invoke(input_dict)

        assert isinstance(result, RankDriversOutput)
        assert result.success is True
        assert result.target_variable == "Y"

    @pytest.mark.asyncio
    async def test_invoke_with_pydantic_input(self):
        """Test invocation with Pydantic input."""
        tool = DriverRankerTool(opik_enabled=False)

        mock_ranker = MagicMock()

        from src.causal_engine.discovery import DriverRankingResult, FeatureRanking

        mock_ranking = MagicMock(spec=FeatureRanking)
        mock_ranking.feature_name = "X"
        mock_ranking.causal_rank = 1
        mock_ranking.predictive_rank = 1
        mock_ranking.rank_difference = 0
        mock_ranking.causal_score = 0.9
        mock_ranking.predictive_score = 0.85
        mock_ranking.is_direct_cause = True
        mock_ranking.path_length = 1

        mock_result = MagicMock(spec=DriverRankingResult)
        mock_result.target_variable = "Y"
        mock_result.rankings = [mock_ranking]
        mock_result.rank_correlation = 0.9
        mock_result.causal_only_features = ["X"]
        mock_result.predictive_only_features = []
        mock_result.concordant_features = ["X"]

        mock_ranker.rank_drivers.return_value = mock_result
        tool._ranker = mock_ranker

        input_data = RankDriversInput(
            dag_edge_list=[{"source": "X", "target": "Y"}],
            target="Y",
            shap_values=[[0.5]],
            feature_names=["X"],
        )

        result = await tool.invoke(input_data)

        assert result.success is True
        assert len(result.rankings) == 1
        assert result.rankings[0].feature_name == "X"

    @pytest.mark.asyncio
    async def test_invoke_handles_errors(self):
        """Test error handling during invocation."""
        tool = DriverRankerTool(opik_enabled=False)

        mock_ranker = MagicMock()
        mock_ranker.rank_drivers.side_effect = KeyError("Target not in DAG")

        tool._ranker = mock_ranker

        input_dict = {
            "dag_edge_list": [],
            "target": "Y",
            "shap_values": [[]],
            "feature_names": [],
        }

        result = await tool.invoke(input_dict)

        assert result.success is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_invoke_with_custom_thresholds(self):
        """Test invocation with custom thresholds."""
        tool = DriverRankerTool(opik_enabled=False)

        mock_ranker = MagicMock()
        mock_ranker.concordance_threshold = 2
        mock_ranker.importance_percentile = 0.25

        from src.causal_engine.discovery import DriverRankingResult

        mock_result = MagicMock(spec=DriverRankingResult)
        mock_result.target_variable = "Y"
        mock_result.rankings = []
        mock_result.rank_correlation = 0.8
        mock_result.causal_only_features = []
        mock_result.predictive_only_features = []
        mock_result.concordant_features = []

        mock_ranker.rank_drivers.return_value = mock_result
        tool._ranker = mock_ranker

        input_dict = {
            "dag_edge_list": [{"source": "X", "target": "Y"}],
            "target": "Y",
            "shap_values": [[0.5]],
            "feature_names": ["X"],
            "concordance_threshold": 5,
            "importance_percentile": 0.1,
        }

        result = await tool.invoke(input_dict)

        assert result.success is True
        # Verify thresholds were updated
        assert mock_ranker.concordance_threshold == 5
        assert mock_ranker.importance_percentile == 0.1


class TestToolFunctions:
    """Tests for top-level tool functions."""

    def test_get_discovery_tool_singleton(self):
        """Test that get_discovery_tool returns singleton."""
        tool1 = get_discovery_tool()
        tool2 = get_discovery_tool()

        assert tool1 is tool2

    def test_get_ranker_tool_singleton(self):
        """Test that get_ranker_tool returns singleton."""
        tool1 = get_ranker_tool()
        tool2 = get_ranker_tool()

        assert tool1 is tool2

    @pytest.mark.asyncio
    async def test_discover_dag_with_dataframe(self):
        """Test discover_dag function with DataFrame input."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # Mock the tool
        with patch("src.tool_registry.tools.causal_discovery.get_discovery_tool") as mock_get_tool:
            mock_tool = MagicMock()
            mock_output = DiscoverDagOutput(
                success=True,
                ensemble_threshold=0.5,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            mock_tool.invoke = AsyncMock(return_value=mock_output)
            mock_get_tool.return_value = mock_tool

            result = await discover_dag(df)

            assert isinstance(result, dict)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_discover_dag_with_dict(self):
        """Test discover_dag function with dict input."""
        data_dict = {"x": [1, 2, 3], "y": [4, 5, 6]}

        with patch("src.tool_registry.tools.causal_discovery.get_discovery_tool") as mock_get_tool:
            mock_tool = MagicMock()
            mock_output = DiscoverDagOutput(
                success=True,
                ensemble_threshold=0.5,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            mock_tool.invoke = AsyncMock(return_value=mock_output)
            mock_get_tool.return_value = mock_tool

            result = await discover_dag(data_dict)

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_rank_drivers_with_numpy_array(self):
        """Test rank_drivers function with numpy array."""
        shap_values = np.array([[0.5, 0.3], [0.6, 0.4]])

        with patch("src.tool_registry.tools.causal_discovery.get_ranker_tool") as mock_get_tool:
            mock_tool = MagicMock()
            mock_output = RankDriversOutput(
                success=True,
                target_variable="Y",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            mock_tool.invoke = AsyncMock(return_value=mock_output)
            mock_get_tool.return_value = mock_tool

            result = await rank_drivers(
                dag_edge_list=[{"source": "X", "target": "Y"}],
                target="Y",
                shap_values=shap_values,
                feature_names=["X", "Z"],
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_rank_drivers_with_list(self):
        """Test rank_drivers function with list input."""
        shap_values = [[0.5, 0.3], [0.6, 0.4]]

        with patch("src.tool_registry.tools.causal_discovery.get_ranker_tool") as mock_get_tool:
            mock_tool = MagicMock()
            mock_output = RankDriversOutput(
                success=True,
                target_variable="Y",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            mock_tool.invoke = AsyncMock(return_value=mock_output)
            mock_get_tool.return_value = mock_tool

            result = await rank_drivers(
                dag_edge_list=[{"source": "X", "target": "Y"}],
                target="Y",
                shap_values=shap_values,
                feature_names=["X", "Z"],
            )

            assert result["success"] is True


class TestToolRegistration:
    """Tests for tool registration functions."""

    @patch("src.tool_registry.tools.causal_discovery.get_registry")
    def test_register_discover_dag_tool(self, mock_get_registry):
        """Test discover_dag tool registration."""
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry

        register_discover_dag_tool()

        assert mock_registry.register.called
        call_args = mock_registry.register.call_args
        schema = call_args[1]["schema"]

        assert schema.name == "discover_dag"
        assert schema.source_agent == "causal_impact"
        assert schema.tier == 2

    @patch("src.tool_registry.tools.causal_discovery.get_registry")
    def test_register_rank_drivers_tool(self, mock_get_registry):
        """Test rank_drivers tool registration."""
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry

        register_rank_drivers_tool()

        assert mock_registry.register.called
        call_args = mock_registry.register.call_args
        schema = call_args[1]["schema"]

        assert schema.name == "rank_drivers"
        assert schema.source_agent == "causal_impact"
        assert schema.tier == 2

    @patch("src.tool_registry.tools.causal_discovery.register_discover_dag_tool")
    @patch("src.tool_registry.tools.causal_discovery.register_rank_drivers_tool")
    def test_register_all_discovery_tools(self, mock_rank, mock_discover):
        """Test registering all discovery tools."""
        register_all_discovery_tools()

        assert mock_discover.called
        assert mock_rank.called
