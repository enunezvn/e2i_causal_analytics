"""Tests for CohortConstructorAgent wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.agents.cohort_constructor import (
    CohortConfig,
    CohortConstructorAgent,
    CohortExecutionResult,
)
from src.agents.cohort_constructor.agent import create_cohort_constructor_agent


class TestCohortConstructorAgentInit:
    """Tests for agent initialization."""

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    @patch("src.agents.cohort_constructor.agent._get_cohort_mlflow_logger")
    @patch("src.agents.cohort_constructor.agent._get_cohort_opik_tracer")
    @patch("src.agents.cohort_constructor.agent.create_cohort_constructor_graph")
    def test_init_defaults(self, mock_graph, mock_opik, mock_mlflow, mock_supabase):
        """Test initialization with defaults."""
        mock_supabase.return_value = MagicMock()
        mock_mlflow.return_value = MagicMock()
        mock_opik.return_value = MagicMock()
        mock_graph.return_value = MagicMock()

        agent = CohortConstructorAgent()

        assert agent.use_graph is True
        assert agent.enable_observability is True

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    @patch("src.agents.cohort_constructor.agent._get_cohort_mlflow_logger")
    @patch("src.agents.cohort_constructor.agent._get_cohort_opik_tracer")
    def test_init_without_graph(self, mock_opik, mock_mlflow, mock_supabase):
        """Test initialization without graph mode."""
        mock_supabase.return_value = MagicMock()
        mock_mlflow.return_value = None
        mock_opik.return_value = None

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)

        assert agent.use_graph is False
        assert agent._graph is None

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    @patch("src.agents.cohort_constructor.agent._get_cohort_mlflow_logger")
    @patch("src.agents.cohort_constructor.agent._get_cohort_opik_tracer")
    def test_init_without_observability(self, mock_opik, mock_mlflow, mock_supabase):
        """Test initialization without observability."""
        mock_supabase.return_value = MagicMock()
        mock_mlflow.return_value = MagicMock()
        mock_opik.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)

        assert agent._mlflow_logger is None
        assert agent._opik_tracer is None


class TestCohortConstructorAgentProperties:
    """Tests for agent properties."""

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_metadata_property(self, mock_supabase):
        """Test metadata property returns agent info."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        metadata = agent.metadata

        assert "name" in metadata
        assert metadata["name"] == "cohort_constructor"
        assert "tier" in metadata
        assert metadata["tier"] == 0

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_supported_brands_property(self, mock_supabase):
        """Test supported brands property."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        brands = agent.supported_brands

        assert isinstance(brands, list)
        assert "remibrutinib" in brands
        assert "fabhalta" in brands
        assert "kisqali" in brands


class TestCohortConstructorAgentConfiguration:
    """Tests for configuration methods."""

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_get_brand_config(self, mock_supabase):
        """Test getting brand configuration."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        config = agent.get_brand_config("remibrutinib")

        assert isinstance(config, CohortConfig)
        assert config.brand == "remibrutinib"

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_get_brand_config_with_indication(self, mock_supabase):
        """Test getting brand configuration with indication."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        config = agent.get_brand_config("fabhalta", "c3g")

        assert config.brand == "fabhalta"
        assert config.indication == "c3g"

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_list_configurations(self, mock_supabase):
        """Test listing all configurations."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        configs = agent.list_configurations()

        assert isinstance(configs, dict)
        assert len(configs) > 0


class TestCohortConstructorAgentDirectExecution:
    """Tests for direct execution mode."""

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_run_sync_with_brand(self, mock_supabase, remibrutinib_patient_df):
        """Test synchronous execution with brand configuration."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        eligible_df, result = agent.run_sync(remibrutinib_patient_df, brand="remibrutinib")

        assert isinstance(result, CohortExecutionResult)
        assert result.status == "success"

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_run_sync_with_config(self, mock_supabase, sample_patient_df, sample_config):
        """Test synchronous execution with explicit config."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        eligible_df, result = agent.run_sync(sample_patient_df, config=sample_config)

        assert result.status == "success"

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_run_sync_no_config_raises(self, mock_supabase, sample_patient_df):
        """Test that run_sync without brand or config raises error."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)

        with pytest.raises(ValueError, match="Either brand or config must be provided"):
            agent.run_sync(sample_patient_df)


class TestCohortConstructorAgentAsyncExecution:
    """Tests for async execution."""

    @pytest.mark.asyncio
    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    async def test_run_direct_mode(self, mock_supabase, remibrutinib_patient_df):
        """Test async run in direct mode."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        eligible_df, result = await agent.run(remibrutinib_patient_df, brand="remibrutinib")

        assert isinstance(result, CohortExecutionResult)
        assert result.status == "success"

    @pytest.mark.asyncio
    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    @patch("src.agents.cohort_constructor.agent._get_cohort_mlflow_logger")
    @patch("src.agents.cohort_constructor.agent._get_cohort_opik_tracer")
    async def test_run_with_observability(
        self, mock_opik, mock_mlflow, mock_supabase, remibrutinib_patient_df
    ):
        """Test async run with observability enabled."""
        mock_supabase.return_value = MagicMock()
        mock_mlflow_instance = MagicMock()
        mock_mlflow.return_value = mock_mlflow_instance
        mock_opik.return_value = None  # Disable Opik to simplify test

        agent = CohortConstructorAgent(use_graph=False, enable_observability=True)
        eligible_df, result = await agent.run(remibrutinib_patient_df, brand="remibrutinib")

        assert result.status == "success"
        # MLflow should have been called
        mock_mlflow_instance.log_cohort_execution.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    async def test_run_with_environment(self, mock_supabase, remibrutinib_patient_df):
        """Test async run with environment parameter."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        eligible_df, result = await agent.run(
            remibrutinib_patient_df,
            brand="remibrutinib",
            environment="staging",
            executed_by="test_user",
        )

        assert result.status == "success"


class TestCohortConstructorAgentValidation:
    """Tests for configuration validation."""

    @pytest.mark.asyncio
    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    async def test_validate_config_with_brand(self, mock_supabase):
        """Test validating a brand configuration."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        result = await agent.validate_config(brand="remibrutinib")

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["config_summary"] is not None
        assert result["config_summary"]["brand"] == "remibrutinib"

    @pytest.mark.asyncio
    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    async def test_validate_config_no_params(self, mock_supabase):
        """Test validation fails without parameters."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        result = await agent.validate_config()

        assert result["valid"] is False
        assert "No brand or configuration provided" in result["errors"]

    @pytest.mark.asyncio
    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    async def test_validate_config_with_explicit_config(self, mock_supabase, sample_config):
        """Test validating explicit configuration."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        result = await agent.validate_config(config=sample_config)

        assert result["valid"] is True
        assert result["config_summary"]["cohort_name"] == sample_config.cohort_name

    @pytest.mark.asyncio
    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    async def test_validate_config_warnings(self, mock_supabase):
        """Test validation returns warnings for empty criteria."""
        mock_supabase.return_value = MagicMock()

        empty_config = CohortConfig(
            cohort_name="Empty Config",
            brand="test",
            indication="test",
            inclusion_criteria=[],
            exclusion_criteria=[],
            required_fields=[],
        )

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        result = await agent.validate_config(config=empty_config)

        assert result["valid"] is True
        assert len(result["warnings"]) >= 2  # Warnings for empty criteria


class TestCohortConstructorAgentFactory:
    """Tests for agent factory function."""

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_create_agent_default(self, mock_supabase):
        """Test creating agent with defaults."""
        mock_supabase.return_value = MagicMock()

        with patch(
            "src.agents.cohort_constructor.agent.create_cohort_constructor_graph"
        ) as mock_graph:
            mock_graph.return_value = MagicMock()
            agent = create_cohort_constructor_agent()

            assert isinstance(agent, CohortConstructorAgent)
            assert agent.enable_observability is True

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_create_agent_without_graph(self, mock_supabase):
        """Test creating agent without graph."""
        mock_supabase.return_value = MagicMock()

        agent = create_cohort_constructor_agent(use_graph=False)

        assert agent.use_graph is False

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_create_agent_without_observability(self, mock_supabase):
        """Test creating agent without observability."""
        mock_supabase.return_value = MagicMock()

        agent = create_cohort_constructor_agent(use_graph=False, enable_observability=False)

        assert agent.enable_observability is False
        assert agent._mlflow_logger is None


class TestCohortConstructorAgentGraphExecution:
    """Tests for graph-based execution."""

    @pytest.mark.asyncio
    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    @patch("src.agents.cohort_constructor.agent._get_cohort_mlflow_logger")
    @patch("src.agents.cohort_constructor.agent._get_cohort_opik_tracer")
    @patch("src.agents.cohort_constructor.agent.create_cohort_constructor_graph")
    async def test_run_graph_mode(
        self, mock_graph_fn, mock_opik, mock_mlflow, mock_supabase, sample_patient_df
    ):
        """Test async run in graph mode."""
        mock_supabase.return_value = MagicMock()
        mock_mlflow.return_value = None
        mock_opik.return_value = None

        # Mock the graph
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "cohort_id": "test_cohort",
                "eligible_patient_ids": ["P001", "P002"],
                "eligibility_stats": {
                    "total_input_patients": 5,
                    "eligible_patient_count": 2,
                },
                "execution_metadata": {
                    "execution_id": "exec_test",
                    "execution_time_ms": 100,
                },
                "status": "success",
            }
        )
        mock_graph_fn.return_value = mock_graph

        agent = CohortConstructorAgent(use_graph=True, enable_observability=False)
        eligible_df, result = await agent.run(sample_patient_df, brand="remibrutinib")

        assert isinstance(result, CohortExecutionResult)
        assert result.status == "success"
        mock_graph.ainvoke.assert_called_once()


class TestCohortConstructorAgentErrorHandling:
    """Tests for error handling in agent."""

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_handles_missing_required_fields(self, mock_supabase):
        """Test agent handles missing required fields gracefully."""
        mock_supabase.return_value = MagicMock()

        df = pd.DataFrame({"patient_journey_id": ["P001"]})  # Missing other fields

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        eligible_df, result = agent.run_sync(df, brand="remibrutinib")

        assert result.status == "failed"
        assert result.error_code == "CC_002"

    @pytest.mark.asyncio
    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    @patch("src.agents.cohort_constructor.agent._get_cohort_mlflow_logger")
    @patch("src.agents.cohort_constructor.agent._get_cohort_opik_tracer")
    async def test_observability_error_doesnt_fail_execution(
        self, mock_opik, mock_mlflow, mock_supabase, remibrutinib_patient_df
    ):
        """Test that observability errors don't fail the main execution."""
        mock_supabase.return_value = MagicMock()

        # MLflow that raises an error
        mock_mlflow_instance = MagicMock()
        mock_mlflow_instance.log_cohort_execution.side_effect = Exception("MLflow error")
        mock_mlflow.return_value = mock_mlflow_instance
        mock_opik.return_value = None

        agent = CohortConstructorAgent(use_graph=False, enable_observability=True)
        eligible_df, result = await agent.run(remibrutinib_patient_df, brand="remibrutinib")

        # Execution should still succeed
        assert result.status == "success"


class TestCohortConstructorAgentIntegration:
    """Integration tests for agent with brand configurations."""

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_remibrutinib_cohort_construction(self, mock_supabase, remibrutinib_patient_df):
        """Test cohort construction with Remibrutinib configuration."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        eligible_df, result = agent.run_sync(remibrutinib_patient_df, brand="remibrutinib")

        assert result.status == "success"
        # Should filter based on UAS7, antihistamine status, age
        assert len(eligible_df) <= len(remibrutinib_patient_df)

    @patch("src.agents.cohort_constructor.agent._get_supabase_client")
    def test_kisqali_cohort_construction(self, mock_supabase, kisqali_patient_df):
        """Test cohort construction with Kisqali configuration."""
        mock_supabase.return_value = MagicMock()

        agent = CohortConstructorAgent(use_graph=False, enable_observability=False)
        eligible_df, result = agent.run_sync(kisqali_patient_df, brand="kisqali")

        assert result.status == "success"
        # Should filter based on HR status, HER2 status, ECOG
        if len(eligible_df) > 0:
            # Verify HR+ patients
            assert all(eligible_df["hr_status"] == "positive")
            # Verify HER2- patients
            assert all(eligible_df["her2_status"] == "negative")
