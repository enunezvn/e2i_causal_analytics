"""Tests for Feast integration in ML Foundation Pipeline.

Tests cover:
- Feast adapter initialization in pipeline
- Feature freshness check in QC Gate
- Feature refs passed to model trainer
- Fallback behavior when Feast unavailable
- Pipeline result contains Feast metadata
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.tier_0 import MLFoundationPipeline, PipelineConfig, PipelineResult


class TestPipelineConfigFeast:
    """Test Feast-related pipeline configuration."""

    def test_default_feast_config(self):
        """Test default Feast configuration values."""
        config = PipelineConfig()

        assert config.enable_feast is True
        assert config.feast_feature_refs is None
        assert config.feast_freshness_check is True
        assert config.feast_max_staleness_hours == 24.0
        assert config.feast_fallback_enabled is True

    def test_custom_feast_config(self):
        """Test custom Feast configuration."""
        feature_refs = [
            "hcp_conversion_features:engagement_score",
            "hcp_conversion_features:trx_count",
        ]

        config = PipelineConfig(
            enable_feast=True,
            feast_feature_refs=feature_refs,
            feast_freshness_check=True,
            feast_max_staleness_hours=12.0,
            feast_fallback_enabled=False,
        )

        assert config.enable_feast is True
        assert len(config.feast_feature_refs) == 2
        assert config.feast_max_staleness_hours == 12.0
        assert config.feast_fallback_enabled is False

    def test_feast_disabled_config(self):
        """Test config with Feast disabled."""
        config = PipelineConfig(enable_feast=False)

        assert config.enable_feast is False


class TestPipelineResultFeast:
    """Test Feast-related pipeline result fields."""

    def test_result_feast_fields(self):
        """Test that PipelineResult has Feast fields."""
        result = PipelineResult(
            pipeline_run_id="test_123",
            status="completed",
            current_stage="completed",
            feast_enabled=True,
            feature_refs_used=["hcp_conversion_features:engagement_score"],
            feature_freshness={"fresh": True, "stale_features": []},
        )

        assert result.feast_enabled is True
        assert len(result.feature_refs_used) == 1
        assert result.feature_freshness["fresh"] is True

    def test_result_feast_defaults(self):
        """Test default values for Feast fields."""
        result = PipelineResult(
            pipeline_run_id="test_123",
            status="in_progress",
            current_stage="scope_definition",
        )

        assert result.feast_enabled is False
        assert result.feature_refs_used is None
        assert result.feature_freshness is None


class TestGetFeastAdapter:
    """Test Feast adapter lazy initialization."""

    @pytest.mark.asyncio
    async def test_get_feast_adapter_disabled(self):
        """Test that adapter is None when Feast disabled."""
        config = PipelineConfig(enable_feast=False)
        pipeline = MLFoundationPipeline(config=config)

        adapter = await pipeline._get_feast_adapter()

        assert adapter is None

    @pytest.mark.asyncio
    async def test_get_feast_adapter_initialization_success(self):
        """Test successful Feast adapter initialization."""
        config = PipelineConfig(enable_feast=True)
        pipeline = MLFoundationPipeline(config=config)

        # Patch where the imports actually come from
        with patch(
            "src.feature_store.client.FeatureStoreClient"
        ) as mock_fs_client, patch(
            "src.feature_store.feature_analyzer_adapter.get_feature_analyzer_adapter"
        ) as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_get_adapter.return_value = mock_adapter
            mock_fs_client.return_value = MagicMock()

            adapter = await pipeline._get_feast_adapter()

            assert adapter is mock_adapter
            assert pipeline._feast_initialized is True
            mock_get_adapter.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_feast_adapter_initialization_failure(self):
        """Test graceful handling of Feast initialization failure."""
        config = PipelineConfig(enable_feast=True)
        pipeline = MLFoundationPipeline(config=config)

        # Patch where the import actually comes from
        with patch(
            "src.feature_store.client.FeatureStoreClient",
            side_effect=Exception("Connection failed"),
        ):
            adapter = await pipeline._get_feast_adapter()

            assert adapter is None
            assert pipeline._feast_initialized is True  # Don't retry

    @pytest.mark.asyncio
    async def test_get_feast_adapter_cached(self):
        """Test that adapter is cached after first call."""
        config = PipelineConfig(enable_feast=True)
        pipeline = MLFoundationPipeline(config=config)

        mock_adapter = MagicMock()
        pipeline._feast_adapter = mock_adapter
        pipeline._feast_initialized = True

        adapter = await pipeline._get_feast_adapter()

        assert adapter is mock_adapter


class TestCheckFeatureFreshness:
    """Test feature freshness checking."""

    @pytest.mark.asyncio
    async def test_check_freshness_no_refs(self):
        """Test freshness check with no feature refs."""
        config = PipelineConfig(enable_feast=True)
        pipeline = MLFoundationPipeline(config=config)

        result = await pipeline._check_feature_freshness(feature_refs=None)

        assert result["fresh"] is True
        assert "No feature refs configured" in result["recommendations"][0]

    @pytest.mark.asyncio
    async def test_check_freshness_no_adapter(self):
        """Test freshness check when adapter not available."""
        config = PipelineConfig(enable_feast=True)
        pipeline = MLFoundationPipeline(config=config)
        pipeline._feast_initialized = True
        pipeline._feast_adapter = None

        result = await pipeline._check_feature_freshness(
            feature_refs=["hcp_conversion_features:engagement_score"]
        )

        assert result["fresh"] is True
        assert "Feast adapter not available" in result["recommendations"][0]

    @pytest.mark.asyncio
    async def test_check_freshness_all_fresh(self):
        """Test freshness check when all features are fresh."""
        config = PipelineConfig(
            enable_feast=True,
            feast_max_staleness_hours=24.0,
        )
        pipeline = MLFoundationPipeline(config=config)

        mock_adapter = AsyncMock()
        mock_adapter.check_feature_freshness = AsyncMock(
            return_value={
                "fresh": True,
                "stale_features": [],
                "recommendations": [],
            }
        )
        pipeline._feast_adapter = mock_adapter
        pipeline._feast_initialized = True

        result = await pipeline._check_feature_freshness(
            feature_refs=["hcp_conversion_features:engagement_score"]
        )

        assert result["fresh"] is True
        assert len(result["stale_features"]) == 0
        mock_adapter.check_feature_freshness.assert_called_once_with(
            feature_refs=["hcp_conversion_features:engagement_score"],
            max_staleness_hours=24.0,
        )

    @pytest.mark.asyncio
    async def test_check_freshness_stale_features(self):
        """Test freshness check when features are stale."""
        config = PipelineConfig(enable_feast=True)
        pipeline = MLFoundationPipeline(config=config)

        mock_adapter = AsyncMock()
        mock_adapter.check_feature_freshness = AsyncMock(
            return_value={
                "fresh": False,
                "stale_features": ["hcp_conversion_features:engagement_score"],
                "recommendations": ["Refresh engagement_score feature"],
            }
        )
        pipeline._feast_adapter = mock_adapter
        pipeline._feast_initialized = True

        result = await pipeline._check_feature_freshness(
            feature_refs=["hcp_conversion_features:engagement_score"]
        )

        assert result["fresh"] is False
        assert len(result["stale_features"]) == 1

    @pytest.mark.asyncio
    async def test_check_freshness_error_handling(self):
        """Test graceful handling of freshness check errors."""
        config = PipelineConfig(enable_feast=True)
        pipeline = MLFoundationPipeline(config=config)

        mock_adapter = AsyncMock()
        mock_adapter.check_feature_freshness = AsyncMock(
            side_effect=Exception("Network error")
        )
        pipeline._feast_adapter = mock_adapter
        pipeline._feast_initialized = True

        result = await pipeline._check_feature_freshness(
            feature_refs=["hcp_conversion_features:engagement_score"]
        )

        # Should not block on errors
        assert result["fresh"] is True
        assert "error" in result


class TestPipelineRunWithFeast:
    """Test pipeline run with Feast integration."""

    @pytest.fixture
    def sample_input(self):
        """Sample input data for pipeline."""
        return {
            "problem_description": "Predict HCP conversion",
            "business_objective": "Increase market share",
            "target_outcome": "conversion",
            "data_source": "business_metrics",
            "feature_refs": ["hcp_conversion_features:engagement_score"],
        }

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for pipeline stages."""
        # Mock scope definer
        scope_output = {
            "experiment_id": "exp_001",
            "scope_spec": {
                "problem_type": "binary_classification",
                "experiment_id": "exp_001",
            },
            "success_criteria": {"min_auc": 0.75},
        }

        # Mock data preparer
        data_output = {
            "gate_passed": True,
            "qc_report": {"overall_score": 0.9, "qc_passed": True},
            "baseline_metrics": {},
        }

        # Mock model selector
        selector_output = {
            "model_candidate": {"algorithm_name": "lightgbm", "selection_score": 0.85},
        }

        # Mock model trainer
        trainer_output = {
            "training_run_id": "run_001",
            "success_criteria_met": True,
            "test_metrics": {"auc_roc": 0.82},
        }

        return {
            "scope_definer": AsyncMock(return_value=scope_output),
            "data_preparer": AsyncMock(return_value=data_output),
            "model_selector": AsyncMock(return_value=selector_output),
            "model_trainer": AsyncMock(return_value=trainer_output),
        }

    @pytest.mark.asyncio
    async def test_pipeline_initializes_feast_fields(self, sample_input, mock_agents):
        """Test that pipeline result has Feast fields initialized."""
        config = PipelineConfig(
            enable_feast=True,
            feast_feature_refs=["hcp_conversion_features:engagement_score"],
            skip_deployment=True,
            skip_feature_analysis=True,
        )
        pipeline = MLFoundationPipeline(config=config)

        # Mock all agents
        with patch.object(
            pipeline, "_get_agent", side_effect=lambda name: MagicMock(
                run=mock_agents.get(name, AsyncMock(return_value={}))
            )
        ), patch.object(
            pipeline, "_get_feast_adapter", new_callable=AsyncMock
        ) as mock_get_adapter:
            mock_adapter = AsyncMock()
            mock_adapter.check_feature_freshness = AsyncMock(
                return_value={"fresh": True, "stale_features": [], "recommendations": []}
            )
            mock_get_adapter.return_value = mock_adapter

            result = await pipeline.run(sample_input)

            assert result.feast_enabled is True
            # feature_refs comes from input, not config when provided
            assert result.feature_refs_used == ["hcp_conversion_features:engagement_score"]

    @pytest.mark.asyncio
    async def test_pipeline_feature_freshness_in_qc(self, sample_input, mock_agents):
        """Test that feature freshness is checked during QC gate."""
        config = PipelineConfig(
            enable_feast=True,
            feast_freshness_check=True,
            skip_deployment=True,
            skip_feature_analysis=True,
        )
        pipeline = MLFoundationPipeline(config=config)

        with patch.object(
            pipeline, "_get_agent", side_effect=lambda name: MagicMock(
                run=mock_agents.get(name, AsyncMock(return_value={}))
            )
        ), patch.object(
            pipeline, "_check_feature_freshness", new_callable=AsyncMock
        ) as mock_freshness:
            mock_freshness.return_value = {
                "fresh": True,
                "stale_features": [],
                "recommendations": [],
            }

            result = await pipeline.run(sample_input)

            # Freshness should be checked during data preparation
            mock_freshness.assert_called_once()
            assert result.feature_freshness is not None
            assert result.feature_freshness["fresh"] is True

    @pytest.mark.asyncio
    async def test_pipeline_stale_features_warning(self, sample_input, mock_agents):
        """Test that stale features add warning but don't block."""
        config = PipelineConfig(
            enable_feast=True,
            feast_freshness_check=True,
            skip_deployment=True,
            skip_feature_analysis=True,
        )
        pipeline = MLFoundationPipeline(config=config)

        with patch.object(
            pipeline, "_get_agent", side_effect=lambda name: MagicMock(
                run=mock_agents.get(name, AsyncMock(return_value={}))
            )
        ), patch.object(
            pipeline, "_check_feature_freshness", new_callable=AsyncMock
        ) as mock_freshness:
            mock_freshness.return_value = {
                "fresh": False,
                "stale_features": ["hcp_conversion_features:engagement_score"],
                "recommendations": ["Refresh features"],
            }

            result = await pipeline.run(sample_input)

            # Pipeline should complete (stale features are non-blocking)
            assert result.status == "completed"
            # But should have warning
            assert len(result.warnings) > 0
            assert any("Stale features" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_pipeline_feast_disabled_skips_freshness(self, sample_input, mock_agents):
        """Test that freshness check is skipped when Feast disabled."""
        config = PipelineConfig(
            enable_feast=False,
            skip_deployment=True,
            skip_feature_analysis=True,
        )
        pipeline = MLFoundationPipeline(config=config)

        with patch.object(
            pipeline, "_get_agent", side_effect=lambda name: MagicMock(
                run=mock_agents.get(name, AsyncMock(return_value={}))
            )
        ), patch.object(
            pipeline, "_check_feature_freshness", new_callable=AsyncMock
        ) as mock_freshness:
            result = await pipeline.run(sample_input)

            # Freshness should NOT be checked
            mock_freshness.assert_not_called()
            assert result.feature_freshness is None
            assert result.feast_enabled is False


class TestTrainerReceivesFeatureRefs:
    """Test that model trainer receives feature refs."""

    @pytest.mark.asyncio
    async def test_trainer_input_includes_feature_refs(self):
        """Test that trainer input includes feature refs."""
        config = PipelineConfig(
            enable_feast=True,
            feast_feature_refs=["hcp_conversion_features:engagement_score"],
            skip_deployment=True,
            skip_feature_analysis=True,
        )
        pipeline = MLFoundationPipeline(config=config)

        # Mock scope result
        pipeline_result = MagicMock()
        pipeline_result.model_candidate = {"algorithm_name": "lightgbm"}
        pipeline_result.qc_report = {"overall_score": 0.9}
        pipeline_result.experiment_id = "exp_001"
        pipeline_result.success_criteria = {"min_auc": 0.75}
        pipeline_result.scope_spec = {"problem_type": "binary_classification"}
        pipeline_result.feature_refs_used = ["hcp_conversion_features:engagement_score"]
        pipeline_result.feast_enabled = True

        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer.run = AsyncMock(
            return_value={
                "training_run_id": "run_001",
                "success_criteria_met": True,
                "test_metrics": {"auc_roc": 0.82},
            }
        )

        with patch.object(
            pipeline, "_get_agent", return_value=mock_trainer
        ):
            await pipeline._run_model_training(
                input_data={},
                result=pipeline_result,
                obs_context=None,
            )

            # Verify trainer was called with feature refs
            call_args = mock_trainer.run.call_args[0][0]
            assert "feature_refs" in call_args
            assert call_args["feature_refs"] == ["hcp_conversion_features:engagement_score"]
            assert call_args["feast_enabled"] is True
