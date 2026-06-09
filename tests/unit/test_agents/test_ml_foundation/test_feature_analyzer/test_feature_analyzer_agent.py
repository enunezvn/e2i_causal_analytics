"""Integration tests for feature_analyzer agent."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from src.agents.ml_foundation.feature_analyzer.agent import FeatureAnalyzerAgent


@pytest.mark.asyncio
class TestFeatureAnalyzerAgent:
    """Integration tests for complete feature analysis workflow."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return FeatureAnalyzerAgent()

    @pytest.fixture
    def mock_random_forest_model(self):
        """Create mock RandomForest model."""
        model = Mock()
        model.__class__.__name__ = "RandomForestClassifier"
        model.feature_names_in_ = ["feat_1", "feat_2", "feat_3", "feat_4", "feat_5"]
        model.n_features_in_ = 5
        model.predict = Mock(return_value=np.random.randint(0, 2, 100))
        model.predict_proba = Mock(return_value=np.random.rand(100, 2))
        return model

    @pytest.fixture
    def valid_input_data(self, mock_random_forest_model):
        """Create valid input data for analysis.

        Includes ``trained_model`` so the C1 in-memory passthrough wires the
        model into ``state["loaded_model"]`` and the SHAP node skips the
        MLflow loader entirely. Tests that need to exercise the URI loader
        path should construct their own input dict instead.
        """
        return {
            "model_uri": "runs:/abc123/model",
            "experiment_id": "exp_test_001",
            "training_run_id": "run_001",
            "max_samples": 100,
            "compute_interactions": True,
            "store_in_semantic_memory": True,
            "trained_model": mock_random_forest_model,
            # F8: provide real caller-supplied sample data so SHAP runs on a real
            # (not synthetic) background — matches the (100, 5) shape the mock
            # explainer returns. Without real data the node now fails closed (skips).
            "X_sample": np.random.rand(100, 5),
        }

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.importance_narrator.Anthropic")
    async def test_complete_analysis_workflow(
        self,
        mock_anthropic_class,
        mock_shap,
        mock_mlflow,
        agent,
        valid_input_data,
        mock_random_forest_model,
    ):
        """Should complete full analysis workflow successfully."""
        # Mock MLflow
        mock_mlflow.sklearn.load_model.return_value = mock_random_forest_model
        mock_mlflow.get_run.return_value = Mock(
            info=Mock(run_id="abc123"), data=Mock(params={"model_version": "v1"})
        )

        # Mock SHAP
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(100, 5)
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        # Mock Anthropic
        mock_response = Mock()
        mock_response.content = [
            Mock(
                text='{"executive_summary": "Test summary", "feature_explanations": {"feat_1": "Explanation"}, "key_insights": ["Insight"], "recommendations": ["Recommendation"], "cautions": ["Caution"]}'
            )
        ]
        mock_response.usage = Mock(input_tokens=400, output_tokens=200)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        # Execute
        result = await agent.run(valid_input_data)

        # Assert
        assert result["status"] == "completed"
        assert "shap_analysis" in result
        assert "interpretation" in result
        assert "top_features" in result
        assert "top_interactions" in result
        assert result["experiment_id"] == "exp_test_001"

    async def test_validates_required_fields(self, agent):
        """Should validate required input fields."""
        with pytest.raises(ValueError, match="Missing required field"):
            await agent.run({})

    async def test_runs_without_model_uri(self, agent):
        """model_uri is optional — SHAP analysis is skipped when not provided."""
        result = await agent.run({"experiment_id": "exp_002"})

        # Agent should succeed but skip SHAP
        assert result is not None
        assert result.get("experiment_id") == "exp_002"

    async def test_validates_experiment_id(self, agent):
        """Should validate experiment_id is provided."""
        with pytest.raises(ValueError, match="experiment_id"):
            await agent.run({"model_uri": "runs:/abc/model"})

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.importance_narrator.Anthropic")
    async def test_returns_shap_analysis(
        self,
        mock_anthropic_class,
        mock_shap,
        mock_mlflow,
        agent,
        valid_input_data,
        mock_random_forest_model,
    ):
        """Should return SHAP analysis structure."""
        # Mock setup
        mock_mlflow.sklearn.load_model.return_value = mock_random_forest_model
        mock_mlflow.get_run.return_value = Mock(
            info=Mock(run_id="abc123"), data=Mock(params={"model_version": "v1"})
        )

        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(100, 5)
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        mock_response = Mock()
        mock_response.content = [
            Mock(
                text='{"executive_summary": "Summary", "feature_explanations": {}, "key_insights": [], "recommendations": [], "cautions": []}'
            )
        ]
        mock_response.usage = Mock(input_tokens=300, output_tokens=150)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        # Execute
        result = await agent.run(valid_input_data)

        # Assert SHAP analysis structure
        shap_analysis = result["shap_analysis"]
        assert "experiment_id" in shap_analysis
        assert "model_version" in shap_analysis
        assert "shap_analysis_id" in shap_analysis
        assert "feature_importance" in shap_analysis
        assert "interactions" in shap_analysis
        assert "samples_analyzed" in shap_analysis
        assert "computation_time_seconds" in shap_analysis

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.importance_narrator.Anthropic")
    async def test_returns_feature_importance_list(
        self,
        mock_anthropic_class,
        mock_shap,
        mock_mlflow,
        agent,
        valid_input_data,
        mock_random_forest_model,
    ):
        """Should return feature importance as structured list."""
        # Mock setup
        mock_mlflow.sklearn.load_model.return_value = mock_random_forest_model
        mock_mlflow.get_run.return_value = Mock(info=Mock(run_id="abc123"), data=Mock(params={}))

        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(100, 5)
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        mock_response = Mock()
        mock_response.content = [
            Mock(
                text='{"executive_summary": "Summary", "feature_explanations": {}, "key_insights": [], "recommendations": [], "cautions": []}'
            )
        ]
        mock_response.usage = Mock(input_tokens=300, output_tokens=150)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        # Execute
        result = await agent.run(valid_input_data)

        # Assert feature importance
        feature_importance = result["feature_importance"]
        assert len(feature_importance) == 5  # 5 features

        # Check structure
        for fi in feature_importance:
            assert "feature" in fi
            assert "importance" in fi
            assert "rank" in fi

        # Check ranking
        ranks = [fi["rank"] for fi in feature_importance]
        assert ranks == sorted(ranks)  # Should be ranked

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.importance_narrator.Anthropic")
    async def test_returns_interpretation(
        self,
        mock_anthropic_class,
        mock_shap,
        mock_mlflow,
        agent,
        valid_input_data,
        mock_random_forest_model,
    ):
        """Should return natural language interpretation."""
        # Mock setup
        mock_mlflow.sklearn.load_model.return_value = mock_random_forest_model
        mock_mlflow.get_run.return_value = Mock(info=Mock(run_id="abc123"), data=Mock(params={}))

        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(100, 5)
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        mock_response = Mock()
        mock_response.content = [
            Mock(
                text='{"executive_summary": "Model prioritizes engagement", "feature_explanations": {}, "key_insights": ["Engagement dominates"], "recommendations": ["Target high engagement"], "cautions": ["Watch confounders"]}'
            )
        ]
        mock_response.usage = Mock(input_tokens=400, output_tokens=250)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        # Execute
        result = await agent.run(valid_input_data)

        # Assert interpretation
        assert "interpretation" in result
        assert "executive_summary" in result
        assert "key_insights" in result
        assert "recommendations" in result
        assert "cautions" in result

        # Check content
        assert "Model prioritizes engagement" in result["executive_summary"]
        assert len(result["key_insights"]) > 0
        assert len(result["recommendations"]) > 0

    async def test_agent_properties(self, agent):
        """Should have correct agent properties."""
        assert agent.tier == 0
        assert agent.tier_name == "ml_foundation"
        assert agent.agent_type == "hybrid"
        assert agent.sla_seconds == 120

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.importance_narrator.Anthropic")
    async def test_respects_max_samples(
        self,
        mock_anthropic_class,
        mock_shap,
        mock_mlflow,
        agent,
        valid_input_data,
        mock_random_forest_model,
    ):
        """Should respect max_samples configuration."""
        # Set max_samples to 50
        valid_input_data["max_samples"] = 50

        # Mock setup
        mock_mlflow.sklearn.load_model.return_value = mock_random_forest_model
        mock_mlflow.get_run.return_value = Mock(info=Mock(run_id="abc123"), data=Mock(params={}))

        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(50, 5)
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        mock_response = Mock()
        mock_response.content = [
            Mock(
                text='{"executive_summary": "Summary", "feature_explanations": {}, "key_insights": [], "recommendations": [], "cautions": []}'
            )
        ]
        mock_response.usage = Mock(input_tokens=300, output_tokens=150)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        # Execute
        result = await agent.run(valid_input_data)

        # Assert
        assert result["samples_analyzed"] <= 50

    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.mlflow")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.shap_computer.shap")
    @patch("src.agents.ml_foundation.feature_analyzer.nodes.importance_narrator.Anthropic")
    async def test_skips_interactions_when_disabled(
        self,
        mock_anthropic_class,
        mock_shap,
        mock_mlflow,
        agent,
        valid_input_data,
        mock_random_forest_model,
    ):
        """Should skip interaction detection when compute_interactions is False."""
        # Disable interactions
        valid_input_data["compute_interactions"] = False

        # Mock setup
        mock_mlflow.sklearn.load_model.return_value = mock_random_forest_model
        mock_mlflow.get_run.return_value = Mock(info=Mock(run_id="abc123"), data=Mock(params={}))

        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.rand(100, 5)
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer

        mock_response = Mock()
        mock_response.content = [
            Mock(
                text='{"executive_summary": "Summary", "feature_explanations": {}, "key_insights": [], "recommendations": [], "cautions": []}'
            )
        ]
        mock_response.usage = Mock(input_tokens=300, output_tokens=150)

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        # Execute
        result = await agent.run(valid_input_data)

        # Assert
        assert result["top_interactions"] == []


# ============================================================================
# Block 5 (#14): auto-register surviving features in Feast
# ============================================================================


@pytest.mark.asyncio
class TestAutoRegisterInFeast:
    """The agent calls FeastClient.register_feature_view for surviving
    features when ``feast_registration_config`` is supplied. Best-effort:
    failures must NOT raise; they're captured in result["feast_registration"].
    """

    async def test_skipped_when_no_config(self):
        """Without feast_registration_config, the call is skipped — the
        existing tier0 path keeps working unchanged."""
        agent = FeatureAnalyzerAgent()
        state = {
            "experiment_id": "exp_001",
            "selected_features": ["f1", "f2"],
        }
        with patch.object(agent, "_get_full_graph"), patch.object(agent, "_get_shap_graph"):
            result = await agent._auto_register_in_feast(state, input_data={})
        assert result["registered"] is False
        assert result["skipped_reason"] == "feast_registration_config not provided"

    async def test_skipped_when_no_surviving_features(self):
        """When the selection step removed everything, registration is skipped."""
        agent = FeatureAnalyzerAgent()
        state = {
            "experiment_id": "exp_002",
            "selected_features": [],
            "selected_features_all": [],
        }
        input_data = {
            "feast_registration_config": {
                "entity_name": "patient",
                "batch_source_name": "patient_journeys_source",
            }
        }
        result = await agent._auto_register_in_feast(state, input_data=input_data)
        assert result["registered"] is False
        assert result["skipped_reason"] == "no surviving features in state"

    async def test_calls_register_feature_view_with_surviving_features(self):
        """Survivors round-trip into FeastClient.register_feature_view with
        the experiment-scoped name + the configured entity / batch source."""
        agent = FeatureAnalyzerAgent()
        state = {
            "experiment_id": "exp_003",
            "selected_features": ["lag_1_trx", "rolling_mean_engagement"],
        }
        input_data = {
            "feast_registration_config": {
                "entity_name": "hcp",
                "batch_source_name": "business_metrics_source",
            }
        }

        fake_client = MagicMock()
        fake_client.register_feature_view = AsyncMock(
            return_value={
                "registered": True,
                "feature_view_name": "tier0_exp_003_features",
                "features_count": 2,
                "error": None,
            }
        )

        with patch(
            "src.agents.ml_foundation.feature_analyzer.agent._get_feast_client",
            new=AsyncMock(return_value=fake_client),
        ):
            result = await agent._auto_register_in_feast(state, input_data=input_data)

        # The shape of the call into FeastClient.
        fake_client.register_feature_view.assert_awaited_once()
        call_kwargs = fake_client.register_feature_view.await_args.kwargs
        assert call_kwargs["name"] == "tier0_exp_003_features"
        assert call_kwargs["entity_name"] == "hcp"
        assert call_kwargs["batch_source_name"] == "business_metrics_source"
        assert call_kwargs["feature_names"] == [
            "lag_1_trx",
            "rolling_mean_engagement",
        ]
        # Round-trip metadata is propagated back to the agent caller.
        assert result["registered"] is True
        assert result["feature_view_name"] == "tier0_exp_003_features"
        assert result["features_count"] == 2

    async def test_failure_is_swallowed_not_raised(self):
        """Any registration error is captured in the result dict, not raised."""
        agent = FeatureAnalyzerAgent()
        state = {
            "experiment_id": "exp_004",
            "selected_features": ["f1"],
        }
        input_data = {
            "feast_registration_config": {
                "entity_name": "hcp",
                "batch_source_name": "business_metrics_source",
            }
        }

        fake_client = MagicMock()
        fake_client.register_feature_view = AsyncMock(side_effect=RuntimeError("Feast unreachable"))
        with patch(
            "src.agents.ml_foundation.feature_analyzer.agent._get_feast_client",
            new=AsyncMock(return_value=fake_client),
        ):
            result = await agent._auto_register_in_feast(state, input_data=input_data)

        # Helper must swallow the exception so tier0 keeps running.
        assert result["registered"] is False
        assert "Feast unreachable" in (result["error"] or "")

    async def test_falls_back_to_selected_features_all(self):
        """When ``selected_features`` is missing, the helper uses
        ``selected_features_all`` (covers the SHAP-only / interpretation
        paths where numeric-only selection isn't run)."""
        agent = FeatureAnalyzerAgent()
        state = {
            "experiment_id": "exp_005",
            "selected_features_all": ["f_categorical", "f_numeric"],
        }
        input_data = {
            "feast_registration_config": {
                "entity_name": "hcp",
                "batch_source_name": "business_metrics_source",
            }
        }
        fake_client = MagicMock()
        fake_client.register_feature_view = AsyncMock(
            return_value={
                "registered": True,
                "feature_view_name": "tier0_exp_005_features",
                "features_count": 2,
                "error": None,
            }
        )
        with patch(
            "src.agents.ml_foundation.feature_analyzer.agent._get_feast_client",
            new=AsyncMock(return_value=fake_client),
        ):
            result = await agent._auto_register_in_feast(state, input_data=input_data)

        fake_client.register_feature_view.assert_awaited_once()
        assert fake_client.register_feature_view.await_args.kwargs["feature_names"] == [
            "f_categorical",
            "f_numeric",
        ]
        assert result["registered"] is True


@pytest.mark.asyncio
class TestFeastClientRegisterFeatureView:
    """The FeastClient.register_feature_view helper itself: gracefully
    declines when the store isn't initialised, and returns clear errors
    when entity / batch source aren't registered."""

    async def test_returns_error_when_store_uninitialised(self):
        from src.feature_store.feast_client import FeastClient

        client = FeastClient()
        # Force the no-store path without invoking real Feast init.
        client._initialized = True
        client._store = None
        result = await client.register_feature_view(
            name="tier0_test_features",
            entity_name="patient",
            feature_names=["f1"],
            batch_source_name="source_x",
        )
        assert result["registered"] is False
        assert "not initialised" in (result["error"] or "")

    async def test_returns_error_for_empty_feature_list(self):
        from src.feature_store.feast_client import FeastClient

        client = FeastClient()
        client._initialized = True
        client._store = MagicMock()
        result = await client.register_feature_view(
            name="tier0_test_features",
            entity_name="patient",
            feature_names=[],
            batch_source_name="source_x",
        )
        assert result["registered"] is False
        assert result["error"] == "feature_names is empty; nothing to register"
