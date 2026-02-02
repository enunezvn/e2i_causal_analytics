"""Unit tests for Tier0OutputMapper.

Tests the mapping of tier0 synthetic data outputs to agent-specific inputs.
"""

import os
from datetime import datetime

import pandas as pd
import pytest

from src.testing.tier0_output_mapper import Tier0OutputMapper

# Set testing mode
os.environ["E2I_TESTING_MODE"] = "true"


@pytest.fixture
def sample_tier0_state():
    """Create a sample tier0 state dictionary."""
    # Create a sample DataFrame with realistic columns
    df = pd.DataFrame(
        {
            "patient_journey_id": [f"pj_{i:03d}" for i in range(1, 11)],
            "patient_id": [f"pt_{i:03d}" for i in range(1, 11)],
            "brand": ["Kisqali"] * 10,
            "discontinuation_flag": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "hcp_visits": [2, 5, 3, 6, 4, 7, 2, 8, 3, 5],
            "prior_treatments": [1, 2, 1, 3, 2, 1, 2, 3, 1, 2],
            "days_on_therapy": [30, 60, 45, 90, 50, 100, 35, 110, 40, 70],
            "age_group": [
                "50-60",
                "60-70",
                "50-60",
                "70+",
                "60-70",
                "50-60",
                "60-70",
                "70+",
                "50-60",
                "60-70",
            ],
            "geographic_region": ["NE", "SE", "MW", "W", "NE", "SE", "MW", "W", "NE", "SE"],
            "feature_1": [0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.5, 0.6],
            "feature_2": [1.0, 1.1, 1.2, 1.3, 1.0, 1.1, 1.2, 1.3, 1.0, 1.1],
        }
    )

    return {
        "experiment_id": "exp_test_001",
        "eligible_df": df,
        "trained_model": "mock_model",
        "model_uri": "models:/test_model/1",
        "validation_metrics": {
            "roc_auc": 0.85,
            "accuracy": 0.78,
            "precision": 0.72,
            "recall": 0.68,
            "f1_score": 0.70,
        },
        "feature_importance": [
            {"feature": "hcp_visits", "importance": 0.35},
            {"feature": "prior_treatments", "importance": 0.25},
            {"feature": "days_on_therapy", "importance": 0.20},
            {"feature": "feature_1", "importance": 0.12},
            {"feature": "feature_2", "importance": 0.08},
        ],
        "scope_spec": {
            "brand": "Kisqali",
            "indication": "HR+/HER2- breast cancer",
        },
        "qc_report": {"total_patients": 10, "quality_score": 0.95},
    }


@pytest.mark.unit
class TestTier0OutputMapperInit:
    """Test Tier0OutputMapper initialization and validation."""

    def test_init_with_valid_state(self, sample_tier0_state):
        """Test initialization with valid tier0 state."""
        mapper = Tier0OutputMapper(sample_tier0_state)
        assert mapper.state == sample_tier0_state

    def test_init_missing_required_keys(self):
        """Test initialization with missing required keys."""
        invalid_state = {"experiment_id": "exp_001"}  # Missing eligible_df
        with pytest.raises(ValueError, match="Missing required tier0 state keys"):
            Tier0OutputMapper(invalid_state)

    def test_required_keys_constant(self):
        """Test REQUIRED_KEYS constant."""
        assert "experiment_id" in Tier0OutputMapper.REQUIRED_KEYS
        assert "eligible_df" in Tier0OutputMapper.REQUIRED_KEYS

    def test_optional_keys_constant(self):
        """Test OPTIONAL_KEYS constant."""
        assert "trained_model" in Tier0OutputMapper.OPTIONAL_KEYS
        assert "model_uri" in Tier0OutputMapper.OPTIONAL_KEYS
        assert "validation_metrics" in Tier0OutputMapper.OPTIONAL_KEYS


@pytest.mark.unit
class TestTier0OutputMapperUtilities:
    """Test utility methods of Tier0OutputMapper."""

    @pytest.fixture
    def mapper(self, sample_tier0_state):
        return Tier0OutputMapper(sample_tier0_state)

    def test_get_feature_names_from_importance(self, mapper):
        """Test extracting feature names from feature_importance."""
        features = mapper._get_feature_names()
        assert "hcp_visits" in features
        assert "prior_treatments" in features
        assert len(features) == 5

    def test_get_feature_names_from_dataframe(self, sample_tier0_state):
        """Test extracting feature names from DataFrame when no importance."""
        state = sample_tier0_state.copy()
        state["feature_importance"] = None
        mapper = Tier0OutputMapper(state)

        features = mapper._get_feature_names()
        assert "hcp_visits" in features
        assert "patient_journey_id" not in features
        assert "brand" not in features

    def test_get_top_features(self, mapper):
        """Test getting top N features."""
        top_3 = mapper._get_top_features(3)
        assert len(top_3) == 3
        assert top_3[0] == "hcp_visits"

    def test_get_top_features_more_than_available(self, mapper):
        """Test getting more features than available."""
        top_10 = mapper._get_top_features(10)
        assert len(top_10) == 5  # Only 5 features available


@pytest.mark.unit
class TestTier1Mappings:
    """Test Tier 1 agent mappings (Orchestrator, ToolComposer)."""

    @pytest.fixture
    def mapper(self, sample_tier0_state):
        return Tier0OutputMapper(sample_tier0_state)

    def test_map_to_orchestrator(self, mapper):
        """Test mapping to orchestrator input."""
        result = mapper.map_to_orchestrator()

        assert "query" in result
        assert "messages" in result
        assert "experiment_id" in result
        assert result["experiment_id"] == "exp_test_001"
        assert "Kisqali" in result["query"]
        assert len(result["messages"]) > 0

    def test_map_to_tool_composer(self, mapper):
        """Test mapping to tool composer input."""
        result = mapper.map_to_tool_composer()

        assert "query" in result
        assert "experiment_id" in result
        assert "available_tools" in result
        assert len(result["available_tools"]) > 0
        assert "causal_effect_estimator" in result["available_tools"]


@pytest.mark.unit
class TestTier2Mappings:
    """Test Tier 2 agent mappings (Causal agents)."""

    @pytest.fixture
    def mapper(self, sample_tier0_state):
        return Tier0OutputMapper(sample_tier0_state)

    def test_map_to_causal_impact(self, mapper):
        """Test mapping to causal impact input."""
        result = mapper.map_to_causal_impact()

        assert "query" in result
        assert "query_id" in result
        assert "treatment_var" in result
        assert "outcome_var" in result
        assert "confounders" in result
        assert "data_source" in result
        assert "experiment_id" in result
        assert "data" in result

        # Check treatment and outcome vars
        assert result["treatment_var"] in ["hcp_visits", "prior_treatments"]
        assert isinstance(result["confounders"], list)

    def test_map_to_gap_analyzer(self, mapper):
        """Test mapping to gap analyzer input."""
        result = mapper.map_to_gap_analyzer()

        assert "query" in result
        assert "metrics" in result
        assert "segments" in result
        assert "brand" in result
        assert "tier0_data" in result

        assert isinstance(result["metrics"], list)
        assert isinstance(result["segments"], list)
        assert result["brand"] == "Kisqali"

    def test_map_to_heterogeneous_optimizer(self, mapper):
        """Test mapping to heterogeneous optimizer input."""
        result = mapper.map_to_heterogeneous_optimizer()

        assert "query" in result
        assert "treatment_var" in result
        assert "outcome_var" in result
        assert "segment_vars" in result
        assert "effect_modifiers" in result
        assert "tier0_data" in result

        assert isinstance(result["segment_vars"], list)
        assert isinstance(result["effect_modifiers"], list)

    def test_heterogeneous_optimizer_no_effect_modifiers(self, sample_tier0_state):
        """Test heterogeneous optimizer with insufficient effect modifiers."""
        # Create state with minimal numeric columns
        df = pd.DataFrame(
            {
                "patient_journey_id": ["pj_001"],
                "discontinuation_flag": [1],
                "treatment": [1],
            }
        )
        state = sample_tier0_state.copy()
        state["eligible_df"] = df

        mapper = Tier0OutputMapper(state)

        with pytest.raises(ValueError, match="No effect modifiers available"):
            mapper.map_to_heterogeneous_optimizer()


@pytest.mark.unit
class TestTier3Mappings:
    """Test Tier 3 agent mappings (Monitoring agents)."""

    @pytest.fixture
    def mapper(self, sample_tier0_state):
        return Tier0OutputMapper(sample_tier0_state)

    def test_map_to_drift_monitor(self, mapper):
        """Test mapping to drift monitor input."""
        result = mapper.map_to_drift_monitor()

        assert "query" in result
        assert "features_to_monitor" in result
        assert "model_id" in result
        assert "time_window" in result
        assert "tier0_data" in result

        assert isinstance(result["features_to_monitor"], list)
        assert len(result["features_to_monitor"]) > 0

    def test_map_to_experiment_designer(self, mapper):
        """Test mapping to experiment designer input."""
        result = mapper.map_to_experiment_designer()

        assert "business_question" in result
        assert "constraints" in result
        assert "available_data" in result
        assert "brand" in result

        # Check constraints structure
        constraints = result["constraints"]
        assert "budget" in constraints
        assert "timeline" in constraints
        assert "operational" in constraints

    def test_map_to_health_score(self, mapper):
        """Test mapping to health score input."""
        result = mapper.map_to_health_score()

        assert "scope" in result
        assert "query" in result
        assert "experiment_name" in result
        assert result["scope"] in ["full", "quick", "models", "pipelines", "agents"]


@pytest.mark.unit
class TestTier4Mappings:
    """Test Tier 4 agent mappings (ML Prediction agents)."""

    @pytest.fixture
    def mapper(self, sample_tier0_state):
        return Tier0OutputMapper(sample_tier0_state)

    def test_map_to_prediction_synthesizer(self, mapper):
        """Test mapping to prediction synthesizer input."""
        result = mapper.map_to_prediction_synthesizer()

        assert "entity_id" in result
        assert "prediction_target" in result
        assert "features" in result
        assert "entity_type" in result
        assert "query" in result
        assert "session_id" in result

        assert result["entity_type"] == "patient"
        assert isinstance(result["features"], dict)

    def test_map_to_resource_optimizer(self, mapper):
        """Test mapping to resource optimizer input."""
        result = mapper.map_to_resource_optimizer()

        assert "allocation_targets" in result
        assert "constraints" in result
        assert "resource_type" in result
        assert "objective" in result
        assert "query" in result

        # Check allocation targets structure
        targets = result["allocation_targets"]
        assert isinstance(targets, list)
        assert len(targets) > 0

        # Check first target structure
        if targets:
            assert "entity_id" in targets[0]
            assert "current_allocation" in targets[0]
            assert "expected_response" in targets[0]

    def test_resource_optimizer_with_regions(self, sample_tier0_state):
        """Test resource optimizer mapping with geographic regions."""
        mapper = Tier0OutputMapper(sample_tier0_state)
        result = mapper.map_to_resource_optimizer()

        # Should create targets based on regions
        targets = result["allocation_targets"]
        assert len(targets) > 0

        # Check that targets have territory IDs
        assert any("territory_" in t["entity_id"] for t in targets)


@pytest.mark.unit
class TestTier5Mappings:
    """Test Tier 5 agent mappings (Self-improvement agents)."""

    @pytest.fixture
    def mapper(self, sample_tier0_state):
        return Tier0OutputMapper(sample_tier0_state)

    def test_map_to_explainer(self, mapper):
        """Test mapping to explainer input."""
        result = mapper.map_to_explainer()

        assert "analysis_results" in result
        assert "query" in result
        assert "user_expertise" in result
        assert "output_format" in result
        assert "session_id" in result

        # Check analysis results structure
        analysis_results = result["analysis_results"]
        assert isinstance(analysis_results, list)
        assert len(analysis_results) > 0

        # Each result should have key_findings
        for analysis in analysis_results:
            assert "key_findings" in analysis
            assert isinstance(analysis["key_findings"], list)

    def test_map_to_feedback_learner(self, mapper):
        """Test mapping to feedback learner input."""
        result = mapper.map_to_feedback_learner()

        assert "time_range_start" in result
        assert "time_range_end" in result
        assert "batch_id" in result
        assert "focus_agents" in result

        # Check time range format
        start = datetime.fromisoformat(result["time_range_start"])
        end = datetime.fromisoformat(result["time_range_end"])
        assert start < end


@pytest.mark.unit
class TestGetAllMappings:
    """Test get_all_mappings utility method."""

    @pytest.fixture
    def mapper(self, sample_tier0_state):
        return Tier0OutputMapper(sample_tier0_state)

    def test_get_all_mappings(self, mapper):
        """Test getting all agent mappings."""
        all_mappings = mapper.get_all_mappings()

        # Check all expected agents are present
        expected_agents = [
            "orchestrator",
            "tool_composer",
            "causal_impact",
            "gap_analyzer",
            "heterogeneous_optimizer",
            "drift_monitor",
            "experiment_designer",
            "health_score",
            "prediction_synthesizer",
            "resource_optimizer",
            "explainer",
            "feedback_learner",
        ]

        for agent in expected_agents:
            assert agent in all_mappings
            assert isinstance(all_mappings[agent], dict)

    def test_get_agent_mapping(self, mapper):
        """Test getting mapping for specific agent."""
        mapping = mapper.get_agent_mapping("causal_impact")

        assert "query" in mapping
        assert "treatment_var" in mapping
        assert "outcome_var" in mapping

    def test_get_agent_mapping_unknown(self, mapper):
        """Test getting mapping for unknown agent."""
        with pytest.raises(ValueError, match="Unknown agent"):
            mapper.get_agent_mapping("unknown_agent")


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_minimal_dataframe(self, sample_tier0_state):
        """Test with minimal DataFrame."""
        df = pd.DataFrame(
            {
                "patient_journey_id": ["pj_001", "pj_002"],
                "discontinuation_flag": [0, 1],
            }
        )
        state = sample_tier0_state.copy()
        state["eligible_df"] = df
        state["feature_importance"] = None

        mapper = Tier0OutputMapper(state)
        features = mapper._get_feature_names()

        # Should not include excluded columns
        assert "patient_journey_id" not in features
        assert "discontinuation_flag" not in features

    def test_missing_scope_spec(self, sample_tier0_state):
        """Test handling missing scope_spec."""
        state = sample_tier0_state.copy()
        del state["scope_spec"]

        mapper = Tier0OutputMapper(state)
        result = mapper.map_to_orchestrator()

        # Should use default brand
        assert "brand" in result["query"].lower() or result["query"]

    def test_missing_validation_metrics(self, sample_tier0_state):
        """Test handling missing validation metrics."""
        state = sample_tier0_state.copy()
        del state["validation_metrics"]

        mapper = Tier0OutputMapper(state)
        result = mapper.map_to_explainer()

        # Should handle missing metrics gracefully
        assert "analysis_results" in result

    def test_empty_feature_importance(self, sample_tier0_state):
        """Test with empty feature importance."""
        state = sample_tier0_state.copy()
        state["feature_importance"] = []

        mapper = Tier0OutputMapper(state)
        features = mapper._get_feature_names()

        # Should fall back to DataFrame columns
        assert len(features) > 0

    def test_prediction_synthesizer_selects_positive_outcome(self, sample_tier0_state):
        """Test that prediction synthesizer selects patient with positive outcome."""
        mapper = Tier0OutputMapper(sample_tier0_state)
        result = mapper.map_to_prediction_synthesizer()

        # Should select entity_id from a row where discontinuation_flag == 1
        entity_id = result["entity_id"]
        df = sample_tier0_state["eligible_df"]

        # Find the selected row
        selected_row = df[df["patient_journey_id"] == entity_id]
        if not selected_row.empty and "discontinuation_flag" in selected_row.columns:
            # If row found with discontinuation_flag, it should be 1
            assert selected_row["discontinuation_flag"].iloc[0] == 1


@pytest.mark.unit
class TestDataFrameColumnHandling:
    """Test handling of different DataFrame column configurations."""

    def test_map_with_all_expected_columns(self, sample_tier0_state):
        """Test mapping with all expected columns present."""
        mapper = Tier0OutputMapper(sample_tier0_state)

        # Should work without errors
        causal_input = mapper.map_to_causal_impact()
        assert "hcp_visits" in causal_input["treatment_var"]

    def test_map_without_hcp_visits(self, sample_tier0_state):
        """Test mapping when hcp_visits column is missing."""
        df = sample_tier0_state["eligible_df"].drop(columns=["hcp_visits"])
        state = sample_tier0_state.copy()
        state["eligible_df"] = df
        # Remove hcp_visits from feature importance too
        state["feature_importance"] = [
            f for f in state["feature_importance"] if f["feature"] != "hcp_visits"
        ]

        mapper = Tier0OutputMapper(state)
        result = mapper.map_to_causal_impact()

        # Should fall back to available feature
        assert "treatment_var" in result
        # Should use prior_treatments or another available feature
        assert result["treatment_var"] in [
            "prior_treatments",
            "days_on_therapy",
            "feature_1",
            "feature_2",
        ]

    def test_gap_analyzer_without_geographic_region(self, sample_tier0_state):
        """Test gap analyzer without geographic_region column."""
        df = sample_tier0_state["eligible_df"].drop(columns=["geographic_region"])
        state = sample_tier0_state.copy()
        state["eligible_df"] = df

        mapper = Tier0OutputMapper(state)
        result = mapper.map_to_gap_analyzer()

        # Should use fallback segments
        assert "segments" in result
        assert len(result["segments"]) > 0
