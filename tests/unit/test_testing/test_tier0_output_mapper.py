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

    def test_init_rejects_missing_required_keys(self):
        """Missing a required contract key raises ``TypeError``.

        ``Tier0StateContract`` declares ``experiment_id`` and
        ``eligible_df`` as ``Required``; omitting either must surface at
        the boundary so downstream agents don't fail with cryptic
        ``KeyError``s deep in their handlers.
        """
        invalid_state = {"experiment_id": "exp_001"}  # Missing eligible_df
        with pytest.raises(TypeError, match="missing required state keys"):
            Tier0OutputMapper(invalid_state)

    def test_init_rejects_extra_keys(self, sample_tier0_state):
        """Keys outside the contract raise ``TypeError`` naming them.

        The contract is the single source of truth: any caller passing a
        key not declared in ``Tier0StateContract`` is signaling either a
        typo or a tier0 schema change that wasn't propagated. Both must
        fail loudly here, not silently pass through to the agents.
        """
        state_with_extra = {**sample_tier0_state, "foo_bar": 42}
        with pytest.raises(TypeError, match=r"unexpected state keys not in contract.*foo_bar"):
            Tier0OutputMapper(state_with_extra)

    def test_init_accepts_valid_contract(self, sample_tier0_state):
        """All required + a subset of NotRequired keys is accepted.

        Anti-regression test for the ``__required_keys__`` vs
        ``__annotations__`` distinction in ``_validate_contract``: the
        sample state intentionally omits several optional keys
        (``business_utility``, ``prediction_timestamp``,
        ``cohort_result``, etc.) — none of those omissions may trigger
        the missing-required path.
        """
        # Confirm the fixture exercises the optional-omission case so
        # this test is a meaningful regression guard, not vacuous.
        assert "business_utility" not in sample_tier0_state
        assert "prediction_timestamp" not in sample_tier0_state
        assert "cohort_result" not in sample_tier0_state

        mapper = Tier0OutputMapper(sample_tier0_state)
        assert mapper.state is sample_tier0_state


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

        # #606: the threaded estimation frame carries a BINARY engagement
        # treatment (median split) for step_1's genuine treated/control contrast,
        # and EXCLUDES the raw hcp_visits count (collinear with that split and
        # degenerate on its own — zero control units).
        est = result["context"]["estimation_data"]
        assert "high_hcp_engagement" in est.columns
        assert "hcp_visits" not in est.columns
        assert {int(v) for v in est["high_hcp_engagement"].unique()} <= {0, 1}
        assert est["high_hcp_engagement"].nunique() == 2


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

        # #606: treatment is a DERIVED BINARY indicator (median-split engagement),
        # NOT the raw hcp_visits count — the count has 0 control units and
        # degenerates econml/dowhy (meaningless ATE + LinearDML/DRLearner crashes).
        assert result["treatment_var"] == "high_hcp_engagement"
        assert isinstance(result["confounders"], list)
        assert "hcp_visits" not in result["confounders"]  # treatment basis excluded
        # The derived treatment lives in the estimation frame and is binary with a
        # real treated/control split.
        tcol = result["data"][result["treatment_var"]]
        assert {int(v) for v in tcol.unique()} <= {0, 1}
        assert tcol.nunique() == 2
        # Smoke-test tuning: real OLS estimator + bounded (real) refutation suite.
        assert result["parameters"]["method"] == "ols"
        assert "refutation_config" in result["parameters"]

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
class TestPredictionTimestampPropagation:
    """Block 1B: ``prediction_timestamp`` plumbing through the mapper.

    The plan adds ``prediction_timestamp`` to the tier0 contract and asks
    every mapping that may consume it later (drift_monitor, heterogeneous
    optimizer) to surface it. These tests lock the propagation in place
    so Block 4+ can rely on it.
    """

    def test_resolves_from_top_level_state(self, sample_tier0_state):
        """Top-level ``prediction_timestamp`` wins over scope_spec."""
        ts = pd.Timestamp("2026-04-01T12:00:00Z")
        state = sample_tier0_state.copy()
        state["prediction_timestamp"] = ts
        # Even with a different value on scope_spec, top-level wins.
        state["scope_spec"] = {**state.get("scope_spec", {}), "prediction_timestamp": "1999-01-01"}

        mapper = Tier0OutputMapper(state)
        resolved = mapper._get_prediction_timestamp()

        assert resolved is not None
        assert resolved == ts

    def test_resolves_from_scope_spec_fallback(self, sample_tier0_state):
        """Falls back to ``scope_spec.prediction_timestamp`` when top-level absent."""
        state = sample_tier0_state.copy()
        state["scope_spec"] = {
            **state.get("scope_spec", {}),
            "prediction_timestamp": "2026-04-26T00:00:00",
        }
        # Ensure no top-level override.
        state.pop("prediction_timestamp", None)

        mapper = Tier0OutputMapper(state)
        resolved = mapper._get_prediction_timestamp()

        assert resolved is not None
        assert resolved == pd.Timestamp("2026-04-26T00:00:00")

    def test_returns_none_when_unset(self, sample_tier0_state):
        """No timestamp anywhere → None (not an error, not a default)."""
        state = sample_tier0_state.copy()
        state.pop("prediction_timestamp", None)
        scope = state.get("scope_spec", {}).copy()
        scope.pop("prediction_timestamp", None)
        state["scope_spec"] = scope

        mapper = Tier0OutputMapper(state)
        assert mapper._get_prediction_timestamp() is None

    def test_drift_monitor_carries_prediction_timestamp(self, sample_tier0_state):
        """``map_to_drift_monitor`` must surface ``prediction_timestamp``."""
        ts = pd.Timestamp("2026-04-26T00:00:00")
        state = sample_tier0_state.copy()
        state["prediction_timestamp"] = ts

        mapper = Tier0OutputMapper(state)
        result = mapper.map_to_drift_monitor()

        assert "prediction_timestamp" in result
        assert result["prediction_timestamp"] == ts

    def test_drift_monitor_prediction_timestamp_none_when_unset(self, sample_tier0_state):
        """When tier0 state has no timestamp, the mapping shows None."""
        state = sample_tier0_state.copy()
        state.pop("prediction_timestamp", None)
        scope = state.get("scope_spec", {}).copy()
        scope.pop("prediction_timestamp", None)
        state["scope_spec"] = scope

        mapper = Tier0OutputMapper(state)
        result = mapper.map_to_drift_monitor()

        assert "prediction_timestamp" in result
        assert result["prediction_timestamp"] is None

    def test_heterogeneous_optimizer_carries_prediction_timestamp(self, sample_tier0_state):
        """``map_to_heterogeneous_optimizer`` must surface ``prediction_timestamp``."""
        ts = pd.Timestamp("2026-05-01")
        state = sample_tier0_state.copy()
        state["scope_spec"] = {
            **state.get("scope_spec", {}),
            "prediction_timestamp": ts.isoformat(),
        }

        mapper = Tier0OutputMapper(state)
        result = mapper.map_to_heterogeneous_optimizer()

        assert "prediction_timestamp" in result
        assert result["prediction_timestamp"] == ts


@pytest.mark.unit
class TestDataFrameColumnHandling:
    """Test handling of different DataFrame column configurations."""

    def test_map_with_all_expected_columns(self, sample_tier0_state):
        """Test mapping with all expected columns present."""
        mapper = Tier0OutputMapper(sample_tier0_state)

        # Should work without errors; treatment is the derived binary indicator (#606)
        causal_input = mapper.map_to_causal_impact()
        assert causal_input["treatment_var"] == "high_hcp_engagement"
        tcol = causal_input["data"]["high_hcp_engagement"]
        assert {int(v) for v in tcol.unique()} <= {0, 1}
        assert tcol.nunique() == 2

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

        # treatment is ALWAYS the derived binary indicator; with hcp_visits gone it
        # is derived from the first available numeric feature instead (#606).
        assert result["treatment_var"] == "high_hcp_engagement"
        tcol = result["data"]["high_hcp_engagement"]
        assert {int(v) for v in tcol.unique()} <= {0, 1}
        assert tcol.nunique() == 2
        # The fallback basis is excluded from confounders (collinear with treatment).
        assert "hcp_visits" not in result["confounders"]

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
