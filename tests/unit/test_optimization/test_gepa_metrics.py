"""Unit tests for GEPA metric classes.

Tests the E2IGEPAMetric protocol implementation for each agent type:
- CausalImpactGEPAMetric (Tier 2 Hybrid)
- ExperimentDesignerGEPAMetric (Tier 3 Hybrid)
- FeedbackLearnerGEPAMetric (Tier 5 Deep)
- StandardAgentGEPAMetric (Standard agents)

Version: 4.3
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# Mark all tests in this module to run on the same worker (DSPy import safety)
pytestmark = pytest.mark.xdist_group(name="gepa_metrics")


class TestGEPAMetricImports:
    """Test that GEPA metric imports work correctly."""

    def test_base_metric_import(self):
        """Test base metric protocol import."""
        from src.optimization.gepa.metrics.base import E2IGEPAMetric, ScoreWithFeedback

        assert E2IGEPAMetric is not None
        assert ScoreWithFeedback is not None

    def test_causal_impact_metric_import(self):
        """Test CausalImpactGEPAMetric import."""
        from src.optimization.gepa.metrics.causal_impact_metric import (
            CausalImpactGEPAMetric,
        )

        assert CausalImpactGEPAMetric is not None

    def test_experiment_designer_metric_import(self):
        """Test ExperimentDesignerGEPAMetric import."""
        from src.optimization.gepa.metrics.experiment_designer_metric import (
            ExperimentDesignerGEPAMetric,
        )

        assert ExperimentDesignerGEPAMetric is not None

    def test_feedback_learner_metric_import(self):
        """Test FeedbackLearnerGEPAMetric import."""
        from src.optimization.gepa.metrics.feedback_learner_metric import (
            FeedbackLearnerGEPAMetric,
        )

        assert FeedbackLearnerGEPAMetric is not None

    def test_standard_agent_metric_import(self):
        """Test StandardAgentGEPAMetric import."""
        from src.optimization.gepa.metrics.standard_agent_metric import (
            StandardAgentGEPAMetric,
        )

        assert StandardAgentGEPAMetric is not None

    def test_factory_import(self):
        """Test metric factory import."""
        from src.optimization.gepa.metrics import get_metric_for_agent

        assert get_metric_for_agent is not None


class TestCausalImpactGEPAMetric:
    """Test CausalImpactGEPAMetric (Tier 2 Hybrid agent)."""

    @pytest.fixture
    def metric(self):
        """Create CausalImpactGEPAMetric instance."""
        from src.optimization.gepa.metrics.causal_impact_metric import (
            CausalImpactGEPAMetric,
        )

        return CausalImpactGEPAMetric()

    def test_metric_initialization(self, metric):
        """Test metric initializes with correct attributes."""
        assert hasattr(metric, "name")
        assert hasattr(metric, "description")
        assert "causal" in metric.name.lower() or "impact" in metric.name.lower()

    def test_metric_call_returns_score(self, metric):
        """Test metric __call__ returns valid score."""
        # Create mock example and prediction with proper attribute configuration
        example = MagicMock()
        example.query = "What caused the TRx increase?"
        example.context = {"brand": "Remibrutinib"}
        example.data_characteristics = {"heterogeneous": True}
        example.expected_kpis = ["TRx", "NRx"]

        prediction = MagicMock()
        prediction.response = "The TRx increase was caused by increased HCP engagement."
        prediction.ate = 0.15
        prediction.confidence = 0.85
        # Configure getattr returns for CausalImpactGEPAMetric
        prediction.refutation_results = {
            "placebo_treatment": {"status": "passed"},
            "random_common_cause": {"status": "passed"},
            "data_subset": {"status": "passed"},
            "bootstrap": {"status": "passed"},
            "sensitivity_e_value": {"status": "passed"},
        }
        prediction.sensitivity_analysis = {"e_value": 2.5}
        prediction.dag_approved = True
        prediction.estimation_method = "CausalForest"
        prediction.kpi_attribution = ["TRx", "NRx"]
        prediction.recommendations = ["Increase HCP engagement"]

        result = metric(example, prediction, trace=None)

        # Result should be a score (float) or ScoreWithFeedback (dict)
        assert isinstance(result, (float, int, dict))
        if isinstance(result, (float, int)):
            assert 0.0 <= result <= 1.0
        else:
            assert "score" in result

    def test_metric_with_missing_fields(self, metric):
        """Test metric handles missing fields gracefully."""
        example = MagicMock(spec=[])  # spec=[] means no auto-created attributes
        example.query = "Test query"
        example.data_characteristics = {}
        example.expected_kpis = []

        prediction = MagicMock(spec=[])  # spec=[] means no auto-created attributes
        prediction.response = ""
        # Set None for all optional attributes to test missing fields
        prediction.refutation_results = None
        prediction.sensitivity_analysis = None
        prediction.dag_approved = False
        prediction.estimation_method = None
        prediction.kpi_attribution = []
        prediction.recommendations = None

        # Should not raise, should return low score
        result = metric(example, prediction, trace=None)
        assert isinstance(result, (float, int, dict))


class TestExperimentDesignerGEPAMetric:
    """Test ExperimentDesignerGEPAMetric (Tier 3 Hybrid agent)."""

    @pytest.fixture
    def metric(self):
        """Create ExperimentDesignerGEPAMetric instance."""
        from src.optimization.gepa.metrics.experiment_designer_metric import (
            ExperimentDesignerGEPAMetric,
        )

        return ExperimentDesignerGEPAMetric()

    def test_metric_initialization(self, metric):
        """Test metric initializes with correct attributes."""
        assert hasattr(metric, "name")
        assert hasattr(metric, "description")

    def test_metric_call_returns_score(self, metric):
        """Test metric __call__ returns valid score."""
        example = MagicMock()
        example.query = "Design an A/B test for email campaign"
        example.target_metric = "conversion_rate"
        example.expected_effect_size = 0.2

        prediction = MagicMock()
        prediction.response = "Recommended A/B test design with 95% power."
        prediction.power = 0.95
        prediction.sample_size = 1000
        # Configure attributes for ExperimentDesignerGEPAMetric
        prediction.power_calculation = {"power": 0.95, "required_n": 500}
        prediction.randomization_method = "stratified"
        prediction.control_group = True
        prediction.blinding = True
        prediction.past_learnings_applied = [{"applied": True}, {"applied": True}]
        prediction.preregistration = {
            "hypothesis": "test hypothesis",
            "primary_outcome": "conversion_rate",
            "sample_size_justification": "80% power",
            "analysis_plan": "t-test",
            "stopping_rules": "early stopping allowed",
        }

        result = metric(example, prediction, trace=None)
        assert isinstance(result, (float, int, dict))


class TestFeedbackLearnerGEPAMetric:
    """Test FeedbackLearnerGEPAMetric (Tier 5 Deep agent)."""

    @pytest.fixture
    def metric(self):
        """Create FeedbackLearnerGEPAMetric instance."""
        from src.optimization.gepa.metrics.feedback_learner_metric import (
            FeedbackLearnerGEPAMetric,
        )

        return FeedbackLearnerGEPAMetric()

    def test_metric_initialization(self, metric):
        """Test metric initializes with correct attributes."""
        assert hasattr(metric, "name")
        assert hasattr(metric, "description")

    def test_metric_call_returns_score(self, metric):
        """Test metric __call__ returns valid score."""
        example = MagicMock()
        example.query = "Analyze feedback patterns"
        example.signals = [{"type": "positive", "reward": 0.9}]
        example.expected_patterns = ["pattern1", "pattern2"]

        prediction = MagicMock()
        prediction.response = "Identified 3 improvement patterns."
        prediction.patterns = ["pattern1", "pattern2"]
        # Configure attributes for FeedbackLearnerGEPAMetric
        prediction.extracted_learnings = ["pattern1 improvement", "pattern2 enhancement"]
        prediction.storage_format = {"compressed": True, "indexed": True}
        prediction.application_results = {"performance_delta": 0.1}

        result = metric(example, prediction, trace=None)
        assert isinstance(result, (float, int, dict))


class TestStandardAgentGEPAMetric:
    """Test StandardAgentGEPAMetric (standard agents)."""

    @pytest.fixture
    def metric(self):
        """Create StandardAgentGEPAMetric instance."""
        from src.optimization.gepa.metrics.standard_agent_metric import (
            StandardAgentGEPAMetric,
        )

        return StandardAgentGEPAMetric()

    def test_metric_initialization(self, metric):
        """Test metric initializes with correct attributes."""
        assert hasattr(metric, "name")
        assert hasattr(metric, "description")

    def test_metric_call_returns_score(self, metric):
        """Test metric __call__ returns valid score."""
        example = MagicMock()
        example.query = "General analytics query"
        example.expected_output = "analysis result"

        prediction = MagicMock()
        prediction.response = "Here is the analysis."
        # Configure attributes for StandardAgentGEPAMetric
        prediction.latency_ms = 500
        prediction.output = "analysis result"

        result = metric(example, prediction, trace=None)
        assert isinstance(result, (float, int, dict))


class TestMetricFactory:
    """Test the get_metric_for_agent factory function."""

    def test_get_metric_for_causal_impact(self):
        """Test factory returns CausalImpactGEPAMetric for causal_impact agent."""
        from src.optimization.gepa.metrics import get_metric_for_agent
        from src.optimization.gepa.metrics.causal_impact_metric import (
            CausalImpactGEPAMetric,
        )

        metric = get_metric_for_agent("causal_impact")
        assert isinstance(metric, CausalImpactGEPAMetric)

    def test_get_metric_for_experiment_designer(self):
        """Test factory returns ExperimentDesignerGEPAMetric for experiment_designer agent."""
        from src.optimization.gepa.metrics import get_metric_for_agent
        from src.optimization.gepa.metrics.experiment_designer_metric import (
            ExperimentDesignerGEPAMetric,
        )

        metric = get_metric_for_agent("experiment_designer")
        assert isinstance(metric, ExperimentDesignerGEPAMetric)

    def test_get_metric_for_feedback_learner(self):
        """Test factory returns FeedbackLearnerGEPAMetric for feedback_learner agent."""
        from src.optimization.gepa.metrics import get_metric_for_agent
        from src.optimization.gepa.metrics.feedback_learner_metric import (
            FeedbackLearnerGEPAMetric,
        )

        metric = get_metric_for_agent("feedback_learner")
        assert isinstance(metric, FeedbackLearnerGEPAMetric)

    def test_get_metric_for_unknown_agent_returns_standard(self):
        """Test factory returns StandardAgentGEPAMetric for unknown agents."""
        from src.optimization.gepa.metrics import get_metric_for_agent
        from src.optimization.gepa.metrics.standard_agent_metric import (
            StandardAgentGEPAMetric,
        )

        metric = get_metric_for_agent("unknown_agent")
        assert isinstance(metric, StandardAgentGEPAMetric)

    def test_get_metric_for_tier_0_agents(self):
        """Test factory returns StandardAgentGEPAMetric for Tier 0 agents."""
        from src.optimization.gepa.metrics import get_metric_for_agent
        from src.optimization.gepa.metrics.standard_agent_metric import (
            StandardAgentGEPAMetric,
        )

        tier_0_agents = [
            "scope_definer",
            "data_preparer",
            "feature_analyzer",
            "model_selector",
            "model_trainer",
            "model_deployer",
            "observability_connector",
        ]

        for agent_name in tier_0_agents:
            metric = get_metric_for_agent(agent_name)
            assert isinstance(metric, StandardAgentGEPAMetric), f"Failed for {agent_name}"


class TestScoreWithFeedback:
    """Test ScoreWithFeedback type and usage."""

    def test_score_with_feedback_dict_format(self):
        """Test ScoreWithFeedback dict format."""
        from src.optimization.gepa.metrics.base import ScoreWithFeedback

        # ScoreWithFeedback is a TypeAlias for dict
        score_result: ScoreWithFeedback = {
            "score": 0.85,
            "feedback": "Good causal reasoning with proper confounding control.",
        }

        assert score_result["score"] == 0.85
        assert "causal" in score_result["feedback"].lower()

    def test_metric_returns_feedback_when_available(self):
        """Test that metrics can return ScoreWithFeedback."""
        from src.optimization.gepa.metrics.causal_impact_metric import (
            CausalImpactGEPAMetric,
        )

        metric = CausalImpactGEPAMetric()

        example = MagicMock()
        example.query = "Causal analysis query"
        example.data_characteristics = {"heterogeneous": True}
        example.expected_kpis = ["TRx"]

        prediction = MagicMock()
        prediction.response = "Detailed causal analysis with ATE estimation."
        prediction.ate = 0.2
        prediction.confidence = 0.9
        # Configure required attributes
        prediction.refutation_results = {
            "placebo_treatment": {"status": "passed"},
            "random_common_cause": {"status": "passed"},
            "data_subset": {"status": "passed"},
            "bootstrap": {"status": "passed"},
            "sensitivity_e_value": {"status": "passed"},
        }
        prediction.sensitivity_analysis = {"e_value": 2.5}
        prediction.dag_approved = True
        prediction.estimation_method = "CausalForest"
        prediction.kpi_attribution = ["TRx"]
        prediction.recommendations = ["Action item"]

        result = metric(example, prediction, trace=None)

        # Result could be float or dict with feedback
        if isinstance(result, dict):
            assert "score" in result
            # Feedback is optional
            if "feedback" in result:
                assert isinstance(result["feedback"], str)
