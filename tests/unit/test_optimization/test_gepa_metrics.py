"""Unit tests for GEPA metric classes.

Tests the E2IGEPAMetric protocol implementation for each agent type:
- CausalImpactGEPAMetric (Tier 2 Hybrid)
- ExperimentDesignerGEPAMetric (Tier 3 Hybrid)
- FeedbackLearnerGEPAMetric (Tier 5 Deep)
- StandardAgentGEPAMetric (Standard agents)

Version: 4.3
"""

from unittest.mock import MagicMock

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

    def test_tool_composer_metric_import(self):
        """Test ToolComposerGEPAMetric import."""
        from src.optimization.gepa.metrics.tool_composer_metric import (
            ToolComposerGEPAMetric,
        )

        assert ToolComposerGEPAMetric is not None

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


class TestToolComposerGEPAMetric:
    """Test ToolComposerGEPAMetric (Tier 1 Orchestration agent)."""

    @pytest.fixture
    def metric(self):
        """Create ToolComposerGEPAMetric instance."""
        from src.optimization.gepa.metrics.tool_composer_metric import (
            ToolComposerGEPAMetric,
        )

        return ToolComposerGEPAMetric()

    def test_metric_initialization(self, metric):
        """Test metric initializes with correct attributes."""
        assert hasattr(metric, "name")
        assert hasattr(metric, "description")
        assert "tool_composer" in metric.name.lower() or "composer" in metric.name.lower()
        assert metric.decomposition_weight == 0.25
        assert metric.planning_weight == 0.25
        assert metric.execution_weight == 0.25
        assert metric.synthesis_weight == 0.25

    def test_metric_call_returns_score_with_feedback(self, metric):
        """Test metric __call__ returns ScoreWithFeedback dict."""
        # Create mock example
        example = MagicMock()
        example.query = "Compare causal impact of X vs Y"
        example.expected_entities = ["X", "Y"]
        example.expected_tools = ["causal_effect_estimator", "cate_analyzer"]

        # Create mock decomposition
        sub_q1 = MagicMock()
        sub_q1.id = "sq_1"
        sub_q1.question = "What is causal effect of X?"
        sub_q1.intent = "CAUSAL"

        sub_q2 = MagicMock()
        sub_q2.id = "sq_2"
        sub_q2.question = "What is causal effect of Y?"
        sub_q2.intent = "CAUSAL"

        decomposition = MagicMock()
        decomposition.sub_questions = [sub_q1, sub_q2]
        decomposition.extracted_entities = ["X", "Y"]

        # Create mock plan
        step1 = MagicMock()
        step1.tool_name = "causal_effect_estimator"
        step1.depends_on_steps = []

        step2 = MagicMock()
        step2.tool_name = "cate_analyzer"
        step2.depends_on_steps = ["step_1"]

        mapping1 = MagicMock()
        mapping1.confidence = 0.9

        mapping2 = MagicMock()
        mapping2.confidence = 0.85

        plan = MagicMock()
        plan.steps = [step1, step2]
        plan.tool_mappings = [mapping1, mapping2]
        plan.parallel_groups = [["step_1"], ["step_2"]]

        # Create mock execution
        execution = MagicMock()
        execution.tools_executed = 2
        execution.tools_succeeded = 2
        execution.step_results = []

        # Create mock response
        response = MagicMock()
        response.answer = "Detailed comparison of X and Y causal impacts with confidence 0.85."
        response.confidence = 0.85
        response.caveats = []
        response.failed_components = []
        response.supporting_data = {"X_ate": 0.15, "Y_ate": 0.22}

        # Create prediction with all 4-phase outputs
        prediction = MagicMock()
        prediction.decomposition = decomposition
        prediction.plan = plan
        prediction.execution = execution
        prediction.response = response
        prediction.total_duration_ms = 5000

        result = metric(example, prediction, trace=None)

        # Should return ScoreWithFeedback dict
        assert isinstance(result, dict)
        assert "score" in result
        assert "feedback" in result
        assert 0.0 <= result["score"] <= 1.0
        assert "DECOMPOSE" in result["feedback"]
        assert "PLAN" in result["feedback"]
        assert "EXECUTE" in result["feedback"]
        assert "SYNTHESIZE" in result["feedback"]

    def test_score_decomposition_missing(self, metric):
        """Test decomposition scoring with missing decomposition."""
        example = MagicMock()
        prediction = MagicMock()
        prediction.decomposition = None

        score, feedback = metric._score_decomposition(prediction, example)
        assert score == 0.0
        assert "CRITICAL" in feedback

    def test_score_decomposition_empty(self, metric):
        """Test decomposition scoring with empty sub-questions."""
        example = MagicMock()
        decomposition = MagicMock()
        decomposition.sub_questions = []

        prediction = MagicMock()
        prediction.decomposition = decomposition

        score, feedback = metric._score_decomposition(prediction, example)
        assert score == 0.0
        assert "Zero sub-questions" in feedback

    def test_score_planning_missing(self, metric):
        """Test planning scoring with missing plan."""
        example = MagicMock()
        prediction = MagicMock()
        prediction.plan = None

        score, feedback = metric._score_planning(prediction, example)
        assert score == 0.0
        assert "CRITICAL" in feedback

    def test_score_execution_all_succeed(self, metric):
        """Test execution scoring when all tools succeed."""
        execution = MagicMock()
        execution.tools_executed = 3
        execution.tools_succeeded = 3
        execution.step_results = []

        prediction = MagicMock()
        prediction.execution = execution
        prediction.total_duration_ms = 5000

        score, feedback = metric._score_execution(prediction)
        assert score > 0.7  # High score for all success
        assert "All 3 tools succeeded" in feedback

    def test_score_execution_partial_failure(self, metric):
        """Test execution scoring with partial failures."""
        execution = MagicMock()
        execution.tools_executed = 4
        execution.tools_succeeded = 3
        execution.step_results = []

        prediction = MagicMock()
        prediction.execution = execution
        prediction.total_duration_ms = 5000

        score, feedback = metric._score_execution(prediction)
        assert score > 0.3  # Some score for partial success
        assert "1/4 tools failed" in feedback

    def test_score_synthesis_empty_answer(self, metric):
        """Test synthesis scoring with empty answer."""
        example = MagicMock()
        response = MagicMock()
        response.answer = ""

        prediction = MagicMock()
        prediction.response = response
        prediction.execution = None

        score, feedback = metric._score_synthesis(prediction, example)
        assert score == 0.0
        assert "Empty answer" in feedback

    def test_score_synthesis_with_caveats(self, metric):
        """Test synthesis scoring with failed components noted in caveats."""
        example = MagicMock()

        execution = MagicMock()
        execution.tools_executed = 2
        execution.tools_succeeded = 1

        response = MagicMock()
        response.answer = "A detailed analysis with partial results due to tool failure."
        response.confidence = 0.5
        response.caveats = ["Tool X failed to execute"]
        response.failed_components = ["tool_x"]
        response.supporting_data = {}

        prediction = MagicMock()
        prediction.response = response
        prediction.execution = execution

        score, feedback = metric._score_synthesis(prediction, example)
        assert score > 0.3  # Some credit for noting failures
        assert "0.5" in feedback  # Confidence mentioned

    def test_metric_with_all_phases_successful(self, metric):
        """Test full metric with all phases working correctly."""
        # Create complete successful composition
        example = MagicMock()
        example.expected_entities = ["brand", "region"]
        example.expected_tools = ["tool_a", "tool_b"]

        # Sub-questions with valid intents
        sq1 = MagicMock()
        sq1.id = "sq_1"
        sq1.intent = "CAUSAL"

        sq2 = MagicMock()
        sq2.id = "sq_2"
        sq2.intent = "COMPARATIVE"

        decomposition = MagicMock()
        decomposition.sub_questions = [sq1, sq2]
        decomposition.extracted_entities = ["brand", "region"]

        # Plan with high-confidence mappings
        step1 = MagicMock()
        step1.tool_name = "tool_a"
        step1.depends_on_steps = []

        step2 = MagicMock()
        step2.tool_name = "tool_b"
        step2.depends_on_steps = ["step_1"]

        m1 = MagicMock()
        m1.confidence = 0.95

        m2 = MagicMock()
        m2.confidence = 0.92

        plan = MagicMock()
        plan.steps = [step1, step2]
        plan.tool_mappings = [m1, m2]
        plan.parallel_groups = [["step_1"], ["step_2"]]

        # Successful execution
        execution = MagicMock()
        execution.tools_executed = 2
        execution.tools_succeeded = 2
        execution.step_results = []

        # High-quality response
        response = MagicMock()
        response.answer = (
            "Comprehensive analysis comparing brand performance across regions with clear insights."
        )
        response.confidence = 0.9
        response.caveats = []
        response.failed_components = []
        response.supporting_data = {"result": "data"}

        prediction = MagicMock()
        prediction.decomposition = decomposition
        prediction.plan = plan
        prediction.execution = execution
        prediction.response = response
        prediction.total_duration_ms = 8000

        result = metric(example, prediction, trace=None)

        assert isinstance(result, dict)
        assert result["score"] >= 0.7  # High overall score


class TestMetricFactory:
    """Test the get_metric_for_agent factory function."""

    def test_get_metric_for_tool_composer(self):
        """Test factory returns ToolComposerGEPAMetric for tool_composer agent."""
        from src.optimization.gepa.metrics import get_metric_for_agent
        from src.optimization.gepa.metrics.tool_composer_metric import (
            ToolComposerGEPAMetric,
        )

        metric = get_metric_for_agent("tool_composer")
        assert isinstance(metric, ToolComposerGEPAMetric)

    def test_get_metric_for_causal_impact(self):
        """Test factory returns EvidenceSynthesisGEPAMetric for causal_impact agent DSPy module optimization."""
        from src.optimization.gepa.metrics import get_metric_for_agent
        from src.optimization.gepa.metrics.evidence_synthesis_metric import (
            EvidenceSynthesisGEPAMetric,
        )

        # causal_impact now uses EvidenceSynthesisGEPAMetric for DSPy module optimization
        # Use causal_impact_pipeline for full pipeline evaluation with CausalImpactGEPAMetric
        metric = get_metric_for_agent("causal_impact")
        assert isinstance(metric, EvidenceSynthesisGEPAMetric)

    def test_get_metric_for_causal_impact_pipeline(self):
        """Test factory returns CausalImpactGEPAMetric for causal_impact_pipeline (full eval)."""
        from src.optimization.gepa.metrics import get_metric_for_agent
        from src.optimization.gepa.metrics.causal_impact_metric import (
            CausalImpactGEPAMetric,
        )

        metric = get_metric_for_agent("causal_impact_pipeline")
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
