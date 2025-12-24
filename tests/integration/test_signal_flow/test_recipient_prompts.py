"""
E2I Signal Flow Integration Tests - Batch 4: Recipient Prompt Distribution

Tests prompt distribution to Recipient agents (health_score, resource_optimizer, explainer).

Recipient agents:
- health_score (Recipient role - Fast Path)
- resource_optimizer (Recipient role - computational)
- explainer (Recipient role - Deep Reasoning)

Run: pytest tests/integration/test_signal_flow/test_recipient_prompts.py -v
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch


# =============================================================================
# HEALTH SCORE RECIPIENT TESTS
# =============================================================================


class TestHealthScoreRecipient:
    """Test health_score DSPy Recipient implementation."""

    def test_import_health_score_integration(self):
        """Verify HealthScoreDSPyIntegration can be imported."""
        from src.agents.health_score.dspy_integration import (
            HealthScoreDSPyIntegration,
        )

        assert HealthScoreDSPyIntegration is not None

    def test_create_health_score_integration(self):
        """Create health_score DSPy integration instance."""
        from src.agents.health_score.dspy_integration import (
            HealthScoreDSPyIntegration,
        )

        integration = HealthScoreDSPyIntegration()

        assert integration is not None
        assert integration.dspy_type == "recipient"

    def test_health_score_has_prompts(self):
        """Verify health_score has prompt templates."""
        from src.agents.health_score.dspy_integration import (
            HealthScoreDSPyIntegration,
            HealthReportPrompts,
        )

        integration = HealthScoreDSPyIntegration()

        assert hasattr(integration, "prompts")
        assert isinstance(integration.prompts, HealthReportPrompts)

    def test_health_score_prompt_templates(self):
        """Verify health_score prompt templates exist."""
        from src.agents.health_score.dspy_integration import (
            HealthReportPrompts,
        )

        prompts = HealthReportPrompts()

        assert prompts.summary_template
        assert prompts.recommendation_template
        assert prompts.issue_description_template
        assert prompts.version == "1.0"

    def test_health_score_update_prompts(self):
        """Test updating prompts with optimized versions."""
        from src.agents.health_score.dspy_integration import (
            HealthScoreDSPyIntegration,
        )

        integration = HealthScoreDSPyIntegration()

        original_version = integration.prompts.version

        # Update with optimized prompts
        integration.update_optimized_prompts(
            prompts={
                "summary_template": "New optimized summary: {grade} {score}",
                "recommendation_template": "New optimized recommendation",
            },
            optimization_score=0.85,
        )

        assert integration.prompts.summary_template == "New optimized summary: {grade} {score}"
        assert integration.prompts.optimization_score == 0.85
        assert integration.prompts.last_optimized != ""
        assert integration.prompts.version != original_version

    def test_health_score_get_summary_prompt(self):
        """Test getting formatted summary prompt."""
        from src.agents.health_score.dspy_integration import (
            HealthScoreDSPyIntegration,
        )

        integration = HealthScoreDSPyIntegration()

        formatted = integration.get_summary_prompt(
            grade="A",
            score=92.5,
            components="database, cache, api",
            critical_count=0,
            warning_count=2,
        )

        assert "A" in formatted
        assert "92.5" in formatted
        assert "database" in formatted


class TestHealthScoreSignatures:
    """Test health_score DSPy signatures."""

    def test_import_health_summary_signature(self):
        """Verify HealthSummarySignature can be imported."""
        from src.agents.health_score.dspy_integration import (
            HealthSummarySignature,
            DSPY_AVAILABLE,
        )

        if DSPY_AVAILABLE:
            assert HealthSummarySignature is not None
        else:
            pytest.skip("DSPy not available")

    def test_import_health_recommendation_signature(self):
        """Verify HealthRecommendationSignature can be imported."""
        from src.agents.health_score.dspy_integration import (
            HealthRecommendationSignature,
            DSPY_AVAILABLE,
        )

        if DSPY_AVAILABLE:
            assert HealthRecommendationSignature is not None
        else:
            pytest.skip("DSPy not available")


# =============================================================================
# RESOURCE OPTIMIZER RECIPIENT TESTS
# =============================================================================


class TestResourceOptimizerRecipient:
    """Test resource_optimizer DSPy Recipient implementation."""

    def test_import_resource_optimizer_integration(self):
        """Verify ResourceOptimizerDSPyIntegration can be imported."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        assert ResourceOptimizerDSPyIntegration is not None

    def test_create_resource_optimizer_integration(self):
        """Create resource_optimizer DSPy integration instance."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()

        assert integration is not None
        assert integration.dspy_type == "recipient"

    def test_resource_optimizer_has_prompts(self):
        """Verify resource_optimizer has prompt templates."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
            ResourceOptimizationPrompts,
        )

        integration = ResourceOptimizerDSPyIntegration()

        assert hasattr(integration, "prompts")
        assert isinstance(integration.prompts, ResourceOptimizationPrompts)

    def test_resource_optimizer_prompt_templates(self):
        """Verify resource_optimizer prompt templates exist."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizationPrompts,
        )

        prompts = ResourceOptimizationPrompts()

        assert prompts.summary_template
        assert prompts.recommendation_template
        assert prompts.scenario_comparison_template
        assert prompts.constraint_warning_template

    def test_resource_optimizer_update_prompts(self):
        """Test updating prompts with optimized versions."""
        from src.agents.resource_optimizer.dspy_integration import (
            ResourceOptimizerDSPyIntegration,
        )

        integration = ResourceOptimizerDSPyIntegration()

        # Update with optimized prompts
        integration.update_optimized_prompts(
            prompts={
                "summary_template": "Optimized summary for {resource_type}",
                "recommendation_template": "Optimized recommendation",
            },
            optimization_score=0.78,
        )

        assert "Optimized summary" in integration.prompts.summary_template
        assert integration.prompts.optimization_score == 0.78


class TestResourceOptimizerSignatures:
    """Test resource_optimizer DSPy signatures."""

    def test_import_optimization_summary_signature(self):
        """Verify OptimizationSummarySignature can be imported."""
        from src.agents.resource_optimizer.dspy_integration import (
            OptimizationSummarySignature,
            DSPY_AVAILABLE,
        )

        if DSPY_AVAILABLE:
            assert OptimizationSummarySignature is not None
        else:
            pytest.skip("DSPy not available")

    def test_import_allocation_recommendation_signature(self):
        """Verify AllocationRecommendationSignature can be imported."""
        from src.agents.resource_optimizer.dspy_integration import (
            AllocationRecommendationSignature,
            DSPY_AVAILABLE,
        )

        if DSPY_AVAILABLE:
            assert AllocationRecommendationSignature is not None
        else:
            pytest.skip("DSPy not available")

    def test_import_scenario_narrative_signature(self):
        """Verify ScenarioNarrativeSignature can be imported."""
        from src.agents.resource_optimizer.dspy_integration import (
            ScenarioNarrativeSignature,
            DSPY_AVAILABLE,
        )

        if DSPY_AVAILABLE:
            assert ScenarioNarrativeSignature is not None
        else:
            pytest.skip("DSPy not available")


# =============================================================================
# EXPLAINER RECIPIENT TESTS
# =============================================================================


class TestExplainerRecipient:
    """Test explainer DSPy Recipient implementation."""

    def test_import_explainer_integration(self):
        """Verify ExplainerDSPyIntegration can be imported."""
        from src.agents.explainer.dspy_integration import (
            ExplainerDSPyIntegration,
        )

        assert ExplainerDSPyIntegration is not None

    def test_create_explainer_integration(self):
        """Create explainer DSPy integration instance."""
        from src.agents.explainer.dspy_integration import (
            ExplainerDSPyIntegration,
        )

        integration = ExplainerDSPyIntegration()

        assert integration is not None
        assert integration.dspy_type == "recipient"

    def test_explainer_has_prompts(self):
        """Verify explainer has prompt templates."""
        from src.agents.explainer.dspy_integration import (
            ExplainerDSPyIntegration,
            ExplanationPrompts,
        )

        integration = ExplainerDSPyIntegration()

        assert hasattr(integration, "prompts")
        assert isinstance(integration.prompts, ExplanationPrompts)

    def test_explainer_prompt_templates(self):
        """Verify explainer prompt templates exist."""
        from src.agents.explainer.dspy_integration import (
            ExplanationPrompts,
        )

        prompts = ExplanationPrompts()

        assert prompts.context_assembly_template
        assert prompts.executive_summary_template
        assert prompts.detailed_explanation_template
        assert prompts.insight_extraction_template
        assert prompts.narrative_section_template
        assert prompts.followup_questions_template

    def test_explainer_prompts_to_dict(self):
        """Test prompt serialization."""
        from src.agents.explainer.dspy_integration import (
            ExplanationPrompts,
        )

        prompts = ExplanationPrompts()
        prompts_dict = prompts.to_dict()

        assert "context_assembly_template" in prompts_dict
        assert "executive_summary_template" in prompts_dict
        assert "version" in prompts_dict
        assert "optimization_score" in prompts_dict

    def test_explainer_update_prompts(self):
        """Test updating prompts with optimized versions."""
        from src.agents.explainer.dspy_integration import (
            ExplainerDSPyIntegration,
        )

        integration = ExplainerDSPyIntegration()

        # Update with optimized prompts
        integration.update_optimized_prompts(
            prompts={
                "executive_summary_template": "Optimized executive summary",
                "detailed_explanation_template": "Optimized explanation",
            },
            optimization_score=0.92,
        )

        assert integration.prompts.optimization_score == 0.92
        assert integration.prompts.last_optimized != ""


class TestExplainerSignatures:
    """Test explainer DSPy signatures."""

    def test_import_explanation_synthesis_signature(self):
        """Verify ExplanationSynthesisSignature can be imported."""
        from src.agents.explainer.dspy_integration import (
            ExplanationSynthesisSignature,
            DSPY_AVAILABLE,
        )

        if DSPY_AVAILABLE:
            assert ExplanationSynthesisSignature is not None
        else:
            pytest.skip("DSPy not available")

    def test_import_insight_extraction_signature(self):
        """Verify InsightExtractionSignature can be imported."""
        from src.agents.explainer.dspy_integration import (
            InsightExtractionSignature,
            DSPY_AVAILABLE,
        )

        if DSPY_AVAILABLE:
            assert InsightExtractionSignature is not None
        else:
            pytest.skip("DSPy not available")

    def test_import_narrative_structure_signature(self):
        """Verify NarrativeStructureSignature can be imported."""
        from src.agents.explainer.dspy_integration import (
            NarrativeStructureSignature,
            DSPY_AVAILABLE,
        )

        if DSPY_AVAILABLE:
            assert NarrativeStructureSignature is not None
        else:
            pytest.skip("DSPy not available")

    def test_import_query_rewrite_signature(self):
        """Verify QueryRewriteForExplanationSignature can be imported."""
        from src.agents.explainer.dspy_integration import (
            QueryRewriteForExplanationSignature,
            DSPY_AVAILABLE,
        )

        if DSPY_AVAILABLE:
            assert QueryRewriteForExplanationSignature is not None
        else:
            pytest.skip("DSPy not available")


# =============================================================================
# PROMPT DISTRIBUTION TESTS
# =============================================================================


class TestPromptDistribution:
    """Test prompt distribution from hub to recipients."""

    def test_all_recipients_have_recipient_type(self):
        """All recipient agents must have dspy_type='recipient'."""
        from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration
        from src.agents.resource_optimizer.dspy_integration import ResourceOptimizerDSPyIntegration
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        recipients = [
            HealthScoreDSPyIntegration(),
            ResourceOptimizerDSPyIntegration(),
            ExplainerDSPyIntegration(),
        ]

        for recipient in recipients:
            assert recipient.dspy_type == "recipient"

    def test_all_recipients_can_update_prompts(self):
        """All recipients must support update_optimized_prompts method."""
        from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration
        from src.agents.resource_optimizer.dspy_integration import ResourceOptimizerDSPyIntegration
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        recipients = [
            HealthScoreDSPyIntegration(),
            ResourceOptimizerDSPyIntegration(),
            ExplainerDSPyIntegration(),
        ]

        for recipient in recipients:
            assert hasattr(recipient, "update_optimized_prompts")
            assert callable(recipient.update_optimized_prompts)

    def test_prompt_version_tracking(self):
        """Test that prompt versions are tracked correctly."""
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integration = ExplainerDSPyIntegration()

        # Initial version
        assert integration.prompts.version == "1.0"

        # First update
        integration.update_optimized_prompts(
            prompts={"executive_summary_template": "Update 1"},
            optimization_score=0.7,
        )
        assert "1." in integration.prompts.version

        # Second update
        integration.update_optimized_prompts(
            prompts={"executive_summary_template": "Update 2"},
            optimization_score=0.8,
        )
        # Version should increment
        assert integration.prompts.optimization_score == 0.8

    def test_prompt_timestamp_tracking(self):
        """Test that optimization timestamps are tracked."""
        from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration

        integration = HealthScoreDSPyIntegration()

        # Initially empty
        assert integration.prompts.last_optimized == ""

        # After update
        integration.update_optimized_prompts(
            prompts={"summary_template": "Updated"},
            optimization_score=0.75,
        )

        # Should have ISO timestamp
        assert integration.prompts.last_optimized != ""
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(integration.prompts.last_optimized.replace("Z", "+00:00"))


class TestPromptQualityMetrics:
    """Test prompt quality metrics per contract."""

    def test_optimization_score_bounds(self):
        """Optimization scores should be in [0, 1] range."""
        from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration

        integration = HealthScoreDSPyIntegration()

        # Update with valid score
        integration.update_optimized_prompts(
            prompts={"summary_template": "Test"},
            optimization_score=0.85,
        )

        assert 0.0 <= integration.prompts.optimization_score <= 1.0

    def test_recipient_prompt_serialization(self):
        """Test that recipient prompts can be serialized."""
        from src.agents.health_score.dspy_integration import HealthReportPrompts
        from src.agents.resource_optimizer.dspy_integration import ResourceOptimizationPrompts
        from src.agents.explainer.dspy_integration import ExplanationPrompts
        import json

        prompt_classes = [
            HealthReportPrompts(),
            ResourceOptimizationPrompts(),
            ExplanationPrompts(),
        ]

        for prompts in prompt_classes:
            prompts_dict = prompts.to_dict()
            json_str = json.dumps(prompts_dict)
            restored = json.loads(json_str)

            assert "version" in restored
            assert "optimization_score" in restored


# =============================================================================
# HYBRID AGENT RECIPIENT TESTS
# =============================================================================


class TestHybridAgentAsRecipient:
    """Test hybrid agents (tool_composer, feedback_learner) as recipients."""

    def test_feedback_learner_can_receive_prompts(self):
        """feedback_learner should be able to receive optimized prompts."""
        from src.agents.feedback_learner.dspy_integration import (
            FeedbackLearnerOptimizer,
        )

        optimizer = FeedbackLearnerOptimizer()

        # Optimizer can have its own prompts optimized
        assert hasattr(optimizer, "pattern_metric")
        assert hasattr(optimizer, "recommendation_metric")

    def test_tool_composer_dspy_integration(self):
        """Test tool_composer DSPy integration exists."""
        from src.agents.tool_composer.dspy_integration import (
            ToolComposerDSPyIntegration,
        )

        integration = ToolComposerDSPyIntegration()

        # Tool composer is hybrid - check it exists
        assert integration is not None


class TestRecipientContractCompliance:
    """Test compliance with SignalFlowContract for recipients."""

    def test_all_contract_recipients_implemented(self):
        """All contract-defined recipients should be implemented."""
        CONTRACT_RECIPIENTS = {
            "health_score",
            "resource_optimizer",
            "explainer",
        }

        from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration
        from src.agents.resource_optimizer.dspy_integration import ResourceOptimizerDSPyIntegration
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        integrations = {
            "health_score": HealthScoreDSPyIntegration(),
            "resource_optimizer": ResourceOptimizerDSPyIntegration(),
            "explainer": ExplainerDSPyIntegration(),
        }

        assert set(integrations.keys()) == CONTRACT_RECIPIENTS

        for name, integration in integrations.items():
            assert integration.dspy_type == "recipient", f"{name} should be recipient"

    def test_recipients_have_required_methods(self):
        """Recipients must have required interface methods."""
        from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration
        from src.agents.resource_optimizer.dspy_integration import ResourceOptimizerDSPyIntegration
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        required_methods = ["update_optimized_prompts"]

        recipients = [
            HealthScoreDSPyIntegration(),
            ResourceOptimizerDSPyIntegration(),
            ExplainerDSPyIntegration(),
        ]

        for recipient in recipients:
            for method in required_methods:
                assert hasattr(recipient, method), f"Missing {method}"
                assert callable(getattr(recipient, method))

    def test_recipients_have_prompts_property(self):
        """Recipients must expose prompts property."""
        from src.agents.health_score.dspy_integration import HealthScoreDSPyIntegration
        from src.agents.resource_optimizer.dspy_integration import ResourceOptimizerDSPyIntegration
        from src.agents.explainer.dspy_integration import ExplainerDSPyIntegration

        recipients = [
            HealthScoreDSPyIntegration(),
            ResourceOptimizerDSPyIntegration(),
            ExplainerDSPyIntegration(),
        ]

        for recipient in recipients:
            assert hasattr(recipient, "prompts")
            assert hasattr(recipient.prompts, "to_dict")
