"""
Tests for Score Composer Node
"""

from datetime import datetime

import pytest

from src.agents.health_score.metrics import ScoreWeights
from src.agents.health_score.nodes.score_composer import ScoreComposerNode


class TestScoreComposerNode:
    """Tests for ScoreComposerNode"""

    @pytest.fixture
    def full_state(self):
        """State with all health scores.

        F1: all four dimensions carry ``<dim>_health_measured = True`` because
        these tests exercise the weighting/grade math for a fully-MEASURED
        result. The composer now ignores any score whose measured flag is not
        True, so the flags are required here (a score alone is no longer enough).
        """
        return {
            "query": "",
            "check_scope": "full",
            "component_statuses": [],
            "component_health_score": 0.9,
            "component_health_measured": True,
            "model_metrics": [],
            "model_health_score": 0.8,
            "model_health_measured": True,
            "pipeline_statuses": [],
            "pipeline_health_score": 0.85,
            "pipeline_health_measured": True,
            "agent_statuses": [],
            "agent_health_score": 0.95,
            "agent_health_measured": True,
            "overall_health_score": None,
            "health_grade": None,
            "critical_issues": None,
            "warnings": None,
            "health_summary": None,
            "total_latency_ms": 100,
            "timestamp": "",
            "errors": [],
            "status": "checking",
        }

    @pytest.mark.asyncio
    async def test_weighted_score_calculation(self, full_state):
        """Test weighted average calculation"""
        node = ScoreComposerNode()
        result = await node.execute(full_state)

        # Expected: 0.9*0.30 + 0.8*0.30 + 0.85*0.25 + 0.95*0.15 = 0.865
        expected_score = 0.865 * 100
        assert abs(result["overall_health_score"] - expected_score) < 0.01

    @pytest.mark.asyncio
    async def test_grade_a(self, full_state):
        """Test grade A (>=90%)"""
        full_state["component_health_score"] = 1.0
        full_state["model_health_score"] = 1.0
        full_state["pipeline_health_score"] = 1.0
        full_state["agent_health_score"] = 1.0

        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["health_grade"] == "A"
        assert result["overall_health_score"] == 100.0

    @pytest.mark.asyncio
    async def test_grade_b(self, full_state):
        """Test grade B (>=80%, <90%)"""
        # Set scores to get ~85%
        full_state["component_health_score"] = 0.85
        full_state["model_health_score"] = 0.85
        full_state["pipeline_health_score"] = 0.85
        full_state["agent_health_score"] = 0.85

        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["health_grade"] == "B"

    @pytest.mark.asyncio
    async def test_grade_c(self, full_state):
        """Test grade C (>=70%, <80%)"""
        full_state["component_health_score"] = 0.75
        full_state["model_health_score"] = 0.75
        full_state["pipeline_health_score"] = 0.75
        full_state["agent_health_score"] = 0.75

        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["health_grade"] == "C"

    @pytest.mark.asyncio
    async def test_grade_d(self, full_state):
        """Test grade D (>=60%, <70%)"""
        full_state["component_health_score"] = 0.65
        full_state["model_health_score"] = 0.65
        full_state["pipeline_health_score"] = 0.65
        full_state["agent_health_score"] = 0.65

        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["health_grade"] == "D"

    @pytest.mark.asyncio
    async def test_grade_f(self, full_state):
        """Test grade F (<60%)"""
        full_state["component_health_score"] = 0.5
        full_state["model_health_score"] = 0.5
        full_state["pipeline_health_score"] = 0.5
        full_state["agent_health_score"] = 0.5

        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["health_grade"] == "F"

    @pytest.mark.asyncio
    async def test_status_completed(self, full_state):
        """Test that status is set to completed"""
        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_timestamp_recorded(self, full_state):
        """Test that timestamp is recorded"""
        node = ScoreComposerNode()
        result = await node.execute(full_state)

        assert result["timestamp"] is not None
        # Should be valid ISO format
        datetime.fromisoformat(result["timestamp"])

    @pytest.mark.asyncio
    async def test_missing_scores_fail_closed_to_unknown(self):
        """F1 (was test_missing_scores_default_to_healthy): missing scores with
        NO measured flags must NOT default to a fabricated healthy 100/grade-A.
        With zero measured dimensions the composer fails closed to an UNKNOWN
        state (score 0.0, provenance 'unknown', grade not 'A')."""
        minimal_state = {
            "query": "",
            "check_scope": "full",
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(minimal_state)

        # Fail-closed: nothing measured => unknown, not a fabricated grade A.
        assert result["data_provenance"] == "unknown"
        assert result["overall_health_score"] == 0.0
        assert result["health_grade"] != "A"


class TestComposerProvenance:
    """F1: composer must build the overall score ONLY from measured dimensions
    and tag the result with honest provenance (measured/partial/unknown)."""

    @pytest.mark.asyncio
    async def test_all_measured_is_measured_provenance(self):
        """All 4 dimensions measured => provenance 'measured'."""
        state = {
            "component_health_score": 0.9,
            "component_health_measured": True,
            "model_health_score": 0.8,
            "model_health_measured": True,
            "pipeline_health_score": 0.85,
            "pipeline_health_measured": True,
            "agent_health_score": 0.95,
            "agent_health_measured": True,
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }
        node = ScoreComposerNode()
        result = await node.execute(state)
        assert result["data_provenance"] == "measured"

    @pytest.mark.asyncio
    async def test_only_component_measured_is_partial(self):
        """Only the component dimension measured => 'partial', and the overall
        score is computed over the component dimension ALONE (renormalized),
        not diluted by fail-open 1.0 defaults for the unmeasured dims."""
        state = {
            "component_health_score": 0.6,
            "component_health_measured": True,
            # model/pipeline/agent NOT measured (no real backend)
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }
        node = ScoreComposerNode()
        result = await node.execute(state)

        assert result["data_provenance"] == "partial"
        # Renormalized over the single measured dim => 0.6 * 100 = 60.0.
        # (If the bug were present, unmeasured dims would default to 1.0 and
        # inflate this well above 60.)
        assert abs(result["overall_health_score"] - 60.0) < 0.01

    @pytest.mark.asyncio
    async def test_zero_measured_is_unknown_not_grade_a(self):
        """No dimension measured => provenance 'unknown', score 0.0, grade not
        'A', and a critical issue explaining the gap. This is the core
        anti-fabrication guard at the composer level."""
        state = {
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }
        node = ScoreComposerNode()
        result = await node.execute(state)

        assert result["data_provenance"] == "unknown"
        assert result["overall_health_score"] == 0.0
        assert result["health_grade"] != "A"
        assert any("could be measured" in c.lower() for c in result["critical_issues"])


class TestIssueIdentification:
    """Tests for issue and warning identification"""

    @pytest.mark.asyncio
    async def test_identifies_unhealthy_components(self):
        """Test identification of unhealthy components"""
        state = {
            "component_statuses": [
                {"component_name": "db", "status": "unhealthy"},
                {"component_name": "cache", "status": "healthy"},
            ],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "Component 'db' is unhealthy" in result["critical_issues"]

    @pytest.mark.asyncio
    async def test_identifies_degraded_components(self):
        """Test identification of degraded components as warnings"""
        state = {
            "component_statuses": [
                {"component_name": "cache", "status": "degraded"},
            ],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "Component 'cache' is degraded" in result["warnings"]

    @pytest.mark.asyncio
    async def test_identifies_unhealthy_models(self):
        """Test identification of unhealthy models"""
        state = {
            "component_statuses": [],
            "model_metrics": [
                {"model_id": "model_1", "status": "unhealthy"},
            ],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "Model 'model_1' is unhealthy" in result["critical_issues"]

    @pytest.mark.asyncio
    async def test_identifies_failed_pipelines(self):
        """Test identification of failed pipelines"""
        state = {
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [
                {"pipeline_name": "etl", "status": "failed"},
            ],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "Pipeline 'etl' has failed" in result["critical_issues"]

    @pytest.mark.asyncio
    async def test_identifies_unavailable_agents(self):
        """Test identification of unavailable agents"""
        state = {
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [
                {"agent_name": "agent_1", "available": False, "success_rate": 0.0},
            ],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "Agent 'agent_1' is unavailable" in result["critical_issues"]

    @pytest.mark.asyncio
    async def test_identifies_low_success_rate_agents(self):
        """Test identification of low success rate agents as warnings"""
        state = {
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [
                {"agent_name": "agent_1", "available": True, "success_rate": 0.7},
            ],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert any("low success rate" in w for w in result["warnings"])


class TestSummaryGeneration:
    """Tests for summary generation"""

    @pytest.mark.asyncio
    async def test_excellent_summary(self):
        """Test excellent health summary (all four dimensions measured)"""
        state = {
            "component_health_score": 1.0,
            "component_health_measured": True,
            "model_health_score": 1.0,
            "model_health_measured": True,
            "pipeline_health_score": 1.0,
            "pipeline_health_measured": True,
            "agent_health_score": 1.0,
            "agent_health_measured": True,
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "excellent" in result["health_summary"]
        assert "Grade: A" in result["health_summary"]
        assert "All systems operational" in result["health_summary"]

    @pytest.mark.asyncio
    async def test_critical_summary(self):
        """Test critical health summary (all four dimensions measured)"""
        state = {
            "component_health_score": 0.3,
            "component_health_measured": True,
            "model_health_score": 0.3,
            "model_health_measured": True,
            "pipeline_health_score": 0.3,
            "pipeline_health_measured": True,
            "agent_health_score": 0.3,
            "agent_health_measured": True,
            "component_statuses": [
                {"component_name": "db", "status": "unhealthy"},
            ],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        assert "critical" in result["health_summary"]
        assert "Grade: F" in result["health_summary"]
        assert "critical issue" in result["health_summary"]


class TestCustomWeightsAndGrades:
    """Tests for custom weights and grade thresholds"""

    @pytest.mark.asyncio
    async def test_custom_weights(self):
        """Test with custom score weights (all four dimensions measured)"""
        state = {
            "component_health_score": 1.0,
            "component_health_measured": True,
            "model_health_score": 0.5,
            "model_health_measured": True,
            "pipeline_health_score": 0.5,
            "pipeline_health_measured": True,
            "agent_health_score": 0.5,
            "agent_health_measured": True,
            "component_statuses": [],
            "model_metrics": [],
            "pipeline_statuses": [],
            "agent_statuses": [],
            "total_latency_ms": 0,
            "errors": [],
            "status": "checking",
        }

        # Custom weights: component = 1.0, others = 0
        custom_weights = ScoreWeights(
            component=1.0,
            model=0.0,
            pipeline=0.0,
            agent=0.0,
        )

        node = ScoreComposerNode(weights=custom_weights)
        result = await node.execute(state)

        # Should be 100% since only component matters
        assert result["overall_health_score"] == 100.0
        assert result["health_grade"] == "A"


class TestDegradedModelNullErrorRate:
    """Regression (#952): a degraded/unhealthy model whose ``error_rate`` is
    UNMEASURED (``None``, as the real ml_model_health_dashboard adapter passes
    through) must NOT crash the composer's diagnosis (the ``None > 0.1``
    TypeError in ``_analyze_model_health``). The composite must still COMPLETE
    with a real, non-null model score — never collapse to a failed/unknown
    composite.

    This is the pure-logic half of PR #948's
    ``test_full_health_score_degraded_model_with_null_error_rate_no_crash``: it
    is kept in the unit lane (direct ``ScoreComposerNode``, no full graph) while
    the real-graph version moves to ``tests/integration/`` (issue #952), so the
    serviceless unit shard never instantiates the real agent's hanging
    observability/memory clients.
    """

    @staticmethod
    def _unmeasured_model(model_id: str, status: str) -> dict:
        """A model row matching the real adapter: ``status`` is sourced; every
        numeric sub-field (accuracy/error_rate/...) is UNMEASURED (``None``),
        never a fabricated 0."""
        return {
            "model_id": model_id,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "auc_roc": None,
            "prediction_latency_p50_ms": None,
            "prediction_latency_p99_ms": None,
            "predictions_last_24h": None,
            "error_rate": None,
            "status": status,
        }

    @pytest.mark.asyncio
    async def test_degraded_model_null_error_rate_does_not_crash(self):
        state = {
            "query": "",
            "check_scope": "full",
            "component_statuses": [],
            "component_health_score": 1.0,
            "component_health_measured": True,
            # model dimension measured with a LOW score (< 0.8) so
            # _analyze_model_health iterates the models and reaches the
            # error_rate comparison that historically crashed on None.
            "model_metrics": [
                self._unmeasured_model("m1", "degraded"),
                self._unmeasured_model("m2", "unhealthy"),
            ],
            "model_health_score": 0.25,  # 1 degraded (0.5) + 1 unhealthy (0.0) / 2
            "model_health_measured": True,
            "pipeline_statuses": [],
            "agent_statuses": [],
            "overall_health_score": None,
            "health_grade": None,
            "critical_issues": None,
            "warnings": None,
            "health_summary": None,
            "total_latency_ms": 0,
            "timestamp": "",
            "errors": [],
            "status": "checking",
        }

        node = ScoreComposerNode()
        result = await node.execute(state)

        # Diagnosis did NOT crash on the None error_rate: the run COMPLETED,
        # not the execute() except-branch (which would set status="failed").
        assert result["status"] == "completed"
        assert not any(
            (e or {}).get("node") == "score_composer" for e in (result.get("errors") or [])
        ), f"score_composer raised on a null error_rate: {result.get('errors')}"
        # The model dimension stays a REAL, non-null measurement.
        assert result["model_health_score"] == 0.25
        # Two measured dims (component + model) -> honest partial provenance.
        assert result["data_provenance"] == "partial"
        assert result["health_grade"] in {"A", "B", "C", "D", "F"}
