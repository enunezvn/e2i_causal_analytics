"""Tests for F-010-backend (#429): env-gated fail-closed on agent ImportError.

Each route handler — gaps, segments, resource_optimizer, feedback,
health_score — previously caught ``ImportError`` from agent module imports
and silently returned a mock response with HTTP 200. These tests pin the
new behavior:

* When ``E2I_REQUIRE_AGENT_IMPORT=1`` (or ENVIRONMENT=production), each
  endpoint raises ``HTTPException(503)`` instead of returning fabricated data.
* When the env-gate is OFF (development), the mock-fallback path is still
  reachable AND still emits the ``warnings: ["Using mock data - ... not
  available"]`` field that Agent 3 renders.
"""

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
from fastapi import HTTPException

# =============================================================================
# Helpers
# =============================================================================


def _fail_closed_env():
    """Env dict that forces fail-closed mode."""
    return {"E2I_REQUIRE_AGENT_IMPORT": "1", "ENVIRONMENT": "production"}


def _mock_allowed_env():
    """Env dict that explicitly enables mock-fallback (dev mode)."""
    return {"E2I_REQUIRE_AGENT_IMPORT": "0", "ENVIRONMENT": "development"}


# =============================================================================
# Codex iter-1 M1: public route preserves 503 (does not mask as 500)
# =============================================================================


class TestRoutePreserves503:
    """Public route wrappers must NOT mask the 503 raised by ``_execute_*``
    helpers as a 500.
    """

    @pytest.mark.asyncio
    async def test_public_gap_route_preserves_503(self):
        """run_gap_analysis bubbles up the 503 (not 500) when helper fails."""
        from fastapi import BackgroundTasks

        from src.api.routes.gaps import (
            GapType,
            RunGapAnalysisRequest,
            run_gap_analysis,
        )

        request = RunGapAnalysisRequest(
            query="Q",
            brand="kisqali",
            metrics=["trx"],
            segments=["region"],
            gap_type=GapType.ALL,
        )
        user = {"user_id": "u", "role": "analyst"}
        with patch.dict("os.environ", _fail_closed_env(), clear=False):
            with patch(
                "src.agents.gap_analyzer.graph.create_gap_analyzer_graph",
                side_effect=ImportError("module missing"),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await run_gap_analysis(
                        request,
                        BackgroundTasks(),
                        async_mode=False,
                        user=user,
                    )
                # Must remain 503 — not collapsed to 500
                assert exc_info.value.status_code == 503


# =============================================================================
# gaps.py
# =============================================================================


class TestGapsFailClosed:
    @pytest.mark.asyncio
    async def test_gaps_raises_503_when_fail_closed(self):
        """_execute_gap_analysis raises 503 under fail-closed env when import fails."""
        from src.api.routes.gaps import (
            GapType,
            RunGapAnalysisRequest,
            _execute_gap_analysis,
        )

        request = RunGapAnalysisRequest(
            query="Q",
            brand="kisqali",
            metrics=["trx"],
            segments=["region"],
            gap_type=GapType.ALL,
        )
        with patch.dict("os.environ", _fail_closed_env(), clear=False):
            with patch(
                "src.agents.gap_analyzer.graph.create_gap_analyzer_graph",
                side_effect=ImportError("module missing"),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await _execute_gap_analysis(request)
                assert exc_info.value.status_code == 503
                assert exc_info.value.detail["agent"] == "Gap Analyzer"

    @pytest.mark.asyncio
    async def test_gaps_falls_back_to_mock_when_allowed(self):
        """When env explicitly allows mock, response carries warnings[] payload."""
        from src.api.routes.gaps import (
            AnalysisStatus,
            GapType,
            RunGapAnalysisRequest,
            _execute_gap_analysis,
        )

        request = RunGapAnalysisRequest(
            query="Q",
            brand="kisqali",
            metrics=["trx"],
            segments=["region"],
            gap_type=GapType.ALL,
        )
        with patch.dict("os.environ", _mock_allowed_env(), clear=False):
            with patch(
                "src.agents.gap_analyzer.graph.create_gap_analyzer_graph",
                side_effect=ImportError("module missing"),
            ):
                response = await _execute_gap_analysis(request)

        assert response.status == AnalysisStatus.COMPLETED
        assert len(response.warnings) > 0
        assert "mock data" in response.warnings[0].lower()


# =============================================================================
# segments.py
# =============================================================================


class TestSegmentsFailClosed:
    # Clinical-HTE rebuild: _execute_segment_analysis now loads the curated
    # patient_journeys frame SERVER-SIDE before the agent import-guard. These
    # tests target the import-guard mechanism, so they use a CURATED request
    # (default treatment_arm -> persistent_180d) and stub the loader so the
    # ImportError path is reached (a placeholder treatment/outcome would 400 at
    # the allowlist first).
    @staticmethod
    def _patch_loader():
        frame = pd.DataFrame(
            {
                "treatment_arm": [i % 2 for i in range(120)],
                "persistent_180d": [(i + 1) % 2 for i in range(120)],
            }
        )
        return patch(
            "src.api.routes.segments._load_segment_hte_frame",
            new=AsyncMock(return_value=frame),
        )

    @pytest.mark.asyncio
    async def test_segments_raises_503_when_fail_closed(self):
        from src.api.routes.segments import (
            QuestionType,
            RunSegmentAnalysisRequest,
            _execute_segment_analysis,
        )

        request = RunSegmentAnalysisRequest(
            query="Q",
            question_type=QuestionType.EFFECT_HETEROGENEITY,
        )
        with patch.dict("os.environ", _fail_closed_env(), clear=False):
            with self._patch_loader():
                with patch(
                    "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
                    side_effect=ImportError("module missing"),
                ):
                    with pytest.raises(HTTPException) as exc_info:
                        await _execute_segment_analysis(request)
                    assert exc_info.value.status_code == 503
                    assert exc_info.value.detail["agent"] == "Heterogeneous Optimizer"

    @pytest.mark.asyncio
    async def test_segments_mock_fallback_preserves_warnings(self):
        from src.api.routes.segments import (
            QuestionType,
            RunSegmentAnalysisRequest,
            _execute_segment_analysis,
        )

        request = RunSegmentAnalysisRequest(
            query="Q",
            question_type=QuestionType.EFFECT_HETEROGENEITY,
        )
        with patch.dict("os.environ", _mock_allowed_env(), clear=False):
            with self._patch_loader():
                with patch(
                    "src.agents.heterogeneous_optimizer.graph.create_heterogeneous_optimizer_graph",
                    side_effect=ImportError("module missing"),
                ):
                    response = await _execute_segment_analysis(request)
        assert len(response.warnings) > 0
        assert "mock data" in response.warnings[0].lower()


# =============================================================================
# resource_optimizer.py
# =============================================================================


class TestResourceOptimizerFailClosed:
    @pytest.mark.asyncio
    async def test_resource_optimizer_raises_503_when_fail_closed(self):
        from src.api.routes.resource_optimizer import (
            AllocationTarget,
            OptimizationObjective,
            ResourceType,
            RunOptimizationRequest,
            _execute_optimization,
        )

        request = RunOptimizationRequest(
            query="Q",
            resource_type=ResourceType.BUDGET,
            allocation_targets=[
                AllocationTarget(
                    entity_id="e1",
                    entity_type="hcp",
                    current_allocation=100.0,
                    expected_response=1.2,
                )
            ],
            objective=OptimizationObjective.MAXIMIZE_OUTCOME,
        )
        with patch.dict("os.environ", _fail_closed_env(), clear=False):
            with patch(
                "src.agents.resource_optimizer.graph.build_resource_optimizer_graph",
                side_effect=ImportError("module missing"),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await _execute_optimization(request)
                assert exc_info.value.status_code == 503
                assert exc_info.value.detail["agent"] == "Resource Optimizer"

    @pytest.mark.asyncio
    async def test_resource_optimizer_mock_fallback_preserves_warnings(self):
        from src.api.routes.resource_optimizer import (
            AllocationTarget,
            OptimizationObjective,
            ResourceType,
            RunOptimizationRequest,
            _execute_optimization,
        )

        request = RunOptimizationRequest(
            query="Q",
            resource_type=ResourceType.BUDGET,
            allocation_targets=[
                AllocationTarget(
                    entity_id="e1",
                    entity_type="hcp",
                    current_allocation=100.0,
                    expected_response=1.2,
                )
            ],
            objective=OptimizationObjective.MAXIMIZE_OUTCOME,
        )
        with patch.dict("os.environ", _mock_allowed_env(), clear=False):
            with patch(
                "src.agents.resource_optimizer.graph.build_resource_optimizer_graph",
                side_effect=ImportError("module missing"),
            ):
                response = await _execute_optimization(request)
        assert len(response.warnings) > 0
        assert "mock data" in response.warnings[0].lower()


# =============================================================================
# feedback.py (learning cycle)
# =============================================================================


class TestFeedbackLearningFailClosed:
    @pytest.mark.asyncio
    async def test_feedback_learning_raises_503_when_fail_closed(self):
        from src.api.routes.feedback import (
            RunLearningRequest,
            _execute_learning_cycle,
        )

        request = RunLearningRequest()
        with patch.dict("os.environ", _fail_closed_env(), clear=False):
            with patch(
                "src.agents.feedback_learner.graph.build_feedback_learner_graph",
                side_effect=ImportError("module missing"),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await _execute_learning_cycle(request)
                assert exc_info.value.status_code == 503
                assert exc_info.value.detail["agent"] == "Feedback Learner"

    @pytest.mark.asyncio
    async def test_feedback_learning_mock_fallback_preserves_warnings(self):
        from src.api.routes.feedback import (
            RunLearningRequest,
            _execute_learning_cycle,
        )

        request = RunLearningRequest()
        with patch.dict("os.environ", _mock_allowed_env(), clear=False):
            with patch(
                "src.agents.feedback_learner.graph.build_feedback_learner_graph",
                side_effect=ImportError("module missing"),
            ):
                response = await _execute_learning_cycle(request)
        assert len(response.warnings) > 0
        assert "mock data" in response.warnings[0].lower()


# =============================================================================
# health_score.py
# =============================================================================


class TestHealthScoreFailClosed:
    """Route imports ``HealthScoreAgent`` lazily inside the function, so we
    sabotage ``builtins.__import__`` for that path. Mirrors the established
    test pattern at tests/unit/test_api/test_routes/test_gaps.py::
    test_gap_health_agent_unavailable.
    """

    @staticmethod
    def _patched_import():
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "src.agents.health_score" or name.startswith("src.agents.health_score."):
                raise ImportError("module 'src.agents.health_score' unavailable")
            return real_import(name, *args, **kwargs)

        return fake_import

    @pytest.mark.asyncio
    async def test_health_score_raises_503_when_fail_closed(self):
        from src.api.routes.health_score import (
            CheckScope,
            _execute_health_check,
        )

        with patch.dict("os.environ", _fail_closed_env(), clear=False):
            with patch("builtins.__import__", side_effect=self._patched_import()):
                with pytest.raises(HTTPException) as exc_info:
                    await _execute_health_check(CheckScope.FULL)
                assert exc_info.value.status_code == 503
                assert exc_info.value.detail["agent"] == "Health Score"

    @pytest.mark.asyncio
    async def test_health_score_mock_fallback_preserves_warnings(self):
        from src.api.routes.health_score import (
            CheckScope,
            _execute_health_check,
        )

        with patch.dict("os.environ", _mock_allowed_env(), clear=False):
            with patch("builtins.__import__", side_effect=self._patched_import()):
                response = await _execute_health_check(CheckScope.FULL)
        assert len(response.warnings) > 0
        assert "mock data" in response.warnings[0].lower()
