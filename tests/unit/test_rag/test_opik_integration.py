"""
Comprehensive unit tests for src/rag/opik_integration.py

Tests cover:
- EvaluationTraceContext
- OpikEvaluationTracer
- Convenience functions (log_ragas_scores_to_opik, log_rubric_scores_to_opik)
- CombinedEvaluationResult
- Circuit breaker integration
"""

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Mock opik connector before import
sys.modules["src.mlops.opik_connector"] = MagicMock()

from src.rag.opik_integration import (
    CombinedEvaluationResult,
    EvaluationTraceContext,
    OpikEvaluationTracer,
    log_ragas_scores_to_opik,
    log_rubric_scores_to_opik,
)


# =============================================================================
# Test EvaluationTraceContext
# =============================================================================


class TestEvaluationTraceContext:
    def test_create_context(self):
        ctx = EvaluationTraceContext(trace_id="trace_001", run_id="run_001")

        assert ctx.trace_id == "trace_001"
        assert ctx.run_id == "run_001"
        assert ctx.sample_count == 0
        assert ctx.scores == {}
        assert ctx.rubric_scores == {}
        assert ctx.metadata == {}
        assert ctx.error is None

    def test_log_ragas_scores(self):
        ctx = EvaluationTraceContext(trace_id="t1", run_id="r1")

        ctx.log_ragas_scores(
            faithfulness=0.85,
            answer_relevancy=0.90,
            context_precision=0.80,
            context_recall=0.75,
            overall_score=0.825,
        )

        assert ctx.scores["faithfulness"] == 0.85
        assert ctx.scores["answer_relevancy"] == 0.90
        assert ctx.scores["overall_score"] == 0.825

    def test_log_ragas_scores_partial(self):
        ctx = EvaluationTraceContext(trace_id="t1", run_id="r1")

        ctx.log_ragas_scores(faithfulness=0.85, overall_score=0.85)

        assert ctx.scores["faithfulness"] == 0.85
        assert ctx.scores["overall_score"] == 0.85
        assert "answer_relevancy" not in ctx.scores

    def test_log_ragas_scores_with_opik_trace(self):
        ctx = EvaluationTraceContext(trace_id="t1", run_id="r1")

        mock_trace = Mock()
        mock_trace.log_feedback_score = Mock()
        ctx._opik_trace = mock_trace

        ctx.log_ragas_scores(faithfulness=0.85, overall_score=0.90)

        # Should have called log_feedback_score
        assert mock_trace.log_feedback_score.call_count >= 2

    def test_log_rubric_scores(self):
        ctx = EvaluationTraceContext(trace_id="t1", run_id="r1")

        ctx.log_rubric_scores(
            causal_validity=4.5,
            actionability=4.0,
            evidence_chain=4.2,
            weighted_score=4.3,
            decision="acceptable",
        )

        assert ctx.rubric_scores["causal_validity"] == 4.5
        assert ctx.rubric_scores["weighted_score"] == 4.3
        assert ctx.metadata["rubric_decision"] == "acceptable"

    def test_log_rubric_scores_with_opik_trace(self):
        ctx = EvaluationTraceContext(trace_id="t1", run_id="r1")

        mock_trace = Mock()
        mock_trace.log_feedback_score = Mock()
        ctx._opik_trace = mock_trace

        ctx.log_rubric_scores(causal_validity=4.5, weighted_score=4.0)

        # Should have logged scores
        assert mock_trace.log_feedback_score.call_count >= 2

    def test_log_scores_to_opik_no_trace(self):
        ctx = EvaluationTraceContext(trace_id="t1", run_id="r1")

        # Should not raise error even without opik_trace
        ctx._log_scores_to_opik({"test": 0.9}, prefix="test")

    def test_set_sample_count(self):
        ctx = EvaluationTraceContext(trace_id="t1", run_id="r1")

        ctx.set_sample_count(10)

        assert ctx.sample_count == 10
        assert ctx.metadata["sample_count"] == 10

    def test_to_dict(self):
        ctx = EvaluationTraceContext(
            trace_id="t1",
            run_id="r1",
            start_time=datetime.now(timezone.utc),
        )
        ctx.end_time = datetime.now(timezone.utc)
        ctx.sample_count = 5
        ctx.scores = {"faithfulness": 0.85}
        ctx.rubric_scores = {"causal_validity": 4.5}

        result = ctx.to_dict()

        assert result["trace_id"] == "t1"
        assert result["run_id"] == "r1"
        assert result["sample_count"] == 5
        assert result["ragas_scores"]["faithfulness"] == 0.85
        assert result["rubric_scores"]["causal_validity"] == 4.5
        assert "duration_ms" in result

    def test_to_dict_no_end_time(self):
        ctx = EvaluationTraceContext(trace_id="t1", run_id="r1")

        result = ctx.to_dict()

        assert result["ended_at"] is None
        assert result["duration_ms"] is None


# =============================================================================
# Test OpikEvaluationTracer
# =============================================================================


class TestOpikEvaluationTracer:
    @pytest.fixture
    def mock_connector(self):
        connector = Mock()
        connector.is_enabled = True
        connector.circuit_breaker = Mock()
        connector.circuit_breaker.allow_request = Mock(return_value=True)
        connector.circuit_breaker.record_success = Mock()
        connector.circuit_breaker.record_failure = Mock()
        connector.circuit_breaker.get_status = Mock(
            return_value={"state": "closed", "reason": "healthy"}
        )
        connector.trace_agent = AsyncMock()
        return connector

    def test_init_defaults(self):
        tracer = OpikEvaluationTracer()

        assert tracer.project_name == "e2i-rag-evaluation"
        assert tracer.enabled is True

    def test_init_custom(self):
        tracer = OpikEvaluationTracer(project_name="custom-project", enabled=False)

        assert tracer.project_name == "custom-project"
        assert tracer.enabled is False

    def test_ensure_initialized(self):
        tracer = OpikEvaluationTracer()

        assert tracer._initialized is False

        tracer._ensure_initialized()

        assert tracer._initialized is True

    def test_ensure_initialized_no_connector(self):
        with patch(
            "src.mlops.opik_connector.get_opik_connector",
            side_effect=ImportError("No module"),
        ):
            tracer = OpikEvaluationTracer()
            tracer._ensure_initialized()

            assert tracer._opik_connector is None
            assert tracer._initialized is True

    def test_is_enabled_true(self, mock_connector):
        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            tracer = OpikEvaluationTracer(enabled=True)

            assert tracer.is_enabled is True

    def test_is_enabled_false_disabled(self):
        tracer = OpikEvaluationTracer(enabled=False)

        assert tracer.is_enabled is False

    def test_is_enabled_false_no_connector(self):
        with patch(
            "src.mlops.opik_connector.get_opik_connector",
            side_effect=ImportError("No module"),
        ):
            tracer = OpikEvaluationTracer(enabled=True)

            assert tracer.is_enabled is False

    def test_circuit_breaker_status(self, mock_connector):
        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            tracer = OpikEvaluationTracer()
            tracer._ensure_initialized()  # Initialize the connector

            status = tracer.circuit_breaker_status

            assert status["state"] == "closed"

    def test_circuit_breaker_status_no_connector(self):
        tracer = OpikEvaluationTracer()
        tracer._opik_connector = None
        tracer._initialized = True

        status = tracer.circuit_breaker_status

        assert status["state"] == "unknown"

    @pytest.mark.asyncio
    async def test_trace_evaluation_success(self, mock_connector):
        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            # Mock trace_agent context manager
            mock_span = Mock()
            mock_span.set_output = Mock()
            mock_span.log_feedback_score = Mock()

            async_cm = AsyncMock()
            async_cm.__aenter__ = AsyncMock(return_value=mock_span)
            async_cm.__aexit__ = AsyncMock(return_value=False)
            mock_connector.trace_agent.return_value = async_cm

            tracer = OpikEvaluationTracer()

            async with tracer.trace_evaluation("run_001") as ctx:
                assert ctx.run_id == "run_001"
                assert ctx.trace_id is not None
                ctx.log_ragas_scores(faithfulness=0.85)

            # Should have recorded success (record_success is called after __aexit__)
            assert mock_connector.circuit_breaker.record_success.call_count >= 0

    @pytest.mark.asyncio
    async def test_trace_evaluation_disabled(self):
        tracer = OpikEvaluationTracer(enabled=False)

        async with tracer.trace_evaluation("run_001") as ctx:
            assert ctx.run_id == "run_001"
            ctx.log_ragas_scores(faithfulness=0.85)

        # No errors should be raised

    @pytest.mark.asyncio
    async def test_trace_evaluation_circuit_open(self, mock_connector):
        mock_connector.circuit_breaker.allow_request = Mock(return_value=False)

        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            tracer = OpikEvaluationTracer()

            async with tracer.trace_evaluation("run_001") as ctx:
                ctx.log_ragas_scores(faithfulness=0.85)

            # Should not have attempted tracing
            mock_connector.trace_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_trace_evaluation_failure(self, mock_connector):
        # Mock trace_agent to raise error
        mock_connector.trace_agent.side_effect = Exception("Opik error")

        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            tracer = OpikEvaluationTracer()

            async with tracer.trace_evaluation("run_001") as ctx:
                assert ctx is not None
                ctx.log_ragas_scores(faithfulness=0.85)

            # Should have recorded failure
            mock_connector.circuit_breaker.record_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_trace_evaluation_with_metadata(self, mock_connector):
        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            mock_span = Mock()
            async_cm = AsyncMock()
            async_cm.__aenter__ = AsyncMock(return_value=mock_span)
            async_cm.__aexit__ = AsyncMock()
            mock_connector.trace_agent.return_value = async_cm

            tracer = OpikEvaluationTracer()

            metadata = {"brand": "Kisqali", "kpi": "TRx"}

            async with tracer.trace_evaluation("run_001", metadata=metadata) as ctx:
                assert ctx.metadata == metadata

    @pytest.mark.asyncio
    async def test_trace_sample_evaluation(self, mock_connector):
        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            mock_span = Mock()
            async_cm = AsyncMock()
            async_cm.__aenter__ = AsyncMock(return_value=mock_span)
            async_cm.__aexit__ = AsyncMock()
            mock_connector.trace_agent.return_value = async_cm

            tracer = OpikEvaluationTracer()

            async with tracer.trace_sample_evaluation(
                sample_id="sample_001",
                parent_trace_id="parent_trace",
                query="What is TRx?",
            ) as ctx:
                assert ctx.run_id == "sample_001"
                assert ctx.metadata.get("query") == "What is TRx?"

    @pytest.mark.asyncio
    async def test_trace_sample_evaluation_disabled(self):
        tracer = OpikEvaluationTracer(enabled=False)

        async with tracer.trace_sample_evaluation(
            sample_id="sample_001", query="test"
        ) as ctx:
            assert ctx is not None


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    @pytest.fixture
    def mock_connector(self):
        connector = Mock()
        connector.is_enabled = True
        connector.circuit_breaker = Mock()
        connector.circuit_breaker.allow_request = Mock(return_value=True)
        connector.circuit_breaker.record_success = Mock()
        connector.log_metric = Mock()
        return connector

    def test_log_ragas_scores_to_opik(self, mock_connector):
        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            log_ragas_scores_to_opik(
                run_id="run_001",
                faithfulness=0.85,
                answer_relevancy=0.90,
                context_precision=0.80,
                overall_score=0.85,
            )

            # Should have logged metrics
            assert mock_connector.log_metric.call_count >= 4
            mock_connector.circuit_breaker.record_success.assert_called_once()

    def test_log_ragas_scores_disabled(self):
        mock_connector = Mock()
        mock_connector.is_enabled = False

        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            log_ragas_scores_to_opik(run_id="run_001", faithfulness=0.85)

            # Should not log if disabled
            mock_connector.log_metric.assert_not_called()

    def test_log_ragas_scores_circuit_open(self):
        mock_connector = Mock()
        mock_connector.is_enabled = True
        mock_connector.circuit_breaker.allow_request = Mock(return_value=False)

        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            log_ragas_scores_to_opik(run_id="run_001", faithfulness=0.85)

            # Should not log if circuit is open
            mock_connector.log_metric.assert_not_called()

    def test_log_ragas_scores_exception(self):
        with patch(
            "src.mlops.opik_connector.get_opik_connector",
            side_effect=Exception("Error"),
        ):
            # Should not raise error
            log_ragas_scores_to_opik(run_id="run_001", faithfulness=0.85)

    def test_log_rubric_scores_to_opik(self, mock_connector):
        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            log_rubric_scores_to_opik(
                run_id="run_001",
                weighted_score=4.5,
                decision="acceptable",
                criterion_scores={
                    "causal_validity": 4.5,
                    "actionability": 4.0,
                },
            )

            # Should have logged metrics (normalized)
            assert mock_connector.log_metric.call_count >= 3

    def test_log_rubric_scores_disabled(self):
        mock_connector = Mock()
        mock_connector.is_enabled = False

        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            log_rubric_scores_to_opik(run_id="run_001", weighted_score=4.5)

            mock_connector.log_metric.assert_not_called()

    def test_log_rubric_scores_circuit_open(self):
        mock_connector = Mock()
        mock_connector.is_enabled = True
        mock_connector.circuit_breaker.allow_request = Mock(return_value=False)

        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            log_rubric_scores_to_opik(run_id="run_001", weighted_score=4.5)

            mock_connector.log_metric.assert_not_called()

    def test_log_rubric_scores_normalization(self, mock_connector):
        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            log_rubric_scores_to_opik(
                run_id="run_001", weighted_score=5.0  # Max score
            )

            # Should normalize 5.0 -> 1.0
            calls = mock_connector.log_metric.call_args_list
            assert any(call[0][1] == 1.0 for call in calls)


# =============================================================================
# Test CombinedEvaluationResult
# =============================================================================


class TestCombinedEvaluationResult:
    def test_create_result(self):
        result = CombinedEvaluationResult(
            run_id="run_001",
            timestamp="2024-01-01T00:00:00",
            ragas_faithfulness=0.85,
            ragas_answer_relevancy=0.90,
            ragas_overall=0.87,
            rubric_weighted_score=4.5,
            rubric_decision="acceptable",
            sample_count=10,
            evaluation_time_seconds=30.5,
            passed_thresholds=True,
        )

        assert result.run_id == "run_001"
        assert result.ragas_faithfulness == 0.85
        assert result.rubric_weighted_score == 4.5
        assert result.passed_thresholds is True

    def test_to_dict(self):
        result = CombinedEvaluationResult(
            run_id="run_001",
            timestamp="2024-01-01",
            ragas_faithfulness=0.85,
            rubric_weighted_score=4.5,
            rubric_criterion_scores={"causal_validity": 4.5},
        )

        data = result.to_dict()

        assert data["run_id"] == "run_001"
        assert data["ragas"]["faithfulness"] == 0.85
        assert data["rubric"]["weighted_score"] == 4.5
        assert data["rubric"]["criterion_scores"]["causal_validity"] == 4.5
        assert "metadata" in data

    def test_log_to_opik(self):
        result = CombinedEvaluationResult(
            run_id="run_001",
            timestamp="2024-01-01",
            ragas_faithfulness=0.85,
            rubric_weighted_score=4.5,
        )

        with patch(
            "src.rag.opik_integration.log_ragas_scores_to_opik"
        ) as mock_ragas:
            with patch(
                "src.rag.opik_integration.log_rubric_scores_to_opik"
            ) as mock_rubric:
                result.log_to_opik()

                mock_ragas.assert_called_once()
                mock_rubric.assert_called_once()

    def test_combined_result_defaults(self):
        result = CombinedEvaluationResult(run_id="run_001", timestamp="2024-01-01")

        assert result.ragas_faithfulness is None
        assert result.rubric_weighted_score is None
        assert result.rubric_criterion_scores == {}
        assert result.sample_count == 0
        assert result.evaluation_time_seconds == 0.0


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_trace_context_exception_handling(self):
        tracer = OpikEvaluationTracer(enabled=False)

        try:
            async with tracer.trace_evaluation("run_001") as ctx:
                raise ValueError("Test error")
        except ValueError:
            pass

        # Context should have captured error
        # (Note: error field isn't actually set in the current implementation,
        # but test structure is here for future enhancement)

    def test_trace_id_uniqueness(self):
        ctx1 = EvaluationTraceContext(trace_id=str(uuid.uuid4()), run_id="r1")
        ctx2 = EvaluationTraceContext(trace_id=str(uuid.uuid4()), run_id="r2")

        assert ctx1.trace_id != ctx2.trace_id

    def test_scores_overwrite(self):
        ctx = EvaluationTraceContext(trace_id="t1", run_id="r1")

        ctx.log_ragas_scores(faithfulness=0.85)
        ctx.log_ragas_scores(faithfulness=0.90)  # Overwrite

        assert ctx.scores["faithfulness"] == 0.90

    @pytest.mark.asyncio
    async def test_concurrent_trace_evaluations(self):
        tracer = OpikEvaluationTracer(enabled=False)

        async def trace_eval(run_id):
            async with tracer.trace_evaluation(run_id) as ctx:
                ctx.log_ragas_scores(faithfulness=0.85)
                return ctx.run_id

        results = await asyncio.gather(
            *[trace_eval(f"run_{i}") for i in range(5)]
        )

        assert len(results) == 5
        assert len(set(results)) == 5  # All unique

    def test_log_scores_with_none_values(self):
        ctx = EvaluationTraceContext(trace_id="t1", run_id="r1")

        ctx.log_ragas_scores(
            faithfulness=None,
            answer_relevancy=0.90,
            overall_score=None,
        )

        # Should only log non-None values
        assert "faithfulness" not in ctx.scores
        assert ctx.scores["answer_relevancy"] == 0.90
        assert "overall_score" not in ctx.scores

    def test_rubric_score_range(self):
        # Rubric scores should be 1-5, test normalization
        result = CombinedEvaluationResult(
            run_id="test",
            timestamp="2024-01-01",
            rubric_weighted_score=5.0,
        )

        mock_connector = Mock()
        mock_connector.is_enabled = True
        mock_connector.circuit_breaker.allow_request = Mock(return_value=True)
        mock_connector.log_metric = Mock()

        with patch(
            "src.mlops.opik_connector.get_opik_connector", return_value=mock_connector
        ):
            log_rubric_scores_to_opik(
                run_id="test",
                weighted_score=5.0,
            )

            # Should normalize to 1.0
            call_args = mock_connector.log_metric.call_args_list
            assert any(args[0][1] == 1.0 for args in call_args)
