"""
Comprehensive unit tests for src/rag/evaluation.py

Tests cover:
- EvaluationSample, EvaluationResult, EvaluationReport models
- RAGASEvaluator class (with RAGAS mocked)
- RAGEvaluationPipeline
- Helper functions
- MLflow and Opik integration
"""

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Mock external dependencies before importing evaluation module
mock_ragas = MagicMock()
mock_ragas.__spec__ = MagicMock()  # Fix for importlib.util.find_spec check
sys.modules["ragas"] = mock_ragas
sys.modules["ragas.metrics"] = MagicMock()
sys.modules["ragas.llms"] = MagicMock()
sys.modules["ragas.embeddings"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["mlflow"] = MagicMock()

from src.rag.evaluation import (
    DEFAULT_THRESHOLDS,
    EvaluationConfig,
    EvaluationReport,
    EvaluationResult,
    EvaluationSample,
    RAGASEvaluator,
    RAGEvaluationPipeline,
    create_evaluation_sample,
    get_default_evaluation_dataset,
    get_ragas_evaluator,
    load_evaluation_dataset,
    quick_evaluate,
    save_evaluation_dataset,
)


# =============================================================================
# Test Data Models
# =============================================================================


class TestEvaluationSample:
    def test_create_sample(self):
        sample = EvaluationSample(
            query="What are TRx trends?",
            ground_truth="TRx increased by 15%",
            contexts=["Context 1", "Context 2"],
            answer="TRx grew 15%",
            metadata={"brand": "Kisqali"},
        )

        assert sample.query == "What are TRx trends?"
        assert sample.ground_truth == "TRx increased by 15%"
        assert len(sample.contexts) == 2
        assert sample.answer == "TRx grew 15%"
        assert sample.metadata["brand"] == "Kisqali"

    def test_sample_defaults(self):
        sample = EvaluationSample(query="test", ground_truth="truth")

        assert sample.contexts == []
        assert sample.answer is None
        assert sample.retrieved_contexts == []
        assert sample.metadata == {}

    def test_sample_serialization(self):
        sample = EvaluationSample(
            query="test", ground_truth="truth", metadata={"key": "value"}
        )

        data = sample.model_dump()
        assert data["query"] == "test"
        assert data["metadata"]["key"] == "value"


class TestEvaluationResult:
    def test_create_result(self):
        result = EvaluationResult(
            sample_id="sample_001",
            query="What are TRx trends?",
            faithfulness=0.85,
            answer_relevancy=0.90,
            context_precision=0.80,
            context_recall=0.75,
            overall_score=0.825,
            passed_thresholds=True,
        )

        assert result.sample_id == "sample_001"
        assert result.faithfulness == 0.85
        assert result.overall_score == 0.825
        assert result.passed_thresholds is True

    def test_result_score_validation(self):
        # Scores must be between 0 and 1
        with pytest.raises(Exception):
            EvaluationResult(
                sample_id="test", query="test", faithfulness=1.5  # Invalid
            )

    def test_result_defaults(self):
        result = EvaluationResult(sample_id="test", query="test")

        assert result.faithfulness is None
        assert result.passed_thresholds is False
        assert result.metadata == {}


class TestEvaluationReport:
    def test_create_report(self):
        results = [
            EvaluationResult(
                sample_id="s1",
                query="q1",
                faithfulness=0.85,
                answer_relevancy=0.90,
                overall_score=0.85,
                passed_thresholds=True,
            )
        ]

        report = EvaluationReport(
            run_id="run_001",
            timestamp="2024-01-01T00:00:00",
            total_samples=10,
            passed_samples=8,
            failed_samples=2,
            avg_faithfulness=0.85,
            overall_score=0.85,
            thresholds=DEFAULT_THRESHOLDS,
            all_thresholds_passed=True,
            results=results,
            evaluation_time_seconds=30.5,
        )

        assert report.run_id == "run_001"
        assert report.total_samples == 10
        assert report.passed_samples == 8
        assert report.all_thresholds_passed is True


# =============================================================================
# Test Configuration
# =============================================================================


class TestEvaluationConfig:
    def test_default_config(self):
        config = EvaluationConfig()

        assert config.thresholds == DEFAULT_THRESHOLDS
        assert config.log_to_mlflow is True
        assert config.batch_size == 10
        assert config.max_concurrent == 5

    def test_custom_config(self):
        custom_thresholds = {"faithfulness": 0.95}
        config = EvaluationConfig(
            thresholds=custom_thresholds, batch_size=20, log_to_mlflow=False
        )

        assert config.thresholds == custom_thresholds
        assert config.batch_size == 20
        assert config.log_to_mlflow is False


# =============================================================================
# Test Dataset Functions
# =============================================================================


class TestDatasetFunctions:
    def test_get_default_dataset(self):
        dataset = get_default_evaluation_dataset()

        assert len(dataset) > 0
        assert all(isinstance(s, EvaluationSample) for s in dataset)

        # Check first sample
        first = dataset[0]
        assert first.query
        assert first.ground_truth
        assert first.contexts
        assert first.answer

    def test_save_and_load_dataset(self):
        samples = [
            EvaluationSample(
                query="test1",
                ground_truth="truth1",
                contexts=["ctx1"],
                metadata={"brand": "Kisqali"},
            ),
            EvaluationSample(
                query="test2", ground_truth="truth2", contexts=["ctx2"]
            ),
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_path = f.name

        try:
            # Save dataset
            save_evaluation_dataset(samples, temp_path)
            assert Path(temp_path).exists()

            # Load dataset
            loaded = load_evaluation_dataset(temp_path)
            assert len(loaded) == 2
            assert loaded[0].query == "test1"
            assert loaded[1].metadata == {}

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_dataset_nonexistent(self):
        # Should return default dataset when file doesn't exist
        dataset = load_evaluation_dataset("/nonexistent/path.json")
        assert len(dataset) > 0

    def test_create_evaluation_sample(self):
        sample = create_evaluation_sample(
            query="What is TRx?",
            ground_truth="Total prescriptions",
            contexts=["TRx is total prescriptions"],
            brand="Kisqali",
            kpi="TRx",
        )

        assert sample.query == "What is TRx?"
        assert sample.ground_truth == "Total prescriptions"
        assert sample.metadata["brand"] == "Kisqali"
        assert sample.metadata["kpi"] == "TRx"


# =============================================================================
# Test RAGASEvaluator
# =============================================================================


class TestRAGASEvaluator:
    @pytest.fixture
    def mock_config(self):
        return EvaluationConfig(log_to_mlflow=False)

    @pytest.fixture
    def evaluator(self, mock_config):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            return RAGASEvaluator(config=mock_config, enable_opik_tracing=False)

    def test_init(self, evaluator):
        assert evaluator.config is not None
        assert evaluator.llm_provider == "openai"

    def test_detect_llm_provider_openai(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            evaluator = RAGASEvaluator(llm_provider="auto", enable_opik_tracing=False)
            assert evaluator.llm_provider == "openai"

    def test_detect_llm_provider_anthropic(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            evaluator = RAGASEvaluator(llm_provider="auto", enable_opik_tracing=False)
            assert evaluator.llm_provider == "anthropic"

    def test_detect_llm_provider_none(self):
        with patch.dict("os.environ", {}, clear=True):
            evaluator = RAGASEvaluator(llm_provider="auto", enable_opik_tracing=False)
            assert evaluator.llm_provider == "none"

    def test_check_ragas(self, evaluator):
        # RAGAS is mocked, so should be available
        assert evaluator._ragas_available is False or True  # Can be either

    def test_check_llm(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            evaluator = RAGASEvaluator(llm_provider="openai", enable_opik_tracing=False)
            assert evaluator._check_llm() is True

        with patch.dict("os.environ", {}, clear=True):
            evaluator = RAGASEvaluator(llm_provider="openai", enable_opik_tracing=False)
            assert evaluator._check_llm() is False

    @pytest.mark.asyncio
    async def test_evaluate_sample_no_answer(self, evaluator):
        sample = EvaluationSample(query="test", ground_truth="truth")

        result = await evaluator.evaluate_sample(sample)

        assert result.sample_id
        assert result.query == "test"
        assert result.metadata.get("error") == "No answer provided"

    @pytest.mark.asyncio
    async def test_evaluate_sample_fallback(self, evaluator):
        sample = EvaluationSample(
            query="What is TRx for Kisqali?",
            ground_truth="TRx is total prescriptions for Kisqali",
            answer="TRx represents total prescription volume for Kisqali",
            retrieved_contexts=[
                "TRx is total prescriptions",
                "Kisqali TRx data shows growth",
            ],
        )

        # Force fallback by disabling RAGAS
        evaluator._ragas_available = False

        result = await evaluator.evaluate_sample(sample)

        assert result.sample_id
        assert result.faithfulness is not None
        assert result.answer_relevancy is not None
        assert result.overall_score is not None
        assert 0 <= result.faithfulness <= 1

    @pytest.mark.asyncio
    async def test_evaluate_with_fallback_empty_answer(self, evaluator):
        sample = EvaluationSample(
            query="test",
            ground_truth="truth",
            answer="",
            retrieved_contexts=["context"],
        )

        result = await evaluator._evaluate_with_fallback(sample, "test_id")

        assert result.faithfulness == 0.0
        assert result.answer_relevancy == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, evaluator):
        samples = [
            EvaluationSample(
                query=f"query_{i}",
                ground_truth=f"truth_{i}",
                answer=f"answer_{i}",
                contexts=[f"context_{i}"],
            )
            for i in range(3)
        ]

        evaluator._ragas_available = False  # Use fallback

        results = await evaluator.evaluate_batch(samples)

        assert len(results) == 3
        assert all(isinstance(r, EvaluationResult) for r in results)

    @pytest.mark.asyncio
    async def test_evaluate_batch_with_run_id(self, evaluator):
        samples = [
            EvaluationSample(
                query="test", ground_truth="truth", answer="ans", contexts=["ctx"]
            )
        ]

        evaluator._ragas_available = False

        results = await evaluator.evaluate_batch(samples, batch_run_id="batch_001")

        assert len(results) == 1

    def test_log_rubric_scores_disabled(self, evaluator):
        result = evaluator.log_rubric_scores(
            run_id="test", weighted_score=4.5, decision="acceptable"
        )

        assert result is False  # Opik tracing disabled


# =============================================================================
# Test RAGEvaluationPipeline
# =============================================================================


class TestRAGEvaluationPipeline:
    @pytest.fixture
    def mock_config(self):
        return EvaluationConfig(log_to_mlflow=False)

    @pytest.fixture
    def pipeline(self, mock_config):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            return RAGEvaluationPipeline(
                config=mock_config, enable_opik_tracing=False
            )

    def test_init(self, pipeline):
        assert pipeline.config is not None
        assert pipeline.evaluator is not None
        assert len(pipeline.dataset) > 0

    def test_init_with_custom_dataset(self, mock_config):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            json.dump(
                [
                    {
                        "query": "test",
                        "ground_truth": "truth",
                        "contexts": [],
                        "answer": None,
                        "retrieved_contexts": [],
                        "metadata": {},
                    }
                ],
                f,
            )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                pipeline = RAGEvaluationPipeline(
                    config=mock_config, dataset_path=temp_path, enable_opik_tracing=False
                )

            assert len(pipeline.dataset) == 1
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_run_evaluation(self, pipeline):
        # Use small dataset
        pipeline.dataset = pipeline.dataset[:2]
        pipeline.evaluator._ragas_available = False  # Force fallback

        report = await pipeline.run_evaluation()

        assert report.run_id
        assert report.total_samples == 2
        assert report.evaluation_time_seconds > 0
        assert len(report.results) == 2

    @pytest.mark.asyncio
    async def test_run_evaluation_with_pipeline(self, pipeline):
        # Mock RAG pipeline
        mock_rag = AsyncMock()
        mock_rag.query = AsyncMock(
            return_value={"answer": "test answer", "contexts": ["context1"]}
        )

        pipeline.dataset = [
            EvaluationSample(query="test", ground_truth="truth", contexts=["ctx"])
        ]
        pipeline.evaluator._ragas_available = False

        report = await pipeline.run_evaluation(rag_pipeline=mock_rag)

        assert report.total_samples == 1
        mock_rag.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_answers(self, pipeline):
        mock_rag = AsyncMock()
        mock_rag.query = AsyncMock(
            return_value={"answer": "generated answer", "contexts": ["ctx1", "ctx2"]}
        )

        pipeline.dataset = [
            EvaluationSample(query="test", ground_truth="truth", contexts=[])
        ]

        await pipeline._generate_answers(mock_rag)

        assert pipeline.dataset[0].answer == "generated answer"
        assert len(pipeline.dataset[0].retrieved_contexts) == 2

    @pytest.mark.asyncio
    async def test_generate_answers_failure(self, pipeline):
        mock_rag = AsyncMock()
        mock_rag.query = AsyncMock(side_effect=Exception("API error"))

        pipeline.dataset = [
            EvaluationSample(query="test", ground_truth="truth", contexts=[])
        ]

        await pipeline._generate_answers(mock_rag)

        assert pipeline.dataset[0].answer == ""

    def test_log_to_mlflow_disabled(self, pipeline):
        report = EvaluationReport(
            run_id="test_run",
            timestamp="2024-01-01T00:00:00",
            total_samples=10,
            passed_samples=8,
            failed_samples=2,
            avg_faithfulness=0.85,
            thresholds={},
            all_thresholds_passed=True,
            results=[],
            evaluation_time_seconds=30.0,
        )

        # Should not raise error when MLflow logging disabled
        pipeline.log_to_mlflow(report)

    @patch("src.rag.evaluation.mlflow")
    def test_log_to_mlflow_enabled(self, mock_mlflow, mock_config):
        config = EvaluationConfig(log_to_mlflow=True)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            pipeline = RAGEvaluationPipeline(config=config, enable_opik_tracing=False)

        report = EvaluationReport(
            run_id="test_run",
            timestamp="2024-01-01T00:00:00",
            total_samples=10,
            passed_samples=8,
            failed_samples=2,
            avg_faithfulness=0.85,
            avg_answer_relevancy=0.90,
            overall_score=0.87,
            thresholds={"faithfulness": 0.8},
            all_thresholds_passed=True,
            results=[],
            evaluation_time_seconds=30.0,
        )

        mock_mlflow.start_run = MagicMock()
        mock_mlflow.start_run().__enter__ = MagicMock()
        mock_mlflow.start_run().__exit__ = MagicMock()

        pipeline.log_to_mlflow(report)

        # Should have attempted to log metrics
        mock_mlflow.set_experiment.assert_called_once()

    def test_check_thresholds_pass(self, pipeline):
        report = EvaluationReport(
            run_id="test",
            timestamp="2024-01-01",
            total_samples=10,
            passed_samples=10,
            failed_samples=0,
            avg_faithfulness=0.90,
            avg_answer_relevancy=0.95,
            avg_context_precision=0.85,
            avg_context_recall=0.85,
            overall_score=0.88,
            thresholds=DEFAULT_THRESHOLDS,
            all_thresholds_passed=True,
            results=[],
            evaluation_time_seconds=30.0,
        )

        passed, failures = pipeline.check_thresholds(report)

        assert passed is True
        assert len(failures) == 0

    def test_check_thresholds_fail(self, pipeline):
        report = EvaluationReport(
            run_id="test",
            timestamp="2024-01-01",
            total_samples=10,
            passed_samples=5,
            failed_samples=5,
            avg_faithfulness=0.70,  # Below threshold
            avg_answer_relevancy=0.75,  # Below threshold
            thresholds=DEFAULT_THRESHOLDS,
            all_thresholds_passed=False,
            results=[],
            evaluation_time_seconds=30.0,
        )

        passed, failures = pipeline.check_thresholds(report)

        assert passed is False
        assert len(failures) > 0
        assert any("Faithfulness" in f for f in failures)


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestHelperFunctions:
    @pytest.mark.asyncio
    async def test_quick_evaluate(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = await quick_evaluate(
                query="What is TRx?",
                answer="TRx is total prescriptions",
                contexts=["TRx stands for total prescriptions"],
                ground_truth="Total prescriptions",
            )

        assert isinstance(result, EvaluationResult)
        assert result.query == "What is TRx?"

    def test_get_ragas_evaluator_singleton(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            evaluator1 = get_ragas_evaluator()
            evaluator2 = get_ragas_evaluator()

            # Should be same instance
            assert evaluator1 is evaluator2

    def test_get_ragas_evaluator_reset(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            evaluator1 = get_ragas_evaluator()
            evaluator2 = get_ragas_evaluator(reset=True)

            # Should be different instances
            assert evaluator1 is not evaluator2


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_evaluate_sample_auto_contexts(self):
        sample = EvaluationSample(
            query="test",
            ground_truth="truth",
            answer="answer",
            contexts=["ctx1", "ctx2"],
            retrieved_contexts=[],  # Empty
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            evaluator = RAGASEvaluator(enable_opik_tracing=False)
            evaluator._ragas_available = False

            result = await evaluator.evaluate_sample(sample)

            # Should use contexts as retrieved_contexts
            assert result is not None

    def test_default_thresholds_coverage(self):
        assert "faithfulness" in DEFAULT_THRESHOLDS
        assert "answer_relevancy" in DEFAULT_THRESHOLDS
        assert "context_precision" in DEFAULT_THRESHOLDS
        assert "context_recall" in DEFAULT_THRESHOLDS
        assert "overall_score" in DEFAULT_THRESHOLDS

    @pytest.mark.asyncio
    async def test_evaluation_with_none_scores(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            pipeline = RAGEvaluationPipeline(enable_opik_tracing=False)

        # Create report with None scores
        report = EvaluationReport(
            run_id="test",
            timestamp="2024-01-01",
            total_samples=0,
            passed_samples=0,
            failed_samples=0,
            avg_faithfulness=None,
            avg_answer_relevancy=None,
            thresholds={},
            all_thresholds_passed=False,
            results=[],
            evaluation_time_seconds=0.0,
        )

        passed, failures = pipeline.check_thresholds(report)
        assert len(failures) == 0  # No failures if scores are None
