"""
Tests for RAGAS evaluation framework.

Tests the evaluation pipeline, metrics, and MLflow integration.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.evaluation import (
    EvaluationConfig,
    EvaluationReport,
    EvaluationResult,
    EvaluationSample,
    RAGASEvaluator,
    RAGEvaluationPipeline,
    DEFAULT_THRESHOLDS,
    create_evaluation_sample,
    get_default_evaluation_dataset,
    load_evaluation_dataset,
    quick_evaluate,
    save_evaluation_dataset,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_evaluation_sample():
    """Create a sample evaluation sample for testing."""
    return EvaluationSample(
        query="What are the TRx trends for Kisqali?",
        ground_truth="Kisqali TRx showed 15% growth in Q4 with increased HCP adoption.",
        contexts=[
            "Kisqali Q4 TRx report: Total prescriptions reached 45,000 units, up 15% from Q3.",
            "HCP targeting data shows 850 new prescribers adopted Kisqali in Q4.",
        ],
        answer="Kisqali TRx trends show 15% growth in Q4, driven by increased HCP adoption and 850 new prescribers.",
        retrieved_contexts=[
            "Kisqali Q4 TRx report: Total prescriptions reached 45,000 units, up 15% from Q3.",
        ],
        metadata={"brand": "Kisqali", "kpi": "TRx"},
    )


@pytest.fixture
def sample_evaluation_config():
    """Create a test evaluation configuration."""
    return EvaluationConfig(
        thresholds={
            "faithfulness": 0.80,
            "answer_relevancy": 0.80,
            "context_precision": 0.70,
            "context_recall": 0.70,
        },
        log_to_mlflow=False,
        batch_size=5,
        max_concurrent=2,
    )


@pytest.fixture
def sample_evaluation_result():
    """Create a sample evaluation result."""
    return EvaluationResult(
        sample_id="test_001",
        query="What are the TRx trends?",
        faithfulness=0.92,
        answer_relevancy=0.88,
        context_precision=0.85,
        context_recall=0.80,
        overall_score=0.8625,
        passed_thresholds=True,
        metadata={"brand": "Kisqali"},
    )


# =============================================================================
# Test EvaluationSample Model
# =============================================================================


class TestEvaluationSample:
    """Tests for EvaluationSample model."""

    def test_create_sample(self):
        """Test creating an evaluation sample."""
        sample = EvaluationSample(
            query="Test query",
            ground_truth="Expected answer",
        )
        assert sample.query == "Test query"
        assert sample.ground_truth == "Expected answer"
        assert sample.contexts == []
        assert sample.answer is None

    def test_sample_with_all_fields(self, sample_evaluation_sample):
        """Test sample with all fields populated."""
        sample = sample_evaluation_sample
        assert "Kisqali" in sample.query
        assert sample.metadata["brand"] == "Kisqali"
        assert len(sample.contexts) == 2
        assert sample.answer is not None

    def test_create_evaluation_sample_helper(self):
        """Test the create_evaluation_sample helper function."""
        sample = create_evaluation_sample(
            query="Test query",
            ground_truth="Test answer",
            contexts=["Context 1", "Context 2"],
            brand="Fabhalta",
            kpi="NRx",
        )
        assert sample.query == "Test query"
        assert sample.metadata["brand"] == "Fabhalta"
        assert sample.metadata["kpi"] == "NRx"


# =============================================================================
# Test EvaluationResult Model
# =============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_create_result(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            sample_id="test_001",
            query="Test query",
        )
        assert result.sample_id == "test_001"
        assert result.faithfulness is None
        assert result.passed_thresholds is False

    def test_result_with_scores(self, sample_evaluation_result):
        """Test result with metric scores."""
        result = sample_evaluation_result
        assert result.faithfulness == 0.92
        assert result.answer_relevancy == 0.88
        assert result.overall_score == 0.8625
        assert result.passed_thresholds is True

    def test_result_score_bounds(self):
        """Test that scores are bounded 0-1."""
        result = EvaluationResult(
            sample_id="test",
            query="query",
            faithfulness=0.5,
            answer_relevancy=1.0,
            context_precision=0.0,
            context_recall=0.75,
        )
        assert 0 <= result.faithfulness <= 1
        assert 0 <= result.answer_relevancy <= 1
        assert 0 <= result.context_precision <= 1
        assert 0 <= result.context_recall <= 1


# =============================================================================
# Test EvaluationConfig
# =============================================================================


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluationConfig()
        # Thresholds adjusted to realistic baseline per Phase 6 plan
        assert config.thresholds["faithfulness"] == 0.80
        assert config.thresholds["answer_relevancy"] == 0.85
        assert config.log_to_mlflow is True
        assert config.batch_size == 10

    def test_custom_config(self, sample_evaluation_config):
        """Test custom configuration."""
        config = sample_evaluation_config
        assert config.thresholds["faithfulness"] == 0.80
        assert config.log_to_mlflow is False
        assert config.max_concurrent == 2

    def test_default_thresholds_constant(self):
        """Test that default thresholds are defined."""
        # Thresholds adjusted to realistic baseline per Phase 6 plan
        assert DEFAULT_THRESHOLDS["faithfulness"] == 0.80
        assert DEFAULT_THRESHOLDS["answer_relevancy"] == 0.85
        assert DEFAULT_THRESHOLDS["context_precision"] == 0.80
        assert DEFAULT_THRESHOLDS["context_recall"] == 0.70


# =============================================================================
# Test Dataset Functions
# =============================================================================


class TestDatasetFunctions:
    """Tests for dataset loading and saving."""

    def test_get_default_dataset(self):
        """Test loading default evaluation dataset."""
        dataset = get_default_evaluation_dataset()
        assert len(dataset) > 0
        assert all(isinstance(s, EvaluationSample) for s in dataset)

        # Check that samples have required fields
        for sample in dataset:
            assert sample.query
            assert sample.ground_truth
            assert len(sample.contexts) > 0

    def test_default_dataset_brand_coverage(self):
        """Test that default dataset covers all brands."""
        dataset = get_default_evaluation_dataset()
        brands = {s.metadata.get("brand") for s in dataset if s.metadata.get("brand")}
        assert "Kisqali" in brands
        assert "Fabhalta" in brands
        assert "Remibrutinib" in brands

    def test_save_and_load_dataset(self):
        """Test saving and loading evaluation dataset."""
        samples = [
            EvaluationSample(
                query="Test query 1",
                ground_truth="Answer 1",
                contexts=["Context 1"],
            ),
            EvaluationSample(
                query="Test query 2",
                ground_truth="Answer 2",
                contexts=["Context 2"],
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            save_evaluation_dataset(samples, temp_path)
            loaded = load_evaluation_dataset(temp_path)

            assert len(loaded) == 2
            assert loaded[0].query == "Test query 1"
            assert loaded[1].ground_truth == "Answer 2"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_nonexistent_dataset(self):
        """Test loading from nonexistent path returns default."""
        dataset = load_evaluation_dataset("/nonexistent/path.json")
        assert len(dataset) > 0  # Should return default dataset


# =============================================================================
# Test RAGASEvaluator
# =============================================================================


class TestRAGASEvaluator:
    """Tests for RAGASEvaluator."""

    def test_evaluator_initialization(self, sample_evaluation_config):
        """Test evaluator initialization."""
        evaluator = RAGASEvaluator(config=sample_evaluation_config)
        assert evaluator.config == sample_evaluation_config
        assert evaluator.llm_provider == "anthropic"

    def test_evaluator_ragas_check(self):
        """Test RAGAS availability check."""
        evaluator = RAGASEvaluator()
        # RAGAS should be available in test environment
        assert evaluator._ragas_available is True

    @pytest.mark.asyncio
    async def test_evaluate_sample_no_answer(self, sample_evaluation_config):
        """Test evaluating sample with no answer."""
        evaluator = RAGASEvaluator(config=sample_evaluation_config)
        sample = EvaluationSample(
            query="Test query",
            ground_truth="Test answer",
        )

        result = await evaluator.evaluate_sample(sample)
        assert result.query == "Test query"
        assert "error" in result.metadata

    @pytest.mark.asyncio
    async def test_evaluate_with_fallback(self, sample_evaluation_config, sample_evaluation_sample):
        """Test fallback evaluation when LLM not configured."""
        evaluator = RAGASEvaluator(config=sample_evaluation_config)
        evaluator._llm_configured = False

        result = await evaluator.evaluate_sample(sample_evaluation_sample)

        assert result.query == sample_evaluation_sample.query
        assert result.faithfulness is not None
        assert result.answer_relevancy is not None
        assert result.context_precision is not None
        assert result.context_recall is not None
        assert "fallback_heuristic" in result.metadata.get("evaluation_method", "")

    @pytest.mark.asyncio
    async def test_fallback_heuristic_scoring(self, sample_evaluation_config):
        """Test that fallback heuristic produces reasonable scores."""
        evaluator = RAGASEvaluator(config=sample_evaluation_config)
        evaluator._llm_configured = False

        # Sample with high overlap
        sample_high_overlap = EvaluationSample(
            query="What is the TRx for Kisqali?",
            ground_truth="Kisqali TRx is 45000 units with 15% growth",
            answer="Kisqali TRx reached 45000 units showing 15% growth in Q4",
            retrieved_contexts=["Kisqali TRx is 45000 units with 15% growth in Q4"],
        )

        result = await evaluator.evaluate_sample(sample_high_overlap)
        # High overlap should produce higher scores
        assert result.overall_score is not None
        assert result.overall_score > 0

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, sample_evaluation_config):
        """Test batch evaluation."""
        evaluator = RAGASEvaluator(config=sample_evaluation_config)
        evaluator._llm_configured = False

        samples = [
            EvaluationSample(
                query=f"Query {i}",
                ground_truth=f"Answer {i}",
                answer=f"Generated answer {i}",
                retrieved_contexts=[f"Context {i}"],
            )
            for i in range(3)
        ]

        results = await evaluator.evaluate_batch(samples)
        assert len(results) == 3
        assert all(isinstance(r, EvaluationResult) for r in results)


# =============================================================================
# Test RAGEvaluationPipeline
# =============================================================================


class TestRAGEvaluationPipeline:
    """Tests for RAGEvaluationPipeline."""

    def test_pipeline_initialization(self, sample_evaluation_config):
        """Test pipeline initialization."""
        pipeline = RAGEvaluationPipeline(config=sample_evaluation_config)
        assert pipeline.config == sample_evaluation_config
        assert len(pipeline.dataset) > 0

    def test_pipeline_with_custom_dataset(self, sample_evaluation_config):
        """Test pipeline with custom dataset path."""
        samples = [
            EvaluationSample(query="Q1", ground_truth="A1", contexts=["C1"]),
            EvaluationSample(query="Q2", ground_truth="A2", contexts=["C2"]),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([s.model_dump() for s in samples], f)
            temp_path = f.name

        try:
            pipeline = RAGEvaluationPipeline(
                config=sample_evaluation_config,
                dataset_path=temp_path,
            )
            assert len(pipeline.dataset) == 2
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_run_evaluation(self, sample_evaluation_config):
        """Test running full evaluation."""
        # Use a small custom dataset for faster testing
        samples = [
            EvaluationSample(
                query="Test query",
                ground_truth="Expected answer",
                answer="Generated answer",
                contexts=["Context passage"],
                retrieved_contexts=["Context passage"],
            )
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([s.model_dump() for s in samples], f)
            temp_path = f.name

        try:
            pipeline = RAGEvaluationPipeline(
                config=sample_evaluation_config,
                dataset_path=temp_path,
            )
            pipeline.evaluator._llm_configured = False  # Force fallback

            report = await pipeline.run_evaluation()

            assert isinstance(report, EvaluationReport)
            assert report.total_samples == 1
            assert len(report.results) == 1
            assert report.evaluation_time_seconds > 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_check_thresholds_pass(self, sample_evaluation_config):
        """Test threshold checking when all pass."""
        pipeline = RAGEvaluationPipeline(config=sample_evaluation_config)

        report = EvaluationReport(
            run_id="test",
            timestamp="2024-01-01",
            total_samples=1,
            passed_samples=1,
            failed_samples=0,
            avg_faithfulness=0.90,
            avg_answer_relevancy=0.85,
            avg_context_precision=0.80,
            avg_context_recall=0.75,
            thresholds=sample_evaluation_config.thresholds,
            evaluation_time_seconds=1.0,
        )

        passed, failures = pipeline.check_thresholds(report)
        assert passed is True
        assert len(failures) == 0

    def test_check_thresholds_fail(self, sample_evaluation_config):
        """Test threshold checking when some fail."""
        pipeline = RAGEvaluationPipeline(config=sample_evaluation_config)

        report = EvaluationReport(
            run_id="test",
            timestamp="2024-01-01",
            total_samples=1,
            passed_samples=0,
            failed_samples=1,
            avg_faithfulness=0.50,  # Below 0.80 threshold
            avg_answer_relevancy=0.85,
            avg_context_precision=0.80,
            avg_context_recall=0.75,
            thresholds=sample_evaluation_config.thresholds,
            evaluation_time_seconds=1.0,
        )

        passed, failures = pipeline.check_thresholds(report)
        assert passed is False
        assert len(failures) == 1
        assert "Faithfulness" in failures[0]

    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.log_metric")
    @patch("mlflow.log_param")
    @patch("mlflow.log_artifact")
    def test_log_to_mlflow(
        self,
        mock_log_artifact,
        mock_log_param,
        mock_log_metric,
        mock_start_run,
        mock_set_experiment,
    ):
        """Test MLflow logging."""
        config = EvaluationConfig(log_to_mlflow=True)
        pipeline = RAGEvaluationPipeline(config=config)

        report = EvaluationReport(
            run_id="test_run",
            timestamp="2024-01-01",
            total_samples=10,
            passed_samples=8,
            failed_samples=2,
            avg_faithfulness=0.85,
            avg_answer_relevancy=0.90,
            avg_context_precision=0.80,
            avg_context_recall=0.80,
            overall_score=0.8375,
            thresholds=DEFAULT_THRESHOLDS,
            all_thresholds_passed=True,
            evaluation_time_seconds=5.0,
        )

        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()

        pipeline.log_to_mlflow(report)

        mock_set_experiment.assert_called_once_with("rag-evaluation")
        mock_start_run.assert_called_once()
        assert mock_log_metric.call_count > 0

    def test_no_mlflow_logging_when_disabled(self, sample_evaluation_config):
        """Test that MLflow logging is skipped when disabled."""
        sample_evaluation_config.log_to_mlflow = False
        pipeline = RAGEvaluationPipeline(config=sample_evaluation_config)

        report = EvaluationReport(
            run_id="test",
            timestamp="2024-01-01",
            total_samples=1,
            passed_samples=1,
            failed_samples=0,
            thresholds={},
            evaluation_time_seconds=1.0,
        )

        # Should not raise any errors
        pipeline.log_to_mlflow(report)


# =============================================================================
# Test Quick Evaluate Function
# =============================================================================


class TestQuickEvaluate:
    """Tests for quick_evaluate convenience function."""

    @pytest.mark.asyncio
    async def test_quick_evaluate(self):
        """Test quick evaluation of single query."""
        result = await quick_evaluate(
            query="What is Kisqali TRx?",
            answer="Kisqali TRx is 45000 units",
            contexts=["Kisqali prescription volume is 45000"],
            ground_truth="Kisqali TRx is approximately 45000",
        )

        assert isinstance(result, EvaluationResult)
        assert result.query == "What is Kisqali TRx?"
        assert result.faithfulness is not None

    @pytest.mark.asyncio
    async def test_quick_evaluate_without_ground_truth(self):
        """Test quick evaluation without explicit ground truth."""
        result = await quick_evaluate(
            query="Test query",
            answer="Test answer",
            contexts=["Test context"],
        )

        assert isinstance(result, EvaluationResult)
        # Should use answer as ground truth when not provided


# =============================================================================
# Test EvaluationReport Model
# =============================================================================


class TestEvaluationReport:
    """Tests for EvaluationReport model."""

    def test_create_report(self):
        """Test creating an evaluation report."""
        report = EvaluationReport(
            run_id="test_001",
            timestamp="2024-01-01 12:00:00",
            total_samples=10,
            passed_samples=8,
            failed_samples=2,
            thresholds=DEFAULT_THRESHOLDS,
            evaluation_time_seconds=5.5,
        )

        assert report.run_id == "test_001"
        assert report.total_samples == 10
        assert report.passed_samples == 8
        assert report.all_thresholds_passed is False

    def test_report_with_results(self, sample_evaluation_result):
        """Test report with individual results."""
        report = EvaluationReport(
            run_id="test",
            timestamp="2024-01-01",
            total_samples=1,
            passed_samples=1,
            failed_samples=0,
            avg_faithfulness=0.92,
            avg_answer_relevancy=0.88,
            results=[sample_evaluation_result],
            thresholds=DEFAULT_THRESHOLDS,
            evaluation_time_seconds=1.0,
        )

        assert len(report.results) == 1
        assert report.results[0].sample_id == "test_001"

    def test_report_serialization(self):
        """Test that report can be serialized to JSON."""
        report = EvaluationReport(
            run_id="test",
            timestamp="2024-01-01",
            total_samples=1,
            passed_samples=1,
            failed_samples=0,
            thresholds={},
            evaluation_time_seconds=1.0,
        )

        # Should not raise
        json_str = json.dumps(report.model_dump())
        assert "test" in json_str


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for evaluation pipeline."""

    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self):
        """Test complete evaluation flow with default dataset."""
        config = EvaluationConfig(
            log_to_mlflow=False,
            thresholds={
                "faithfulness": 0.0,  # Low thresholds for testing
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            },
        )

        # Create minimal dataset
        samples = [
            EvaluationSample(
                query="What is Kisqali?",
                ground_truth="Kisqali is a CDK4/6 inhibitor for breast cancer",
                answer="Kisqali treats breast cancer",
                contexts=["Kisqali is used for breast cancer treatment"],
                retrieved_contexts=["Kisqali is used for breast cancer treatment"],
            )
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([s.model_dump() for s in samples], f)
            temp_path = f.name

        try:
            pipeline = RAGEvaluationPipeline(
                config=config,
                dataset_path=temp_path,
            )
            pipeline.evaluator._llm_configured = False

            report = await pipeline.run_evaluation()

            assert report.total_samples == 1
            assert report.avg_faithfulness is not None
            assert report.evaluation_time_seconds > 0

            # Check thresholds
            passed, _ = pipeline.check_thresholds(report)
            assert passed is True  # Should pass with 0.0 thresholds
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_concurrent_evaluation(self):
        """Test concurrent evaluation of multiple samples."""
        config = EvaluationConfig(
            log_to_mlflow=False,
            max_concurrent=3,
        )

        evaluator = RAGASEvaluator(config=config)
        evaluator._llm_configured = False

        samples = [
            EvaluationSample(
                query=f"Query {i}",
                ground_truth=f"Answer {i}",
                answer=f"Response {i}",
                retrieved_contexts=[f"Context {i}"],
            )
            for i in range(5)
        ]

        results = await evaluator.evaluate_batch(samples)
        assert len(results) == 5
        # All should have scores (fallback evaluation)
        assert all(r.faithfulness is not None for r in results)
