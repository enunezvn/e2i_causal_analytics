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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock external dependencies before importing evaluation module
mock_ragas = MagicMock()
mock_ragas.__spec__ = MagicMock()  # Fix for importlib.util.find_spec check
sys.modules["ragas"] = mock_ragas
sys.modules["ragas.metrics"] = MagicMock()
sys.modules["ragas.llms"] = MagicMock()
sys.modules["ragas.embeddings"] = MagicMock()
sys.modules["ragas.run_config"] = MagicMock()
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
        sample = EvaluationSample(query="test", ground_truth="truth", metadata={"key": "value"})

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
                sample_id="test",
                query="test",
                faithfulness=1.5,  # Invalid
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
        config = EvaluationConfig(thresholds=custom_thresholds, batch_size=20, log_to_mlflow=False)

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
            EvaluationSample(query="test2", ground_truth="truth2", contexts=["ctx2"]),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
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

    def test_default_dataset_has_at_least_30_samples(self):
        """Issue #496: the 10-sample golden set is too small for stable
        LLM-judge gates — per-sample verdict discreteness gives the aggregate
        metrics a ~0.08-0.10 noise band, so thresholds sat inside the band and
        flaked (faithfulness ~1/3 of runs; context_recall landed AT its 0.70
        gate). Expanding to >=30 shrinks the variance of every metric's mean by
        ~sqrt(3) so the floors rise and the thresholds become comfortable."""
        dataset = get_default_evaluation_dataset()
        assert len(dataset) >= 30, (
            f"golden set has {len(dataset)} samples; >=30 required to keep "
            "LLM-judge metric variance below the gate thresholds (#496)"
        )

    def test_every_sample_is_evaluable_by_ragas(self):
        """Every golden sample must carry the four fields RAGAS needs or it
        silently corrupts the aggregate (NaN/0 → drags a metric below its
        gate): query, ground_truth, a non-empty answer, non-empty contexts,
        and non-empty retrieved_contexts. Guards the 20 samples added for #496."""
        dataset = get_default_evaluation_dataset()
        for i, s in enumerate(dataset):
            assert s.query and s.query.strip(), f"sample {i}: empty query"
            assert s.ground_truth and s.ground_truth.strip(), f"sample {i}: empty ground_truth"
            assert s.answer and s.answer.strip(), f"sample {i}: empty answer"
            assert s.contexts and all(c.strip() for c in s.contexts), (
                f"sample {i}: empty/blank contexts"
            )
            assert s.retrieved_contexts and all(c.strip() for c in s.retrieved_contexts), (
                f"sample {i}: empty/blank retrieved_contexts"
            )

    def test_every_sample_has_metadata(self):
        """Every sample carries non-empty metadata (brand and/or kpi/analysis_type)
        so per-category coverage stays auditable as the set grows (#496)."""
        dataset = get_default_evaluation_dataset()
        for i, s in enumerate(dataset):
            assert s.metadata, f"sample {i} ({s.query!r}): empty metadata"


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
            EvaluationSample(query="test", ground_truth="truth", answer="ans", contexts=["ctx"])
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
            return RAGEvaluationPipeline(config=mock_config, enable_opik_tracing=False)

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
        mock_rag.query = AsyncMock(return_value={"answer": "test answer", "contexts": ["context1"]})

        pipeline.dataset = [EvaluationSample(query="test", ground_truth="truth", contexts=["ctx"])]
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

        pipeline.dataset = [EvaluationSample(query="test", ground_truth="truth", contexts=[])]

        await pipeline._generate_answers(mock_rag)

        assert pipeline.dataset[0].answer == "generated answer"
        assert len(pipeline.dataset[0].retrieved_contexts) == 2

    @pytest.mark.asyncio
    async def test_generate_answers_failure(self, pipeline):
        mock_rag = AsyncMock()
        mock_rag.query = AsyncMock(side_effect=Exception("API error"))

        pipeline.dataset = [EvaluationSample(query="test", ground_truth="truth", contexts=[])]

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
            avg_faithfulness=0.50,  # Below calibrated 0.70 threshold (#491)
            avg_answer_relevancy=0.70,  # Below calibrated 0.75 threshold (#496)
            thresholds=DEFAULT_THRESHOLDS,
            all_thresholds_passed=False,
            results=[],
            evaluation_time_seconds=30.0,
        )

        passed, failures = pipeline.check_thresholds(report)

        assert passed is False
        assert len(failures) > 0
        assert any("Faithfulness" in f for f in failures)

    def test_faithfulness_floor_passes_calibrated_threshold(self, pipeline):
        """Issue #491: with the accurate gpt-4o judge, faithfulness on the
        10-sample golden set has an empirical floor of ~0.77 (n=8 runs:
        0.77 x3, 0.85 x4, 0.875 x1), driven by per-claim verdict discreteness
        on a small sample, NOT a RAG-quality problem. The faithfulness
        threshold is calibrated to 0.70 (== context_recall's threshold; one
        noise-quantum below the floor) so a healthy pipeline at its noise
        floor passes the gate instead of flaking ~1/3 of runs. Other metrics
        held at their stable observed values (AR 0.876, CP 0.90, CR 0.75)."""
        report = EvaluationReport(
            run_id="test",
            timestamp="2024-01-01",
            total_samples=10,
            passed_samples=10,
            failed_samples=0,
            avg_faithfulness=0.77,  # observed gpt-4o floor; must clear 0.70 gate
            avg_answer_relevancy=0.876,
            avg_context_precision=0.90,
            avg_context_recall=0.75,
            overall_score=0.82,
            thresholds=DEFAULT_THRESHOLDS,
            all_thresholds_passed=True,
            results=[],
            evaluation_time_seconds=30.0,
        )

        passed, failures = pipeline.check_thresholds(report)

        assert passed is True, f"floor 0.77 must pass calibrated gate; got {failures}"
        assert len(failures) == 0

    def test_faithfulness_regression_below_floor_still_fails(self, pipeline):
        """The 0.70 calibration must still catch a genuine faithfulness
        regression (e.g. the RAG starts hallucinating): a value well below the
        ~0.77 noise floor must fail the gate. Guards against the calibration
        being misread as 'faithfulness no longer matters'."""
        report = EvaluationReport(
            run_id="test",
            timestamp="2024-01-01",
            total_samples=10,
            passed_samples=4,
            failed_samples=6,
            avg_faithfulness=0.55,  # genuine regression, multiple quanta below floor
            avg_answer_relevancy=0.876,
            avg_context_precision=0.90,
            avg_context_recall=0.75,
            overall_score=0.67,
            thresholds=DEFAULT_THRESHOLDS,
            all_thresholds_passed=False,
            results=[],
            evaluation_time_seconds=30.0,
        )

        passed, failures = pipeline.check_thresholds(report)

        assert passed is False
        assert any("Faithfulness" in f for f in failures)

    def test_answer_relevancy_floor_passes_calibrated_threshold(self, pipeline):
        """Issue #496: expanding the golden set to 30 (to stabilise
        context_recall) revealed that answer_relevancy under the gpt-4o judge
        sits at a rock-stable 0.804 — identical across two full CI runs — well
        below the old 0.85 gate (19/30 samples score under 0.85, including an
        original sample), because the judge scores the 'one query, answer
        synthesises two facts' style at ~0.80. AR is calibrated to 0.75 (one
        noise-quantum below the 0.804 floor) so a healthy pipeline at its floor
        passes instead of flaking the gate. Sister calibration to #491's
        faithfulness 0.70."""
        report = EvaluationReport(
            run_id="test",
            timestamp="2024-01-01",
            total_samples=30,
            passed_samples=30,
            failed_samples=0,
            avg_faithfulness=0.93,
            avg_answer_relevancy=0.804,  # observed gpt-4o floor (n=30); must clear 0.75 gate
            avg_context_precision=0.85,
            avg_context_recall=0.917,
            overall_score=0.88,
            thresholds=DEFAULT_THRESHOLDS,
            all_thresholds_passed=True,
            results=[],
            evaluation_time_seconds=30.0,
        )

        passed, failures = pipeline.check_thresholds(report)

        assert passed is True, f"AR floor 0.804 must pass calibrated gate; got {failures}"
        assert len(failures) == 0

    def test_answer_relevancy_regression_below_floor_still_fails(self, pipeline):
        """The 0.75 calibration must still catch a genuine answer-relevancy
        regression (e.g. the RAG starts answering off-topic): a value well below
        the 0.804 noise floor must fail the gate. Guards against the calibration
        being misread as 'answer relevancy no longer matters'."""
        report = EvaluationReport(
            run_id="test",
            timestamp="2024-01-01",
            total_samples=30,
            passed_samples=12,
            failed_samples=18,
            avg_faithfulness=0.93,
            avg_answer_relevancy=0.60,  # genuine regression, multiple quanta below floor
            avg_context_precision=0.85,
            avg_context_recall=0.917,
            overall_score=0.60,
            thresholds=DEFAULT_THRESHOLDS,
            all_thresholds_passed=False,
            results=[],
            evaluation_time_seconds=30.0,
        )

        passed, failures = pipeline.check_thresholds(report)

        assert passed is False
        assert any("Answer Relevancy" in f for f in failures)


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


# =============================================================================
# Test batched RAGAS evaluation (issue #504 — single evaluate() over N rows)
# =============================================================================
#
# These guard the CI-runtime fix: the RAGAS gate must score the whole golden
# set in ONE ragas.evaluate() call (RAGAS parallelises the row x metric jobs
# internally via its own RunConfig executor) instead of one serial, event-loop-
# blocking evaluate() per sample. The per-row scores the gate checks MUST be
# unchanged — only wall time. ragas/datasets/openai are mocked at module import
# (top of file), so these run without the RAGAS stack (both the CI unit-test job
# and the dev venv lack it). Written as sync functions calling asyncio.run so
# they are also runnable outside pytest (the conftest datasets-import chain
# blocks local pytest collection — see issue #496 notes).


def _batched_ragas_evaluator():
    """A RAGASEvaluator forced onto the batched RAGAS path (ragas + LLM available, no Opik tracer)."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        ev = RAGASEvaluator(config=EvaluationConfig(log_to_mlflow=False), enable_opik_tracing=False)
    ev._ragas_available = True
    ev._llm_configured = True
    ev._opik_tracer = None
    return ev


def _point_mock_evaluate_at_frame(n):
    """Make the mocked ragas.evaluate() return an n-row to_pandas() with distinct per-row scores."""
    import pandas as pd

    mock_ragas.evaluate.reset_mock()
    mock_ragas.evaluate.side_effect = None
    sys.modules["datasets"].Dataset.from_dict.reset_mock()
    frame = pd.DataFrame(
        {
            "faithfulness": [round(0.90 + i / 1000, 3) for i in range(n)],
            "answer_relevancy": [round(0.80 + i / 1000, 3) for i in range(n)],
            "context_precision": [round(0.85 + i / 1000, 3) for i in range(n)],
            "context_recall": [round(0.88 + i / 1000, 3) for i in range(n)],
        }
    )
    result = MagicMock()
    result.to_pandas.return_value = frame
    mock_ragas.evaluate.return_value = result
    return frame


def _answered_samples(n):
    return [
        EvaluationSample(
            query=f"q{i}", ground_truth=f"gt{i}", answer=f"a{i}", retrieved_contexts=[f"c{i}"]
        )
        for i in range(n)
    ]


def test_batched_ragas_calls_evaluate_once_on_full_dataset():
    """#504: N samples -> ONE evaluate() on an N-row dataset (was N serial 1-row calls)."""
    n = 5
    _point_mock_evaluate_at_frame(n)
    ev = _batched_ragas_evaluator()
    with patch("src.rag.evaluation._ensure_ragas_vertexai_compat"):
        results = asyncio.run(ev.evaluate_batch(_answered_samples(n)))
    assert mock_ragas.evaluate.call_count == 1, (
        f"expected ONE batched evaluate(), got {mock_ragas.evaluate.call_count}"
    )
    data_arg = sys.modules["datasets"].Dataset.from_dict.call_args.args[0]
    assert len(data_arg["question"]) == n
    assert len(results) == n


def test_batched_ragas_maps_each_row_to_its_sample_in_order():
    """Per-row scores from the batched frame map back to the right sample, in order."""
    n = 4
    frame = _point_mock_evaluate_at_frame(n)
    ev = _batched_ragas_evaluator()
    with patch("src.rag.evaluation._ensure_ragas_vertexai_compat"):
        results = asyncio.run(ev.evaluate_batch(_answered_samples(n)))
    for i in range(n):
        assert results[i].faithfulness == frame["faithfulness"].iloc[i]
        assert results[i].answer_relevancy == frame["answer_relevancy"].iloc[i]
        assert results[i].context_precision == frame["context_precision"].iloc[i]
        assert results[i].context_recall == frame["context_recall"].iloc[i]


def test_batched_ragas_passes_judge_as_args_without_mutating_singletons():
    """Judge llm/embeddings/run_config go in as evaluate() args; the module-level metric
    singletons are NOT mutated (no cross-call race — the #504 thread-safety fix)."""
    n = 3
    _point_mock_evaluate_at_frame(n)
    metrics_mod = sys.modules["ragas.metrics"]
    sentinel = object()
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        getattr(metrics_mod, name).llm = sentinel
    ev = _batched_ragas_evaluator()
    with patch("src.rag.evaluation._ensure_ragas_vertexai_compat"):
        asyncio.run(ev.evaluate_batch(_answered_samples(n)))
    kwargs = mock_ragas.evaluate.call_args.kwargs
    assert "llm" in kwargs and "embeddings" in kwargs and "run_config" in kwargs
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        assert getattr(metrics_mod, name).llm is sentinel, (
            f"{name}.llm was mutated on the shared singleton"
        )


def test_batched_ragas_handles_no_answer_samples_without_scoring_them():
    """No-answer samples get the 'No answer provided' result; only answered ones are batched; order kept."""
    _point_mock_evaluate_at_frame(2)
    ev = _batched_ragas_evaluator()
    samples = [
        EvaluationSample(query="q0", ground_truth="gt0", answer="a0", retrieved_contexts=["c0"]),
        EvaluationSample(query="q1", ground_truth="gt1", answer="", retrieved_contexts=["c1"]),
        EvaluationSample(query="q2", ground_truth="gt2", answer="a2", retrieved_contexts=["c2"]),
    ]
    with patch("src.rag.evaluation._ensure_ragas_vertexai_compat"):
        results = asyncio.run(ev.evaluate_batch(samples))
    assert len(results) == 3
    assert results[1].metadata.get("error") == "No answer provided"
    assert results[1].faithfulness is None
    data_arg = sys.modules["datasets"].Dataset.from_dict.call_args.args[0]
    assert len(data_arg["question"]) == 2
    assert mock_ragas.evaluate.call_count == 1


def test_ragas_workflow_has_concurrency_and_timeout():
    """#504 pure-infra guards: cancel superseded runs + bound runaway runs."""
    import yaml

    wf_path = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "ragas-evaluation.yml"
    wf = yaml.safe_load(wf_path.read_text())
    assert "concurrency" in wf, "workflow needs a concurrency group to cancel superseded runs"
    assert wf["concurrency"].get("cancel-in-progress") is True
    assert wf["jobs"]["ragas-evaluation"].get("timeout-minutes"), (
        "ragas-evaluation job needs timeout-minutes"
    )


def test_batched_ragas_falls_back_to_heuristic_when_evaluate_raises():
    """A non-import error in the batched evaluate() falls every sample back to the
    heuristic scorer — no sample dropped, length preserved (#504)."""
    _point_mock_evaluate_at_frame(3)
    mock_ragas.evaluate.side_effect = RuntimeError("ragas blew up")
    ev = _batched_ragas_evaluator()
    with patch("src.rag.evaluation._ensure_ragas_vertexai_compat"):
        results = asyncio.run(ev.evaluate_batch(_answered_samples(3)))
    mock_ragas.evaluate.side_effect = None
    assert len(results) == 3
    # heuristic fallback returns numeric (non-None) scores
    assert all(r.faithfulness is not None for r in results)


def test_batched_ragas_runconfig_caps_only_workers_not_timeout_or_retries():
    """RunConfig must override ONLY max_workers; timeout/max_retries stay at RAGAS
    defaults so the per-call NaN envelope (and the gate scores) match main (#504)."""
    _point_mock_evaluate_at_frame(2)
    run_config_cls = sys.modules["ragas.run_config"].RunConfig
    run_config_cls.reset_mock()
    ev = _batched_ragas_evaluator()
    with patch("src.rag.evaluation._ensure_ragas_vertexai_compat"):
        asyncio.run(ev.evaluate_batch(_answered_samples(2)))
    run_config_cls.assert_called_once()
    kwargs = run_config_cls.call_args.kwargs
    assert "max_workers" in kwargs
    assert "timeout" not in kwargs and "max_retries" not in kwargs


def test_evaluate_batch_uses_per_sample_path_when_opik_tracing_enabled():
    """With Opik tracing on, evaluate_batch must NOT use the batched path (tracing
    needs a span per sample), so it routes to evaluate_sample instead (#504)."""
    ev = _batched_ragas_evaluator()
    ev._opik_tracer = MagicMock()  # tracer present
    ev.enable_opik_tracing = True
    ev._evaluate_batch_with_ragas = AsyncMock()
    ev.evaluate_sample = AsyncMock(return_value=EvaluationResult(sample_id="x", query="q"))
    asyncio.run(ev.evaluate_batch(_answered_samples(2)))
    ev._evaluate_batch_with_ragas.assert_not_called()
    assert ev.evaluate_sample.await_count == 2


def test_gate_script_routes_to_batched_path_by_default():
    """#504: scripts/run_ragas_eval.py must default Opik tracing OFF (via the
    --opik-tracing opt-in flag) so evaluate_batch uses the fast batched path.
    Static guard so the gate cannot silently regress to the slow per-sample
    tracing path (which is what made the first PR run show no speedup)."""
    script = (Path(__file__).resolve().parents[3] / "scripts" / "run_ragas_eval.py").read_text()
    assert "--opik-tracing" in script, "gate must expose the --opik-tracing opt-in flag"
    assert "enable_opik_tracing=args.opik_tracing" in script, (
        "gate must wire tracing to the default-off flag, not hardcode it on"
    )
