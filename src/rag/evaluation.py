"""
RAGAS-based evaluation framework for E2I RAG pipeline.

Evaluates:
1. Faithfulness - Is the answer grounded in retrieved context?
2. Answer Relevancy - Does the answer address the query?
3. Context Precision - Are retrieved documents ranked correctly?
4. Context Recall - Did we retrieve all relevant documents?

CRITICAL: Evaluation is for OPERATIONAL queries only.
NOT for: Medical/clinical query evaluation.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Data Models
# =============================================================================


class EvaluationSample(BaseModel):
    """Single evaluation sample with ground truth."""

    query: str = Field(..., description="User query")
    ground_truth: str = Field(..., description="Expected answer or key points")
    contexts: List[str] = Field(
        default_factory=list, description="Reference context passages"
    )
    answer: Optional[str] = Field(None, description="Generated answer to evaluate")
    retrieved_contexts: List[str] = Field(
        default_factory=list, description="Contexts retrieved by RAG"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (brand, KPI, etc.)"
    )


class EvaluationResult(BaseModel):
    """Evaluation result for a single sample."""

    sample_id: str = Field(..., description="Unique sample identifier")
    query: str = Field(..., description="Query evaluated")
    faithfulness: Optional[float] = Field(None, ge=0, le=1)
    answer_relevancy: Optional[float] = Field(None, ge=0, le=1)
    context_precision: Optional[float] = Field(None, ge=0, le=1)
    context_recall: Optional[float] = Field(None, ge=0, le=1)
    overall_score: Optional[float] = Field(None, ge=0, le=1)
    passed_thresholds: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Aggregated evaluation report."""

    run_id: str = Field(..., description="Unique run identifier")
    timestamp: str = Field(..., description="Evaluation timestamp")
    total_samples: int = Field(..., ge=0)
    passed_samples: int = Field(..., ge=0)
    failed_samples: int = Field(..., ge=0)

    # Aggregate metrics
    avg_faithfulness: Optional[float] = Field(None, ge=0, le=1)
    avg_answer_relevancy: Optional[float] = Field(None, ge=0, le=1)
    avg_context_precision: Optional[float] = Field(None, ge=0, le=1)
    avg_context_recall: Optional[float] = Field(None, ge=0, le=1)
    overall_score: Optional[float] = Field(None, ge=0, le=1)

    # Thresholds
    thresholds: Dict[str, float] = Field(default_factory=dict)
    all_thresholds_passed: bool = Field(default=False)

    # Individual results
    results: List[EvaluationResult] = Field(default_factory=list)

    # Timing
    evaluation_time_seconds: float = Field(..., ge=0)


# =============================================================================
# Threshold Configuration
# =============================================================================

DEFAULT_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.90,
    "context_precision": 0.80,
    "context_recall": 0.80,
    "overall_score": 0.85,
}


@dataclass
class EvaluationConfig:
    """Configuration for RAG evaluation."""

    thresholds: Dict[str, float] = field(default_factory=lambda: DEFAULT_THRESHOLDS.copy())
    log_to_mlflow: bool = True
    mlflow_experiment: str = "rag-evaluation"
    batch_size: int = 10
    max_concurrent: int = 5
    timeout_seconds: int = 60
    retry_count: int = 3


# =============================================================================
# Evaluation Dataset
# =============================================================================


def get_default_evaluation_dataset() -> List[EvaluationSample]:
    """
    Get default evaluation dataset for E2I pharmaceutical commercial analytics.

    Contains representative queries across brands and KPIs with simulated answers.
    """
    return [
        # Kisqali (breast cancer) queries
        EvaluationSample(
            query="What are the TRx trends for Kisqali in Q4?",
            ground_truth="Kisqali TRx showed growth in Q4 with 15% increase in prescription volume. Key drivers include increased HCP adoption in top territories and successful conversion from competitive therapies.",
            contexts=[
                "Kisqali Q4 TRx report: Total prescriptions reached 45,000 units, up 15% from Q3. Northeast region led with 22% growth.",
                "HCP targeting data shows 850 new prescribers adopted Kisqali in Q4, driving market share expansion in HR+ breast cancer segment.",
            ],
            answer="Kisqali TRx trends in Q4 show strong growth with a 15% increase in prescription volume, reaching 45,000 units. The Northeast region led with 22% growth, and 850 new HCPs adopted the drug, expanding market share in the HR+ breast cancer segment.",
            retrieved_contexts=[
                "Kisqali Q4 TRx report: Total prescriptions reached 45,000 units, up 15% from Q3. Northeast region led with 22% growth.",
                "HCP targeting data shows 850 new prescribers adopted Kisqali in Q4, driving market share expansion in HR+ breast cancer segment.",
            ],
            metadata={"brand": "Kisqali", "kpi": "TRx", "period": "Q4"},
        ),
        EvaluationSample(
            query="What is the market share for Kisqali compared to competitors?",
            ground_truth="Kisqali holds approximately 32% market share in the CDK4/6 inhibitor market for HR+/HER2- breast cancer, positioning it as the second-largest player behind Ibrance.",
            contexts=[
                "CDK4/6 inhibitor market share Q4: Ibrance 45%, Kisqali 32%, Verzenio 23%.",
                "Kisqali gained 3 percentage points share from Ibrance in key territories, driven by favorable efficacy data.",
            ],
            answer="Kisqali holds 32% market share in the CDK4/6 inhibitor market for HR+/HER2- breast cancer, making it the second-largest player. Ibrance leads at 45%, while Verzenio has 23%. Kisqali gained 3 percentage points from Ibrance in key territories.",
            retrieved_contexts=[
                "CDK4/6 inhibitor market share Q4: Ibrance 45%, Kisqali 32%, Verzenio 23%.",
                "Kisqali gained 3 percentage points share from Ibrance in key territories, driven by favorable efficacy data.",
            ],
            metadata={"brand": "Kisqali", "kpi": "market_share", "period": "Q4"},
        ),
        # Fabhalta (PNH) queries
        EvaluationSample(
            query="How many new prescriptions for Fabhalta this month?",
            ground_truth="Fabhalta achieved 280 new prescriptions (NRx) this month, representing 18% month-over-month growth in the PNH market.",
            contexts=[
                "Fabhalta NRx monthly report: 280 new starts recorded, up from 237 last month (18% growth).",
                "PNH market new patient starts: Fabhalta capturing 35% of newly diagnosed patients.",
            ],
            answer="Fabhalta achieved 280 new prescriptions (NRx) this month, representing 18% month-over-month growth from 237 last month. Fabhalta is capturing 35% of newly diagnosed PNH patients.",
            retrieved_contexts=[
                "Fabhalta NRx monthly report: 280 new starts recorded, up from 237 last month (18% growth).",
                "PNH market new patient starts: Fabhalta capturing 35% of newly diagnosed patients.",
            ],
            metadata={"brand": "Fabhalta", "kpi": "NRx", "period": "monthly"},
        ),
        EvaluationSample(
            query="What drives Fabhalta adoption among hematologists?",
            ground_truth="Key drivers of Fabhalta adoption include oral administration convenience, complement component 3 inhibition mechanism, and reduced transfusion dependency compared to C5 inhibitors.",
            contexts=[
                "Hematologist survey: 78% cite oral route as primary driver for Fabhalta preference.",
                "Clinical outcomes data: Fabhalta patients show 40% reduction in transfusion requirements vs. baseline.",
            ],
            answer="Key drivers of Fabhalta adoption among hematologists include oral administration convenience (cited by 78% as primary driver) and reduced transfusion dependency, with patients showing 40% reduction in transfusion requirements compared to baseline.",
            retrieved_contexts=[
                "Hematologist survey: 78% cite oral route as primary driver for Fabhalta preference.",
                "Clinical outcomes data: Fabhalta patients show 40% reduction in transfusion requirements vs. baseline.",
            ],
            metadata={"brand": "Fabhalta", "kpi": "adoption_rate"},
        ),
        # Remibrutinib (CSU) queries
        EvaluationSample(
            query="What is the conversion rate for Remibrutinib?",
            ground_truth="Remibrutinib shows a 45% conversion rate from trial to prescription, with allergists demonstrating higher conversion rates (52%) compared to dermatologists (38%).",
            contexts=[
                "Remibrutinib conversion metrics: Overall conversion rate 45%. Allergist segment leads at 52%.",
                "HCP specialty analysis: Dermatologists show 38% conversion, opportunity for targeted education.",
            ],
            answer="Remibrutinib has a 45% overall conversion rate from trial to prescription. Allergists lead with 52% conversion, while dermatologists show 38% conversion, indicating opportunity for targeted education.",
            retrieved_contexts=[
                "Remibrutinib conversion metrics: Overall conversion rate 45%. Allergist segment leads at 52%.",
                "HCP specialty analysis: Dermatologists show 38% conversion, opportunity for targeted education.",
            ],
            metadata={"brand": "Remibrutinib", "kpi": "conversion_rate"},
        ),
        EvaluationSample(
            query="Which territories have the highest Remibrutinib sales?",
            ground_truth="Top territories for Remibrutinib are California (23% of sales), Texas (15%), and Florida (12%), accounting for 50% of total prescription volume.",
            contexts=[
                "Territory performance Q4: California leads with 23% share, Texas 15%, Florida 12%.",
                "Regional analysis: West Coast territories show 30% higher per-HCP prescribing vs. national average.",
            ],
            answer="Top territories for Remibrutinib sales are California (23%), Texas (15%), and Florida (12%), together accounting for 50% of total prescription volume. West Coast territories show 30% higher per-HCP prescribing than the national average.",
            retrieved_contexts=[
                "Territory performance Q4: California leads with 23% share, Texas 15%, Florida 12%.",
                "Regional analysis: West Coast territories show 30% higher per-HCP prescribing vs. national average.",
            ],
            metadata={"brand": "Remibrutinib", "kpi": "territory_performance"},
        ),
        # Cross-brand / analytical queries
        EvaluationSample(
            query="What causal factors affect NRx growth?",
            ground_truth="Key causal factors for NRx growth include HCP detailing frequency (0.3 coefficient), peer education events (0.25), and patient support program enrollment (0.22).",
            contexts=[
                "Causal analysis: HCP detailing shows 0.3 coefficient impact on NRx. Statistical significance p<0.01.",
                "Multi-touch attribution: Peer education events contribute 0.25 lift, patient programs 0.22.",
            ],
            answer="Key causal factors affecting NRx growth include HCP detailing frequency with a 0.3 coefficient impact (p<0.01), peer education events contributing 0.25 lift, and patient support program enrollment with 0.22 impact.",
            retrieved_contexts=[
                "Causal analysis: HCP detailing shows 0.3 coefficient impact on NRx. Statistical significance p<0.01.",
                "Multi-touch attribution: Peer education events contribute 0.25 lift, patient programs 0.22.",
            ],
            metadata={"kpi": "NRx", "analysis_type": "causal"},
        ),
        EvaluationSample(
            query="What gaps exist in our HCP targeting strategy?",
            ground_truth="Key gaps identified: 35% of high-value HCPs have not been contacted in 90 days, Northeast region is underrepresented in speaker programs, and digital engagement is below benchmark for dermatologists.",
            contexts=[
                "HCP coverage analysis: 35% of decile 1-2 prescribers missing recent contact (>90 days).",
                "Gap analysis: Northeast speaker program participation 40% below target. Digital engagement for dermatology 25% below benchmark.",
            ],
            answer="Key gaps in HCP targeting strategy: 35% of high-value decile 1-2 prescribers have not been contacted in 90+ days. Northeast speaker program participation is 40% below target, and digital engagement for dermatology is 25% below benchmark.",
            retrieved_contexts=[
                "HCP coverage analysis: 35% of decile 1-2 prescribers missing recent contact (>90 days).",
                "Gap analysis: Northeast speaker program participation 40% below target. Digital engagement for dermatology 25% below benchmark.",
            ],
            metadata={"analysis_type": "gap_analysis"},
        ),
        # Performance / metrics queries
        EvaluationSample(
            query="What is the ROI on recent marketing campaigns?",
            ground_truth="Marketing campaign ROI ranges from 2.5x to 4.2x depending on channel. Digital campaigns show highest ROI at 4.2x, followed by HCP events at 3.1x and print at 2.5x.",
            contexts=[
                "Campaign ROI analysis: Digital campaigns 4.2x ROI, HCP events 3.1x, print materials 2.5x.",
                "Budget allocation recommendation: Shift 15% from print to digital based on ROI differential.",
            ],
            answer="Marketing campaign ROI ranges from 2.5x to 4.2x by channel. Digital campaigns lead with 4.2x ROI, followed by HCP events at 3.1x and print at 2.5x. Recommendation: shift 15% of budget from print to digital.",
            retrieved_contexts=[
                "Campaign ROI analysis: Digital campaigns 4.2x ROI, HCP events 3.1x, print materials 2.5x.",
                "Budget allocation recommendation: Shift 15% from print to digital based on ROI differential.",
            ],
            metadata={"kpi": "ROI", "analysis_type": "marketing"},
        ),
        EvaluationSample(
            query="How is drift detected in our prediction models?",
            ground_truth="Model drift is monitored via PSI (Population Stability Index) for feature distributions and concept drift detection for label shifts. Alerts trigger when PSI exceeds 0.2 or accuracy drops 5% from baseline.",
            contexts=[
                "Drift monitoring config: PSI threshold 0.2, accuracy degradation threshold 5%, check frequency daily.",
                "Drift detection methods: PSI for feature drift, KL divergence for prediction drift, accuracy monitoring for concept drift.",
            ],
            answer="Model drift is detected using PSI (Population Stability Index) for feature distributions with a 0.2 threshold, KL divergence for prediction drift, and accuracy monitoring for concept drift with a 5% degradation threshold. Checks run daily.",
            retrieved_contexts=[
                "Drift monitoring config: PSI threshold 0.2, accuracy degradation threshold 5%, check frequency daily.",
                "Drift detection methods: PSI for feature drift, KL divergence for prediction drift, accuracy monitoring for concept drift.",
            ],
            metadata={"analysis_type": "mlops"},
        ),
    ]


def load_evaluation_dataset(path: Optional[str] = None) -> List[EvaluationSample]:
    """
    Load evaluation dataset from file or return default.

    Args:
        path: Optional path to JSON file with evaluation samples

    Returns:
        List of evaluation samples
    """
    if path and Path(path).exists():
        with open(path, "r") as f:
            data = json.load(f)
            return [EvaluationSample(**sample) for sample in data]

    return get_default_evaluation_dataset()


def save_evaluation_dataset(samples: List[EvaluationSample], path: str) -> None:
    """Save evaluation dataset to JSON file."""
    with open(path, "w") as f:
        json.dump([sample.model_dump() for sample in samples], f, indent=2)


# =============================================================================
# RAGAS Metric Wrappers
# =============================================================================


class RAGASEvaluator:
    """
    Wrapper for RAGAS evaluation metrics.

    Implements graceful degradation when RAGAS or LLM is unavailable.
    """

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        llm_provider: str = "anthropic",
    ):
        """
        Initialize RAGAS evaluator.

        Args:
            config: Evaluation configuration
            llm_provider: LLM provider for metrics (anthropic, openai)
        """
        self.config = config or EvaluationConfig()
        self.llm_provider = llm_provider
        self._ragas_available = self._check_ragas()
        self._llm_configured = self._check_llm()

    def _check_ragas(self) -> bool:
        """Check if RAGAS is available."""
        try:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            return True
        except ImportError:
            logger.warning("RAGAS not installed. Using fallback metrics.")
            return False

    def _check_llm(self) -> bool:
        """Check if LLM is configured for RAGAS."""
        if self.llm_provider == "anthropic":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        elif self.llm_provider == "openai":
            return bool(os.environ.get("OPENAI_API_KEY"))
        return False

    async def evaluate_sample(
        self,
        sample: EvaluationSample,
    ) -> EvaluationResult:
        """
        Evaluate a single sample using RAGAS metrics.

        Args:
            sample: Evaluation sample with query, answer, and contexts

        Returns:
            Evaluation result with metric scores
        """
        sample_id = f"{sample.metadata.get('brand', 'unknown')}_{int(time.time())}"

        if not sample.answer:
            logger.warning(f"Sample {sample_id} has no answer to evaluate")
            return EvaluationResult(
                sample_id=sample_id,
                query=sample.query,
                metadata={"error": "No answer provided"},
            )

        if not sample.retrieved_contexts:
            sample.retrieved_contexts = sample.contexts

        if self._ragas_available and self._llm_configured:
            return await self._evaluate_with_ragas(sample, sample_id)
        else:
            return await self._evaluate_with_fallback(sample, sample_id)

    async def _evaluate_with_ragas(
        self,
        sample: EvaluationSample,
        sample_id: str,
    ) -> EvaluationResult:
        """Evaluate using RAGAS library."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset

            # Prepare dataset in RAGAS format
            data = {
                "question": [sample.query],
                "answer": [sample.answer],
                "contexts": [sample.retrieved_contexts],
                "ground_truth": [sample.ground_truth],
            }
            dataset = Dataset.from_dict(data)

            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )

            # Extract scores
            scores = result.to_pandas().iloc[0].to_dict()

            faith = scores.get("faithfulness", 0.0)
            relevancy = scores.get("answer_relevancy", 0.0)
            precision = scores.get("context_precision", 0.0)
            recall = scores.get("context_recall", 0.0)

            overall = (faith + relevancy + precision + recall) / 4

            # Check thresholds
            passed = all([
                faith >= self.config.thresholds.get("faithfulness", 0.85),
                relevancy >= self.config.thresholds.get("answer_relevancy", 0.90),
                precision >= self.config.thresholds.get("context_precision", 0.80),
                recall >= self.config.thresholds.get("context_recall", 0.80),
            ])

            return EvaluationResult(
                sample_id=sample_id,
                query=sample.query,
                faithfulness=faith,
                answer_relevancy=relevancy,
                context_precision=precision,
                context_recall=recall,
                overall_score=overall,
                passed_thresholds=passed,
                metadata=sample.metadata,
            )

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return await self._evaluate_with_fallback(sample, sample_id)

    async def _evaluate_with_fallback(
        self,
        sample: EvaluationSample,
        sample_id: str,
    ) -> EvaluationResult:
        """
        Fallback evaluation using simple heuristics.

        Used when RAGAS or LLM is unavailable.
        """
        # Simple heuristic-based scoring
        answer = sample.answer.lower() if sample.answer else ""
        ground_truth = sample.ground_truth.lower()
        contexts = " ".join(sample.retrieved_contexts).lower()

        # Faithfulness: How much of answer is in context?
        answer_words = set(answer.split())
        context_words = set(contexts.split())
        if answer_words:
            faith = len(answer_words & context_words) / len(answer_words)
        else:
            faith = 0.0

        # Answer relevancy: How much of ground truth is in answer?
        truth_words = set(ground_truth.split())
        if truth_words:
            relevancy = len(answer_words & truth_words) / len(truth_words)
        else:
            relevancy = 0.0

        # Context precision: Are contexts related to query?
        query_words = set(sample.query.lower().split())
        if query_words:
            precision = len(context_words & query_words) / len(query_words)
        else:
            precision = 0.0

        # Context recall: Do contexts contain ground truth info?
        if truth_words:
            recall = len(context_words & truth_words) / len(truth_words)
        else:
            recall = 0.0

        # Normalize to 0-1 range
        faith = min(faith, 1.0)
        relevancy = min(relevancy, 1.0)
        precision = min(precision, 1.0)
        recall = min(recall, 1.0)

        overall = (faith + relevancy + precision + recall) / 4

        passed = all([
            faith >= self.config.thresholds.get("faithfulness", 0.85),
            relevancy >= self.config.thresholds.get("answer_relevancy", 0.90),
            precision >= self.config.thresholds.get("context_precision", 0.80),
            recall >= self.config.thresholds.get("context_recall", 0.80),
        ])

        return EvaluationResult(
            sample_id=sample_id,
            query=sample.query,
            faithfulness=faith,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
            overall_score=overall,
            passed_thresholds=passed,
            metadata={**sample.metadata, "evaluation_method": "fallback_heuristic"},
        )

    async def evaluate_batch(
        self,
        samples: List[EvaluationSample],
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple samples concurrently.

        Args:
            samples: List of evaluation samples

        Returns:
            List of evaluation results
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def evaluate_with_semaphore(sample: EvaluationSample) -> EvaluationResult:
            async with semaphore:
                return await self.evaluate_sample(sample)

        tasks = [evaluate_with_semaphore(s) for s in samples]
        return await asyncio.gather(*tasks)


# =============================================================================
# Full Evaluation Pipeline
# =============================================================================


class RAGEvaluationPipeline:
    """
    Complete RAG evaluation pipeline with MLflow integration.

    Usage:
        pipeline = RAGEvaluationPipeline()
        report = await pipeline.run_evaluation()
        pipeline.log_to_mlflow(report)
    """

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        dataset_path: Optional[str] = None,
    ):
        """
        Initialize evaluation pipeline.

        Args:
            config: Evaluation configuration
            dataset_path: Path to custom evaluation dataset
        """
        self.config = config or EvaluationConfig()
        self.evaluator = RAGASEvaluator(config=self.config)
        self.dataset = load_evaluation_dataset(dataset_path)

    async def run_evaluation(
        self,
        rag_pipeline: Optional[Any] = None,
    ) -> EvaluationReport:
        """
        Run full evaluation pipeline.

        Args:
            rag_pipeline: Optional RAG pipeline to generate answers.
                          If not provided, uses pre-defined answers.

        Returns:
            Evaluation report with all metrics
        """
        start_time = time.time()
        run_id = f"eval_{int(start_time)}"

        logger.info(f"Starting evaluation run {run_id} with {len(self.dataset)} samples")

        # Generate answers if pipeline provided
        if rag_pipeline:
            await self._generate_answers(rag_pipeline)

        # Evaluate all samples
        results = await self.evaluator.evaluate_batch(self.dataset)

        # Aggregate metrics
        valid_results = [r for r in results if r.faithfulness is not None]

        if valid_results:
            avg_faith = sum(r.faithfulness for r in valid_results) / len(valid_results)
            avg_relevancy = sum(r.answer_relevancy for r in valid_results) / len(valid_results)
            avg_precision = sum(r.context_precision for r in valid_results) / len(valid_results)
            avg_recall = sum(r.context_recall for r in valid_results) / len(valid_results)
            overall = sum(r.overall_score for r in valid_results) / len(valid_results)
        else:
            avg_faith = avg_relevancy = avg_precision = avg_recall = overall = None

        passed_count = sum(1 for r in results if r.passed_thresholds)

        # Check if all thresholds met
        all_passed = (
            avg_faith is not None
            and avg_faith >= self.config.thresholds.get("faithfulness", 0.85)
            and avg_relevancy >= self.config.thresholds.get("answer_relevancy", 0.90)
            and avg_precision >= self.config.thresholds.get("context_precision", 0.80)
            and avg_recall >= self.config.thresholds.get("context_recall", 0.80)
        )

        elapsed = time.time() - start_time

        report = EvaluationReport(
            run_id=run_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_samples=len(self.dataset),
            passed_samples=passed_count,
            failed_samples=len(self.dataset) - passed_count,
            avg_faithfulness=avg_faith,
            avg_answer_relevancy=avg_relevancy,
            avg_context_precision=avg_precision,
            avg_context_recall=avg_recall,
            overall_score=overall,
            thresholds=self.config.thresholds,
            all_thresholds_passed=all_passed,
            results=results,
            evaluation_time_seconds=elapsed,
        )

        logger.info(
            f"Evaluation complete: {passed_count}/{len(self.dataset)} passed, "
            f"overall score: {f'{overall:.3f}' if overall else 'N/A'}"
        )

        return report

    async def _generate_answers(self, rag_pipeline: Any) -> None:
        """Generate answers using RAG pipeline for each sample."""
        for sample in self.dataset:
            if not sample.answer:
                try:
                    # Assuming rag_pipeline has a query method
                    result = await rag_pipeline.query(sample.query)
                    sample.answer = result.get("answer", "")
                    sample.retrieved_contexts = result.get("contexts", sample.contexts)
                except Exception as e:
                    logger.warning(f"Failed to generate answer for: {sample.query}: {e}")
                    sample.answer = ""

    def log_to_mlflow(self, report: EvaluationReport) -> None:
        """
        Log evaluation results to MLflow.

        Args:
            report: Evaluation report to log
        """
        if not self.config.log_to_mlflow:
            return

        try:
            mlflow.set_experiment(self.config.mlflow_experiment)

            with mlflow.start_run(run_name=report.run_id):
                # Log aggregate metrics
                if report.avg_faithfulness is not None:
                    mlflow.log_metric("avg_faithfulness", report.avg_faithfulness)
                if report.avg_answer_relevancy is not None:
                    mlflow.log_metric("avg_answer_relevancy", report.avg_answer_relevancy)
                if report.avg_context_precision is not None:
                    mlflow.log_metric("avg_context_precision", report.avg_context_precision)
                if report.avg_context_recall is not None:
                    mlflow.log_metric("avg_context_recall", report.avg_context_recall)
                if report.overall_score is not None:
                    mlflow.log_metric("overall_score", report.overall_score)

                mlflow.log_metric("total_samples", report.total_samples)
                mlflow.log_metric("passed_samples", report.passed_samples)
                mlflow.log_metric("failed_samples", report.failed_samples)
                mlflow.log_metric("pass_rate", report.passed_samples / report.total_samples)
                mlflow.log_metric("evaluation_time_seconds", report.evaluation_time_seconds)

                # Log thresholds as params
                for name, value in report.thresholds.items():
                    mlflow.log_param(f"threshold_{name}", value)

                mlflow.log_param("all_thresholds_passed", report.all_thresholds_passed)

                # Log detailed results as artifact
                results_path = f"/tmp/{report.run_id}_results.json"
                with open(results_path, "w") as f:
                    json.dump(report.model_dump(), f, indent=2)
                mlflow.log_artifact(results_path)

                logger.info(f"Logged evaluation results to MLflow experiment: {self.config.mlflow_experiment}")

        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")

    def check_thresholds(self, report: EvaluationReport) -> Tuple[bool, List[str]]:
        """
        Check if evaluation meets quality thresholds.

        Args:
            report: Evaluation report to check

        Returns:
            Tuple of (passed, list of failure messages)
        """
        failures = []

        if report.avg_faithfulness is not None:
            threshold = self.config.thresholds.get("faithfulness", 0.85)
            if report.avg_faithfulness < threshold:
                failures.append(
                    f"Faithfulness {report.avg_faithfulness:.3f} < {threshold}"
                )

        if report.avg_answer_relevancy is not None:
            threshold = self.config.thresholds.get("answer_relevancy", 0.90)
            if report.avg_answer_relevancy < threshold:
                failures.append(
                    f"Answer Relevancy {report.avg_answer_relevancy:.3f} < {threshold}"
                )

        if report.avg_context_precision is not None:
            threshold = self.config.thresholds.get("context_precision", 0.80)
            if report.avg_context_precision < threshold:
                failures.append(
                    f"Context Precision {report.avg_context_precision:.3f} < {threshold}"
                )

        if report.avg_context_recall is not None:
            threshold = self.config.thresholds.get("context_recall", 0.80)
            if report.avg_context_recall < threshold:
                failures.append(
                    f"Context Recall {report.avg_context_recall:.3f} < {threshold}"
                )

        return len(failures) == 0, failures


# =============================================================================
# Convenience Functions
# =============================================================================


async def quick_evaluate(
    query: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> EvaluationResult:
    """
    Quick evaluation of a single query-answer pair.

    Args:
        query: User query
        answer: Generated answer
        contexts: Retrieved contexts
        ground_truth: Optional ground truth answer

    Returns:
        Evaluation result
    """
    sample = EvaluationSample(
        query=query,
        ground_truth=ground_truth or answer,
        answer=answer,
        retrieved_contexts=contexts,
    )

    evaluator = RAGASEvaluator()
    return await evaluator.evaluate_sample(sample)


def create_evaluation_sample(
    query: str,
    ground_truth: str,
    contexts: List[str],
    **metadata: Any,
) -> EvaluationSample:
    """
    Create an evaluation sample.

    Args:
        query: User query
        ground_truth: Expected answer
        contexts: Reference contexts
        **metadata: Additional metadata (brand, kpi, etc.)

    Returns:
        Evaluation sample
    """
    return EvaluationSample(
        query=query,
        ground_truth=ground_truth,
        contexts=contexts,
        metadata=metadata,
    )
