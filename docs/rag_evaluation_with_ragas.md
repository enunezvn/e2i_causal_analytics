# RAG Evaluation with Ragas Framework
**Created**: 2025-12-15
**Status**: Implementation Guide
**Priority**: High

---

## Table of Contents

1. [Overview](#overview)
2. [Ragas Framework Introduction](#ragas-framework-introduction)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Implementation Architecture](#implementation-architecture)
5. [Test Dataset Creation](#test-dataset-creation)
6. [Evaluation Pipeline](#evaluation-pipeline)
7. [Integration with MLflow & Opik](#integration-with-mlflow--opik)
8. [Automated Evaluation Workflows](#automated-evaluation-workflows)
9. [Monitoring & Alerting](#monitoring--alerting)
10. [Best Practices](#best-practices)

---

## Overview

This document describes the implementation of RAG (Retrieval-Augmented Generation) pipeline evaluation using the **Ragas** framework for the E2I Causal Analytics platform.

### Goals

1. **Quantify RAG Performance** - Measure retrieval quality, answer relevance, and faithfulness
2. **Track Improvements** - Monitor metrics over time as system evolves
3. **Identify Weaknesses** - Find specific failure modes in retrieval or generation
4. **Enable A/B Testing** - Compare different retrieval strategies (weights, models, etc.)
5. **Production Monitoring** - Continuous evaluation on real user queries

### Key Metrics (Ragas)

| Metric | Measures | Range | Target |
|--------|----------|-------|--------|
| **Faithfulness** | Answer grounded in retrieved context | 0-1 | >0.8 |
| **Answer Relevancy** | Answer addresses the question | 0-1 | >0.85 |
| **Context Precision** | Relevant chunks ranked high | 0-1 | >0.75 |
| **Context Recall** | All relevant info retrieved | 0-1 | >0.8 |
| **Answer Similarity** | Semantic similarity to ground truth | 0-1 | >0.7 |
| **Answer Correctness** | Factual correctness | 0-1 | >0.75 |

---

## Ragas Framework Introduction

### What is Ragas?

Ragas (Retrieval-Augmented Generation Assessment) is an open-source framework for evaluating RAG pipelines without requiring extensive ground truth labels.

**Key Features**:
- Reference-free metrics (faithfulness, answer relevancy)
- Reference-based metrics (answer similarity, correctness)
- Retrieval-focused metrics (context precision, recall)
- LLM-as-judge for nuanced evaluation
- Integration with LangChain and LangSmith

**Installation**:
```bash
pip install ragas>=0.1.0
```

**Documentation**: https://docs.ragas.io/

### Why Ragas for E2I?

1. **Domain Agnostic** - Works with healthcare/pharma terminology
2. **Minimal Labels** - Can evaluate with just queries (no ground truth needed for some metrics)
3. **LLM-Powered** - Uses Claude/GPT to judge answer quality
4. **Modular** - Evaluate retrieval and generation separately
5. **Production Ready** - Can run on live traffic

---

## Evaluation Metrics

### 1. Faithfulness (Hallucination Detection)

**Measures**: Whether the answer is grounded in the retrieved context.

**How it works**:
1. Extract claims from the generated answer
2. Check if each claim is supported by retrieved contexts
3. Score = (supported claims) / (total claims)

**Implementation**:
```python
from ragas.metrics import faithfulness

# Requires: question, answer, contexts
score = faithfulness.score({
    "question": "Why did Remibrutinib conversion drop in Q4?",
    "answer": "Conversion dropped 8pp due to South region coverage gaps...",
    "contexts": [context1, context2, context3]  # Retrieved chunks
})
```

**When to use**: Critical for medical/pharma domain where accuracy is paramount.

**Threshold**: > 0.8 (80% of claims must be supported)

### 2. Answer Relevancy

**Measures**: Whether the answer actually addresses the question asked.

**How it works**:
1. Generate variations of the question from the answer (using LLM)
2. Calculate similarity between original question and generated variants
3. High similarity = answer is relevant

**Implementation**:
```python
from ragas.metrics import answer_relevancy

score = answer_relevancy.score({
    "question": "What caused the engagement score increase?",
    "answer": "HCP engagement increased by 12% due to...",
    "contexts": [...]  # Optional for this metric
})
```

**Threshold**: > 0.85

### 3. Context Precision

**Measures**: Whether relevant chunks are ranked higher than irrelevant ones.

**How it works**:
1. For each retrieved chunk, determine if it's relevant (using LLM)
2. Calculate precision at k for each position
3. Average across all positions

**Implementation**:
```python
from ragas.metrics import context_precision

score = context_precision.score({
    "question": "What is the causal effect of coverage on TRx?",
    "contexts": [chunk1, chunk2, chunk3],  # Ordered by retriever rank
    "ground_truth": "Coverage has a 0.25 effect size on TRx..."  # Optional
})
```

**When to use**: Evaluate if RRF fusion ranks relevant chunks appropriately.

**Threshold**: > 0.75

### 4. Context Recall

**Measures**: Whether all relevant information was retrieved.

**How it works**:
1. Extract claims from ground truth answer
2. Check what fraction can be attributed to retrieved contexts
3. Score = (attributed claims) / (total ground truth claims)

**Implementation**:
```python
from ragas.metrics import context_recall

score = context_recall.score({
    "question": "...",
    "contexts": [chunk1, chunk2, chunk3],
    "ground_truth": "The complete correct answer..."
})
```

**Threshold**: > 0.8

### 5. Answer Similarity (Optional)

**Measures**: Semantic similarity to reference answer.

**Requires**: Ground truth answers (can be expert-written or human-validated).

**Implementation**:
```python
from ragas.metrics import answer_similarity

score = answer_similarity.score({
    "answer": "Generated answer...",
    "ground_truth": "Expert-written answer..."
})
```

**When to use**: When you have a golden test set with expert answers.

### 6. Answer Correctness (Optional)

**Measures**: Factual correctness considering both similarity and factual overlap.

**Implementation**:
```python
from ragas.metrics import answer_correctness

score = answer_correctness.score({
    "answer": "Generated answer...",
    "ground_truth": "Correct answer..."
})
```

**Combines**: Semantic similarity + fact matching.

---

## Implementation Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   E2I RAG Pipeline                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  Vector    │  │ Fulltext   │  │   Graph    │            │
│  │  Search    │  │  Search    │  │   Search   │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│         │               │               │                    │
│         └───────────────┴───────────────┘                    │
│                     │                                        │
│              ┌──────▼──────┐                                │
│              │ RRF Fusion  │                                │
│              └──────┬──────┘                                │
│                     │                                        │
│              ┌──────▼──────┐                                │
│              │ LLM (Claude)│                                │
│              └──────┬──────┘                                │
│                     │                                        │
│              ┌──────▼──────┐                                │
│              │   Answer    │                                │
│              └──────┬──────┘                                │
│                     │                                        │
└─────────────────────┼────────────────────────────────────────┘
                      │
            ┌─────────▼─────────┐
            │                   │
      ┌─────▼─────┐     ┌──────▼──────┐
      │  Ragas    │     │   Logging   │
      │ Evaluation│     │  (Opik/MLflow)│
      └─────┬─────┘     └──────┬──────┘
            │                  │
      ┌─────▼─────┐     ┌──────▼──────┐
      │  Metrics  │     │  Dashboard  │
      │  Database │     │ (Grafana)   │
      └───────────┘     └─────────────┘
```

### File Structure

```
src/rag/
├── __init__.py
├── hybrid_retriever.py          # Main RAG pipeline
├── health_monitor.py            # Backend health checks
├── evaluation.py                # NEW: Ragas evaluation
└── embeddings.py                # NEW: OpenAI embedding client

tests/evaluation/
├── __init__.py
├── test_ragas_metrics.py        # Unit tests for metrics
├── test_evaluation_pipeline.py  # Integration tests
└── golden_dataset.json          # Test queries + ground truth

config/
├── rag_config.yaml
└── ragas_config.yaml            # NEW: Evaluation configuration
```

---

## Test Dataset Creation

### 1. Golden Test Set Structure

**File**: `tests/evaluation/golden_dataset.json`

```json
{
  "test_cases": [
    {
      "id": "tc_001",
      "category": "causal_impact",
      "question": "What caused the Remibrutinib conversion rate drop in Q4 2024?",
      "ground_truth": "Remibrutinib conversion rate dropped 8 percentage points in Q4 2024 due to South region coverage gaps (70% vs 88% national benchmark). DoWhy analysis confirmed causal effect with p<0.01.",
      "expected_contexts": [
        "causal_paths table - Remibrutinib conversion analysis",
        "agent_activities - gap_analyzer output for Q4 2024"
      ],
      "difficulty": "medium",
      "requires_graph": true,
      "requires_temporal": true
    },
    {
      "id": "tc_002",
      "category": "gap_analysis",
      "question": "What is the ROI opportunity for improving HCP coverage in the South region?",
      "ground_truth": "Improving South region HCP coverage from 70% to 88% (national benchmark) yields estimated $8M captured value over 6 months, driven by 18% coverage increase → -8pp fairness gap reduction → TRx lift.",
      "expected_contexts": [
        "gap_analyzer agent outputs",
        "ROI calculations from agent_activities"
      ],
      "difficulty": "hard",
      "requires_graph": true,
      "requires_calculation": true
    },
    {
      "id": "tc_003",
      "category": "kpi_lookup",
      "question": "What is the current conversion rate for Fabhalta in the Northeast region?",
      "ground_truth": "Fabhalta conversion rate in Northeast is 12.4% as of Dec 2024.",
      "expected_contexts": [
        "patient_journeys table - Fabhalta NE conversions"
      ],
      "difficulty": "easy",
      "requires_graph": false,
      "requires_temporal": false
    }
  ]
}
```

### 2. Test Set Categories

| Category | Count | Description | Example Query |
|----------|-------|-------------|---------------|
| **Causal Impact** | 20 | Causal effect questions | "What caused X?" |
| **Gap Analysis** | 15 | Opportunity identification | "What is the ROI of improving Y?" |
| **KPI Lookup** | 20 | Simple metric retrieval | "What is the current value of Z?" |
| **Temporal Analysis** | 15 | Time-series questions | "How has X changed over time?" |
| **Comparative** | 10 | Cross-brand/region comparisons | "How does A compare to B?" |
| **Multi-Hop** | 10 | Requires multiple reasoning steps | "If we improve X, what happens to Z?" |
| **Graph Traversal** | 10 | Needs causal path walking | "What is the causal chain from A to B?" |

**Total**: 100 test cases

### 3. Creating Test Cases

**Script**: `scripts/generate_test_dataset.py`

```python
"""Generate golden test dataset for RAG evaluation."""

import json
from typing import List, Dict
from dataclasses import dataclass, asdict

@dataclass
class TestCase:
    id: str
    category: str
    question: str
    ground_truth: str
    expected_contexts: List[str]
    difficulty: str
    requires_graph: bool = False
    requires_temporal: bool = False
    requires_calculation: bool = False

def generate_causal_impact_cases() -> List[TestCase]:
    """Generate causal impact test cases."""
    cases = []

    # Example 1: Conversion drop
    cases.append(TestCase(
        id="tc_causal_001",
        category="causal_impact",
        question="What caused the Remibrutinib conversion rate drop in Q4 2024?",
        ground_truth="Remibrutinib conversion rate dropped 8 percentage points in Q4 2024 due to South region coverage gaps (70% vs 88% national benchmark). DoWhy analysis confirmed causal effect with p<0.01.",
        expected_contexts=[
            "causal_paths - Remibrutinib conversion",
            "agent_activities - gap_analyzer Q4 2024"
        ],
        difficulty="medium",
        requires_graph=True,
        requires_temporal=True
    ))

    # Add 19 more causal cases...

    return cases

def generate_gap_analysis_cases() -> List[TestCase]:
    """Generate gap analysis test cases."""
    # Implementation...
    pass

def main():
    """Generate full test dataset."""
    test_cases = []

    test_cases.extend(generate_causal_impact_cases())
    test_cases.extend(generate_gap_analysis_cases())
    # ... other categories

    dataset = {
        "metadata": {
            "version": "1.0",
            "created_at": "2025-12-15",
            "total_cases": len(test_cases),
            "categories": {
                "causal_impact": 20,
                "gap_analysis": 15,
                "kpi_lookup": 20,
                # ... etc
            }
        },
        "test_cases": [asdict(tc) for tc in test_cases]
    }

    with open("tests/evaluation/golden_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(test_cases)} test cases")

if __name__ == "__main__":
    main()
```

---

## Evaluation Pipeline

### 1. Core Evaluation Class

**File**: `src/rag/evaluation.py`

```python
"""RAG evaluation using Ragas framework."""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
from datasets import Dataset

from .hybrid_retriever import HybridRetriever


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    test_case_id: str
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]

    # Metrics
    faithfulness_score: float
    answer_relevancy_score: float
    context_precision_score: Optional[float]
    context_recall_score: Optional[float]
    answer_similarity_score: Optional[float]
    answer_correctness_score: Optional[float]

    # Metadata
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    timestamp: str


class RAGEvaluator:
    """
    Evaluate RAG pipeline using Ragas metrics.

    Supports:
    - Batch evaluation on golden test set
    - Single query evaluation
    - Production traffic sampling
    - A/B testing different configurations
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client,  # Claude client
        enable_all_metrics: bool = True
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.enable_all_metrics = enable_all_metrics

        # Select metrics based on availability of ground truth
        self.metrics_no_gt = [faithfulness, answer_relevancy]
        self.metrics_with_gt = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_similarity,
            answer_correctness
        ]

    def evaluate_test_set(
        self,
        test_dataset_path: str,
        sample_size: Optional[int] = None
    ) -> Dict:
        """
        Evaluate RAG pipeline on golden test dataset.

        Args:
            test_dataset_path: Path to JSON test dataset
            sample_size: Optional limit on number of test cases

        Returns:
            Dictionary with aggregate metrics and per-case results
        """
        # Load test dataset
        with open(test_dataset_path) as f:
            dataset = json.load(f)

        test_cases = dataset["test_cases"]
        if sample_size:
            test_cases = test_cases[:sample_size]

        # Prepare data for Ragas
        questions = []
        answers = []
        contexts_list = []
        ground_truths = []
        results = []

        for tc in test_cases:
            print(f"Evaluating {tc['id']}: {tc['question']}")

            # Run RAG pipeline
            retrieval_start = datetime.utcnow()
            retrieved_results = self.retriever.search(tc["question"])
            retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds() * 1000

            # Extract contexts
            contexts = [r.content for r in retrieved_results]

            # Generate answer using LLM
            generation_start = datetime.utcnow()
            answer = self._generate_answer(tc["question"], contexts)
            generation_time = (datetime.utcnow() - generation_start).total_seconds() * 1000

            # Collect for batch evaluation
            questions.append(tc["question"])
            answers.append(answer)
            contexts_list.append(contexts)
            ground_truths.append(tc.get("ground_truth", ""))

            # Store result
            results.append({
                "test_case_id": tc["id"],
                "question": tc["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": tc.get("ground_truth"),
                "retrieval_latency_ms": retrieval_time,
                "generation_latency_ms": generation_time,
                "total_latency_ms": retrieval_time + generation_time
            })

        # Create Ragas dataset
        ragas_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths
        })

        # Run Ragas evaluation
        metrics = self.metrics_with_gt if all(ground_truths) else self.metrics_no_gt
        evaluation_result = evaluate(ragas_dataset, metrics=metrics)

        # Combine with latency data
        for i, result in enumerate(results):
            for metric in evaluation_result.columns:
                if metric not in ["question", "answer", "contexts", "ground_truth"]:
                    result[metric] = evaluation_result[metric][i]

        # Calculate aggregate statistics
        aggregate = {
            "total_cases": len(test_cases),
            "mean_faithfulness": evaluation_result["faithfulness"].mean(),
            "mean_answer_relevancy": evaluation_result["answer_relevancy"].mean(),
            "mean_retrieval_latency_ms": sum(r["retrieval_latency_ms"] for r in results) / len(results),
            "mean_generation_latency_ms": sum(r["generation_latency_ms"] for r in results) / len(results),
            "mean_total_latency_ms": sum(r["total_latency_ms"] for r in results) / len(results)
        }

        if "context_precision" in evaluation_result.columns:
            aggregate["mean_context_precision"] = evaluation_result["context_precision"].mean()
            aggregate["mean_context_recall"] = evaluation_result["context_recall"].mean()
            aggregate["mean_answer_similarity"] = evaluation_result["answer_similarity"].mean()
            aggregate["mean_answer_correctness"] = evaluation_result["answer_correctness"].mean()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_path": test_dataset_path,
            "aggregate_metrics": aggregate,
            "per_case_results": results,
            "ragas_full_report": evaluation_result.to_pandas().to_dict()
        }

    def evaluate_single_query(
        self,
        question: str,
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a single query.

        Useful for:
        - Debugging specific failures
        - Production traffic sampling
        - Real-time evaluation
        """
        # Retrieve
        retrieval_start = datetime.utcnow()
        retrieved_results = self.retriever.search(question)
        retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds() * 1000

        contexts = [r.content for r in retrieved_results]

        # Generate
        generation_start = datetime.utcnow()
        answer = self._generate_answer(question, contexts)
        generation_time = (datetime.utcnow() - generation_start).total_seconds() * 1000

        # Evaluate
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }

        if ground_truth:
            data["ground_truth"] = [ground_truth]
            metrics = self.metrics_with_gt
        else:
            metrics = self.metrics_no_gt

        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=metrics)

        return EvaluationResult(
            test_case_id="single_query",
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            faithfulness_score=result["faithfulness"][0],
            answer_relevancy_score=result["answer_relevancy"][0],
            context_precision_score=result.get("context_precision", [None])[0],
            context_recall_score=result.get("context_recall", [None])[0],
            answer_similarity_score=result.get("answer_similarity", [None])[0],
            answer_correctness_score=result.get("answer_correctness", [None])[0],
            retrieval_latency_ms=retrieval_time,
            generation_latency_ms=generation_time,
            total_latency_ms=retrieval_time + generation_time,
            timestamp=datetime.utcnow().isoformat()
        )

    def compare_configurations(
        self,
        test_dataset_path: str,
        configs: List[Dict],
        config_names: List[str]
    ) -> Dict:
        """
        A/B test different RAG configurations.

        Example configs to compare:
        - Different weight distributions (vector vs fulltext vs graph)
        - Different RRF k values
        - Different graph boost factors
        - Different top_k values
        """
        results = {}

        for config, name in zip(configs, config_names):
            print(f"\nEvaluating configuration: {name}")

            # Update retriever config
            self.retriever.config.update(config)

            # Run evaluation
            eval_result = self.evaluate_test_set(test_dataset_path)
            results[name] = eval_result["aggregate_metrics"]

        # Compare results
        comparison = {
            "configurations": config_names,
            "results": results,
            "winner": self._determine_winner(results)
        }

        return comparison

    def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate answer using LLM with retrieved contexts."""
        # Construct prompt
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""You are a causal analytics expert for pharmaceutical drug adoption.

Using the following retrieved context, answer the question accurately and concisely.

CONTEXT:
{context_str}

QUESTION: {question}

ANSWER (be factual, cite specific numbers/evidence from context):"""

        # Call Claude
        response = self.llm_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _determine_winner(self, results: Dict[str, Dict]) -> str:
        """Determine best configuration based on aggregate metrics."""
        # Weighted scoring
        scores = {}

        for config_name, metrics in results.items():
            score = (
                metrics["mean_faithfulness"] * 0.3 +
                metrics["mean_answer_relevancy"] * 0.3 +
                metrics.get("mean_context_precision", 0) * 0.2 +
                metrics.get("mean_context_recall", 0) * 0.2
            )
            scores[config_name] = score

        return max(scores, key=scores.get)
```

---

## Integration with MLflow & Opik

### 1. Logging to MLflow

**File**: `src/rag/evaluation.py` (add to `RAGEvaluator`)

```python
import mlflow

class RAGEvaluator:
    # ... existing code ...

    def log_to_mlflow(self, evaluation_result: Dict, experiment_name: str = "rag_evaluation"):
        """Log evaluation results to MLflow."""

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
            # Log aggregate metrics
            aggregate = evaluation_result["aggregate_metrics"]
            mlflow.log_metrics({
                "faithfulness": aggregate["mean_faithfulness"],
                "answer_relevancy": aggregate["mean_answer_relevancy"],
                "context_precision": aggregate.get("mean_context_precision", 0),
                "context_recall": aggregate.get("mean_context_recall", 0),
                "retrieval_latency_ms": aggregate["mean_retrieval_latency_ms"],
                "generation_latency_ms": aggregate["mean_generation_latency_ms"],
                "total_latency_ms": aggregate["mean_total_latency_ms"]
            })

            # Log configuration
            mlflow.log_params({
                "vector_weight": self.retriever.config.vector_weight,
                "fulltext_weight": self.retriever.config.fulltext_weight,
                "graph_weight": self.retriever.config.graph_weight,
                "rrf_k": self.retriever.config.rrf_k,
                "graph_boost_factor": self.retriever.config.graph_boost_factor,
                "final_top_k": self.retriever.config.final_top_k
            })

            # Log full results as artifact
            with open("evaluation_results.json", "w") as f:
                json.dump(evaluation_result, f, indent=2)
            mlflow.log_artifact("evaluation_results.json")

            print(f"Logged to MLflow experiment: {experiment_name}")
```

### 2. Logging to Opik

```python
from opik import track

class RAGEvaluator:
    # ... existing code ...

    @track()
    def evaluate_with_opik_tracking(
        self,
        test_dataset_path: str,
        sample_size: Optional[int] = None
    ) -> Dict:
        """Evaluate with Opik tracking for observability."""

        # Run standard evaluation
        result = self.evaluate_test_set(test_dataset_path, sample_size)

        # Opik will automatically track:
        # - Function inputs/outputs
        # - Latency
        # - Token usage (if configured)

        # Manually log custom metrics
        from opik import opik_context
        opik_context.update_current_trace(
            metadata={
                "faithfulness": result["aggregate_metrics"]["mean_faithfulness"],
                "answer_relevancy": result["aggregate_metrics"]["mean_answer_relevancy"],
                "total_cases": result["aggregate_metrics"]["total_cases"]
            },
            tags=["rag_evaluation", "ragas"]
        )

        return result
```

---

## Automated Evaluation Workflows

### 1. Daily Evaluation Job

**File**: `scripts/run_daily_evaluation.py`

```python
"""Run automated daily RAG evaluation."""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.hybrid_retriever import HybridRetriever
from src.rag.evaluation import RAGEvaluator
from anthropic import Anthropic
from supabase import create_client
from falkordb import FalkorDB


def main():
    """Run daily evaluation and log results."""

    print(f"Starting daily RAG evaluation at {datetime.utcnow()}")

    # Initialize clients
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY")
    )
    falkordb = FalkorDB(host=os.getenv("FALKORDB_HOST", "localhost"))
    claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Initialize RAG components
    retriever = HybridRetriever(supabase=supabase, falkordb=falkordb)
    evaluator = RAGEvaluator(retriever=retriever, llm_client=claude)

    # Run evaluation
    result = evaluator.evaluate_test_set(
        test_dataset_path="tests/evaluation/golden_dataset.json",
        sample_size=None  # Evaluate full test set
    )

    # Log to MLflow
    evaluator.log_to_mlflow(result, experiment_name="daily_rag_evaluation")

    # Check for regressions
    check_for_regressions(result)

    print("Daily evaluation complete")
    return result


def check_for_regressions(result: dict):
    """Check if metrics have regressed below thresholds."""

    aggregate = result["aggregate_metrics"]

    thresholds = {
        "mean_faithfulness": 0.8,
        "mean_answer_relevancy": 0.85,
        "mean_context_precision": 0.75,
        "mean_context_recall": 0.8
    }

    failures = []
    for metric, threshold in thresholds.items():
        if metric in aggregate and aggregate[metric] < threshold:
            failures.append(f"{metric}: {aggregate[metric]:.3f} < {threshold}")

    if failures:
        print("⚠️  REGRESSION DETECTED:")
        for failure in failures:
            print(f"  - {failure}")

        # Send alert (email, Slack, etc.)
        send_alert(failures)
    else:
        print("✅ All metrics above thresholds")


def send_alert(failures: list):
    """Send alert for metric regressions."""
    # Implement notification logic
    # - Email
    # - Slack webhook
    # - PagerDuty
    pass


if __name__ == "__main__":
    main()
```

### 2. Cron Job Setup

```bash
# Add to crontab
# Run daily at 2 AM UTC
0 2 * * * /path/to/venv/bin/python /path/to/scripts/run_daily_evaluation.py >> /var/log/rag_evaluation.log 2>&1
```

### 3. GitHub Actions Workflow

**File**: `.github/workflows/rag_evaluation.yml`

```yaml
name: RAG Evaluation

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:      # Manual trigger

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run evaluation
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
          FALKORDB_HOST: ${{ secrets.FALKORDB_HOST }}
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python scripts/run_daily_evaluation.py

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-results
          path: evaluation_results.json
```

---

## Monitoring & Alerting

### 1. Grafana Dashboard

**Metrics to monitor**:
- Faithfulness (line chart over time)
- Answer Relevancy (line chart)
- Context Precision & Recall (line chart)
- Retrieval Latency (histogram)
- Generation Latency (histogram)
- Total Latency P50, P95, P99

**Panels**:
1. **Metric Trends** - All Ragas metrics over last 30 days
2. **Latency Distribution** - Histogram of retrieval/generation times
3. **Failure Rate** - % of queries below threshold
4. **Source Distribution** - % using vector vs fulltext vs graph
5. **Test Case Performance** - Heatmap by category

### 2. Alerting Rules

**Alert on**:
- Faithfulness < 0.8 for 2 consecutive days
- Answer Relevancy < 0.85 for 2 consecutive days
- P95 latency > 5 seconds
- Any metric drops >10% week-over-week

---

## Best Practices

### 1. Test Set Maintenance

- ✅ Review test set quarterly
- ✅ Add new cases for edge cases found in production
- ✅ Remove outdated cases (e.g., deprecated brands)
- ✅ Balance difficulty levels (easy/medium/hard)
- ✅ Cover all query categories

### 2. Evaluation Frequency

- **Daily**: Full test set evaluation (automated)
- **Weekly**: Manual review of failures
- **Monthly**: Update test set, review trends
- **Quarterly**: A/B test new configurations

### 3. Metric Interpretation

- **Faithfulness < 0.7**: High hallucination risk - investigate LLM prompts
- **Answer Relevancy < 0.8**: Retrieval likely off-topic - check query understanding
- **Context Precision < 0.7**: RRF weights may need tuning
- **Context Recall < 0.7**: Missing relevant chunks - increase top_k or improve chunking

### 4. A/B Testing Protocol

1. Define hypothesis (e.g., "Increasing graph weight improves causal queries")
2. Create variant configuration
3. Run on same test set
4. Compare aggregate metrics
5. If improvement > 5%, run production A/B test
6. Monitor for 1 week before full rollout

### 5. Production Sampling

- Sample 1-5% of production queries
- Evaluate without ground truth (faithfulness + relevancy only)
- Log to Opik for debugging
- Flag low-scoring queries for manual review

---

## Appendix

### A. Ragas Configuration File

**File**: `config/ragas_config.yaml`

```yaml
ragas:
  # Evaluation settings
  evaluation:
    enable_evaluation: true
    evaluation_interval: daily  # daily, weekly, on_demand
    test_set_size: 100
    sample_production_queries: true
    production_sample_rate: 0.05  # 5% of queries

  # Metrics to compute
  metrics:
    - faithfulness
    - answer_relevancy
    - context_precision
    - context_recall
    - answer_similarity  # Only if ground truth available
    - answer_correctness  # Only if ground truth available

  # Thresholds for alerts
  thresholds:
    faithfulness: 0.8
    answer_relevancy: 0.85
    context_precision: 0.75
    context_recall: 0.8
    answer_similarity: 0.7
    answer_correctness: 0.75

  # LLM for evaluation (Ragas uses LLM-as-judge)
  evaluation_llm:
    provider: anthropic  # or openai
    model: claude-sonnet-4-5-20250929
    temperature: 0.0

  # Logging
  logging:
    log_to_mlflow: true
    mlflow_experiment_name: rag_evaluation
    log_to_opik: true
    log_to_console: true
    save_results_to_file: true
    results_directory: ./evaluation_results

# A/B Testing
ab_testing:
  enable: false
  configurations:
    - name: baseline
      vector_weight: 0.4
      fulltext_weight: 0.2
      graph_weight: 0.4
    - name: graph_heavy
      vector_weight: 0.3
      fulltext_weight: 0.2
      graph_weight: 0.5
```

### B. Example Evaluation Report

```json
{
  "timestamp": "2025-12-15T10:30:00Z",
  "dataset_path": "tests/evaluation/golden_dataset.json",
  "aggregate_metrics": {
    "total_cases": 100,
    "mean_faithfulness": 0.87,
    "mean_answer_relevancy": 0.91,
    "mean_context_precision": 0.79,
    "mean_context_recall": 0.83,
    "mean_answer_similarity": 0.76,
    "mean_answer_correctness": 0.81,
    "mean_retrieval_latency_ms": 423,
    "mean_generation_latency_ms": 1547,
    "mean_total_latency_ms": 1970
  },
  "per_category_metrics": {
    "causal_impact": {
      "count": 20,
      "mean_faithfulness": 0.89,
      "mean_answer_relevancy": 0.93
    },
    "gap_analysis": {
      "count": 15,
      "mean_faithfulness": 0.85,
      "mean_answer_relevancy": 0.88
    }
  },
  "failures": [
    {
      "test_case_id": "tc_042",
      "question": "...",
      "faithfulness_score": 0.65,
      "reason": "Answer included unsupported claim about Q1 2025"
    }
  ]
}
```

---

**Document Version**: 1.0
**Last Updated**: 2025-12-15
**Next Review**: After first evaluation run
