"""Contract tests for the manual hybrid-pipeline RAGAS lane."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import scripts.run_hybrid_pipeline_ragas as runner

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_hybrid_pipeline_ragas.py"


def _grounded_sample():
    return SimpleNamespace(
        query="What is the KPI?",
        ground_truth="Live source content",
        contexts=["Live source content"],
        metadata={
            "dataset_provenance": "live_non_synthetic_rag_document_chunks",
            "source_chunk_id": "chunk-1",
            "source_document_id": "document-1",
        },
    )


def _corpus_rows():
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "document-1",
            "content": "Live source content",
            "is_synthetic": False,
        }
    ]


def test_hybrid_ragas_lane_exercises_live_service():
    text = SCRIPT.read_text()
    assert "RAGService.get_instance()" in text
    assert "run_evaluation(rag_pipeline=service)" in text
    assert "RAG_ENABLE_QUERY_OPTIMIZATION=true" in text
    assert "RAG_ENABLE_RERANKING=true" in text
    assert '"ANTHROPIC_API_KEY", "OPENAI_API_KEY"' in text
    assert "hybrid_runtime_validation_errors" in text


def test_hybrid_ragas_lane_is_distinct_from_fixture_sentinel():
    assert SCRIPT.exists()
    docstring = SCRIPT.read_text().split('"""')[1].lower()
    assert "live hybrid" in docstring
    assert "run_ragas_eval.py" in docstring


@pytest.mark.asyncio
async def test_invalid_runtime_is_not_logged_to_mlflow(monkeypatch, tmp_path):
    output = tmp_path / "report.json"
    dataset = tmp_path / "corpus-grounded.json"
    dataset.write_text("[]")
    args = SimpleNamespace(
        dataset=str(dataset),
        limit=1,
        output=str(output),
        fail_on_threshold=False,
        no_mlflow=False,
    )
    service = SimpleNamespace(query_optimizer=object(), reranker=object())
    report = SimpleNamespace(
        results=[SimpleNamespace(metadata={})],
        all_thresholds_passed=True,
        model_dump=MagicMock(return_value={"total_samples": 1}),
    )
    pipeline = SimpleNamespace(
        dataset=[_grounded_sample()],
        evaluator=SimpleNamespace(
            can_judge=True,
            judged_path_blockers=(),
            verify_dependencies=MagicMock(),
        ),
        run_evaluation=AsyncMock(return_value=report),
        log_to_mlflow=MagicMock(),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner.RAGService, "get_instance", lambda: service)
    monkeypatch.setattr(runner, "RAGEvaluationPipeline", MagicMock(return_value=pipeline))
    monkeypatch.setattr(runner, "load_live_corpus_rows", lambda samples: _corpus_rows())
    monkeypatch.setattr(
        runner,
        "hybrid_runtime_validation_errors",
        lambda samples, results: ["reranker failed"],
    )

    assert await runner.main() == 2
    pipeline.log_to_mlflow.assert_not_called()
    assert output.exists()


@pytest.mark.asyncio
async def test_empty_dataset_is_rejected(monkeypatch, tmp_path):
    dataset = tmp_path / "corpus-grounded.json"
    dataset.write_text("[]")
    args = SimpleNamespace(
        dataset=str(dataset),
        limit=1,
        output=str(tmp_path / "report.json"),
        fail_on_threshold=False,
        no_mlflow=True,
    )
    service = SimpleNamespace(query_optimizer=object(), reranker=object())
    pipeline = SimpleNamespace(
        dataset=[],
        evaluator=SimpleNamespace(),
        run_evaluation=AsyncMock(),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner.RAGService, "get_instance", lambda: service)
    monkeypatch.setattr(runner, "RAGEvaluationPipeline", MagicMock(return_value=pipeline))

    with pytest.raises(SystemExit, match="contains no samples"):
        await runner.main()

    pipeline.run_evaluation.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_simulated_dataset_is_rejected_for_live_quality_gate(monkeypatch, tmp_path):
    args = SimpleNamespace(
        dataset=None,
        limit=1,
        output=str(tmp_path / "report.json"),
        fail_on_threshold=False,
        no_mlflow=True,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(runner, "parse_args", lambda: args)

    with pytest.raises(SystemExit, match="corpus-grounded --dataset"):
        await runner.main()


@pytest.mark.asyncio
async def test_missing_dataset_path_does_not_fall_back_to_simulated_fixture(monkeypatch, tmp_path):
    missing = tmp_path / "missing.json"
    args = SimpleNamespace(
        dataset=str(missing),
        limit=1,
        output=str(tmp_path / "report.json"),
        fail_on_threshold=False,
        no_mlflow=True,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(runner, "parse_args", lambda: args)

    with pytest.raises(SystemExit, match="does not exist"):
        await runner.main()


@pytest.mark.asyncio
async def test_existing_but_unverified_dataset_is_rejected_before_judging(monkeypatch, tmp_path):
    dataset = tmp_path / "simulated.json"
    dataset.write_text("[]")
    args = SimpleNamespace(
        dataset=str(dataset),
        limit=1,
        output=str(tmp_path / "report.json"),
        fail_on_threshold=False,
        no_mlflow=True,
    )
    service = SimpleNamespace(query_optimizer=object(), reranker=object())
    sample = _grounded_sample()
    sample.metadata = {"dataset_provenance": "simulated_fixture"}
    pipeline = SimpleNamespace(
        dataset=[sample],
        evaluator=SimpleNamespace(
            can_judge=True,
            judged_path_blockers=(),
            verify_dependencies=MagicMock(),
        ),
        run_evaluation=AsyncMock(),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner.RAGService, "get_instance", lambda: service)
    monkeypatch.setattr(runner, "RAGEvaluationPipeline", MagicMock(return_value=pipeline))
    monkeypatch.setattr(runner, "load_live_corpus_rows", lambda samples: [])

    with pytest.raises(SystemExit, match="not verified against the live non-synthetic corpus"):
        await runner.main()

    pipeline.run_evaluation.assert_not_awaited()
