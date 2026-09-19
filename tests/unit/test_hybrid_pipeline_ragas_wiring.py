"""Contract tests for the manual hybrid-pipeline RAGAS lane."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import scripts.run_hybrid_pipeline_ragas as runner

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_hybrid_pipeline_ragas.py"


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
    args = SimpleNamespace(
        dataset=None,
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
        dataset=[SimpleNamespace()],
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
    args = SimpleNamespace(
        dataset=None,
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
