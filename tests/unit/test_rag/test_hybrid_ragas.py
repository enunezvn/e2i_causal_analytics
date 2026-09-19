from types import SimpleNamespace

from src.rag.hybrid_ragas import hybrid_runtime_validation_errors


def _sample(runtime, contexts=None):
    return SimpleNamespace(
        query="test query",
        retrieved_contexts=["context"] if contexts is None else contexts,
        metadata={"pipeline_metadata": runtime},
    )


def _result(method="ragas"):
    return SimpleNamespace(metadata={"evaluation_method": method})


def test_hybrid_runtime_validation_accepts_proven_judged_path():
    runtime = {
        "query_optimizer_enabled": True,
        "reranker_enabled": True,
        "reranked": True,
        "retrieval_hit": True,
        "generation_attempted": True,
        "generation_succeeded": True,
    }

    assert hybrid_runtime_validation_errors([_sample(runtime)], [_result()]) == []


def test_hybrid_runtime_validation_rejects_component_and_judge_fallbacks():
    runtime = {
        "query_optimizer_enabled": True,
        "reranker_enabled": True,
        "reranked": False,
        "reranker_error": "TimeoutError",
        "retrieval_hit": True,
        "generation_attempted": True,
        "generation_succeeded": False,
    }

    errors = hybrid_runtime_validation_errors(
        [_sample(runtime)],
        [_result("fallback_heuristic")],
    )

    assert any("reranker failed" in error for error in errors)
    assert any("was not reranked" in error for error in errors)
    assert any("generation did not succeed" in error for error in errors)
    assert any("heuristic fallback" in error for error in errors)


def test_hybrid_runtime_validation_accepts_honest_retrieval_miss():
    runtime = {
        "query_optimizer_enabled": True,
        "reranker_enabled": True,
        "reranked": False,
        "retrieval_hit": False,
        "generation_attempted": False,
        "generation_succeeded": False,
    }

    assert hybrid_runtime_validation_errors([_sample(runtime, contexts=[])], [_result()]) == []


def test_hybrid_runtime_validation_rejects_generation_error():
    runtime = {
        "query_optimizer_enabled": True,
        "reranker_enabled": True,
        "reranked": True,
        "retrieval_hit": True,
        "generation_attempted": True,
        "generation_succeeded": False,
        "generation_error": "insight_enrichment_failed",
    }

    errors = hybrid_runtime_validation_errors([_sample(runtime)], [_result()])

    assert any("answer generation failed" in error for error in errors)
    assert any("answer generation did not succeed" in error for error in errors)


def test_hybrid_runtime_validation_rejects_missing_evaluation_results():
    runtime = {
        "query_optimizer_enabled": True,
        "reranker_enabled": True,
        "reranked": True,
        "retrieval_hit": True,
        "generation_attempted": True,
        "generation_succeeded": True,
    }

    errors = hybrid_runtime_validation_errors([_sample(runtime)], [])

    assert errors == ["evaluation result count mismatch: 0 results for 1 samples"]
