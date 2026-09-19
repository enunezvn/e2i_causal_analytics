from types import SimpleNamespace

from src.rag.hybrid_ragas import (
    corpus_grounding_validation_errors,
    hybrid_runtime_validation_errors,
)


def _sample(runtime, contexts=None):
    return SimpleNamespace(
        query="test query",
        retrieved_contexts=["context"] if contexts is None else contexts,
        metadata={"pipeline_metadata": runtime},
    )


def _result(method="ragas", **overrides):
    values = {
        "faithfulness": 0.9,
        "answer_relevancy": 0.9,
        "context_precision": 0.9,
        "context_recall": 0.9,
        "overall_score": 0.9,
    }
    values.update(overrides)
    return SimpleNamespace(metadata={"evaluation_method": method}, **values)


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


def test_hybrid_runtime_validation_allows_disclosed_degraded_backend_hit():
    runtime = {
        "query_optimizer_enabled": True,
        "reranker_enabled": True,
        "reranked": True,
        "retrieval_hit": True,
        "generation_attempted": True,
        "generation_succeeded": True,
        "errors": ["vector:VectorSearchTimeoutError"],
    }

    assert hybrid_runtime_validation_errors([_sample(runtime)], [_result()]) == []


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


def test_hybrid_runtime_validation_requires_explicit_complete_ragas_judgment():
    runtime = {
        "query_optimizer_enabled": True,
        "reranker_enabled": True,
        "reranked": True,
        "retrieval_hit": True,
        "generation_attempted": True,
        "generation_succeeded": True,
    }
    missing_attestation = _result(method=None)
    missing_attestation.metadata = {}
    incomplete = _result(context_recall=None)
    incomplete.metadata["unmeasured_metrics"] = ["context_recall"]

    unattested_errors = hybrid_runtime_validation_errors(
        [_sample(runtime)], [missing_attestation]
    )
    incomplete_errors = hybrid_runtime_validation_errors([_sample(runtime)], [incomplete])

    assert any("not explicitly attested" in error for error in unattested_errors)
    assert any("context_recall is not a finite score" in error for error in incomplete_errors)
    assert any("unmeasured RAGAS metrics" in error for error in incomplete_errors)


def _grounded_sample(**metadata_overrides):
    metadata = {
        "dataset_provenance": "live_non_synthetic_rag_document_chunks",
        "source_chunk_id": "chunk-1",
        "source_document_id": "document-1",
    }
    metadata.update(metadata_overrides)
    return SimpleNamespace(
        query="What is the KPI?",
        ground_truth="The source KPI content.",
        contexts=["The source KPI content."],
        metadata=metadata,
    )


def _corpus_row(**overrides):
    row = {
        "chunk_id": "chunk-1",
        "document_id": "document-1",
        "content": "The source KPI content.",
        "is_synthetic": False,
    }
    row.update(overrides)
    return row


def test_corpus_grounding_validation_accepts_exact_live_source():
    assert corpus_grounding_validation_errors(
        [_grounded_sample()], [_corpus_row()]
    ) == []


def test_corpus_grounding_validation_rejects_unverified_or_modified_sources():
    errors = corpus_grounding_validation_errors(
        [
            _grounded_sample(dataset_provenance="simulated_fixture"),
            _grounded_sample(source_chunk_id="missing-chunk"),
            _grounded_sample(source_document_id="wrong-document"),
            SimpleNamespace(
                query="Modified reference",
                ground_truth="Invented answer",
                contexts=["Invented context"],
                metadata={
                    "dataset_provenance": "live_non_synthetic_rag_document_chunks",
                    "source_chunk_id": "chunk-1",
                    "source_document_id": "document-1",
                },
            ),
        ],
        [_corpus_row()],
    )

    assert any("unsupported dataset provenance" in error for error in errors)
    assert any("source chunk was not found" in error for error in errors)
    assert any("source document ID does not match" in error for error in errors)
    assert any("ground truth does not exactly match" in error for error in errors)
    assert any("reference contexts do not contain" in error for error in errors)


def test_corpus_grounding_validation_rejects_synthetic_source():
    errors = corpus_grounding_validation_errors(
        [_grounded_sample()], [_corpus_row(is_synthetic=True)]
    )

    assert any("synthetic" in error for error in errors)
