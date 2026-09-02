"""Real model-load smoke for the CrossEncoder reranker (no mocks).

Added with the 2026-09-02 transformers 4.57.3->5.10.1 + huggingface-hub
0.36.0->1.5.0 bump: the sentence-transformers CrossEncoder in
src/rag/reranker.py is the ONLY HuggingFace model load in the codebase, and
no other CI test exercises a real hub download + weight load, so a
transformers/hub major bump would otherwise false-green.

Two guards against vacuous greens:

- the load is triggered via ``reranker.model`` DIRECTLY, because
  ``_batch_score`` swallows every exception into fallback scores of exactly
  0.5 — a broken model load would otherwise still "pass" a rerank-only test;
- the score assertions exclude that 0.5 fallback on both sides.

Skips (visibly, never silently) only on genuine network unreachability;
compat breaks (ImportError / TypeError / AttributeError) fail hard.
"""

import pytest

from src.rag.models.retrieval_models import RetrievalResult
from src.rag.reranker import CrossEncoderReranker


def _network_exceptions() -> tuple:
    """Exception types that mean "hub unreachable", not "stack broken".

    Built dynamically because the HTTP client under huggingface-hub differs
    across major versions (requests in 0.x, httpx in 1.x).
    """
    excs: list = [ConnectionError, TimeoutError]
    try:
        import requests

        excs.append(requests.exceptions.RequestException)
    except ImportError:
        pass
    try:
        import httpx

        excs.append(httpx.HTTPError)
    except ImportError:
        pass
    try:
        from huggingface_hub import errors as hub_errors

        for name in ("HfHubHTTPError", "LocalEntryNotFoundError"):
            exc = getattr(hub_errors, name, None)
            if exc is not None:
                excs.append(exc)
    except ImportError:
        pass
    return tuple(excs)


@pytest.mark.timeout(120)
def test_cross_encoder_loads_and_ranks_real_model() -> None:
    reranker = CrossEncoderReranker()

    try:
        model = reranker.model  # hub download (first run) + weight load
    except _network_exceptions() as exc:
        pytest.skip(f"HuggingFace Hub unreachable; model-load smoke not exercised: {exc}")

    assert model is reranker.model  # module-level cache returns the same object

    query = "what drives market share gaps between regions"
    relevant = RetrievalResult(
        content="Kisqali shows a 21% market-share gap in the west region driven by HCP access",
        source="kpi_summaries",
        source_id="smoke-relevant",
        score=0.0,
        retrieval_method="dense",
    )
    irrelevant = RetrievalResult(
        content="the cafeteria menu changed on tuesday to include soup",
        source="kpi_summaries",
        source_id="smoke-irrelevant",
        score=0.0,
        retrieval_method="dense",
    )

    reranked = reranker.rerank([irrelevant, relevant], query, top_k=2)

    assert [r.source_id for r in reranked] == ["smoke-relevant", "smoke-irrelevant"]
    # Real cross-encoder scores, not _batch_score's uniform 0.5 fallback:
    assert reranked[0].score > 0.6
    assert reranked[1].score < 0.4
    for result in reranked:
        assert 0.0 <= result.score <= 1.0
        assert result.metadata["original_score"] == 0.0
