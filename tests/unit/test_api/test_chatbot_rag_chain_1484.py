"""Red-first tests for #1484: retrieve_rag chain async migration + hop economics + spans.

Measured on the prod box (2026-08-05, real LLM + DB calls, novel queries):

* Live steady-state band on fb268e00: retrieve_rag 23.0-32.4s (n=6, classify
  in-band 1.2-1.9s => structural cost, not provider elevation).
* Local per-leg attribution (n=2, totals 34.0s/27.7s): evidence scoring
  15 calls / 17.8s (~53%), rewrite ~5s, hop decider ~4.9s, search ~6.3s.
  The 15 score calls had ZERO overlapping intervals — the F5 asyncio.gather
  is serialized end-to-end because ``score_evidence_dspy`` calls the DSPy
  module SYNCHRONOUSLY inside each coroutine (the existing
  test_chatbot_dspy_evidence_scoring_parallel.py proves concurrency only at
  the coroutine layer, with async stubs that hide the sync seam below).
* dspy 3.1.0 ``Predict.acall`` on the SAME standard-tier model: 5 novel
  scores in 2.6s wall vs 10.0s serial (3.8x). Fast-tier (gpt-5.6-luna) was
  measured SLOWER (median 2412ms vs 1555ms) with 1/5 keep-gate flips —
  that lever is dead, the async one is real.
* Both attribution queries ran to the 3-hop max, kept ZERO evidence, and
  hops 2-3 largely re-retrieved (and re-scored) the same rows — pure waste.

These tests pin:

1.  ``score_evidence_dspy`` / ``rewrite_query_dspy`` / the hop-decider call
    use the module's ASYNC interface (``acall``), never sync ``__call__``.
2.  Each LLM leg has a fail-open timeout (``CHATBOT_RAG_LLM_TIMEOUT_S``).
3.  Rows already scored this request are not re-scored on later hops.
4.  Two consecutive hops keeping zero NEW evidence stop the loop before the
    next decider call (``CHATBOT_RAG_DRY_HOP_LIMIT``, 0 disables).
5.  Chain legs record onto the #1475 stage-timing ledger (``rag.*`` keys).
6.  ``CognitiveRAGResult.score_calls`` reports real LLM-scoring call count.
7.  ``ChatbotTraceContext`` gains ``rag_stage_ms`` / ``rag_meta``; the span
    payload, the request-span INFO line, MLflow metrics, and the node-level
    ledger transfer all carry them (same surfaces as #1471/#1475).
"""

import asyncio
import logging
import time

import pytest

import src.api.routes.chatbot_dspy as cd
import src.api.routes.chatbot_graph as cg
from src.api.routes.chatbot_tracer import ChatbotTraceContext
from src.rag.models.retrieval_models import RetrievalResult
from src.utils.stage_timing import (
    activate_stage_ledger,
    deactivate_stage_ledger,
)

pytestmark = pytest.mark.asyncio


class _Pred:
    """dspy.Prediction stand-in — attribute bag."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _rows(n: int, prefix: str = "row") -> list[RetrievalResult]:
    return [
        RetrievalResult(
            content=f"evidence {prefix}-{i} about kisqali trx northeast",
            source="episodic_memories",
            source_id=f"{prefix}-{i}",
            score=0.5,
            retrieval_method="dense",
        )
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _enable_cognitive(monkeypatch):
    """Force the DSPy cognitive path on and neutralize real LM configuration."""
    monkeypatch.setattr(cd, "DSPY_AVAILABLE", True)
    monkeypatch.setattr(cd, "CHATBOT_COGNITIVE_RAG_ENABLED", True)
    monkeypatch.setattr(cd, "_ensure_dspy_configured", lambda: None)


# =============================================================================
# 1. Async interface contract — the seam the F5 test could not see
# =============================================================================


class _StubScorer:
    """Sync path and async path return DIFFERENT scores so the test can tell
    which interface production code used."""

    def __call__(self, **kw):
        return _Pred(relevance_score=0.9, key_insight="sync", follow_up_needed=False)

    async def acall(self, **kw):
        return _Pred(relevance_score=0.7, key_insight="async", follow_up_needed=True)

    def set_lm(self, lm):  # pragma: no cover - interface parity
        pass


async def test_score_evidence_uses_async_module_interface(monkeypatch):
    monkeypatch.setattr(cd, "ChatbotEvidenceScorer", _StubScorer)
    score, insight, follow = await cd.score_evidence_dspy(
        investigation_goal="Answer: q",
        evidence_item="some evidence",
        source_memory="episodic",
    )
    assert score == pytest.approx(0.7), (
        "score_evidence_dspy must call the scorer's async interface (acall); "
        f"got the sync __call__ result instead (score={score})"
    )
    assert insight == "async"


class _StubRewriter:
    def __call__(self, **kw):
        return _Pred(
            rewritten_query="SYNC", search_keywords="kisqali,trx", graph_entities="Kisqali"
        )

    async def acall(self, **kw):
        return _Pred(
            rewritten_query="ASYNC", search_keywords="kisqali,trx", graph_entities="Kisqali"
        )


async def test_rewrite_uses_async_module_interface(monkeypatch):
    monkeypatch.setattr(cd, "_get_dspy_query_rewriter", lambda: _StubRewriter())
    rewritten, keywords, entities, method = await cd.rewrite_query_dspy(query="q")
    assert method == "dspy"
    assert rewritten == "ASYNC", (
        "rewrite_query_dspy must call the rewriter's async interface (acall)"
    )


async def test_hop_decider_uses_async_module_interface(monkeypatch):
    """Drive one extra hop; the hop-2 search query must come from acall."""
    seen_queries: list[str] = []

    async def fake_rewrite(**kw):
        return ("rewritten q", ["kisqali"], ["Kisqali"], "dspy")

    async def fake_score(investigation_goal, evidence_item, source_memory="episodic"):
        return (0.1, "meh", False)  # dry: nothing clears the 0.3 keep-gate

    async def fake_hybrid_search(query, k, entities=None, kpi_name=None, filters=None, **kwargs):
        seen_queries.append(query)
        return _rows(5)

    class _StubDecider:
        def __call__(self, **kw):
            return _Pred(next_memory="episodic", confidence=0.9, retrieval_query="sync hop q")

        async def acall(self, **kw):
            return _Pred(next_memory="episodic", confidence=0.9, retrieval_query="async hop q")

    monkeypatch.setattr(cd, "rewrite_query_dspy", fake_rewrite)
    monkeypatch.setattr(cd, "score_evidence_dspy", fake_score)
    monkeypatch.setattr(cd, "_get_dspy_hop_decider", lambda: _StubDecider())
    # #1518: this test pins the decider's ASYNC interface; the empty-board
    # skip would bypass the decider entirely on this dry scenario, so pin the
    # legacy path explicitly (still live for non-empty boards / knob=false).
    monkeypatch.setattr(cd, "_SKIP_EMPTY_DECIDER", False)
    import src.rag.retriever as rt

    monkeypatch.setattr(rt, "hybrid_search", fake_hybrid_search)

    await cd.cognitive_rag_retrieve(query="q", enable_multi_hop=True, collect_signal=False)

    assert "async hop q" in seen_queries, (
        "the hop decider must be invoked via its async interface (acall); "
        f"hop queries seen: {seen_queries}"
    )
    assert "sync hop q" not in seen_queries


# =============================================================================
# 2. Fail-open timeouts on every RAG LLM leg
# =============================================================================


async def test_score_timeout_fails_open(monkeypatch):
    class _HungScorer(_StubScorer):
        async def acall(self, **kw):
            await asyncio.sleep(5.0)
            return _Pred(relevance_score=0.7, key_insight="late", follow_up_needed=False)

    monkeypatch.setattr(cd, "ChatbotEvidenceScorer", _HungScorer)
    monkeypatch.setattr(cd, "_RAG_LLM_TIMEOUT_S", 0.05, raising=False)
    t0 = time.monotonic()
    score, insight, follow = await cd.score_evidence_dspy(
        investigation_goal="Answer: q", evidence_item="ev", source_memory="episodic"
    )
    assert time.monotonic() - t0 < 1.0, "hung scorer must be cut off by the leg timeout"
    assert score == pytest.approx(0.5), "timeout must fail open to the neutral 0.5 score"
    assert follow is False


async def test_rewrite_timeout_falls_back_to_hardcoded(monkeypatch):
    class _HungRewriter(_StubRewriter):
        async def acall(self, **kw):
            await asyncio.sleep(5.0)
            return _Pred(rewritten_query="late", search_keywords="", graph_entities="")

    monkeypatch.setattr(cd, "_get_dspy_query_rewriter", lambda: _HungRewriter())
    monkeypatch.setattr(cd, "_RAG_LLM_TIMEOUT_S", 0.05, raising=False)
    t0 = time.monotonic()
    rewritten, keywords, entities, method = await cd.rewrite_query_dspy(
        query="TRx trend for Kisqali"
    )
    assert time.monotonic() - t0 < 1.0
    assert method == "hardcoded", "rewrite timeout must fail open to the hardcoded rewriter"


# =============================================================================
# 3+4. Hop-loop economics: dedupe-before-scoring + dry-hop early exit
# =============================================================================


def _dry_loop_fixtures(monkeypatch, hop_rows_fn):
    """Wire a dry multi-hop run: every row scores 0.1 (< 0.3 keep-gate)."""
    score_items: list[str] = []
    decider_calls = {"n": 0}
    hop_queries: list[str] = []

    async def fake_rewrite(**kw):
        return ("rewritten q", ["kisqali"], ["Kisqali"], "dspy")

    async def fake_score(investigation_goal, evidence_item, source_memory="episodic"):
        score_items.append(evidence_item)
        return (0.1, "meh", False)

    async def fake_hybrid_search(query, k, entities=None, kpi_name=None, filters=None, **kwargs):
        hop_queries.append(query)
        return hop_rows_fn(len(hop_queries))

    class _Decider:
        def __call__(self, **kw):  # pragma: no cover - async contract pinned above
            return self._pred()

        async def acall(self, **kw):
            return self._pred()

        def _pred(self):
            decider_calls["n"] += 1
            return _Pred(
                next_memory="episodic",
                confidence=0.9,
                retrieval_query=f"refined q {decider_calls['n']}",
            )

    monkeypatch.setattr(cd, "rewrite_query_dspy", fake_rewrite)
    monkeypatch.setattr(cd, "score_evidence_dspy", fake_score)
    monkeypatch.setattr(cd, "_get_dspy_hop_decider", lambda: _Decider())
    import src.rag.retriever as rt

    monkeypatch.setattr(rt, "hybrid_search", fake_hybrid_search)
    return score_items, decider_calls, hop_queries


async def test_rows_already_scored_are_not_rescored_on_later_hops(monkeypatch):
    """Attribution showed hops 2-3 re-retrieving the same rows and re-scoring
    them (visible locally as cache-hit scores; in prod each is a real call)."""
    score_items, _, _ = _dry_loop_fixtures(monkeypatch, lambda hop: _rows(5))
    result = await cd.cognitive_rag_retrieve(query="q", enable_multi_hop=True, collect_signal=False)
    assert len(score_items) == 5, (
        f"identical rows must be scored once per request, not per hop; "
        f"saw {len(score_items)} scoring calls"
    )
    assert result.score_calls == 5


async def test_two_consecutive_dry_hops_stop_the_loop(monkeypatch):
    """Hop 1 kept nothing -> ONE refinement chance; if that hop also keeps
    nothing new, stop — never pay the second decider + third hop.

    #1518: pins the LLM-decider refinement path (knob=false); the default
    empty-board skip variant is covered in test_chatbot_rag_chain_1518.py."""
    _, decider_calls, hop_queries = _dry_loop_fixtures(monkeypatch, lambda hop: _rows(5))
    monkeypatch.setattr(cd, "_SKIP_EMPTY_DECIDER", False)
    result = await cd.cognitive_rag_retrieve(query="q", enable_multi_hop=True, collect_signal=False)
    assert decider_calls["n"] == 1, (
        f"expected exactly one decider call after a dry hop-1, got {decider_calls['n']}"
    )
    assert result.hop_count == 2
    assert len(hop_queries) == 2


async def test_dry_hop_limit_zero_disables_early_exit(monkeypatch):
    # #1518: legacy run-to-max means the LLM decider fires each hop — pin the
    # knob=false path (the default skip variant lives in the 1518 file).
    _, decider_calls, hop_queries = _dry_loop_fixtures(monkeypatch, lambda hop: _rows(5))
    monkeypatch.setattr(cd, "_SKIP_EMPTY_DECIDER", False)
    monkeypatch.setattr(cd, "_DRY_HOP_LIMIT", 0, raising=False)
    result = await cd.cognitive_rag_retrieve(query="q", enable_multi_hop=True, collect_signal=False)
    # legacy behavior: decider fires until max hops
    assert result.hop_count == cd._MULTIHOP_MAX_HOPS
    assert decider_calls["n"] == cd._MULTIHOP_MAX_HOPS - 1


async def test_rich_evidence_still_stops_before_any_decider(monkeypatch):
    """Control: the existing strong-evidence early stop is unchanged."""
    score_items, decider_calls, hop_queries = _dry_loop_fixtures(monkeypatch, lambda hop: _rows(5))

    async def rich_score(investigation_goal, evidence_item, source_memory="episodic"):
        score_items.append(evidence_item)
        return (0.8, "great", False)

    monkeypatch.setattr(cd, "score_evidence_dspy", rich_score)
    result = await cd.cognitive_rag_retrieve(query="q", enable_multi_hop=True, collect_signal=False)
    assert decider_calls["n"] == 0
    assert result.hop_count == 1
    assert len(result.evidence) == 5


# =============================================================================
# 5. Ledger recording for chain legs
# =============================================================================


async def test_chain_legs_record_on_stage_ledger(monkeypatch):
    _dry_loop_fixtures(monkeypatch, lambda hop: _rows(5))
    ledger, token = activate_stage_ledger()
    try:
        await cd.cognitive_rag_retrieve(query="q", enable_multi_hop=True, collect_signal=False)
    finally:
        deactivate_stage_ledger(token)
    for key in ("rag.rewrite", "rag.search", "rag.score", "rag.hop_decider"):
        assert key in ledger, f"expected {key} on the stage ledger, got {sorted(ledger)}"
        assert ledger[key] >= 0.0


# =============================================================================
# 6+7. Trace context, span payload, INFO line, MLflow
# =============================================================================


def _ctx(**kw) -> ChatbotTraceContext:
    return ChatbotTraceContext(trace_id="t", span_id="s", query="q", **kw)


async def test_trace_context_rag_fields_accumulate():
    ctx = _ctx()
    assert ctx.rag_stage_ms == {}
    assert ctx.rag_meta == {}
    ctx.record_rag_stage_time("rewrite", 10.0)
    ctx.record_rag_stage_time("rewrite", 5.0)
    ctx.record_rag_stage_time("score", 2.0)
    assert ctx.rag_stage_ms == {"rewrite": 15.0, "score": 2.0}


async def test_span_payload_and_log_line_carry_rag_attribution(caplog):
    ctx = _ctx()
    ctx.record_rag_stage_time("rewrite", 4100.2)
    ctx.rag_meta.update({"hops": 2, "score_calls": 5, "evidence_kept": 0})
    payload = cg._build_latency_span_payload("req-1", ctx, 1000.0, False)
    assert payload["rag_stage_ms"] == {"rewrite": 4100.2}
    assert payload["rag_meta"] == {"hops": 2, "score_calls": 5, "evidence_kept": 0}

    with caplog.at_level(logging.INFO, logger="src.api.routes.chatbot_graph"):
        cg._log_request_span(payload)
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "rag_stage_ms=" in line
    assert "rag_meta=" in line


async def test_mlflow_metrics_carry_rag_attribution():
    ctx = _ctx()
    ctx.record_rag_stage_time("rewrite", 4100.0)
    ctx.record_rag_stage_time("hop_decider", 2300.0)
    ctx.rag_meta.update({"hops": 2, "score_calls": 5, "evidence_kept": 1})
    metrics = cg._build_chat_mlflow_metrics(None, 1000.0, False, trace_ctx=ctx)
    assert metrics["rag_rewrite_ms"] == 4100.0
    assert metrics["rag_hop_decider_ms"] == 2300.0
    assert metrics["rag_hops"] == 2
    assert metrics["rag_score_calls"] == 5
    assert metrics["rag_evidence_kept"] == 1


async def test_retrieve_rag_node_transfers_ledger_and_meta(monkeypatch):
    """The node activates a ledger around the chain and moves rag.* stages +
    meta onto the trace context (mirrors the orchestrator node's #1475 flow)."""
    from src.utils.stage_timing import record_stage_wall_time

    async def fake_cognitive_rag_retrieve(**kw):
        record_stage_wall_time("rag.rewrite", 7.5)
        return cd.CognitiveRAGResult(
            rewritten_query="rq",
            search_keywords=["k"],
            graph_entities=["e"],
            evidence=[
                {
                    "source_id": "row-0",
                    "content": "c",
                    "score": 0.5,
                    "relevance_score": 0.9,
                    "key_insight": "i",
                    "source": "episodic",
                }
            ],
            hop_count=2,
            avg_relevance_score=0.9,
            retrieval_method="cognitive",
            score_calls=7,
        )

    monkeypatch.setattr(cg, "cognitive_rag_retrieve", fake_cognitive_rag_retrieve)
    monkeypatch.setattr(cg, "CHATBOT_COGNITIVE_RAG_ENABLED", True)

    ctx = _ctx()
    tok = cg._active_trace_context.set(ctx)
    try:
        out = await cg.retrieve_rag_node({"query": "q", "messages": []})
    finally:
        cg._active_trace_context.reset(tok)

    assert ctx.rag_stage_ms.get("rewrite") == pytest.approx(7.5)
    assert ctx.rag_meta["hops"] == 2
    assert ctx.rag_meta["score_calls"] == 7
    assert ctx.rag_meta["evidence_kept"] == 1
    assert out["rag_context"], "node output must still carry the evidence"
