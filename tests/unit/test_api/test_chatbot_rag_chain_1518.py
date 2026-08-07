"""Red-first tests for #1518: remaining RAG-chain latency levers after #1490.

Local measurements backing these tests (2026-08-07, real LLM openai/gpt-5.6-terra
+ real DB, novel text per run, dspy cache disabled):

* Lever A — rewrite Predict vs CoT (n=8 novel pharma queries, interleaved
  order): CoT 4311-7537ms median 5133; Predict 2631-4650ms median 3425;
  Predict faster 8/8, per-pair delta 568-3521ms median 1602ms. Quality guard:
  0/8 queries where Predict drops a domain term CoT kept (brands, KPIs,
  regions, specialties checked across rewritten/keywords/entities outputs).
  => ChatbotQueryRewriter goes Predict-only by default; CoT restorable via
  CHATBOT_RAG_REWRITE_COT=true.
* Lever B1 — decider Predict vs CoT (n=8 evidence boards): CoT 2202-4738ms
  median 3075 vs Predict 2404-4130ms median 3149 — parity (the signature
  already demands a `reasoning` output field). REJECTED, no code change.
* Lever B2 — empty-board decider skip: on an EMPTY evidence board the CoT
  decider said continue->episodic in 8/8 samples (conf >= 0.92), so the
  1.9-3.7s (median ~3.3s) LLM call carries information only in its refined
  retrieval_query. Retrieval comparison over 6 goals (chain dedupe keys):
  decider-refined query 26 new-rows-vs-hop1 vs original-user-query 24
  (never worse by >1 per goal; keywords-join 21). => skip the LLM decider
  when the board is empty and retry with the original query (fallback
  keywords-join), preserving dry-limit/max-hop economics.
  CHATBOT_RAG_SKIP_EMPTY_DECIDER=false restores the LLM decider.
* Lever C — search overlap: hybrid_search legs are ALREADY concurrent
  (dense/sparse/graph rel starts 0-10ms; dense dominates: median 1013ms of
  1019ms total; graph 2-196ms, 0 rows on 6/6). REJECTED, no code change.

These tests pin:

1.  ChatbotQueryRewriter builds dspy.Predict by default (no CoT reasoning
    field => fewer output tokens), ChainOfThought only when the knob asks.
2.  The prod singleton (_get_dspy_query_rewriter) serves the real module
    with the Predict inner — seam-pin, no stubs.
3.  An empty evidence board skips the decider LLM call entirely and retries
    with the original user query; dry-limit economics unchanged.
4.  Heuristic candidates exhaust (original -> keywords-join -> stop): the
    loop can never spin on repeated queries.
5.  A non-empty (but insufficient) board still consults the LLM decider.
6.  CHATBOT_RAG_SKIP_EMPTY_DECIDER=false restores the legacy decider path.
7.  The skip path still records the rag.hop_decider stage (near-zero ms is
    the live-verification signal on SSE/INFO/MLflow surfaces).
"""

import dspy
import pytest

import src.api.routes.chatbot_dspy as cd
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
# 1+2. Lever A — rewriter is Predict-only by default (CoT via knob)
# =============================================================================


async def test_rewriter_uses_predict_only_by_default():
    rewriter = cd.ChatbotQueryRewriter()
    assert type(rewriter.rewrite) is dspy.Predict, (
        "ChatbotQueryRewriter must build dspy.Predict by default (#1518: CoT "
        "measured 8/8 slower, median +1602ms, with no domain-term quality "
        f"gain); got {type(rewriter.rewrite).__name__}"
    )


async def test_rewriter_predict_signature_has_no_reasoning_field():
    """The latency win comes from not generating reasoning tokens — pin that
    the inner module's signature is the raw QueryRewriteSignature."""
    rewriter = cd.ChatbotQueryRewriter()
    out_fields = set(rewriter.rewrite.signature.output_fields.keys())
    assert out_fields == {"rewritten_query", "search_keywords", "graph_entities"}, (
        f"unexpected output fields (reasoning injected?): {sorted(out_fields)}"
    )


async def test_rewrite_cot_knob_restores_chain_of_thought(monkeypatch):
    monkeypatch.setattr(cd, "_RAG_REWRITE_USE_COT", True)
    rewriter = cd.ChatbotQueryRewriter()
    assert isinstance(rewriter.rewrite, dspy.ChainOfThought), (
        "CHATBOT_RAG_REWRITE_COT=true must restore the ChainOfThought rewriter"
    )


async def test_prod_singleton_serves_real_predict_rewriter(monkeypatch):
    """Seam-pin: the prod accessor returns the REAL module class with the
    Predict inner — not a stub, not CoT."""
    monkeypatch.setattr(cd, "_dspy_query_rewriter", None)
    rewriter = cd._get_dspy_query_rewriter()
    assert isinstance(rewriter, cd.ChatbotQueryRewriter)
    assert type(rewriter.rewrite) is dspy.Predict


# =============================================================================
# 3+4. Lever B2 — empty-board decider skip
# =============================================================================


def _dry_loop_fixtures(monkeypatch, score_fn=None):
    """Wire a multi-hop run over 5-row hops; default scoring keeps nothing."""
    decider_calls = {"n": 0}
    hop_queries: list[str] = []

    async def fake_rewrite(**kw):
        return ("rewritten q", ["kisqali"], ["Kisqali"], "dspy")

    async def default_score(investigation_goal, evidence_item, source_memory="episodic"):
        return (0.1, "meh", False)  # dry: nothing clears the 0.3 keep-gate

    async def fake_hybrid_search(query, k, entities=None, kpi_name=None, filters=None, **kwargs):
        hop_queries.append(query)
        return _rows(5, prefix=f"hop{len(hop_queries)}")

    class _Decider:
        def __call__(self, **kw):  # pragma: no cover - async contract pinned in 1484
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
    monkeypatch.setattr(cd, "score_evidence_dspy", score_fn or default_score)
    monkeypatch.setattr(cd, "_get_dspy_hop_decider", lambda: _Decider())
    import src.rag.retriever as rt

    monkeypatch.setattr(rt, "hybrid_search", fake_hybrid_search)
    return decider_calls, hop_queries


async def test_empty_board_skips_decider_llm_and_retries_with_original_query(monkeypatch):
    """Hop-1 kept nothing => the board the decider would see is []. Measured
    8/8 such calls say continue->episodic, so skip the LLM and retry with the
    original user query directly."""
    decider_calls, hop_queries = _dry_loop_fixtures(monkeypatch)
    result = await cd.cognitive_rag_retrieve(
        query="original q", enable_multi_hop=True, collect_signal=False
    )
    assert decider_calls["n"] == 0, (
        f"empty evidence board must not pay a decider LLM call; got {decider_calls['n']}"
    )
    assert result.hop_count == 2, "the dry-hop retry economics must be preserved"
    assert hop_queries == ["rewritten q", "original q"], (
        f"hop-2 must retry with the ORIGINAL user query; saw {hop_queries}"
    )


async def test_empty_board_skip_exhausts_heuristic_candidates_then_stops(monkeypatch):
    """With the dry-limit disabled (legacy run-to-max), the heuristic loop
    must exhaust its candidates (original -> keywords-join) and stop — it can
    never spin on repeated queries."""
    decider_calls, hop_queries = _dry_loop_fixtures(monkeypatch)
    monkeypatch.setattr(cd, "_DRY_HOP_LIMIT", 0)
    result = await cd.cognitive_rag_retrieve(
        query="original q", enable_multi_hop=True, collect_signal=False
    )
    assert decider_calls["n"] == 0
    assert hop_queries == ["rewritten q", "original q", "kisqali"], (
        f"expected original-query then keywords-join retries; saw {hop_queries}"
    )
    assert result.hop_count == 3


async def test_non_empty_insufficient_board_still_calls_decider(monkeypatch):
    """CONTROL (green pre-change): one kept row (0.4) is a non-empty but
    insufficient board — the LLM decider still owns that decision."""

    async def one_keeper_score(investigation_goal, evidence_item, source_memory="episodic"):
        # keep exactly one row per hop: hopN-0 scores 0.4, the rest 0.1
        return (
            (0.4, "ok", False)
            if evidence_item.endswith("-0 about kisqali trx northeast")
            else (0.1, "meh", False)
        )

    decider_calls, hop_queries = _dry_loop_fixtures(monkeypatch, score_fn=one_keeper_score)
    await cd.cognitive_rag_retrieve(query="original q", enable_multi_hop=True, collect_signal=False)
    assert decider_calls["n"] >= 1, (
        "a non-empty (insufficient) board must still consult the LLM decider"
    )
    assert "refined q 1" in hop_queries


async def test_skip_knob_disabled_restores_llm_decider_on_empty_board(monkeypatch):
    decider_calls, hop_queries = _dry_loop_fixtures(monkeypatch)
    monkeypatch.setattr(cd, "_SKIP_EMPTY_DECIDER", False)
    result = await cd.cognitive_rag_retrieve(
        query="original q", enable_multi_hop=True, collect_signal=False
    )
    assert decider_calls["n"] == 1, (
        "CHATBOT_RAG_SKIP_EMPTY_DECIDER=false must restore the legacy LLM decider"
    )
    assert hop_queries == ["rewritten q", "refined q 1"]
    assert result.hop_count == 2


async def test_empty_board_skip_records_hop_decider_stage(monkeypatch):
    """The skip path must still record rag.hop_decider on the stage ledger —
    a near-zero value there is the live-verification signal that the skip
    fired (SSE dispatch_info / INFO line / MLflow all read this ledger)."""
    decider_calls, _ = _dry_loop_fixtures(monkeypatch)
    ledger, token = activate_stage_ledger()
    try:
        await cd.cognitive_rag_retrieve(
            query="original q", enable_multi_hop=True, collect_signal=False
        )
    finally:
        deactivate_stage_ledger(token)
    assert decider_calls["n"] == 0
    assert "rag.hop_decider" in ledger, (
        f"skip path must keep the rag.hop_decider stage observable; got {sorted(ledger)}"
    )
    assert ledger["rag.hop_decider"] < 100.0, "skip path should be near-zero ms"


# =============================================================================
# 5. The heuristic helper itself
# =============================================================================


async def test_empty_board_hop_query_candidates_and_exhaustion():
    f = cd._empty_board_hop_query
    # primary: the original user query
    assert f("orig q", ["kisqali", "trx"], {"rewritten q"}) == "orig q"
    # fallback: keywords-join when the original was already used as a hop query
    assert f("orig q", ["kisqali", "trx"], {"rewritten q", "orig q"}) == "kisqali, trx"
    # exhausted -> None (caller stops the loop)
    assert f("orig q", ["kisqali", "trx"], {"rewritten q", "orig q", "kisqali, trx"}) is None
    # blank keywords never yield an empty-string query
    assert f("orig q", [], {"orig q"}) is None
    assert f("  ", [" "], {"x"}) is None
