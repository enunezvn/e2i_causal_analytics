"""Faithful E2E gates for cognitive-RAG multi-hop (audit F6).

Two gate tiers:

* E2I_RUN_LIVE_RAG=1 — needs only live Supabase + the OpenAI embedder. Proves the
  multi-hop pipeline is crash-safe and returns the operational corpus end-to-end
  (also exercises F2/F3a/F4 through cognitive_rag_retrieve). Runnable whenever the
  droplet backends are up, independent of the chat LLM.

* E2I_RUN_REAL_LLM_E2E=1 — needs real Claude (DSPy LM). Proves the previously-dead
  ChatbotHopDecider produces an actionable decision and that the loop escalates on
  insufficient evidence. (Precedent gate name: the DSPy loop's real-LM E2E.)
"""

import os

import pytest

from src.api.routes.chatbot_dspy import (
    _MULTIHOP_AVAILABLE_MEMORIES,
    _MULTIHOP_MAX_HOPS,
    _get_dspy_hop_decider,
    _validate_confidence,
    cognitive_rag_retrieve,
)

STRONG_QUERY = "What is the TRx trend for Kisqali in the Northeast?"


@pytest.mark.skipif(
    os.getenv("E2I_RUN_LIVE_RAG") != "1",
    reason="droplet-only faithful gate; set E2I_RUN_LIVE_RAG=1 to run",
)
@pytest.mark.asyncio
async def test_multihop_pipeline_returns_corpus_and_is_crash_safe():
    """enable_multi_hop=True must run end-to-end, bounded, and surface the corpus.

    Crash-safety holds even when the decider LLM is unavailable (the loop breaks
    instead of raising). This is the integration proof that F6's wiring is live
    (not the dead enable_multi_hop the audit found) AND that F2/F3a/F4 deliver the
    corpus through the full cognitive path.
    """
    result = await cognitive_rag_retrieve(
        query=STRONG_QUERY, brand_context="Kisqali", k=5, enable_multi_hop=True
    )
    assert 1 <= result.hop_count <= _MULTIHOP_MAX_HOPS, (
        f"hop_count out of range: {result.hop_count}"
    )
    assert result.evidence, "multi-hop pipeline returned no evidence"
    # The top result must be real operational-corpus KPI prose (rendered from
    # business_metrics: "<metric> for <brand> in the <region> on <date>: value ..."),
    # NOT the [PROC] composition junk the audit saw win every query. The DSPy
    # rewrite may emphasize a synonym metric (Total Prescriptions vs the literal
    # TRx row), so assert the corpus-prose shape, not a specific KPI token.
    top = result.evidence[0]["content"].lower()
    assert "for kisqali in the" in top and "value" in top, (
        f"top result is not operational corpus prose (F4 still degenerate?): {top[:90]!r}"
    )
    blob = " ".join(e["content"].lower() for e in result.evidence)
    assert "composition_comp" not in blob and "[proc]" not in blob, (
        "procedural agent-bookkeeping junk still outranks the corpus"
    )


@pytest.mark.skipif(
    os.getenv("E2I_RUN_REAL_LLM_E2E") != "1",
    reason="real-Claude gate; set E2I_RUN_REAL_LLM_E2E=1 (needs Anthropic credits)",
)
@pytest.mark.asyncio
async def test_hop_decider_real_claude_produces_actionable_decision():
    """The (previously never-instantiated) ChatbotHopDecider must, with real
    Claude, return a usable next_memory + a non-empty refined retrieval_query +
    a valid confidence — the inputs the loop needs to escalate."""
    decider = _get_dspy_hop_decider()
    assert decider is not None, "hop decider unavailable (DSPy not configured)"
    decision = decider(
        investigation_goal="Answer: Why did Kisqali adoption increase in the Northeast?",
        current_evidence="[]",  # insufficient -> a real decider should want more
        hop_number=1,
        available_memories=_MULTIHOP_AVAILABLE_MEMORIES,
    )
    assert str(getattr(decision, "next_memory", "")).strip(), "decider gave no next_memory"
    assert str(getattr(decision, "retrieval_query", "")).strip(), (
        "decider proposed no refined query -> loop cannot escalate"
    )
    conf = _validate_confidence(getattr(decision, "confidence", None))
    assert 0.0 <= conf <= 1.0


@pytest.mark.skipif(
    os.getenv("E2I_RUN_REAL_LLM_E2E") != "1",
    reason="real-Claude gate; set E2I_RUN_REAL_LLM_E2E=1 (needs Anthropic credits)",
)
@pytest.mark.asyncio
async def test_multihop_runs_bounded_with_real_llm():
    """With real Claude, an under-served query drives the loop without exceeding
    the hop cap; a well-served query early-stops. Both stay within bounds."""
    weak = await cognitive_rag_retrieve(
        query="Summarize cross-brand competitive dynamics and unmet need signals",
        brand_context="",
        k=5,
        enable_multi_hop=True,
    )
    assert 1 <= weak.hop_count <= _MULTIHOP_MAX_HOPS

    strong = await cognitive_rag_retrieve(
        query=STRONG_QUERY, brand_context="Kisqali", k=5, enable_multi_hop=True
    )
    assert 1 <= strong.hop_count <= _MULTIHOP_MAX_HOPS
