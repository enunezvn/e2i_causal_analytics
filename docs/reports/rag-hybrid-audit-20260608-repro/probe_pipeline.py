"""
FAITHFUL diagnostic probe for the cognitive RAG hybrid-search pipeline.

Runs the REAL production pipeline (chatbot_dspy.cognitive_rag_retrieve) against
LIVE backends (Supabase pgvector :5433/:6543, FalkorDB :6381) and REAL Claude
(DSPy LM anthropic/claude-sonnet-4) — NO MOCKS.

Instruments the 4 steps the audit targets:
  1. DSPy query rewrite        -> rewrite_query_dspy
  2. parallel hybrid (3 bk)    -> retriever.DenseRetriever/BM25Retriever/GraphRetriever
  3. Reciprocal Rank Fusion    -> HybridRetriever._reciprocal_rank_fusion
  4. per-result evidence score -> score_evidence_dspy

Reports, per query: rewrite output+method, which backends actually fired + their
result counts + latency, RRF fan-in, evidence kept, and total latency. Also runs
a control basic hybrid_search WITH a kpi_name to prove the graph backend CAN fire.

Run:  .venv/bin/dotenv run -- .venv/bin/python docs/reports/rag-hybrid-audit-20260608-repro/probe_pipeline.py
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List

# Quiet the libraries; we print our own structured trace.
logging.basicConfig(level=logging.ERROR)
for noisy in ("httpx", "openai", "dspy", "LiteLLM", "litellm", "urllib3", "falkordb"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

import src.rag.retriever as retr
import src.api.routes.chatbot_dspy as cb

CALL_LOG: List[Dict[str, Any]] = []


def _wrap_async(cls, name, label):
    orig = getattr(cls, name)

    async def wrapper(self, *a, **k):
        t0 = time.time()
        try:
            res = await orig(self, *a, **k)
            n = len(res) if hasattr(res, "__len__") else "?"
            CALL_LOG.append(
                {"backend": label, "ok": True, "n": n, "ms": round((time.time() - t0) * 1000, 1)}
            )
            return res
        except Exception as e:  # noqa: BLE001
            CALL_LOG.append(
                {"backend": label, "ok": False, "err": repr(e)[:160], "ms": round((time.time() - t0) * 1000, 1)}
            )
            raise

    setattr(cls, name, wrapper)


def _wrap_sync(cls, name, label):
    orig = getattr(cls, name)

    def wrapper(self, *a, **k):
        t0 = time.time()
        try:
            res = orig(self, *a, **k)
            n = len(res) if hasattr(res, "__len__") else "?"
            CALL_LOG.append(
                {"backend": label, "ok": True, "n": n, "ms": round((time.time() - t0) * 1000, 1)}
            )
            return res
        except Exception as e:  # noqa: BLE001
            CALL_LOG.append(
                {"backend": label, "ok": False, "err": repr(e)[:160], "ms": round((time.time() - t0) * 1000, 1)}
            )
            raise

    setattr(cls, name, wrapper)


# Instrument the three backends at the class level (hybrid_search builds fresh
# instances each call, so class-level patches capture every invocation).
_wrap_async(retr.DenseRetriever, "search", "DENSE(vector)")
_wrap_async(retr.BM25Retriever, "search", "SPARSE(fulltext)")
_wrap_sync(retr.GraphRetriever, "traverse", "GRAPH.traverse(entities)")
_wrap_sync(retr.GraphRetriever, "traverse_kpi", "GRAPH.traverse_kpi(kpi)")

# Instrument evidence scoring (count + timing of the per-result LLM calls).
SCORE_LOG: List[Dict[str, Any]] = []
_orig_score = cb.score_evidence_dspy


async def _score_probe(investigation_goal, evidence_item, source_memory="episodic"):
    t0 = time.time()
    out = await _orig_score(investigation_goal, evidence_item, source_memory)
    SCORE_LOG.append({"ms": round((time.time() - t0) * 1000, 1), "score": out[0], "src_arg": source_memory})
    return out


cb.score_evidence_dspy = _score_probe


def _reset():
    CALL_LOG.clear()
    SCORE_LOG.clear()


async def probe_cognitive(query: str, brand: str = "Kisqali"):
    _reset()
    t0 = time.time()
    # capture rewrite separately too
    rq, kw, ge, method = await cb.rewrite_query_dspy(query=query, brand_context=brand)
    rewrite_ms = round((time.time() - t0) * 1000, 1)

    t1 = time.time()
    result = await cb.cognitive_rag_retrieve(
        query=query,
        conversation_context="",
        brand_context=brand,
        intent="",
        k=5,
        enable_multi_hop=False,
        collect_signal=False,
    )
    total_ms = round((time.time() - t1) * 1000, 1)

    print(f"\n{'='*78}\nQUERY: {query!r}  (brand={brand})\n{'='*78}")
    print("STEP 1 — DSPy rewrite:")
    print(f"   method      : {method}")
    print(f"   rewritten   : {rq[:140]!r}")
    print(f"   keywords    : {kw}")
    print(f"   graph_ents  : {ge}   <-- extracted; check if passed to graph backend below")
    print(f"   rewrite_ms  : {rewrite_ms}")
    print("STEP 2+3 — backends fired during cognitive_rag_retrieve (parallel hybrid + RRF):")
    if not CALL_LOG:
        print("   (NONE)")
    for c in CALL_LOG:
        print(f"   {c}")
    fired = {c['backend'].split('(')[0].split('.')[0] for c in CALL_LOG if c.get('ok')}
    print(f"   distinct backends that FIRED: {sorted(fired)}")
    print("STEP 4 — per-result evidence scoring:")
    print(f"   scoring LLM calls: {len(SCORE_LOG)}  (sequential)")
    if SCORE_LOG:
        print(f"   scoring latency ms: {[s['ms'] for s in SCORE_LOG]}  (sum={round(sum(s['ms'] for s in SCORE_LOG),1)})")
        print(f"   src_memory arg always: {sorted({s['src_arg'] for s in SCORE_LOG})}")
    print("RESULT:")
    print(f"   retrieval_method : {result.retrieval_method}")
    print(f"   evidence kept    : {len(result.evidence)}")
    print(f"   hop_count        : {result.hop_count}")
    print(f"   avg_relevance    : {round(result.avg_relevance_score,3)}")
    for i, e in enumerate(result.evidence[:5]):
        print(f"     [{i}] src={e.get('source')!s:18.18} rrf={e.get('score')!s:8.8} rel={e.get('relevance_score')} :: {e.get('content','')[:70]!r}")
    print(f"   TOTAL cognitive_rag_retrieve latency: {total_ms} ms")
    return result


async def probe_basic_with_kpi(query: str, kpi_name: str, brand: str = "Kisqali"):
    """Control: basic hybrid_search WITH kpi_name -> proves graph backend reachable."""
    _reset()
    t0 = time.time()
    results = await retr.hybrid_search(
        query=query, k=5, kpi_name=kpi_name, filters={"brand": brand} if brand else None
    )
    ms = round((time.time() - t0) * 1000, 1)
    print(f"\n{'-'*78}\nCONTROL basic hybrid_search(kpi_name={kpi_name!r}) for {query!r}\n{'-'*78}")
    for c in CALL_LOG:
        print(f"   {c}")
    fired = {c['backend'].split('(')[0].split('.')[0] for c in CALL_LOG if c.get('ok')}
    print(f"   distinct backends that FIRED: {sorted(fired)}")
    print(f"   results: {len(results)}  latency: {ms} ms")
    return results


async def main():
    print("FAITHFUL cognitive-RAG pipeline probe — live Supabase + FalkorDB + real Claude\n")

    queries = [
        ("What is the TRx trend for Kisqali in the Northeast this quarter?", "Kisqali"),  # KPI
        ("Why did Kisqali adoption increase in the Northeast last quarter?", "Kisqali"),   # explanation/causal
        ("Summarize recent agent activity and context for Fabhalta.", "Fabhalta"),          # context
    ]
    for q, b in queries:
        try:
            await probe_cognitive(q, b)
        except Exception as e:  # noqa: BLE001
            print(f"   !! cognitive probe raised: {e!r}")

    # Control — does graph fire when kpi_name is supplied (basic path)?
    try:
        await probe_basic_with_kpi(
            "What is the TRx trend for Kisqali in the Northeast?", kpi_name="trx", brand="Kisqali"
        )
    except Exception as e:  # noqa: BLE001
        print(f"   !! basic probe raised: {e!r}")

    print("\nDONE.")


if __name__ == "__main__":
    asyncio.run(main())
