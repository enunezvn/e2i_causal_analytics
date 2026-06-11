"""
Second faithful probe — answers three follow-ups from probe_pipeline.py:

  A) Is SPARSE (fulltext) genuinely broken, or just no matching rows?
  B) Is per-result evidence scoring miscalibrated, or is the retrieved corpus
     simply irrelevant to KPI/causal queries?
  C) What actually lives in the memory tables the pipeline searches?

Live Supabase + FalkorDB + real Claude. NO MOCKS.
Run: PYTHONPATH=. .venv/bin/dotenv run -- .venv/bin/python docs/reports/rag-hybrid-audit-20260608-repro/probe_corpus_sparse_scoring.py
"""
import asyncio
import logging

logging.basicConfig(level=logging.ERROR)

from src.rag.memory_connector import get_memory_connector
from src.memory.services.factories import get_supabase_client
import src.api.routes.chatbot_dspy as cb


async def main():
    conn = get_memory_connector()

    print("="*78)
    print("C) Row counts in the corpora the pipeline searches")
    print("="*78)
    sb = get_supabase_client()
    for table in ["episodic_memories", "procedural_memories", "agent_activities", "causal_paths", "triggers"]:
        try:
            r = sb.table(table).select("*", count="exact").limit(1).execute()
            cnt = getattr(r, "count", None)
            print(f"   {table:22} count={cnt}")
        except Exception as e:  # noqa: BLE001
            print(f"   {table:22} ERROR {repr(e)[:90]}")

    print("\n" + "="*78)
    print("B) Raw DENSE results for a KPI query + their DSPy relevance scores")
    print("="*78)
    q = "Analyze TRx prescription trends and quarterly performance for Kisqali in Northeast"
    dense = await conn.vector_search_by_text(query_text=q, k=8, min_similarity=0.5)
    print(f"   vector_search_by_text returned {len(dense)} rows (min_similarity=0.5)")
    for i, r in enumerate(dense):
        score, insight, _ = await cb.score_evidence_dspy(
            investigation_goal=f"Answer: {q}", evidence_item=r.content[:500], source_memory="episodic"
        )
        print(f"   [{i}] vec_score={round(r.score,3)} dspy_rel={score} src={r.source} :: {r.content[:80]!r}")

    # Try a lower similarity floor to see if relevant rows exist but are filtered.
    dense_lo = await conn.vector_search_by_text(query_text=q, k=8, min_similarity=0.0)
    print(f"\n   same query with min_similarity=0.0 -> {len(dense_lo)} rows")
    for i, r in enumerate(dense_lo[:8]):
        print(f"      [{i}] vec_score={round(r.score,3)} src={r.source} :: {r.content[:70]!r}")

    print("\n" + "="*78)
    print("A) SPARSE / fulltext search — does it ever return anything?")
    print("="*78)
    for qq in ["Kisqali", "TRx", "Kisqali Northeast prescription", "adoption", "agent"]:
        try:
            res = await conn.fulltext_search(query_text=qq, k=5)
            print(f"   fulltext_search({qq!r}) -> {len(res)} rows; sample={[x.content[:40] for x in res[:2]]}")
        except Exception as e:  # noqa: BLE001
            print(f"   fulltext_search({qq!r}) ERROR {repr(e)[:120]}")

    # Inspect the underlying RPC the fulltext path calls.
    print("\n   --- direct RPC probe (hybrid_fulltext_search) ---")
    try:
        rpc = sb.rpc("hybrid_fulltext_search", {"search_query": "Kisqali", "match_count": 5, "filters": {}}).execute()
        data = rpc.data if hasattr(rpc, "data") else rpc
        print(f"   RPC hybrid_fulltext_search('Kisqali') -> {len(data) if data else 0} rows")
        if data:
            print(f"   sample keys: {list(data[0].keys())}")
    except Exception as e:  # noqa: BLE001
        print(f"   RPC hybrid_fulltext_search ERROR {repr(e)[:160]}")

    print("\nDONE.")


if __name__ == "__main__":
    asyncio.run(main())
