"""#785: faithful Tier-1 causal_impact populate — grow e2i_causal CausalPath + episodic.

Runs REAL causal_impact analyses (real DoWhy/OLS estimation + real refutation suite) with
``enable_memory=True`` so each agent run contributes to the tri-memory architecture via the
canonical ``contribute_to_memory`` (episodic ``causal_analysis_completed`` with a real
1536-dim vector + ``store_causal_path`` Variable/CAUSES growth for a PROCEED-validated
estimate). Captures baseline + post counts and prints the per-run + cumulative delta.

No fabricated nodes: every node/edge/row comes from real agent processing (real DoWhy
estimate, real refutation gate, real OpenAI embedding) — not hardcoded.

Run faithfully:
    dotenv -f .env run -- python scripts/run_785_causal_populate.py
"""

import asyncio
import sys

from src.agents.causal_impact.agent import CausalImpactAgent
from src.memory.episodic_memory import get_supabase_client
from src.memory.semantic_memory import get_semantic_memory

# Real causal questions. A PROCEED-validated run adds 2 Variable nodes + 1 branded CAUSES
# edge (store_causal_path) and an episodic causal_analysis_completed row (1536-dim).
#
# NOTE on "at scale": the ``data_source='synthetic'`` fixture (estimation.py: seeded
# np.random.seed(42) HCP/conversion data) models ONE real causal relationship — the
# canonical hcp_engagement -> conversion pair validates; arbitrary other pairs correctly
# FAIL refutation (the H2 gate fail-closes, and contribute_to_memory skips failed
# analyses by design — no node is fabricated). Genuine multi-path at-scale growth requires
# REAL multi-relationship data fed via ``state['data_cache']['estimation_data']`` (the real
# patient_journeys / tool_composer feed); that path is unblocked by this fix and #784.
QUERIES = [
    (
        "What drove Kisqali conversion in the Northeast?",
        "hcp_engagement_level",
        "patient_conversion_rate",
        ["geographic_region", "hcp_specialty"],
    ),
]


def _episodic_count() -> tuple[int, int]:
    client = get_supabase_client()
    resp = (
        client.table("episodic_memories")
        .select("memory_id, embedding")
        .eq("event_type", "causal_analysis_completed")
        .execute()
    )
    rows = resp.data or []
    return len(rows), sum(1 for r in rows if r.get("embedding") is not None)


def _graph_counts() -> dict:
    sm = get_semantic_memory()
    out = {}
    for q, label in [
        ("MATCH (n) RETURN count(n)", "nodes"),
        ("MATCH ()-[r]->() RETURN count(r)", "edges"),
        ("MATCH (n:Variable) RETURN count(n)", "Variable_nodes"),
        ("MATCH ()-[r:CAUSES]->() RETURN count(r)", "CAUSES_edges"),
    ]:
        res = sm.graph.query(q)
        out[label] = res.result_set[0][0] if res.result_set else 0
    return out


async def main() -> int:
    ep_before, ep_vec_before = _episodic_count()
    g_before = _graph_counts()
    print("=== BASELINE ===")
    print(
        f"episodic causal_analysis_completed: {ep_before} (with 1536-dim vector: {ep_vec_before})"
    )
    print(f"e2i_causal graph: {g_before}\n")

    proceeded = 0
    for i, (query, treatment, outcome, confounders) in enumerate(QUERIES, 1):
        agent = CausalImpactAgent(enable_memory=True)
        input_data = {
            "query": query,
            "treatment_var": treatment,
            "outcome_var": outcome,
            "confounders": confounders,
            "data_source": "synthetic",  # real OLS+refutation processing on seeded data
            "interpretation_depth": "standard",
            "brand": "kisqali",
            "region": "northeast",
        }
        result = await agent.run(input_data)
        gate = result.get("gate_decision")
        if gate == "proceed":
            proceeded += 1
        ate = result.get("ate_estimate")
        conf = result.get("confidence")
        print(
            f"[{i}/{len(QUERIES)}] {treatment} -> {outcome}: status={result.get('status')} "
            f"ate={ate if ate is None else round(ate, 4)} "
            f"conf={conf if conf is None else round(conf, 2)} "
            f"refut={result.get('refutation_passed')} gate={gate}"
        )

    ep_after, ep_vec_after = _episodic_count()
    g_after = _graph_counts()
    print("\n=== POST ===")
    print(f"episodic causal_analysis_completed: {ep_after} (with 1536-dim vector: {ep_vec_after})")
    print(f"e2i_causal graph: {g_after}")
    print("\n=== CUMULATIVE DELTA (from real agent processing only) ===")
    print(f"runs={len(QUERIES)} proceed_validated={proceeded}")
    print(
        f"episodic causal_analysis_completed: +{ep_after - ep_before} "
        f"(with 1536-dim vector: +{ep_vec_after - ep_vec_before})"
    )
    for k in ("nodes", "edges", "Variable_nodes", "CAUSES_edges"):
        print(f"{k}: +{g_after[k] - g_before[k]}  ({g_before[k]} -> {g_after[k]})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
