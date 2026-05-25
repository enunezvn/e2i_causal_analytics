"""Deterministic corpus seeder for the HybridRetriever latency substrate (#414).

Usage: BENCH_PG_DSN=postgresql://... python -m tests.benchmarks.substrate.seed
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import psycopg2

from tests.benchmarks._loader import load_queries
from tests.benchmarks.substrate.embedder import embed_text, to_pgvector_literal

_HERE = Path(__file__).resolve().parent
_QUERY_FILE = _HERE.parent / "data" / "retrieval_queries.jsonl"

FILLER_VOCAB = (
    "adoption persistence titration formulary access copay specialty pharmacy "
    "infusion oncology hematology dermatology nephrology biologic adherence "
    "claims cohort uptake share growth decline region quarter segment"
).split()
RELEVANT_PER_QUERY = 5
FILLER_EPISODIC = 1500
FILLER_PROCEDURAL = 500
FILLER_FULLTEXT = 300  # per full-text table


def _vec(text: str) -> str:
    return to_pgvector_literal(embed_text(text))


def _filler_text(rng: random.Random) -> str:
    return " ".join(rng.sample(FILLER_VOCAB, k=min(8, len(FILLER_VOCAB))))


def seed(dsn: str) -> None:
    rng = random.Random(0)  # determinism
    queries = load_queries(_QUERY_FILE)
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE episodic_memories, procedural_memories, causal_paths, "
                "agent_activities, triggers;"
            )

            # Relevant rows: echo each query's text so both streams return hits.
            for qi, q in enumerate(queries):
                qtext = q.query_text
                for j in range(RELEVANT_PER_QUERY):
                    content = f"{qtext} confidence score high relevant document {j}"
                    cur.execute(
                        "INSERT INTO episodic_memories (memory_id, description, embedding, "
                        "event_type, agent_name, occurred_at, brand, region, importance_score) "
                        "VALUES (%s,%s,%s::vector,'analysis','gap_analyzer',now(),'bench','west',0.9)",
                        (f"em-rel-{qi}-{j}", content, _vec(content)),
                    )
                    cur.execute(
                        "INSERT INTO procedural_memories (procedure_id, procedure_name, "
                        "trigger_pattern, trigger_embedding, is_active, success_count, "
                        "procedure_type, success_rate, usage_count) "
                        "VALUES (%s,%s,%s,%s::vector,true,5,'analysis',0.8,10)",
                        (f"pm-rel-{qi}-{j}", f"proc {qtext}", content, _vec(content)),
                    )
                # Full-text relevant rows (one per table is enough to guarantee a hit)
                cur.execute(
                    "INSERT INTO causal_paths (path_id, start_node, end_node, method_used, "
                    "causal_chain, causal_effect_size, confidence_level, created_at) "
                    "VALUES (%s,%s,%s,'dowhy','{}'::jsonb,0.3,0.9,now())",
                    (f"cp-rel-{qi}", qtext, "outcome"),
                )
                cur.execute(
                    "INSERT INTO agent_activities (activity_id, agent_name, activity_type, "
                    "analysis_results, agent_tier, status, created_at, workstream) "
                    "VALUES (%s,%s,%s,'{}'::jsonb,'tier2','complete',now(),'bench')",
                    (f"aa-rel-{qi}", qtext, "analysis"),
                )
                cur.execute(
                    "INSERT INTO triggers (trigger_id, trigger_reason, trigger_type, "
                    "recommended_action, priority, confidence_score, created_at, invalidated_at) "
                    "VALUES (%s,%s,'opportunity',%s,'high',0.9,now(),NULL)",
                    (f"trg-rel-{qi}", qtext, "engage"),
                )

            # Filler rows: bulk so the indexes traverse a realistic corpus.
            for i in range(FILLER_EPISODIC):
                txt = _filler_text(rng)
                cur.execute(
                    "INSERT INTO episodic_memories (memory_id, description, embedding, "
                    "event_type, agent_name, occurred_at, brand, region, importance_score) "
                    "VALUES (%s,%s,%s::vector,'analysis','drift_monitor',now(),'bench','east',0.5)",
                    (f"em-fill-{i}", txt, _vec(txt)),
                )
            for i in range(FILLER_PROCEDURAL):
                txt = _filler_text(rng)
                cur.execute(
                    "INSERT INTO procedural_memories (procedure_id, procedure_name, "
                    "trigger_pattern, trigger_embedding, is_active, success_count, "
                    "procedure_type, success_rate, usage_count) "
                    "VALUES (%s,%s,%s,%s::vector,true,3,'analysis',0.6,5)",
                    (f"pm-fill-{i}", f"proc {txt}", txt, _vec(txt)),
                )
            for i in range(FILLER_FULLTEXT):
                txt = _filler_text(rng)
                cur.execute(
                    "INSERT INTO causal_paths (path_id, start_node, end_node, method_used, "
                    "causal_chain, causal_effect_size, confidence_level, created_at) "
                    "VALUES (%s,%s,%s,'dowhy','{}'::jsonb,0.1,0.5,now())",
                    (f"cp-fill-{i}", txt, "outcome"),
                )
                cur.execute(
                    "INSERT INTO agent_activities (activity_id, agent_name, activity_type, "
                    "analysis_results, agent_tier, status, created_at, workstream) "
                    "VALUES (%s,%s,%s,'{}'::jsonb,'tier3','complete',now(),'bench')",
                    (f"aa-fill-{i}", txt, "monitoring"),
                )
                cur.execute(
                    "INSERT INTO triggers (trigger_id, trigger_reason, trigger_type, "
                    "recommended_action, priority, confidence_score, created_at, invalidated_at) "
                    "VALUES (%s,%s,'alert',%s,'low',0.5,now(),NULL)",
                    (f"trg-fill-{i}", txt, "monitor"),
                )
    finally:
        conn.close()


if __name__ == "__main__":
    dsn = os.environ["BENCH_PG_DSN"]
    seed(dsn)
    print("substrate seeded")
