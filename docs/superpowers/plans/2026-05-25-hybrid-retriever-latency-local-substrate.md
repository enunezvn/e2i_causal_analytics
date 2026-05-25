# HybridRetriever Latency Local Substrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the `hybrid_retriever_search_p50/p95` benchmark baselines from honest-0.0 placeholders into a secret-free, reproducible, in-CI latency regression guard by running `hybrid_search` against a seeded local Postgres+pgvector substrate.

**Architecture:** A test-side `DirectSQLMemoryConnector` (psycopg2) is injected into the existing `get_memory_connector()` singleton; it calls the **real** `hybrid_vector_search` / `hybrid_fulltext_search` SQL functions against a local pgvector container seeded with a deterministic corpus. A deterministic bag-of-token-hash embedder replaces OpenAI. No production code changes; fail-closed (a broken/empty substrate errors rather than blessing 0.0).

**Tech Stack:** Python 3.12, pytest, psycopg2-binary, pgvector (`pgvector/pgvector:pg16`), PostgreSQL plpgsql functions, GitHub Actions service containers.

**Spec:** `docs/superpowers/specs/2026-05-25-hybrid-retriever-latency-local-substrate-design.md`

---

## File Structure

| Path | Responsibility | Action |
|------|----------------|--------|
| `tests/benchmarks/substrate/__init__.py` | package marker | Create |
| `tests/benchmarks/substrate/embedder.py` | deterministic bag-of-token-hash embedder + pgvector literal formatter | Create |
| `tests/benchmarks/substrate/direct_sql_connector.py` | `DirectSQLMemoryConnector` — 4 retriever methods via psycopg2, fail-closed | Create |
| `tests/benchmarks/substrate/bench_schema.sql` | minimal DDL for the 5 tables the SQL functions read | Create |
| `tests/benchmarks/substrate/seed.py` | deterministic corpus generator (query-echoing relevant rows + filler) | Create |
| `tests/benchmarks/substrate/fixture.py` | `substrate_ready()` + `inject_local_substrate` pytest fixture | Create |
| `tests/benchmarks/substrate/test_embedder.py` | unit tests for the embedder | Create |
| `tests/benchmarks/substrate/test_substrate_integration.py` | seed→query non-empty + fail-loud-on-broken-substrate | Create |
| `tests/benchmarks/test_hybrid_retriever_latency.py` | swap skip logic to `substrate_ready()`; fail on all-empty | Modify |
| `.github/workflows/benchmarks.yml` | add pgvector service container + load/seed steps to Box 2 | Modify |
| `tests/benchmarks/baselines/performance.json` | re-bless the 2 hybrid baselines (after CI runs) | Modify |
| `tests/unit/test_benchmarks_meta/test_baseline_no_placeholder.py` | move hybrid boxes placeholder→reblessed | Modify |

**Reference facts (verified, do not re-derive):**
- `hybrid_vector_search(query_embedding vector(1536), match_count int, filters jsonb)` → `(id text, content text, similarity float, metadata jsonb, source_table text)`. Reads `episodic_memories` + `procedural_memories`. **Hardcoded `similarity > 0.5` floor** (`database/memory/011_hybrid_search_functions_fixed.sql:126,148`).
- `hybrid_fulltext_search(search_query text, match_count int, filters jsonb)` → `(id text, content text, rank double precision, metadata jsonb, source_table text)`. Reads `causal_paths` + `agent_activities` + `triggers`. The authoritative definition is `database/memory/022_hybrid_search_max_staleness.sql` (overrides 011).
- The latency benchmark calls `hybrid_search(query, k, filters, max_staleness)` with **no** `entities`/`kpi_name`, so the graph stream is never invoked (`src/rag/retriever.py:305-307`). No FalkorDB needed.
- Injection seam: `src/rag/memory_connector.py:480-496` (`get_memory_connector()` singleton + `reset_memory_connector()`); the retriever resolves the connector per-call, so monkeypatching the module global suffices.
- Latency test consts: `_QUERY_FILE = data/retrieval_queries.jsonl`, `_TOP_K = 10`, `_BASELINE_FILE = baselines/performance.json` (`tests/benchmarks/test_hybrid_retriever_latency.py:74-76`).

---

## Task 1: Deterministic embedder

**Files:**
- Create: `tests/benchmarks/substrate/__init__.py`
- Create: `tests/benchmarks/substrate/embedder.py`
- Test: `tests/benchmarks/substrate/test_embedder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/substrate/test_embedder.py
import math

from tests.benchmarks.substrate.embedder import EMBED_DIM, embed_text, to_pgvector_literal


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_dimension_and_unit_norm():
    v = embed_text("Kisqali TRx growth West region")
    assert len(v) == EMBED_DIM
    assert math.isclose(_cos(v, v), 1.0, abs_tol=1e-6)


def test_identical_text_identical_vector():
    assert embed_text("fabhalta PNH discontinuation") == embed_text("fabhalta PNH discontinuation")


def test_token_overlap_gives_high_cosine():
    q = embed_text("kisqali trx growth west region q3")
    doc = embed_text("kisqali trx growth west region q3 confidence score high")
    assert _cos(q, doc) > 0.5  # must clear hybrid_vector_search's hardcoded 0.5 floor


def test_disjoint_tokens_near_zero():
    a = embed_text("alpha beta gamma delta")
    b = embed_text("xenon yttrium zirconium niobium")
    assert _cos(a, b) < 0.1


def test_empty_text_is_deterministic_unit_vector():
    v = embed_text("!!! ???")
    assert math.isclose(_cos(v, v), 1.0, abs_tol=1e-6)


def test_pgvector_literal_format():
    lit = to_pgvector_literal([0.0, 1.0, -0.5])
    assert lit == "[0.000000,1.000000,-0.500000]"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/benchmarks/substrate/test_embedder.py -v -o addopts=""`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.benchmarks.substrate.embedder'`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/benchmarks/substrate/__init__.py
```

```python
# tests/benchmarks/substrate/embedder.py
"""Deterministic, network-free embedder for the HybridRetriever latency substrate.

Bag-of-token-hash: identical text -> identical vector; texts that share tokens
have positive cosine similarity. This matters because hybrid_vector_search
hardcodes a `similarity > 0.5` floor (011_hybrid_search_functions_fixed.sql),
so RANDOM embeddings would return zero rows. A corpus doc that echoes a query's
tokens clears the floor, guaranteeing the vector stream returns non-empty.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

EMBED_DIM = 1536
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def embed_text(text: str, dim: int = EMBED_DIM) -> List[float]:
    vec = [0.0] * dim
    for tok in _TOKEN_RE.findall(text.lower()):
        h = int.from_bytes(hashlib.sha1(tok.encode("utf-8")).digest()[:8], "big")
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        vec[0] = 1.0  # punctuation-only / empty text -> deterministic unit vector
        return vec
    return [v / norm for v in vec]


def to_pgvector_literal(vec: List[float]) -> str:
    """Format an embedding as a pgvector text literal, e.g. '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/benchmarks/substrate/test_embedder.py -v -o addopts=""`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/benchmarks/substrate/__init__.py tests/benchmarks/substrate/embedder.py tests/benchmarks/substrate/test_embedder.py
git commit -m "feat(#414): deterministic bag-of-token-hash embedder for latency substrate"
```

---

## Task 2: Minimal substrate schema DDL

**Files:**
- Create: `tests/benchmarks/substrate/bench_schema.sql`

This DDL creates only the columns the two SQL functions reference (verified against
`011_hybrid_search_functions_fixed.sql` + `022_hybrid_search_max_staleness.sql`). The
`search_vector` generated columns, indexes, and the functions themselves are added by
loading 011 then 022 on top of this schema (Task 6 CI step). It also creates the
`authenticated` role so the `GRANT ... TO authenticated` lines in 011/022 succeed on a
bare Postgres.

- [ ] **Step 1: Write the schema file**

```sql
-- tests/benchmarks/substrate/bench_schema.sql
-- Minimal schema for the HybridRetriever latency substrate (#414).
-- Load order: this file -> 011_hybrid_search_functions_fixed.sql -> 022_...sql -> seed.
CREATE EXTENSION IF NOT EXISTS vector;

-- 011/022 GRANT EXECUTE ... TO authenticated; create the role so they don't error.
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        CREATE ROLE authenticated;
    END IF;
END $$;

-- Vector-search source tables (hybrid_vector_search) --------------------------
CREATE TABLE IF NOT EXISTS episodic_memories (
    memory_id        text PRIMARY KEY,
    description      text,
    embedding        vector(1536),
    event_type       text,
    agent_name       text,
    occurred_at      timestamptz,
    brand            text,
    region           text,
    patient_id       text,
    hcp_id           text,
    importance_score double precision
);

CREATE TABLE IF NOT EXISTS procedural_memories (
    procedure_id       text PRIMARY KEY,
    procedure_name     text,
    trigger_pattern    text,
    trigger_embedding  vector(1536),
    is_active          boolean DEFAULT true,
    success_count      integer DEFAULT 0,
    procedure_type     text,
    success_rate       double precision,
    usage_count        integer,
    applicable_brands  text[],
    applicable_regions text[],
    detected_intent    text
);

-- Full-text source tables (hybrid_fulltext_search) ----------------------------
-- search_vector (GENERATED) + GIN indexes are added by 011's ALTER statements.
CREATE TABLE IF NOT EXISTS causal_paths (
    path_id            text PRIMARY KEY,
    start_node         text,
    end_node           text,
    method_used        text,
    causal_chain       jsonb,
    causal_effect_size double precision,
    confidence_level   double precision,
    created_at         timestamptz
);

CREATE TABLE IF NOT EXISTS agent_activities (
    activity_id      text PRIMARY KEY,
    agent_name       text,
    activity_type    text,
    analysis_results jsonb,
    agent_tier       text,
    status           text,
    created_at       timestamptz,
    workstream       text
);

CREATE TABLE IF NOT EXISTS triggers (
    trigger_id         text PRIMARY KEY,
    trigger_reason     text,
    trigger_type       text,
    recommended_action text,
    priority           text,
    confidence_score   double precision,
    created_at         timestamptz,
    invalidated_at     timestamptz  -- referenced by 022's max_staleness filter
);
```

- [ ] **Step 2: Verify the schema + functions load into a throwaway pgvector container**

Run:
```bash
docker run -d --rm --name bench_pg_ddl -e POSTGRES_PASSWORD=bench -p 55432:5432 pgvector/pgvector:pg16
sleep 6
PGPASSWORD=bench psql -h localhost -p 55432 -U postgres -d postgres \
  -v ON_ERROR_STOP=1 \
  -f tests/benchmarks/substrate/bench_schema.sql \
  -f database/memory/011_hybrid_search_functions_fixed.sql \
  -f database/memory/022_hybrid_search_max_staleness.sql \
  -c "SELECT proname FROM pg_proc WHERE proname IN ('hybrid_vector_search','hybrid_fulltext_search');"
docker stop bench_pg_ddl
```
Expected: no `ON_ERROR_STOP` failures; the final query lists both `hybrid_vector_search` and `hybrid_fulltext_search`.

- [ ] **Step 3: Commit**

```bash
git add tests/benchmarks/substrate/bench_schema.sql
git commit -m "feat(#414): minimal pg+pgvector schema for hybrid latency substrate"
```

---

## Task 3: DirectSQLMemoryConnector

**Files:**
- Create: `tests/benchmarks/substrate/direct_sql_connector.py`

Implements the 4 methods the retriever calls (`vector_search_by_text`, `fulltext_search`,
`graph_traverse`, `graph_traverse_kpi`), mirroring the result-shaping logic in
`src/rag/memory_connector.py` but calling the SQL functions directly via psycopg2.
**Fail-closed (part 1 of 2):** no `except: return []` — DB errors raise from this
connector. NOTE (codex audit HIGH-1): `HybridRetriever`'s own dense/sparse paths swallow
exceptions (`retriever.py:88,146`), so this raise alone is NOT sufficient — Task 6 adds a
direct-connector preflight that bypasses that swallow. Behavioral tests live in Task 5
(they need the seeded DB).

- [ ] **Step 1: Confirm the RetrievalResult / RetrievalSource imports**

Run: `sed -n '1,30p' src/rag/memory_connector.py | grep -nE 'import|RetrievalSource|RetrievalResult'`
Expected: shows the exact import line for `RetrievalResult` and `RetrievalSource`. Use the
SAME import paths in the next step (do not guess).

- [ ] **Step 2: Write the connector**

```python
# tests/benchmarks/substrate/direct_sql_connector.py
"""Test-side MemoryConnector that calls the real hybrid_vector_search /
hybrid_fulltext_search SQL functions via psycopg2 against a local pgvector
substrate (#414). Fail-closed: raises on DB errors (no silent return []).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras

# Imports verified against src/rag/memory_connector.py:20-21 — RetrievalSource
# lives in src.rag.types, NOT retrieval_models (codex audit MED-1):
from src.rag.models.retrieval_models import RetrievalResult
from src.rag.types import RetrievalSource
from tests.benchmarks.substrate.embedder import embed_text, to_pgvector_literal


class DirectSQLMemoryConnector:
    def __init__(self, dsn: str) -> None:
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True

    async def get_embedding_service(self):  # API parity; unused on this path
        return None

    async def vector_search_by_text(
        self, query_text, k=10, filters=None, min_similarity=0.5, max_staleness=None
    ) -> List[RetrievalResult]:
        embedding = embed_text(query_text)
        return await self.vector_search(
            embedding, k=k, filters=filters, min_similarity=min_similarity,
            max_staleness=max_staleness,
        )

    async def vector_search(
        self, query_embedding, k=10, filters=None, min_similarity=0.5, max_staleness=None
    ) -> List[RetrievalResult]:
        rpc_filters: Dict[str, Any] = dict(filters or {})
        if max_staleness is not None:
            rpc_filters["max_staleness"] = max_staleness
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM hybrid_vector_search(%s::vector, %s, %s::jsonb)",
                (to_pgvector_literal(query_embedding), k, json.dumps(rpc_filters)),
            )
            rows = cur.fetchall()
        results: List[RetrievalResult] = []
        for row in rows:
            similarity = float(row.get("similarity") or 0.0)
            if similarity < min_similarity:
                continue
            md = dict(row.get("metadata") or {})
            md["source_name"] = row.get("source_table", "unknown")
            results.append(
                RetrievalResult(
                    source_id=row.get("id", ""), content=row.get("content", ""),
                    source=RetrievalSource.VECTOR.value, score=similarity,
                    retrieval_method="dense", metadata=md,
                )
            )
        return results

    async def fulltext_search(
        self, query_text, k=10, filters=None, max_staleness=None
    ) -> List[RetrievalResult]:
        rpc_filters: Dict[str, Any] = dict(filters or {})
        if max_staleness is not None:
            rpc_filters["max_staleness"] = max_staleness
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM hybrid_fulltext_search(%s, %s, %s::jsonb)",
                (query_text, k, json.dumps(rpc_filters)),
            )
            rows = cur.fetchall()
        max_rank = max((float(r.get("rank") or 0.0) for r in rows), default=0.0)
        results: List[RetrievalResult] = []
        for row in rows:
            rank = float(row.get("rank") or 0.0)
            score = rank / max_rank if max_rank > 0 else 0.0
            md = dict(row.get("metadata") or {})
            md["source_name"] = row.get("source_table", "unknown")
            results.append(
                RetrievalResult(
                    source_id=row.get("id", ""), content=row.get("content", ""),
                    source=RetrievalSource.FULLTEXT.value, score=score,
                    retrieval_method="sparse", metadata=md,
                )
            )
        return results

    def graph_traverse(self, entity_id, relationship="causal_path", max_depth=3):
        return []  # latency benchmark never invokes the graph stream

    def graph_traverse_kpi(self, kpi_name, min_confidence=0.5):
        return []

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
```

- [ ] **Step 3: Verify it imports cleanly (no syntax/import errors)**

Run: `python -c "from tests.benchmarks.substrate.direct_sql_connector import DirectSQLMemoryConnector; print('ok')"`
Expected: prints `ok`. If the `RetrievalSource` import path differs, fix it to match Step 1.

- [ ] **Step 4: Commit**

```bash
git add tests/benchmarks/substrate/direct_sql_connector.py
git commit -m "feat(#414): DirectSQLMemoryConnector (real SQL fns via psycopg2, fail-closed)"
```

---

## Task 4: Deterministic seed generator

**Files:**
- Create: `tests/benchmarks/substrate/seed.py`

Seeds the 5 tables deterministically. For each benchmark query it plants a handful of
**relevant** rows whose text echoes the query tokens (so the vector stream clears the
0.5 floor and the full-text stream matches), plus bulk **filler** rows so the HNSW/GIN
indexes do real traversal work. Idempotent: truncates the 5 tables first.

- [ ] **Step 1: Write the seeder**

```python
# tests/benchmarks/substrate/seed.py
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
```

- [ ] **Step 2: Commit (behavioral verification happens in Task 5 against the seeded DB)**

```bash
git add tests/benchmarks/substrate/seed.py
git commit -m "feat(#414): deterministic corpus seeder for hybrid latency substrate"
```

---

## Task 5: Fixture + integration tests (seed→non-empty, fail-loud)

**Files:**
- Create: `tests/benchmarks/substrate/fixture.py`
- Create: `tests/benchmarks/substrate/test_substrate_integration.py`

- [ ] **Step 1: Write the fixture + `substrate_ready()` helper**

```python
# tests/benchmarks/substrate/fixture.py
"""Substrate readiness + connector injection for the hybrid latency benchmark (#414)."""

from __future__ import annotations

import os

import src.rag.memory_connector as _mc
from tests.benchmarks.substrate.direct_sql_connector import DirectSQLMemoryConnector


def substrate_ready() -> bool:
    """True when the local pg substrate is configured for this run."""
    return os.getenv("BENCH_SUBSTRATE") == "local_pg" and bool(os.getenv("BENCH_PG_DSN"))


def make_connector() -> DirectSQLMemoryConnector:
    return DirectSQLMemoryConnector(os.environ["BENCH_PG_DSN"])


def inject(connector) -> None:
    """Install a connector as the process-wide singleton."""
    _mc._memory_connector = connector


def reset() -> None:
    _mc.reset_memory_connector()
```

- [ ] **Step 2: Write the failing integration test**

```python
# tests/benchmarks/substrate/test_substrate_integration.py
import asyncio

import pytest

from tests.benchmarks.substrate.fixture import substrate_ready
from tests.benchmarks.substrate.direct_sql_connector import DirectSQLMemoryConnector

pytestmark = pytest.mark.skipif(
    not substrate_ready(), reason="local pg substrate not configured (set BENCH_SUBSTRATE/BENCH_PG_DSN)"
)


def test_vector_stream_returns_nonempty():
    import os

    conn = DirectSQLMemoryConnector(os.environ["BENCH_PG_DSN"])
    try:
        results = asyncio.run(
            conn.vector_search_by_text("kisqali trx growth west region q3", k=10)
        )
    finally:
        conn.close()
    assert results, "vector stream returned empty against a seeded substrate"


def test_fulltext_stream_returns_nonempty():
    import os

    conn = DirectSQLMemoryConnector(os.environ["BENCH_PG_DSN"])
    try:
        results = asyncio.run(conn.fulltext_search("kisqali trx growth", k=10))
    finally:
        conn.close()
    assert results, "full-text stream returned empty against a seeded substrate"


def test_broken_substrate_fails_loud():
    """A bad DSN must RAISE, never silently return [] (the #403 failure mode)."""
    with pytest.raises(Exception):
        DirectSQLMemoryConnector("postgresql://nobody@127.0.0.1:1/none")
```

- [ ] **Step 3: Bring up a seeded substrate and run the integration test**

Run:
```bash
docker run -d --rm --name bench_pg -e POSTGRES_PASSWORD=bench -p 55432:5432 pgvector/pgvector:pg16
sleep 6
export BENCH_PG_DSN="postgresql://postgres:bench@localhost:55432/postgres"
PGPASSWORD=bench psql "$BENCH_PG_DSN" -v ON_ERROR_STOP=1 \
  -f tests/benchmarks/substrate/bench_schema.sql \
  -f database/memory/011_hybrid_search_functions_fixed.sql \
  -f database/memory/022_hybrid_search_max_staleness.sql
python -m tests.benchmarks.substrate.seed
BENCH_SUBSTRATE=local_pg pytest tests/benchmarks/substrate/test_substrate_integration.py -v -o addopts=""
docker stop bench_pg
```
Expected: 3 passed — both streams non-empty; broken-DSN raises.

- [ ] **Step 4: Commit**

```bash
git add tests/benchmarks/substrate/fixture.py tests/benchmarks/substrate/test_substrate_integration.py
git commit -m "test(#414): substrate fixture + seed->non-empty + fail-loud integration tests"
```

---

## Task 6: Refactor the latency benchmark to use the substrate

**Files:**
- Modify: `tests/benchmarks/test_hybrid_retriever_latency.py`

Replace the `requires_supabase` + `sk-*`-only skip with `substrate_ready()`-aware logic,
inject the connector when in substrate mode, and **fail on all-empty results** (a
reachable-but-unseeded substrate must not bless a fast 0.0).

- [ ] **Step 1: Read the current skip + run block**

Run: `sed -n '160,230p' tests/benchmarks/test_hybrid_retriever_latency.py`
Expected: shows the `@pytest.mark.requires_supabase` decorator, the `_retrieval_env_ready()`
skip, the per-query timing loop, and the p50/p95 computation.

- [ ] **Step 2: Add the substrate import + readiness gate (replace the marker/skip)**

Replace the `@pytest.mark.requires_supabase` decorator line and the opening
`if not _retrieval_env_ready(): pytest.skip(...)` block with:

```python
from tests.benchmarks.substrate.fixture import (  # add to imports near top
    inject as _inject_substrate,
    make_connector as _make_substrate_connector,
    reset as _reset_substrate,
    substrate_ready as _substrate_ready,
)
```

```python
# (decorator: keep @pytest.mark.timeout(600); REMOVE @pytest.mark.requires_supabase)
@pytest.mark.timeout(600)
def test_hybrid_retriever_latency_against_baseline() -> None:
    # ... existing docstring (updated in Task 9) ...
    use_substrate = _substrate_ready()
    if not use_substrate and not _retrieval_env_ready():
        pytest.skip(
            "no benchmark substrate: set BENCH_SUBSTRATE=local_pg + BENCH_PG_DSN "
            "for the local pgvector path, or provide live SUPABASE/OPENAI creds."
        )

    queries = load_queries(_QUERY_FILE)
    baseline = _load_baseline()

    connector = _make_substrate_connector() if use_substrate else None
    if use_substrate:
        _inject_substrate(connector)
    try:
        # FAIL-CLOSED preflight (codex audit HIGH-1 + HIGH-2). HybridRetriever's
        # dense/sparse paths wrap the connector in `except: return []`
        # (retriever.py:88,146), so a broken/unseeded substrate would otherwise
        # surface as empty results, NOT an error. Here we call the connector
        # DIRECTLY (bypassing that swallow): a DB/connection/SQL error RAISES and
        # fails the run, and we require EVERY query to return rows on BOTH streams
        # (not just one) so a partially-seeded substrate also fails loudly.
        if use_substrate:
            pre_loop = asyncio.new_event_loop()
            try:
                empties = []
                for q in queries:
                    dense = pre_loop.run_until_complete(
                        connector.vector_search_by_text(
                            q.query_text, k=_TOP_K, filters=q.filters or None,
                            max_staleness=q.max_staleness,
                        )
                    )
                    sparse = pre_loop.run_until_complete(
                        connector.fulltext_search(
                            q.query_text, k=_TOP_K, filters=q.filters or None,
                            max_staleness=q.max_staleness,
                        )
                    )
                    if not dense or not sparse:
                        empties.append((q.query_text[:40], len(dense), len(sparse)))
            finally:
                pre_loop.close()
            assert not empties, (
                "FAIL-CLOSED: substrate returned an empty stream for "
                f"{len(empties)}/{len(queries)} queries "
                f"(query, dense_n, sparse_n): {empties[:5]}. The substrate is "
                "broken or unseeded — refusing to bless (issue #403 mode)."
            )

        # Timed fused run through the real retriever path.
        timings_ms: List[float] = []
        loop = asyncio.new_event_loop()
        try:
            for q in queries:
                elapsed = loop.run_until_complete(_run_one_query_timed(q, k=_TOP_K))
                timings_ms.append(elapsed)
        finally:
            loop.close()
        # ... existing p50/p95 compute + print + write_measurements + placeholder
        #     early-return + tolerance-band assertion continue UNCHANGED below ...
    finally:
        if use_substrate:
            _reset_substrate()
            connector.close()
```

- [ ] **Step 3: (no change to `_run_one_query_timed`)**

The fail-closed guarantee now comes from the Step 2 direct-connector preflight, so
`_run_one_query_timed` keeps its production signature (returns `float`). No edit to it is
needed — do NOT add a result-count return (that earlier approach was rejected because the
retriever's `except: return []` swallow made an in-loop count guard unreliable).

- [ ] **Step 4: Run the benchmark locally against the seeded substrate**

Run (with the seeded container from Task 5 still up, or re-create + re-seed it):
```bash
export BENCH_SUBSTRATE=local_pg
export BENCH_PG_DSN="postgresql://postgres:bench@localhost:55432/postgres"
pytest tests/benchmarks/test_hybrid_retriever_latency.py -v -o addopts="" -p no:xdist
```
Expected: the test RUNS (not skipped) and **PASSES** — with the baseline still at `0.0`,
the existing placeholder early-return (`if p50_baseline == 0.0 ...: return`,
`test_hybrid_retriever_latency.py:264`) returns before the tolerance assertion, while still
printing the `[issue-#391 box-2]` p50/p95 line and writing `measurements.json`. Record the
printed p50/p95. The fail-closed preflight must NOT trip; if it does, the substrate is
broken/unseeded — fix before proceeding. (After Task 11 sets `mean_ms > 0`, the early-return
is bypassed and the tolerance assertion guards for real.)

- [ ] **Step 5: Commit**

```bash
git add tests/benchmarks/test_hybrid_retriever_latency.py
git commit -m "feat(#414): run hybrid latency benchmark against local pg substrate, fail-closed"
```

---

## Task 7: Cheapest-disproof — local stability check

**Files:** none (validation task; resolves the spec §4 embedder-faithfulness premise)

- [ ] **Step 1: Measure p50/p95 across 3 local repeats**

Run (seeded container up):
```bash
export BENCH_SUBSTRATE=local_pg BENCH_PG_DSN="postgresql://postgres:bench@localhost:55432/postgres"
for i in 1 2 3; do
  pytest tests/benchmarks/test_hybrid_retriever_latency.py -q -o addopts="" -p no:xdist 2>&1 \
    | grep -E "p50_ms|p95_ms" || true
done
```
Expected: p50/p95 are **non-zero and stable** across the 3 repeats (cross-run spread well
within the tolerance bands we'll derive). If timings are sub-millisecond / collapse to
noise, the corpus is too small — bump `FILLER_EPISODIC` in `seed.py` (e.g. to 4000),
re-seed, and re-measure. **Do not proceed to CI re-bless until timings are stable.**

- [ ] **Step 2: Tear down the local container**

Run: `docker stop bench_pg 2>/dev/null || true`

No commit (unless `seed.py` corpus size was tuned — then `git commit -am "tune(#414): bump filler corpus for stable substrate timings"`).

---

## Task 8: Wire the substrate into CI

**Files:**
- Modify: `.github/workflows/benchmarks.yml`

- [ ] **Step 1: Read the current Box 2 job/steps**

Run: `sed -n '62,135p' .github/workflows/benchmarks.yml`
Expected: shows the `performance-benchmarks` job (`runs-on`, `env`, install step, the three
benchmark-run steps).

- [ ] **Step 2: Add the pgvector service container to the job**

Under the `performance-benchmarks:` job (sibling of `runs-on`/`strategy`/`env`), add:

```yaml
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_PASSWORD: bench
          POSTGRES_DB: postgres
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U postgres"
          --health-interval 5s
          --health-timeout 5s
          --health-retries 10
```

- [ ] **Step 3: Add a "Provision substrate" step BEFORE the hybrid (Box 2) step**

Insert immediately before the `Run HybridRetriever latency benchmark (box 2)` step:

```yaml
      - name: Provision hybrid-retriever substrate (box 2)
        if: always()  # codex audit LOW: don't skip provisioning if box 1 failed
        env:
          PGPASSWORD: bench
          BENCH_PG_DSN: postgresql://postgres:bench@localhost:5432/postgres
        run: |
          sudo apt-get update && sudo apt-get install -y postgresql-client
          psql "$BENCH_PG_DSN" -v ON_ERROR_STOP=1 \
            -f tests/benchmarks/substrate/bench_schema.sql \
            -f database/memory/011_hybrid_search_functions_fixed.sql \
            -f database/memory/022_hybrid_search_max_staleness.sql
          python -m tests.benchmarks.substrate.seed
```

- [ ] **Step 4: Point the Box 2 step at the substrate**

Edit the `Run HybridRetriever latency benchmark (box 2)` step: add an `env:` block and
keep `if: always()`:

```yaml
      - name: Run HybridRetriever latency benchmark (box 2)
        if: always()
        env:
          BENCH_SUBSTRATE: local_pg
          BENCH_PG_DSN: postgresql://postgres:bench@localhost:5432/postgres
        run: scripts/run_benchmarks.sh hybrid
```

- [ ] **Step 5: Validate the workflow YAML parses**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/benchmarks.yml')); print('yaml ok')"`
Expected: `yaml ok`

- [ ] **Step 6: Lint + commit**

Run: `ruff check tests/benchmarks/substrate/ && ruff format --check tests/benchmarks/substrate/`
Expected: clean (run `ruff format tests/benchmarks/substrate/` first if needed).

```bash
git add .github/workflows/benchmarks.yml
git commit -m "ci(#414): seed local pgvector substrate for hybrid latency benchmark"
```

---

## Task 9: Update the benchmark docstring + push for CI

**Files:**
- Modify: `tests/benchmarks/test_hybrid_retriever_latency.py` (module + test docstrings)

- [ ] **Step 1: Update the module docstring**

Replace the "skips in CI without SUPABASE/OPENAI / placeholder strategy retained for
hybrid" language (`tests/benchmarks/test_hybrid_retriever_latency.py:29-44`) with a
description of the local-substrate path: runs against a seeded `pgvector` container via
`BENCH_SUBSTRATE=local_pg` + `BENCH_PG_DSN`, deterministic embedder, fail-closed on
all-empty; the live SUPABASE/OPENAI path is the legacy fallback. Keep it factual and
concise.

- [ ] **Step 2: Push the branch and open the PR (CI runs the substrate path)**

```bash
git push -u origin feat/414-hybrid-latency-local-substrate
gh pr create --title "feat(#414): local pg+pgvector substrate for HybridRetriever latency guard" \
  --body-file docs/superpowers/specs/2026-05-25-hybrid-retriever-latency-local-substrate-design.md \
  --base main
```
Note: per repo memory, `gh pr edit --body-file` silently fails — if the body needs edits
later, use `gh api repos/enunezvn/e2i_causal_analytics/pulls/<N> -X PATCH -f body=...`.
At this point CI box-2 **passes** via the placeholder early-return (baseline still 0.0)
while still emitting `measurements.json` + the printed p50/p95; the fail-closed preflight
still guards a broken substrate. Task 11 re-blesses from those measurements — once
`mean_ms > 0` the early-return is bypassed and the tolerance assertion guards for real.

- [ ] **Step 3: Commit**

```bash
git add tests/benchmarks/test_hybrid_retriever_latency.py
git commit -m "docs(#414): document local-substrate path in hybrid latency benchmark"
git push
```

---

## Task 10: Collect ≥3 CI measurements

**Files:** none (operational)

- [ ] **Step 1: Trigger ≥3 workflow_dispatch runs on the branch**

```bash
gh workflow run benchmarks.yml --ref feat/414-hybrid-latency-local-substrate
# repeat 3x (wait for each to register), then:
gh run list --workflow benchmarks.yml --branch feat/414-hybrid-latency-local-substrate
```
Expected: ≥3 completed runs. Record their run IDs (for `_blessed_from_ci_runs`).

- [ ] **Step 2: Download each run's measurements artifact**

```bash
for rid in <RUN_ID_1> <RUN_ID_2> <RUN_ID_3>; do
  gh run download "$rid" -n performance-benchmark-results -D "/tmp/bench_$rid"
done
```
Expected: each dir contains the box-2 `measurements.json` with per-query p50/p95.
Compute **median-of-p50s** and **median-of-p95s** across the 3 runs, and the cross-run
**stdev** for each (used for tolerance bands). Note: box-2 **passes** in these runs via the
placeholder early-return at 0.0 baseline, but `measurements.json` is written before that
return — so the numbers are available even though the tolerance assertion hasn't run yet.

---

## Task 11: Re-bless the baselines

**Files:**
- Modify: `tests/benchmarks/baselines/performance.json`

- [ ] **Step 1: Edit the two hybrid entries**

For `hybrid_retriever_search_p50` and `hybrid_retriever_search_p95`:
- Set `mean_ms` to the median-of-medians from Task 10 (p50 and p95 respectively).
- **Remove** `_placeholder_rationale` and `_followup_issue`.
- **Add** `_ci_observation` (mirror `cascade_5hop_bfs`'s shape): the 3 per-run medians, the
  med-of-meds, cross-run stdev (absolute + %), and a one-line note that this is measured
  against the seeded local `pgvector` substrate (not live Supabase/OpenAI), so absolute ms
  reflect deterministic-embedder + local-index cost, not production network latency.
- **Add** per-box `_blessed_from_ci_runs` (list of the 3 run IDs).
- Set `tolerance_pct` / `tolerance_abs_ms` **derived from observed cross-run stdev** (e.g.
  band ≈ max(3×stdev%, a small absolute floor)); replace the `_tolerance_rationale` to
  state the bands were derived from the 3-run substrate variance, not the PR #401
  placeholder defaults.

- [ ] **Step 2: Verify the JSON is well-formed and means are positive**

Run:
```bash
python -c "import json; d=json.load(open('tests/benchmarks/baselines/performance.json')); \
print(d['hybrid_retriever_search_p50']['mean_ms'], d['hybrid_retriever_search_p95']['mean_ms']); \
assert d['hybrid_retriever_search_p50']['mean_ms']>0 and d['hybrid_retriever_search_p95']['mean_ms']>0"
```
Expected: prints two positive numbers; no assertion error.

- [ ] **Step 3: Commit**

```bash
git add tests/benchmarks/baselines/performance.json
git commit -m "perf(#414): re-bless hybrid p50/p95 from local-substrate CI medians"
```

---

## Task 12: Update the drift-guard meta-test

**Files:**
- Modify: `tests/unit/test_benchmarks_meta/test_baseline_no_placeholder.py`

- [ ] **Step 1: Move the hybrid boxes from placeholder → reblessed**

- Add the two hybrid boxes to `_REBLESSED_BOXES` (`test_baseline_no_placeholder.py:45-50`):

```python
_REBLESSED_BOXES = (
    "cascade_5hop_bfs",
    "bm25_build_1k",
    "bm25_build_5k",
    "bm25_build_10k",
    "hybrid_retriever_search_p50",
    "hybrid_retriever_search_p95",
)
```

- Delete the `_PLACEHOLDER_BOXES_HYBRID` tuple (lines 52-59) and delete the now-obsolete
  `test_hybrid_boxes_are_still_placeholder_with_refreshed_breadcrumb` test (lines 191-226).
- Update the module docstring (lines 1-30) to state all 6 boxes are now CI-blessed
  (`cascade` + `bm25` from #403; the 2 `hybrid` boxes from #414 via the local pgvector
  substrate), removing the "2 hybrid baselines remain placeholder" language.

- [ ] **Step 2: Run the full meta-test + the substrate unit tests**

Run: `pytest tests/unit/test_benchmarks_meta/test_baseline_no_placeholder.py tests/benchmarks/substrate/test_embedder.py -v -o addopts=""`
Expected: all pass — the parametrized `_REBLESSED_BOXES` tests now cover the 2 hybrid boxes
(positive `mean_ms`, no placeholder breadcrumbs, provenance present).

- [ ] **Step 3: Commit + push; confirm CI is green**

```bash
git add tests/unit/test_benchmarks_meta/test_baseline_no_placeholder.py
git commit -m "test(#414): drift-guard pins hybrid baselines as CI-blessed"
git push
gh run list --workflow benchmarks.yml --branch feat/414-hybrid-latency-local-substrate --limit 1
```
Expected: a fresh benchmarks run is GREEN (box-2 observed p50/p95 now compare against the
blessed baseline within the derived band). The PR is ready for review/merge with
`Closes #414`.

---

## Notes for the executor
- Run benchmark/substrate tests with `-o addopts="" -p no:xdist` — the repo's default
  `-n 4 --dist=loadscope` xdist can starve async retriever calls (per
  `retrieval-benchmarks.yml:80-86`).
- The minimal RAGAS-style venv caveats don't apply here; use the normal project env.
- This substrate also unblocks #377 (retrieval-quality), but re-blessing that is **out of
  scope** — it would need a real embedder swapped in for the hash one.
