"""Test-side MemoryConnector that calls the real hybrid_vector_search /
hybrid_fulltext_search SQL functions via psycopg2 against a local pgvector
substrate (#414). Fail-closed: raises on DB errors (no silent return []).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import psycopg2
import psycopg2.extras

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
            embedding,
            k=k,
            filters=filters,
            min_similarity=min_similarity,
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
                    source_id=row.get("id", ""),
                    content=row.get("content", ""),
                    source=RetrievalSource.VECTOR.value,
                    score=similarity,
                    retrieval_method="dense",
                    metadata=md,
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
                    source_id=row.get("id", ""),
                    content=row.get("content", ""),
                    source=RetrievalSource.FULLTEXT.value,
                    score=score,
                    retrieval_method="sparse",
                    metadata=md,
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
