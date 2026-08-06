"""One shape for ``learning_signals.retrieved_chunks`` (#1489).

``database/ml/022_self_improvement_tables.sql:143`` added the column "for
RAGAS evaluation" and it had no producer at all — measured on the live DB
2026-08-06, non-default on 0 of 3,959 rows. #1489 deferral 1 wires two:

* the cognitive Reflector, from ``state.evidence_board`` (source and hop are
  known there), and
* ``RubricNode._store_evaluation``, from ``EvaluationContext
  .retrieved_contexts`` (bare strings — that is all the rubric path has).

They write the same column, so they must agree on its shape or every reader
has to branch on which producer wrote the row. ``content`` is the key both
always set and the one a RAGAS judge scores against; producers with more
provenance add keys, they never rename that one.

Deliberately stdlib-only and dependency-free: the cognitive path imports it
under dspy and the feedback-learner path under pydantic, and neither should
drag the other's imports along.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

__all__ = ["MAX_CHUNK_CONTENT_CHARS", "chunk_payload", "chunks_from_texts"]

# Retrieval chunks are normally hundreds to a couple of thousand characters,
# so this cap is not reached on real traffic; it exists so one pathological
# evidence item cannot write an unbounded JSONB blob on a live turn.
MAX_CHUNK_CONTENT_CHARS = 4000


def chunk_payload(content: Any, **provenance: Any) -> Dict[str, Any]:
    """One chunk, capped and marked when the cap bit.

    A cut chunk carries ``truncated: True``. An unmarked cut would understate
    what the answer was grounded in, and faithfulness is judged against
    exactly this text — a reader comparing a score to a silently shortened
    context would blame the pipeline for the storage layer's trimming.
    """
    text = str(content)
    chunk: Dict[str, Any] = {"content": text[:MAX_CHUNK_CONTENT_CHARS]}
    chunk.update(provenance)
    if len(text) > MAX_CHUNK_CONTENT_CHARS:
        chunk["truncated"] = True
    return chunk


def chunks_from_texts(texts: Iterable[Any]) -> List[Dict[str, Any]]:
    """Chunk payloads for a producer that has only the context strings.

    Every text produces exactly one chunk: dropping any would misreport what
    retrieval returned.
    """
    return [chunk_payload(text) for text in texts]
