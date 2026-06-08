"""Shared canonicalization for Reciprocal Rank Fusion dedup.

Both RRF implementations -- ``src/rag/retriever.py`` (the LIVE chatbot
``_reciprocal_rank_fusion``) and ``src/rag/hybrid_retriever.py`` (the
orchestrator/REST ``_apply_rrf_fusion``) -- MUST use this single helper so a
dedup fix reaches both. Dedup is content-aware: two rows with identical content
under different source_ids collapse to one fused row (they would otherwise both
accrue RRF mass and waste top-k slots). Empty/blank content falls back to the
row id so unrelated empty rows are NOT collapsed.
"""

from __future__ import annotations


def dedup_key(content: str, source_id: str) -> str:
    """Return a stable dedup key for an RRF result.

    Identical (case/whitespace-insensitive) content -> identical key, so
    duplicate-content rows collapse. Blank content -> fall back to ``source_id``
    so distinct empty rows stay distinct (rows with no usable content are
    identified solely by their id).
    """
    normalized = " ".join((content or "").split()).casefold()
    if normalized:
        return f"content::{normalized}"
    return f"id::{source_id}"
