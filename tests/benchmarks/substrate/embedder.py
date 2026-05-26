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
