"""
E2I Hybrid RAG - Backend Clients

This module contains individual backend client implementations:
- VectorBackend: Supabase/pgvector for semantic search
- FulltextBackend: PostgreSQL full-text search
- GraphBackend: FalkorDB/Graphiti for knowledge graph queries

Part of Phase 1, Checkpoint 1.3.
"""

from src.rag.backends.vector import VectorBackend
from src.rag.backends.fulltext import FulltextBackend
from src.rag.backends.graph import GraphBackend

__all__ = [
    "VectorBackend",
    "FulltextBackend",
    "GraphBackend",
]
