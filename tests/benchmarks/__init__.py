"""Retrieval-quality benchmark harness for HybridRetriever.

See issue #377 + plan §"Phase 2 — Remaining" line 74 + §"Recommended sequencing"
item 4. Provides Recall@10 / MRR metrics + a regression-gated harness so
retrieval-quality drift gets caught when domain vocab, indexes, or weights
change.
"""
