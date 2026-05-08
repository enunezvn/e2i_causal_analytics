"""Live integration test for KnowledgeGraphQuerier (Phase 2.3).

Skipped automatically when ``UMLS_UTS_API_KEY`` is absent. When present,
this test exercises ``query_disease_hierarchy`` against the real UTS
endpoint for ``C0011615`` (atopic dermatitis) and asserts that:

1. At least one taxonomic edge is returned (atopic dermatitis is well-known
   to have multiple parent and child concepts in the metathesaurus).
2. Every returned edge has the ``KGEdge`` shape contract — non-empty
   subject_id starting with ``C``, evidence_source ``"umls_relations"``,
   and a fine-grained predicate.

Open Targets has a separate live test slot reserved for the v1 zero-auth
verification (deferred — needs no API key but adds external network
dependency to the unit suite). The Open Targets path is fully covered by
unit tests with ``httpx.MockTransport``.
"""

from __future__ import annotations

import os

import pytest

from src.data.kg.kg_querier import KnowledgeGraphQuerier
from src.data.kg.umls_uts import reset_caches

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_API_KEY = os.environ.get("UMLS_UTS_API_KEY")
_REASON = "UMLS_UTS_API_KEY not set; skipping live KGQuerier test."

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _API_KEY, reason=_REASON),
]


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def test_query_disease_hierarchy_returns_taxonomic_edges() -> None:
    """C0011615 (atopic dermatitis) has known parents/children in UMLS."""
    querier = KnowledgeGraphQuerier()
    edges = querier.query_disease_hierarchy("C0011615")
    assert len(edges) >= 1, "C0011615 should produce at least one taxonomic edge in UMLS; got 0"
    for edge in edges:
        assert edge.subject_id == "C0011615"
        assert edge.evidence_source == "umls_relations"
        # Object must be a CUI (starts with C).
        assert edge.object_id.startswith("C")
        # Predicate is one of the taxonomic shape labels (fine or coarse).
        assert edge.predicate in {
            "isa",
            "inverse_isa",
            "is_a",
            "subclass_of",
            "superclass_of",
            "par",
            "chd",
            "rb",
            "rn",
        }
