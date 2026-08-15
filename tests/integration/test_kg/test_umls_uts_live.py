"""Live UMLS UTS integration test.

Skipped automatically when UMLS_UTS_API_KEY is absent. When the key is
present, this test hits the real UTS endpoint and asserts:

1. ICD-10-CM ``L20.9`` (atopic dermatitis, unspecified) cross-walks to
   CUI ``C0011615``.
2. CUI ``C0011615`` has the canonical preferred name and at least one
   ``Disease or Syndrome`` semantic type.
3. EntityLinker.resolve_icd10 returns a resolved EntityLink end-to-end.

This test exercises the full ``EntityLinker → UMLSClient → UTS REST``
chain. It is marked ``slow`` (alongside ``integration``) so it runs on
``slow-tests.yml`` and NOT on the PR lane — the same marking as every
sibling live-contract file.

``integration`` alone did not achieve that (#1629). The PR lane runs
``pytest tests/integration/ -m "not slow"`` and Job A of slow-tests runs
``pytest tests/ -m slow``, so an ``integration``-only file is selected by
the PR lane and excluded from slow-tests — the exact opposite of the
intent. It went unnoticed because CI holds no ``UMLS_UTS_API_KEY``, so the
skipif below fired on every run: the test had never actually executed
anywhere, and would have started hitting the live network on the
PR-blocking lane the moment a key was added.

UTS calls are gentle (3 endpoints, single round-trip each) and each is
covered by an in-process LRU cache, so re-runs in the same process are
free.
"""

from __future__ import annotations

import os

import pytest

from src.data.kg.entity_linker import EntityLinker
from src.data.kg.umls_uts import UMLSClient, reset_caches

# Load .env so UMLS_UTS_API_KEY is available even when pytest is launched
# from a context where the env var hasn't been exported.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


_API_KEY = os.environ.get("UMLS_UTS_API_KEY")
_REASON = "UMLS_UTS_API_KEY not set; skipping live UMLS test."

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not _API_KEY, reason=_REASON),
]


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()


def test_icd10_atopic_dermatitis_resolves_to_c0011615() -> None:
    with UMLSClient() as client:
        cui = client.code_to_cui("L20.9", source="ICD10CM")
        assert cui == "C0011615"


def test_cui_lookup_returns_disease_semantic_type() -> None:
    with UMLSClient() as client:
        concept = client.cui_lookup("C0011615")
        assert concept.preferred_name.lower().startswith("dermatitis")
        # UTS returns "Disease or Syndrome" for the atopic dermatitis CUI.
        assert any("disease" in s.lower() for s in concept.semantic_types)
        # Atom count is in the hundreds; just sanity-check it's > 1.
        assert (concept.atom_count or 0) > 10


def test_entity_linker_end_to_end_for_icd10() -> None:
    with EntityLinker() as linker:
        link = linker.resolve_icd10("L20.9")
        assert link.resolved
        assert link.concept is not None
        assert link.concept.cui == "C0011615"
        assert link.input_system == "ICD10CM"
        assert "ICD10CM" in link.sources
