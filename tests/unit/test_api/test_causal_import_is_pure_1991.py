"""Importing the causal routes package builds no external HTTP clients, and the
lazy clinical-context accessor caches its instance (#1991 debt 4).

Test 1 deletes ``src.api.routes.causal`` (and its submodules) from
``sys.modules`` and re-imports it under a patch that makes
``ClinicalContextService.__init__`` raise, to prove the import path never
constructs the service (whose constructor builds four HTTP clients: ChEMBL,
ClinicalTrials, PubMed, OpenFDA). Re-importing rebinds TWO things: the
``sys.modules`` entries for the deleted keys, AND the ``causal`` attribute
that the import machinery sets on the parent package module
(``src.api.routes``) — restoring only the former leaves the parent pointing
at the fresh (orphan) module while every existing importer of
``src.api.routes.causal`` (and the already-built FastAPI app) still holds the
old one. So both are snapshotted before the delete and put back in a
``finally``, and the test asserts the thing the ``sys.modules`` restore alone
cannot guarantee: that ``src.api.routes.causal`` (the attribute access other
code actually uses) is again the original module object.

Test 2 proves the module-level cache both builds the service through
``ClinicalContextService.__init__`` exactly once and then reuses that same
instance (identity across two accessor calls) — ``__init__`` is patched to a
no-op so no real HTTP clients are built.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

_PREFIX = "src.api.routes.causal"
_PARENT = "src.api.routes"


def test_import_does_not_construct_clinical_context_service():
    # Deliberately IN-PROCESS (sys.modules surgery), not a subprocess like the
    # isolation guard in tests/unit/test_scripts/test_seed_falkordb_import_
    # isolation_1761.py: measured here, a cold subprocess import of this
    # package takes ~12-17s (fastapi/pydantic/the whole dependency closure
    # cold) against a ~0.1-0.2s warm in-process re-import once the interpreter
    # already has those modules cached. A subprocess per test would turn this
    # into a slow test for no gain — the property under test (no HTTP client
    # construction) is fully observable via the patched __init__ without
    # paying for a second interpreter. Do not "simplify" this into a
    # subprocess call.
    saved = {
        k: sys.modules[k] for k in list(sys.modules) if k == _PREFIX or k.startswith(_PREFIX + ".")
    }
    for k in saved:
        del sys.modules[k]
    try:
        with patch(
            "src.services.clinical_context.service.ClinicalContextService.__init__",
            side_effect=AssertionError("built at import"),
        ):
            importlib.import_module(_PREFIX)
    finally:
        fresh = [k for k in list(sys.modules) if k == _PREFIX or k.startswith(_PREFIX + ".")]
        for k in fresh:
            del sys.modules[k]
        sys.modules.update(saved)
        # The import machinery also rebound the `causal` attribute on the
        # parent package module to the fresh module it just created; the
        # sys.modules restore above does not touch that attribute, so it must
        # be put back (or removed) explicitly.
        parent = sys.modules.get(_PARENT)
        if parent is not None:
            if _PREFIX in saved:
                parent.causal = saved[_PREFIX]
            elif hasattr(parent, "causal"):
                delattr(parent, "causal")

    if _PREFIX in saved:
        # The assertion that would have caught the parent-attribute leak: a
        # plain `sys.modules[_PREFIX] is saved[_PREFIX]` check is tautological
        # (it compares dict entries this same function just assigned via
        # `sys.modules.update(saved)`), but `parent.causal` is only correct if
        # the explicit restore above ran.
        assert sys.modules[_PARENT].causal is saved[_PREFIX]
    else:
        # The package had never been imported in this process before this test;
        # the restore removed the fresh entries, so it must be back to that state.
        assert _PREFIX not in sys.modules
        if _PARENT in sys.modules:
            assert not hasattr(sys.modules[_PARENT], "causal")


def test_accessor_builds_the_service_once_and_caches_it(monkeypatch):
    from src.api.routes.causal import catalog
    from src.services.clinical_context import ClinicalContextService

    monkeypatch.setattr(catalog, "_clinical_context_service", None)
    with patch.object(ClinicalContextService, "__init__", return_value=None) as mock_init:
        a = catalog._get_clinical_context_service()
        b = catalog._get_clinical_context_service()
    assert mock_init.call_count == 1
    assert a is b
    assert isinstance(a, ClinicalContextService)
