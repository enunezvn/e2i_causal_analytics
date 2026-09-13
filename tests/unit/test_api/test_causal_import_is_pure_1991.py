"""Importing the causal routes package builds no external HTTP clients, and the
lazy clinical-context accessor caches its instance (#1991 debt 4).

Test 1 deletes ``src.api.routes.causal`` (and its submodules) from
``sys.modules`` and re-imports it under a patch that makes
``ClinicalContextService.__init__`` raise, to prove the import path never
constructs the service (whose constructor builds four HTTP clients: ChEMBL,
ClinicalTrials, PubMed, OpenFDA). Re-importing creates NEW module objects,
which would silently orphan every OTHER module in this pytest process still
holding the OLD ones (sibling test modules, and the already-built FastAPI
app) — so ``sys.modules`` is snapshotted before the delete and restored in a
``finally``, and the test asserts the restore put back the exact objects that
were there before it ran.

Test 2 proves the module-level cache actually caches (identity across two
accessor calls) WITHOUT constructing the real service — ``__init__`` is
patched to a no-op, so the accessor builds a bare instance and only identity
is checked.

Test 3 gives test 1 teeth: it proves the accessor's construction really does
route through ``ClinicalContextService.__init__`` (called exactly once across
two accessor calls) under the same kind of patch, so a regression that made
import eager would surface as test 1's ``AssertionError`` instead of a
vacuously-passing test.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

_PREFIX = "src.api.routes.causal"


def test_import_does_not_construct_clinical_context_service():
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

    if _PREFIX in saved:
        assert sys.modules[_PREFIX] is saved[_PREFIX]
        assert sys.modules[f"{_PREFIX}.catalog"] is saved[f"{_PREFIX}.catalog"]
    else:
        # The package had never been imported in this process before this test;
        # the restore removed the fresh entries, so it must be back to that state.
        assert _PREFIX not in sys.modules


def test_accessor_builds_once(monkeypatch):
    from src.api.routes.causal import catalog
    from src.services.clinical_context import ClinicalContextService

    monkeypatch.setattr(catalog, "_clinical_context_service", None)
    with patch.object(ClinicalContextService, "__init__", return_value=None):
        a = catalog._get_clinical_context_service()
        b = catalog._get_clinical_context_service()
    assert a is b
    assert isinstance(a, ClinicalContextService)


def test_accessor_constructs_the_service_exactly_once_on_first_use(monkeypatch):
    from src.api.routes.causal import catalog
    from src.services.clinical_context import ClinicalContextService

    monkeypatch.setattr(catalog, "_clinical_context_service", None)
    with patch.object(ClinicalContextService, "__init__", return_value=None) as mock_init:
        catalog._get_clinical_context_service()
        catalog._get_clinical_context_service()
    assert mock_init.call_count == 1
