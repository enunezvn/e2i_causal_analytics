"""Shared fixtures for the standalone BentoML serving tests.

Why this conftest exists:
    ``scripts/bentoml/e2i_serving_service.py`` is a self-contained service
    deliberately isolated from ``src.*`` and meant to run inside the BentoML
    container. The host venv intentionally does NOT install ``bentoml``, so
    importing the module raises ``ModuleNotFoundError`` unless we stub the
    decorator surface it touches. We stub the minimum needed for the module
    to import: ``bentoml.service``, ``bentoml.api``, ``bentoml.exceptions``,
    and a ``bentoml.models.list`` that returns no models (forcing the
    "graceful no-model" code path).

This is mocking an EXTERNAL boundary (the bentoml SDK), not business logic,
so it complies with the Tier-0 "no mocks of business logic" constraint.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import pytest


def _install_bentoml_stub() -> None:
    """Install a minimal bentoml stub into sys.modules.

    The stub provides only what scripts/bentoml/e2i_serving_service.py touches
    at import time and during runtime in the test paths exercised here.
    """
    if "bentoml" in sys.modules and getattr(sys.modules["bentoml"], "_e2i_test_stub", False):
        return  # Already installed

    bentoml_module = types.ModuleType("bentoml")
    bentoml_module._e2i_test_stub = True  # type: ignore[attr-defined]

    # @bentoml.service(...) → returns identity decorator
    def _service_decorator(*_args: Any, **_kwargs: Any) -> Any:
        def _wrap(cls: Any) -> Any:
            return cls

        return _wrap

    # @bentoml.api → identity decorator
    def _api_decorator(func: Any) -> Any:
        return func

    bentoml_module.service = _service_decorator  # type: ignore[attr-defined]
    bentoml_module.api = _api_decorator  # type: ignore[attr-defined]

    # bentoml.exceptions.NotFound — touched in framework loader try/except
    exceptions_module = types.ModuleType("bentoml.exceptions")

    class _NotFound(Exception):
        pass

    exceptions_module.NotFound = _NotFound  # type: ignore[attr-defined]
    bentoml_module.exceptions = exceptions_module  # type: ignore[attr-defined]

    # bentoml.models.list() / .get() — return empty list / raise to force degraded mode
    models_module = types.ModuleType("bentoml.models")

    def _empty_list() -> list[Any]:
        return []

    def _get_raises(*_a: Any, **_k: Any) -> Any:
        raise _NotFound("test stub: no models")

    models_module.list = _empty_list  # type: ignore[attr-defined]
    models_module.get = _get_raises  # type: ignore[attr-defined]
    bentoml_module.models = models_module  # type: ignore[attr-defined]

    sys.modules["bentoml"] = bentoml_module
    sys.modules["bentoml.exceptions"] = exceptions_module
    sys.modules["bentoml.models"] = models_module


@pytest.fixture(scope="session")
def serving_module() -> Any:
    """Import and return the e2i_serving_service module under a stubbed bentoml.

    Session-scoped: the import is expensive and the module's globals are
    safe to share across tests because each test that mutates state creates
    its own ``E2IModelService`` instance.
    """
    _install_bentoml_stub()

    # Load the module from its file path so we don't pollute sys.path globally.
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "bentoml" / "e2i_serving_service.py"
    spec = importlib.util.spec_from_file_location("e2i_serving_service", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["e2i_serving_service"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _reset_feast_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure each test starts with a clean Feast endpoint env."""
    monkeypatch.delenv("FEAST_HTTP_ENDPOINT", raising=False)
    monkeypatch.delenv("FEAST_URL", raising=False)
    yield
