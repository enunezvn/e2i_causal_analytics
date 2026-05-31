"""Unit-test conftest — silences real Opik/MLflow emissions during the
unit-test sweep.

Background
----------
The #391 monitoring-slice instrumentation in
:mod:`src.mlops.lifecycle_monitoring` is wired into the production
``cascade_invalidate`` / ``Consolidator.run`` / ``_crystallize_group``
paths. Existing unit tests for those modules (e.g.
``test_invalidator.py``) exercise those paths end-to-end with a fake
Supabase / fake Redis — but they also reach the REAL lifecycle-
monitoring helpers, which submit MLflow runs to whatever tracking
server is reachable (``http://localhost:5000`` by default).

This conftest disables that side-effect at unit-test boundary by
flipping the module-level availability sentinels to False, BEFORE the
producer modules import. Tests that need to exercise the monitoring
boundary (``tests/unit/test_observability/``) override at per-test
scope via fixture-injected fake recorders.

Without this conftest, every ``cascade_invalidate`` call in the unit
suite would create one or more MLflow runs on the operator's local
tracking server — polluting the experiment store and slowing the
unit sweep with network I/O.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _silence_lifecycle_monitoring_in_unit_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autouse: every unit test runs with Opik + MLflow emission
    disabled at the lifecycle-monitoring boundary.

    Tests in ``tests/unit/test_observability/`` override the
    boundary functions (``_emit_opik_trace`` / ``_emit_mlflow_metric``)
    directly to capture emissions — they don't need (and don't want)
    the disable behavior, but the override happens AFTER this
    autouse so the function-replacement wins.
    """
    try:
        from src.mlops import lifecycle_monitoring as lm
    except ImportError:
        # Module unavailable in some test environments — silent skip.
        return
    monkeypatch.setattr(lm, "_MLFLOW_AVAILABLE", False, raising=False)
    monkeypatch.setattr(lm, "_OPIK_AVAILABLE", False, raising=False)


@pytest.fixture(autouse=True)
def _reset_service_client_singletons() -> None:
    """Autouse: reset ``src.memory.services.factories`` singletons before each test.

    The factory memoises clients in module globals (``_redis_client``,
    ``_supabase_client``, ``_falkordb_client`` …) plus an ``@lru_cache`` for
    production reuse. Those globals leak across tests: a test that initialises or
    mocks a client (e.g. ``test_service_factories`` / ``test_embedding_fallback``,
    which set them on purpose) pollutes a later test expecting a clean slate.
    Under ``-n 2 --dist=loadscope`` the worker distribution decides ordering, so
    widening the unit allowlist (#555) surfaced this as ``ServiceConnectionError``
    / ``MagicMock can't be used in 'await'`` in ``test_memory`` / ``test_api``
    even though each dir passes in isolation.

    ``reset_all_clients()`` is the maintained reset hook; running it before every
    test makes the singletons order-independent. Locked by
    ``test_memory/test_service_singleton_isolation.py``.
    """
    try:
        from src.memory.services.factories import reset_all_clients
    except ImportError:
        return
    reset_all_clients()
