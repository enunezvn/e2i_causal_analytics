"""Cross-test isolation guard for the service-client singletons (issue #555).

``src.memory.services.factories`` memoises clients in module-level globals
(``_supabase_client``, ``_redis_client``, ``_falkordb_client`` …) for production
reuse. Without a reset between tests those globals leak: a test that sets one
(e.g. ``test_service_factories`` / ``test_embedding_fallback``) pollutes a later
test that expected a clean slate. Under ``-n 2 --dist=loadscope`` the worker
distribution decides ordering, so widening the unit allowlist (#555) surfaced
this as ``ServiceConnectionError`` / ``MagicMock can't be used in 'await'`` in
``test_memory`` / ``test_api``.

These two ordered tests lock in the autouse reset fixture
(``tests/unit/conftest.py::_reset_service_client_singletons``): the first
pollutes a singleton, the second must observe a clean slate. Remove the autouse
reset and the second test fails — exactly the regression we are preventing.
"""

from __future__ import annotations

import src.memory.services.factories as factories

_SENTINEL = object()


class TestServiceSingletonIsolation:
    def test_a_pollutes_a_singleton(self) -> None:
        factories._supabase_client = _SENTINEL
        assert factories._supabase_client is _SENTINEL

    def test_b_sees_a_clean_slate(self) -> None:
        # The autouse reset must have cleared test_a's pollution before this test.
        assert factories._supabase_client is None
