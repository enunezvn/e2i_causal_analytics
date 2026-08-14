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

from unittest.mock import AsyncMock

import pytest

# DEAD-SUPABASE PIN (#1420, incident 2026-07-31). The root conftest's
# load_dotenv(override=True) walks up from nested worktrees into the
# repo-root .env; on the droplet (PROD == DEV) that hands every unit test
# the REAL local-Supabase URL + key. Lazy live-writers — e.g.
# ``refute_causal_estimate`` building a real ``CausalValidationRepository``
# — then persist fixture output into prod tables: 1,760 test-artifact rows
# landed in ``causal_validations`` on 2026-07-31 alone, unmasked by the
# #1352 uuid-cast fix (before it, the writes failed the cast silently).
# Third recurrence of the live-writer family (#1371 MLflow HTTP, #1355
# agent_activities kill switch in the root conftest).
#
# CI is the faithful baseline: unit jobs run with a dead Supabase endpoint
# and fake keys (no Supabase service exists on the runner) and are green,
# so no unit test may depend on a live DB unless it opts in explicitly.
# The pin must be a per-test fixture, NOT an import-time os.environ write:
# the root conftest's ``pytest_configure`` re-runs load_dotenv(override=True)
# AFTER every conftest imports, clobbering import-time values. The sentinel
# is 127.0.0.1:1 (reserved port, never listening, immediate ECONNREFUSED)
# — NOT CI's literal localhost:54321, which is dead on runners but is the
# LIVE prod Supabase on the droplet.
#
# Escape hatches (both deliberate):
# * ``@pytest.mark.real_supabase`` — for the rare reachability-gated,
#   READ-ONLY faithful checks in the unit tree (e.g.
#   test_kpi_resolution.py::test_resolve_conversion_frame_real_supabase);
#   they keep the ambient env and their own skipif gates.
# * per-test ``monkeypatch.setenv`` — test-requested fixtures apply after
#   this autouse pin, so a test's own env values win.
# Integration trees are deliberately untouched (function scope cannot leak
# past the unit tree in mixed runs).
# Locked by tests/unit/test_utils/test_unit_tree_dead_supabase_1420.py.
_DEAD_SUPABASE_ENV = {
    "SUPABASE_URL": "http://127.0.0.1:1",
    "SUPABASE_KEY": "test-key",
    "SUPABASE_ANON_KEY": "test-key",
    "SUPABASE_SERVICE_KEY": "test-key",
    "SUPABASE_SERVICE_ROLE_KEY": "test-key",
}


@pytest.fixture(autouse=True)
def _pin_dead_supabase_env(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autouse: every unit test runs with dead Supabase credentials."""
    if request.node.get_closest_marker("real_supabase") is not None:
        return
    for var, value in _DEAD_SUPABASE_ENV.items():
        monkeypatch.setenv(var, value)


# REAL-MODE SYNTHETIC PIN (#1602, same contamination class as #1420). This
# droplet is a synthetic-gold showcase instance, so the repo-root .env sets
# E2I_INCLUDE_SYNTHETIC=true (line 100) and E2I_KPI_INCLUDE_SYNTHETIC=true
# (line 52). Both are CORRECT for the running services and must not be edited
# — the tests are what needs isolating.
#
# The root conftest's ``load_dotenv(override=True)`` + ``find_dotenv`` walk-up
# hands those flags to every local pytest process (nested worktrees have no
# ``.env`` of their own, so the walk reaches the repo root). Production then
# legitimately reports ``deployment_includes_synthetic()`` /
# ``kpi_include_synthetic()`` True, ``apply_provenance_filter`` skips its
# ``.eq('is_synthetic', False)`` link, and ``resolve_kpi_query_id`` swaps in
# the ``_include_synthetic`` RPC twin. Measured on main @ e4654355: 56 unit
# tests that pin the real-mode default fail on the droplet and pass in CI
# (no ``.env`` anywhere up the runner's checkout path) — 47 in
# tests/unit/test_repositories/ + 9 in tests/unit/test_kpi/. Most are direct
# ``*_excludes_synthetic`` predicate pins; the rest are collateral, because a
# skipped ``.eq`` link makes a default-exclude mock chain miss its AsyncMock
# and ``await`` land on a bare MagicMock.
#
# PR #1414 established the delenv precedent and #1495/#1497 extended it, but
# always per-module/per-class, so every new provenance-asserting test file
# re-introduced the failure. Pinning once at the unit-tree root closes the
# class instead of the instances. Those five local fixtures are deliberately
# LEFT in place: their docstrings carry the incident history at the point of
# use, and they keep the modules self-pinning if ever collected under a
# different conftest root.
#
# Must be a per-test fixture, not an import-time ``os.environ`` write, for the
# same reason as the dead-Supabase pin above: ``pytest_configure`` re-runs
# ``load_dotenv(override=True)`` AFTER every conftest imports.
#
# Nor can it be worked around by dropping an empty ``.env`` in the worktree:
# measured 2026-08-14, the flags STILL arrive. ``find_dotenv`` walks up from
# the CALLING frame, and at least one loader's frame lives in site-packages —
# ``.venv/`` sits at the repo root, so that walk reaches the real ``.env``
# regardless of what the worktree contains. (This is #1414's second vector.
# It is easy to mis-measure: wrapping ``load_dotenv`` to trace it inserts a
# worktree-rooted frame into the walk and masks the injection.) A per-test
# delenv is the only isolation that holds.
#
# Showcase-mode tests are unaffected: they re-set the flag with their own
# FUNCTION-scope ``monkeypatch.setenv`` (test body, or a function-scope
# fixture), which runs after this conftest-level autouse and therefore wins.
# A BROADER-scope autouse fixture (module/class/session) would lose instead —
# it runs before this function-scope pin, which then deletes its value. No
# unit module sets either flag at module scope today, and the lock test
# exercises exactly that ordering deliberately.
#
# ``@pytest.mark.real_supabase`` opts out, reusing the #1420 escape hatch: a
# reachability-gated READ-ONLY faithful check runs against the live deployment,
# so it must keep the WHOLE ambient env, not just the credentials. Measured:
# without this opt-out, test_kpi_resolution's faithful frame check fails 3/3
# ("no KPI frame resolved for Kisqali/northeast") — this droplet's substrate is
# entirely ``is_synthetic=true``, so pinning real mode makes the live query
# return zero rows. That is an environment artifact, not a defect.
#
# Scope is the unit tree ONLY. tests/integration/ holds deliberate live probes
# of this showcase deployment (test_chatbot_kpi_tool_live.py,
# test_577_action_rate_uplift_live.py) that must keep reading the ambient
# flags; the ones needing real mode already delenv per-test
# (test_rag_feedstock_realdb_1489.py).
# Locked by tests/unit/test_utils/test_unit_tree_real_mode_synthetic_1602.py.
_REAL_MODE_SYNTHETIC_FLAGS = (
    "E2I_INCLUDE_SYNTHETIC",
    "E2I_KPI_INCLUDE_SYNTHETIC",
)


@pytest.fixture(autouse=True)
def _pin_real_mode_synthetic_flags(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autouse: every unit test runs with the deployment synthetic switches OFF."""
    if request.node.get_closest_marker("real_supabase") is not None:
        return
    for var in _REAL_MODE_SYNTHETIC_FLAGS:
        monkeypatch.delenv(var, raising=False)


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


@pytest.fixture(autouse=True)
def _neutralize_drift_monitoring_client(request, monkeypatch):
    """Autouse (#845): the drift / monitoring / retraining call sites now resolve
    a real async Supabase client via
    ``src.repositories.drift_monitoring.get_drift_monitoring_client`` and wire it
    into the repositories (replacing the old client-less, silently-no-op'ing
    construction). Unit tests must stay hermetic: CI's unit job sets a
    ``SUPABASE_URL`` but has no reachable Supabase and no usable service key
    (``_resolve_supabase_key`` ignores ``SUPABASE_KEY``), so a live resolve raises
    ``ServiceConnectionError`` -> wired endpoints/tasks would 5xx/fail where they
    previously read empty.

    Stub the resolver to return ``None`` so every call site falls back to the
    EXACT pre-#845 client-less no-op (mocked-repo tests ignore the value; tests
    that relied on the no-op still get empty results). The real wiring and
    fail-closed behavior are covered by
    ``tests/unit/test_repositories/test_drift_monitoring_client_wiring_845.py``,
    which opts OUT below (it must exercise the real resolver)."""
    if "test_drift_monitoring_client_wiring_845" in request.node.nodeid:
        return
    try:
        import src.repositories.drift_monitoring as _dm
    except Exception:  # pragma: no cover - module always importable in unit env
        return
    monkeypatch.setattr(
        _dm, "get_drift_monitoring_client", AsyncMock(return_value=None), raising=False
    )
