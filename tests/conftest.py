"""Root conftest.py - Global pytest fixtures and service availability management.

This module provides:
1. Service availability detection at session start
2. Auto-skip fixtures for external services (Redis, FalkorDB, Supabase)
3. Safe async fixtures with built-in timeouts
4. pytest hooks for service-based test filtering
5. Memory-aware test grouping for heavy ML imports

Usage:
    # In test files, use the fixtures:
    @pytest.mark.requires_redis
    async def test_redis_operation(redis_client):
        await redis_client.ping()

    # Or use skip markers directly:
    @pytest.mark.skipif(not SERVICES_AVAILABLE["redis"], reason="Redis not available")
    def test_something():
        ...

    # For memory-heavy tests (dspy, econml, etc.):
    @pytest.mark.xdist_group(name="dspy_integration")
    def test_heavy_ml_operation():
        import dspy  # Heavy import grouped on single worker
        ...

Memory Management:
    - Default: 4 parallel workers (safe for 7.5GB RAM systems)
    - xdist_group markers ensure heavy imports share workers
    - Use `pytest -n 0` for sequential runs on low-memory systems
"""

from __future__ import annotations

import asyncio
import os
import time
import warnings
from typing import Dict, Optional

import pytest
import pytest_asyncio
from dotenv import load_dotenv

# =============================================================================
# LOAD ENVIRONMENT VARIABLES from .env file IMMEDIATELY
# =============================================================================
# Load .env at module import time to ensure API keys are available before
# any test files are collected. Use override=True so real .env values win
# over any placeholder test keys that may have been set earlier.
load_dotenv(override=True)

# =============================================================================
# ASYNCIO POLLUTION PROBE (issue #218 — follow-up to #215)
# =============================================================================
# ``nest_asyncio.apply()`` is a PROCESS-WIDE monkey-patch of ``asyncio.run``.
# Once any test (directly or via a dependency such as DSPy's ``syncify`` or
# mlflow.genai's ``_make_sync_wrapper``) triggers ``apply()`` on an xdist
# worker, every subsequent ``asyncio.run(coro)`` on that worker routes through
# nest_asyncio's patched runner — which holds a reference to the *original*
# event loop. If that loop has since been closed (e.g., by pytest-asyncio
# tearing down a per-test loop), the next sync ``asyncio.run`` call raises
# ``RuntimeError: Event loop is closed``.
#
# PR #217 closed one polluter (``experiment_designer.graph``'s eager singleton)
# but left the broader bug class open: any of the 9 gated callsites in ``src/``
# or the 2 known third-party callers (``dspy/utils/syncify.py:20`` and
# ``mlflow/genai/utils/trace_utils.py:395``) can pollute if invoked from
# inside a running loop.
#
# This probe instruments ``nest_asyncio.apply`` at session start so we can:
#   1. Record the FIRST ``apply()`` call's full stack trace (the polluter).
#   2. Detect the test boundary AT WHICH ``asyncio.run`` first becomes
#      monkey-patched, and surface BOTH the test nodeid and the call stack
#      in a session-summary line + a pytest warning.
#
# Default behaviour is *observational* — we do not fail the session, because
# the existing victim-site mitigations (PR #217 commit ``a321b64f``,
# ``test_layer_5_pipeline_integration.py``, etc.) keep the suite green. Set
# ``E2I_ASSERT_NO_ASYNCIO_POLLUTION=1`` in CI to promote a detected polluter
# into a hard ``pytest.exit`` so the offending test is named on a red CI run.
_ORIG_ASYNCIO_RUN = asyncio.run
_ASYNCIO_POLLUTION_STATE: Dict[str, object] = {
    "apply_first_stack": None,  # type: Optional[str]
    "apply_first_nodeid": None,  # type: Optional[str]
    "polluter_nodeid": None,  # type: Optional[str]
    "current_nodeid": None,  # type: Optional[str]
    "apply_count": 0,
    "installed_trace": False,
}


def _install_nest_asyncio_apply_trace() -> None:
    """Wrap ``nest_asyncio.apply`` so we record the first-call stack trace.

    Idempotent: re-installing on top of itself is a no-op. The wrapper still
    delegates to the real ``apply`` (we instrument, not block) so PR #217's
    lazy-apply pattern continues to work for legitimate nested-loop cases.
    """
    if _ASYNCIO_POLLUTION_STATE["installed_trace"]:
        return
    try:
        import nest_asyncio  # type: ignore[import-not-found]
    except ImportError:
        # nest_asyncio is a project dep; absence means the suite cannot be
        # polluted via it. Nothing to instrument.
        _ASYNCIO_POLLUTION_STATE["installed_trace"] = True
        return

    _real_apply = nest_asyncio.apply

    def _traced_apply(*args, **kwargs):  # type: ignore[no-untyped-def]
        import traceback

        # Increment for every apply() call so a follow-up regression test
        # can pin "apply_count==0" on clean sessions.
        _ASYNCIO_POLLUTION_STATE["apply_count"] = (
            int(_ASYNCIO_POLLUTION_STATE["apply_count"]) + 1
        )
        if _ASYNCIO_POLLUTION_STATE["apply_first_stack"] is None:
            _ASYNCIO_POLLUTION_STATE["apply_first_stack"] = "".join(
                traceback.format_stack()
            )
            _ASYNCIO_POLLUTION_STATE["apply_first_nodeid"] = (
                _ASYNCIO_POLLUTION_STATE["current_nodeid"]
            )
        return _real_apply(*args, **kwargs)

    nest_asyncio.apply = _traced_apply  # type: ignore[assignment]
    _ASYNCIO_POLLUTION_STATE["installed_trace"] = True


# Install ASAP so we catch apply() calls fired during fixture setup, not just
# inside the test body itself.
_install_nest_asyncio_apply_trace()

# =============================================================================
# TESTING MODE - Set before any src imports to bypass JWT auth
# =============================================================================
os.environ["E2I_TESTING_MODE"] = "1"


def pytest_configure(config):
    """Run load_dotenv again at configure time for safety."""
    load_dotenv(override=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Service URLs from environment with defaults.
#
# Port mapping (per docker/docker-compose.yml):
#   - e2i_redis (Working Memory): 6382
#   - e2i_falkordb (Semantic Memory): 6381
#   - opik-redis (Opik, separate): 6390
#   - auto-claude-falkordb (external): 6380
#
# CI note (issue #183): GitHub Actions integration-tests/unit-tests/slow-tests
# jobs do NOT stand up a FalkorDB service container. Historically those
# workflows still exported ``FALKORDB_URL='redis://localhost:6380'``, which
# caused ``_check_redis_service`` to log a noisy ECONNREFUSED on every CI run
# and confuse the failure picture when an xdist worker died for unrelated
# reasons (see PR #181's run 25809588036 where a transient gw0 message was
# misread as a FalkorDB outage). The workflow YAML has been updated to drop
# FALKORDB_URL when no FalkorDB service is declared; the conftest now opts
# the probe out cleanly when ``FALKORDB_REQUIRED`` is not set, so the session
# header reads ``FALKORDB: UNCONFIGURED (expected in this lane)`` instead of
# producing an ECONNREFUSED traceback.
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")  # No password in docker/docker-compose.yml
_redis_url_env = os.getenv("REDIS_URL")
if _redis_url_env:
    REDIS_URL = _redis_url_env
elif REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@localhost:6382"
else:
    REDIS_URL = "redis://localhost:6382"

# FALKORDB_URL is optional. When it is unset AND the lane has not opted into
# FALKORDB_REQUIRED=1, the service-check probe is skipped (returns False)
# rather than attempting a TCP connect that we know cannot succeed. This
# preserves the existing skip semantics for ``requires_falkordb`` tests while
# stopping the misleading ECONNREFUSED log line.
FALKORDB_URL = os.getenv("FALKORDB_URL", "")
FALKORDB_REQUIRED = os.getenv("FALKORDB_REQUIRED", "").lower() in {"1", "true", "yes"}
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "") or os.getenv("SUPABASE_SERVICE_KEY", "")

# Connection timeout for service checks (seconds)
SERVICE_CHECK_TIMEOUT = 3.0

# Global service availability cache (populated at session start)
SERVICES_AVAILABLE: Dict[str, bool] = {
    "redis": False,
    "falkordb": False,
    "supabase": False,
}


# =============================================================================
# SERVICE AVAILABILITY CHECKING
# =============================================================================


async def _check_redis_service(url: str, timeout: float = SERVICE_CHECK_TIMEOUT) -> bool:
    """Check if a Redis-compatible service is available.

    Args:
        url: Redis connection URL
        timeout: Connection timeout in seconds

    Returns:
        True if service is reachable and responds to PING
    """
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            url,
            socket_timeout=timeout,
            socket_connect_timeout=timeout,
        )
        try:
            await asyncio.wait_for(client.ping(), timeout=timeout)
            return True
        finally:
            await client.aclose()
    except Exception as e:
        # Debug: print exception to help diagnose connectivity issues
        import sys

        print(f"  DEBUG _check_redis_service({url}): {type(e).__name__}: {e}", file=sys.stderr)
        return False


def _check_supabase_service() -> bool:
    """Check if Supabase credentials are configured.

    Note: We only check for credentials, not actual connectivity,
    because Supabase is a remote service and connectivity checks
    would add latency to every test run.

    Returns:
        True if SUPABASE_URL and key are configured
    """
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _run_service_checks() -> Dict[str, bool]:
    """Run all service availability checks.

    Returns:
        Dictionary mapping service names to availability status
    """
    results = {}

    # Use a fresh event loop to avoid conflicts with pytest-asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Check Redis
        import sys as _debug_sys

        print(f"  DEBUG _run_service_checks: REDIS_URL={REDIS_URL}", file=_debug_sys.stderr)
        results["redis"] = loop.run_until_complete(_check_redis_service(REDIS_URL))
    except Exception as e:
        import sys

        print(f"  DEBUG: Redis check failed: {e}", file=sys.stderr)
        results["redis"] = False

    # FalkorDB check is skipped cleanly when no URL is configured (issue #183).
    # The probe ONLY fires when a non-empty ``FALKORDB_URL`` is exported
    # (local-droplet runs, where docker-compose auto-injects the env var
    # pointing at :6381). The independent ``FALKORDB_REQUIRED`` env var does
    # NOT itself trigger a probe — it only surfaces a warning when the URL is
    # empty so future FalkorDB-backed CI lanes can't silently mis-configure
    # the env without log evidence (codex pass-1 LOW-1 wording clarification).
    if FALKORDB_URL:
        try:
            results["falkordb"] = loop.run_until_complete(_check_redis_service(FALKORDB_URL))
        except Exception as e:
            import sys

            print(f"  DEBUG: FalkorDB check failed: {e}", file=sys.stderr)
            results["falkordb"] = False
        finally:
            loop.close()
    else:
        # No URL configured. Treat as unconfigured (skip-equivalent) instead of
        # attempting a TCP connect that we know will fail with ECONNREFUSED.
        results["falkordb"] = False
        loop.close()
        if FALKORDB_REQUIRED:
            import sys

            print(
                "  WARN: FALKORDB_REQUIRED=1 set but FALKORDB_URL is empty; "
                "FalkorDB-marked tests will skip.",
                file=sys.stderr,
            )

    # Check Supabase (credentials only)
    results["supabase"] = _check_supabase_service()

    return results


# =============================================================================
# PYTEST HOOKS
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with service availability information.

    This runs once at the start of the test session to:
    1. Disable rate limiting for tests
    2. Check which services are available
    3. Store results for skip decision making
    4. Print service status to console
    """
    global SERVICES_AVAILABLE

    # Disable rate limiting for tests to prevent 429 errors from state accumulation
    os.environ["DISABLE_RATE_LIMITING"] = "1"

    # Run service checks
    start_time = time.time()
    SERVICES_AVAILABLE = _run_service_checks()
    check_duration = time.time() - start_time

    # Store in config for access by other hooks
    config._service_availability = SERVICES_AVAILABLE

    # Print service status (only if not in quiet mode)
    quiet = getattr(config.option, "quiet", 0) or getattr(config.option, "q", 0)
    if not quiet:
        print("\n" + "=" * 60)
        print("SERVICE AVAILABILITY CHECK")
        print("=" * 60)
        for service, available in SERVICES_AVAILABLE.items():
            if service == "falkordb" and not FALKORDB_URL and not available:
                # Distinguish "intentionally unconfigured" from "configured but
                # unreachable" so a session header in CI reads
                # ``FALKORDB: UNCONFIGURED`` rather than implying a service
                # outage (issue #183).
                status = "UNCONFIGURED (skipping FalkorDB-marked tests)"
                icon = "\u2013"  # en dash
            elif available:
                status = "AVAILABLE"
                icon = "\u2713"
            else:
                status = "UNAVAILABLE"
                icon = "\u2717"
            print(f"  {icon} {service.upper()}: {status}")
        print(f"  (checked in {check_duration:.2f}s)")
        print("=" * 60 + "\n")


# =============================================================================
# ASYNCIO POLLUTION HOOKS (issue #218)
# =============================================================================


def pytest_runtest_protocol(item, nextitem):  # type: ignore[no-untyped-def]
    """Record which test is currently executing so the apply() trace can
    attribute the first ``nest_asyncio.apply()`` call to a specific nodeid."""

    _ASYNCIO_POLLUTION_STATE["current_nodeid"] = item.nodeid
    # Don't return a value — we only observe; pytest continues the protocol.
    return None


def pytest_runtest_logfinish(nodeid, location):  # type: ignore[no-untyped-def]
    """Detect the first test boundary at which ``asyncio.run`` is polluted.

    Fires after every test (passed/failed/skipped/xfailed). Once we observe
    ``asyncio.run is not _ORIG_ASYNCIO_RUN``, we record the polluter nodeid
    and emit a pytest warning so the session summary shows the culprit.
    """

    if _ASYNCIO_POLLUTION_STATE["polluter_nodeid"] is not None:
        # Already recorded — pollution is a one-shot process state, no
        # follow-up signal needed.
        return

    if asyncio.run is _ORIG_ASYNCIO_RUN:
        return

    _ASYNCIO_POLLUTION_STATE["polluter_nodeid"] = nodeid
    warnings.warn(
        (
            f"[issue-218] asyncio.run was monkey-patched by nest_asyncio.apply() "
            f"during test: {nodeid}. First apply() call attributed to: "
            f"{_ASYNCIO_POLLUTION_STATE.get('apply_first_nodeid')!r}. "
            f"Set E2I_ASSERT_NO_ASYNCIO_POLLUTION=1 to promote this to a hard "
            f"failure; inspect ``_ASYNCIO_POLLUTION_STATE['apply_first_stack']`` "
            f"in a debugger or via the session-summary log line for the call site."
        ),
        category=RuntimeWarning,
        stacklevel=1,
    )

    if os.getenv("E2I_ASSERT_NO_ASYNCIO_POLLUTION", "").lower() in {"1", "true", "yes"}:
        # Hard-fail when explicitly requested. We use pytest.exit (not raise)
        # so the surrounding test isn't blamed for the pollution — the
        # session summary will name the actual culprit via ``polluter_nodeid``.
        stack = _ASYNCIO_POLLUTION_STATE.get("apply_first_stack") or "<not captured>"
        pytest.exit(
            (
                f"[issue-218] FATAL: nest_asyncio.apply() polluted asyncio.run "
                f"during test {nodeid!r}. First apply() called from: "
                f"{_ASYNCIO_POLLUTION_STATE.get('apply_first_nodeid')!r}.\n"
                f"--- apply() call stack ---\n{stack}\n--- end stack ---"
            ),
            returncode=2,
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # type: ignore[no-untyped-def]
    """Append an issue-218 diagnostic block to the terminal summary when
    pollution was detected during the session."""

    if _ASYNCIO_POLLUTION_STATE["polluter_nodeid"] is None:
        return
    terminalreporter.write_sep("=", "issue-218 asyncio pollution diagnostic")
    terminalreporter.write_line(
        f"polluter_nodeid (first test after pollution detected): "
        f"{_ASYNCIO_POLLUTION_STATE['polluter_nodeid']!r}"
    )
    terminalreporter.write_line(
        f"apply_first_nodeid (running when first apply() fired): "
        f"{_ASYNCIO_POLLUTION_STATE['apply_first_nodeid']!r}"
    )
    terminalreporter.write_line(
        f"nest_asyncio.apply() call count: "
        f"{_ASYNCIO_POLLUTION_STATE['apply_count']}"
    )
    stack = _ASYNCIO_POLLUTION_STATE.get("apply_first_stack")
    if stack:
        terminalreporter.write_line("--- first apply() call stack ---")
        for line in str(stack).splitlines():
            terminalreporter.write_line(line)
        terminalreporter.write_line("--- end stack ---")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify test collection to skip tests based on service availability.

    This automatically skips tests marked with requires_* markers
    when the corresponding service is not available.
    """
    services = getattr(config, "_service_availability", SERVICES_AVAILABLE)

    skip_markers = {
        "requires_redis": ("redis", "Redis not available"),
        "requires_falkordb": ("falkordb", "FalkorDB not available"),
        "requires_supabase": ("supabase", "Supabase not configured"),
    }

    for item in items:
        for marker_name, (service, reason) in skip_markers.items():
            if marker_name in [m.name for m in item.iter_markers()]:
                if not services.get(service, False):
                    item.add_marker(pytest.mark.skip(reason=reason))


# =============================================================================
# SERVICE AVAILABILITY FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def service_availability() -> Dict[str, bool]:
    """Get service availability status.

    Returns:
        Dictionary mapping service names to availability boolean
    """
    return SERVICES_AVAILABLE.copy()


@pytest.fixture
def skip_without_redis():
    """Skip test if Redis is not available."""
    if not SERVICES_AVAILABLE["redis"]:
        pytest.skip("Redis not available")


@pytest.fixture
def skip_without_falkordb():
    """Skip test if FalkorDB is not available."""
    if not SERVICES_AVAILABLE["falkordb"]:
        pytest.skip("FalkorDB not available")


@pytest.fixture
def skip_without_supabase():
    """Skip test if Supabase is not configured."""
    if not SERVICES_AVAILABLE["supabase"]:
        pytest.skip("Supabase not configured")


# =============================================================================
# SAFE ASYNC CLIENT FIXTURES
# =============================================================================


@pytest_asyncio.fixture
async def redis_client():
    """Create a Redis client with automatic skip if unavailable.

    This fixture:
    1. Skips if Redis is not available
    2. Creates client with connection timeout
    3. Verifies connection with PING
    4. Cleans up after test

    Yields:
        redis.asyncio.Redis: Connected Redis client
    """
    if not SERVICES_AVAILABLE["redis"]:
        pytest.skip("Redis not available")

    import redis.asyncio as aioredis

    client = aioredis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_timeout=SERVICE_CHECK_TIMEOUT,
        socket_connect_timeout=SERVICE_CHECK_TIMEOUT,
    )

    try:
        # Verify connection
        await asyncio.wait_for(client.ping(), timeout=SERVICE_CHECK_TIMEOUT)
        yield client
    except asyncio.TimeoutError:
        pytest.skip(f"Redis connection timeout ({REDIS_URL})")
    except Exception as e:
        pytest.skip(f"Redis connection failed: {e}")
    finally:
        await client.aclose()


def _enforce_falkordb_preconditions() -> None:
    """Shared skip-or-proceed guard for the ``falkordb_client`` fixture.

    Extracted as a top-level helper so unit tests can exercise it without
    pulling the full pytest_asyncio fixture machinery (codex pass-2 LOW-2).

    Raises:
        pytest.skip.Exception: when either the global availability flag is
            False (no probe success) or ``FALKORDB_URL`` is empty (issue
            #183 defensive guard against ``aioredis.from_url("")``).
    """
    if not SERVICES_AVAILABLE["falkordb"]:
        pytest.skip("FalkorDB not available")
    if not FALKORDB_URL:
        pytest.skip("FalkorDB URL not configured")


@pytest_asyncio.fixture
async def falkordb_client():
    """Create a FalkorDB client with automatic skip if unavailable.

    This fixture:
    1. Skips if FalkorDB is not available
    2. Creates client with connection timeout
    3. Verifies connection with PING
    4. Cleans up after test

    Yields:
        redis.asyncio.Redis: Connected FalkorDB client (Redis protocol)
    """
    _enforce_falkordb_preconditions()

    import redis.asyncio as aioredis

    client = aioredis.from_url(
        FALKORDB_URL,
        decode_responses=True,
        socket_timeout=SERVICE_CHECK_TIMEOUT,
        socket_connect_timeout=SERVICE_CHECK_TIMEOUT,
    )

    try:
        # Verify connection
        await asyncio.wait_for(client.ping(), timeout=SERVICE_CHECK_TIMEOUT)
        yield client
    except asyncio.TimeoutError:
        pytest.skip(f"FalkorDB connection timeout ({FALKORDB_URL})")
    except Exception as e:
        pytest.skip(f"FalkorDB connection failed: {e}")
    finally:
        await client.aclose()


@pytest.fixture
def supabase_client():
    """Create a Supabase client with automatic skip if not configured.

    This fixture:
    1. Skips if Supabase credentials are not set
    2. Creates sync Supabase client

    Returns:
        supabase.Client: Connected Supabase client
    """
    if not SERVICES_AVAILABLE["supabase"]:
        pytest.skip("Supabase not configured")

    from supabase import create_client

    return create_client(SUPABASE_URL, SUPABASE_KEY)


# =============================================================================
# ASYNC UTILITIES
# =============================================================================


@pytest.fixture
def async_timeout():
    """Provide an async timeout wrapper for use in tests.

    Usage:
        async def test_something(async_timeout):
            result = await async_timeout(some_async_func(), timeout=5.0)

    Returns:
        Callable that wraps coroutines with asyncio.wait_for
    """

    async def _timeout_wrapper(coro, timeout: float = 5.0):
        return await asyncio.wait_for(coro, timeout=timeout)

    return _timeout_wrapper


# =============================================================================
# TEST ISOLATION FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment state before each test.

    This ensures tests don't leak state through environment variables.
    """
    # Store original environment
    original_env = os.environ.copy()

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================


@pytest.fixture
def timer():
    """Provide a simple timer for performance measurement.

    Usage:
        def test_performance(timer):
            timer.start()
            do_something()
            elapsed = timer.stop()
            assert elapsed < 1.0, "Too slow!"
    """

    class Timer:
        def __init__(self):
            self._start: Optional[float] = None
            self._elapsed: Optional[float] = None

        def start(self) -> None:
            self._start = time.perf_counter()
            self._elapsed = None

        def stop(self) -> float:
            if self._start is None:
                raise RuntimeError("Timer not started")
            self._elapsed = time.perf_counter() - self._start
            return self._elapsed

        @property
        def elapsed(self) -> Optional[float]:
            return self._elapsed

    return Timer()


# ---------------------------------------------------------------------------
# D1.3 — audit_workflow_id fixture for ml_foundation agent tests
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_workflow_id():
    """Workflow-scoped UUID for threading through ml_foundation agent
    ``input_data`` dicts in tests.

    Tests that call ``agent.run({...})`` should include
    ``"audit_workflow_id": audit_workflow_id`` in input_data so the State
    receives a caller-provided UUID rather than relying on the
    ``Field(default_factory=uuid4)`` transition shim. After sub-shard
    D1.4 drops the default_factory, this fixture becomes load-bearing
    for any test that constructs initial agent state.

    Example::

        @pytest.mark.asyncio
        async def test_thing(audit_workflow_id):
            input_data = {
                "audit_workflow_id": audit_workflow_id,
                ...
            }
            result = await agent.run(input_data)
    """
    from uuid import uuid4

    return uuid4()
