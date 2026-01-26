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
from typing import Any, Dict, Optional

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
# TESTING MODE - Set before any src imports to bypass JWT auth
# =============================================================================
os.environ["E2I_TESTING_MODE"] = "1"


def pytest_configure(config):
    """Run load_dotenv again at configure time for safety."""
    load_dotenv(override=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Service URLs from environment with defaults
# Port mapping (per docker/docker-compose.yml):
#   - e2i_redis (Working Memory): 6382
#   - e2i_falkordb (Semantic Memory): 6381
#   - opik-redis (Opik, separate): 6390
#   - auto-claude-falkordb (external): 6380
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")  # No password in docker/docker-compose.yml
_redis_url_env = os.getenv("REDIS_URL")
if _redis_url_env:
    REDIS_URL = _redis_url_env
elif REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@localhost:6382"
else:
    REDIS_URL = "redis://localhost:6382"
FALKORDB_URL = os.getenv("FALKORDB_URL", "redis://localhost:6381")
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

    try:
        # Check FalkorDB
        results["falkordb"] = loop.run_until_complete(_check_redis_service(FALKORDB_URL))
    except Exception as e:
        import sys
        print(f"  DEBUG: FalkorDB check failed: {e}", file=sys.stderr)
        results["falkordb"] = False
    finally:
        loop.close()

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
            status = "AVAILABLE" if available else "UNAVAILABLE"
            icon = "\u2713" if available else "\u2717"
            print(f"  {icon} {service.upper()}: {status}")
        print(f"  (checked in {check_duration:.2f}s)")
        print("=" * 60 + "\n")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
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
    if not SERVICES_AVAILABLE["falkordb"]:
        pytest.skip("FalkorDB not available")

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
