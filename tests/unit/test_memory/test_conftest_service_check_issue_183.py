"""Regression tests for issue #183 — FalkorDB service-check probe behavior.

Pins the contract from PR fixing issue #183:

1. When ``FALKORDB_URL`` is unset (empty string), ``_run_service_checks``
   MUST NOT attempt a TCP connection. It returns ``falkordb=False`` and
   skips the probe entirely. Rationale: CI lanes that have no FalkorDB
   service container historically still exported
   ``FALKORDB_URL='redis://localhost:6380'`` which produced a noisy
   ECONNREFUSED on every CI run and confused the failure picture when an
   xdist worker died for unrelated reasons (issue #183).

2. When ``FALKORDB_URL`` is set, the probe MUST still fire (preserves
   local-droplet semantics where docker-compose auto-injects the env var
   pointing at :6381).

3. The ``falkordb_client`` fixture defensively skips with a clean message
   when ``FALKORDB_URL`` is empty even if the global availability flag
   somehow indicates True. This guards against the prior crash mode where
   ``aioredis.from_url("")`` would have raised inside the fixture.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest


def _reload_root_conftest():
    """Reload ``tests.conftest`` after mutating env so module-level
    ``FALKORDB_URL`` / ``FALKORDB_REQUIRED`` constants pick up the change."""
    # tests/conftest.py is loaded by pytest as a special module — we reload
    # the importable copy by clearing it from sys.modules and re-importing.
    mod_name = "tests.conftest"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


def test_falkordb_url_unset_skips_probe_cleanly(monkeypatch):
    """When FALKORDB_URL is empty, _run_service_checks must NOT attempt a
    FalkorDB-specific network connection (it may still probe Redis)."""
    monkeypatch.delenv("FALKORDB_URL", raising=False)
    monkeypatch.delenv("FALKORDB_REQUIRED", raising=False)
    conftest = _reload_root_conftest()

    assert conftest.FALKORDB_URL == ""

    probed_urls: list[str] = []

    async def _fake_probe(url, timeout=3.0):
        probed_urls.append(url)
        return True  # pretend everything is up

    monkeypatch.setattr(conftest, "_check_redis_service", _fake_probe)
    results = conftest._run_service_checks()

    assert results["falkordb"] is False
    # Probe must have been called only for Redis (REDIS_URL), never for an
    # empty FalkorDB URL.
    assert "" not in probed_urls, (
        f"Probe must NOT fire with empty FALKORDB_URL (issue #183); probed: {probed_urls}"
    )
    # Defensive: no probe targeted :6380 (the stale CI port).
    assert not any(":6380" in u for u in probed_urls), (
        f"Probe must NOT target :6380 when FALKORDB_URL is unset; got: {probed_urls}"
    )


def test_falkordb_url_set_fires_probe(monkeypatch):
    """When FALKORDB_URL is set, _run_service_checks must invoke the probe."""
    monkeypatch.setenv("FALKORDB_URL", "redis://localhost:6381")
    monkeypatch.delenv("FALKORDB_REQUIRED", raising=False)
    conftest = _reload_root_conftest()

    assert conftest.FALKORDB_URL == "redis://localhost:6381"

    calls = []

    async def _fake_probe(url, timeout=3.0):
        calls.append(url)
        return True

    monkeypatch.setattr(conftest, "_check_redis_service", _fake_probe)
    results = conftest._run_service_checks()

    # Probe must have been called at least twice (redis + falkordb).
    assert any(c == "redis://localhost:6381" for c in calls), (
        f"Expected FalkorDB probe at :6381, got: {calls}"
    )
    assert results["falkordb"] is True


def test_falkordb_required_without_url_emits_warning(monkeypatch, capsys):
    """If FALKORDB_REQUIRED=1 but FALKORDB_URL is empty, the conftest emits
    a clear warning so the misconfiguration is visible — but still does NOT
    attempt to probe an empty/dead FalkorDB URL."""
    monkeypatch.setenv("FALKORDB_REQUIRED", "1")
    monkeypatch.delenv("FALKORDB_URL", raising=False)
    conftest = _reload_root_conftest()

    probed_urls: list[str] = []

    async def _fake_probe(url, timeout=3.0):
        probed_urls.append(url)
        return True

    monkeypatch.setattr(conftest, "_check_redis_service", _fake_probe)
    results = conftest._run_service_checks()

    captured = capsys.readouterr()
    assert results["falkordb"] is False
    # No FalkorDB-targeted probe: only Redis should have been probed.
    assert "" not in probed_urls
    assert not any(":6380" in u for u in probed_urls)
    assert "FALKORDB_REQUIRED=1" in captured.err
    assert "FALKORDB_URL is empty" in captured.err


def test_falkordb_client_fixture_skips_when_url_empty(monkeypatch):
    """The ``falkordb_client`` fixture must skip cleanly (not raise) when the
    URL is empty even if the global availability flag somehow indicates True.
    This pins the defensive guard added for issue #183 — previously the
    fixture would fall through into ``aioredis.from_url("")`` which raises
    a confusing ``ValueError`` deep in the redis client.

    Pass-2 LOW-2 fix: exercise the actual shared helper
    ``_enforce_falkordb_preconditions`` that the fixture invokes, instead
    of mirroring guard clauses inline (which would silently pass if the
    fixture were ever refactored to drop a guard)."""
    monkeypatch.delenv("FALKORDB_URL", raising=False)
    conftest = _reload_root_conftest()

    # Force the availability flag to True so we hit the second skip guard,
    # not the first ``not SERVICES_AVAILABLE["falkordb"]`` short-circuit.
    monkeypatch.setitem(conftest.SERVICES_AVAILABLE, "falkordb", True)

    assert conftest.FALKORDB_URL == ""

    # Invoke the actual shared helper that the fixture calls. If the
    # fixture's empty-URL guard is removed, this test fails.
    with pytest.raises(pytest.skip.Exception, match="FalkorDB URL not configured"):
        conftest._enforce_falkordb_preconditions()


def test_falkordb_client_fixture_skips_when_unavailable(monkeypatch):
    """First-guard coverage for ``_enforce_falkordb_preconditions``: skips
    when the global availability flag is False, regardless of URL state."""
    monkeypatch.setenv("FALKORDB_URL", "redis://localhost:6381")
    conftest = _reload_root_conftest()

    monkeypatch.setitem(conftest.SERVICES_AVAILABLE, "falkordb", False)

    with pytest.raises(pytest.skip.Exception, match="FalkorDB not available"):
        conftest._enforce_falkordb_preconditions()


def test_falkordb_preconditions_proceed_when_both_ok(monkeypatch):
    """When availability=True AND URL is non-empty, the helper must NOT
    skip (returns ``None``)."""
    monkeypatch.setenv("FALKORDB_URL", "redis://localhost:6381")
    conftest = _reload_root_conftest()

    monkeypatch.setitem(conftest.SERVICES_AVAILABLE, "falkordb", True)

    # Should NOT raise.
    result = conftest._enforce_falkordb_preconditions()
    assert result is None


def test_no_econnrefused_logged_on_unset_url(monkeypatch, capsys):
    """The hallmark symptom of issue #183 — the ECONNREFUSED debug line for
    redis://localhost:6380 — must NOT appear when FALKORDB_URL is unset."""
    monkeypatch.delenv("FALKORDB_URL", raising=False)
    monkeypatch.delenv("FALKORDB_REQUIRED", raising=False)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    conftest = _reload_root_conftest()

    # Real probe stub that simulates Redis up + ensures no second probe fires
    async def _fake_probe(url, timeout=3.0):
        return True  # pretend redis is up

    monkeypatch.setattr(conftest, "_check_redis_service", _fake_probe)
    conftest._run_service_checks()

    captured = capsys.readouterr()
    assert ":6380" not in captured.err, (
        f"FALKORDB :6380 must not be probed when URL is unset; stderr: {captured.err!r}"
    )
    assert "ConnectionError" not in captured.err
    assert "ECONNREFUSED" not in captured.err


# ---------------------------------------------------------------------------
# Restore environment after this module runs so the rest of the suite sees
# the original FALKORDB_URL (if any).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _restore_falkordb_env():
    original = {
        "FALKORDB_URL": os.environ.get("FALKORDB_URL"),
        "FALKORDB_REQUIRED": os.environ.get("FALKORDB_REQUIRED"),
    }
    yield
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _reload_root_conftest()
