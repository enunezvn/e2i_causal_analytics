"""Tests for the OpenTelemetry per-worker re-init helper.

Under gunicorn --preload, init_opentelemetry runs once in the master and is
idempotency-guarded by the module global ``_otel_initialized``. Forked workers
inherit that guard set, so a naive re-call is a no-op and the worker has no live
BatchSpanProcessor export thread. ``reinitialize_opentelemetry`` resets the guard
so a re-call actually rebuilds the provider in the child.
"""

from __future__ import annotations

from unittest import mock

import pytest

otel = pytest.importorskip("src.api.dependencies.opentelemetry_config")


def test_reinitialize_helper_exists() -> None:
    assert callable(getattr(otel, "reinitialize_opentelemetry", None)), (
        "reinitialize_opentelemetry helper must exist"
    )


def test_reinitialize_resets_guard_then_calls_init() -> None:
    """Even when the guard is already True, reinit must rebuild (not no-op)."""
    with mock.patch.object(otel, "OTEL_ENABLED", True):
        # Simulate inherited-from-master state: guard set, stale provider present.
        otel._otel_initialized = True
        otel._tracer = mock.sentinel.stale_tracer
        otel._tracer_provider = mock.MagicMock()

        with mock.patch.object(otel, "init_opentelemetry", wraps=lambda app=None: None) as init_spy:
            otel.reinitialize_opentelemetry()

        # The guard must have been reset to False BEFORE init was invoked,
        # otherwise init's own idempotency check would short-circuit.
        assert otel._otel_initialized is False
        init_spy.assert_called_once()


def test_reinitialize_noop_when_disabled() -> None:
    """When OTEL is disabled, reinit must not attempt to rebuild."""
    with mock.patch.object(otel, "OTEL_ENABLED", False):
        with mock.patch.object(otel, "init_opentelemetry") as init_spy:
            otel.reinitialize_opentelemetry()
        init_spy.assert_not_called()


def test_reinitialize_swaps_global_provider_despite_api_once_guard() -> None:
    """PR #569 gate 1 (#1560): the OTel API's ``set_tracer_provider`` is
    one-shot (``Once``). After the master's import-time init, a forked child's
    rebuild must still be able to REGISTER its new provider — without the
    guard reset the API logs "Overriding of current TracerProvider is not
    allowed" and keeps the inherited dead-exporter provider (measured on
    opentelemetry-api 1.39.1).
    """
    trace = pytest.importorskip("opentelemetry.trace")
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")

    prev_provider = trace._TRACER_PROVIDER
    prev_once = trace._TRACER_PROVIDER_SET_ONCE
    try:
        # Arrange: simulate the master's import-time init having consumed the
        # one-shot guard with its own provider.
        from opentelemetry.util._once import Once

        trace._TRACER_PROVIDER_SET_ONCE = Once()
        trace._TRACER_PROVIDER = None
        master_provider = sdk_trace.TracerProvider()
        trace.set_tracer_provider(master_provider)
        assert trace.get_tracer_provider() is master_provider

        # Act: the forked child re-inits; its init sets a NEW provider.
        child_provider = sdk_trace.TracerProvider()

        def _child_init(app=None):
            trace.set_tracer_provider(child_provider)
            return True

        with (
            mock.patch.object(otel, "OTEL_ENABLED", True),
            mock.patch.object(otel, "init_opentelemetry", side_effect=_child_init),
        ):
            otel.reinitialize_opentelemetry()

        # Assert: the child's provider actually took (the whole point).
        assert trace.get_tracer_provider() is child_provider, (
            "reinitialize_opentelemetry must reset the OTel API Once guard so "
            "the forked worker's provider registers; otherwise spans keep "
            "flowing to the inherited provider whose export thread died in fork"
        )
    finally:
        trace._TRACER_PROVIDER = prev_provider
        trace._TRACER_PROVIDER_SET_ONCE = prev_once


def test_reinitialize_clears_stale_provider_reference() -> None:
    """Stale inherited provider/tracer references are cleared before rebuild."""
    with mock.patch.object(otel, "OTEL_ENABLED", True):
        otel._otel_initialized = True
        otel._tracer = mock.sentinel.stale
        otel._tracer_provider = mock.MagicMock()

        captured = {}

        def _fake_init(app=None):
            # at the point init runs, guard+refs must be cleared
            captured["initialized"] = otel._otel_initialized
            captured["tracer"] = otel._tracer
            captured["provider"] = otel._tracer_provider

        with mock.patch.object(otel, "init_opentelemetry", side_effect=_fake_init):
            otel.reinitialize_opentelemetry()

    assert captured["initialized"] is False
    assert captured["tracer"] is None
    assert captured["provider"] is None
