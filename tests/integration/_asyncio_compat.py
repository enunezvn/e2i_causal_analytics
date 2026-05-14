"""Shared asyncio-pollution-safe helpers for integration tests (issue #220).

The default ``asyncio.run(coro)`` call is unsafe in this repo's integration
test suite because at least one third-party dependency (``ragas`` —
see ``ragas/async_utils.py:49``, identified by PR #219's runtime probe)
calls ``nest_asyncio.apply()`` unconditionally at the top of one of its
helpers. Once that fires on an xdist worker, every subsequent
``asyncio.run(coro)`` on the same worker routes through
``nest_asyncio.run``, which references the loop captured at apply-time.
If that loop has been closed (the common pytest-asyncio per-test teardown
path), the next sync ``asyncio.run`` call raises ``RuntimeError: Event
loop is closed`` — issue #215's victim pattern.

The fix is to call ``loop.run_until_complete(coro)`` on a freshly-created
loop, then close it deterministically. PR #217 commit ``a321b64f``
shipped this pattern inline; issue #220 extracted it here so that the
~15 integration test callsites consume one centralised implementation.

Usage in a synchronous test body::

    from tests.integration._asyncio_compat import run_sync

    def test_thing():
        result = run_sync(some_coro())
        ...
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")


def run_sync(coro: Awaitable[T]) -> T:
    """Run ``coro`` on a freshly-created event loop, then close the loop.

    Issue #220: this replaces bare ``asyncio.run(coro)`` so the call is
    robust against the RAGAS / DSPy / mlflow.genai ``nest_asyncio.apply()``
    chain that may have already monkey-patched ``asyncio.run`` on the same
    xdist worker. ``loop.run_until_complete`` does NOT route through the
    patched ``asyncio.run``; it goes directly to the loop runner, which
    is what we want.

    Args:
        coro: An awaitable (coroutine, Future, Task) to run to completion.

    Returns:
        The value returned by ``coro``.

    Raises:
        Whatever ``coro`` raises. The loop is always closed in ``finally``.

    Notes:
        - Creates a new event loop on every call. Do NOT reuse the
          returned loop; it is closed before this function returns.
        - Safe to call from any synchronous context, including pytest
          test bodies, fixtures, and module-level helpers.
        - Inside ``async def`` code, use ``await coro`` directly — this
          helper is only for the sync-to-async boundary.
    """

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def run_sync_returning(coro: Awaitable[Any]) -> Any:
    """Alias for :func:`run_sync` for call sites that prefer the more
    explicit name when the return value is consumed.

    Kept thin so call sites that read ``result = run_sync_returning(...)``
    are visually distinct from fire-and-forget ``run_sync(...)`` lines.
    """

    return run_sync(coro)
