"""Regression: the autouse finalizer in ``tests/conftest.py`` must RESTORE
``asyncio.run`` after a test that monkey-patches it (issue #218 follow-up).

Root cause recap (confirmed from the 2026-06-05 / 2026-06-06 nightly "Slow
Tests (tracked)" lane A runs): ``ragas/async_utils.py:49`` calls
``nest_asyncio.apply()`` unconditionally inside ``test_gepa_integration.py
::TestRAGASGEPAOpikIntegration::test_ragas_feedback_provider_evaluate``.
That monkey-patches ``asyncio.run`` PROCESS-WIDE on the xdist worker and
leaves it bound to a loop that pytest-asyncio later closes. The NEXT test on
the same worker that calls bare ``asyncio.run`` — e.g.
``tests/ml/synthetic_v2/test_integration_diagnostic_runner.py
::TestDiagnosticRunnerScenarioA`` — then crashes with
``RuntimeError: Event loop is closed``.

Under the slow lane's ``--dist=loadscope`` mode (``-n 2 --dist=loadscope``)
``xdist_group`` markers are ignored, so we cannot pin polluter and victim to
separate workers — co-location is left to scheduling, hence the INTERMITTENT
failure. The fix is the function-scoped autouse finalizer
``_restore_asyncio_run_after_pollution`` in ``tests/conftest.py``: it
restores the pristine ``_ORIG_ASYNCIO_RUN`` (and the loop policy) after every
test, so the next test always gets a working ``asyncio.run`` regardless of
co-scheduling or which test tree it lives in.

This module pins that contract with a two-test sequence executed IN ORDER:

  1. ``test_a_pollutes_asyncio_run`` faithfully reproduces the failure mode
     (patches ``asyncio.run`` with a runner bound to a CLOSED loop, exactly
     as nest_asyncio leaves it). The autouse finalizer should heal this on
     teardown.
  2. ``test_b_asyncio_run_is_pristine_and_works`` runs next and asserts
     ``asyncio.run`` is back to the genuine stdlib runner AND that a real
     ``asyncio.run(coro)`` succeeds (the exact victim operation).

Without the finalizer, step 1's pollution would leak into step 2 and the
``asyncio.run(coro)`` in ``test_b`` would raise ``RuntimeError: Event loop
is closed`` — the production symptom. With it, ``test_b`` is green.

We simulate the polluter instead of importing ``nest_asyncio`` directly so
the regression runs even in minimal environments and does not itself depend
on ragas being installed; the simulated patch is byte-for-byte the failure
mode (a wrapper that calls ``loop.run_until_complete`` on a closed loop),
validated against the real symptom.

Placement: this lives in ``tests/ml/synthetic_v2/`` — the SAME tree as the
real victims (``test_integration_diagnostic_runner.py``) — rather than in
``tests/integration/``. That is deliberate: (1) it pins the contract next to
the code it protects, and (2) the integration-only lint
``tests/integration/test_no_bare_asyncio_run_in_integration_tests.py``
forbids bare ``asyncio.run`` in sync test bodies, but this regression MUST
call bare ``asyncio.run`` (that is the exact victim operation it proves is
healed). The lint scans only ``tests/integration/``, so placing the pin in
the victim tree keeps the lint's intent intact without adding a carve-out.
"""

from __future__ import annotations

import asyncio

import pytest

# Import the conftest-captured pristine runner so the test asserts against the
# SAME identity the finalizer restores to.
from tests.conftest import _ORIG_ASYNCIO_RUN


async def _sample_coro() -> int:
    return 42


def _install_closed_loop_pollution() -> None:
    """Patch ``asyncio.run`` exactly the way nest_asyncio leaves it after a
    pytest-asyncio teardown: a runner that delegates to a loop which has
    already been closed. Calling the patched ``asyncio.run`` raises
    ``RuntimeError: Event loop is closed`` — the production symptom.
    """

    dead_loop = asyncio.new_event_loop()
    dead_loop.close()  # mimic pytest-asyncio closing the per-test loop

    def _polluted_run(coro, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return dead_loop.run_until_complete(coro)
        finally:
            # Avoid "coroutine was never awaited" noise if the closed loop
            # rejects the coro before consuming it.
            coro.close()

    asyncio.run = _polluted_run  # type: ignore[assignment]


@pytest.mark.slow
class TestAsyncioRunRestoredAfterPollution:
    """The two tests run in definition order within the class; pytest
    preserves intra-class ordering, so ``test_a`` always pollutes before
    ``test_b`` checks the heal."""

    def test_a_pollutes_asyncio_run(self) -> None:
        """Sanity-check the simulation faithfully reproduces the symptom,
        then leave ``asyncio.run`` polluted for the finalizer to heal."""

        _install_closed_loop_pollution()
        assert asyncio.run is not _ORIG_ASYNCIO_RUN
        # The polluted runner reproduces the exact production failure.
        with pytest.raises(RuntimeError, match="Event loop is closed"):
            asyncio.run(_sample_coro())
        # Intentionally do NOT restore here — the autouse finalizer in
        # tests/conftest.py is the thing under test.

    def test_b_asyncio_run_is_pristine_and_works(self) -> None:
        """After ``test_a`` polluted ``asyncio.run``, the autouse finalizer
        must have restored the pristine stdlib runner so this test's
        ``asyncio.run`` (the victim operation) succeeds."""

        # Identity restored to the genuine stdlib runner captured at import.
        assert asyncio.run is _ORIG_ASYNCIO_RUN, (
            "asyncio.run was NOT restored after the previous test polluted it "
            "— the autouse _restore_asyncio_run_after_pollution finalizer in "
            "tests/conftest.py regressed. This is the issue-#218 cross-test "
            "RuntimeError('Event loop is closed') leak."
        )
        # The exact victim operation (bare asyncio.run) must now work.
        assert asyncio.run(_sample_coro()) == 42


@pytest.mark.slow
def test_finalizer_is_noop_for_unpolluted_tests() -> None:
    """A test that never touches ``asyncio.run`` is unaffected: the runner is
    pristine before AND after, and a plain ``asyncio.run`` works. Guards
    against the finalizer introducing overhead/regressions on the common
    path."""

    assert asyncio.run is _ORIG_ASYNCIO_RUN
    assert asyncio.run(_sample_coro()) == 42
    assert asyncio.run is _ORIG_ASYNCIO_RUN
