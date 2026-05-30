"""Unit tests for the bounded heavy-compute limiter (Priority 1 OOM fix).

These tests pin the per-worker concurrency limiter that prevents concurrent
heavy requests from OOM-killing the e2i_api container (5G cgroup cap; each
heavy request peaks ~1.3 GiB).

The limiter must:
- Admit up to ``HEAVY_COMPUTE_MAX_CONCURRENCY`` in-flight heavy ops.
- REJECT FAST the (N+1)th op (raise ``HeavyComputeSaturated``) rather than queue
  unboundedly (queuing would rebuild the memory/latency we are bounding).
- Free a slot on release (including on exception).
- Run blocking callables OFF the event loop via a shared bounded executor.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.api.dependencies.compute import (
    HeavyComputeSaturated,
    get_heavy_compute_limiter,
    heavy_compute_slot,
    run_in_bounded_executor,
)


@pytest.fixture(autouse=True)
def _reset_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test controls concurrency explicitly and starts from a clean limiter."""
    monkeypatch.setenv("HEAVY_COMPUTE_MAX_CONCURRENCY", "2")
    monkeypatch.delenv("HEAVY_COMPUTE_EXECUTOR_WORKERS", raising=False)
    import src.api.dependencies.compute as compute_mod

    compute_mod._reset_limiter_cache_for_tests()


async def test_concurrency_limit_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEAVY_COMPUTE_MAX_CONCURRENCY", "3")
    import src.api.dependencies.compute as compute_mod

    compute_mod._reset_limiter_cache_for_tests()

    limiter = get_heavy_compute_limiter()
    assert limiter.max_concurrency == 3


async def test_acquires_up_to_n_then_rejects() -> None:
    limiter = get_heavy_compute_limiter()
    assert limiter.max_concurrency == 2

    limiter.acquire()
    limiter.acquire()
    assert limiter.in_flight == 2

    # The (N+1)th acquire must fail fast, not block / queue.
    with pytest.raises(HeavyComputeSaturated):
        limiter.acquire()


async def test_release_frees_a_slot() -> None:
    limiter = get_heavy_compute_limiter()
    limiter.acquire()
    limiter.acquire()
    with pytest.raises(HeavyComputeSaturated):
        limiter.acquire()

    limiter.release()
    assert limiter.in_flight == 1

    # A freed slot can be re-acquired.
    limiter.acquire()
    assert limiter.in_flight == 2
    with pytest.raises(HeavyComputeSaturated):
        limiter.acquire()


async def test_slot_context_manager_acquires_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEAVY_COMPUTE_MAX_CONCURRENCY", "1")
    import src.api.dependencies.compute as compute_mod

    compute_mod._reset_limiter_cache_for_tests()
    limiter = get_heavy_compute_limiter()

    async with heavy_compute_slot():
        assert limiter.in_flight == 1
        # While the single slot is held, a second slot must be rejected.
        with pytest.raises(HeavyComputeSaturated):
            async with heavy_compute_slot():
                pass  # pragma: no cover - should never enter

    assert limiter.in_flight == 0


async def test_slot_releases_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEAVY_COMPUTE_MAX_CONCURRENCY", "1")
    import src.api.dependencies.compute as compute_mod

    compute_mod._reset_limiter_cache_for_tests()
    limiter = get_heavy_compute_limiter()

    with pytest.raises(RuntimeError):
        async with heavy_compute_slot():
            assert limiter.in_flight == 1
            raise RuntimeError("boom")

    assert limiter.in_flight == 0


async def test_run_in_bounded_executor_runs_off_event_loop() -> None:
    """A slow blocking call in the executor must NOT block a concurrent coroutine.

    Behavioral proof that the work runs off the single-threaded event loop:
    launch one slow blocking call and one fast coroutine concurrently; the fast
    coroutine must finish while the slow blocking call is still running. If the
    blocking call ran ON the loop, the fast coroutine could not progress until it
    returned.
    """
    order: list[str] = []

    def _slow_blocking() -> str:
        time.sleep(0.30)  # real blocking sleep (not asyncio.sleep)
        order.append("slow_done")
        return "slow"

    async def _fast() -> str:
        await asyncio.sleep(0.05)
        order.append("fast_done")
        return "fast"

    slow_task = asyncio.create_task(run_in_bounded_executor(_slow_blocking))
    fast_task = asyncio.create_task(_fast())

    fast_result = await fast_task
    assert fast_result == "fast"
    # Fast finished while slow is still running off-loop.
    assert order == ["fast_done"], order

    slow_result = await slow_task
    assert slow_result == "slow"
    assert order == ["fast_done", "slow_done"], order


async def test_run_in_bounded_executor_passes_args_and_kwargs() -> None:
    def _add(a: int, b: int, *, c: int = 0) -> int:
        return a + b + c

    result = await run_in_bounded_executor(_add, 2, 3, c=4)
    assert result == 9
