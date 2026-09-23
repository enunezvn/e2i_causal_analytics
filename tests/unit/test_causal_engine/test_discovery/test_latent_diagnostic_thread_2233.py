"""#2233 defect 3: the FCI latent diagnostic's timeout must abandon its thread
deterministically.

Live (2026-09-22): the diagnostic reported ``timed out after 12.1s`` and a thread
was still inside causallearn FCI 68 s later — on the event loop's SHARED default
executor (``loop.run_in_executor(None, …)``), whose threads ``asyncio.run`` JOINS
at shutdown (the Lane D acceptance script hung to rc=124 on exactly that). The
diagnostic now runs on a dedicated DAEMON thread: the timeout path never waits on
it, nothing joins it at interpreter or worker exit, the default executor keeps
its slots, and the result records that the thread outlived the timeout.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.base import (
    AlgorithmResult,
    BaseDiscoveryAlgorithm,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
)
from src.causal_engine.discovery.runner import DiscoveryRunner


class _Instant(BaseDiscoveryAlgorithm):
    def __init__(self, algo: DiscoveryAlgorithmType, seconds: float = 0.0) -> None:
        self._algo = algo
        self.seconds = seconds
        self.threads: list[threading.Thread] = []

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        return self._algo

    def supports_latent_confounders(self) -> bool:
        return self._algo is DiscoveryAlgorithmType.FCI

    def get_bidirected_edges(self, result: AlgorithmResult) -> list:
        return []

    def discover(self, data: pd.DataFrame, config: DiscoveryConfig) -> AlgorithmResult:
        self.threads.append(threading.current_thread())
        start = time.time()
        time.sleep(self.seconds)
        n = len(data.columns)
        return AlgorithmResult(
            algorithm=self._algo,
            adjacency_matrix=np.zeros((n, n), dtype=int),
            edge_list=[("a", "b")],
            runtime_seconds=time.time() - start,
            converged=True,
        )


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])


def _runner(fci_seconds: float) -> tuple[DiscoveryRunner, _Instant]:
    runner = DiscoveryRunner(enable_tracing=False)
    runner._algorithms[DiscoveryAlgorithmType.PC] = _Instant(DiscoveryAlgorithmType.PC, 0.0)
    fci = _Instant(DiscoveryAlgorithmType.FCI, fci_seconds)
    runner._algorithms[DiscoveryAlgorithmType.FCI] = fci
    return runner, fci


@pytest.mark.asyncio
async def test_timed_out_diagnostic_abandons_a_daemon_thread_and_says_so() -> None:
    runner, fci = _runner(fci_seconds=0.8)
    config = DiscoveryConfig(
        algorithms=[DiscoveryAlgorithmType.PC], latent_diagnostic=True, time_budget_s=0.2
    )
    t0 = time.monotonic()
    result = await runner.discover_dag(_frame(), config)
    wall = time.monotonic() - t0
    payload = result.metadata["latent_diagnostic"]
    assert payload["ran"] is True and payload["converged"] is False
    assert "timeout" in payload["error"]
    # The run did NOT wait for the fit: it returned at the budget, not at 0.8 s.
    assert wall < 0.7, wall
    assert payload["fci_thread_outlived_timeout"] is True
    assert len(fci.threads) == 1
    thread = fci.threads[0]
    assert payload["fci_thread_name"] == thread.name
    assert thread.name.startswith("fci-latent-diagnostic")
    assert thread.daemon is True, "a non-daemon thread is joined at interpreter exit"
    assert thread.is_alive()  # abandoned, still computing — by design, and recorded
    thread.join(timeout=2.0)  # keep the test process tidy


@pytest.mark.asyncio
async def test_within_budget_records_no_outlived_thread() -> None:
    runner, fci = _runner(fci_seconds=0.0)
    config = DiscoveryConfig(
        algorithms=[DiscoveryAlgorithmType.PC], latent_diagnostic=True, time_budget_s=5.0
    )
    result = await runner.discover_dag(_frame(), config)
    payload = result.metadata["latent_diagnostic"]
    assert payload["ran"] is True and payload["converged"] is True
    assert payload["fci_thread_outlived_timeout"] is False
    assert fci.threads and fci.threads[0].daemon is True


@pytest.mark.asyncio
async def test_diagnostic_does_not_use_the_shared_default_executor(monkeypatch) -> None:
    """Teeth: with the loop's ``run_in_executor`` disabled the diagnostic still
    runs — it holds no slot of the executor ``asyncio.to_thread`` and the
    bootstrap share, and that ``asyncio.run`` joins at shutdown."""
    loop = asyncio.get_running_loop()

    def boom(*args, **kwargs):
        raise AssertionError("latent diagnostic must not use the default executor")

    monkeypatch.setattr(loop, "run_in_executor", boom)
    runner, fci = _runner(fci_seconds=0.0)
    config = DiscoveryConfig(
        algorithms=[DiscoveryAlgorithmType.PC], latent_diagnostic=True, time_budget_s=5.0
    )
    result = await runner.discover_dag(_frame(), config)
    payload = result.metadata["latent_diagnostic"]
    assert payload["ran"] is True and payload["converged"] is True
    assert fci.threads and fci.threads[0] is not threading.main_thread()


# ---------------------------------------------------------------------------
# Codex r1 HIGH: abandoned fits must not ACCUMULATE. The heavy-compute slot is
# released when the graph ends, not when the abandoned thread ends, so a later
# guided run could start while a ~380 s FCI fit still holds the GIL (measured
# 2.5x slowdown of the loop thread under one busy thread). Documented bound:
# at most ONE outstanding diagnostic thread per process; a new diagnostic is
# skipped (ran=False, reason named) while one is alive.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_diagnostic_is_skipped_while_an_outlived_thread_is_alive() -> None:
    from src.causal_engine.discovery import runner as runner_mod

    first, fci1 = _runner(fci_seconds=1.0)
    budget = DiscoveryConfig(
        algorithms=[DiscoveryAlgorithmType.PC], latent_diagnostic=True, time_budget_s=0.2
    )
    r1 = await first.discover_dag(_frame(), budget)
    assert r1.metadata["latent_diagnostic"]["fci_thread_outlived_timeout"] is True
    assert runner_mod.outlived_diagnostic_threads() == [fci1.threads[0].name]

    second, fci2 = _runner(fci_seconds=0.0)
    roomy = DiscoveryConfig(
        algorithms=[DiscoveryAlgorithmType.PC], latent_diagnostic=True, time_budget_s=5.0
    )
    r2 = await second.discover_dag(_frame(), roomy)
    payload = r2.metadata["latent_diagnostic"]
    assert payload["ran"] is False
    assert "still running" in payload["error"] and fci1.threads[0].name in payload["error"]
    assert payload["fci_outlived_threads_alive"] == [fci1.threads[0].name]
    assert fci2.threads == []  # never started
    assert r2.success is True  # the diagnostic annotates, never gates

    fci1.threads[0].join(timeout=3.0)
    assert runner_mod.outlived_diagnostic_threads() == []
    r3 = await second.discover_dag(_frame(), roomy)
    assert r3.metadata["latent_diagnostic"]["ran"] is True
    assert len(fci2.threads) == 1
