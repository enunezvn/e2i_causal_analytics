"""Bootstrap under a time budget (Lane D item 2).

Spec: docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md,
"Lane D — guided discovery on claims frames", item 2: ``time_budget_s``
bounds the resample loop; the ACHIEVED resample count is reported and the
gate's corroboration is computed over it; a run that achieves fewer than
``min_resamples`` is reported as uncorroborated, never as corroborated.

The slow algorithm here sleeps a fixed wall-clock per call, so the number of
resamples the budget admits is known up front; the primary fit's runtime is
charged against the budget too (the loop cannot pretend it started at zero).
"""

from __future__ import annotations

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
from src.causal_engine.discovery.gate import DiscoveryGate
from src.causal_engine.discovery.hasher import hash_config
from src.causal_engine.discovery.runner import DiscoveryRunner


class _SlowAlgorithm(BaseDiscoveryAlgorithm):
    """Every call sleeps ``seconds`` and returns a->b; every second resample
    also returns c->d, so a->b has stability 1.0 and c->d about 0.5."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.calls = 0

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        return DiscoveryAlgorithmType.PC

    def supports_latent_confounders(self) -> bool:
        return False

    def discover(self, data: pd.DataFrame, config: DiscoveryConfig) -> AlgorithmResult:
        self.calls += 1
        start = time.time()
        time.sleep(self.seconds)
        edges = [("a", "b")] if self.calls % 2 else [("a", "b"), ("c", "d")]
        if self.calls == 1:
            edges = [("a", "b"), ("c", "d")]
        n = len(data.columns)
        return AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.PC,
            adjacency_matrix=np.zeros((n, n), dtype=int),
            edge_list=edges,
            runtime_seconds=time.time() - start,
            converged=True,
        )


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(60, 4)), columns=["a", "b", "c", "d"])


def _runner(algorithm: BaseDiscoveryAlgorithm) -> DiscoveryRunner:
    runner = DiscoveryRunner(enable_tracing=False)
    runner._algorithms[DiscoveryAlgorithmType.PC] = algorithm
    return runner


class TestConfigFields:
    def test_defaults_are_unbounded_and_legacy(self) -> None:
        config = DiscoveryConfig()
        assert config.time_budget_s is None
        assert config.min_resamples is None

    def test_round_trip_and_hash(self) -> None:
        config = DiscoveryConfig(bootstrap_resamples=20, time_budget_s=180.0, min_resamples=10)
        restored = DiscoveryConfig.from_dict(config.to_dict())
        assert restored.time_budget_s == 180.0
        assert restored.min_resamples == 10
        # A budget changes what discovery can achieve, so it is part of the
        # cache identity.
        assert hash_config(config) != hash_config(DiscoveryConfig(bootstrap_resamples=20))
        assert hash_config(config) != hash_config(
            DiscoveryConfig(bootstrap_resamples=20, time_budget_s=180.0)
        )


class TestBudgetBoundsTheLoop:
    @pytest.mark.asyncio
    async def test_budget_stops_the_loop_and_reports_the_achieved_count(self) -> None:
        """0.05 s per fit, budget 0.42 s: the primary fit spends ~0.05 s, so at
        most 7 resamples can START before the loop would overrun; 20 were
        requested. The count reported is the achieved one, stability is
        computed over it, and the run says the budget bit."""
        algorithm = _SlowAlgorithm(0.05)
        runner = _runner(algorithm)
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC],
            bootstrap_resamples=20,
            time_budget_s=0.42,
            min_resamples=2,
        )
        result = await runner.discover_dag(_frame(), config)
        summary = result.metadata["bootstrap"]
        assert summary["n_resamples"] == 20
        assert 2 <= summary["n_attempted"] <= 8
        assert summary["n_attempted"] == algorithm.calls - 1
        assert summary["n_succeeded"] == summary["n_attempted"]
        assert summary["budget_exhausted"] is True
        assert summary["time_budget_s"] == 0.42
        assert summary["corroborated"] is True
        assert summary["elapsed_s"] <= 0.42 + 0.05 + 0.05  # one fit of overshoot at most
        by_edge = {(e.source, e.target): e for e in result.edges}
        assert by_edge[("a", "b")].bootstrap_stability == 1.0
        # c->d on every second call: computed over the achieved count, so it is
        # a multiple of 1/n_succeeded, not of 1/20.
        stability = by_edge[("c", "d")].bootstrap_stability
        assert stability is not None
        assert abs(stability * summary["n_succeeded"] - round(stability * summary["n_succeeded"])) < 1e-9

    @pytest.mark.asyncio
    async def test_no_budget_runs_every_requested_resample(self) -> None:
        algorithm = _SlowAlgorithm(0.0)
        runner = _runner(algorithm)
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC], bootstrap_resamples=6, min_resamples=2
        )
        result = await runner.discover_dag(_frame(), config)
        summary = result.metadata["bootstrap"]
        assert summary["n_attempted"] == 6
        assert summary["n_succeeded"] == 6
        assert summary["budget_exhausted"] is False
        assert summary["time_budget_s"] is None
        assert algorithm.calls == 7

    @pytest.mark.asyncio
    async def test_primary_fit_is_charged_against_the_budget(self) -> None:
        """A primary fit that already consumed the whole budget leaves no room
        for a single resample: zero attempted, reported, uncorroborated."""
        algorithm = _SlowAlgorithm(0.2)
        runner = _runner(algorithm)
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC],
            bootstrap_resamples=20,
            time_budget_s=0.1,
            min_resamples=2,
        )
        result = await runner.discover_dag(_frame(), config)
        summary = result.metadata["bootstrap"]
        assert summary["n_attempted"] == 0
        assert summary["n_succeeded"] == 0
        assert summary["budget_exhausted"] is True
        assert summary["corroborated"] is False
        assert algorithm.calls == 1
        assert all(e.bootstrap_stability is None for e in result.edges)


class TestMinResamples:
    @pytest.mark.asyncio
    async def test_fewer_than_min_resamples_is_uncorroborated(self) -> None:
        """Budget admits ~4 resamples, min_resamples=10: the stabilities are
        NOT written (the gate scores the run uncorroborated) but the achieved
        count is still reported."""
        algorithm = _SlowAlgorithm(0.05)
        runner = _runner(algorithm)
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC],
            bootstrap_resamples=20,
            time_budget_s=0.28,
            min_resamples=10,
        )
        result = await runner.discover_dag(_frame(), config)
        summary = result.metadata["bootstrap"]
        assert 1 <= summary["n_succeeded"] < 10
        assert summary["min_resamples"] == 10
        assert summary["corroborated"] is False
        assert all(e.bootstrap_stability is None for e in result.edges)
        evaluation = DiscoveryGate().evaluate(result)
        assert evaluation.metadata["corroboration_basis"] == "uncorroborated_single_run"
        assert evaluation.confidence == 0.0

    @pytest.mark.asyncio
    async def test_min_resamples_met_is_corroborated(self) -> None:
        algorithm = _SlowAlgorithm(0.0)
        runner = _runner(algorithm)
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC], bootstrap_resamples=10, min_resamples=10
        )
        result = await runner.discover_dag(_frame(), config)
        summary = result.metadata["bootstrap"]
        assert summary["n_succeeded"] == 10
        assert summary["corroborated"] is True
        evaluation = DiscoveryGate().evaluate(result)
        assert evaluation.metadata["corroboration_basis"] == "bootstrap_stability"

    @pytest.mark.asyncio
    async def test_legacy_rule_when_min_resamples_unset(self) -> None:
        """No ``min_resamples``: the pre-existing ``max(2, B // 2)`` rule holds
        for consumers that never set one (pinned in test_bootstrap_stability)."""
        algorithm = _SlowAlgorithm(0.0)
        runner = _runner(algorithm)
        config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC], bootstrap_resamples=4)
        result = await runner.discover_dag(_frame(), config)
        summary = result.metadata["bootstrap"]
        assert summary["min_resamples"] == 2
        assert summary["corroborated"] is True
