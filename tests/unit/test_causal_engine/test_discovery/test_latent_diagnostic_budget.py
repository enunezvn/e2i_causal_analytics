"""The FCI latent diagnostic falls under the discovery time budget, and the
PC wrapper honours a forced independence test (Lane D items 2 and 4).

Measured (docs/demos/results/2026-09-22_lane_d_guided_discovery_claims/): one
unguided FCI fit on the capped real frame is 382.5 s against a 180 s budget.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.algorithms.pc_wrapper import PCAlgorithm
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
        self.calls = 0

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        return self._algo

    def supports_latent_confounders(self) -> bool:
        return self._algo is DiscoveryAlgorithmType.FCI

    def get_bidirected_edges(self, result: AlgorithmResult) -> list:
        return []

    def discover(self, data: pd.DataFrame, config: DiscoveryConfig) -> AlgorithmResult:
        self.calls += 1
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


def _runner(pc_seconds: float, fci_seconds: float) -> tuple[DiscoveryRunner, _Instant]:
    runner = DiscoveryRunner(enable_tracing=False)
    runner._algorithms[DiscoveryAlgorithmType.PC] = _Instant(DiscoveryAlgorithmType.PC, pc_seconds)
    fci = _Instant(DiscoveryAlgorithmType.FCI, fci_seconds)
    runner._algorithms[DiscoveryAlgorithmType.FCI] = fci
    return runner, fci


class TestLatentDiagnosticUnderBudget:
    @pytest.mark.asyncio
    async def test_not_started_when_the_budget_is_spent(self) -> None:
        runner, fci = _runner(pc_seconds=0.15, fci_seconds=0.0)
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC], latent_diagnostic=True, time_budget_s=0.1
        )
        result = await runner.discover_dag(_frame(), config)
        payload = result.metadata["latent_diagnostic"]
        assert payload["ran"] is False
        assert "budget" in payload["error"]
        assert fci.calls == 0

    @pytest.mark.asyncio
    async def test_timed_out_diagnostic_is_reported_not_silent(self) -> None:
        runner, fci = _runner(pc_seconds=0.0, fci_seconds=0.6)
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC], latent_diagnostic=True, time_budget_s=0.2
        )
        result = await runner.discover_dag(_frame(), config)
        payload = result.metadata["latent_diagnostic"]
        assert payload["ran"] is True
        assert payload["converged"] is False
        assert "timeout" in payload["error"]
        assert payload["bidirected_edges"] == []
        assert fci.calls == 1
        # discovery itself still succeeded: the diagnostic annotates, never gates
        assert result.success is True

    @pytest.mark.asyncio
    async def test_within_budget_runs_and_records_the_budget(self) -> None:
        runner, fci = _runner(pc_seconds=0.0, fci_seconds=0.0)
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC], latent_diagnostic=True, time_budget_s=5.0
        )
        result = await runner.discover_dag(_frame(), config)
        payload = result.metadata["latent_diagnostic"]
        assert payload["ran"] is True and payload["converged"] is True
        assert payload["time_budget_s"] == 5.0
        assert fci.calls == 1

    @pytest.mark.asyncio
    async def test_no_budget_is_unbounded_legacy(self) -> None:
        runner, fci = _runner(pc_seconds=0.0, fci_seconds=0.05)
        config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC], latent_diagnostic=True)
        result = await runner.discover_dag(_frame(), config)
        payload = result.metadata["latent_diagnostic"]
        assert payload["ran"] is True and payload["converged"] is True
        assert "time_budget_s" not in payload


def _mixed_frame(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sev = rng.normal(size=n)
    t = (0.9 * sev + rng.normal(size=n) > 0).astype(float)
    y = (0.8 * t + 0.8 * sev + rng.normal(size=n) > 0).astype(float)
    return pd.DataFrame({"t": t, "y": y, "sev": sev})


class TestForcedIndependenceTest:
    def test_default_is_the_measured_auto_selection(self) -> None:
        result = PCAlgorithm().discover(_mixed_frame(), DiscoveryConfig())
        assert result.metadata["indep_test"] == "fisherz"
        assert result.metadata["indep_test_forced"] is False

    def test_forced_test_is_used_and_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The forced test must reach causal-learn's ``pc`` call, not only the
        metadata (codex r1 finding 7: a wrapper that ran fisherz and wrote
        'gsq' would pass a metadata-only assertion)."""
        import causallearn.search.ConstraintBased.PC as pc_module

        real_pc = pc_module.pc
        seen: list = []

        def spy(X, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(kwargs.get("indep_test"))
            return real_pc(X, **kwargs)

        monkeypatch.setattr(pc_module, "pc", spy)
        frame = _mixed_frame()
        frame["sev"] = (frame["sev"] > 0).astype(float)  # gsq needs discrete data
        result = PCAlgorithm().discover(frame, DiscoveryConfig(indep_test="gsq"))
        assert result.converged, result.metadata
        assert seen == ["gsq"]
        assert result.metadata["indep_test"] == "gsq"
        assert result.metadata["indep_test_forced"] is True
