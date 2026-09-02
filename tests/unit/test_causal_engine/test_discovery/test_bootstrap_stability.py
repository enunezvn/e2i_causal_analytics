"""Bootstrap edge stability: data model, hashing, cache round-trip, runner.

Fix 2 of the causal-DAG grading sequence (gate corroboration). The runner
tests live here too (Task 3) so the whole bootstrap seam is one file.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.base import (
    AlgorithmResult,
    BaseDiscoveryAlgorithm,
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
)
from src.causal_engine.discovery.cache import DiscoveryCache
from src.causal_engine.discovery.hasher import hash_config
from src.causal_engine.discovery.runner import DiscoveryRunner


class TestBootstrapStabilityModel:
    def test_edge_defaults_to_no_stability(self) -> None:
        edge = DiscoveredEdge(source="a", target="b")
        assert edge.bootstrap_stability is None

    def test_edge_to_dict_carries_stability(self) -> None:
        edge = DiscoveredEdge(source="a", target="b", bootstrap_stability=0.85)
        assert edge.to_dict()["bootstrap_stability"] == 0.85

    def test_config_defaults_bootstrap_off_and_serializes(self) -> None:
        config = DiscoveryConfig()
        assert config.bootstrap_resamples == 0
        assert config.to_dict()["bootstrap_resamples"] == 0

    def test_bootstrap_resamples_changes_config_hash(self) -> None:
        assert hash_config(DiscoveryConfig()) != hash_config(
            DiscoveryConfig(bootstrap_resamples=20)
        )

    def test_cache_round_trips_stability(self) -> None:
        cache = DiscoveryCache()
        edge = DiscoveredEdge(source="a", target="b", bootstrap_stability=0.4)
        payload = json.dumps(
            {
                "success": True,
                "config": DiscoveryConfig().to_dict(),
                "edges": [edge.to_dict()],
                "gate_decision": None,
                "gate_confidence": 0.0,
                "created_at": "2026-09-01T00:00:00+00:00",
                "session_id": None,
                "metadata": {},
            }
        )
        restored = cache._deserialize_result(payload)
        assert restored is not None
        assert restored.edges[0].bootstrap_stability == 0.4

    def test_cache_round_trip_missing_stability_key_defaults_none(self) -> None:
        """Legacy cache payload written before bootstrap_stability existed:
        the edge dict lacks the key entirely, not just carries None."""
        cache = DiscoveryCache()
        edge = DiscoveredEdge(source="a", target="b", bootstrap_stability=0.4)
        edge_dict = edge.to_dict()
        del edge_dict["bootstrap_stability"]
        payload = json.dumps(
            {
                "success": True,
                "config": DiscoveryConfig().to_dict(),
                "edges": [edge_dict],
                "gate_decision": None,
                "gate_confidence": 0.0,
                "created_at": "2026-09-01T00:00:00+00:00",
                "session_id": None,
                "metadata": {},
            }
        )
        restored = cache._deserialize_result(payload)
        assert restored is not None
        assert restored.edges[0].bootstrap_stability is None


class _ScriptedAlgorithm(BaseDiscoveryAlgorithm):
    """Primary call returns two edges; resample calls return a->b always
    and c->d on every second call — so stability is 1.0 and 0.5."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        return DiscoveryAlgorithmType.PC

    def supports_latent_confounders(self) -> bool:
        return False

    def discover(self, data: pd.DataFrame, config: DiscoveryConfig) -> AlgorithmResult:
        self.calls += 1
        if self.calls == 1:
            edges = [("a", "b"), ("c", "d")]
        elif self.calls % 2 == 0:
            edges = [("a", "b"), ("c", "d")]
        else:
            edges = [("a", "b")]
        n = len(data.columns)
        return AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.PC,
            adjacency_matrix=np.zeros((n, n), dtype=int),
            edge_list=edges,
            runtime_seconds=0.0,
            converged=True,
        )


class _Flaky(BaseDiscoveryAlgorithm):
    """Primary call converges with a single edge; every subsequent call
    (i.e. every bootstrap resample) raises."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        return DiscoveryAlgorithmType.PC

    def supports_latent_confounders(self) -> bool:
        return False

    def discover(self, data: pd.DataFrame, config: DiscoveryConfig) -> AlgorithmResult:
        self.calls += 1
        if self.calls == 1:
            n = len(data.columns)
            return AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((n, n), dtype=int),
                edge_list=[("a", "b")],
                runtime_seconds=0.0,
                converged=True,
            )
        raise RuntimeError("degenerate resample")


class _PartialFailureAlgorithm(BaseDiscoveryAlgorithm):
    """Primary call converges with a->b. Of the 10 resample calls that
    follow: calls 1-2 raise, call 3 returns converged=False, and of the
    remaining 7 (calls 4-10), calls 4-7 recover a->b and calls 8-10 do not.
    Pins the stability denominator to n_succeeded (7), not n_resamples (10):
    4/7 != 4/10."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        return DiscoveryAlgorithmType.PC

    def supports_latent_confounders(self) -> bool:
        return False

    def discover(self, data: pd.DataFrame, config: DiscoveryConfig) -> AlgorithmResult:
        self.calls += 1
        n = len(data.columns)
        if self.calls == 1:
            return AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((n, n), dtype=int),
                edge_list=[("a", "b")],
                runtime_seconds=0.0,
                converged=True,
            )
        resample_call = self.calls - 1  # 1..10
        if resample_call in (1, 2):
            raise RuntimeError("degenerate resample")
        if resample_call == 3:
            return AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((n, n), dtype=int),
                edge_list=[],
                runtime_seconds=0.0,
                converged=False,
            )
        edges = [("a", "b")] if resample_call <= 7 else []
        return AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.PC,
            adjacency_matrix=np.zeros((n, n), dtype=int),
            edge_list=edges,
            runtime_seconds=0.0,
            converged=True,
        )


class _FixedAlgorithm(BaseDiscoveryAlgorithm):
    """Always converges and returns the same fixed edge list. Used to
    populate a >=2-algorithm ensemble; tracks call count so a test can
    assert it was never re-invoked for bootstrap resampling."""

    def __init__(self, algo_type: DiscoveryAlgorithmType, edges: list[tuple[str, str]]) -> None:
        self._algo_type = algo_type
        self._edges = edges
        self.calls = 0

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        return self._algo_type

    def supports_latent_confounders(self) -> bool:
        return False

    def discover(self, data: pd.DataFrame, config: DiscoveryConfig) -> AlgorithmResult:
        self.calls += 1
        n = len(data.columns)
        return AlgorithmResult(
            algorithm=self._algo_type,
            adjacency_matrix=np.zeros((n, n), dtype=int),
            edge_list=self._edges,
            runtime_seconds=0.0,
            converged=True,
        )


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(size=40) for c in ("a", "b", "c", "d")})


class TestRunnerBootstrapStability:
    @pytest.mark.asyncio
    async def test_stability_measured_and_confidence_overwritten(self) -> None:
        runner = DiscoveryRunner(enable_tracing=False)
        algo = _ScriptedAlgorithm()
        runner._algorithms[DiscoveryAlgorithmType.PC] = algo
        config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC], bootstrap_resamples=10)
        result = await runner.discover_dag(_frame(), config)

        by_edge = {(e.source, e.target): e for e in result.edges}
        assert by_edge[("a", "b")].bootstrap_stability == 1.0
        assert by_edge[("a", "b")].confidence == 1.0
        assert by_edge[("c", "d")].bootstrap_stability == 0.5
        assert by_edge[("c", "d")].confidence == 0.5
        assert result.metadata["bootstrap"] == {"n_resamples": 10, "n_succeeded": 10}
        assert algo.calls == 11  # primary + B resamples

    @pytest.mark.asyncio
    async def test_bootstrap_off_by_default_leaves_confidence_alone(self) -> None:
        runner = DiscoveryRunner(enable_tracing=False)
        runner._algorithms[DiscoveryAlgorithmType.PC] = _ScriptedAlgorithm()
        config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC])
        result = await runner.discover_dag(_frame(), config)
        assert all(e.bootstrap_stability is None for e in result.edges)
        assert all(e.confidence == 1.0 for e in result.edges)
        assert "bootstrap" not in result.metadata

    @pytest.mark.asyncio
    async def test_too_many_failed_resamples_leaves_stability_unknown(self) -> None:
        runner = DiscoveryRunner(enable_tracing=False)
        runner._algorithms[DiscoveryAlgorithmType.PC] = _Flaky()
        config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC], bootstrap_resamples=10)
        result = await runner.discover_dag(_frame(), config)
        assert all(e.bootstrap_stability is None for e in result.edges)
        assert all(e.confidence == 1.0 for e in result.edges)
        assert result.metadata["bootstrap"] is None

    @pytest.mark.asyncio
    async def test_stability_denominator_is_succeeded_not_total_resamples(self) -> None:
        """Of 10 resample calls: 2 raise, 1 returns converged=False, and of
        the 7 that succeed, 4 recover a->b. Stability must be 4/7 (over the
        SUCCEEDED resamples) — flipping the denominator to n_resamples would
        give 0.4 instead of ~0.571, so this fails loudly if runner.py divides
        by the wrong count."""
        runner = DiscoveryRunner(enable_tracing=False)
        algo = _PartialFailureAlgorithm()
        runner._algorithms[DiscoveryAlgorithmType.PC] = algo
        config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC], bootstrap_resamples=10)
        result = await runner.discover_dag(_frame(), config)

        by_edge = {(e.source, e.target): e for e in result.edges}
        assert by_edge[("a", "b")].bootstrap_stability == pytest.approx(4 / 7)
        assert by_edge[("a", "b")].confidence == pytest.approx(4 / 7)
        assert result.metadata["bootstrap"] == {"n_resamples": 10, "n_succeeded": 7}

    @pytest.mark.asyncio
    async def test_bootstrap_skipped_when_multiple_algorithms_converge(self) -> None:
        """Agreement across >=2 converged algorithms is already a corroboration
        signal, so bootstrap must not run — and, more precisely, the scripted
        doubles must never be re-invoked for resampling at all."""
        runner = DiscoveryRunner(enable_tracing=False)
        pc = _FixedAlgorithm(DiscoveryAlgorithmType.PC, [("a", "b")])
        ges = _FixedAlgorithm(DiscoveryAlgorithmType.GES, [("a", "b")])
        runner._algorithms[DiscoveryAlgorithmType.PC] = pc
        runner._algorithms[DiscoveryAlgorithmType.GES] = ges
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC, DiscoveryAlgorithmType.GES],
            bootstrap_resamples=10,
        )
        result = await runner.discover_dag(_frame(), config)

        assert "bootstrap" not in result.metadata
        assert all(e.bootstrap_stability is None for e in result.edges)
        by_edge = {(e.source, e.target): e for e in result.edges}
        # Both algorithms agree on a->b: untouched ensemble-vote confidence.
        assert by_edge[("a", "b")].confidence == 1.0
        # Neither double was called beyond its one primary discovery run.
        assert pc.calls == 1
        assert ges.calls == 1
