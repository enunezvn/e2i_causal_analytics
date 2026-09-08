"""Tests for DiscoveryRunner class.

Version: 1.0.0
Tests the causal discovery runner with ensemble algorithms.
"""

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
)
from src.causal_engine.discovery.runner import DiscoveryRunner


class TestDiscoveryRunnerInit:
    """Test DiscoveryRunner initialization."""

    def test_default_init(self):
        """Test default initialization."""
        runner = DiscoveryRunner()

        assert runner.max_workers == 4
        assert runner.timeout_seconds == 300.0
        assert runner._algorithms == {}

    def test_custom_init(self):
        """Test custom initialization."""
        runner = DiscoveryRunner(max_workers=2, timeout_seconds=60.0)

        assert runner.max_workers == 2
        assert runner.timeout_seconds == 60.0


class TestGetAlgorithm:
    """Test algorithm retrieval."""

    def test_get_registered_algorithm(self):
        """Test getting a registered algorithm."""
        runner = DiscoveryRunner()

        algo = runner._get_algorithm(DiscoveryAlgorithmType.GES)
        assert algo is not None

        # Should be cached
        algo2 = runner._get_algorithm(DiscoveryAlgorithmType.GES)
        assert algo is algo2

    def test_get_pc_algorithm(self):
        """Test getting PC algorithm."""
        runner = DiscoveryRunner()

        algo = runner._get_algorithm(DiscoveryAlgorithmType.PC)
        assert algo is not None

    def test_get_fci_algorithm(self):
        """Test getting FCI algorithm."""
        runner = DiscoveryRunner()

        algo = runner._get_algorithm(DiscoveryAlgorithmType.FCI)
        assert algo is not None
        assert algo.supports_latent_confounders() is True

    def test_get_direct_lingam_algorithm(self):
        """Test getting DirectLiNGAM algorithm."""
        runner = DiscoveryRunner()

        algo = runner._get_algorithm(DiscoveryAlgorithmType.DIRECT_LINGAM)
        assert algo is not None
        assert algo.supports_latent_confounders() is False

    def test_get_ica_lingam_algorithm(self):
        """Test getting ICA-LiNGAM algorithm."""
        runner = DiscoveryRunner()

        algo = runner._get_algorithm(DiscoveryAlgorithmType.ICA_LINGAM)
        assert algo is not None
        assert algo.supports_latent_confounders() is False


class TestGetAvailableAlgorithms:
    """Test get_available_algorithms class method."""

    def test_returns_registered_algorithms(self):
        """Test that it returns list of registered algorithms."""
        algorithms = DiscoveryRunner.get_available_algorithms()

        assert DiscoveryAlgorithmType.GES in algorithms
        assert DiscoveryAlgorithmType.PC in algorithms
        assert len(algorithms) >= 2


class TestBuildEnsemble:
    """Test ensemble building logic."""

    @pytest.fixture
    def runner(self):
        """Create runner instance."""
        return DiscoveryRunner()

    def test_empty_results(self, runner):
        """Test with no algorithm results."""
        edges, dag = runner._build_ensemble([], ["A", "B", "C"], 0.5)

        assert edges == []
        assert dag.number_of_nodes() == 0

    def test_single_algorithm(self, runner):
        """Test with single algorithm result."""
        results = [
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.GES,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("A", "B"), ("B", "C")],
                runtime_seconds=1.0,
                converged=True,
            )
        ]

        edges, dag = runner._build_ensemble(results, ["A", "B", "C"], 0.5)

        # With threshold 0.5 and 1 algorithm, min_votes = 1
        assert len(edges) == 2
        assert dag.has_edge("A", "B")
        assert dag.has_edge("B", "C")

    def test_multiple_algorithms_voting(self, runner):
        """Test edge voting with multiple algorithms."""
        results = [
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.GES,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("A", "B"), ("B", "C")],
                runtime_seconds=1.0,
                converged=True,
            ),
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("A", "B"), ("A", "C")],  # Different edges
                runtime_seconds=1.0,
                converged=True,
            ),
        ]

        # With threshold 0.5 and 2 algorithms, min_votes = 1
        edges, dag = runner._build_ensemble(results, ["A", "B", "C"], 0.5)

        # A->B has 2 votes (both agree), B->C has 1 vote, A->C has 1 vote
        edge_dict = {(e.source, e.target): e for e in edges}

        assert ("A", "B") in edge_dict
        assert edge_dict[("A", "B")].algorithm_votes == 2
        assert edge_dict[("A", "B")].confidence == 1.0  # 2/2

    def test_threshold_filtering(self, runner):
        """Test that threshold filters low-vote edges."""
        results = [
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.GES,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("A", "B")],
                runtime_seconds=1.0,
                converged=True,
            ),
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("B", "C")],  # Different edges
                runtime_seconds=1.0,
                converged=True,
            ),
        ]

        # With threshold 1.0, need all algorithms to agree
        edges, dag = runner._build_ensemble(results, ["A", "B", "C"], 1.0)

        # No edges have unanimous agreement
        # Actually, with 2 algos and threshold 1.0, min_votes = max(1, int(2*1.0)) = 2
        assert len(edges) == 0

    def test_unconverged_algorithm_skipped(self, runner):
        """Test that unconverged algorithm results are skipped."""
        results = [
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.GES,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("A", "B")],
                runtime_seconds=1.0,
                converged=True,
            ),
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((3, 3)),
                edge_list=[("B", "C"), ("C", "D")],
                runtime_seconds=0.0,
                converged=False,  # Did not converge
            ),
        ]

        edges, dag = runner._build_ensemble(results, ["A", "B", "C", "D"], 0.5)

        # Only GES edges should be counted
        edge_sources = [(e.source, e.target) for e in edges]
        assert ("A", "B") in edge_sources
        # PC edges should not be included since it didn't converge


class TestRemoveCycles:
    """Test cycle removal logic."""

    @pytest.fixture
    def runner(self):
        """Create runner instance."""
        return DiscoveryRunner()

    def test_no_cycles(self, runner):
        """Test DAG without cycles remains unchanged."""
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C")])
        dag.edges["A", "B"]["confidence"] = 0.9
        dag.edges["B", "C"]["confidence"] = 0.8

        result = runner._remove_cycles(dag)

        assert result.has_edge("A", "B")
        assert result.has_edge("B", "C")

    def test_simple_cycle_removed(self, runner):
        """Test simple cycle is broken."""
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        dag.edges["A", "B"]["confidence"] = 0.9
        dag.edges["B", "C"]["confidence"] = 0.8
        dag.edges["C", "A"]["confidence"] = 0.5  # Lowest confidence

        result = runner._remove_cycles(dag)

        # Should remove C->A (lowest confidence)
        assert result.has_edge("A", "B")
        assert result.has_edge("B", "C")
        assert not result.has_edge("C", "A")

    def test_multiple_cycles(self, runner):
        """Test multiple cycles are resolved."""
        dag = nx.DiGraph()
        dag.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("C", "A"),  # Cycle 1
                ("D", "E"),
                ("E", "D"),  # Cycle 2
            ]
        )
        for u, v in dag.edges():
            dag.edges[u, v]["confidence"] = 0.5

        result = runner._remove_cycles(dag)

        # Should be acyclic now
        assert nx.is_directed_acyclic_graph(result)


class TestDiscoverDagSync:
    """Test synchronous discover_dag method."""

    def test_discover_dag_sync_calls_async(self):
        """Test that sync version calls async version."""
        runner = DiscoveryRunner()

        # Create simple test data
        data = pd.DataFrame(
            {
                "A": np.random.randn(100),
                "B": np.random.randn(100),
                "C": np.random.randn(100),
            }
        )

        # Mock the async discover_dag
        expected_result = DiscoveryResult(
            success=True,
            config=DiscoveryConfig(),
            edges=[],
        )

        with patch.object(runner, "discover_dag", new_callable=AsyncMock) as mock_discover:
            mock_discover.return_value = expected_result

            result = runner.discover_dag_sync(data)

            assert result == expected_result
            mock_discover.assert_called_once()


class TestDiscoverDagIntegration:
    """Integration tests for discover_dag (requires causal-learn)."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data with known structure: A -> B -> C."""
        np.random.seed(42)
        n = 200

        # A causes B causes C
        a = np.random.randn(n)
        b = 0.8 * a + 0.2 * np.random.randn(n)
        c = 0.7 * b + 0.3 * np.random.randn(n)

        return pd.DataFrame({"A": a, "B": b, "C": c})

    @pytest.mark.asyncio
    async def test_discover_dag_basic(self, synthetic_data):
        """Test basic discovery with synthetic data."""
        runner = DiscoveryRunner()
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.GES],
            alpha=0.05,
        )

        result = await runner.discover_dag(synthetic_data, config)

        assert result.success is True
        assert result.ensemble_dag is not None
        assert len(result.algorithm_results) == 1
        assert "total_runtime_seconds" in result.metadata

    @pytest.mark.asyncio
    async def test_discover_dag_with_session_id(self, synthetic_data):
        """Test discovery with session ID."""
        runner = DiscoveryRunner()
        session_id = uuid4()

        result = await runner.discover_dag(
            synthetic_data,
            DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.GES]),
            session_id=session_id,
        )

        assert result.session_id == session_id

    @pytest.mark.asyncio
    async def test_discover_dag_ensemble(self, synthetic_data):
        """Test discovery with multiple algorithms."""
        runner = DiscoveryRunner()
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC],
            ensemble_threshold=0.5,
        )

        result = await runner.discover_dag(synthetic_data, config)

        assert result.success is True
        assert len(result.algorithm_results) == 2

        # Check metadata
        assert result.metadata["n_samples"] == len(synthetic_data)
        assert result.metadata["node_names"] == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_discover_dag_default_config(self, synthetic_data):
        """Test discovery with default config."""
        runner = DiscoveryRunner()

        result = await runner.discover_dag(synthetic_data)

        assert result.success is True
        assert result.config is not None


class TestRegisterAlgorithm:
    """Test custom algorithm registration."""

    def test_register_custom_algorithm(self):
        """Test registering a custom algorithm."""

        # Create mock algorithm class
        class MockAlgorithm:
            def discover(self, data, config):
                return AlgorithmResult(
                    algorithm=DiscoveryAlgorithmType.LINGAM,
                    adjacency_matrix=np.zeros((3, 3)),
                    edge_list=[],
                    runtime_seconds=0.1,
                )

        # Register
        DiscoveryRunner.register_algorithm(
            DiscoveryAlgorithmType.LINGAM,
            MockAlgorithm,
        )

        # Verify it's registered
        assert DiscoveryAlgorithmType.LINGAM in DiscoveryRunner.ALGORITHM_REGISTRY

        # Clean up
        del DiscoveryRunner.ALGORITHM_REGISTRY[DiscoveryAlgorithmType.LINGAM]


class TestPoolWorkerFallback:
    """#1978: the pool path must fall back to the runner's own ``max_workers``.

    The timeout half of #1978 is pinned by TestPerAlgorithmTimeout; this is the
    other half, which shipped untested. ``max_workers`` was the *other* attribute
    the issue named as "stored but never read", and the fix
    (``config.max_workers or self.max_workers``) is exactly the kind of one-line
    expression that silently reverts.

    The real ``ProcessPoolExecutor`` is deliberately NOT spawned: forking a pool
    that re-imports causal-learn on a memory-capped box is a genuine OOM risk,
    and the behaviour under test is which value reaches the constructor, not what
    the pool computes. The recorder captures the real call from the real code
    path and then aborts, so nothing is faked -- no substitute result is ever
    returned to the code under test.
    """

    @staticmethod
    async def _record_max_workers(runner, config):
        seen = {}

        class _Abort(RuntimeError):
            pass

        class _Recorder:
            def __init__(self, max_workers=None):
                seen["max_workers"] = max_workers
                raise _Abort

        data = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        with patch("src.causal_engine.discovery.runner.ProcessPoolExecutor", _Recorder):
            try:
                await runner._run_algorithms_parallel(data, config)
            except _Abort:
                pass
        return seen

    @pytest.mark.asyncio
    async def test_runner_max_workers_is_used_when_config_does_not_pin_one(self):
        runner = DiscoveryRunner(max_workers=7)
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC, DiscoveryAlgorithmType.GES],
            use_process_pool=True,
            max_workers=None,
        )
        seen = await self._record_max_workers(runner, config)
        assert seen["max_workers"] == 7, (
            "the runner's max_workers must be the fallback -- it was stored and "
            "never read before #1978"
        )

    @pytest.mark.asyncio
    async def test_config_max_workers_still_wins_when_pinned(self):
        """Negative control: the fallback must not override an explicit config."""
        runner = DiscoveryRunner(max_workers=7)
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC, DiscoveryAlgorithmType.GES],
            use_process_pool=True,
            max_workers=2,
        )
        seen = await self._record_max_workers(runner, config)
        assert seen["max_workers"] == 2


class TestPerAlgorithmTimeout:
    """#1978: ``timeout_seconds`` bounds each algorithm on the sequential path."""

    @pytest.mark.asyncio
    async def test_slow_algorithm_times_out_as_a_failed_run(self):
        import time as _time

        import pandas as pd

        class SlowAlgorithm:
            def discover(self, data, config):
                _time.sleep(0.5)
                return AlgorithmResult(
                    algorithm=DiscoveryAlgorithmType.LINGAM,
                    adjacency_matrix=np.zeros((2, 2), dtype=int),
                    edge_list=[],
                    runtime_seconds=0.5,
                )

        DiscoveryRunner.register_algorithm(DiscoveryAlgorithmType.LINGAM, SlowAlgorithm)
        try:
            runner = DiscoveryRunner(timeout_seconds=0.05, enable_tracing=False)
            data = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 1.0, 4.0, 3.0]})
            config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.LINGAM])
            result = await runner.discover_dag(data, config)
        finally:
            del DiscoveryRunner.ALGORITHM_REGISTRY[DiscoveryAlgorithmType.LINGAM]

        assert len(result.algorithm_results) == 1
        failed = result.algorithm_results[0]
        assert failed.converged is False
        assert "timeout" in failed.metadata["error"]
        # No converged run -> no edges, never a silently empty converged DAG.
        assert result.n_edges == 0
        # Allow the abandoned worker thread to finish before the loop closes.
        await asyncio.sleep(0.6)

    @pytest.mark.asyncio
    async def test_fast_algorithm_is_unaffected_by_the_bound(self):
        import pandas as pd

        class FastAlgorithm:
            def discover(self, data, config):
                return AlgorithmResult(
                    algorithm=DiscoveryAlgorithmType.LINGAM,
                    adjacency_matrix=np.array([[0, 1], [0, 0]], dtype=int),
                    edge_list=[],
                    runtime_seconds=0.001,
                    metadata={"node_names": list(data.columns)},
                )

        DiscoveryRunner.register_algorithm(DiscoveryAlgorithmType.LINGAM, FastAlgorithm)
        try:
            runner = DiscoveryRunner(timeout_seconds=5.0, enable_tracing=False)
            data = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 1.0, 4.0, 3.0]})
            config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.LINGAM])
            result = await runner.discover_dag(data, config)
        finally:
            del DiscoveryRunner.ALGORITHM_REGISTRY[DiscoveryAlgorithmType.LINGAM]

        assert result.algorithm_results[0].converged is True
