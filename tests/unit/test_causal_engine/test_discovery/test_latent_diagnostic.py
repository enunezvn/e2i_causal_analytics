"""Latent-confounding diagnostic (FCI): config plumbing, runner orchestration,
gate metadata carry, and graph_builder flag annotation.

Fix 3 of the causal-DAG grading sequence. Guided discovery runs PC, which
assumes causal sufficiency — nothing in the pipeline could NOTICE latent
confounding. The diagnostic runs FCI unguided alongside discovery and reports
its bidirected-edge testimony as structured metadata plus a warning flag. It
never changes gate decisions (fix 2's accept/reject calibration is preserved).

The measured operating point behind the flag predicate lives in
``test_structural_recovery.py`` (module docstring item 6): the estimand-pair
bidirected mark is the only location with true-positive signal on this
platform's binary-logit frames; covariate-level bidirected pairs are reported
in the payload but do not raise the flag.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.algorithms.fci_wrapper import FCIAlgorithm
from src.causal_engine.discovery.base import (
    AlgorithmResult,
    BaseDiscoveryAlgorithm,
    CausalPriorKnowledge,
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
)
from src.causal_engine.discovery.cache import DiscoveryCache
from src.causal_engine.discovery.gate import DiscoveryGate
from src.causal_engine.discovery.hasher import hash_config
from src.causal_engine.discovery.runner import DiscoveryRunner


class TestLatentDiagnosticConfig:
    def test_config_defaults_diagnostic_off_and_serializes(self) -> None:
        config = DiscoveryConfig()
        assert config.latent_diagnostic is False
        assert config.to_dict()["latent_diagnostic"] is False

    def test_latent_diagnostic_changes_config_hash(self) -> None:
        assert hash_config(DiscoveryConfig()) != hash_config(
            DiscoveryConfig(latent_diagnostic=True)
        )

    def test_cache_round_trips_latent_diagnostic_config_and_payload(self) -> None:
        cache = DiscoveryCache()
        payload = json.dumps(
            {
                "success": True,
                "config": DiscoveryConfig(latent_diagnostic=True, bootstrap_resamples=20).to_dict(),
                "edges": [],
                "gate_decision": None,
                "gate_confidence": 0.0,
                "created_at": "2026-09-02T00:00:00+00:00",
                "session_id": None,
                "metadata": {
                    "latent_diagnostic": {
                        "ran": True,
                        "converged": True,
                        "runtime_seconds": 0.03,
                        "bidirected_edges": [["treatment_arm", "persistent_180d"]],
                    }
                },
            }
        )
        restored = cache._deserialize_result(payload)
        assert restored is not None
        assert restored.config.latent_diagnostic is True
        # Same field-by-field reconstruction trap as latent_diagnostic (the
        # deserializer enumerates config fields manually): bootstrap_resamples
        # was hashed but never restored, silently reverting to 0.
        assert restored.config.bootstrap_resamples == 20
        assert restored.metadata["latent_diagnostic"]["bidirected_edges"] == [
            ["treatment_arm", "persistent_180d"]
        ]


class _PrimaryPC(BaseDiscoveryAlgorithm):
    """Stands in for guided PC as the primary discovery algorithm."""

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        return DiscoveryAlgorithmType.PC

    def supports_latent_confounders(self) -> bool:
        return False

    def discover(self, data: pd.DataFrame, config: DiscoveryConfig) -> AlgorithmResult:
        n = len(data.columns)
        return AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.PC,
            adjacency_matrix=np.zeros((n, n), dtype=int),
            edge_list=[("t", "y")],
            runtime_seconds=0.0,
            converged=True,
        )


class _ScriptedFCI(FCIAlgorithm):
    """FCIAlgorithm whose discover() is replaced by a scripted result, keeping
    the real ``get_bidirected_edges`` name mapping in the loop. Captures the
    config it receives so the priors-stripping contract is assertable."""

    def __init__(self, metadata: dict, converged: bool = True) -> None:
        self._metadata = metadata
        self._converged = converged
        self.seen_configs: list = []

    def discover(self, data: pd.DataFrame, config: DiscoveryConfig) -> AlgorithmResult:
        self.seen_configs.append(config)
        n = len(data.columns)
        return AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.FCI,
            adjacency_matrix=np.zeros((n, n), dtype=int),
            edge_list=[],
            runtime_seconds=0.01,
            converged=self._converged,
            metadata=self._metadata,
        )


class _ExplodingFCI(FCIAlgorithm):
    def discover(self, data: pd.DataFrame, config: DiscoveryConfig) -> AlgorithmResult:
        raise RuntimeError("fci blew up")


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(size=40) for c in ("t", "y", "c")})


_INDEX_KEYED_BIDIRECTED = {
    # Real discover() keys edge_types by integer node indices; node_names is
    # the mapping the extraction must apply.
    "edge_types": {"0->1": "bidirected", "0->2": "directed"},
    "node_names": ["t", "y", "c"],
}


class TestRunnerLatentDiagnostic:
    @pytest.mark.asyncio
    async def test_off_by_default_runs_no_fci(self) -> None:
        runner = DiscoveryRunner(enable_tracing=False)
        runner._algorithms[DiscoveryAlgorithmType.PC] = _PrimaryPC()
        fci = _ScriptedFCI(_INDEX_KEYED_BIDIRECTED)
        runner._algorithms[DiscoveryAlgorithmType.FCI] = fci
        config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC])
        result = await runner.discover_dag(_frame(), config)
        assert "latent_diagnostic" not in result.metadata
        assert fci.seen_configs == []

    @pytest.mark.asyncio
    async def test_on_reports_bidirected_pairs_by_column_name(self) -> None:
        runner = DiscoveryRunner(enable_tracing=False)
        runner._algorithms[DiscoveryAlgorithmType.PC] = _PrimaryPC()
        runner._algorithms[DiscoveryAlgorithmType.FCI] = _ScriptedFCI(_INDEX_KEYED_BIDIRECTED)
        config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC], latent_diagnostic=True)
        result = await runner.discover_dag(_frame(), config)
        payload = result.metadata["latent_diagnostic"]
        assert payload["ran"] is True
        assert payload["converged"] is True
        assert payload["bidirected_edges"] == [["t", "y"]]
        assert payload["runtime_seconds"] == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_diagnostic_run_strips_priors_and_bootstrap(self) -> None:
        """Constraint: the diagnostic is the data's OWN testimony — no guided
        priors — and must not multiply FCI's runtime by the resample count."""
        runner = DiscoveryRunner(enable_tracing=False)
        runner._algorithms[DiscoveryAlgorithmType.PC] = _PrimaryPC()
        fci = _ScriptedFCI(_INDEX_KEYED_BIDIRECTED)
        runner._algorithms[DiscoveryAlgorithmType.FCI] = fci
        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC],
            latent_diagnostic=True,
            alpha=0.01,
            prior_knowledge=CausalPriorKnowledge(required_edges=[("t", "y")]),
        )
        await runner.discover_dag(_frame(), config)
        assert len(fci.seen_configs) == 1
        seen = fci.seen_configs[0]
        assert seen.prior_knowledge is None
        assert seen.bootstrap_resamples == 0
        assert seen.algorithms == [DiscoveryAlgorithmType.FCI]
        assert seen.alpha == 0.01  # data-relevant knobs pass through

    @pytest.mark.asyncio
    async def test_fci_failure_degrades_to_not_run_without_failing_discovery(self) -> None:
        runner = DiscoveryRunner(enable_tracing=False)
        runner._algorithms[DiscoveryAlgorithmType.PC] = _PrimaryPC()
        runner._algorithms[DiscoveryAlgorithmType.FCI] = _ExplodingFCI()
        config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC], latent_diagnostic=True)
        result = await runner.discover_dag(_frame(), config)
        assert result.success is True
        assert result.edges, "primary discovery must be unaffected"
        payload = result.metadata["latent_diagnostic"]
        assert payload["ran"] is False
        assert "fci blew up" in payload["error"]

    @pytest.mark.asyncio
    async def test_unconverged_fci_reports_converged_false_and_no_pairs(self) -> None:
        runner = DiscoveryRunner(enable_tracing=False)
        runner._algorithms[DiscoveryAlgorithmType.PC] = _PrimaryPC()
        runner._algorithms[DiscoveryAlgorithmType.FCI] = _ScriptedFCI(
            {"error": "did not converge"}, converged=False
        )
        config = DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC], latent_diagnostic=True)
        result = await runner.discover_dag(_frame(), config)
        payload = result.metadata["latent_diagnostic"]
        assert payload["ran"] is True
        assert payload["converged"] is False
        assert payload["bidirected_edges"] == []


def _result_with_payload(payload) -> DiscoveryResult:
    import networkx as nx

    dag = nx.DiGraph()
    dag.add_edge("t", "y")
    metadata = {} if payload is None else {"latent_diagnostic": payload}
    return DiscoveryResult(
        success=True,
        config=DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC]),
        ensemble_dag=dag,
        edges=[DiscoveredEdge(source="t", target="y", confidence=1.0, algorithms=["pc"])],
        algorithm_results=[
            AlgorithmResult(
                algorithm=DiscoveryAlgorithmType.PC,
                adjacency_matrix=np.zeros((2, 2), dtype=int),
                edge_list=[("t", "y")],
                runtime_seconds=0.0,
                converged=True,
            )
        ],
        metadata=metadata,
    )


class TestGateCarriesDiagnosticWithoutGatingOnIt:
    PAYLOAD = {
        "ran": True,
        "converged": True,
        "runtime_seconds": 0.03,
        "bidirected_edges": [["t", "y"]],
        "flag": True,
    }

    def test_evaluation_metadata_carries_payload(self) -> None:
        evaluation = DiscoveryGate().evaluate(_result_with_payload(self.PAYLOAD))
        assert evaluation.metadata["latent_diagnostic"] == self.PAYLOAD
        # to_dict is the serialization graph_builder consumes; the payload must
        # survive it (fix 2's AUGMENT was inert for a day on exactly this trap).
        assert evaluation.to_dict()["metadata"]["latent_diagnostic"] == self.PAYLOAD

    def test_decision_and_confidence_identical_with_and_without_payload(self) -> None:
        """Constraint 1: the diagnostic annotates, it never gates. Fix 2's
        accept/reject calibration must be untouched by its presence."""
        with_payload = DiscoveryGate().evaluate(_result_with_payload(self.PAYLOAD))
        without = DiscoveryGate().evaluate(_result_with_payload(None))
        assert with_payload.decision == without.decision
        assert with_payload.confidence == without.confidence

    def test_min_edges_reject_path_still_carries_payload(self) -> None:
        result = _result_with_payload(self.PAYLOAD)
        result.edges = []
        evaluation = DiscoveryGate().evaluate(result)
        assert evaluation.metadata["latent_diagnostic"] == self.PAYLOAD


class TestGraphBuilderFlagAnnotation:
    @staticmethod
    def _annotate(payload, treatment="t", outcome="y"):
        from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode

        result = _result_with_payload(payload)
        GraphBuilderNode._annotate_latent_diagnostic(result, treatment, outcome)
        return result.metadata.get("latent_diagnostic")

    def test_estimand_pair_bidirected_raises_flag_either_order(self) -> None:
        for pair in (["t", "y"], ["y", "t"]):
            payload = self._annotate({"ran": True, "converged": True, "bidirected_edges": [pair]})
            assert payload["flag"] is True
            assert payload["treatment"] == "t"
            assert payload["outcome"] == "y"

    def test_covariate_pairs_do_not_raise_flag(self) -> None:
        """Measured: covariate-level bidirected pairs occur on observed-
        confounder and noise controls (false alarms); only the estimand-pair
        mark carries true-positive signal. The pairs stay in the payload."""
        payload = self._annotate(
            {"ran": True, "converged": True, "bidirected_edges": [["y", "c"], ["c", "d"]]}
        )
        assert payload["flag"] is False
        assert payload["bidirected_edges"] == [["y", "c"], ["c", "d"]]

    def test_not_run_payload_gets_no_flag(self) -> None:
        payload = self._annotate({"ran": False, "error": "boom"})
        assert payload["flag"] is False

    def test_absent_payload_is_left_absent(self) -> None:
        assert self._annotate(None) is None


class TestConfigReconstructionContract:
    """One shared ``DiscoveryConfig.from_dict`` behind every manual
    field-by-field reconstruction (cache deserializer, process-pool worker):
    the fix-2/fix-3 trap class is a field added to the dataclass but not to a
    hand-enumerated copy, silently reverting to its default."""

    def _guided_config(self) -> DiscoveryConfig:
        return DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC],
            alpha=0.01,
            bootstrap_resamples=20,
            latent_diagnostic=True,
            prior_knowledge=CausalPriorKnowledge(
                tiers=[["c"], ["t"], ["y"]],
                required_edges=[("t", "y"), ("c", "y")],
                forbidden_edges=[("y", "t")],
            ),
        )

    def test_from_dict_round_trips_every_field(self) -> None:
        config = self._guided_config()
        restored = DiscoveryConfig.from_dict(config.to_dict())
        assert restored.to_dict() == config.to_dict()
        assert restored.prior_knowledge is not None
        assert restored.prior_knowledge.required_edges == [("t", "y"), ("c", "y")]
        assert restored.prior_knowledge.forbidden_edges == [("y", "t")]
        assert restored.prior_knowledge.tiers == [["c"], ["t"], ["y"]]

    def test_cache_round_trip_preserves_gate_corroboration_basis(self) -> None:
        """A guided result whose every edge is prior-required must evaluate as
        'prior_determined' AFTER a cache round-trip too — losing
        prior_knowledge in deserialization would silently re-basis the gate
        to 'uncorroborated_single_run' (score 0.0) on every cache hit."""
        import networkx as nx

        config = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC],
            prior_knowledge=CausalPriorKnowledge(required_edges=[("t", "y")]),
        )
        dag = nx.DiGraph()
        dag.add_edge("t", "y")
        result = DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=dag,
            edges=[DiscoveredEdge(source="t", target="y", confidence=1.0, algorithms=["pc"])],
            algorithm_results=[
                AlgorithmResult(
                    algorithm=DiscoveryAlgorithmType.PC,
                    adjacency_matrix=np.zeros((2, 2), dtype=int),
                    edge_list=[("t", "y")],
                    runtime_seconds=0.0,
                    converged=True,
                )
            ],
        )
        before = DiscoveryGate().evaluate(result)
        assert before.metadata["corroboration_basis"] == "prior_determined"

        cache = DiscoveryCache()
        restored = cache._deserialize_result(cache._serialize_result(result))
        assert restored is not None
        # The deserialized result drops algorithm_results (not serialized), so
        # re-evaluating the full gate is not faithful — but the config's prior
        # must survive, which is what the basis computation reads.
        assert restored.config.prior_knowledge is not None
        assert restored.config.prior_knowledge.required_edges == [("t", "y")]

    def test_process_pool_worker_receives_the_full_config(self) -> None:
        from src.causal_engine.discovery.runner import _run_algorithm_in_process

        config = self._guided_config()
        frame = _frame()
        result_dict = _run_algorithm_in_process(
            _ConfigEchoAlgorithm, frame.to_dict(), config.to_dict()
        )
        seen = result_dict["metadata"]
        assert seen["prior_required_edges"] == [("t", "y"), ("c", "y")]
        assert seen["bootstrap_resamples"] == 20
        assert seen["latent_diagnostic"] is True
        assert seen["alpha"] == 0.01


class _ConfigEchoAlgorithm(BaseDiscoveryAlgorithm):
    """Echoes the config it received back through result metadata, so tests
    can assert what a reconstruction actually delivered."""

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        return DiscoveryAlgorithmType.PC

    def supports_latent_confounders(self) -> bool:
        return False

    def discover(self, data: pd.DataFrame, config: DiscoveryConfig) -> AlgorithmResult:
        n = len(data.columns)
        prior = config.prior_knowledge
        return AlgorithmResult(
            algorithm=DiscoveryAlgorithmType.PC,
            adjacency_matrix=np.zeros((n, n), dtype=int),
            edge_list=[],
            runtime_seconds=0.0,
            converged=True,
            metadata={
                "prior_required_edges": list(prior.required_edges) if prior else None,
                "bootstrap_resamples": config.bootstrap_resamples,
                "latent_diagnostic": config.latent_diagnostic,
                "alpha": config.alpha,
            },
        )
