"""P12 (MED) — ensemble edge confidence must be relative to CONVERGED algorithms.

runner._build_ensemble divided edge votes by the TOTAL algorithm count
(including failed/non-converged ones), deflating every edge's confidence whenever
an algorithm failed. It must divide by the number that actually converged.
"""

from __future__ import annotations

import numpy as np

from src.causal_engine.discovery.base import AlgorithmResult, DiscoveryAlgorithmType
from src.causal_engine.discovery.runner import DiscoveryRunner


def _result(algorithm, edges, converged):
    return AlgorithmResult(
        algorithm=algorithm,
        adjacency_matrix=np.zeros((2, 2), dtype=int),
        edge_list=list(edges),
        runtime_seconds=0.01,
        converged=converged,
        metadata={},
    )


class TestEnsembleConfidenceUsesConvergedCount:
    def test_failed_algorithms_do_not_deflate_confidence(self):
        runner = DiscoveryRunner()
        results = [
            _result(DiscoveryAlgorithmType.PC, [("X", "Y")], converged=True),
            _result(DiscoveryAlgorithmType.GES, [("X", "Y")], converged=True),
            _result(DiscoveryAlgorithmType.FCI, [], converged=False),  # failed
            _result(DiscoveryAlgorithmType.DIRECT_LINGAM, [], converged=False),  # failed
        ]
        edges, _graph = runner._build_ensemble(results, ["X", "Y"], threshold=0.5)
        by_edge = {(e.source, e.target): e for e in edges}
        assert ("X", "Y") in by_edge, "an edge both converged algorithms agree on must survive"
        # 2 of 2 CONVERGED algorithms agree → confidence 1.0, NOT 2/4 = 0.5.
        assert by_edge[("X", "Y")].confidence == 1.0

    def test_all_failed_returns_empty(self):
        runner = DiscoveryRunner()
        results = [
            _result(DiscoveryAlgorithmType.PC, [("X", "Y")], converged=False),
            _result(DiscoveryAlgorithmType.GES, [("X", "Y")], converged=False),
        ]
        edges, _graph = runner._build_ensemble(results, ["X", "Y"], threshold=0.5)
        assert edges == []
