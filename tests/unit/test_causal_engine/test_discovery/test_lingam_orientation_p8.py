"""P8 / H7 — LiNGAM edge orientation + driver-rank propagation.

Finding H7: the lingam library defines ``adjacency_matrix_`` as B where
x_i = Σ_j B[i,j]·x_j, so a nonzero B[i,j] means edge j → i. The wrapper fed B
UNTRANSPOSED into ``_adjacency_to_edge_list`` (which reads A[i,j]≠0 as i → j),
so every DirectLiNGAM / ICA-LiNGAM edge came back REVERSED, propagating into
driver_ranker's directed-path rankings.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_lingam_module():
    mock_module = types.ModuleType("lingam")
    mock_module.DirectLiNGAM = MagicMock
    mock_module.ICALiNGAM = MagicMock
    original = sys.modules.get("lingam")
    sys.modules["lingam"] = mock_module
    yield mock_module
    if original is not None:
        sys.modules["lingam"] = original
    else:
        sys.modules.pop("lingam", None)


from src.causal_engine.discovery.algorithms.lingam_wrapper import (  # noqa: E402
    DirectLiNGAMAlgorithm,
    ICALiNGAMAlgorithm,
)
from src.causal_engine.discovery.base import DiscoveryConfig  # noqa: E402
from src.causal_engine.discovery.driver_ranker import DriverRanker  # noqa: E402


def _xy_data():
    rng = np.random.RandomState(0)
    x = rng.normal(0, 1, 50)
    y = 0.8 * x + rng.normal(0, 0.3, 50)
    return pd.DataFrame({"X": x, "Y": y})


class TestLiNGAMOrientation:
    def test_direct_lingam_x_to_y_edge_not_reversed(self, mock_lingam_module):
        """True X → Y must come back as edge (X, Y), not (Y, X)."""
        algo = DirectLiNGAMAlgorithm()
        config = DiscoveryConfig(random_state=42)
        with patch("lingam.DirectLiNGAM") as MockLiNGAM:
            mock_model = MagicMock()
            # CORRECT lingam B for X→Y (Y=0.8·X): x_1 = 0.8·x_0 → B[1,0]=0.8.
            mock_model.adjacency_matrix_ = np.array([[0.0, 0.0], [0.8, 0.0]])
            mock_model.causal_order_ = [0, 1]
            MockLiNGAM.return_value = mock_model
            result = algo.discover(_xy_data(), config)
        assert ("X", "Y") in result.edge_list, f"expected X→Y, got {result.edge_list}"
        assert ("Y", "X") not in result.edge_list

    def test_ica_lingam_x_to_y_edge_not_reversed(self, mock_lingam_module):
        algo = ICALiNGAMAlgorithm()
        config = DiscoveryConfig(random_state=42)
        with patch("lingam.ICALiNGAM") as MockLiNGAM:
            mock_model = MagicMock()
            mock_model.adjacency_matrix_ = np.array([[0.0, 0.0], [0.8, 0.0]])
            mock_model.causal_order_ = [0, 1]
            MockLiNGAM.return_value = mock_model
            result = algo.discover(_xy_data(), config)
        assert ("X", "Y") in result.edge_list, f"expected X→Y, got {result.edge_list}"
        assert ("Y", "X") not in result.edge_list


class TestDriverRankPropagation:
    def test_true_cause_ranks_above_spurious_with_correct_dag(self):
        """With the corrected DAG X→Y→Z, a true cause outranks a spurious node.

        Under the reversed orientation (Z→Y→X) the true cause would have NO
        directed path to the target and would be mis-ranked.
        """
        dag = nx.DiGraph()
        dag.add_edges_from([("X", "Y"), ("Y", "Z")])  # correct orientation
        dag.add_node("S")  # spurious associate, no path to Z
        ranker = DriverRanker()
        feature_names = ["X", "Y", "S"]
        shap_values = np.zeros((20, 3))  # isolate the causal contribution
        result = ranker.rank_drivers(dag, "Z", shap_values, feature_names)

        scores = {r.feature_name: r for r in result.rankings}
        # X and Y are causal ancestors of Z; S is not.
        assert scores["X"].causal_score > scores["S"].causal_score
        assert scores["Y"].causal_score > scores["S"].causal_score
        assert scores["X"].path_length is not None  # X→Y→Z reachable
        assert scores["S"].path_length is None  # no path
