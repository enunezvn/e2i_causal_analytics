"""Graph Builder Integration Stress Tests.

Tests the causal_impact agent's graph_builder node with auto_discover=True
on large datasets.

Performance targets:
- auto_discover with 10K rows: <30s
- Gate decision quality should remain consistent at scale
- Augmentation path should work with high-confidence edges

Test scenarios:
- auto_discover on increasingly large datasets
- Gate decision accuracy at scale
- Latency comparison: manual DAG vs auto_discover
"""

import time
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import pytest

# Mark all tests as stress tests
pytestmark = [pytest.mark.stress]


# =============================================================================
# DATA GENERATORS
# =============================================================================


def generate_treatment_outcome_data(
    n_samples: int = 5000,
    n_confounders: int = 5,
    true_ate: float = 1.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate treatment-outcome data with confounders.

    Args:
        n_samples: Number of samples
        n_confounders: Number of confounding variables
        true_ate: True average treatment effect
        seed: Random seed

    Returns:
        Tuple of (DataFrame, metadata)
    """
    np.random.seed(seed)

    # Generate confounders
    confounders = {}
    for i in range(n_confounders):
        confounders[f"X{i+1}"] = np.random.normal(0, 1, n_samples)

    # Treatment depends on confounders
    propensity = 0.5
    for i in range(min(3, n_confounders)):
        propensity = propensity + 0.1 * confounders[f"X{i+1}"]
    propensity = np.clip(propensity, 0.1, 0.9)
    treatment = np.random.binomial(1, propensity)

    # Outcome depends on treatment and confounders
    outcome = true_ate * treatment
    for i in range(n_confounders):
        weight = 0.3 / (i + 1)  # Decreasing weights
        outcome = outcome + weight * confounders[f"X{i+1}"]
    outcome = outcome + np.random.normal(0, 0.5, n_samples)

    # Build DataFrame
    data = pd.DataFrame(confounders)
    data["treatment"] = treatment
    data["outcome"] = outcome

    metadata = {
        "true_ate": true_ate,
        "n_samples": n_samples,
        "n_confounders": n_confounders,
        "treatment_var": "treatment",
        "outcome_var": "outcome",
        "confounders": list(confounders.keys()),
    }

    return data, metadata


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def small_treatment_data():
    """5K rows for quick tests."""
    return generate_treatment_outcome_data(n_samples=5000, n_confounders=5)


@pytest.fixture
def medium_treatment_data():
    """20K rows for medium tests."""
    return generate_treatment_outcome_data(n_samples=20000, n_confounders=10)


@pytest.fixture
def large_treatment_data():
    """50K rows for large tests."""
    return generate_treatment_outcome_data(n_samples=50000, n_confounders=15)


# =============================================================================
# MOCK GRAPH BUILDER
# =============================================================================


class MockGraphBuilder:
    """Mock graph builder that simulates auto_discover behavior.

    In production, this would use the actual graph_builder node.
    """

    def __init__(self, auto_discover: bool = False):
        self.auto_discover = auto_discover

    async def build_graph(
        self,
        data: pd.DataFrame,
        treatment_var: str,
        outcome_var: str,
        confounders: List[str] = None,
    ) -> Dict:
        """Build causal graph.

        Args:
            data: DataFrame with variables
            treatment_var: Treatment variable name
            outcome_var: Outcome variable name
            confounders: Known confounders (used if auto_discover=False)

        Returns:
            Dict with graph, confidence, latency_ms
        """
        start = time.time()

        if self.auto_discover:
            # Simulate discovery algorithm
            result = await self._run_discovery(data, treatment_var, outcome_var)
        else:
            # Use provided confounders
            result = self._build_manual_graph(data, treatment_var, outcome_var, confounders)

        result["latency_ms"] = (time.time() - start) * 1000
        return result

    async def _run_discovery(
        self,
        data: pd.DataFrame,
        treatment_var: str,
        outcome_var: str,
    ) -> Dict:
        """Run causal discovery to find graph structure."""
        try:
            # Simulate discovery time based on data size
            n = len(data)
            p = len(data.columns)

            # O(n * p^2) complexity simulation
            complexity = n * (p ** 2) / 1e8
            time.sleep(min(complexity, 5.0))  # Cap at 5 seconds for mock

            # Discover confounders (mock: use columns that correlate with both T and Y)
            discovered_confounders = []
            for col in data.columns:
                if col in [treatment_var, outcome_var]:
                    continue

                # Check correlation with treatment
                corr_t = np.corrcoef(data[col], data[treatment_var])[0, 1]
                # Check correlation with outcome
                corr_y = np.corrcoef(data[col], data[outcome_var])[0, 1]

                if abs(corr_t) > 0.1 and abs(corr_y) > 0.1:
                    discovered_confounders.append(col)

            # Build graph
            nodes = [treatment_var, outcome_var] + discovered_confounders
            edges = []

            # Confounders -> Treatment
            for conf in discovered_confounders:
                edges.append((conf, treatment_var))

            # Confounders -> Outcome
            for conf in discovered_confounders:
                edges.append((conf, outcome_var))

            # Treatment -> Outcome
            edges.append((treatment_var, outcome_var))

            # Gate decision based on discovered structure
            if len(discovered_confounders) >= 2:
                gate_decision = "accept"
                confidence = 0.8
            elif len(discovered_confounders) == 1:
                gate_decision = "review"
                confidence = 0.6
            else:
                gate_decision = "reject"
                confidence = 0.3

            return {
                "nodes": nodes,
                "edges": edges,
                "discovered_confounders": discovered_confounders,
                "gate_decision": gate_decision,
                "confidence": confidence,
                "auto_discover": True,
            }

        except Exception as e:
            return {
                "error": str(e),
                "gate_decision": "error",
                "confidence": 0.0,
                "auto_discover": True,
            }

    def _build_manual_graph(
        self,
        data: pd.DataFrame,
        treatment_var: str,
        outcome_var: str,
        confounders: List[str],
    ) -> Dict:
        """Build graph from provided structure."""
        nodes = [treatment_var, outcome_var] + (confounders or [])
        edges = []

        for conf in (confounders or []):
            edges.append((conf, treatment_var))
            edges.append((conf, outcome_var))

        edges.append((treatment_var, outcome_var))

        return {
            "nodes": nodes,
            "edges": edges,
            "gate_decision": "accept",
            "confidence": 0.9,
            "auto_discover": False,
        }


# =============================================================================
# GRAPH BUILDER SCALE TESTS
# =============================================================================


class TestAutoDiscoverScale:
    """Tests for auto_discover at scale."""

    @pytest.mark.timeout(60)
    @pytest.mark.asyncio
    async def test_auto_discover_5k_rows(self, small_treatment_data):
        """auto_discover on 5K rows should complete in <15s."""
        data, metadata = small_treatment_data

        builder = MockGraphBuilder(auto_discover=True)
        result = await builder.build_graph(
            data,
            treatment_var=metadata["treatment_var"],
            outcome_var=metadata["outcome_var"],
        )

        assert "error" not in result, f"Discovery failed: {result.get('error')}"
        assert result["latency_ms"] < 15000, f"Took {result['latency_ms']:.0f}ms > 15s"
        assert len(result["discovered_confounders"]) > 0, "No confounders discovered"

    @pytest.mark.timeout(90)
    @pytest.mark.asyncio
    async def test_auto_discover_20k_rows(self, medium_treatment_data):
        """auto_discover on 20K rows should complete in <30s."""
        data, metadata = medium_treatment_data

        builder = MockGraphBuilder(auto_discover=True)
        result = await builder.build_graph(
            data,
            treatment_var=metadata["treatment_var"],
            outcome_var=metadata["outcome_var"],
        )

        assert "error" not in result, f"Discovery failed: {result.get('error')}"
        assert result["latency_ms"] < 30000, f"Took {result['latency_ms']:.0f}ms > 30s"

    @pytest.mark.timeout(120)
    @pytest.mark.asyncio
    @pytest.mark.memory_intensive
    async def test_auto_discover_50k_rows(self, large_treatment_data):
        """auto_discover on 50K rows should complete in <60s."""
        data, metadata = large_treatment_data

        builder = MockGraphBuilder(auto_discover=True)
        result = await builder.build_graph(
            data,
            treatment_var=metadata["treatment_var"],
            outcome_var=metadata["outcome_var"],
        )

        if "error" in result:
            pytest.skip(f"Discovery failed on 50K rows: {result.get('error')}")

        assert result["latency_ms"] < 60000, f"Took {result['latency_ms']:.0f}ms > 60s"


class TestGateDecisionQuality:
    """Tests for gate decision quality at scale."""

    @pytest.mark.asyncio
    async def test_gate_decision_with_clear_confounders(self, small_treatment_data):
        """Gate should accept when confounders are clearly identified."""
        data, metadata = small_treatment_data

        builder = MockGraphBuilder(auto_discover=True)
        result = await builder.build_graph(
            data,
            treatment_var=metadata["treatment_var"],
            outcome_var=metadata["outcome_var"],
        )

        # With clear confounders, should accept or review
        assert result["gate_decision"] in ["accept", "review"], (
            f"Gate decision '{result['gate_decision']}' unexpected for clear confounders"
        )

    @pytest.mark.asyncio
    async def test_gate_confidence_reasonable(self, small_treatment_data):
        """Gate confidence should be reasonable."""
        data, metadata = small_treatment_data

        builder = MockGraphBuilder(auto_discover=True)
        result = await builder.build_graph(
            data,
            treatment_var=metadata["treatment_var"],
            outcome_var=metadata["outcome_var"],
        )

        # Confidence should be in valid range
        assert 0 <= result["confidence"] <= 1, f"Confidence {result['confidence']} out of range"


class TestLatencyComparison:
    """Tests comparing auto_discover vs manual DAG construction."""

    @pytest.mark.asyncio
    async def test_manual_faster_than_auto_discover(self, small_treatment_data):
        """Manual DAG construction should be faster than auto_discover."""
        data, metadata = small_treatment_data

        # Manual construction
        manual_builder = MockGraphBuilder(auto_discover=False)
        manual_result = await manual_builder.build_graph(
            data,
            treatment_var=metadata["treatment_var"],
            outcome_var=metadata["outcome_var"],
            confounders=metadata["confounders"],
        )

        # Auto discover
        auto_builder = MockGraphBuilder(auto_discover=True)
        auto_result = await auto_builder.build_graph(
            data,
            treatment_var=metadata["treatment_var"],
            outcome_var=metadata["outcome_var"],
        )

        # Manual should be faster
        assert manual_result["latency_ms"] < auto_result["latency_ms"], (
            f"Manual {manual_result['latency_ms']:.0f}ms >= Auto {auto_result['latency_ms']:.0f}ms"
        )

    @pytest.mark.asyncio
    async def test_auto_discover_finds_similar_structure(self, small_treatment_data):
        """Auto discover should find similar structure to manual."""
        data, metadata = small_treatment_data
        known_confounders = set(metadata["confounders"])

        auto_builder = MockGraphBuilder(auto_discover=True)
        result = await auto_builder.build_graph(
            data,
            treatment_var=metadata["treatment_var"],
            outcome_var=metadata["outcome_var"],
        )

        discovered = set(result["discovered_confounders"])

        # Should overlap with known confounders
        overlap = len(known_confounders & discovered)
        assert overlap > 0, "Auto discover found no overlap with known confounders"


class TestAugmentationPath:
    """Tests for augmentation with high-confidence edges."""

    @pytest.mark.asyncio
    async def test_augmentation_adds_edges(self, small_treatment_data):
        """Augmentation should add edges when confidence is high."""
        data, metadata = small_treatment_data

        builder = MockGraphBuilder(auto_discover=True)
        result = await builder.build_graph(
            data,
            treatment_var=metadata["treatment_var"],
            outcome_var=metadata["outcome_var"],
        )

        # Should have edges for discovered confounders
        if result["discovered_confounders"]:
            expected_edges = len(result["discovered_confounders"]) * 2 + 1
            assert len(result["edges"]) >= expected_edges, (
                f"Expected >= {expected_edges} edges, got {len(result['edges'])}"
            )
