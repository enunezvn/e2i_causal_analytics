"""Tests for DriverRanker class.

Version: 1.0.0
Tests the driver ranking logic comparing causal vs predictive importance.
"""

import networkx as nx
import numpy as np
import pytest

from src.causal_engine.discovery.base import (
    DiscoveryConfig,
    DiscoveryResult,
)
from src.causal_engine.discovery.driver_ranker import (
    DriverRanker,
    DriverRankingResult,
    FeatureRanking,
    ImportanceType,
)


class TestImportanceTypeEnum:
    """Test ImportanceType enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert ImportanceType.CAUSAL.value == "causal"
        assert ImportanceType.PREDICTIVE.value == "predictive"
        assert ImportanceType.COMBINED.value == "combined"


class TestFeatureRanking:
    """Test FeatureRanking dataclass."""

    def test_create_ranking(self):
        """Test creating a feature ranking."""
        ranking = FeatureRanking(
            feature_name="feature_a",
            causal_rank=1,
            predictive_rank=3,
            causal_score=0.9,
            predictive_score=0.7,
            is_direct_cause=True,
            path_length=1,
        )

        assert ranking.feature_name == "feature_a"
        assert ranking.causal_rank == 1
        assert ranking.predictive_rank == 3
        assert ranking.rank_difference == 2  # 3 - 1

    def test_rank_difference_calculation(self):
        """Test that rank_difference is computed correctly."""
        # Predictive rank higher than causal (feature is more causal than predictive)
        ranking1 = FeatureRanking(
            feature_name="x",
            causal_rank=2,
            predictive_rank=5,
            causal_score=0.8,
            predictive_score=0.5,
        )
        assert ranking1.rank_difference == 3  # More causal than predictive

        # Causal rank higher than predictive (feature is more predictive than causal)
        ranking2 = FeatureRanking(
            feature_name="y",
            causal_rank=5,
            predictive_rank=2,
            causal_score=0.5,
            predictive_score=0.8,
        )
        assert ranking2.rank_difference == -3  # More predictive than causal

    def test_to_dict(self):
        """Test serialization to dict."""
        ranking = FeatureRanking(
            feature_name="test",
            causal_rank=1,
            predictive_rank=2,
            causal_score=0.9,
            predictive_score=0.8,
            is_direct_cause=True,
            path_length=1,
        )

        d = ranking.to_dict()

        assert d["feature_name"] == "test"
        assert d["causal_rank"] == 1
        assert d["predictive_rank"] == 2
        assert d["rank_difference"] == 1
        assert d["is_direct_cause"] is True
        assert d["path_length"] == 1


class TestDriverRankingResult:
    """Test DriverRankingResult dataclass."""

    @pytest.fixture
    def sample_rankings(self):
        """Create sample rankings for testing."""
        return [
            FeatureRanking("A", 1, 3, 0.9, 0.7),
            FeatureRanking("B", 2, 2, 0.8, 0.8),
            FeatureRanking("C", 3, 1, 0.6, 0.9),
            FeatureRanking("D", 4, 4, 0.5, 0.5),
            FeatureRanking("E", 5, 5, 0.3, 0.3),
        ]

    def test_get_by_causal_rank(self, sample_rankings):
        """Test getting rankings sorted by causal importance."""
        result = DriverRankingResult(
            rankings=sample_rankings,
            target_variable="Y",
        )

        top_2 = result.get_by_causal_rank(top_k=2)

        assert len(top_2) == 2
        assert top_2[0].feature_name == "A"  # Rank 1
        assert top_2[1].feature_name == "B"  # Rank 2

    def test_get_by_predictive_rank(self, sample_rankings):
        """Test getting rankings sorted by predictive importance."""
        result = DriverRankingResult(
            rankings=sample_rankings,
            target_variable="Y",
        )

        top_2 = result.get_by_predictive_rank(top_k=2)

        assert len(top_2) == 2
        assert top_2[0].feature_name == "C"  # Predictive rank 1
        assert top_2[1].feature_name == "B"  # Predictive rank 2

    def test_get_discordant_features(self, sample_rankings):
        """Test finding features with large rank differences."""
        result = DriverRankingResult(
            rankings=sample_rankings,
            target_variable="Y",
        )

        # Features with abs(rank_difference) >= 2
        discordant = result.get_discordant_features(threshold=2)

        # A has diff 2, C has diff -2
        assert len(discordant) == 2
        names = [r.feature_name for r in discordant]
        assert "A" in names
        assert "C" in names

    def test_to_dict(self, sample_rankings):
        """Test serialization."""
        result = DriverRankingResult(
            rankings=sample_rankings,
            target_variable="Y",
            causal_only_features=["A"],
            predictive_only_features=["C"],
            concordant_features=["B", "D", "E"],
            rank_correlation=0.7,
        )

        d = result.to_dict()

        assert d["target_variable"] == "Y"
        assert len(d["rankings"]) == 5
        assert d["causal_only_features"] == ["A"]
        assert d["predictive_only_features"] == ["C"]
        assert d["rank_correlation"] == 0.7


class TestDriverRanker:
    """Test DriverRanker class."""

    @pytest.fixture
    def ranker(self):
        """Create DriverRanker instance."""
        return DriverRanker(concordance_threshold=2, importance_percentile=0.4)

    @pytest.fixture
    def simple_dag(self):
        """Create simple DAG: A -> B -> Target, C -> Target."""
        dag = nx.DiGraph()
        dag.add_edges_from(
            [
                ("A", "B"),
                ("B", "Target"),
                ("C", "Target"),
            ]
        )
        return dag

    @pytest.fixture
    def shap_values(self):
        """Create synthetic SHAP values."""
        # 100 samples, 3 features (A, B, C)
        np.random.seed(42)
        return np.array(
            [
                np.random.randn(100) * 0.5,  # A - medium SHAP
                np.random.randn(100) * 0.3,  # B - lower SHAP
                np.random.randn(100) * 0.8,  # C - highest SHAP
            ]
        ).T

    def test_rank_drivers_basic(self, ranker, simple_dag, shap_values):
        """Test basic driver ranking."""
        result = ranker.rank_drivers(
            dag=simple_dag,
            target="Target",
            shap_values=shap_values,
            feature_names=["A", "B", "C"],
        )

        assert result.target_variable == "Target"
        assert len(result.rankings) == 3

        # All features should have rankings
        names = [r.feature_name for r in result.rankings]
        assert "A" in names
        assert "B" in names
        assert "C" in names

    def test_direct_cause_identified(self, ranker, simple_dag, shap_values):
        """Test that direct causes are properly identified."""
        result = ranker.rank_drivers(
            dag=simple_dag,
            target="Target",
            shap_values=shap_values,
            feature_names=["A", "B", "C"],
        )

        rankings_by_name = {r.feature_name: r for r in result.rankings}

        # B and C are direct causes of Target
        assert rankings_by_name["B"].is_direct_cause is True
        assert rankings_by_name["C"].is_direct_cause is True

        # A is not a direct cause (A -> B -> Target)
        assert rankings_by_name["A"].is_direct_cause is False

    def test_path_length_calculation(self, ranker, simple_dag, shap_values):
        """Test path length calculation."""
        result = ranker.rank_drivers(
            dag=simple_dag,
            target="Target",
            shap_values=shap_values,
            feature_names=["A", "B", "C"],
        )

        rankings_by_name = {r.feature_name: r for r in result.rankings}

        # A -> B -> Target = path length 2
        assert rankings_by_name["A"].path_length == 2

        # B -> Target = path length 1
        assert rankings_by_name["B"].path_length == 1

        # C -> Target = path length 1
        assert rankings_by_name["C"].path_length == 1

    def test_rank_correlation(self, ranker, simple_dag, shap_values):
        """Test that rank correlation is computed."""
        result = ranker.rank_drivers(
            dag=simple_dag,
            target="Target",
            shap_values=shap_values,
            feature_names=["A", "B", "C"],
        )

        # Correlation should be between -1 and 1
        assert -1 <= result.rank_correlation <= 1

    def test_target_not_in_dag(self, ranker):
        """Test handling when target is not in DAG."""
        dag = nx.DiGraph()
        dag.add_nodes_from(["A", "B", "C"])
        dag.add_edge("A", "B")

        # SHAP values for A, B, C
        shap_values = np.random.randn(50, 3)

        # Target not in DAG - should handle gracefully
        result = ranker.rank_drivers(
            dag=dag,
            target="Target",
            shap_values=shap_values,
            feature_names=["A", "B", "C"],
        )

        # Should still work, with Target added
        assert result is not None
        assert "Target" in dag.nodes()

    def test_categorization_of_features(self, ranker):
        """Test feature categorization (causal_only, predictive_only, concordant)."""
        # Create DAG where A is strongly causal, C is not
        dag = nx.DiGraph()
        dag.add_nodes_from(["A", "B", "C", "Target"])  # Add all nodes first
        dag.add_edges_from([("A", "Target")])
        # B and C have no path to Target

        # SHAP: C is highly predictive, A is low
        np.random.seed(42)
        shap_values = np.array(
            [
                np.random.randn(100) * 0.2,  # A - low SHAP
                np.random.randn(100) * 0.3,  # B - medium SHAP
                np.random.randn(100) * 0.9,  # C - high SHAP
            ]
        ).T

        result = ranker.rank_drivers(
            dag=dag,
            target="Target",
            shap_values=shap_values,
            feature_names=["A", "B", "C"],
        )

        # Check categorizations exist
        assert isinstance(result.causal_only_features, list)
        assert isinstance(result.predictive_only_features, list)
        assert isinstance(result.concordant_features, list)


class TestCausalImportanceComputation:
    """Test causal importance score computation."""

    def test_direct_cause_high_importance(self):
        """Test that direct causes have high causal importance."""
        ranker = DriverRanker()

        dag = nx.DiGraph()
        dag.add_edges_from(
            [
                ("Direct1", "Target"),
                ("Direct2", "Target"),
                ("Indirect", "Direct1"),
            ]
        )

        shap_values = np.random.randn(50, 3)

        result = ranker.rank_drivers(
            dag=dag,
            target="Target",
            shap_values=shap_values,
            feature_names=["Direct1", "Direct2", "Indirect"],
        )

        rankings_by_name = {r.feature_name: r for r in result.rankings}

        # Direct causes should have higher causal scores
        assert rankings_by_name["Direct1"].causal_score >= rankings_by_name["Indirect"].causal_score

    def test_no_path_to_target(self):
        """Test feature with no path to target."""
        ranker = DriverRanker()

        dag = nx.DiGraph()
        dag.add_edge("A", "Target")
        dag.add_node("Isolated")  # No path to target

        shap_values = np.random.randn(50, 2)

        result = ranker.rank_drivers(
            dag=dag,
            target="Target",
            shap_values=shap_values,
            feature_names=["A", "Isolated"],
        )

        rankings_by_name = {r.feature_name: r for r in result.rankings}

        # Isolated should have path_length None
        assert rankings_by_name["Isolated"].path_length is None


class TestPredictiveImportanceComputation:
    """Test predictive importance from SHAP values."""

    def test_high_shap_high_importance(self):
        """Test that high SHAP values give high predictive importance."""
        ranker = DriverRanker()

        dag = nx.DiGraph()
        dag.add_nodes_from(["A", "B", "Target"])

        # A has much higher SHAP than B
        shap_values = np.array(
            [
                [1.0, 0.1],
                [0.9, 0.1],
                [1.1, 0.2],
            ]
        )

        result = ranker.rank_drivers(
            dag=dag,
            target="Target",
            shap_values=shap_values,
            feature_names=["A", "B"],
        )

        rankings_by_name = {r.feature_name: r for r in result.rankings}

        # A should have higher predictive score
        assert rankings_by_name["A"].predictive_score > rankings_by_name["B"].predictive_score
        assert rankings_by_name["A"].predictive_rank < rankings_by_name["B"].predictive_rank


class TestSpearmanCorrelation:
    """Test Spearman correlation calculation."""

    def test_perfect_correlation(self):
        """Test perfect positive correlation."""
        ranker = DriverRanker()

        ranks1 = [1, 2, 3, 4, 5]
        ranks2 = [1, 2, 3, 4, 5]

        corr = ranker._spearman_correlation(ranks1, ranks2)

        assert corr == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation."""
        ranker = DriverRanker()

        ranks1 = [1, 2, 3, 4, 5]
        ranks2 = [5, 4, 3, 2, 1]

        corr = ranker._spearman_correlation(ranks1, ranks2)

        assert corr == pytest.approx(-1.0)

    def test_empty_ranks(self):
        """Test with empty ranks."""
        ranker = DriverRanker()

        corr = ranker._spearman_correlation([], [])

        assert corr == 0.0

    def test_mismatched_lengths(self):
        """Test with mismatched rank lengths."""
        ranker = DriverRanker()

        corr = ranker._spearman_correlation([1, 2, 3], [1, 2])

        assert corr == 0.0


class TestRankFromDiscoveryResult:
    """Test convenience method for DiscoveryResult."""

    def test_from_discovery_result(self):
        """Test ranking from DiscoveryResult."""
        ranker = DriverRanker()

        dag = nx.DiGraph()
        dag.add_edges_from([("A", "Target"), ("B", "Target")])

        result = DiscoveryResult(
            success=True,
            config=DiscoveryConfig(),
            ensemble_dag=dag,
            edges=[],
            metadata={"node_names": ["A", "B", "Target"]},
        )

        shap_values = np.random.randn(50, 2)

        ranking_result = ranker.rank_from_discovery_result(
            result=result,
            target="Target",
            shap_values=shap_values,
        )

        assert ranking_result.target_variable == "Target"
        assert len(ranking_result.rankings) == 2

    def test_from_discovery_result_no_dag(self):
        """Test error when DiscoveryResult has no DAG."""
        ranker = DriverRanker()

        result = DiscoveryResult(
            success=False,
            config=DiscoveryConfig(),
            ensemble_dag=None,
        )

        with pytest.raises(ValueError, match="no ensemble DAG"):
            ranker.rank_from_discovery_result(
                result=result,
                target="Target",
                shap_values=np.random.randn(10, 2),
            )
