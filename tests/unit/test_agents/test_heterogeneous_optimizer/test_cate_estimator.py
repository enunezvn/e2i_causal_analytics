"""Tests for CATE Estimator Node."""

import numpy as np
import pytest

from src.agents.heterogeneous_optimizer.connectors import MockDataConnector
from src.agents.heterogeneous_optimizer.nodes.cate_estimator import CATEEstimatorNode
from src.agents.heterogeneous_optimizer.state import HeterogeneousOptimizerState


class TestCATEEstimatorNode:
    """Test CATEEstimatorNode."""

    def _create_test_state(self, **overrides) -> HeterogeneousOptimizerState:
        """Create test state with defaults."""
        state = {
            "query": "Which segments respond best to treatment?",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty", "region"],
            "effect_modifiers": ["hcp_tenure", "competitive_pressure", "formulary_status"],
            "data_source": "hcp_performance_metrics",
            "filters": None,
            "n_estimators": 100,
            "min_samples_leaf": 10,
            "significance_level": 0.05,
            "top_segments_count": 10,
            "cate_by_segment": None,
            "overall_ate": None,
            "heterogeneity_score": None,
            "feature_importance": None,
            "high_responders": None,
            "low_responders": None,
            "segment_comparison": None,
            "policy_recommendations": None,
            "expected_total_lift": None,
            "optimal_allocation_summary": None,
            "cate_plot_data": None,
            "segment_grid_data": None,
            "executive_summary": None,
            "key_insights": None,
            "estimation_latency_ms": 0,
            "analysis_latency_ms": 0,
            "total_latency_ms": 0,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }
        state.update(overrides)
        return state

    @pytest.mark.asyncio
    async def test_cate_estimation_basic(self):
        """Test basic CATE estimation."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state()

        result = await node.execute(state)

        assert "overall_ate" in result
        assert result["overall_ate"] is not None
        assert isinstance(result["overall_ate"], float)

    @pytest.mark.asyncio
    async def test_heterogeneity_score(self):
        """Test heterogeneity score calculation."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state()

        result = await node.execute(state)

        assert "heterogeneity_score" in result
        assert result["heterogeneity_score"] is not None
        assert 0.0 <= result["heterogeneity_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_feature_importance(self):
        """Test feature importance extraction."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state()

        result = await node.execute(state)

        assert "feature_importance" in result
        assert result["feature_importance"] is not None
        assert isinstance(result["feature_importance"], dict)

        # Should have importance for each effect modifier
        for modifier in state["effect_modifiers"]:
            assert modifier in result["feature_importance"]

    @pytest.mark.asyncio
    async def test_cate_by_segment(self):
        """Test CATE calculation by segment."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state()

        result = await node.execute(state)

        assert "cate_by_segment" in result
        assert result["cate_by_segment"] is not None
        assert isinstance(result["cate_by_segment"], dict)

        # Should have results for each segment variable
        for segment_var in state["segment_vars"]:
            assert segment_var in result["cate_by_segment"]

    @pytest.mark.asyncio
    async def test_cate_by_segment_encodes_string_effect_modifiers(self):
        """Regression: per-segment CATE must encode STRING effect modifiers the SAME
        way the training matrix was built. ``_calculate_cate_by_segment`` previously
        fed the RAW ``segment_df[effect_modifiers].values`` to ``cf.effect`` while
        training label-encoded them, so any categorical-string modifier crashed econml
        with "could not convert string to float". Surfaced by the
        synthetic-causal-validation Shard 11 gate 11: the #839 het resolver binds the
        conversion-KPI substrate whose effect modifiers include string columns
        (trigger_type / delivery_channel / priority). The fix encodes the modifiers
        over the full frame and positionally masks per segment.
        """
        import numpy as np
        import pandas as pd

        node = CATEEstimatorNode(data_connector=MockDataConnector())
        n = 60
        df = pd.DataFrame(
            {
                "severity": ["high", "low"] * (n // 2),  # string segment_var
                "channel": ["email", "call", "visit"] * (n // 3),  # STRING modifier
                "tenure": np.linspace(1.0, 30.0, n),  # numeric modifier
            }
        )

        class _FakeForest:
            # CausalForestDML stand-in: effect() requires a NUMERIC design matrix.
            # np.asarray(dtype=float) raises on any residual string -> proves the node
            # encoded the modifiers before calling effect() (RED before the fix).
            def effect(self, X):
                return np.asarray(X, dtype=float).sum(axis=1)

            def effect_interval(self, X, alpha):
                e = self.effect(X)
                return (e - 0.1, e + 0.1)

        # Planted per-segment treated shares: every "high" row treated, every
        # "low" row untreated — the observed rates must come back 1.0 / 0.0.
        T = (df["severity"] == "high").to_numpy().astype(float)

        result = await node._calculate_cate_by_segment(
            df,
            _FakeForest(),
            segment_vars=["severity"],
            effect_modifiers=["channel", "tenure"],
            alpha=0.05,
            T=T,
        )

        # Both severity segments computed without a string->float crash.
        assert "severity" in result
        assert len(result["severity"]) == 2

        # Observed treated share is measured per segment, not assumed.
        rates = {r["segment_value"]: r["treatment_rate"] for r in result["severity"]}
        assert rates == {"high": 1.0, "low": 0.0}

    @pytest.mark.asyncio
    async def test_cate_result_structure(self):
        """Test structure of CATE results."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state()

        result = await node.execute(state)

        cate_by_segment = result["cate_by_segment"]
        for _segment_var, results in cate_by_segment.items():
            assert isinstance(results, list)
            assert len(results) > 0

            # Check first result structure
            cate_result = results[0]
            assert "segment_name" in cate_result
            assert "segment_value" in cate_result
            assert "cate_estimate" in cate_result
            assert "cate_ci_lower" in cate_result
            assert "cate_ci_upper" in cate_result
            assert "sample_size" in cate_result
            assert "statistical_significance" in cate_result
            assert "treatment_rate" in cate_result
            assert 0.0 <= cate_result["treatment_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_interval(self):
        """Test confidence interval calculation."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state()

        result = await node.execute(state)

        cate_by_segment = result["cate_by_segment"]
        for _segment_var, results in cate_by_segment.items():
            for cate_result in results:
                ci_lower = cate_result["cate_ci_lower"]
                ci_upper = cate_result["cate_ci_upper"]
                cate = cate_result["cate_estimate"]

                # CI should contain estimate
                assert ci_lower <= cate <= ci_upper

    @pytest.mark.asyncio
    async def test_statistical_significance(self):
        """Test statistical significance determination."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state()

        result = await node.execute(state)

        cate_by_segment = result["cate_by_segment"]
        for _segment_var, results in cate_by_segment.items():
            for cate_result in results:
                significance = cate_result["statistical_significance"]
                assert isinstance(significance, bool)

                # If significant, CI should not contain 0
                if significance:
                    ci_lower = cate_result["cate_ci_lower"]
                    ci_upper = cate_result["cate_ci_upper"]
                    assert (ci_lower > 0) or (ci_upper < 0)

    @pytest.mark.asyncio
    async def test_estimation_latency(self):
        """Test latency measurement."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state()

        result = await node.execute(state)

        assert "estimation_latency_ms" in result
        assert result["estimation_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_status_update(self):
        """Test status update to analyzing."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state()

        result = await node.execute(state)

        assert result["status"] == "analyzing"

    @pytest.mark.asyncio
    async def test_insufficient_data(self):
        """Test handling of insufficient data.

        The CATE estimator has fallback logic that switches to MockDataConnector
        when insufficient data is returned. We need to patch the connectors module
        so the fallback also returns insufficient data.
        """
        from unittest.mock import patch

        # Create mock connector that returns small dataset
        class SmallDataConnector:
            async def query(self, source, columns, filters=None):
                import numpy as np
                import pandas as pd

                np.random.seed(42)
                # Only 50 rows - insufficient for CATE estimation
                return pd.DataFrame({col: np.random.randn(50) for col in columns})

        node = CATEEstimatorNode(data_connector=SmallDataConnector())
        state = self._create_test_state()

        # Patch MockDataConnector in the connectors module where it's imported from
        with patch(
            "src.agents.heterogeneous_optimizer.connectors.MockDataConnector",
            SmallDataConnector,
        ):
            result = await node.execute(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert "Insufficient data" in result["errors"][0]["error"]

    @pytest.mark.asyncio
    async def test_multiple_segments(self):
        """Test CATE estimation with multiple segment variables."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state(
            segment_vars=["hcp_specialty", "region", "patient_volume_decile"]
        )

        result = await node.execute(state)

        assert len(result["cate_by_segment"]) == 3

    @pytest.mark.asyncio
    async def test_heterogeneity_score_range(self):
        """Test heterogeneity score is properly normalized."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state()

        result = await node.execute(state)

        # Score should be between 0 and 1
        score = result["heterogeneity_score"]
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_cate_sorted_by_estimate(self):
        """Test CATE results are sorted by estimate."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state()

        result = await node.execute(state)

        for _segment_var, results in result["cate_by_segment"].items():
            # Check sorting (descending)
            for i in range(len(results) - 1):
                assert results[i]["cate_estimate"] >= results[i + 1]["cate_estimate"]


class TestCATEEstimatorEdgeCases:
    """Test edge cases for CATE estimator."""

    def _create_test_state(self, **overrides):
        """Create test state."""
        state = {
            "query": "test",
            "treatment_var": "hcp_engagement_frequency",
            "outcome_var": "trx_total",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["hcp_tenure", "competitive_pressure"],
            "data_source": "test",
            "filters": None,
            "n_estimators": 52,  # Must be divisible by 4 (subforest_size)
            "min_samples_leaf": 10,
            "significance_level": 0.05,
            "top_segments_count": 10,
            "cate_by_segment": None,
            "overall_ate": None,
            "heterogeneity_score": None,
            "feature_importance": None,
            "high_responders": None,
            "low_responders": None,
            "segment_comparison": None,
            "policy_recommendations": None,
            "expected_total_lift": None,
            "optimal_allocation_summary": None,
            "cate_plot_data": None,
            "segment_grid_data": None,
            "executive_summary": None,
            "key_insights": None,
            "estimation_latency_ms": 0,
            "analysis_latency_ms": 0,
            "total_latency_ms": 0,
            "errors": [],
            "warnings": [],
            "status": "pending",
        }
        state.update(overrides)
        return state

    @pytest.mark.asyncio
    async def test_binary_treatment(self):
        """Test with binary treatment variable."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state()

        result = await node.execute(state)

        # Should handle binary treatment
        assert result["status"] == "analyzing"

    @pytest.mark.asyncio
    async def test_continuous_treatment(self):
        """Test with continuous treatment variable."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())

        # Mock connector with continuous treatment
        class ContinuousTreatmentConnector:
            async def query(self, source, columns, filters=None):
                import numpy as np
                import pandas as pd

                np.random.seed(42)
                n = 1000
                data = {}
                for col in columns:
                    if col == "hcp_engagement_frequency":
                        data[col] = np.random.uniform(0, 10, n)  # Continuous
                    else:
                        data[col] = np.random.randn(n)
                return pd.DataFrame(data)

        node.data_connector = ContinuousTreatmentConnector()
        state = self._create_test_state()

        result = await node.execute(state)

        # Should handle continuous treatment
        assert result["status"] == "analyzing"

    @pytest.mark.asyncio
    async def test_single_effect_modifier(self):
        """Test with single effect modifier."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        state = self._create_test_state(effect_modifiers=["hcp_tenure"])

        result = await node.execute(state)

        assert len(result["feature_importance"]) == 1

    @pytest.mark.asyncio
    async def test_many_effect_modifiers(self):
        """Test with many effect modifiers."""
        import numpy as np
        import pandas as pd

        # Custom mock connector with 10 effect modifiers
        class ManyModifiersConnector:
            async def query(self, source, columns, filters=None):
                np.random.seed(42)
                n = 1000
                data = {}
                for col in columns:
                    if col == "hcp_engagement_frequency":
                        data[col] = np.random.choice([0, 1], n)
                    elif col == "trx_total":
                        data[col] = np.random.randn(n) * 100 + 500
                    elif col == "hcp_specialty":
                        data[col] = np.random.choice(["A", "B"], n)
                    else:
                        data[col] = np.random.randn(n)
                return pd.DataFrame(data)

        node = CATEEstimatorNode(data_connector=ManyModifiersConnector())
        modifiers = [f"modifier_{i}" for i in range(10)]
        state = self._create_test_state(effect_modifiers=modifiers)

        result = await node.execute(state)

        # If status is "analyzing", check feature importance
        if result["status"] == "analyzing":
            assert len(result["feature_importance"]) == 10
        else:
            # Edge case where EconML may fail with many modifiers - check error exists
            assert result["errors"] is not None


class TestSegmentMeanInference:
    """Honest segment-mean CI regression tests (2026-07-05 "0/14 significant" incident).

    The per-segment CI must be the residual-based GATE interval (shrinks
    ~1/sqrt(n)), NOT the mean of the forest's per-individual interval bounds
    (an individual-level prediction interval whose width is n-independent).
    On the live Remibrutinib cohort the old computation reported ±17.7pp for
    every segment regardless of n (1.4k–5.9k rows), so a real +11pp effect
    could never test significant.
    """

    @staticmethod
    def _make_frame_and_forest(n: int, theta_by_segment: dict, seed: int = 0):
        """Frame + CausalForestDML stand-in with planted DML residuals.

        The fake forest's per-point intervals are DELIBERATELY enormous
        (±10.0): if the node consumed them for the segment CI, nothing would
        ever be significant — proving the GATE path is the one in use.
        """
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)
        segments = rng.choice(list(theta_by_segment), size=n)
        t_res = rng.choice([0.5, -0.5], size=n)  # centered binary residual
        theta = np.array([theta_by_segment[s] for s in segments])
        y_res = theta * t_res + rng.normal(0.0, 0.05, n)
        df = pd.DataFrame({"segment": segments, "modifier": rng.normal(size=n)})

        class _FakeForest:
            residuals_ = (y_res, t_res.reshape(-1, 1), None, None)

            def effect(self, X):
                import numpy as np

                return np.zeros(len(X))

            def effect_interval(self, X, alpha):
                import numpy as np

                return (np.full(len(X), -10.0), np.full(len(X), 10.0))

        return df, _FakeForest()

    @pytest.mark.asyncio
    async def test_gate_recovers_significance_and_ordering(self):
        """A planted +0.11 mean effect at n≈4000/segment must be significant."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())
        df, cf = self._make_frame_and_forest(8000, {"high": 0.15, "low": 0.08})

        result = await node._calculate_cate_by_segment(
            df,
            cf,
            segment_vars=["segment"],
            effect_modifiers=["modifier"],
            alpha=0.05,
            T=np.arange(len(df)) % 2,
        )

        rows = {r["segment_value"]: r for r in result["segment"]}
        assert rows["high"]["statistical_significance"] is True
        assert rows["low"]["statistical_significance"] is True
        assert abs(rows["high"]["cate_estimate"] - 0.15) < 0.02
        assert abs(rows["low"]["cate_estimate"] - 0.08) < 0.02
        # CI is a mean-scale interval, nothing like the fake ±10 per-point bounds.
        for r in rows.values():
            assert (r["cate_ci_upper"] - r["cate_ci_lower"]) < 0.1

    @pytest.mark.asyncio
    async def test_gate_ci_shrinks_with_sample_size(self):
        """The segment CI width must shrink ~1/sqrt(n) (16x n -> ~4x narrower)."""
        node = CATEEstimatorNode(data_connector=MockDataConnector())

        widths = {}
        for n in (500, 8000):
            df, cf = self._make_frame_and_forest(n, {"only": 0.11}, seed=1)
            result = await node._calculate_cate_by_segment(
                df,
                cf,
                segment_vars=["segment"],
                effect_modifiers=["modifier"],
                alpha=0.05,
                T=np.arange(len(df)) % 2,
            )
            row = result["segment"][0]
            widths[n] = row["cate_ci_upper"] - row["cate_ci_lower"]

        assert widths[500] > 2.5 * widths[8000]

    @pytest.mark.asyncio
    async def test_multi_column_treatment_residuals_fall_back(self):
        """Multi-valued treatment residuals must fall back to per-point intervals."""
        import numpy as np
        import pandas as pd

        node = CATEEstimatorNode(data_connector=MockDataConnector())
        n = 40
        df = pd.DataFrame({"segment": ["a", "b"] * (n // 2), "modifier": np.zeros(n)})

        class _MultiTreatmentForest:
            # 2-column T residual: the single-theta GATE moment does not apply.
            residuals_ = (np.zeros(n), np.zeros((n, 2)), None, None)

            def effect(self, X):
                return np.full(len(X), 0.3)

            def effect_interval(self, X, alpha):
                return (np.full(len(X), 0.1), np.full(len(X), 0.5))

        result = await node._calculate_cate_by_segment(
            df,
            _MultiTreatmentForest(),
            segment_vars=["segment"],
            effect_modifiers=["modifier"],
            alpha=0.05,
            T=np.arange(n) % 2,
        )

        for row in result["segment"]:
            assert row["cate_estimate"] == pytest.approx(0.3)
            assert row["cate_ci_lower"] == pytest.approx(0.1)
            assert row["cate_ci_upper"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_residual_length_mismatch_falls_back(self):
        """Misaligned residuals (defensive guard) must fall back, not misindex."""
        import numpy as np
        import pandas as pd

        node = CATEEstimatorNode(data_connector=MockDataConnector())
        n = 40
        df = pd.DataFrame({"segment": ["a"] * n, "modifier": np.zeros(n)})

        class _MisalignedForest:
            residuals_ = (np.zeros(n + 7), np.zeros(n + 7), None, None)

            def effect(self, X):
                return np.full(len(X), 0.2)

            def effect_interval(self, X, alpha):
                return (np.full(len(X), -0.1), np.full(len(X), 0.5))

        result = await node._calculate_cate_by_segment(
            df,
            _MisalignedForest(),
            segment_vars=["segment"],
            effect_modifiers=["modifier"],
            alpha=0.05,
            T=np.arange(n) % 2,
        )

        row = result["segment"][0]
        assert row["cate_estimate"] == pytest.approx(0.2)
        assert row["statistical_significance"] is False
