"""Tests for Profile Generator Node."""

import pytest

from src.agents.heterogeneous_optimizer.nodes.profile_generator import ProfileGeneratorNode
from src.agents.heterogeneous_optimizer.state import (
    HeterogeneousOptimizerState,
)


class TestProfileGeneratorNode:
    """Test ProfileGeneratorNode."""

    def _create_test_state(self) -> HeterogeneousOptimizerState:
        """Create test state with complete analysis results."""
        cate_by_segment = {
            "hcp_specialty": [
                {
                    "segment_name": "hcp_specialty",
                    "segment_value": "Oncology",
                    "cate_estimate": 0.50,
                    "cate_ci_lower": 0.40,
                    "cate_ci_upper": 0.60,
                    "sample_size": 200,
                    "statistical_significance": True,
                },
                {
                    "segment_name": "hcp_specialty",
                    "segment_value": "Primary Care",
                    "cate_estimate": 0.10,
                    "cate_ci_lower": 0.05,
                    "cate_ci_upper": 0.15,
                    "sample_size": 300,
                    "statistical_significance": True,
                },
            ]
        }

        high_responders = [
            {
                "segment_id": "hcp_specialty_Oncology",
                "responder_type": "high",
                "cate_estimate": 0.50,
                "defining_features": [
                    {"variable": "hcp_specialty", "value": "Oncology", "effect_size": 2.0}
                ],
                "size": 200,
                "size_percentage": 20.0,
                "recommendation": "Prioritize treatment",
            }
        ]

        low_responders = [
            {
                "segment_id": "hcp_specialty_Primary Care",
                "responder_type": "low",
                "cate_estimate": 0.10,
                "defining_features": [
                    {"variable": "hcp_specialty", "value": "Primary Care", "effect_size": 0.4}
                ],
                "size": 300,
                "size_percentage": 30.0,
                "recommendation": "De-prioritize treatment",
            }
        ]

        segment_comparison = {
            "overall_ate": 0.25,
            "high_responder_avg_cate": 0.50,
            "low_responder_avg_cate": 0.10,
            "effect_ratio": 5.0,
            "high_responder_count": 1,
            "low_responder_count": 1,
        }

        return {
            "query": "test",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "segment_vars": ["hcp_specialty"],
            "effect_modifiers": ["modifier1"],
            "data_source": "test",
            "filters": None,
            "n_estimators": 100,
            "min_samples_leaf": 10,
            "significance_level": 0.05,
            "top_segments_count": 10,
            "cate_by_segment": cate_by_segment,
            "overall_ate": 0.25,
            "heterogeneity_score": 0.6,
            "feature_importance": {"modifier1": 0.5, "modifier2": 0.3},
            "high_responders": high_responders,
            "low_responders": low_responders,
            "segment_comparison": segment_comparison,
            "policy_recommendations": [],
            "expected_total_lift": 150.0,
            "optimal_allocation_summary": "Test summary",
            "cate_plot_data": None,
            "segment_grid_data": None,
            "executive_summary": None,
            "key_insights": None,
            "estimation_latency_ms": 100,
            "analysis_latency_ms": 50,
            "total_latency_ms": 200,
            "errors": [],
            "warnings": [],
            "status": "completed",
        }

    @pytest.mark.asyncio
    async def test_generate_cate_plot_data(self):
        """Test CATE plot data generation."""
        node = ProfileGeneratorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "cate_plot_data" in result
        assert result["cate_plot_data"] is not None
        assert "overall_ate" in result["cate_plot_data"]
        assert "segments" in result["cate_plot_data"]

    @pytest.mark.asyncio
    async def test_cate_plot_data_structure(self):
        """Test structure of CATE plot data."""
        node = ProfileGeneratorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        plot_data = result["cate_plot_data"]
        segments = plot_data["segments"]

        assert len(segments) > 0

        # Check first segment structure
        seg = segments[0]
        assert "segment_var" in seg
        assert "segment_value" in seg
        assert "cate" in seg
        assert "ci_lower" in seg
        assert "ci_upper" in seg
        assert "sample_size" in seg
        assert "significant" in seg

    @pytest.mark.asyncio
    async def test_cate_plot_sorted(self):
        """Test CATE plot data is sorted by CATE."""
        node = ProfileGeneratorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        segments = result["cate_plot_data"]["segments"]
        for i in range(len(segments) - 1):
            assert segments[i]["cate"] >= segments[i + 1]["cate"]

    @pytest.mark.asyncio
    async def test_generate_segment_grid_data(self):
        """Test segment grid data generation."""
        node = ProfileGeneratorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "segment_grid_data" in result
        assert result["segment_grid_data"] is not None

    @pytest.mark.asyncio
    async def test_segment_grid_data_structure(self):
        """Test structure of segment grid data."""
        node = ProfileGeneratorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        grid_data = result["segment_grid_data"]

        assert "comparison_metrics" in grid_data
        assert "high_responder_segments" in grid_data
        assert "low_responder_segments" in grid_data

    @pytest.mark.asyncio
    async def test_executive_summary_generation(self):
        """Test executive summary generation."""
        node = ProfileGeneratorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "executive_summary" in result
        assert result["executive_summary"] is not None
        assert isinstance(result["executive_summary"], str)
        assert len(result["executive_summary"]) > 0

    @pytest.mark.asyncio
    async def test_executive_summary_content(self):
        """Test executive summary contains key information."""
        node = ProfileGeneratorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        summary = result["executive_summary"]

        # Should mention key metrics
        assert "0.25" in summary or "0.6" in summary  # ATE or heterogeneity
        assert "high-responder" in summary.lower() or "low-responder" in summary.lower()

    @pytest.mark.asyncio
    async def test_key_insights_generation(self):
        """Test key insights generation."""
        node = ProfileGeneratorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert "key_insights" in result
        assert result["key_insights"] is not None
        assert isinstance(result["key_insights"], list)
        assert len(result["key_insights"]) > 0

    @pytest.mark.asyncio
    async def test_key_insights_limit(self):
        """Test key insights limited to 5."""
        node = ProfileGeneratorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert len(result["key_insights"]) <= 5

    @pytest.mark.asyncio
    async def test_key_insights_content(self):
        """Test key insights contain meaningful information."""
        node = ProfileGeneratorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        insights = result["key_insights"]

        # Should have insights about treatment effect and heterogeneity
        assert any("treatment" in insight.lower() for insight in insights)
        assert any(
            "heterogeneity" in insight.lower() or "segment" in insight.lower()
            for insight in insights
        )

    @pytest.mark.asyncio
    async def test_status_remains_completed(self):
        """Test status remains completed."""
        node = ProfileGeneratorNode()
        state = self._create_test_state()

        result = await node.execute(state)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_failed_status_passthrough(self):
        """Test failed status is passed through."""
        node = ProfileGeneratorNode()
        state = self._create_test_state()
        state["status"] = "failed"

        result = await node.execute(state)

        assert result["status"] == "failed"


class TestProfileGeneratorEdgeCases:
    """Test edge cases for profile generator."""

    def _create_test_state(self, **overrides):
        """Create test state."""
        state = {
            "query": "test",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "segment_vars": ["segment1"],
            "effect_modifiers": ["modifier1"],
            "data_source": "test",
            "filters": None,
            "n_estimators": 100,
            "min_samples_leaf": 10,
            "significance_level": 0.05,
            "top_segments_count": 10,
            "cate_by_segment": {},
            "overall_ate": 0.25,
            "heterogeneity_score": 0.0,
            "feature_importance": {},
            "high_responders": [],
            "low_responders": [],
            "segment_comparison": {},
            "policy_recommendations": [],
            "expected_total_lift": 0.0,
            "optimal_allocation_summary": "",
            "cate_plot_data": None,
            "segment_grid_data": None,
            "executive_summary": None,
            "key_insights": None,
            "estimation_latency_ms": 0,
            "analysis_latency_ms": 0,
            "total_latency_ms": 0,
            "errors": [],
            "warnings": [],
            "status": "completed",
        }
        state.update(overrides)
        return state

    @pytest.mark.asyncio
    async def test_no_high_or_low_responders(self):
        """Test with no high or low responders."""
        node = ProfileGeneratorNode()
        state = self._create_test_state(high_responders=[], low_responders=[])

        result = await node.execute(state)

        # Should still generate summary
        assert result["executive_summary"] is not None
        assert (
            "uniform" in result["executive_summary"].lower()
            or "limited heterogeneity" in result["executive_summary"].lower()
        )

    @pytest.mark.asyncio
    async def test_high_heterogeneity(self):
        """Test with high heterogeneity score."""
        node = ProfileGeneratorNode()
        state = self._create_test_state(
            heterogeneity_score=0.9,
            high_responders=[
                {
                    "segment_id": "test",
                    "cate_estimate": 0.50,
                    "size_percentage": 10.0,
                    "responder_type": "high",
                    "defining_features": [],
                    "size": 100,
                    "recommendation": "test",
                }
            ],
            low_responders=[
                {
                    "segment_id": "test2",
                    "cate_estimate": 0.05,
                    "size_percentage": 10.0,
                    "responder_type": "low",
                    "defining_features": [],
                    "size": 100,
                    "recommendation": "test",
                }
            ],
        )

        result = await node.execute(state)

        # Should mention high heterogeneity
        insights = " ".join(result["key_insights"])
        assert "high" in insights.lower() and "heterogeneity" in insights.lower()

    @pytest.mark.asyncio
    async def test_low_heterogeneity(self):
        """Test with low heterogeneity score."""
        node = ProfileGeneratorNode()
        state = self._create_test_state(heterogeneity_score=0.1)

        result = await node.execute(state)

        # Should mention low heterogeneity
        insights = " ".join(result["key_insights"])
        assert "low" in insights.lower() and "heterogeneity" in insights.lower()

    @pytest.mark.asyncio
    async def test_negative_ate(self):
        """Test with negative ATE."""
        node = ProfileGeneratorNode()
        state = self._create_test_state(overall_ate=-0.15)

        result = await node.execute(state)

        # Should mention negative effect
        insights = " ".join(result["key_insights"])
        assert "negative" in insights.lower()

    @pytest.mark.asyncio
    async def test_empty_cate_by_segment(self):
        """Test with empty CATE by segment."""
        node = ProfileGeneratorNode()
        state = self._create_test_state(cate_by_segment={})

        result = await node.execute(state)

        # Should still generate outputs
        assert result["cate_plot_data"] is not None
        assert len(result["cate_plot_data"]["segments"]) == 0


class TestProfileGeneratorLLMInterpretation:
    """The explanation is LLM-generated (CATEInterpretationSignature) when an LM is
    configured, with an honest factual fallback otherwise."""

    def _state(self) -> HeterogeneousOptimizerState:
        return {  # type: ignore[return-value]
            "overall_ate": 0.243,
            "heterogeneity_score": 0.13,
            "expected_lift_pp": 0.0,
            "optimal_allocation_summary": "No segment responds significantly above average.",
            "feature_importance": {"disease_severity": 0.35, "egfr": 0.18},
            "cate_by_segment": {
                "disease_severity_band": [
                    {
                        "segment_name": "disease_severity_band",
                        "segment_value": "high",
                        "cate_estimate": 0.295,
                        "cate_ci_lower": 0.141,
                        "cate_ci_upper": 0.448,
                        "sample_size": 1354,
                        "statistical_significance": True,
                    }
                ]
            },
            "high_responders": [],
            "low_responders": [],
            "errors": [],
            "warnings": [],
            "status": "optimizing",
        }

    @pytest.mark.asyncio
    async def test_llm_output_is_used_when_available(self, monkeypatch):
        """When the runner returns content, the executive_summary / strategic_
        interpretation / key_insights come from the LLM, NOT the templates."""

        def _fake_runner(**kwargs):
            # Echo a couple of inputs to prove it was fed the real numbers.
            assert kwargs["overall_ate"] == pytest.approx(0.243)
            assert "disease_severity_band=high" in kwargs["cate_by_segment_text"]
            assert "significantly ABOVE" not in kwargs["cate_by_segment_text"]  # CI overlaps ATE
            return {
                "executive_summary": "LLM-EXEC: uniform effect, no targeting edge.",
                "interpretation": "LLM-INTERP: all segment CIs overlap the ATE.",
                "key_insights": ["LLM-INSIGHT-1", "LLM-INSIGHT-2"],
                "high_responder_description": "none significantly above average",
                "low_responder_description": "none significantly below average",
            }

        monkeypatch.setattr(
            "src.agents.heterogeneous_optimizer.dspy_integration.generate_cate_interpretation",
            _fake_runner,
        )

        result = await ProfileGeneratorNode().execute(self._state())

        assert result["executive_summary"] == "LLM-EXEC: uniform effect, no targeting edge."
        assert result["strategic_interpretation"] == "LLM-INTERP: all segment CIs overlap the ATE."
        assert result["key_insights"] == ["LLM-INSIGHT-1", "LLM-INSIGHT-2"]

    @pytest.mark.asyncio
    async def test_falls_back_to_factual_when_llm_unavailable(self, monkeypatch):
        """When the runner returns None (no LM / failure), a factual, non-empty,
        non-fabricated summary is produced — the run never blocks."""
        monkeypatch.setattr(
            "src.agents.heterogeneous_optimizer.dspy_integration.generate_cate_interpretation",
            lambda **kwargs: None,
        )

        result = await ProfileGeneratorNode().execute(self._state())

        assert isinstance(result["executive_summary"], str) and result["executive_summary"]
        assert result["key_insights"] and isinstance(result["key_insights"], list)
        # Factual fallback cites the real ATE; it is NOT the LLM sentinel.
        assert "LLM-EXEC" not in result["executive_summary"]

    def test_serialize_cate_flags_above_ate(self):
        """The CATE serialization marks which segments clear the above-ATE bar so the
        LLM can reconcile nominal tiers with significance-gated targeting."""
        node = ProfileGeneratorNode()
        state = {  # type: ignore[assignment]
            "cate_by_segment": {
                "d": [
                    # CI lower (0.30) > ATE (0.20) -> above
                    {
                        "segment_value": "a",
                        "cate_estimate": 0.40,
                        "cate_ci_lower": 0.30,
                        "cate_ci_upper": 0.50,
                        "sample_size": 100,
                    },
                    # CI straddles ATE -> not distinguishable
                    {
                        "segment_value": "b",
                        "cate_estimate": 0.22,
                        "cate_ci_lower": 0.10,
                        "cate_ci_upper": 0.34,
                        "sample_size": 100,
                    },
                    # CI upper (-0.05) < 0 -> harmful
                    {
                        "segment_value": "c",
                        "cate_estimate": -0.15,
                        "cate_ci_lower": -0.25,
                        "cate_ci_upper": -0.05,
                        "sample_size": 100,
                    },
                ]
            }
        }
        text = node._serialize_cate_for_llm(state, overall_ate=0.20)  # type: ignore[arg-type]
        assert "d=a:" in text and "significantly ABOVE the ATE" in text
        assert "d=b:" in text and "NOT statistically distinguishable" in text
        assert "d=c:" in text and "significantly HARMFUL" in text
