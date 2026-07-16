"""
Tests for Pattern Analyzer node.
"""

from unittest.mock import AsyncMock

import pytest

from src.agents.feedback_learner.nodes.pattern_analyzer import PatternAnalyzerNode


class TestPatternAnalyzerNode:
    """Tests for PatternAnalyzerNode."""

    @pytest.mark.asyncio
    async def test_execute_with_feedback(self, state_with_feedback):
        """Test execution with feedback items."""
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state_with_feedback)

        assert result["status"] == "extracting"
        assert result["detected_patterns"] is not None
        assert result["pattern_clusters"] is not None
        assert result["analysis_latency_ms"] >= 0
        assert result["model_used"] == "deterministic"

    @pytest.mark.asyncio
    async def test_execute_empty_feedback(self, base_state):
        """Test execution with no feedback items."""
        state = {**base_state, "feedback_items": [], "status": "analyzing"}
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        assert result["status"] == "extracting"
        assert result["detected_patterns"] == []
        assert result["pattern_clusters"] == {}

    @pytest.mark.asyncio
    async def test_skip_if_already_failed(self, base_state):
        """Test that node skips execution if already failed."""
        state = {**base_state, "status": "failed"}
        node = PatternAnalyzerNode()

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_detect_low_rating_pattern(self, base_state, low_rating_feedback):
        """Test detection of low rating pattern."""
        state = {
            **base_state,
            "feedback_items": low_rating_feedback,
            "feedback_summary": {
                "total_count": len(low_rating_feedback),
                "by_type": {"rating": len(low_rating_feedback)},
                "by_agent": {"explainer": len(low_rating_feedback)},
                "average_rating": 1.5,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        patterns = result["detected_patterns"]
        assert len(patterns) > 0
        # Should detect accuracy issue from low ratings
        accuracy_patterns = [p for p in patterns if p["pattern_type"] == "accuracy_issue"]
        assert len(accuracy_patterns) > 0
        assert accuracy_patterns[0]["severity"] in ["medium", "high"]

    @pytest.mark.asyncio
    async def test_detect_correction_pattern(self, base_state, correction_heavy_feedback):
        """Test detection of correction-heavy pattern."""
        state = {
            **base_state,
            "feedback_items": correction_heavy_feedback,
            "feedback_summary": {
                "total_count": len(correction_heavy_feedback),
                "by_type": {"correction": len(correction_heavy_feedback)},
                "by_agent": {"causal_impact": len(correction_heavy_feedback)},
                "average_rating": None,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        patterns = result["detected_patterns"]
        assert len(patterns) > 0
        # Should detect accuracy issue from corrections
        accuracy_patterns = [p for p in patterns if p["pattern_type"] == "accuracy_issue"]
        assert len(accuracy_patterns) > 0

    @pytest.mark.asyncio
    async def test_detect_outcome_error_pattern(self, base_state, outcome_error_feedback):
        """Test detection of outcome error pattern."""
        state = {
            **base_state,
            "feedback_items": outcome_error_feedback,
            "feedback_summary": {
                "total_count": len(outcome_error_feedback),
                "by_type": {"outcome": len(outcome_error_feedback)},
                "by_agent": {"prediction_synthesizer": len(outcome_error_feedback)},
                "average_rating": None,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        patterns = result["detected_patterns"]
        assert len(patterns) > 0
        # Should detect accuracy issue from prediction errors
        accuracy_patterns = [p for p in patterns if p["pattern_type"] == "accuracy_issue"]
        assert len(accuracy_patterns) > 0

    @pytest.mark.asyncio
    async def test_detect_agent_specific_pattern(self, base_state):
        """Test detection of agent-specific high negative feedback rate."""
        # Create feedback where one agent has many negative ratings
        feedback_items = [
            {
                "feedback_id": f"F{i:03d}",
                "source_agent": "problematic_agent",
                "query": f"Query {i}",
                "agent_response": f"Response {i}",
                "user_feedback": 1 if i < 6 else 4,  # 6 low, 4 high
                "feedback_type": "rating",
                "timestamp": f"2024-01-{15 + i % 15:02d}T10:00:00Z",
            }
            for i in range(10)
        ]

        state = {
            **base_state,
            "feedback_items": feedback_items,
            "feedback_summary": {
                "total_count": 10,
                "by_type": {"rating": 10},
                "by_agent": {"problematic_agent": 10},
                "average_rating": 2.2,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        patterns = result["detected_patterns"]
        # Should detect relevance issue for high negative rate agent
        relevance_patterns = [p for p in patterns if p["pattern_type"] == "relevance_issue"]
        assert len(relevance_patterns) > 0
        assert "problematic_agent" in relevance_patterns[0]["affected_agents"]

    @pytest.mark.asyncio
    async def test_agent_specific_pattern_counts_thumbs_strings(self, base_state):
        """The per-agent negative detector must normalize thumbs strings like
        the overall low-rating detector does — an isinstance gate silently
        dropped every thumbs_down, hiding per-agent negative streaks (codex
        round-1 MED)."""
        feedback_items = [
            {
                "feedback_id": f"F{i:03d}",
                "source_agent": "problematic_agent",
                "query": f"Query {i}",
                "agent_response": f"Response {i}",
                "user_feedback": "thumbs_down" if i < 6 else "thumbs_up",
                "feedback_type": "rating",
                "timestamp": f"2024-01-{15 + i % 15:02d}T10:00:00Z",
            }
            for i in range(10)
        ]

        state = {
            **base_state,
            "feedback_items": feedback_items,
            "feedback_summary": {
                "total_count": 10,
                "by_type": {"rating": 10},
                "by_agent": {"problematic_agent": 10},
                "average_rating": 2.6,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        relevance_patterns = [
            p for p in result["detected_patterns"] if p["pattern_type"] == "relevance_issue"
        ]
        assert len(relevance_patterns) > 0
        assert "problematic_agent" in relevance_patterns[0]["affected_agents"]

    @pytest.mark.asyncio
    async def test_pattern_clustering(self, state_with_feedback):
        """Test that patterns are properly clustered by type."""
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state_with_feedback)

        clusters = result["pattern_clusters"]
        patterns = result["detected_patterns"]

        # All pattern IDs should be in clusters
        all_ids_in_clusters = set()
        for ids in clusters.values():
            all_ids_in_clusters.update(ids)

        for pattern in patterns:
            assert pattern["pattern_id"] in all_ids_in_clusters

    @pytest.mark.asyncio
    async def test_llm_mode_fallback(self, state_with_feedback, mock_llm):
        """Test LLM mode falls back to deterministic on error."""
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        node = PatternAnalyzerNode(use_llm=True, llm=mock_llm)

        result = await node.execute(state_with_feedback)

        # Should fall back to deterministic mode
        assert result["status"] == "extracting"
        assert result["model_used"] == "deterministic"

    @pytest.mark.asyncio
    async def test_llm_mode_success(self, state_with_feedback, mock_llm):
        """Test LLM mode when successful."""
        node = PatternAnalyzerNode(use_llm=True, llm=mock_llm)

        result = await node.execute(state_with_feedback)

        assert result["status"] == "extracting"
        assert len(result["detected_patterns"]) > 0
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_mode_without_llm_instance(self, state_with_feedback):
        """Test LLM mode without LLM instance falls back to deterministic."""
        node = PatternAnalyzerNode(use_llm=True, llm=None)

        result = await node.execute(state_with_feedback)

        assert result["status"] == "extracting"
        assert result["model_used"] == "deterministic"

    @pytest.mark.asyncio
    async def test_pattern_structure(self, state_with_feedback):
        """Test that detected patterns have correct structure."""
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state_with_feedback)

        for pattern in result["detected_patterns"]:
            assert "pattern_id" in pattern
            assert "pattern_type" in pattern
            assert "description" in pattern
            assert "frequency" in pattern
            assert "severity" in pattern
            assert "affected_agents" in pattern
            assert "example_feedback_ids" in pattern
            assert "root_cause_hypothesis" in pattern

    @pytest.mark.asyncio
    async def test_error_handling(self, base_state):
        """Test error handling in pattern analysis."""
        # Create invalid state that would cause an error
        state = {
            **base_state,
            "feedback_items": "invalid",  # Should be a list
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0
        assert result["errors"][0]["node"] == "pattern_analyzer"

    @pytest.mark.asyncio
    async def test_latency_tracking(self, state_with_feedback):
        """Test that latency is properly tracked."""
        node = PatternAnalyzerNode(use_llm=False)

        result = await node.execute(state_with_feedback)

        assert "analysis_latency_ms" in result
        assert isinstance(result["analysis_latency_ms"], int)
        assert result["analysis_latency_ms"] >= 0


class TestLLMEnumValidation:
    """The LLM/DSPy paths emit free-form values into Literal-constrained
    contract fields. Measured 2026-06-11 (canonical tier1-5 runs): the LLM
    invented pattern_type='baseline_establishment' → pydantic literal_error
    at FeedbackLearnerOutput validation, failing the whole agent. Out-of-
    contract pattern types must be DROPPED with a log (the category IS the
    semantic payload — clamping would fabricate meaning); out-of-contract
    severities clamp to 'medium' (the pattern is real, the adjective is
    malformed)."""

    def _analyzer(self):
        from src.agents.feedback_learner.nodes.pattern_analyzer import PatternAnalyzerNode

        return PatternAnalyzerNode.__new__(PatternAnalyzerNode)  # no LLM client needed

    def test_hallucinated_pattern_type_is_dropped(self):
        content = """```json
{"patterns": [
  {"pattern_id": "P1", "pattern_type": "baseline_establishment",
   "description": "hallucinated", "frequency": 2, "severity": "low",
   "affected_agents": [], "example_feedback_ids": [], "root_cause_hypothesis": ""},
  {"pattern_id": "P2", "pattern_type": "accuracy_issue",
   "description": "real", "frequency": 3, "severity": "high",
   "affected_agents": [], "example_feedback_ids": [], "root_cause_hypothesis": ""}
]}
```"""
        patterns = self._analyzer()._parse_patterns(content)
        assert [p["pattern_id"] for p in patterns] == ["P2"]
        assert patterns[0]["pattern_type"] == "accuracy_issue"

    def test_invalid_severity_drops_pattern(self):
        """codex R4: severity is LOAD-BEARING (learning_extractor emits the
        model_retrain recommendation only for high/critical) — clamping an
        invented severity would fabricate that decision, so the pattern
        drops (counted in pattern_parse_anomalies)."""
        content = """```json
{"patterns": [
  {"pattern_id": "P1", "pattern_type": "coverage_gap",
   "description": "ok", "frequency": 1, "severity": "catastrophic",
   "affected_agents": [], "example_feedback_ids": [], "root_cause_hypothesis": ""}
]}
```"""
        analyzer = self._analyzer()
        patterns = analyzer._parse_patterns(content)
        assert patterns == []
        assert analyzer._enum_drop_count == 1

    def test_all_dropped_surfaces_parse_anomalies_not_clean_success(self, state_with_feedback):
        """codex R4 fail-open guard: when the LLM emits ONLY out-of-contract
        patterns, the node must surface pattern_parse_anomalies so '0
        patterns detected' is distinguishable from a clean no-findings run."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from src.agents.feedback_learner.nodes.pattern_analyzer import PatternAnalyzerNode

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content="""```json
{"patterns": [
  {"pattern_id": "P1", "pattern_type": "baseline_establishment",
   "description": "hallucinated", "frequency": 2, "severity": "low",
   "affected_agents": [], "example_feedback_ids": [], "root_cause_hypothesis": ""}
]}
```"""
            )
        )
        node = PatternAnalyzerNode(use_llm=True, llm=mock_llm, prefer_optimized=False)
        result = asyncio.run(node.execute(state_with_feedback))
        assert result["status"] == "extracting"  # NOT failed — drops are not crashes
        assert result["detected_patterns"] == []
        assert result["pattern_parse_anomalies"]["dropped_out_of_contract"] == 1

    def test_all_valid_pass_through_unchanged(self):
        content = """```json
{"patterns": [
  {"pattern_id": "P1", "pattern_type": "latency_issue",
   "description": "ok", "frequency": 1, "severity": "critical",
   "affected_agents": ["a"], "example_feedback_ids": ["f1"], "root_cause_hypothesis": "h"}
]}
```"""
        patterns = self._analyzer()._parse_patterns(content)
        assert patterns[0]["pattern_type"] == "latency_issue"
        assert patterns[0]["severity"] == "critical"


def _rating_item(i: int, rating, agent: str = "agent1", metadata=None) -> dict:
    """One rating-type feedback item; metadata omitted entirely when None
    (mirrors legacy items that predate surface metadata)."""
    item = {
        "feedback_id": f"F{i:03d}",
        "source_agent": agent,
        "query": f"Query {i}",
        "agent_response": f"Response {i}",
        "user_feedback": rating,
        "feedback_type": "rating",
        "timestamp": f"2024-01-{15 + i % 15:02d}T10:00:00Z",
    }
    if metadata is not None:
        item["metadata"] = metadata
    return item


# Metadata shapes as LearningSignalsFeedbackStore emits them: cognitive rows
# carry source_path=None (the Reflector never sets it), copilot rows carry
# the explicit marker written by _collect_copilot_learning_signal (#1240).
_COPILOT_META = {
    "source": "learning_signals",
    "signal_component": "agent",
    "source_path": "copilotkit",
    "reward": 0.35,
}
_COGNITIVE_META = {
    "source": "learning_signals",
    "signal_component": "agent",
    "source_path": None,
    "reward": 0.95,
}


class TestRatingSurface:
    """#1251: surface derivation + ceiling map live at the aggregation layer."""

    def test_copilot_metadata_maps_to_copilot_surface(self):
        from src.agents.feedback_learner.rating_utils import rating_surface

        assert rating_surface(_COPILOT_META) == "copilotkit"

    def test_learning_signals_without_source_path_is_cognitive(self):
        from src.agents.feedback_learner.rating_utils import rating_surface

        assert rating_surface(_COGNITIVE_META) == "cognitive"
        assert rating_surface({"source": "learning_signals"}) == "cognitive"

    def test_everything_else_is_explicit(self):
        from src.agents.feedback_learner.rating_utils import rating_surface

        assert rating_surface(None) == "explicit"
        assert rating_surface({}) == "explicit"
        assert rating_surface({"source": "chatbot_ui"}) == "explicit"

    def test_ceiling_map_pins_copilot_honest_ceiling(self):
        """Copilot's reward ceiling is 0.8 (#1240) → rating ceiling 4.2; the
        other surfaces reach the full 5.0. Any future top-anchored consumer
        must read these, so pin them."""
        from src.agents.feedback_learner.rating_utils import (
            SURFACE_RATING_CEILINGS,
            rating_surface,
        )

        assert SURFACE_RATING_CEILINGS["copilotkit"] == pytest.approx(4.2)
        assert SURFACE_RATING_CEILINGS["cognitive"] == 5.0
        assert SURFACE_RATING_CEILINGS["explicit"] == 5.0
        # every derivable surface has a declared ceiling (fail-closed pairing)
        for meta in (None, _COPILOT_META, _COGNITIVE_META):
            assert rating_surface(meta) in SURFACE_RATING_CEILINGS


class TestSourceAwareRatingAggregation:
    """#1251: the low-ratings gate groups by reward surface so a low pool on
    one surface is never masked by a high pool on another (the pooled mean's
    distance-to-gate must not depend on source mix)."""

    @pytest.mark.asyncio
    async def test_low_copilot_pool_not_masked_by_high_cognitive_pool(self, base_state):
        """Headline #1251 case: 6 cognitive items @4.8 + 4 copilot items @2.4
        pool to 3.84 — today's pooled gate stays silent and the copilot low
        streak is invisible. Source-aware grouping must flag the copilot pool."""
        feedback_items = [
            _rating_item(i, 4.8, agent="gap_analyzer", metadata=dict(_COGNITIVE_META))
            for i in range(6)
        ] + [
            _rating_item(10 + i, 2.4, agent="copilotkit", metadata=dict(_COPILOT_META))
            for i in range(4)
        ]
        state = {
            **base_state,
            "feedback_items": feedback_items,
            "feedback_summary": {
                "total_count": 10,
                "by_type": {"rating": 10},
                "by_agent": {"gap_analyzer": 6, "copilotkit": 4},
                "average_rating": 3.84,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False, prefer_optimized=False)

        result = await node.execute(state)

        low_rating = [
            p
            for p in result["detected_patterns"]
            if p["pattern_type"] == "accuracy_issue" and "Low average" in p["description"]
        ]
        assert len(low_rating) == 1
        pattern = low_rating[0]
        assert pattern["frequency"] == 4  # copilot pool only, not all 10
        assert pattern["affected_agents"] == ["copilotkit"]
        assert pattern["severity"] == "medium"  # pool avg 2.4 (>= 2.0)
        assert "copilotkit" in pattern["description"]

    @pytest.mark.asyncio
    async def test_single_surface_pool_behavior_unchanged(self, base_state):
        """Acceptance #2 regression: a single-surface pool must gate exactly
        as before — one pattern over the whole pool, severity from pool avg."""
        feedback_items = [_rating_item(i, 1.5, agent="explainer") for i in range(4)]
        state = {
            **base_state,
            "feedback_items": feedback_items,
            "feedback_summary": {
                "total_count": 4,
                "by_type": {"rating": 4},
                "by_agent": {"explainer": 4},
                "average_rating": 1.5,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False, prefer_optimized=False)

        result = await node.execute(state)

        low_rating = [
            p
            for p in result["detected_patterns"]
            if p["pattern_type"] == "accuracy_issue" and "Low average" in p["description"]
        ]
        assert len(low_rating) == 1
        assert low_rating[0]["frequency"] == 4
        assert low_rating[0]["severity"] == "high"  # pool avg 1.5 < 2.0
        assert low_rating[0]["affected_agents"] == ["explainer"]

    @pytest.mark.asyncio
    async def test_two_low_surfaces_emit_one_pattern_each(self, base_state):
        """Each low pool gates independently with ITS OWN severity: pooled
        (today) this collapses into ONE high-severity pattern over all 6."""
        feedback_items = [_rating_item(i, 1.0, agent="expl_agent") for i in range(3)] + [
            _rating_item(10 + i, 2.5, agent="copilotkit", metadata=dict(_COPILOT_META))
            for i in range(3)
        ]
        state = {
            **base_state,
            "feedback_items": feedback_items,
            "feedback_summary": {
                "total_count": 6,
                "by_type": {"rating": 6},
                "by_agent": {"expl_agent": 3, "copilotkit": 3},
                "average_rating": 1.75,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False, prefer_optimized=False)

        result = await node.execute(state)

        low_rating = [
            p
            for p in result["detected_patterns"]
            if p["pattern_type"] == "accuracy_issue" and "Low average" in p["description"]
        ]
        assert len(low_rating) == 2
        # deterministic surface order: copilotkit sorts before explicit
        assert low_rating[0]["affected_agents"] == ["copilotkit"]
        assert low_rating[0]["severity"] == "medium"  # copilot pool avg 2.5
        assert low_rating[1]["affected_agents"] == ["expl_agent"]
        assert low_rating[1]["severity"] == "high"  # explicit pool avg 1.0
        assert low_rating[0]["pattern_id"] != low_rating[1]["pattern_id"]

    @pytest.mark.asyncio
    async def test_healthy_pools_on_both_surfaces_stay_silent(self, base_state):
        """Grouping must not FABRICATE patterns: two healthy pools (each avg
        >= 3.0) emit nothing, exactly as the pooled gate did."""
        feedback_items = [
            _rating_item(i, 4.6, agent="gap_analyzer", metadata=dict(_COGNITIVE_META))
            for i in range(3)
        ] + [
            _rating_item(10 + i, 3.8, agent="copilotkit", metadata=dict(_COPILOT_META))
            for i in range(3)
        ]
        state = {
            **base_state,
            "feedback_items": feedback_items,
            "feedback_summary": {
                "total_count": 6,
                "by_type": {"rating": 6},
                "by_agent": {"gap_analyzer": 3, "copilotkit": 3},
                "average_rating": 4.2,
            },
            "status": "analyzing",
        }
        node = PatternAnalyzerNode(use_llm=False, prefer_optimized=False)

        result = await node.execute(state)

        low_rating = [
            p
            for p in result["detected_patterns"]
            if p["pattern_type"] == "accuracy_issue" and "Low average" in p["description"]
        ]
        assert low_rating == []
