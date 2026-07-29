"""Mode-gating tests for the 4-stage pipeline wiring in IntentClassifierNode.

ORCHESTRATOR_CLASSIFIER_MODE: off | shadow (default) | active. The pipeline
runs in shadow/active, is fail-open, and only spawns the classification_logs
write outside test environments.
"""

import asyncio
from unittest.mock import AsyncMock, patch

from src.agents.orchestrator.nodes import intent_classifier as ic_module
from src.agents.orchestrator.nodes.intent_classifier import (
    IntentClassifierNode,
    _classifier_mode,
    _should_log_classification,
)

QUERY = "What is the causal impact of rep visits on TRx for Kisqali?"


class TestClassifierMode:
    def test_default_is_shadow(self, monkeypatch):
        monkeypatch.delenv("ORCHESTRATOR_CLASSIFIER_MODE", raising=False)
        assert _classifier_mode() == "shadow"

    def test_explicit_modes(self, monkeypatch):
        for mode in ("off", "shadow", "active"):
            monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", mode)
            assert _classifier_mode() == mode

    def test_normalized(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", "  ACTIVE ")
        assert _classifier_mode() == "active"


class TestShouldLogClassification:
    def test_requires_supabase_url(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("E2I_TESTING_MODE", raising=False)
        assert _should_log_classification() is False

    def test_testing_mode_blocks_logging(self, monkeypatch):
        """Hermeticity (883-A lesson): tests/conftest.py loads the dev .env,
        so SUPABASE_URL alone must not be enough to write real rows."""
        monkeypatch.setenv("SUPABASE_URL", "http://localhost:54321")
        monkeypatch.setenv("E2I_TESTING_MODE", "1")
        assert _should_log_classification() is False

    def test_prod_shape_logs(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "http://localhost:54321")
        monkeypatch.delenv("E2I_TESTING_MODE", raising=False)
        assert _should_log_classification() is True


class TestPipelineWiringInNode:
    async def test_off_mode_no_pipeline_keys(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", "off")
        node = IntentClassifierNode()
        state = await node.execute({"query": QUERY})
        assert "routing_pattern" not in state
        assert "classification" not in state
        assert state["intent"]["primary_intent"] == "causal_effect"

    async def test_shadow_mode_surfaces_classification(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", "shadow")
        node = IntentClassifierNode()
        state = await node.execute({"query": QUERY})
        assert state["routing_pattern"] in {
            "SINGLE_AGENT",
            "PARALLEL_DELEGATION",
            "TOOL_COMPOSER",
            "CLARIFICATION_NEEDED",
        }
        assert isinstance(state["used_llm_layer"], bool)
        clf = state["classification"]
        assert clf["routing_pattern"] == state["routing_pattern"]
        assert clf["classification_latency_ms"] >= 0.0
        # stages excluded from graph state (only the log writer gets them)
        assert clf.get("stages") is None
        # Legacy classification is untouched
        assert state["intent"]["primary_intent"] == "causal_effect"

    async def test_pipeline_failure_is_fail_open(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", "shadow")
        broken = AsyncMock()
        broken.classify.side_effect = RuntimeError("pipeline exploded")
        with patch.object(ic_module, "_get_classification_pipeline", return_value=broken):
            node = IntentClassifierNode()
            state = await node.execute({"query": QUERY})
        assert state["intent"]["primary_intent"] == "causal_effect"
        assert "routing_pattern" not in state
        assert "classification" not in state

    async def test_no_log_task_in_testing_mode(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", "shadow")
        monkeypatch.setenv("E2I_TESTING_MODE", "1")
        recorder = AsyncMock()
        with patch.object(ic_module, "_log_classification", recorder):
            node = IntentClassifierNode()
            await node.execute({"query": QUERY, "session_id": "s", "user_id": "u"})
            await asyncio.sleep(0)
        recorder.assert_not_awaited()

    async def test_log_task_spawned_outside_testing_mode(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", "shadow")
        monkeypatch.setenv("SUPABASE_URL", "http://localhost:54321")
        monkeypatch.delenv("E2I_TESTING_MODE", raising=False)
        recorder = AsyncMock()
        with patch.object(ic_module, "_log_classification", recorder):
            node = IntentClassifierNode()
            await node.execute({"query": QUERY, "session_id": "sess-1", "user_id": "user-1"})
            # Let the fire-and-forget task run
            await asyncio.sleep(0)
        recorder.assert_awaited_once()
        args = recorder.await_args[0]
        assert args[0] == QUERY
        assert args[2] == "sess-1"
        assert args[3] == "user-1"
