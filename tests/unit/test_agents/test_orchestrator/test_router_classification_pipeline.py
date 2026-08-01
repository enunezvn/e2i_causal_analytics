"""Active-mode routing tests: RouterNode consulting the 4-stage classifier.

In active mode a confident ClassificationResult takes routing authority;
CLARIFICATION_NEEDED / low confidence / unknown patterns abstain to legacy
intent routing. Shadow/off modes must be byte-identical to legacy routing.
"""

from src.agents.orchestrator.nodes.router import RouterNode

QUERY = "test query"


def _intent(primary="prediction", requires_multi=False, secondary=None):
    return {
        "primary_intent": primary,
        "confidence": 0.9,
        "secondary_intents": secondary or [],
        "requires_multi_agent": requires_multi,
    }


def _state(classification=None, **kwargs):
    state = {"query": QUERY, "intent": _intent(**kwargs)}
    if classification is not None:
        state["classification"] = classification
    return state


def _clf(pattern, targets, confidence):
    return {
        "routing_pattern": pattern,
        "target_agents": targets,
        "confidence": confidence,
    }


class TestActiveModeDispatch:
    def setup_method(self):
        self.router = RouterNode()

    async def _route(self, state, monkeypatch, mode="active"):
        monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", mode)
        return await self.router.execute(state)

    async def test_single_agent_pattern(self, monkeypatch):
        out = await self._route(_state(_clf("SINGLE_AGENT", ["causal_impact"], 0.8)), monkeypatch)
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["causal_impact"]
        # Canonical per-agent config preserved (timeout from INTENT_TO_AGENTS)
        assert out["dispatch_plan"][0]["timeout_ms"] == 300000  # #1419
        assert out["dispatch_plan"][0]["priority"] == "critical"

    async def test_parallel_delegation_pattern(self, monkeypatch):
        out = await self._route(
            _state(_clf("PARALLEL_DELEGATION", ["causal_impact", "gap_analyzer"], 0.7)),
            monkeypatch,
        )
        names = [d["agent_name"] for d in out["dispatch_plan"]]
        assert names == ["causal_impact", "gap_analyzer"]
        priorities = [d["priority"] for d in out["dispatch_plan"]]
        assert priorities == ["critical", "high"]
        assert out["parallel_groups"] == [["causal_impact"], ["gap_analyzer"]]

    async def test_tool_composer_pattern(self, monkeypatch):
        out = await self._route(_state(_clf("TOOL_COMPOSER", ["tool_composer"], 0.6)), monkeypatch)
        plan = out["dispatch_plan"]
        assert [d["agent_name"] for d in plan] == ["tool_composer"]
        # Canonical multi_faceted dispatch: 3-minute SLA + explainer fallback
        assert plan[0]["timeout_ms"] == 180000
        assert plan[0]["fallback_agent"] == "explainer"

    async def test_heavy_agent_timeout_preserved(self, monkeypatch):
        out = await self._route(
            _state(_clf("SINGLE_AGENT", ["heterogeneous_optimizer"], 0.8)), monkeypatch
        )
        assert out["dispatch_plan"][0]["timeout_ms"] == 420000


class TestActiveModeAbstention:
    def setup_method(self):
        self.router = RouterNode()

    async def _route(self, state, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", "active")
        return await self.router.execute(state)

    async def test_clarification_falls_back_to_legacy(self, monkeypatch):
        out = await self._route(
            _state(_clf("CLARIFICATION_NEEDED", [], 0.0), primary="prediction"), monkeypatch
        )
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["prediction_synthesizer"]

    async def test_low_confidence_falls_back_to_legacy(self, monkeypatch):
        out = await self._route(
            _state(_clf("SINGLE_AGENT", ["causal_impact"], 0.3), primary="prediction"),
            monkeypatch,
        )
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["prediction_synthesizer"]

    async def test_unknown_pattern_falls_back_to_legacy(self, monkeypatch):
        out = await self._route(
            _state(_clf("SOMETHING_NEW", ["causal_impact"], 0.9), primary="prediction"),
            monkeypatch,
        )
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["prediction_synthesizer"]

    async def test_empty_targets_falls_back_to_legacy(self, monkeypatch):
        out = await self._route(
            _state(_clf("SINGLE_AGENT", [], 0.9), primary="prediction"), monkeypatch
        )
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["prediction_synthesizer"]

    async def test_no_classification_falls_back_to_legacy(self, monkeypatch):
        out = await self._route(_state(primary="prediction"), monkeypatch)
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["prediction_synthesizer"]

    async def test_self_dispatch_guard_applies(self, monkeypatch):
        """A classification naming 'orchestrator' must be stripped (#251 F1);
        the guard's empty-plan fallback then routes explainer."""
        out = await self._route(
            _state(_clf("SINGLE_AGENT", ["orchestrator"], 0.9), primary="prediction"),
            monkeypatch,
        )
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["explainer"]


class TestShadowAndOffModesAreLegacy:
    def setup_method(self):
        self.router = RouterNode()

    async def test_shadow_ignores_classification(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", "shadow")
        out = await self.router.execute(
            _state(_clf("SINGLE_AGENT", ["causal_impact"], 0.99), primary="prediction")
        )
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["prediction_synthesizer"]

    async def test_off_ignores_classification(self, monkeypatch):
        monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", "off")
        out = await self.router.execute(
            _state(_clf("TOOL_COMPOSER", ["tool_composer"], 0.99), primary="prediction")
        )
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["prediction_synthesizer"]

    async def test_active_multi_agent_legacy_path_unaffected_on_abstain(self, monkeypatch):
        """Abstention must leave the legacy multi-agent pattern intact."""
        monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", "active")
        state = _state(
            _clf("CLARIFICATION_NEEDED", [], 0.0),
            primary="causal_effect",
            requires_multi=True,
            secondary=["segment_analysis"],
        )
        out = await self.router.execute(state)
        names = [d["agent_name"] for d in out["dispatch_plan"]]
        assert names == ["causal_impact", "heterogeneous_optimizer"]
