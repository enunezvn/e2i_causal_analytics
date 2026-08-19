"""#1714: the router must honor an explicit caller-requested target agent.

``orchestrator_tool`` (src/api/routes/chatbot_tools.py) stashes the chat
model's explicit choice under ``user_context["target_agent"]`` — but until
#1714 NOTHING in the orchestrator consumed that key, so the request was
silently ignored for EVERY agent and routing fell to intent classification
alone. The 2026-08-19 full eval measured the visible half of that defect:
turn 5.5 dispatched ``target_agent='explainer'`` and got
``agents_dispatched: ['heterogeneous_optimizer', 'gap_analyzer']`` — the
requests that LOOKED honored (experiment_designer 3.4/3.6,
heterogeneous_optimizer 4.4) were intent-routing coincidences, not the
explicit target working.

These tests pin the fixed contract:

- an explicit target naming a router-dispatchable agent takes routing
  authority (``routing_authority == "explicit_target"``) over intent routing,
  multi-agent patterns, AND the active-mode classification pipeline;
- an explicit target survives intent-classification failure (the
  ``_default_routing`` path must not shadow it);
- an unknown / non-dispatchable target falls through to intent routing
  unchanged (the tool payload's ``target_agent_requested`` vs
  ``agents_dispatched`` pair keeps that mismatch visible to the model);
- ``"orchestrator"`` can never be smuggled in as an explicit target
  (issue #251 F1 invariant);
- absent target keeps routing byte-identical (``routing_authority ==
  "legacy"``).
"""

import pytest

import src.agents.orchestrator.nodes.router as router_module
from src.agents.orchestrator.nodes.router import RouterNode


def _intent(primary="segment_analysis", requires_multi=False, secondary=None):
    return {
        "primary_intent": primary,
        "confidence": 0.9,
        "secondary_intents": secondary or [],
        "requires_multi_agent": requires_multi,
    }


def _agent_names(result):
    return [d["agent_name"] for d in result["dispatch_plan"]]


class TestExplicitTargetHonored:
    """The eval-5.5 defect shape: explicit explainer request must dispatch explainer."""

    @pytest.mark.asyncio
    async def test_explicit_explainer_beats_segment_intent(self):
        """RED-first core: 5.5's exact shape — target explainer, segment intent.

        Pre-#1714 this dispatched heterogeneous_optimizer (intent routing);
        the explicit request was silently substituted.
        """
        router = RouterNode()
        state = {
            "intent": _intent("segment_analysis"),
            "user_context": {"target_agent": "explainer"},
        }

        result = await router.execute(state)

        assert _agent_names(result) == ["explainer"]
        assert result["dispatch_plan"][0]["priority"] == "critical"
        # Canonical explainer dispatch config resolved via INTENT_TO_AGENTS.
        assert result["dispatch_plan"][0]["timeout_ms"] == 45000
        assert result["routing_authority"] == "explicit_target"
        assert result["current_phase"] == "dispatching"
        assert result["parallel_groups"] == [["explainer"]]

    @pytest.mark.asyncio
    async def test_explicit_target_beats_multi_agent_pattern(self):
        """Explicit target wins over a hard-coded multi-agent pattern too."""
        router = RouterNode()
        state = {
            "intent": _intent(
                "segment_analysis",
                requires_multi=True,
                secondary=["performance_gap"],
            ),
            "user_context": {"target_agent": "explainer"},
        }

        result = await router.execute(state)

        assert _agent_names(result) == ["explainer"]
        assert result["routing_authority"] == "explicit_target"

    @pytest.mark.asyncio
    async def test_explicit_target_beats_active_classification_pipeline(self, monkeypatch):
        """Explicit target outranks a confident active-mode pipeline verdict."""
        monkeypatch.setattr(router_module, "_classifier_mode", lambda: "active")
        router = RouterNode()
        state = {
            "intent": _intent("segment_analysis"),
            "classification": {
                "routing_pattern": "SINGLE_AGENT",
                "target_agents": ["heterogeneous_optimizer"],
                "confidence": 0.95,
            },
            "user_context": {"target_agent": "explainer"},
        }

        result = await router.execute(state)

        assert _agent_names(result) == ["explainer"]
        assert result["routing_authority"] == "explicit_target"

    @pytest.mark.asyncio
    async def test_explicit_target_survives_missing_intent(self):
        """No classified intent must not shadow the explicit target.

        Pre-#1714 the ``if not intent`` early return dispatched the default
        explainer regardless of the requested agent.
        """
        router = RouterNode()
        state = {
            "intent": None,
            "user_context": {"target_agent": "gap_analyzer"},
        }

        result = await router.execute(state)

        assert _agent_names(result) == ["gap_analyzer"]
        assert result["routing_authority"] == "explicit_target"

    @pytest.mark.asyncio
    async def test_explicit_target_whitespace_and_case_normalized(self):
        """' Explainer ' resolves to the canonical explainer dispatch."""
        router = RouterNode()
        state = {
            "intent": _intent("segment_analysis"),
            "user_context": {"target_agent": "  Explainer "},
        }

        result = await router.execute(state)

        assert _agent_names(result) == ["explainer"]
        assert result["routing_authority"] == "explicit_target"


class TestExplicitTargetFallthrough:
    """Non-dispatchable targets keep intent routing (and its authority marker)."""

    @pytest.mark.asyncio
    async def test_unknown_target_falls_through_to_intent_routing(self):
        router = RouterNode()
        state = {
            "intent": _intent("segment_analysis"),
            "user_context": {"target_agent": "not_a_real_agent"},
        }

        result = await router.execute(state)

        assert _agent_names(result) == ["heterogeneous_optimizer"]
        assert result["routing_authority"] == "legacy"

    @pytest.mark.asyncio
    async def test_orchestrator_target_never_self_dispatches(self):
        """Issue #251 F1: 'orchestrator' is not dispatchable, explicit or not."""
        router = RouterNode()
        state = {
            "intent": _intent("performance_gap"),
            "user_context": {"target_agent": "orchestrator"},
        }

        result = await router.execute(state)

        assert "orchestrator" not in _agent_names(result)
        assert _agent_names(result) == ["gap_analyzer"]
        assert result["routing_authority"] == "legacy"

    @pytest.mark.asyncio
    async def test_non_string_target_ignored(self):
        router = RouterNode()
        state = {
            "intent": _intent("performance_gap"),
            "user_context": {"target_agent": 42},
        }

        result = await router.execute(state)

        assert _agent_names(result) == ["gap_analyzer"]
        assert result["routing_authority"] == "legacy"

    @pytest.mark.asyncio
    async def test_absent_target_keeps_legacy_routing(self):
        """No target: routing stays byte-identical to pre-#1714 behavior."""
        router = RouterNode()
        state = {
            "intent": _intent("segment_analysis"),
            "user_context": {},
        }

        result = await router.execute(state)

        assert _agent_names(result) == ["heterogeneous_optimizer"]
        assert result["routing_authority"] == "legacy"

    @pytest.mark.asyncio
    async def test_missing_intent_and_no_target_defaults_unchanged(self):
        """The _default_routing fallback is untouched when no target exists."""
        router = RouterNode()
        state = {"intent": None, "user_context": {}}

        result = await router.execute(state)

        assert _agent_names(result) == ["explainer"]
        assert result["routing_authority"] == "legacy"
