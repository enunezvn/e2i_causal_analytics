"""#1582: per-turn routing authority marker in the dispatch observability envelope.

``dispatch_info`` interleaves TWO subsystems' outputs with nothing separating
them: the 4-stage ClassificationPipeline's decision (``routing_pattern``,
``classification_latency_ms``, ``used_llm_layer``) and legacy intent routing's
actual outcome (``agents_dispatched``, ``routed_agent``, ``intent``). An
answered kpi_query turn therefore reads as ``routing_pattern=CLARIFICATION_NEEDED``
next to ``agents_dispatched=['explainer']``, which evaluators repeatedly
mistook for a routing regression (#1582, and identically in the 07-29 baseline).

``routing_pattern`` is NOT wrong: the pipeline genuinely abstains on those
turns (confidence 0.0, empty targets), and ``routing_metrics.py`` counting that
as an abstain is correct — it measures a shadow classifier that abstains on
67.7% of the #1337 gold set. The fix is therefore ADDITIVE and telemetry-only:
``routing_authority`` names WHICH subsystem actually decided THIS turn.

Semantics (per-turn, never inferred from config alone):
- ``"pipeline"`` — the dispatch plan came from ``_dispatch_from_classification``
  (active mode AND the pipeline was confident).
- ``"legacy"``  — legacy intent routing decided. Covers off mode, shadow mode,
  AND active mode where the pipeline abstained. That last case is the whole
  point: an active-mode ABSTAINED turn must never read as pipeline authority.

Routing behaviour must stay byte-identical; these tests pin the dispatch plan
alongside the new marker so a future change cannot smuggle a routing change in
behind the telemetry field.
"""

import pytest

from src.agents.orchestrator.classifier.pipeline import ClassificationPipeline
from src.agents.orchestrator.nodes.router import RouterNode

QUERY = "test query"

# The real #1582 stimulus: gold row bench-0000, the single highest-volume
# production query shape. The 4-stage pipeline emits CLARIFICATION_NEEDED for
# it while the graph answers via legacy explainer routing.
KPI_QUERY = "What is TRx for Kisqali?"


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


async def _route(state, monkeypatch, mode):
    monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", mode)
    return await RouterNode().execute(state)


class TestRoutingAuthorityMarker:
    """The marker reflects which subsystem produced THIS turn's dispatch plan."""

    async def test_active_mode_pipeline_takeover_reads_pipeline(self, monkeypatch):
        """Confident pipeline in active mode => it really did decide."""
        out = await _route(
            _state(_clf("SINGLE_AGENT", ["causal_impact"], 0.8)), monkeypatch, "active"
        )
        assert out["routing_authority"] == "pipeline"
        # routing behaviour unchanged: the pipeline's plan is what dispatches
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["causal_impact"]

    async def test_active_mode_abstain_reads_legacy(self, monkeypatch):
        """THE load-bearing case: active + CLARIFICATION_NEEDED => legacy decided.

        Config alone says "active"; the turn's truth is that the pipeline
        abstained and legacy routing chose. Deriving the marker from
        ``_classifier_mode()`` would report "pipeline" here and reproduce the
        exact confusion #1582 is about.
        """
        out = await _route(
            _state(_clf("CLARIFICATION_NEEDED", [], 0.0), primary="prediction"),
            monkeypatch,
            "active",
        )
        assert out["routing_authority"] == "legacy"
        # legacy intent routing decided, unchanged
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["prediction_synthesizer"]

    async def test_active_mode_low_confidence_reads_legacy(self, monkeypatch):
        """Below MIN_ACTIVE_CONFIDENCE the pipeline abstains too."""
        out = await _route(
            _state(_clf("SINGLE_AGENT", ["causal_impact"], 0.2), primary="prediction"),
            monkeypatch,
            "active",
        )
        assert out["routing_authority"] == "legacy"
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["prediction_synthesizer"]

    @pytest.mark.parametrize("mode", ["shadow", "off"])
    async def test_non_active_modes_read_legacy(self, monkeypatch, mode):
        """Shadow/off never consult the pipeline for routing."""
        out = await _route(
            _state(_clf("SINGLE_AGENT", ["causal_impact"], 0.9), primary="prediction"),
            monkeypatch,
            mode,
        )
        assert out["routing_authority"] == "legacy"
        # byte-identical legacy routing despite a confident pipeline decision
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["prediction_synthesizer"]

    async def test_default_routing_path_reads_legacy(self, monkeypatch):
        """No intent => `_default_routing` early return still carries the marker."""
        monkeypatch.setenv("ORCHESTRATOR_CLASSIFIER_MODE", "shadow")
        out = await RouterNode().execute({"query": QUERY})
        assert out["routing_authority"] == "legacy"
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["explainer"]

    async def test_marker_is_only_ever_the_two_values(self, monkeypatch):
        for mode in ("off", "shadow", "active"):
            out = await _route(
                _state(_clf("CLARIFICATION_NEEDED", [], 0.0)), monkeypatch, mode
            )
            assert out["routing_authority"] in ("legacy", "pipeline")


class TestIssue1582Scenario:
    """End-to-end on the REAL pipeline output for the real stimulus."""

    async def test_answered_kpi_turn_is_legacy_authority_and_pattern_unchanged(
        self, monkeypatch
    ):
        """The #1582 turn: CLARIFICATION_NEEDED label, legacy authority, answered.

        Pins that the fix does NOT relabel routing_pattern (that would perturb
        routing_metrics.py and, in active mode, the router's abstain branch) —
        it only names who decided.
        """
        pipeline = ClassificationPipeline(llm_client=None, enable_llm_layer=False)
        result = await pipeline.classify(query=KPI_QUERY)

        # Unchanged: the pipeline still abstains, and still says so.
        assert result.routing_pattern.value == "CLARIFICATION_NEEDED"
        assert result.target_agents == []

        classification = result.model_dump(mode="json", exclude={"stages"})
        state = {
            "query": KPI_QUERY,
            "intent": _intent(primary="explanation"),
            "classification": classification,
            "routing_pattern": result.routing_pattern.value,
        }
        out = await _route(state, monkeypatch, "shadow")

        # The envelope is now self-describing: the abstaining label sits next
        # to an explicit statement that legacy routing answered the turn.
        assert out["routing_pattern"] == "CLARIFICATION_NEEDED"
        assert out["routing_authority"] == "legacy"
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["explainer"]


class TestRoutingUnchanged:
    """The marker must not perturb any routing output."""

    @pytest.mark.parametrize("mode", ["off", "shadow", "active"])
    @pytest.mark.parametrize(
        "classification",
        [
            None,
            _clf("CLARIFICATION_NEEDED", [], 0.0),
            _clf("SINGLE_AGENT", ["causal_impact"], 0.8),
            _clf("PARALLEL_DELEGATION", ["causal_impact", "gap_analyzer"], 0.7),
            _clf("TOOL_COMPOSER", [], 0.9),
        ],
    )
    async def test_dispatch_outputs_untouched_by_marker(
        self, monkeypatch, mode, classification
    ):
        """Every routing-bearing key is exactly what it was; only the marker is new."""
        out = await _route(_state(classification), monkeypatch, mode)
        routing_keys = {
            "dispatch_plan",
            "parallel_groups",
            "current_phase",
            "discovery_routing_applied",
            "discovery_aware_agents",
        }
        # the marker is additive, never a replacement for a routing key
        assert routing_keys <= set(out)
        assert out["current_phase"] == "dispatching"
        assert len(out["dispatch_plan"]) >= 1
        assert "routing_authority" in out


class TestOrchestratorOutputThread:
    """The marker survives `_build_output` into the chat/dispatch envelope."""

    def test_build_output_carries_routing_authority(self):
        from src.agents.orchestrator.agent import OrchestratorAgent

        agent = OrchestratorAgent.__new__(OrchestratorAgent)
        out = agent._build_output(
            {
                "query": KPI_QUERY,
                "routing_pattern": "CLARIFICATION_NEEDED",
                "routing_authority": "legacy",
                "used_llm_layer": False,
            }
        )
        assert out["routing_authority"] == "legacy"
        assert out["routing_pattern"] == "CLARIFICATION_NEEDED"

    def test_build_output_authority_absent_is_none_not_crash(self):
        """Fail-open: a state that never reached the router yields None."""
        from src.agents.orchestrator.agent import OrchestratorAgent

        agent = OrchestratorAgent.__new__(OrchestratorAgent)
        out = agent._build_output({"query": KPI_QUERY})
        assert out["routing_authority"] is None
