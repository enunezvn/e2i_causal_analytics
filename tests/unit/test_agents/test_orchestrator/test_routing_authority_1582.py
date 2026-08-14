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
# production query shape. #1593 taught DomainMapper the KPI-value-lookup SSOT,
# so the pipeline now decides SINGLE_AGENT[explainer] here instead of
# abstaining — which makes this the STRONGER form of the #1582 pin: even a
# confident pipeline verdict must not be credited as pipeline authority in
# shadow mode.
KPI_QUERY = "What is TRx for Kisqali?"

# Gold row bench-0008 — a population breakdown #1593 deliberately keeps
# abstaining on (gold cohort_profiler; a lone explainer would be a measured
# active-mode degradation). It carries the original #1582 scenario forward:
# an abstaining label sitting beside an answered turn.
ABSTAINING_QUERY = "Give me an NRx breakdown by patient clinical segment for Remibrutinib"


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
            out = await _route(_state(_clf("CLARIFICATION_NEEDED", [], 0.0)), monkeypatch, mode)
            assert out["routing_authority"] in ("legacy", "pipeline")


class TestIssue1582Scenario:
    """End-to-end on the REAL pipeline output for the real stimulus."""

    async def _shadow_route(self, query, monkeypatch):
        pipeline = ClassificationPipeline(llm_client=None, enable_llm_layer=False)
        result = await pipeline.classify(query=query)
        state = {
            "query": query,
            "intent": _intent(primary="explanation"),
            "classification": result.model_dump(mode="json", exclude={"stages"}),
            "routing_pattern": result.routing_pattern.value,
        }
        return result, await _route(state, monkeypatch, "shadow")

    async def test_confident_pipeline_verdict_is_still_legacy_authority_in_shadow(
        self, monkeypatch
    ):
        """#1593 made this turn a CONFIDENT pipeline decision. Shadow mode must
        still credit legacy — authority is per-turn provenance, never a
        restatement of what the pipeline happened to conclude."""
        result, out = await self._shadow_route(KPI_QUERY, monkeypatch)

        assert result.routing_pattern.value == "SINGLE_AGENT"
        assert result.target_agents == ["explainer"]

        assert out["routing_pattern"] == "SINGLE_AGENT"
        assert out["routing_authority"] == "legacy"
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["explainer"]

    async def test_abstaining_turn_is_legacy_authority_and_pattern_unchanged(self, monkeypatch):
        """The original #1582 shape, on a query #1593 keeps abstaining:
        CLARIFICATION_NEEDED label, legacy authority, answered turn.

        Pins that #1593 does NOT relabel routing_pattern wholesale — genuine
        abstentions still say so, so routing_metrics.py keeps measuring a real
        abstain rate rather than a retrofitted one.
        """
        result, out = await self._shadow_route(ABSTAINING_QUERY, monkeypatch)

        assert result.routing_pattern.value == "CLARIFICATION_NEEDED"
        assert result.target_agents == []

        # The envelope is self-describing: the abstaining label sits next to an
        # explicit statement that legacy routing answered the turn.
        assert out["routing_pattern"] == "CLARIFICATION_NEEDED"
        assert out["routing_authority"] == "legacy"
        assert [d["agent_name"] for d in out["dispatch_plan"]] == ["explainer"]


# Exact dispatch expectations as (agent_name, priority, timeout_ms, fallback_agent).
# Spelled out rather than derived so a change to INTENT_TO_AGENTS or to the
# active-mode branch has to edit THIS table — a self-derived expectation would
# move with the regression it is supposed to catch.
_LEGACY_PREDICTION = [("prediction_synthesizer", "critical", 15000, None)]
_PIPELINE_SINGLE = [("causal_impact", "critical", 300000, "explainer")]
_PIPELINE_PARALLEL = [
    ("causal_impact", "critical", 300000, "explainer"),
    ("gap_analyzer", "high", 20000, None),
]
_PIPELINE_COMPOSER = [("tool_composer", "critical", 180000, "explainer")]

_CLF_NONE = None
_CLF_ABSTAIN = _clf("CLARIFICATION_NEEDED", [], 0.0)
_CLF_LOWCONF = _clf("SINGLE_AGENT", ["causal_impact"], 0.2)
_CLF_SINGLE = _clf("SINGLE_AGENT", ["causal_impact"], 0.8)
_CLF_PARALLEL = _clf("PARALLEL_DELEGATION", ["causal_impact", "gap_analyzer"], 0.7)
_CLF_COMPOSER = _clf("TOOL_COMPOSER", [], 0.9)

# (mode, classification, expected dispatch tuples, expected authority)
_DISPATCH_MATRIX = [
    # off/shadow ignore the pipeline entirely, however confident it is
    ("off", _CLF_NONE, _LEGACY_PREDICTION, "legacy"),
    ("off", _CLF_SINGLE, _LEGACY_PREDICTION, "legacy"),
    ("off", _CLF_PARALLEL, _LEGACY_PREDICTION, "legacy"),
    ("shadow", _CLF_NONE, _LEGACY_PREDICTION, "legacy"),
    ("shadow", _CLF_SINGLE, _LEGACY_PREDICTION, "legacy"),
    ("shadow", _CLF_PARALLEL, _LEGACY_PREDICTION, "legacy"),
    ("shadow", _CLF_COMPOSER, _LEGACY_PREDICTION, "legacy"),
    # active: abstentions fall through to the SAME legacy plan
    ("active", _CLF_NONE, _LEGACY_PREDICTION, "legacy"),
    ("active", _CLF_ABSTAIN, _LEGACY_PREDICTION, "legacy"),
    ("active", _CLF_LOWCONF, _LEGACY_PREDICTION, "legacy"),
    # active: confident pipeline takes authority
    ("active", _CLF_SINGLE, _PIPELINE_SINGLE, "pipeline"),
    ("active", _CLF_PARALLEL, _PIPELINE_PARALLEL, "pipeline"),
    ("active", _CLF_COMPOSER, _PIPELINE_COMPOSER, "pipeline"),
]


class TestRoutingUnchanged:
    """The marker must not perturb any routing output."""

    @pytest.mark.parametrize(
        "mode,classification,expected_dispatch,expected_authority",
        _DISPATCH_MATRIX,
    )
    async def test_dispatch_plan_is_exactly_unchanged(
        self, monkeypatch, mode, classification, expected_dispatch, expected_authority
    ):
        """Byte-identical dispatch across every mode x pipeline-decision cell.

        Asserts the FULL per-agent config (name, priority, timeout, fallback),
        not merely that something was dispatched — the marker must be provably
        additive, and #1582 would be a bad trade if it moved any SLA.
        """
        out = await _route(_state(classification), monkeypatch, mode)

        actual = [
            (d["agent_name"], d["priority"], d["timeout_ms"], d["fallback_agent"])
            for d in out["dispatch_plan"]
        ]
        assert actual == expected_dispatch
        assert out["routing_authority"] == expected_authority
        # parallel_groups stays consistent with the plan it describes
        assert [name for group in out["parallel_groups"] for name in group] == [
            d[0] for d in expected_dispatch
        ]
        assert out["current_phase"] == "dispatching"


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
