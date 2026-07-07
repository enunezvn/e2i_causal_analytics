"""
Regression tests for issue #953 follow-up (#954 deploy exposed two more bugs).

#954 fixed the WORKER TIMEOUT (to_thread) and reached the store_episode
write-back, but the live call still returned error-as-data. Two distinct
bugs surfaced:

BUG A — ``'dict' object has no attribute 'evidence_board'``.
    ``CausalRAG.cognitive_search`` consumes the result of
    ``workflow.ainvoke(...)`` via ATTRIBUTE access (result_state.evidence_board,
    .response, .hop_count, ... 10 fields), but a compiled LangGraph returns a
    **dict** of channel values, not the CognitiveState dataclass. The first
    attribute access (evidence_board) raises AttributeError, which is caught by
    cognitive_search's ``except Exception`` and surfaced in-band as the
    HTTP-200 ``error`` field. It was never reachable before because the
    pipeline always errored earlier (.store / thread_id / WORKER TIMEOUT); the
    #945/#954 fixes ran the graph to completion and exposed it.

BUG B — ``invalid input value for enum e2i_agent_name: "cognitive_rag"`` (22P02).
    #954's store_episode metadata passed ``agent_name="cognitive_rag"``, but
    ``e2i_agent_name`` is a constrained enum that does NOT include
    "cognitive_rag" (it is not a registered agent). The episodic insert fails
    with a postgrest 22P02. ``episodic_memories.agent_name`` is nullable and the
    writer drops None keys, so the fix is to OMIT agent_name (don't invent a
    fake-but-valid agent).

These tests mock only the boundary (LM via stubbed module.forward; DB transport
for the episodic insert) and exercise the REAL logic under test (the
cognitive_search result consumption; the store_episode -> EpisodicMemoryInput
metadata mapping).
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# DSPy import has parallel-worker race conditions; pin to one worker.
pytestmark = pytest.mark.xdist_group(name="dspy_integration")


# =============================================================================
# BUG A — cognitive_search must consume the LangGraph dict result without
#         raising AttributeError, and must surface the real fields.
# =============================================================================


class TestCognitiveSearchConsumesGraphResult:
    """The REAL CausalRAG.cognitive_search consumes a real workflow.ainvoke
    result. A compiled LangGraph returns a dict; the consumer must not raise
    'dict' object has no attribute 'evidence_board' and must return the real
    fields (response/evidence/hop_count/...)."""

    @pytest.mark.asyncio
    async def test_cognitive_search_returns_real_fields_no_error(self):
        import src.rag.cognitive_rag_dspy as mod
        from src.rag.causal_rag import CausalRAG
        from src.rag.cognitive_rag_dspy import Evidence, MemoryType

        # Stub the four DSPy module forwards at class level so the REAL graph
        # wiring + the REAL ainvoke return-type behavior run with no LM. This is
        # the LM boundary only; the graph orchestration and cognitive_search's
        # result consumption (the logic under test) are NOT mocked.
        def _sum_fwd(self, original_query, conversation_context, domain_vocabulary):
            return {
                "rewritten_query": "rewritten q",
                "search_keywords": [],
                "graph_entities": [],
                "extracted_entities": "{}",
                "primary_intent": "CAUSAL_ANALYSIS",
                "secondary_intents": [],
                "requires_visualization": False,
                "complexity": "SIMPLE",
            }

        async def _inv_fwd(self, rewritten_query, intent, entities):
            return {
                "investigation_goal": "goal",
                "evidence_board": [
                    Evidence(
                        source=MemoryType.EPISODIC,
                        hop_number=1,
                        content="Kisqali adoption rose in Q3",
                        relevance_score=0.8,
                    )
                ],
                "hop_count": 1,
                "sufficient_evidence": False,
            }

        async def _agent_fwd(self, state):
            state.response = "A real synthesized answer about Kisqali adoption."
            state.routed_agents = ["orchestrator"]
            return state

        async def _refl_fwd(self, state, user_feedback=None):
            state.worth_remembering = False
            return state

        with (
            patch.object(mod.SummarizerModule, "forward", _sum_fwd),
            patch.object(mod.InvestigatorModule, "forward", _inv_fwd),
            patch.object(mod.AgentModule, "forward", _agent_fwd),
            patch.object(mod.ReflectorModule, "forward", _refl_fwd),
            # Avoid configuring a real DSPy LM; the stubbed forwards never touch
            # an LM, but cognitive_search checks dspy.settings.lm and would try
            # to construct one. Seed a dummy so it skips that branch.
            patch("dspy.settings") as dspy_settings,
        ):
            dspy_settings.lm = object()  # truthy => skip LM construction branch

            rag = CausalRAG()
            result = await rag.cognitive_search(
                query="Why did Kisqali adoption increase?",
                conversation_id="conv-953b",
            )

        # RED before the fix: cognitive_search hits result_state.evidence_board
        # on a dict -> AttributeError -> caught -> result["error"] is set with
        # "'dict' object has no attribute 'evidence_board'".
        # GREEN after: no error key, real fields surfaced.
        assert "error" not in result or result.get("error") is None, (
            f"cognitive_search returned error-as-data: {result.get('error')!r}"
        )
        assert result["response"] == "A real synthesized answer about Kisqali adoption."
        assert result["hop_count"] == 1
        assert len(result["evidence"]) == 1
        assert result["evidence"][0]["content"] == "Kisqali adoption rose in Q3"
        assert result["intent"] == "CAUSAL_ANALYSIS"
        assert result["rewritten_query"] == "rewritten q"
        assert result["routed_agents"] == ["orchestrator"]
        assert result["worth_remembering"] is False


# =============================================================================
# BUG B — store_episode must not write an invalid e2i_agent_name enum.
# =============================================================================


class TestStoreEpisodeAgentNameEnum:
    """The cognitive-RAG episodic write must NOT pass an agent_name that is
    outside the e2i_agent_name enum (cognitive_rag is not a registered agent).
    Passing it triggers a postgrest 22P02 and the episode is lost."""

    # The valid e2i_agent_name enum values (verified against the live DB; see
    # database/migrations/008_agentic_memory_schema.sql + enum-extend migs).
    VALID_AGENTS = {
        "scope_definer",
        "data_preparer",
        "feature_analyzer",
        "model_selector",
        "model_trainer",
        "model_deployer",
        "observability_connector",
        "cohort_constructor",
        "corpus_ingestion",
        "orchestrator",
        "tool_composer",
        "causal_impact",
        "gap_analyzer",
        "drift_monitor",
        "heterogeneous_optimizer",
        "fairness_guardian",
        "health_score",
        "experiment_designer",
        "experiment_monitor",
        "prediction_synthesizer",
        "feedback_learner",
        "explainer",
        "resource_optimizer",
        # "cognitive_rag" is NOT among them.
    }

    # The valid memory_event_type enum values (NOT NULL column; verified against
    # the live DB). "conversation" is NOT among them; "agent_action" is.
    VALID_EVENT_TYPES = {
        "user_query",
        "agent_action",
        "system_event",
        "feedback",
        "error",
        "causal_discovery",
        "trigger_generated",
        "experiment_completed",
        "composition_completed",
        "optimization_completed",
        "explanation_generated",
        "scope_definition_completed",
        "qc_report_completed",
        "model_selection_completed",
        "model_training_completed",
        "feature_analysis_completed",
        "model_deployment_completed",
        "observability_metrics_collected",
        "cohort_construction_completed",
        "causal_analysis_completed",
        "causal_analysis",
        "prediction_completed",
        "prediction_delivered",
        "cate_analysis_completed",
        "health_check_completed",
        "experiment_alert_generated",
        "experiment_monitoring_completed",
        "gap_analysis_completed",
        "orchestration_completed",
    }

    @pytest.mark.asyncio
    async def test_reflector_episodic_writeback_omits_invalid_agent_name(self):
        """The ReflectorModule episodic write-back must not send
        agent_name='cognitive_rag' (which is not in e2i_agent_name)."""
        from src.rag.cognitive_rag_dspy import CognitiveState, ReflectorModule

        captured: Dict[str, Any] = {}

        class CapturingEpisodic:
            async def store_episode(
                self,
                content: str,
                episode_type: str,
                metadata: Optional[Dict[str, Any]] = None,
            ) -> Optional[str]:
                captured["content"] = content
                captured["episode_type"] = episode_type
                captured["metadata"] = metadata or {}
                return "ep-1"

        writers = {
            "episodic": CapturingEpisodic(),
            "semantic": MagicMock(),
            "procedural": MagicMock(),
        }
        collector = MagicMock()

        async def _collect(signals):
            return None

        collector.collect = _collect

        module = ReflectorModule(writers, collector)

        class _Eval:
            worth_remembering = True
            memory_type = "episodic"
            importance_score = 0.8
            key_facts: List[Any] = []

        module.evaluate = MagicMock(return_value=_Eval())  # type: ignore[method-assign]

        state = CognitiveState(user_query="q", conversation_id="c")
        state.response = "r"

        await module.forward(state)

        meta = captured["metadata"]
        # RED before fix: meta["agent_name"] == "cognitive_rag" (invalid enum).
        # GREEN after fix: agent_name is absent OR a valid enum value.
        agent_name = meta.get("agent_name")
        assert agent_name is None or agent_name in self.VALID_AGENTS, (
            f"store_episode metadata carried an invalid e2i_agent_name: {agent_name!r}"
        )
        # event_type maps to the NOT-NULL memory_event_type enum; it must be a
        # valid value (the live DB rejects "conversation" with 22P02 too).
        assert captured["episode_type"] in self.VALID_EVENT_TYPES, (
            f"store_episode used an invalid memory_event_type: {captured['episode_type']!r}"
        )
        # The useful provenance must still be persisted.
        assert meta["query"] == "q"
        assert meta["importance_score"] == pytest.approx(0.8)
        assert meta["session_id"] == "c"

    @pytest.mark.asyncio
    async def test_store_episode_maps_to_valid_episodic_input(self):
        """Through the REAL EpisodicMemoryBackend.store_episode -> the metadata
        is mapped onto an EpisodicMemoryInput whose event_type/agent_name are
        valid against the DB enums (the logic under test).

        We capture the EpisodicMemoryInput constructor kwargs by substituting a
        capturing fake for ``cognitive_backends.EpisodicMemoryInput`` and stub
        the insert boundary, so the test asserts purely on what store_episode
        BUILDS -- immune to any sibling test that has stubbed the heavy memory
        modules in this process.
        """
        from src.rag import cognitive_backends as cb
        from src.rag.cognitive_backends import EpisodicMemoryBackend

        captured: Dict[str, Any] = {}

        class _CaptureInput:
            def __init__(self, **kwargs):
                captured["input_kwargs"] = kwargs
                self.event_type = kwargs.get("event_type")
                self.agent_name = kwargs.get("agent_name")
                self.description = kwargs.get("description")

        async def _capture_insert(memory, text_to_embed=None, session_id=None, cycle_id=None):
            captured["memory"] = memory
            captured["session_id"] = session_id
            return "ep-1"

        backend = EpisodicMemoryBackend()

        # Substitute the input model + insert boundary in cognitive_backends'
        # namespace (where store_episode references them). Context-managed, so
        # restored on exit -- and independent of the real DB/embedding chain.
        with (
            patch.object(cb, "EpisodicMemoryInput", _CaptureInput),
            patch.object(cb, "insert_episodic_memory_with_text", _capture_insert),
        ):
            episode_id = await backend.store_episode(
                content="A real synthesized answer.",
                episode_type="agent_action",
                metadata={
                    "query": "Why did Kisqali adoption increase?",
                    "importance_score": 0.8,
                    "session_id": "conv-953b",
                },
            )

        assert episode_id is not None, "episode insert should succeed"
        memory = captured["memory"]
        # event_type must be a valid memory_event_type ("conversation" is not).
        assert memory.event_type in self.VALID_EVENT_TYPES, (
            f"EpisodicMemoryInput.event_type={memory.event_type!r} "
            "-> would trigger memory_event_type 22P02"
        )
        assert memory.event_type == "agent_action"
        # agent_name must be absent/None or a valid enum (never "cognitive_rag").
        assert memory.agent_name is None or memory.agent_name in self.VALID_AGENTS, (
            f"EpisodicMemoryInput.agent_name={memory.agent_name!r} "
            "-> would trigger e2i_agent_name 22P02"
        )
        # Real provenance still persisted.
        assert memory.description == "A real synthesized answer."
        assert captured["session_id"] == "conv-953b"


# =============================================================================
# Synthesis reward grading — the previous constant `0.8 if response` had zero
# variance, so GEPA gating and feedback-learner pattern analysis learned
# nothing from the `type=agent` signal (live data: 40/40 rows at exactly 0.8).
# =============================================================================


class TestAgentRewardGrading:
    """_collect_training_signals must grade the synthesis (`agent`) reward from
    observable outcome quality, mirroring its summarizer/investigator siblings."""

    def _module(self):
        from src.rag.cognitive_rag_dspy import ReflectorModule

        return ReflectorModule(
            {"episodic": MagicMock(), "semantic": MagicMock(), "procedural": MagicMock()},
            MagicMock(),
        )

    def _agent_signal(self, state):
        signals = self._module()._collect_training_signals(state, user_feedback=None)
        return next(s for s in signals if s["type"] == "agent")

    def test_no_response_scores_zero(self):
        from src.rag.cognitive_rag_dspy import CognitiveState

        state = CognitiveState(user_query="q", conversation_id="c")
        assert self._agent_signal(state)["reward"] == 0.0

    def test_bare_response_scores_base_in_fuel_band(self):
        """Any completed response stays in the GEPA fuel band (>= 0.5), as the
        old constant did — but a bare thin answer no longer scores 0.8."""
        from src.rag.cognitive_rag_dspy import CognitiveState

        state = CognitiveState(user_query="q", conversation_id="c")
        state.response = "short"
        reward = self._agent_signal(state)["reward"]
        assert reward == pytest.approx(0.5)

    def test_grounded_substantive_response_scores_higher(self):
        """Evidence, substance, sufficiency, and an artifact each add credit —
        the signal now has variance for downstream learners."""
        from src.rag.cognitive_rag_dspy import CognitiveState, Evidence, MemoryType

        state = CognitiveState(user_query="q", conversation_id="c")
        state.response = "x" * 250
        state.sufficient_evidence = True
        state.visualization_config = {"type": "bar"}
        state.evidence_board = [
            Evidence(source=list(MemoryType)[0], hop_number=1, content="e", relevance_score=0.9)
            for _ in range(4)
        ]
        reward = self._agent_signal(state)["reward"]
        assert reward == pytest.approx(1.0)

    def test_reward_never_exceeds_one(self):
        from src.rag.cognitive_rag_dspy import CognitiveState, Evidence, MemoryType

        state = CognitiveState(user_query="q", conversation_id="c")
        state.response = "x" * 1000
        state.sufficient_evidence = True
        state.visualization_config = {"type": "bar"}
        state.evidence_board = [
            Evidence(source=list(MemoryType)[0], hop_number=1, content="e", relevance_score=1.0)
            for _ in range(10)
        ]
        assert self._agent_signal(state)["reward"] <= 1.0
