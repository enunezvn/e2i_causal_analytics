"""
Regression tests for issue #953 — AIAgentInsights cognitive-RAG.

Two pre-existing defects block a real grounded insight from
``POST /api/cognitive/rag``:

1. **Error-as-data (.store AttributeError).** ``ReflectorModule.forward``
   (Phase-4 write-back) called ``memory_writers["episodic"].store({...})``,
   but the real ``EpisodicMemoryBackend`` exposes ``store_episode(content,
   episode_type, metadata)`` — there is no ``.store``. The resulting
   ``AttributeError`` unwound out of the workflow and was surfaced in-band
   as the HTTP-200 ``error`` field by ``CausalRAG.cognitive_search``.
   The sibling semantic/procedural writers had the same class of
   name/signature mismatch (``add_fact`` does not exist; ``store_procedure``
   expects positional fields, not a single dict).

2. **Worker timeout → 502.** The pipeline made ~8 *synchronous* DSPy/LLM
   calls that blocked the uvicorn event loop, starving the gunicorn
   heartbeat → ``WORKER TIMEOUT`` → nginx 502 at ~72s. Every sync DSPy
   call must run off the loop via ``asyncio.to_thread`` so the worker stays
   responsive.

These tests are RED before the fix and GREEN after. They mock only the
*boundary* (the DSPy LM predictors and the memory backends), never the
logic under test (the Reflector write-back routing and the event-loop
scheduling).
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# DSPy import has parallel-worker race conditions; pin to one worker.
pytestmark = pytest.mark.xdist_group(name="dspy_integration")


# =============================================================================
# Fakes that expose the REAL backend signatures (no .store on episodic)
# =============================================================================


class FakeEpisodicBackend:
    """Mirror of EpisodicMemoryBackend's WRITE contract.

    Exposes ``store_episode(content, episode_type, metadata)`` and NOTHING
    named ``store`` — so a call to ``.store`` raises AttributeError exactly
    like production did.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def store_episode(
        self,
        content: str,
        episode_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        self.calls.append(
            {"content": content, "episode_type": episode_type, "metadata": metadata or {}}
        )
        return "episode-id-1"


class FakeSemanticBackend:
    """Mirror of SemanticMemoryBackend's WRITE contract.

    Exposes ``store_relationship(...)`` and NO ``add_fact`` (which never
    existed on the real backend).
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def store_relationship(
        self,
        source_entity: str,
        target_entity: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self.calls.append(
            {
                "source_entity": source_entity,
                "target_entity": target_entity,
                "relationship_type": relationship_type,
                "properties": properties or {},
            }
        )
        return True


class FakeProceduralBackend:
    """Mirror of ProceduralMemoryBackend's WRITE contract.

    Exposes ``store_procedure(procedure_name, tool_sequence, trigger_pattern,
    intent, embedding)`` — positional fields, not a single dict.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def store_procedure(
        self,
        procedure_name: str,
        tool_sequence: List[Dict[str, Any]],
        trigger_pattern: Optional[str] = None,
        intent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> Optional[str]:
        self.calls.append(
            {
                "procedure_name": procedure_name,
                "tool_sequence": tool_sequence,
                "trigger_pattern": trigger_pattern,
                "intent": intent,
            }
        )
        return "procedure-id-1"


class _Evaluation:
    """Stand-in for the DSPy MemoryWorthiness prediction object."""

    def __init__(
        self,
        worth_remembering: bool,
        memory_type: str,
        importance_score: float,
        key_facts: Any,
    ) -> None:
        self.worth_remembering = worth_remembering
        self.memory_type = memory_type
        self.importance_score = importance_score
        self.key_facts = key_facts


# =============================================================================
# ROOT CAUSE 1 — store_episode write-back arg mapping
# =============================================================================


class TestReflectorStoreEpisode:
    """The episodic write-back must call store_episode (not .store) with
    correctly-mapped args, and must never raise into the response."""

    @pytest.mark.asyncio
    async def test_episodic_writeback_calls_store_episode_with_mapped_args(self):
        from src.rag.cognitive_rag_dspy import CognitiveState, ReflectorModule

        episodic = FakeEpisodicBackend()
        writers = {
            "episodic": episodic,
            "semantic": FakeSemanticBackend(),
            "procedural": FakeProceduralBackend(),
        }
        collector = MagicMock()

        async def _collect(signals):
            return None

        collector.collect = _collect

        module = ReflectorModule(writers, collector)

        # Replace the DSPy worthiness evaluator (the LM boundary) with a sync
        # callable returning a deterministic prediction. The logic under test
        # (the write-back routing) is NOT mocked.
        module.evaluate = MagicMock(  # type: ignore[method-assign]
            return_value=_Evaluation(
                worth_remembering=True,
                memory_type="episodic",
                importance_score=0.91,
                key_facts=[],
            )
        )

        state = CognitiveState(
            user_query="Why did Kisqali adoption increase in the Northeast?",
            conversation_id="conv-953",
        )
        state.response = "Adoption rose because of higher oncologist detailing in Q3."

        # RED before the fix: ReflectorModule.forward calls
        # ``episodic.store({...})`` -> AttributeError (no such attr on the fake
        # / real backend). GREEN after the fix: store_episode is awaited.
        result = await module.forward(state)

        assert len(episodic.calls) == 1, "store_episode should be awaited exactly once"
        call = episodic.calls[0]
        # content is the synthesized response (what we want to remember)
        assert call["content"] == state.response
        # event_type maps to the memory_event_type enum; "agent_action" is the
        # faithful, valid value ("conversation" is NOT in the enum -> #953b).
        assert call["episode_type"] == "agent_action"
        meta = call["metadata"]
        assert meta["query"] == state.user_query
        assert meta["importance_score"] == pytest.approx(0.91)
        # agent_name is deliberately omitted: cognitive_rag is not a valid
        # e2i_agent_name enum value (#953b). Must be absent, never the invalid
        # literal.
        assert meta.get("agent_name") is None
        # No in-band error / AttributeError leaked through
        assert result.worth_remembering is True

    @pytest.mark.asyncio
    async def test_episodic_writeback_is_best_effort_never_raises(self):
        """A failure in the episodic writer must not abort the Reflector."""
        from src.rag.cognitive_rag_dspy import CognitiveState, ReflectorModule

        class ExplodingEpisodic:
            async def store_episode(self, content, episode_type, metadata=None):
                raise RuntimeError("supabase down")

        writers = {
            "episodic": ExplodingEpisodic(),
            "semantic": FakeSemanticBackend(),
            "procedural": FakeProceduralBackend(),
        }
        collector = MagicMock()

        async def _collect(signals):
            return None

        collector.collect = _collect

        module = ReflectorModule(writers, collector)
        module.evaluate = MagicMock(  # type: ignore[method-assign]
            return_value=_Evaluation(
                worth_remembering=True,
                memory_type="episodic",
                importance_score=0.5,
                key_facts=[],
            )
        )

        state = CognitiveState(user_query="q", conversation_id="c")
        state.response = "r"

        # Must complete without propagating the backend error.
        result = await module.forward(state)
        assert result is state

    @pytest.mark.asyncio
    async def test_semantic_facts_use_store_relationship_not_add_fact(self):
        """key_facts must route through store_relationship (add_fact never
        existed on the semantic backend)."""
        from src.rag.cognitive_rag_dspy import CognitiveState, ReflectorModule

        semantic = FakeSemanticBackend()
        writers = {
            "episodic": FakeEpisodicBackend(),
            "semantic": semantic,
            "procedural": FakeProceduralBackend(),
        }
        collector = MagicMock()

        async def _collect(signals):
            return None

        collector.collect = _collect

        module = ReflectorModule(writers, collector)
        module.evaluate = MagicMock(  # type: ignore[method-assign]
            return_value=_Evaluation(
                worth_remembering=True,
                memory_type="episodic",
                importance_score=0.7,
                key_facts=[
                    {
                        "source": "brand:kisqali",
                        "target": "region:northeast",
                        "relationship": "ADOPTED_IN",
                    }
                ],
            )
        )

        state = CognitiveState(user_query="q", conversation_id="c")
        state.response = "r"

        await module.forward(state)

        assert len(semantic.calls) == 1, "store_relationship should be awaited for the fact"
        # add_fact does not exist; the test fake would AttributeError if called.


# =============================================================================
# ROOT CAUSE 2 — sync DSPy calls must not block the event loop
# =============================================================================


async def _heartbeat(ticks: List[float], stop: asyncio.Event) -> None:
    """Tick rapidly; each tick proves the event loop got control."""
    while not stop.is_set():
        ticks.append(time.monotonic())
        await asyncio.sleep(0.01)


class TestSyncDspyDoesNotBlockEventLoop:
    """If a sync DSPy call runs on the event loop, a concurrent heartbeat
    coroutine cannot tick during it (RED). Running it via asyncio.to_thread
    keeps the loop responsive so the heartbeat ticks many times (GREEN)."""

    @pytest.mark.asyncio
    async def test_summarizer_node_does_not_block_loop(self):
        """The REAL summarizer_node closure (Phase 1) must run the sync
        SummarizerModule.forward off the event loop.

        We drive the actual node extracted from the compiled workflow
        (wf.nodes['summarizer'].bound) and patch the exact sync call it makes
        (SummarizerModule.forward) to sleep. RED: the node awaits a sync call
        on the loop -> heartbeat starves (0-1 ticks). GREEN: the node offloads
        via asyncio.to_thread -> heartbeat ticks many times.
        """
        from src.rag import cognitive_rag_dspy as mod
        from src.rag.cognitive_rag_dspy import (
            CognitiveState,
            create_dspy_cognitive_workflow,
        )

        SLEEP = 0.5

        # Patch the exact sync callable the node invokes. forward() is what the
        # production node calls; making it block lets us prove the node either
        # blocks the loop (RED) or offloads it (GREEN).
        def _slow_forward(self, original_query, conversation_context, domain_vocabulary):
            time.sleep(SLEEP)
            return {
                "rewritten_query": "rq",
                "search_keywords": [],
                "graph_entities": [],
                "extracted_entities": "{}",
                "primary_intent": "GENERAL",
                "secondary_intents": [],
                "requires_visualization": False,
                "complexity": "SIMPLE",
            }

        backends = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "procedural": MagicMock(),
        }
        wf = create_dspy_cognitive_workflow(
            memory_backends=backends,
            memory_writers=backends,
            agent_registry={},
            signal_collector=MagicMock(),
            domain_vocabulary="v",
        )
        node = wf.nodes["summarizer"].bound

        ticks: List[float] = []
        stop = asyncio.Event()
        state = CognitiveState(user_query="q", conversation_id="c")

        original_forward = mod.SummarizerModule.forward
        mod.SummarizerModule.forward = _slow_forward  # type: ignore[assignment]
        try:
            hb = asyncio.create_task(_heartbeat(ticks, stop))
            await node.ainvoke(state)
            stop.set()
            await hb
        finally:
            mod.SummarizerModule.forward = original_forward  # type: ignore[assignment]

        # ~0.5s of sync sleep; a responsive loop ticks every ~10ms => dozens.
        # A blocked loop ticks 0-1 times.
        assert len(ticks) >= 5, f"event loop was blocked: only {len(ticks)} heartbeat ticks"

    @pytest.mark.asyncio
    async def test_investigator_forward_does_not_block_loop(self):
        """InvestigatorModule.forward (async) must run its sync DSPy
        self.plan / self.decide_hop off the loop."""
        from src.rag.cognitive_rag_dspy import InvestigatorModule

        backend = MagicMock()

        async def _empty_vector(*a, **k):
            return []

        backend.vector_search = _empty_vector
        investigator = InvestigatorModule({"episodic": backend})

        SLEEP = 0.3

        def _slow_plan(*a, **k):
            time.sleep(SLEEP)
            r = MagicMock()
            r.investigation_goal = "goal"
            r.hop_strategy = ["episodic"]
            r.max_hops = 1
            r.early_stop_criteria = ""
            return r

        def _slow_decide(*a, **k):
            time.sleep(SLEEP)
            r = MagicMock()
            r.next_memory = "STOP"
            r.retrieval_query = ""
            r.reasoning = ""
            r.confidence = 0.9
            return r

        investigator.plan = _slow_plan  # type: ignore[assignment]
        investigator.decide_hop = _slow_decide  # type: ignore[assignment]

        ticks: List[float] = []
        stop = asyncio.Event()

        hb = asyncio.create_task(_heartbeat(ticks, stop))
        result = await investigator.forward(rewritten_query="q", intent="GENERAL", entities="{}")
        stop.set()
        await hb

        assert "investigation_goal" in result
        assert len(ticks) >= 5, f"event loop was blocked: only {len(ticks)} heartbeat ticks"
