"""
Regression tests for the cognitive-RAG checkpointer thread_id bug.

Background
----------
`create_dspy_cognitive_workflow` compiles its LangGraph graph with a
`MemorySaver` checkpointer. LangGraph REQUIRES a ``thread_id`` in the run
config whenever a checkpointer is present; otherwise ``ainvoke`` raises::

    ValueError: Checkpointer requires one or more of the following
    'configurable' keys: thread_id, checkpoint_ns, checkpoint_id

The production call site (`CausalRAG.cognitive_search`) used to call
``await workflow.ainvoke(initial_state)`` with NO config. The broad
``except Exception`` then swallowed the ValueError and returned an HTTP-200
payload whose ``response`` field held the error STRING and ``error`` was set
-- which the Executive AI Brief panel rendered to users as a real insight
(the #932/#939 "error-as-data" anti-fabrication class).

These tests are FAITHFUL to the real failure mechanism:

1. ``test_real_workflow_requires_thread_id`` exercises the genuine compiled
   workflow (with its real MemorySaver checkpointer) and proves the contract:
   no thread_id => ValueError; with thread_id => the run proceeds.

2. ``test_cognitive_search_passes_thread_id`` exercises the production call
   site with a workflow stub that ENFORCES the same contract (rejects an
   ``ainvoke`` lacking a non-empty ``configurable.thread_id``), so a fix that
   silently drops the config cannot pass.
"""

import sys
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.rag.causal_rag import CausalRAG

# ============================================================================
# 1. FAITHFUL: the real compiled workflow enforces the thread_id contract
# ============================================================================


class _Node:
    """Trivial node that marks the state so we can prove the run executed."""

    async def __call__(self, state: Any) -> Any:
        state.response = "real-grounded-answer"
        return state


def _build_real_checkpointed_workflow():
    """
    Compile a LangGraph workflow over the *real* CognitiveState dataclass with
    the *real* MemorySaver checkpointer -- the same checkpointer used by
    ``create_dspy_cognitive_workflow`` -- but with a trivial node so the run
    does not need an LLM. The checkpointer guard fires before any node runs,
    so this is a faithful reproduction of the production failure mode.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

    from src.rag.cognitive_rag_dspy import CognitiveState

    graph = StateGraph(CognitiveState)
    graph.add_node("only", _Node())
    graph.set_entry_point("only")
    graph.add_edge("only", END)
    return graph.compile(checkpointer=MemorySaver())


@pytest.mark.asyncio
async def test_real_workflow_requires_thread_id():
    """The genuine checkpointed workflow raises without a thread_id config."""
    from src.rag.cognitive_rag_dspy import CognitiveState

    workflow = _build_real_checkpointed_workflow()

    # Without config -> the live error we reproduced against prod.
    with pytest.raises(ValueError) as exc_info:
        await workflow.ainvoke(CognitiveState(user_query="q", conversation_id="conv-1"))
    assert "thread_id" in str(exc_info.value)
    assert "Checkpointer requires" in str(exc_info.value)

    # With a thread_id config -> the run proceeds and produces real content.
    result = await workflow.ainvoke(
        CognitiveState(user_query="q", conversation_id="conv-1"),
        config={"configurable": {"thread_id": "conv-1"}},
    )
    response = result["response"] if isinstance(result, dict) else result.response
    assert response == "real-grounded-answer"


# ============================================================================
# 2. PRODUCTION CALL SITE: cognitive_search must thread the thread_id through
# ============================================================================


class _ThreadIdEnforcingWorkflow:
    """
    Stand-in for the compiled workflow that enforces the SAME contract as the
    real MemorySaver: ``ainvoke`` MUST receive a non-empty
    ``config['configurable']['thread_id']`` or it raises the production error.

    This deliberately refuses to be a permissive ``AsyncMock`` -- a permissive
    mock is exactly why the original bug shipped (the call site dropped the
    config and the test never noticed). Here, a fix that omits the config
    cannot pass.
    """

    def __init__(self, result_state: Any) -> None:
        self._result_state = result_state
        self.received_config: Optional[Dict[str, Any]] = None

    async def ainvoke(self, state: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        self.received_config = config
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        if not thread_id:
            raise ValueError(
                "Checkpointer requires one or more of the following "
                "'configurable' keys: thread_id, checkpoint_ns, checkpoint_id"
            )
        return self._result_state


def _make_result_state() -> Mock:
    state = Mock()
    state.response = "Prescribing gaps are concentrated in the West region."
    state.evidence_board = []
    state.hop_count = 2
    state.visualization_config = {}
    state.routed_agents = []
    state.extracted_entities = ["Remibrutinib"]
    state.detected_intent = "gap"
    state.rewritten_query = "top prescribing gaps for Remibrutinib"
    state.dspy_signals = []
    state.worth_remembering = True
    return state


async def _run_cognitive_search(conversation_id: Optional[str], workflow: Any):
    """
    Invoke ``CausalRAG.cognitive_search`` with dspy + backends patched to the
    real code path, substituting only the compiled workflow object so we can
    assert how it is invoked.
    """
    rag = CausalRAG()

    mock_dspy = MagicMock()
    mock_dspy.settings.lm = None
    mock_dspy.LM.return_value = Mock()
    mock_dspy.configure = Mock()

    mock_cog_state_class = Mock(side_effect=lambda **kwargs: type("S", (), kwargs)())

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch.dict(sys.modules, {"dspy": mock_dspy}):
            with patch(
                "src.rag.cognitive_rag_dspy.create_dspy_cognitive_workflow",
                return_value=workflow,
            ):
                with patch(
                    "src.rag.cognitive_rag_dspy.CognitiveState",
                    mock_cog_state_class,
                ):
                    with patch(
                        "src.rag.cognitive_backends.get_cognitive_memory_backends"
                    ) as mock_backends:
                        mock_backends.return_value = {
                            "readers": {},
                            "writers": {},
                            "signal_collector": Mock(),
                        }
                        return await rag.cognitive_search(
                            query="What are the top prescribing gaps?",
                            conversation_id=conversation_id,
                        )


@pytest.mark.asyncio
async def test_cognitive_search_passes_thread_id():
    """
    cognitive_search supplies a thread_id config and returns a REAL result
    (no error key, real response) instead of the swallowed error payload.
    """
    workflow = _ThreadIdEnforcingWorkflow(_make_result_state())

    result = await _run_cognitive_search("conv-abc", workflow)

    # The config was threaded through to the (checkpointer-bound) workflow.
    assert workflow.received_config is not None, "ainvoke received no config"
    thread_id = workflow.received_config["configurable"]["thread_id"]
    assert thread_id, "thread_id must be a non-empty value"
    # The thread_id is seeded from the conversation id so a conversation maps
    # to a single LangGraph thread.
    assert thread_id == "conv-abc"

    # And the call returns a real grounded answer, NOT the error-as-data shape.
    assert "error" not in result
    assert result["response"] == "Prescribing gaps are concentrated in the West region."
    assert not result["response"].startswith("Unable to complete cognitive search")


@pytest.mark.asyncio
async def test_cognitive_search_generates_thread_id_when_no_conversation_id():
    """When no conversation_id is supplied, a thread_id is still generated."""
    workflow = _ThreadIdEnforcingWorkflow(_make_result_state())

    result = await _run_cognitive_search(None, workflow)

    assert workflow.received_config is not None
    thread_id = workflow.received_config["configurable"]["thread_id"]
    assert thread_id, "a thread_id must be generated even without conversation_id"
    assert "error" not in result
