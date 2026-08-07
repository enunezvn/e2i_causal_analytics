"""#1489 deferral 1: the live producer for ``learning_signals.retrieved_chunks``.

``database/ml/022_self_improvement_tables.sql:143-150`` added
``retrieved_chunks`` and ``retrieval_scores`` "for RAGAS evaluation" and
nothing ever wrote them: measured on the live DB 2026-08-06, 3,959 rows, 0
with a non-default value in either column. The #1489 close-out recorded the
same fact and deferred the producer here.

This pins the free half of that producer — no LLM call is added anywhere. The
cognitive workflow's Phase-4 Reflector already holds ``state.evidence_board``,
the evidence the turn ACTUALLY retrieved, when it builds its training signals
(``_collect_training_signals``). The chain under test is:

    _collect_training_signals  ->  SignalCollector.collect
                               ->  LearningSignalInput
                               ->  record_learning_signal  ->  the columns

Only the ``agent`` (synthesis) signal carries the chunks. The three signals
describe one turn, so replicating the evidence onto all three would triple the
stored JSONB and make any per-row count of retrieved chunks read 3x the truth;
the ``agent`` row is the one whose ``response`` those chunks grounded, and so
the one a RAGAS judge scores.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


def _evidence(content: str, score: float, hop: int = 1):
    from src.rag.cognitive_rag_dspy import Evidence, MemoryType

    return Evidence(
        source=MemoryType.SEMANTIC,
        hop_number=hop,
        content=content,
        relevance_score=score,
    )


def _reflector():
    """Reflector with inert collaborators — ``_collect_training_signals`` is a
    pure function of the state and touches neither."""
    from src.rag.cognitive_rag_dspy import ReflectorModule

    return ReflectorModule(memory_writers={}, signal_collector=None)


def _state_with_evidence(evidence: List[Any]):
    from src.rag.cognitive_rag_dspy import CognitiveState

    state = CognitiveState(
        user_query="Why did Kisqali TRx fall in the Northeast?", conversation_id="conv-1489"
    )
    state.evidence_board = list(evidence)
    state.response = "Kisqali TRx fell 8% on payer mix. " * 20
    state.sufficient_evidence = True
    state.hop_count = 2
    state.detected_intent = "kpi_investigation"
    state.routed_agents = ["causal_impact"]
    return state


# ---------------------------------------------------------------------------
# 1. The Reflector attaches the turn's real evidence to the agent signal
# ---------------------------------------------------------------------------


def test_agent_signal_carries_retrieved_chunks_and_scores():
    """RED: ``_collect_training_signals`` emitted no chunk payload at all, so
    the column had no producer even though the evidence was in hand."""

    state = _state_with_evidence(
        [_evidence("payer mix shifted to Tier 3", 0.91), _evidence("NRx held flat", 0.42, hop=2)]
    )
    signals = _reflector()._collect_training_signals(state, user_feedback=None)

    agent = [s for s in signals if s["type"] == "agent"]
    assert len(agent) == 1, "the synthesis signal is the one a RAGAS judge scores"
    chunks = agent[0]["retrieved_chunks"]
    scores = agent[0]["retrieval_scores"]

    assert [c["content"] for c in chunks] == [
        "payer mix shifted to Tier 3",
        "NRx held flat",
    ]
    assert scores == [0.91, 0.42]
    assert [c["hop"] for c in chunks] == [1, 2]
    assert all(c["source"] == "semantic" for c in chunks)


def test_retrieval_scores_are_index_aligned_with_chunks():
    """Two columns, one retrieval: position i in ``retrieval_scores`` must
    describe position i in ``retrieved_chunks``, or neither can be read."""

    state = _state_with_evidence([_evidence(f"chunk-{i}", i / 10.0) for i in range(5)])
    signals = _reflector()._collect_training_signals(state, user_feedback=None)
    agent = next(s for s in signals if s["type"] == "agent")

    assert len(agent["retrieved_chunks"]) == len(agent["retrieval_scores"]) == 5
    for i, (chunk, score) in enumerate(
        zip(agent["retrieved_chunks"], agent["retrieval_scores"], strict=True)
    ):
        assert chunk["content"] == f"chunk-{i}"
        assert score == pytest.approx(i / 10.0)


def test_only_the_agent_signal_carries_chunks():
    """The summarizer/investigator signals describe phases, not the answer;
    copying the evidence onto them would triple-count one turn's retrieval."""

    state = _state_with_evidence([_evidence("only chunk", 0.5)])
    signals = _reflector()._collect_training_signals(state, user_feedback=None)

    for signal in signals:
        if signal["type"] == "agent":
            continue
        assert "retrieved_chunks" not in signal
        assert "retrieval_scores" not in signal


def test_zero_retrieval_turn_records_empty_lists_not_absence():
    """A turn that retrieved nothing is the pipeline's most common outcome
    (hit rate 3/10 on the #1489 close-out run). Recording [] says "measured,
    and it was empty"; omitting the key would be indistinguishable from the
    3,959 rows written before this producer existed."""

    state = _state_with_evidence([])
    signals = _reflector()._collect_training_signals(state, user_feedback=None)
    agent = next(s for s in signals if s["type"] == "agent")

    assert agent["retrieved_chunks"] == []
    assert agent["retrieval_scores"] == []


def test_oversized_chunk_content_is_capped_and_marked():
    """One pathological chunk must not write an unbounded JSONB blob on a live
    turn. The cap is per-chunk and marked, so a reader can tell a truncated
    chunk from a short one — an unmarked cut would silently understate what
    the answer was grounded in."""
    from src.rag.retrieved_chunks import MAX_CHUNK_CONTENT_CHARS

    state = _state_with_evidence([_evidence("x" * (MAX_CHUNK_CONTENT_CHARS + 500), 0.7)])
    signals = _reflector()._collect_training_signals(state, user_feedback=None)
    chunk = next(s for s in signals if s["type"] == "agent")["retrieved_chunks"][0]

    assert len(chunk["content"]) == MAX_CHUNK_CONTENT_CHARS
    assert chunk["truncated"] is True


def test_short_chunk_is_not_marked_truncated():
    state = _state_with_evidence([_evidence("short", 0.7)])
    signals = _reflector()._collect_training_signals(state, user_feedback=None)
    chunk = next(s for s in signals if s["type"] == "agent")["retrieved_chunks"][0]

    assert chunk["content"] == "short"
    assert chunk.get("truncated", False) is False


# ---------------------------------------------------------------------------
# 2. SignalCollector passes them through to LearningSignalInput
# ---------------------------------------------------------------------------


@pytest.fixture
def real_learning_signal_input(monkeypatch):
    """Undo a co-located module's import-time sys.modules stubbing.

    ``test_cognitive_backends.py`` replaces ``src.memory.procedural_memory``
    with a MagicMock while it imports ``cognitive_backends``, then restores
    sys.modules — but ``cognitive_backends.LearningSignalInput`` stays bound to
    the mock for the rest of the process. Its comment says nothing leaks "under
    pytest-xdist loadscope"; this repo's lanes run ``-n 0``, where it does. Without
    this fixture these two tests pass alone and fail in a full run, which is a
    guard that guards nothing.
    """
    from src.memory.procedural_memory import LearningSignalInput
    from src.rag import cognitive_backends

    monkeypatch.setattr(cognitive_backends, "LearningSignalInput", LearningSignalInput)
    return LearningSignalInput


@pytest.mark.asyncio
async def test_signal_collector_passes_chunks_into_learning_signal_input(
    monkeypatch, real_learning_signal_input
):
    """RED: ``SignalCollector.collect`` built ``LearningSignalInput`` from four
    keys only, so a chunk payload on the signal dict was dropped on the floor
    between the Reflector and the writer."""
    from src.rag import cognitive_backends

    captured: Dict[str, Any] = {}

    async def _fake_record(signal, cycle_id=None, session_id=None):
        captured["signal"] = signal
        return "sig-1"

    monkeypatch.setattr(cognitive_backends, "record_learning_signal", _fake_record)

    await cognitive_backends.SignalCollector().collect(
        [
            {
                "type": "agent",
                "signature_name": "agent",
                "input": "q",
                "output": "a",
                "metric": 0.8,
                "retrieved_chunks": [{"content": "c", "source": "semantic", "hop": 1}],
                "retrieval_scores": [0.55],
            }
        ]
    )

    signal = captured["signal"]
    assert signal.retrieved_chunks == [{"content": "c", "source": "semantic", "hop": 1}]
    assert signal.retrieval_scores == [0.55]


@pytest.mark.asyncio
async def test_signal_collector_omits_chunks_when_the_signal_has_none(
    monkeypatch, real_learning_signal_input
):
    """The summarizer/investigator signals carry no chunks; the writer strips
    None, leaving the column at its '[]' default rather than writing a null."""
    from src.rag import cognitive_backends

    captured: Dict[str, Any] = {}

    async def _fake_record(signal, cycle_id=None, session_id=None):
        captured["signal"] = signal
        return "sig-1"

    monkeypatch.setattr(cognitive_backends, "record_learning_signal", _fake_record)

    await cognitive_backends.SignalCollector().collect(
        [
            {
                "type": "summarizer",
                "signature_name": "summarizer",
                "input": "q",
                "output": "a",
                "metric": 0.4,
            }
        ]
    )

    assert captured["signal"].retrieved_chunks is None
    assert captured["signal"].retrieval_scores is None


# ---------------------------------------------------------------------------
# 3. record_learning_signal maps them onto the columns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_learning_signal_writes_the_two_columns(monkeypatch):
    """RED: ``LearningSignalInput`` had no such fields and the record builder
    never emitted the column names, so no value could reach the table."""
    from src.memory import procedural_memory

    captured: Dict[str, Any] = {}

    class _Table:
        def insert(self, record):
            captured["record"] = record
            return self

        def execute(self):
            return type("R", (), {"data": [{"signal_id": "x"}]})()

    class _Client:
        def table(self, name):
            captured["table"] = name
            return _Table()

    monkeypatch.setattr(procedural_memory, "get_supabase_client", lambda: _Client())

    await procedural_memory.record_learning_signal(
        signal=procedural_memory.LearningSignalInput(
            signal_type="rating",
            signal_value=0.8,
            retrieved_chunks=[{"content": "c", "source": "semantic", "hop": 1}],
            retrieval_scores=[0.55],
        )
    )

    record = captured["record"]
    assert captured["table"] == "learning_signals"
    # Raw structures, never json.dumps: a pre-dumped string is double-encoded
    # by postgrest into a JSON string scalar (#883, migration 073).
    assert record["retrieved_chunks"] == [{"content": "c", "source": "semantic", "hop": 1}]
    assert record["retrieval_scores"] == [0.55]


@pytest.mark.asyncio
async def test_record_learning_signal_omits_absent_chunk_columns(monkeypatch):
    """A signal with no retrieval must leave the columns at their schema
    defaults ('[]'), not overwrite them with NULL — ``retrieval_scores`` is
    read as an array by any consumer and a NULL would break it."""
    from src.memory import procedural_memory

    captured: Dict[str, Any] = {}

    class _Table:
        def insert(self, record):
            captured["record"] = record
            return self

        def execute(self):
            return type("R", (), {"data": [{"signal_id": "x"}]})()

    class _Client:
        def table(self, name):
            return _Table()

    monkeypatch.setattr(procedural_memory, "get_supabase_client", lambda: _Client())

    await procedural_memory.record_learning_signal(
        signal=procedural_memory.LearningSignalInput(signal_type="rating", signal_value=0.8)
    )

    assert "retrieved_chunks" not in captured["record"]
    assert "retrieval_scores" not in captured["record"]
