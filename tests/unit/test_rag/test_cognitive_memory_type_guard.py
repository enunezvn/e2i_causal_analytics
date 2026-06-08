"""C2 guard: InvestigatorModule.forward must not crash (and must not silently
truncate) when the DSPy hop-decider returns a non-enum next_memory string.

Faithful: the unit under test (InvestigatorModule.forward) runs for real.
Only the DSPy predictors (plan / decide_hop / score_evidence) -- the true
external LLM -- are doubled, mirroring
tests/rag/test_cognitive_rag_dspy.py::TestAgentModule::test_agent_forward_updates_state.

Placed under tests/unit/test_rag/ (collected by the CI Heavy Unit lane) rather
than tests/rag/ (uncollected) so the regression guard actually runs in CI. The
test is faithful and offline (doubles only the DSPy predictors), so it is safe
in the unit lane.

IMPORTANT (pre-fix reality, verified empirically): against the CURRENT unfixed
code, the off-vocabulary string ('Episodic'/'causal') is passed RAW to
self.memory_backends.get(...) in _retrieve_from_memory, which returns [] before
the MemoryType() cast ever runs. So pre-fix the module returns a clean dict and
NEVER raises. The ONE red test below is
test_forward_coerces_offvocab_to_valid_enum_source (it expects the item to
SURVIVE coercion -> board len 1, but pre-fix the board is empty). The other two
tests already pass pre-fix and serve as POST-FIX REGRESSION GUARDS.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.cognitive_rag_dspy import InvestigatorModule, MemoryType


def _make_investigator_with_decider(next_memory_value: str) -> InvestigatorModule:
    """Real InvestigatorModule whose decider returns the given next_memory once,
    then STOP; episodic backend yields one retrievable, high-scoring item.

    The backend is registered under the canonical lowercase key 'episodic' so
    that AFTER the hardening (case-insensitive coercion) an off-vocab decider
    value like 'Episodic' routes to it and drives content into the cast.
    """
    episodic_backend = MagicMock()
    episodic_backend.vector_search = AsyncMock(
        return_value=[{"content": "TRx up 15% in Northeast"}]
    )
    module = InvestigatorModule({"episodic": episodic_backend})

    # plan() -> fixed investigation goal
    plan_result = MagicMock()
    plan_result.investigation_goal = "find TRx drivers"
    module.plan = MagicMock(return_value=plan_result)

    # decide_hop() -> off-vocabulary value on hop 1, STOP on hop 2
    decision_bad = MagicMock()
    decision_bad.next_memory = next_memory_value  # e.g. "Episodic"
    decision_bad.retrieval_query = "TRx drivers"
    decision_bad.confidence = 0.9  # >= 0.3 so the loop does not early-stop
    decision_stop = MagicMock()
    decision_stop.next_memory = "STOP"
    decision_stop.retrieval_query = ""
    decision_stop.confidence = 0.9
    module.decide_hop = MagicMock(side_effect=[decision_bad, decision_stop])

    # score_evidence() -> relevance 0.9 so the item passes the >= 0.5 gate
    scored = MagicMock()
    scored.relevance_score = 0.9
    scored.key_insight = "TRx rose"
    module.score_evidence = MagicMock(return_value=scored)
    return module


@pytest.mark.asyncio
async def test_forward_does_not_raise_on_offvocab_next_memory():
    """Regression guard (passes pre AND post fix): off-vocabulary next_memory
    must NOT raise out of forward -- forward always returns a dict with the
    'evidence_board' key. Pre-fix this holds because retrieval returns [] on the
    raw key; post-fix it holds because the coercion skips-and-logs uncoercible
    hops. Either way: no exception escapes."""
    module = _make_investigator_with_decider("Episodic")  # capitalized -> not an enum value
    result = await module.forward(
        rewritten_query="TRx trend", intent="kpi", entities="Kisqali,Northeast"
    )
    assert isinstance(result, dict)
    assert "evidence_board" in result


@pytest.mark.asyncio
async def test_forward_coerces_offvocab_to_valid_enum_source():
    """THE RED TEST (fails pre-fix, passes post-fix). A coercible off-vocabulary
    value ('Episodic') must route to the episodic backend and yield an Evidence
    whose source is a real MemoryType -- not be silently dropped. Pre-fix the
    board is EMPTY (raw 'Episodic' is not a backend key, so retrieval returns []
    before the cast), so len(board) == 0 != 1 -> RED."""
    module = _make_investigator_with_decider("Episodic")
    result = await module.forward(
        rewritten_query="TRx trend", intent="kpi", entities="Kisqali,Northeast"
    )
    board = result["evidence_board"]
    assert len(board) == 1, "the retrievable, high-scoring item must survive coercion"
    assert isinstance(board[0].source, MemoryType)
    assert board[0].source == MemoryType.EPISODIC


@pytest.mark.asyncio
async def test_forward_skips_uncoercible_next_memory_without_crashing():
    """POST-FIX regression guard (NON-DISCRIMINATING: passes pre AND post fix).
    Kept to lock in that a truly off-vocabulary value ('causal', which no enum
    maps to) is skipped, not crashed: the investigation returns cleanly with an
    empty board."""
    module = _make_investigator_with_decider("causal")  # no enum maps to this
    result = await module.forward(
        rewritten_query="TRx trend", intent="kpi", entities="Kisqali,Northeast"
    )
    assert isinstance(result, dict)
    # uncoercible -> that hop's items are dropped, but forward returns cleanly
    assert result["evidence_board"] == []
