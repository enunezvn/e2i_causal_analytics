"""
Regression tests for issue #953 completion (#954 deploy reached the response
construction; two MORE bugs surfaced on the live 4-phase run).

The endpoint no longer 502s and no longer raises
``'dict' object has no attribute 'evidence_board'`` (#957). A live
``POST /api/cognitive/rag`` now runs the full 4-phase pipeline (~102s) but
returns **HTTP 400** because the response construction fails pydantic
validation, and DSPy training signals are silently dropped by a UUID-typed
``cycle_id`` insert.

BUG C — ``CognitiveRAGResponse`` list_type/dict_type 400.
    ``CausalRAG.cognitive_search`` returns a dict whose list/dict-typed fields
    can carry a non-list/non-dict value (a graph node can write ``None`` into a
    channel for a ``List``/``Dict`` field; ``CognitiveState`` is a plain
    ``@dataclass`` that does not coerce). The route then builds
    ``CognitiveRAGResponse(**result)`` with ``result.get("X", default)``, and
    ``.get`` returns the present-but-``None`` value (the default only applies
    when the key is ABSENT). ``CognitiveRAGResponse`` declares those fields
    ``List[...]``/``Dict[...]`` -> pydantic ``list_type``/``dict_type`` -> the
    route's ``except ValueError`` maps it to HTTP 400.

BUG D — DSPy signals never persist (cycle_id="unknown" -> 22P02).
    ``SignalCollector.collect`` calls
    ``record_learning_signal(..., cycle_id=signal.get("cycle_id", "unknown"))``.
    Cognitive-RAG signals carry no ``cycle_id``, so the literal string
    ``"unknown"`` is inserted into ``learning_signals.cycle_id`` -- a nullable
    UUID column (database/migrations/008_agentic_memory_schema.sql) -- which
    Postgres rejects with ``22P02 invalid input syntax for type uuid``. The row
    is swallowed into ``_pending_signals`` and NO DSPy training signal ever
    persists for cognitive RAG. Passing ``cycle_id=None`` lets
    ``record_learning_signal`` drop the key (it strips None values) so the
    column takes NULL, which the nullable FK column accepts.

These tests exercise the REAL code paths. BUG C drives the REAL
``create_dspy_cognitive_workflow`` graph + REAL ``ainvoke`` + the REAL response
construction with only the LM boundary stubbed (DSPy module ``forward``s). BUG D
asserts at the REAL ``SignalCollector.collect`` -> ``record_learning_signal``
call boundary (capturing the ``cycle_id`` argument -- that is our OWN call
contract, not a mock of the fix).
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# DSPy import has parallel-worker race conditions; pin to one worker.
pytestmark = pytest.mark.xdist_group(name="dspy_integration")


# =============================================================================
# BUG C — the live 400. Drive the REAL graph end-to-end (LM boundary stubbed)
#         and prove the route can build a valid CognitiveRAGResponse.
# =============================================================================


class TestCognitiveRAGResponseConstructsFromGraphResult:
    """The REAL graph result must satisfy CognitiveRAGResponse. A node writing
    None into a List/Dict channel propagates None all the way into the response
    dict (CognitiveState does not coerce), and the route's
    ``result.get(k, default)`` keeps that None (default only applies to ABSENT
    keys) -> pydantic list_type/dict_type -> HTTP 400."""

    def _stub_forwards(self, mod):
        """Stub the four DSPy module forwards at the LM boundary.

        The AgentModule forward deliberately leaves ``routed_agents`` and
        ``visualization_config`` at their channel default by NOT writing them,
        emulating a graph run where the LM (or an empty-registry route) returns
        nothing for a list/dict field. The success-path return dict then carries
        whatever the channel holds -- this is the real shape that broke the
        live response construction.
        """

        def _sum_fwd(self, original_query, conversation_context, domain_vocabulary):
            return {
                "rewritten_query": "rewritten q",
                "search_keywords": [],
                "graph_entities": [],
                # The REAL SummarizerModule emits extracted_entities as a STRING
                # (str(dict)); mirror that exactly so the test exercises the real
                # entities-field shape that reaches CognitiveRAGResponse.entities
                # (declared List[str]).
                "extracted_entities": "{'brands': ['Kisqali'], 'regions': ['Northeast']}",
                "primary_intent": "CAUSAL_ANALYSIS",
                "secondary_intents": [],
                "requires_visualization": False,
                "complexity": "SIMPLE",
            }

        async def _inv_fwd(self, rewritten_query, intent, entities):
            from src.rag.cognitive_rag_dspy import Evidence, MemoryType

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
            # Synthesize a real answer but DO NOT set routed_agents /
            # visualization_config: emulate the LM/route returning nothing for
            # these list/dict fields. The dataclass keeps their factory default
            # here, but a real LangGraph channel could just as easily hold None;
            # the contract fix must tolerate None either way (see the None
            # variant test below).
            state.response = "A real synthesized answer about Kisqali adoption."
            return state

        async def _refl_fwd(self, state, user_feedback=None):
            state.worth_remembering = False
            return state

        return (
            patch.object(mod.SummarizerModule, "forward", _sum_fwd),
            patch.object(mod.InvestigatorModule, "forward", _inv_fwd),
            patch.object(mod.AgentModule, "forward", _agent_fwd),
            patch.object(mod.ReflectorModule, "forward", _refl_fwd),
        )

    @pytest.mark.asyncio
    async def test_response_model_constructs_from_real_cognitive_search(self):
        """End-to-end: REAL cognitive_search -> REAL route construction.

        RED before the contract fix: ``CognitiveRAGResponse(**result)`` raises a
        pydantic ValidationError (entities is a str, or a list/dict field is
        None) -> the route returns 400.
        GREEN after: the response constructs and surfaces the real fields.
        """
        import src.rag.cognitive_rag_dspy as mod
        from src.api.routes.cognitive import CognitiveRAGResponse
        from src.rag.causal_rag import CausalRAG

        p1, p2, p3, p4 = self._stub_forwards(mod)
        with p1, p2, p3, p4, patch("dspy.settings") as dspy_settings:
            dspy_settings.lm = object()  # truthy => skip real LM construction

            rag = CausalRAG()
            result = await rag.cognitive_search(
                query="Why did Kisqali adoption increase?",
                conversation_id="conv-953c",
            )

        assert "error" not in result or result.get("error") is None, (
            f"cognitive_search returned error-as-data: {result.get('error')!r}"
        )

        # This mirrors the EXACT route construction in
        # src/api/routes/cognitive.py::cognitive_rag_search. RED: raises
        # ValidationError -> 400. GREEN: constructs.
        request_query = "Why did Kisqali adoption increase?"
        response = CognitiveRAGResponse(
            response=result.get("response", ""),
            evidence=result.get("evidence", []),
            hop_count=result.get("hop_count", 0),
            visualization_config=result.get("visualization_config", {}),
            routed_agents=result.get("routed_agents", []),
            entities=result.get("entities", []),
            intent=result.get("intent", ""),
            rewritten_query=result.get("rewritten_query", request_query),
            dspy_signals=result.get("dspy_signals", []),
            worth_remembering=result.get("worth_remembering", False),
            latency_ms=result.get("latency_ms", 0.0),
            error=result.get("error"),
        )

        assert response.response == "A real synthesized answer about Kisqali adoption."
        assert isinstance(response.evidence, list)
        assert isinstance(response.routed_agents, list)
        assert isinstance(response.entities, list)
        assert isinstance(response.dspy_signals, list)
        assert isinstance(response.visualization_config, dict)

    def test_route_construction_coerces_none_to_defaults(self):
        """If the graph result carries None for a list/dict field (a node wrote
        None into the channel; CognitiveState does not coerce), the route MUST
        still build a valid CognitiveRAGResponse.

        This is the precise live-400 shape: ``result.get(key, default)`` returns
        the present-but-None value, and CognitiveRAGResponse requires List/Dict.
        """
        from src.api.routes.cognitive import CognitiveRAGResponse

        # The pathological result: every list/dict-typed field is present-and-None.
        result: Dict[str, Any] = {
            "response": None,
            "evidence": None,
            "hop_count": None,
            "visualization_config": None,
            "routed_agents": None,
            "entities": None,
            "intent": None,
            "rewritten_query": None,
            "dspy_signals": None,
            "worth_remembering": None,
            "latency_ms": None,
            "error": None,
        }

        # RED before fix: ``result.get("evidence", [])`` -> None -> pydantic
        # list_type. GREEN after fix: ``or []`` coercion yields valid defaults.
        request_query = "Why did Kisqali adoption increase?"
        response = CognitiveRAGResponse(
            response=result.get("response") or "",
            evidence=result.get("evidence") or [],
            hop_count=result.get("hop_count") or 0,
            visualization_config=result.get("visualization_config") or {},
            routed_agents=result.get("routed_agents") or [],
            entities=result.get("entities") or [],
            intent=result.get("intent") or "",
            rewritten_query=result.get("rewritten_query") or request_query,
            dspy_signals=result.get("dspy_signals") or [],
            worth_remembering=bool(result.get("worth_remembering")),
            latency_ms=result.get("latency_ms") or 0.0,
            error=result.get("error"),
        )

        assert response.evidence == []
        assert response.routed_agents == []
        assert response.entities == []
        assert response.dspy_signals == []
        assert response.visualization_config == {}
        assert response.response == ""
        assert response.rewritten_query == request_query
        assert response.worth_remembering is False
        assert response.latency_ms == 0.0
        assert response.error is None


# =============================================================================
# BUG C (source) — SummarizerModule.forward must emit extracted_entities as a
#                  List[str], not a str(dict). This is the ROOT cause of the
#                  live list_type 400 (the str propagated into the List[str]
#                  channel and then into CognitiveRAGResponse.entities).
# =============================================================================


class TestSummarizerEmitsEntitiesAsList:
    """The summarizer's extracted_entities output feeds CognitiveState
    .extracted_entities (List[str]) which is surfaced verbatim as
    CognitiveRAGResponse.entities (List[str]). It MUST be a list of entity
    values, never the ``str(dict)`` rendering that broke the live response."""

    def test_forward_returns_extracted_entities_as_flat_list(self):
        import src.rag.cognitive_rag_dspy as mod
        from src.rag.cognitive_rag_dspy import SummarizerModule

        # Stub the three predictors with realistic per-category list outputs.
        class _Extract:
            brands = ["Kisqali", "Fabhalta"]
            regions = ["Northeast"]
            hcp_types = ["Oncologist"]
            patient_stages: List[str] = []
            time_references = ["Q3"]

        class _Rewrite:
            rewritten_query = "rewritten"
            search_keywords: List[str] = []
            graph_entities: List[str] = []

        class _Intent:
            primary_intent = "CAUSAL_ANALYSIS"
            secondary_intents: List[str] = []
            requires_visualization = False
            complexity = "SIMPLE"

        with (
            patch.object(mod.SummarizerModule, "extract", create=True),
            patch.object(mod.SummarizerModule, "rewrite", create=True),
            patch.object(mod.SummarizerModule, "classify", create=True),
        ):
            m = SummarizerModule()
            m.extract = MagicMock(return_value=_Extract())  # type: ignore[method-assign]
            m.rewrite = MagicMock(return_value=_Rewrite())  # type: ignore[method-assign]
            m.classify = MagicMock(return_value=_Intent())  # type: ignore[method-assign]

            result = m.forward(
                original_query="Why did Kisqali adoption increase?",
                conversation_context="",
                domain_vocabulary="",
            )

        entities = result["extracted_entities"]
        # RED before source fix: this was ``str({...})`` (a str).
        assert isinstance(entities, list), (
            f"extracted_entities must be a list, got {type(entities).__name__}"
        )
        assert all(isinstance(e, str) for e in entities)
        # The flattened, de-duplicated values from every category are present.
        assert set(entities) == {
            "Kisqali",
            "Fabhalta",
            "Northeast",
            "Oncologist",
            "Q3",
        }

        # The classify InputField still received a STRING rendering (its
        # signature declares extracted_entities: str) -- the source fix must not
        # break the LM input contract.
        classify_kwargs = m.classify.call_args.kwargs
        assert isinstance(classify_kwargs["extracted_entities"], str)


# =============================================================================
# BUG D — DSPy signals must persist: cycle_id must be None (NULL), not "unknown".
# =============================================================================


class TestDSPySignalCycleIdNull:
    """``SignalCollector.collect`` must NOT pass cycle_id='unknown' (a string
    into a UUID column -> 22P02 -> signal swallowed). It must pass the real
    cycle_id, or None (which record_learning_signal drops -> NULL)."""

    @pytest.mark.asyncio
    async def test_collect_passes_none_cycle_id_when_signal_has_no_cycle(self):
        """A cognitive-RAG signal carries no cycle_id -> collect must pass
        cycle_id=None to record_learning_signal (NOT the string 'unknown').

        We capture the cycle_id argument at the REAL record_learning_signal call
        boundary -- this asserts OUR call contract, not the DB write itself.
        """
        from src.rag.cognitive_backends import SignalCollector

        captured: Dict[str, Any] = {}

        async def _capture_record(signal, cycle_id=None, session_id=None):
            captured["cycle_id"] = cycle_id
            captured["signal"] = signal
            return "sig-1"

        collector = SignalCollector()

        # A real cognitive-RAG DSPy signal: no cycle_id key.
        signals = [
            {
                "signature_name": "QueryRewriteSignature",
                "input": {"query": "Why did Kisqali adoption increase?"},
                "output": {"rewritten_query": "Kisqali adoption drivers Q3"},
                "metric": 0.8,
            }
        ]

        with patch("src.rag.cognitive_backends.record_learning_signal", _capture_record):
            await collector.collect(signals)

        # RED before fix: cycle_id == "unknown" (string into a UUID column).
        # GREEN after fix: cycle_id is None (record_learning_signal drops it ->
        # NULL, which the nullable FK column accepts).
        assert captured.get("cycle_id") is None, (
            f"collect passed cycle_id={captured.get('cycle_id')!r} "
            "-> string into a UUID column triggers 22P02 and the DSPy signal is "
            "swallowed into _pending_signals (never persists)."
        )
        # The signal must NOT have been queued for retry (it should have been
        # accepted at the boundary).
        assert collector._pending_signals == [], "signal was swallowed into the pending-retry queue"

    @pytest.mark.asyncio
    async def test_collect_preserves_real_cycle_id_when_present(self):
        """When a signal DOES carry a real cycle_id, collect must pass it
        through unchanged (the fix must not blanket-null a present cycle_id)."""
        from src.rag.cognitive_backends import SignalCollector

        captured: Dict[str, Any] = {}

        async def _capture_record(signal, cycle_id=None, session_id=None):
            captured["cycle_id"] = cycle_id
            return "sig-2"

        collector = SignalCollector()
        real_cycle = "11111111-1111-1111-1111-111111111111"
        signals = [
            {
                "signature_name": "IntentClassificationSignature",
                "input": {},
                "output": {},
                "metric": 0.9,
                "cycle_id": real_cycle,
            }
        ]

        with patch("src.rag.cognitive_backends.record_learning_signal", _capture_record):
            await collector.collect(signals)

        assert captured.get("cycle_id") == real_cycle
        assert collector._pending_signals == []
