"""
CausalRAG - Main orchestrator for graph-enhanced retrieval.

This is the primary entry point for RAG operations in E2I Causal Analytics.
Supports both traditional hybrid retrieval and DSPy-enhanced cognitive workflows.
"""

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, cast

from src.rag.models.retrieval_models import RetrievalContext, RetrievalResult

logger = logging.getLogger(__name__)


def _coerce_cognitive_state(raw: Any) -> Any:
    """Coerce a compiled-LangGraph ainvoke result into a CognitiveState.

    ``StateGraph(CognitiveState).compile().ainvoke(...)`` returns a **dict** of
    channel values keyed by the CognitiveState field names, not the dataclass
    instance. ``cognitive_search`` consumes the result via attribute access, so
    a bare dict raised ``'dict' object has no attribute 'evidence_board'``
    (#953). This rebuilds the dataclass from the dict (keys == field names),
    ignoring any unknown channel keys so construction can never raise.

    The ONLY problematic shape is a plain ``dict``. Anything else (a real
    CognitiveState, or a test double that already supports attribute access) is
    returned untouched -- we duck-type rather than ``isinstance`` against
    CognitiveState because that symbol may be patched in tests, and the
    consumer only needs attribute access to succeed.
    """
    if isinstance(raw, dict):
        # Lazy import mirrors cognitive_search's own lazy import (circular-dep
        # safe) and picks up a patched CognitiveState in tests.
        from dataclasses import fields

        from src.rag.cognitive_rag_dspy import CognitiveState

        valid = {f.name for f in fields(CognitiveState)}
        return CognitiveState(**{k: v for k, v in raw.items() if k in valid})
    # Already attribute-accessible (real CognitiveState / test double / future
    # LangGraph object): leave as-is. We do not fabricate a state.
    return raw


class CausalRAG:
    """
    Graph-enhanced retrieval for causal insights.

    Retrieval sources:
    1. Vector store (semantic similarity)
    2. Causal graph (path traversal)
    3. Structured queries (SQL for KPIs)

    CRITICAL: Only indexes operational data.
    NEVER indexes: clinical trials, medical literature, regulatory docs.
    """

    def __init__(
        self,
        vector_retriever=None,
        graph_retriever=None,
        kpi_retriever=None,
        reranker=None,
    ):
        """
        Initialize CausalRAG with retrieval components.

        Args:
            vector_retriever: Dense/sparse vector retriever
            graph_retriever: Causal graph traversal
            kpi_retriever: Structured KPI queries
            reranker: Cross-encoder reranker
        """
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.kpi_retriever = kpi_retriever
        self.reranker = reranker

    def retrieve(
        self,
        query,  # ParsedQuery from NLP layer
        top_k: int = 10,
        retrieval_config: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Execute hybrid retrieval for a parsed query.

        Args:
            query: ParsedQuery from NLP layer
            top_k: Maximum results to return
            retrieval_config: Override default retrieval settings

        Returns:
            List of RetrievalResult ordered by relevance
        """
        all_results = []

        # 1. Semantic retrieval from vector store
        if self.vector_retriever:
            vector_results = self.vector_retriever.search(
                query.text if hasattr(query, "text") else str(query), k=top_k
            )
            all_results.extend(vector_results)

        # 2. Graph-based retrieval for causal queries
        if self.graph_retriever and hasattr(query, "intent"):
            # Only use graph retrieval for causal intent queries
            if query.intent and query.intent.value == "causal":
                graph_results = self.graph_retriever.traverse(
                    entities=query.entities if hasattr(query, "entities") else [],
                    relationship="causal_path",
                )
                all_results.extend(graph_results)

        # 3. Structured retrieval for KPI queries
        if self.kpi_retriever and hasattr(query, "entities"):
            if hasattr(query.entities, "kpis") and query.entities.kpis:
                kpi_results = self.kpi_retriever.query(query.entities.kpis)
                all_results.extend(kpi_results)

        # 4. Rerank and deduplicate
        if self.reranker and all_results:
            return cast(
                List[RetrievalResult], self.reranker.rerank(all_results, query, top_k=top_k)
            )

        return all_results[:top_k]

    async def retrieve_async(
        self,
        query,
        top_k: int = 10,
        retrieval_config: Optional[Dict[str, Any]] = None,
    ) -> RetrievalContext:
        """
        Async version of retrieve with full context.

        Returns:
            RetrievalContext with results and metadata
        """
        import time

        start_time = time.time()

        results = self.retrieve(query, top_k, retrieval_config)

        elapsed_ms = (time.time() - start_time) * 1000

        return RetrievalContext(
            query=query,
            results=results,
            total_retrieved=len(results),
            retrieval_time_ms=elapsed_ms,
        )

    async def cognitive_search(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[str] = None,
        agent_registry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute 4-phase DSPy-enhanced cognitive RAG workflow.

        This method provides LLM-powered multi-hop reasoning with:
        - Phase 1 (Summarizer): Query rewriting, entity extraction, intent classification
        - Phase 2 (Investigator): Multi-hop evidence gathering with adaptive retrieval
        - Phase 3 (Agent): Response synthesis and agent routing
        - Phase 4 (Reflector): Memory consolidation and procedure learning

        Args:
            query: Natural language query from user
            conversation_id: Optional conversation ID for session tracking
            conversation_history: Optional compressed conversation history
            agent_registry: Optional registry of available downstream agents

        Returns:
            Dict with:
                - response: Synthesized natural language response
                - evidence: List of evidence pieces gathered
                - hop_count: Number of retrieval hops performed
                - visualization_config: Chart configuration if applicable
                - routed_agents: Agents recommended for further processing
                - dspy_signals: Training signals for optimization
                - latency_ms: Total processing time

        Example:
            result = await rag.cognitive_search(
                query="Why did Kisqali adoption increase in the Northeast?",
                conversation_id="session-123"
            )
            print(result["response"])
        """
        start_time = time.time()

        try:
            # Lazy import to avoid circular dependencies
            import dspy

            from src.rag.cognitive_backends import get_cognitive_memory_backends
            from src.rag.cognitive_rag_dspy import (
                CognitiveState,
                create_dspy_cognitive_workflow,
            )

            # Configure DSPy LM if not already configured. Provider-aware: use
            # the same model the rest of the app uses (resolved from LLM_PROVIDER /
            # ANTHROPIC_MODEL / DSPY_LM_MODEL) rather than a hardcoded Anthropic
            # model the deployed key may not serve — a bare retired model raises
            # litellm.NotFoundError (404), which previously broke the brief.
            if not hasattr(dspy.settings, "lm") or dspy.settings.lm is None:
                from src.optimization.dspy_lm import (
                    dspy_provider_api_key_present,
                    get_default_dspy_model,
                )

                if not dspy_provider_api_key_present():
                    raise ValueError(
                        "No API key for the configured LLM provider "
                        "(set OPENAI_API_KEY or ANTHROPIC_API_KEY per LLM_PROVIDER) "
                        "required for cognitive search"
                    )
                lm = dspy.LM(get_default_dspy_model())
                dspy.configure(lm=lm)
                logger.info("Configured DSPy LM for cognitive workflow")

            # Get real memory backends
            backends = get_cognitive_memory_backends()

            # Domain vocabulary for pharmaceutical context
            domain_vocabulary = """
            Brands: Remibrutinib (CSU), Fabhalta (PNH), Kisqali (HR+/HER2- breast cancer)
            KPIs: TRx, NRx, market_share, conversion_rate, adoption_rate
            Regions: Northeast, Southwest, Midwest, West, South, East
            Entities: HCP, physician, territory, brand, region
            """

            # Create workflow with real backends
            workflow = create_dspy_cognitive_workflow(
                memory_backends=backends["readers"],
                memory_writers=backends["writers"],
                agent_registry=agent_registry or {},
                signal_collector=backends["signal_collector"],
                domain_vocabulary=domain_vocabulary.strip(),
            )

            # Initialize cognitive state
            import uuid

            # Resolve a stable conversation/thread identifier. The compiled
            # LangGraph workflow uses a MemorySaver checkpointer, which REQUIRES
            # a `thread_id` in the run config; without it, `ainvoke` raises
            # "Checkpointer requires one or more of the following 'configurable'
            # keys: thread_id, checkpoint_ns, checkpoint_id" and the whole
            # request fails (returning an error string to the caller). We seed
            # both the state and the checkpointer thread from the same id so a
            # real conversation maps to a single LangGraph thread.
            resolved_conversation_id = conversation_id or str(uuid.uuid4())

            initial_state = CognitiveState(
                user_query=query,
                conversation_id=resolved_conversation_id,
                compressed_history=conversation_history or "",
            )

            # Execute cognitive cycle. The thread_id config is MANDATORY here
            # because the workflow is compiled with a checkpointer (see
            # create_dspy_cognitive_workflow); omitting it raises a ValueError.
            run_config = {"configurable": {"thread_id": resolved_conversation_id}}
            raw_result = await workflow.ainvoke(initial_state, config=run_config)  # type: ignore[attr-defined]

            # A compiled LangGraph returns a *dict* of channel values, not the
            # CognitiveState dataclass it was seeded with. The consumer below
            # uses attribute access (result_state.evidence_board, .response,
            # ... 10 fields), so a raw dict raised
            # "'dict' object has no attribute 'evidence_board'", which the
            # except clause then surfaced in-band as error-as-data. Coerce the
            # channel-value dict back into a CognitiveState (its keys are
            # exactly the CognitiveState field names) so every downstream
            # attribute access is valid and correctly typed. Tolerate an
            # already-CognitiveState return (future LangGraph behavior) and
            # filter to known fields so an extra channel key can never break
            # construction.
            result_state = _coerce_cognitive_state(raw_result)

            elapsed_ms = (time.time() - start_time) * 1000

            # Convert Evidence objects to dicts for serialization
            evidence_list = []
            for ev in result_state.evidence_board:
                if hasattr(ev, "__dict__"):
                    evidence_list.append(
                        asdict(ev) if hasattr(ev, "__dataclass_fields__") else ev.__dict__
                    )
                else:
                    evidence_list.append({"content": str(ev)})

            return {
                "response": result_state.response,
                "evidence": evidence_list,
                "hop_count": result_state.hop_count,
                "visualization_config": result_state.visualization_config,
                "routed_agents": result_state.routed_agents,
                "entities": result_state.extracted_entities,
                "intent": result_state.detected_intent,
                "rewritten_query": result_state.rewritten_query,
                "dspy_signals": result_state.dspy_signals,
                "worth_remembering": result_state.worth_remembering,
                "latency_ms": elapsed_ms,
            }

        except ImportError as e:
            logger.error(f"Cognitive search import error: {e}")
            raise RuntimeError(f"Cognitive search requires additional dependencies: {e}") from e
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Cognitive search failed: {e}")
            return {
                "response": f"Unable to complete cognitive search: {str(e)[:200]}",
                "evidence": [],
                "hop_count": 0,
                "visualization_config": {},
                "routed_agents": [],
                "entities": [],
                "intent": "",
                "rewritten_query": query,
                "dspy_signals": [],
                "worth_remembering": False,
                "latency_ms": elapsed_ms,
                "error": str(e),
            }
