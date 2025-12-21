"""
CausalRAG - Main orchestrator for graph-enhanced retrieval.

This is the primary entry point for RAG operations in E2I Causal Analytics.
Supports both traditional hybrid retrieval and DSPy-enhanced cognitive workflows.
"""

import logging
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from src.rag.models.retrieval_models import RetrievalContext, RetrievalResult

logger = logging.getLogger(__name__)


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
            return self.reranker.rerank(all_results, query, top_k=top_k)

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

            # Configure DSPy LM if not already configured
            if not hasattr(dspy.settings, "lm") or dspy.settings.lm is None:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY required for cognitive search")
                lm = dspy.LM("anthropic/claude-sonnet-4-20250514")
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

            initial_state = CognitiveState(
                user_query=query,
                conversation_id=conversation_id or str(uuid.uuid4()),
                compressed_history=conversation_history or "",
            )

            # Execute cognitive cycle
            result_state = await workflow.ainvoke(initial_state)

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
