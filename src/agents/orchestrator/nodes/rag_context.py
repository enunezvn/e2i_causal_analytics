"""RAG context retrieval node for orchestrator agent.

Retrieves relevant context from the Hybrid RAG system to augment
agent responses with:
- Historical causal insights
- Agent activity history
- Related business metrics
- Causal graph paths
"""

import logging
import time
from typing import Any, Dict, List, Optional, cast

from src.rag.types import ExtractedEntities, RetrievalResult

from ..state import OrchestratorState

logger = logging.getLogger(__name__)


class RAGContextNode:
    """RAG context retrieval node for enriching orchestrator context.

    Retrieves relevant context from hybrid RAG based on:
    - Extracted entities from intent classification
    - Query semantics (via embedding)
    - Causal graph relationships

    Latency target: <500ms
    """

    # Maximum results to retrieve for context
    DEFAULT_TOP_K = 5

    # Timeout for RAG search
    TIMEOUT_MS = 500

    def __init__(
        self,
        retriever: Optional[Any] = None,
        embedding_service: Optional[Any] = None,
        entity_extractor: Optional[Any] = None,
    ):
        """Initialize RAG context node.

        Args:
            retriever: HybridRetriever instance
            embedding_service: Embedding service for query vectorization
            entity_extractor: Entity extractor for query parsing
        """
        self._retriever = retriever
        self._embedding_service = embedding_service
        self._entity_extractor = entity_extractor
        self._initialized = False

    async def initialize(self):
        """Lazy initialization of dependencies.

        Called on first execute() if not already initialized.
        """
        if self._initialized:
            return

        # Import here to avoid circular dependencies
        try:
            if self._retriever is None:
                from src.api.dependencies import get_rag_dependencies

                deps = await get_rag_dependencies()
                self._retriever = deps.get("retriever")
                self._embedding_service = deps.get("embedding_service")
                self._entity_extractor = deps.get("entity_extractor")
        except ImportError:
            logger.warning("RAG dependencies not available - using mock mode")
            self._retriever = None

        self._initialized = True

    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute RAG context retrieval.

        Args:
            state: Current orchestrator state (after classification)

        Returns:
            Updated state with RAG context
        """
        start_time = time.time()

        query = state.get("query", "")
        intent_data = state.get("intent")
        intent: dict[str, Any] = dict(intent_data) if intent_data else {}

        # Skip if no query or if intent is system-related
        primary_intent = intent.get("primary_intent")
        if not query or primary_intent in ["system_health", "drift_check"]:
            logger.debug("Skipping RAG context - not applicable for this intent")
            return {
                **state,
                "rag_context": None,
                "rag_latency_ms": 0,
            }

        try:
            await self.initialize()

            if self._retriever is None:
                # Mock mode - return empty context
                logger.debug("RAG retriever not available - skipping context retrieval")
                return {
                    **state,
                    "rag_context": None,
                    "rag_latency_ms": int((time.time() - start_time) * 1000),
                }

            # Extract entities from query if not already done
            entities = await self._extract_entities(query, state)

            # Generate embedding if embedding service available
            embedding = await self._generate_embedding(query)

            # Build filters from entities
            filters = self._build_filters_from_entities(entities)

            # Execute hybrid search
            results = await self._retriever.search(
                query=query,
                embedding=embedding,
                entities=entities,
                filters=filters,
                top_k=self.DEFAULT_TOP_K,
            )

            # Format context for agents
            rag_context = self._format_context(results)

            rag_latency_ms = int((time.time() - start_time) * 1000)

            logger.info(f"RAG context retrieved: {len(results)} results in {rag_latency_ms}ms")

            return {  # type: ignore[typeddict-unknown-key]
                **state,
                "rag_context": rag_context,
                "rag_results": [self._result_to_dict(r) for r in results],
                "rag_latency_ms": rag_latency_ms,
                "entities_extracted": self._entities_to_dict(entities) if entities else None,
            }

        except Exception as e:
            rag_latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"RAG context retrieval failed: {e}")

            # Don't fail the pipeline - continue without context
            return {
                **state,
                "rag_context": None,
                "rag_latency_ms": rag_latency_ms,
                "warnings": state.get("warnings", []) + [f"RAG context retrieval failed: {str(e)}"],
            }

    async def _extract_entities(
        self, query: str, state: OrchestratorState
    ) -> Optional[ExtractedEntities]:
        """Extract entities from query.

        Uses existing entities from state if available, otherwise extracts.
        """
        # Check if entities already in state
        existing = state.get("entities_extracted")
        if existing:
            return ExtractedEntities(
                brands=existing.get("brands", []),
                regions=existing.get("regions", []),
                kpis=existing.get("kpis", []),
                agents=existing.get("agents", []),
                journey_stages=existing.get("journey_stages", []),
                time_references=existing.get("time_references", []),
                hcp_segments=existing.get("hcp_segments", []),
            )

        # Extract entities
        if self._entity_extractor:
            try:
                return cast(ExtractedEntities, self._entity_extractor.extract(query))
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")

        return ExtractedEntities()

    async def _generate_embedding(self, query: str) -> Optional[List[float]]:
        """Generate query embedding."""
        if not self._embedding_service:
            return None

        try:
            return cast(List[float], await self._embedding_service.embed(query))
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    def _build_filters_from_entities(self, entities: Optional[ExtractedEntities]) -> Dict[str, Any]:
        """Build search filters from extracted entities."""
        if not entities:
            return {}

        filters = {}

        if entities.brands:
            filters["brands"] = entities.brands

        if entities.regions:
            filters["regions"] = entities.regions

        if entities.kpis:
            filters["kpis"] = entities.kpis

        return filters

    def _format_context(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Format RAG results into context for agents.

        Returns:
            Dict with structured context:
            - summary: Brief text summary of relevant context
            - insights: List of relevant insights
            - causal_paths: Any causal relationships found
            - sources: Source attribution
        """
        if not results:
            return {"summary": "", "insights": [], "causal_paths": [], "sources": []}

        # Group by source type
        insights = []
        causal_paths = []
        sources = []

        for result in results:
            # Add to insights
            insights.append(
                {
                    "content": result.content[:500],  # Truncate for context
                    "score": result.score,
                    "source": result.source.value,
                    "metadata": {
                        k: v
                        for k, v in result.metadata.items()
                        if k in ["brand", "kpi", "region", "source_table", "agent_name"]
                    },
                }
            )

            # Extract causal paths if present
            if result.graph_context and hasattr(result.graph_context, "paths"):
                for path in result.graph_context.paths:  # type: ignore[attr-defined]
                    causal_paths.append(
                        {
                            "path": " → ".join(path.node_names) if path.node_names else str(path),
                            "relationship": (
                                path.relationship_type
                                if hasattr(path, "relationship_type")
                                else None
                            ),
                            "strength": path.weight if hasattr(path, "weight") else 1.0,
                        }
                    )

            # Track sources
            sources.append(
                {"id": result.id, "source": result.source.value, "relevance": result.score}
            )

        # Build summary from top insights
        summary_parts = [r.content[:100] for r in results[:3]]
        summary = " | ".join(summary_parts) if summary_parts else ""

        return {
            "summary": summary,
            "insights": insights,
            "causal_paths": causal_paths[:5],  # Top 5 causal paths
            "sources": sources,
        }

    def _result_to_dict(self, result: RetrievalResult) -> Dict[str, Any]:
        """Convert RetrievalResult to dict for state storage."""
        return {
            "id": result.id,
            "content": result.content,
            "score": result.score,
            "source": result.source.value,
            "metadata": result.metadata,
        }

    def _entities_to_dict(self, entities: ExtractedEntities) -> Dict[str, List[str]]:
        """Convert ExtractedEntities to dict."""
        return {
            "brands": entities.brands or [],
            "regions": entities.regions or [],
            "kpis": entities.kpis or [],
            "agents": entities.agents or [],
            "journey_stages": entities.journey_stages or [],
            "time_references": entities.time_references or [],
            "hcp_segments": entities.hcp_segments or [],
        }


# Export for use in graph
async def retrieve_rag_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node function for RAG context retrieval.

    Args:
        state: Current state

    Returns:
        Updated state with RAG context
    """
    node = RAGContextNode()
    result = await node.execute(cast(OrchestratorState, state))
    return cast(Dict[str, Any], result)
