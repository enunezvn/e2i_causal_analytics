"""
Real Memory Backend Adapters for Cognitive RAG Workflow.

Provides concrete implementations that replace the mock backends
in cognitive_rag_dspy.py with real memory system integrations.

Usage:
    from src.rag.cognitive_backends import get_cognitive_memory_backends

    backends = get_cognitive_memory_backends()
    workflow = create_dspy_cognitive_workflow(
        memory_backends=backends["readers"],
        memory_writers=backends["writers"],
        signal_collector=backends["signal_collector"],
        ...
    )
"""

import logging
from typing import Any, Dict, List, Optional

from src.memory.procedural_memory import (
    LearningSignalInput,
    ProceduralMemoryInput,
    find_relevant_procedures_by_text,
    insert_procedural_memory,
    record_learning_signal,
)
from src.rag.memory_connector import get_memory_connector

logger = logging.getLogger(__name__)


class EpisodicMemoryBackend:
    """
    Episodic Memory Backend using Supabase pgvector.

    Provides conversation history and agent action retrieval
    via vector similarity search.
    """

    def __init__(self):
        self._connector = None

    @property
    def connector(self):
        """Lazy-load memory connector."""
        if self._connector is None:
            self._connector = get_memory_connector()
        return self._connector

    async def vector_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search episodic memories by semantic similarity.

        Args:
            query: Search query text
            limit: Maximum results to return

        Returns:
            List of dicts with "content" field for workflow compatibility
        """
        try:
            results = await self.connector.vector_search_by_text(
                query_text=query, k=limit, min_similarity=0.5
            )

            # Convert RetrievalResult to workflow-compatible dict format
            return [
                {
                    "content": r.content,
                    "source": r.source,
                    "source_id": r.source_id,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Episodic memory search failed: {e}")
            return []

    async def store_episode(
        self, content: str, episode_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Store a new episode in episodic memory.

        Args:
            content: Episode content
            episode_type: Type of episode (conversation, agent_action, etc.)
            metadata: Additional metadata

        Returns:
            Episode ID if successful, None otherwise
        """
        # TODO: Implement episode storage via Supabase
        logger.warning("Episode storage not yet implemented")
        return None


class SemanticMemoryBackend:
    """
    Semantic Memory Backend using FalkorDB graph.

    Provides graph-based retrieval for entity relationships,
    causal chains, and network traversal.
    """

    def __init__(self):
        self._connector = None

    @property
    def connector(self):
        """Lazy-load memory connector."""
        if self._connector is None:
            self._connector = get_memory_connector()
        return self._connector

    async def graph_query(self, query: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Query semantic graph for related entities.

        The cognitive workflow passes a text query, which we use
        to identify entities and traverse the graph.

        Args:
            query: Search query (may contain entity references)
            max_depth: Maximum traversal depth

        Returns:
            List of dicts with "content" field for workflow compatibility
        """
        try:
            # Extract potential entity IDs from query
            # For now, use the query as entity ID (will be enhanced with NER)
            entity_id = self._extract_entity_id(query)

            if entity_id:
                results = self.connector.graph_traverse(
                    entity_id=entity_id, relationship="causal_path", max_depth=max_depth
                )
            else:
                # Fall back to KPI-based traversal if query mentions KPIs
                kpi = self._extract_kpi(query)
                if kpi:
                    results = self.connector.graph_traverse_kpi(kpi_name=kpi, min_confidence=0.5)
                else:
                    # No entity found, return empty
                    results = []

            # Convert RetrievalResult to workflow-compatible dict format
            return [
                {
                    "content": r.content,
                    "source": r.source,
                    "source_id": r.source_id,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Semantic memory graph query failed: {e}")
            return []

    def _extract_entity_id(self, query: str) -> Optional[str]:
        """
        Extract entity ID from query text.

        Simple heuristic for now - can be enhanced with NER.
        """
        # Look for common entity patterns in E2I domain
        query_lower = query.lower()

        # Check for brand names
        brands = ["kisqali", "fabhalta", "remibrutinib"]
        for brand in brands:
            if brand in query_lower:
                return f"brand:{brand}"

        # Check for region patterns
        regions = ["northeast", "southwest", "midwest", "west", "south", "east"]
        for region in regions:
            if region in query_lower:
                return f"region:{region}"

        # Check for HCP patterns
        if "hcp" in query_lower or "physician" in query_lower:
            return "entity:hcp"

        return None

    def _extract_kpi(self, query: str) -> Optional[str]:
        """Extract KPI name from query text."""
        query_lower = query.lower()

        kpis = {
            "trx": "TRx",
            "nrx": "NRx",
            "total prescriptions": "TRx",
            "new prescriptions": "NRx",
            "conversion": "conversion_rate",
            "market share": "market_share",
            "adoption": "adoption_rate",
        }

        for pattern, kpi_name in kpis.items():
            if pattern in query_lower:
                return kpi_name

        return None

    async def store_relationship(
        self,
        source_entity: str,
        target_entity: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store a new relationship in semantic graph.

        Args:
            source_entity: Source entity ID
            target_entity: Target entity ID
            relationship_type: Type of relationship
            properties: Relationship properties

        Returns:
            True if successful, False otherwise
        """
        # TODO: Implement relationship storage via FalkorDB
        logger.warning("Relationship storage not yet implemented")
        return False


class ProceduralMemoryBackend:
    """
    Procedural Memory Backend using Supabase pgvector.

    Provides tool sequence patterns and few-shot examples
    for DSPy in-context learning.
    """

    async def procedure_search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant procedures/patterns.

        Args:
            query: Search query text
            limit: Maximum results to return

        Returns:
            List of dicts with "content" field for workflow compatibility
        """
        try:
            results = await find_relevant_procedures_by_text(
                query_text=query, limit=limit, min_similarity=0.5
            )

            # Convert to workflow-compatible format
            formatted_results = []
            for r in results:
                # Build content from procedure data
                tool_sequence = r.get("tool_sequence", [])
                procedure_name = r.get("procedure_name", "Unknown procedure")

                if tool_sequence:
                    content = f"{procedure_name}: " + " → ".join(
                        step.get("tool", str(step)) for step in tool_sequence
                    )
                else:
                    content = procedure_name

                formatted_results.append(
                    {
                        "content": content,
                        "source": "procedural_memory",
                        "source_id": r.get("id", ""),
                        "score": r.get("similarity", 0.7),
                        "metadata": {
                            "procedure_type": r.get("procedure_type"),
                            "success_rate": r.get("success_rate", 0.0),
                            "execution_count": r.get("execution_count", 0),
                        },
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Procedural memory search failed: {e}")
            return []

    async def store_procedure(
        self,
        procedure_name: str,
        tool_sequence: List[Dict[str, Any]],
        trigger_pattern: Optional[str] = None,
        intent: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> Optional[str]:
        """
        Store a new procedure in memory.

        Args:
            procedure_name: Name of the procedure
            tool_sequence: Sequence of tool calls
            trigger_pattern: Pattern that triggers this procedure
            intent: Detected intent for filtering
            embedding: Pre-computed embedding (optional)

        Returns:
            Procedure ID if successful, None otherwise
        """
        try:
            from src.memory.services.factories import get_embedding_service

            # Generate embedding if not provided
            if embedding is None:
                embed_service = get_embedding_service()
                trigger_text = trigger_pattern or procedure_name
                embedding = await embed_service.embed(trigger_text)

            procedure = ProceduralMemoryInput(
                procedure_name=procedure_name,
                tool_sequence=tool_sequence,
                trigger_pattern=trigger_pattern,
                detected_intent=intent,
            )

            result = await insert_procedural_memory(procedure, embedding)
            return result.get("id") if result else None

        except Exception as e:
            logger.error(f"Procedure storage failed: {e}")
            return None


class SignalCollector:
    """
    Signal Collector for DSPy training feedback.

    Collects learning signals from the cognitive workflow
    for optimization via the Feedback Learner agent.
    """

    def __init__(self):
        self._pending_signals: List[Dict[str, Any]] = []

    async def collect(self, signals: List[Dict[str, Any]]) -> None:
        """
        Collect DSPy training signals.

        Args:
            signals: List of signal dictionaries with:
                - signature_name: Name of the DSPy signature
                - input: Input to the signature
                - output: Output from the signature
                - metric: Optional metric value
        """
        for signal in signals:
            try:
                # Convert to LearningSignalInput format
                learning_signal = LearningSignalInput(
                    signal_type="dspy_signal",
                    is_training_example=True,
                    dspy_metric_name=signal.get("signature_name"),
                    dspy_metric_value=signal.get("metric"),
                    training_input=str(signal.get("input", "")),
                    training_output=str(signal.get("output", "")),
                    signal_details=signal,
                )

                await record_learning_signal(
                    signal=learning_signal, cycle_id=signal.get("cycle_id", "unknown")
                )

                logger.debug(f"Collected signal for {signal.get('signature_name')}")

            except Exception as e:
                logger.warning(f"Failed to collect signal: {e}")
                # Queue for retry
                self._pending_signals.append(signal)

        if self._pending_signals:
            logger.warning(f"{len(self._pending_signals)} signals pending retry")

    async def flush_pending(self) -> int:
        """
        Retry pending signals.

        Returns:
            Number of successfully flushed signals
        """
        if not self._pending_signals:
            return 0

        pending = self._pending_signals.copy()
        self._pending_signals = []

        await self.collect(pending)

        flushed = len(pending) - len(self._pending_signals)
        return flushed


def get_cognitive_memory_backends() -> Dict[str, Any]:
    """
    Get configured memory backends for cognitive workflow.

    Returns:
        Dict with:
            - readers: Dict of memory backend readers
            - writers: Dict of memory backend writers
            - signal_collector: Signal collector instance
    """
    episodic = EpisodicMemoryBackend()
    semantic = SemanticMemoryBackend()
    procedural = ProceduralMemoryBackend()
    signal_collector = SignalCollector()

    return {
        "readers": {"episodic": episodic, "semantic": semantic, "procedural": procedural},
        "writers": {"episodic": episodic, "semantic": semantic, "procedural": procedural},
        "signal_collector": signal_collector,
    }
