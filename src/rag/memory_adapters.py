"""
Memory Backend Adapters for Cognitive RAG DSPy Integration.

These adapters wrap the real memory implementations (MemoryConnector, FalkorDB,
Procedural Memory, Feedback Learner) to match the interface expected by the
CognitiveRAGWorkflow in cognitive_rag_dspy.py.

The adapters translate between:
- Real backend APIs (Supabase RPC, FalkorDB Cypher, Pydantic models)
- DSPy workflow expectations (simple dicts with "content" key)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol Definitions (what cognitive_rag_dspy.py expects)
# =============================================================================


@runtime_checkable
class EpisodicMemoryProtocol(Protocol):
    """Protocol for episodic memory access (vector search)."""

    async def vector_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search episodic memory by semantic similarity."""
        ...


@runtime_checkable
class SemanticMemoryProtocol(Protocol):
    """Protocol for semantic memory access (graph queries)."""

    async def graph_query(self, query: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Query semantic graph for related concepts."""
        ...


@runtime_checkable
class ProceduralMemoryProtocol(Protocol):
    """Protocol for procedural memory access."""

    async def procedure_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant procedures/patterns."""
        ...


@runtime_checkable
class SignalCollectorProtocol(Protocol):
    """Protocol for training signal collection."""

    async def collect(self, signals: List[Dict[str, Any]]) -> None:
        """Collect training signals for future optimization."""
        ...


# =============================================================================
# Episodic Memory Adapter (wraps MemoryConnector)
# =============================================================================


class EpisodicMemoryAdapter:
    """
    Adapter for episodic memory access via MemoryConnector.

    Wraps the MemoryConnector's vector_search methods to provide the simple
    interface expected by CognitiveRAGWorkflow.

    Example:
        connector = MemoryConnector(supabase_client)
        adapter = EpisodicMemoryAdapter(connector)
        results = await adapter.vector_search("Kisqali adoption trends", limit=10)
        # Returns: [{"content": "...", "source": "...", "score": 0.9}, ...]
    """

    def __init__(
        self,
        memory_connector: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
    ):
        """
        Initialize episodic memory adapter.

        Args:
            memory_connector: MemoryConnector instance for Supabase access
            embedding_model: Optional embedding model for query vectorization
        """
        self._connector = memory_connector
        self._embedding_model = embedding_model

    async def vector_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search episodic memory using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of results

        Returns:
            List of dicts with "content", "source", "score", and metadata
        """
        if self._connector is None:
            logger.warning("EpisodicMemoryAdapter: No connector configured, returning empty")
            return []

        try:
            # Use text-based search if no embedding model provided
            if self._embedding_model is None:
                results = await self._connector.vector_search_by_text(
                    query_text=query,
                    limit=limit,
                )
            else:
                # Generate embedding and search
                embedding = await self._generate_embedding(query)
                results = await self._connector.vector_search(
                    query_embedding=embedding,
                    limit=limit,
                )

            # Transform to expected format
            return self._transform_results(results)

        except Exception as e:
            logger.error(f"Episodic memory search failed: {e}")
            return []

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using configured model."""
        if hasattr(self._embedding_model, "embed"):
            return await self._embedding_model.embed(text)
        elif hasattr(self._embedding_model, "encode"):
            # Synchronous model (e.g., sentence-transformers)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._embedding_model.encode, text)
        else:
            raise ValueError("Embedding model must have 'embed' or 'encode' method")

    def _transform_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Transform MemoryConnector results to expected format."""
        transformed = []
        for result in results:
            if hasattr(result, "content"):
                # RetrievalResult or similar object
                transformed.append(
                    {
                        "content": result.content,
                        "source": getattr(result, "source", "episodic"),
                        "source_id": getattr(result, "source_id", None),
                        "score": getattr(result, "score", 0.0),
                        "metadata": getattr(result, "metadata", {}),
                    }
                )
            elif isinstance(result, dict):
                transformed.append(
                    {
                        "content": result.get("content", str(result)),
                        "source": result.get("source", "episodic"),
                        "source_id": result.get("source_id"),
                        "score": result.get("score", 0.0),
                        "metadata": result.get("metadata", {}),
                    }
                )
            else:
                transformed.append(
                    {
                        "content": str(result),
                        "source": "episodic",
                        "score": 0.0,
                    }
                )

        return transformed


# =============================================================================
# Semantic Memory Adapter (wraps FalkorDB)
# =============================================================================


class SemanticMemoryAdapter:
    """
    Adapter for semantic memory access via FalkorDB.

    Wraps the FalkorDBSemanticMemory graph operations to provide the simple
    interface expected by CognitiveRAGWorkflow.

    Example:
        semantic_mem = FalkorDBSemanticMemory(client)
        adapter = SemanticMemoryAdapter(semantic_mem)
        results = await adapter.graph_query("Kisqali → HCP targeting", max_depth=2)
        # Returns: [{"content": "HCP targeting CONNECTED_TO Kisqali adoption...", ...}]
    """

    def __init__(
        self,
        falkordb_memory: Optional[Any] = None,
        memory_connector: Optional[Any] = None,
    ):
        """
        Initialize semantic memory adapter.

        Args:
            falkordb_memory: FalkorDBSemanticMemory instance for graph operations
            memory_connector: Optional MemoryConnector for fallback graph traversal
        """
        self._falkordb = falkordb_memory
        self._connector = memory_connector

    async def graph_query(self, query: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Query semantic graph for related concepts.

        Args:
            query: Query describing relationships to find
            max_depth: Maximum traversal depth

        Returns:
            List of dicts with "content" describing graph relationships
        """
        results = []

        # Try FalkorDB first
        if self._falkordb is not None:
            try:
                falkor_results = await self._query_falkordb(query, max_depth)
                results.extend(falkor_results)
            except Exception as e:
                logger.warning(f"FalkorDB query failed: {e}")

        # Fallback to MemoryConnector graph traversal
        if not results and self._connector is not None:
            try:
                connector_results = await self._query_connector_graph(query, max_depth)
                results.extend(connector_results)
            except Exception as e:
                logger.warning(f"MemoryConnector graph query failed: {e}")

        if not results:
            logger.warning("SemanticMemoryAdapter: No graph results found")

        return results

    async def _query_falkordb(self, query: str, max_depth: int) -> List[Dict[str, Any]]:
        """Query FalkorDB semantic graph."""
        # Extract entities from query for targeted graph search
        entities = self._extract_entities(query)

        all_results = []

        for entity in entities:
            # Find related nodes within max_depth hops
            if hasattr(self._falkordb, "find_related"):
                related = await self._falkordb.find_related(
                    entity_type=entity.get("type", "Entity"),
                    entity_id=entity.get("id"),
                    max_hops=max_depth,
                )
                all_results.extend(self._transform_graph_results(related))

            # Also try semantic similarity search in graph
            if hasattr(self._falkordb, "semantic_search"):
                semantic = await self._falkordb.semantic_search(
                    query=query,
                    limit=10,
                )
                all_results.extend(self._transform_graph_results(semantic))

        return all_results

    async def _query_connector_graph(self, query: str, max_depth: int) -> List[Dict[str, Any]]:
        """Query graph via MemoryConnector's graph_traverse."""
        # Extract starting entities
        entities = self._extract_entities(query)

        results = []
        for entity in entities:
            try:
                traversal = await self._connector.graph_traverse(
                    start_node_id=entity.get("id"),
                    relationship_types=None,  # All relationships
                    max_depth=max_depth,
                )
                results.extend(self._transform_graph_results(traversal))
            except Exception as e:
                logger.debug(f"Graph traverse failed for {entity}: {e}")

        return results

    def _extract_entities(self, query: str) -> List[Dict[str, str]]:
        """Extract entity mentions from query for graph lookup."""
        # Simple entity extraction based on known brands/terms
        known_entities = {
            "kisqali": {"type": "Drug", "id": "kisqali"},
            "fabhalta": {"type": "Drug", "id": "fabhalta"},
            "remibrutinib": {"type": "Drug", "id": "remibrutinib"},
            "hcp": {"type": "HCP", "id": "hcp_generic"},
            "trx": {"type": "KPI", "id": "trx"},
            "nrx": {"type": "KPI", "id": "nrx"},
        }

        query_lower = query.lower()
        found = []

        for term, entity in known_entities.items():
            if term in query_lower:
                found.append(entity)

        # Default to generic entity if nothing found
        if not found:
            found.append({"type": "Query", "id": query[:50]})

        return found

    def _transform_graph_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Transform graph results to expected format."""
        transformed = []

        for result in results:
            if isinstance(result, dict):
                # Build content description from graph structure
                content = self._build_content_from_graph_node(result)
                transformed.append(
                    {
                        "content": content,
                        "source": "semantic_graph",
                        "relationship": result.get("relationship"),
                        "metadata": result,
                    }
                )
            elif hasattr(result, "to_dict"):
                node_dict = result.to_dict()
                content = self._build_content_from_graph_node(node_dict)
                transformed.append(
                    {
                        "content": content,
                        "source": "semantic_graph",
                        "metadata": node_dict,
                    }
                )
            else:
                transformed.append(
                    {
                        "content": str(result),
                        "source": "semantic_graph",
                    }
                )

        return transformed

    def _build_content_from_graph_node(self, node: Dict[str, Any]) -> str:
        """Build human-readable content from graph node."""
        parts = []

        # Node type and ID
        if "type" in node:
            parts.append(f"{node['type']}")
        if "id" in node or "name" in node:
            parts.append(f"'{node.get('name', node.get('id'))}'")

        # Relationship info
        if "relationship" in node:
            parts.append(f"CONNECTED_VIA {node['relationship']}")

        # Target node
        if "target" in node:
            target = node["target"]
            if isinstance(target, dict):
                parts.append(
                    f"TO {target.get('type', 'Entity')} '{target.get('name', target.get('id'))}'"
                )
            else:
                parts.append(f"TO {target}")

        # Properties
        props = {
            k: v
            for k, v in node.items()
            if k not in ("type", "id", "name", "relationship", "target")
        }
        if props:
            prop_str = ", ".join(f"{k}={v}" for k, v in list(props.items())[:3])
            parts.append(f"[{prop_str}]")

        return " ".join(parts) if parts else str(node)


# =============================================================================
# Procedural Memory Adapter (wraps procedural_memory.py functions)
# =============================================================================


class ProceduralMemoryAdapter:
    """
    Adapter for procedural memory access.

    Wraps the procedural memory functions to provide the simple interface
    expected by CognitiveRAGWorkflow.

    Example:
        adapter = ProceduralMemoryAdapter(supabase_client)
        procedures = await adapter.procedure_search("adoption analysis workflow", limit=5)
        # Returns: [{"content": "For adoption analysis: 1) Query episodic...", ...}]
    """

    def __init__(
        self,
        supabase_client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
    ):
        """
        Initialize procedural memory adapter.

        Args:
            supabase_client: Supabase client for database access
            embedding_model: Optional embedding model for similarity search
        """
        self._client = supabase_client
        self._embedding_model = embedding_model

    async def procedure_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant procedures/patterns.

        Args:
            query: Query describing the task or problem
            limit: Maximum number of results

        Returns:
            List of dicts with "content" describing procedures
        """
        if self._client is None:
            logger.warning("ProceduralMemoryAdapter: No client configured, returning empty")
            return []

        try:
            # Import procedural memory functions
            from src.memory.procedural_memory import find_relevant_procedures

            # Search for relevant procedures
            procedures = await self._execute_procedure_search(query, limit)

            return self._transform_procedure_results(procedures)

        except ImportError:
            logger.warning("Procedural memory module not available")
            return self._get_fallback_procedures(query)
        except Exception as e:
            logger.error(f"Procedural memory search failed: {e}")
            return self._get_fallback_procedures(query)

    async def _execute_procedure_search(self, query: str, limit: int) -> List[Any]:
        """Execute procedure search via Supabase."""
        # Try RPC function first
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.rpc(
                    "search_procedural_memory", {"query_text": query, "limit_count": limit}
                ).execute(),
            )
            return response.data if response.data else []
        except Exception as e:
            logger.debug(f"RPC search failed, trying table query: {e}")

        # Fallback to direct table query
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.table("procedural_memory").select("*").limit(limit).execute(),
            )
            return response.data if response.data else []
        except Exception as e:
            logger.warning(f"Direct table query failed: {e}")
            return []

    def _transform_procedure_results(self, procedures: List[Any]) -> List[Dict[str, Any]]:
        """Transform procedure results to expected format."""
        transformed = []

        for proc in procedures:
            if isinstance(proc, dict):
                # Build procedural content
                content = self._build_procedure_content(proc)
                transformed.append(
                    {
                        "content": content,
                        "source": "procedural",
                        "procedure_type": proc.get("procedure_type"),
                        "success_rate": proc.get("success_rate", 0.0),
                        "metadata": proc,
                    }
                )
            else:
                transformed.append(
                    {
                        "content": str(proc),
                        "source": "procedural",
                    }
                )

        return transformed

    def _build_procedure_content(self, proc: Dict[str, Any]) -> str:
        """Build human-readable content from procedure."""
        parts = []

        # Title/name
        if "name" in proc:
            parts.append(f"Procedure: {proc['name']}")
        elif "procedure_type" in proc:
            parts.append(f"Procedure Type: {proc['procedure_type']}")

        # Steps
        if "steps" in proc:
            steps = proc["steps"]
            if isinstance(steps, list):
                step_str = " → ".join(str(s) for s in steps[:5])
                parts.append(f"Steps: {step_str}")
            else:
                parts.append(f"Steps: {steps}")

        # Context
        if "context" in proc:
            parts.append(f"Context: {proc['context']}")

        # Pattern
        if "pattern" in proc:
            parts.append(f"Pattern: {proc['pattern']}")

        # Success rate
        if "success_rate" in proc:
            parts.append(f"Success Rate: {proc['success_rate']:.0%}")

        return "; ".join(parts) if parts else str(proc)

    def _get_fallback_procedures(self, query: str) -> List[Dict[str, Any]]:
        """Return fallback procedures when database unavailable."""
        # Common pharmaceutical analytics procedures
        fallback = [
            {
                "content": "For adoption analysis: 1) Query episodic memory for historical trends, 2) Check regional factors in semantic graph, 3) Compare against procedural patterns, 4) Synthesize findings",
                "source": "procedural_fallback",
                "procedure_type": "adoption_analysis",
            },
            {
                "content": "For TRx/NRx trends: 1) Retrieve prescription data from episodic, 2) Identify HCP segments, 3) Calculate growth rates, 4) Flag anomalies",
                "source": "procedural_fallback",
                "procedure_type": "prescription_analysis",
            },
            {
                "content": "For causal investigation: 1) Identify potential confounders, 2) Check for natural experiments, 3) Apply appropriate causal method, 4) Validate with refutation",
                "source": "procedural_fallback",
                "procedure_type": "causal_analysis",
            },
        ]

        # Filter to relevant procedures based on query
        query_lower = query.lower()
        relevant = []

        for proc in fallback:
            proc_lower = proc["content"].lower()
            if any(
                term in query_lower
                for term in ["adopt", "trx", "nrx", "prescri", "causal", "impact", "effect"]
            ):
                if any(term in proc_lower for term in query_lower.split()):
                    relevant.append(proc)

        return relevant if relevant else fallback[:2]


# =============================================================================
# Signal Collector Adapter (wraps Feedback Learner)
# =============================================================================


@dataclass
class CollectedSignal:
    """Represents a collected training signal."""

    signal_type: str
    query: str
    response: str
    feedback: Optional[Dict[str, Any]] = None
    reward: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalCollectorAdapter:
    """
    Adapter for training signal collection via Feedback Learner.

    Wraps the FeedbackLearnerTrainingSignal to collect DSPy optimization signals.

    Example:
        adapter = SignalCollectorAdapter(supabase_client)
        await adapter.collect([
            {"type": "response", "query": "...", "response": "...", "reward": 0.8}
        ])
    """

    def __init__(
        self,
        supabase_client: Optional[Any] = None,
        buffer_size: int = 100,
    ):
        """
        Initialize signal collector adapter.

        Args:
            supabase_client: Supabase client for persisting signals
            buffer_size: Size of in-memory signal buffer before flush
        """
        self._client = supabase_client
        self._buffer_size = buffer_size
        self._signal_buffer: List[CollectedSignal] = []
        self._flush_lock = asyncio.Lock()

    async def collect(self, signals: List[Dict[str, Any]]) -> None:
        """
        Collect training signals for future optimization.

        Args:
            signals: List of signal dicts with type, query, response, reward, etc.
        """
        for signal_dict in signals:
            signal = CollectedSignal(
                signal_type=signal_dict.get("type", "unknown"),
                query=signal_dict.get("query", ""),
                response=signal_dict.get("response", ""),
                feedback=signal_dict.get("feedback"),
                reward=signal_dict.get("reward", 0.0),
                metadata=signal_dict.get("metadata", {}),
            )
            self._signal_buffer.append(signal)

        logger.info(
            f"Collected {len(signals)} training signals (buffer: {len(self._signal_buffer)})"
        )

        # Flush if buffer full
        if len(self._signal_buffer) >= self._buffer_size:
            await self.flush()

    async def flush(self) -> int:
        """
        Flush signal buffer to persistent storage.

        Returns:
            Number of signals flushed
        """
        async with self._flush_lock:
            if not self._signal_buffer:
                return 0

            signals_to_flush = self._signal_buffer.copy()
            self._signal_buffer.clear()

        count = len(signals_to_flush)

        if self._client is None:
            logger.warning(f"No client configured, discarding {count} signals")
            return 0

        try:
            # Convert to storage format
            # Transform signals to match database schema
            records = [
                {
                    "source_agent": s.signal_type,  # Map signal_type to source_agent
                    "batch_id": s.metadata.get("batch_id"),
                    "input_context": {
                        "query": s.query,
                        "timestamp": s.timestamp.isoformat(),
                        **s.metadata,
                    },
                    "output": {
                        "response": s.response[:1000] if s.response else "",
                    },
                    "quality_metrics": (
                        {
                            "feedback": s.feedback,
                        }
                        if s.feedback
                        else {}
                    ),
                    "reward": s.reward,
                    "latency_breakdown": s.metadata.get("latency", {}),
                }
                for s in signals_to_flush
            ]

            # Persist to database
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.table("dspy_agent_training_signals").insert(records).execute(),
            )

            logger.info(f"Flushed {count} training signals to database")
            return count

        except Exception as e:
            logger.error(f"Failed to flush signals: {e}")
            # Re-add to buffer on failure
            self._signal_buffer.extend(signals_to_flush)
            return 0

    async def get_signals_for_optimization(
        self,
        signal_type: Optional[str] = None,
        min_reward: float = 0.0,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve signals for DSPy optimization.

        Args:
            signal_type: Filter by signal type
            min_reward: Minimum reward threshold
            limit: Maximum signals to retrieve

        Returns:
            List of signal dicts suitable for DSPy optimization
        """
        if self._client is None:
            logger.warning("No client configured, returning empty signals")
            return []

        try:
            query = self._client.table("dspy_agent_training_signals").select("*")

            if signal_type:
                query = query.eq("signal_type", signal_type)

            query = query.gte("reward", min_reward).limit(limit)

            response = await asyncio.get_event_loop().run_in_executor(None, lambda: query.execute())

            return response.data if response.data else []

        except Exception as e:
            logger.error(f"Failed to retrieve signals: {e}")
            return []


# =============================================================================
# Factory Function for Creating Adapters
# =============================================================================


def create_memory_adapters(
    supabase_client: Optional[Any] = None,
    falkordb_memory: Optional[Any] = None,
    memory_connector: Optional[Any] = None,
    embedding_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Create all memory adapters for cognitive RAG workflow.

    Args:
        supabase_client: Supabase client for database access
        falkordb_memory: FalkorDBSemanticMemory instance
        memory_connector: MemoryConnector instance
        embedding_model: Embedding model for vector operations

    Returns:
        Dict with keys: episodic, semantic, procedural, signals
    """
    return {
        "episodic": EpisodicMemoryAdapter(
            memory_connector=memory_connector,
            embedding_model=embedding_model,
        ),
        "semantic": SemanticMemoryAdapter(
            falkordb_memory=falkordb_memory,
            memory_connector=memory_connector,
        ),
        "procedural": ProceduralMemoryAdapter(
            supabase_client=supabase_client,
            embedding_model=embedding_model,
        ),
        "signals": SignalCollectorAdapter(
            supabase_client=supabase_client,
        ),
    }
