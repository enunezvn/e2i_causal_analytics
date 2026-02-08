"""
Cognitive Integration Bridge
============================

Bridges the cognitive workflow (004_cognitive_workflow.py) with production
memory services and the API layer.

This module provides:
1. CognitiveService - Main service for executing cognitive cycles
2. Memory integration with production backends
3. Session management via Redis working memory
4. Learning signal routing to Feedback Learner
5. ReflectorTaskManager - Async task tracking with timeout and monitoring

Author: E2I Causal Analytics Team
Version: 1.1.0
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any, Coroutine, Dict, List, Optional, Set, cast

from pydantic import BaseModel, Field

from src.memory.episodic_memory import (
    E2IEntityReferences,
    EpisodicMemoryInput,
    insert_episodic_memory_with_text,
)
from src.memory.procedural_memory import LearningSignalInput, record_learning_signal
from src.memory.working_memory import RedisWorkingMemory, get_working_memory
from src.rag.retriever import hybrid_search

logger = logging.getLogger(__name__)

# Graphiti integration for knowledge graph (mandatory)
from src.memory.graphiti_service import get_graphiti_service

# =============================================================================
# SERVICE MODELS
# =============================================================================


class CognitiveQueryInput(BaseModel):
    """Input for cognitive query processing."""

    query: str = Field(..., min_length=1, description="User's natural language query")
    session_id: Optional[str] = Field(None, description="Existing session ID for continuity")
    user_id: Optional[str] = Field(None, description="User identifier")
    brand: Optional[str] = Field(None, description="Brand context filter")
    region: Optional[str] = Field(None, description="Region context filter")
    include_evidence: bool = Field(True, description="Whether to include evidence trail")
    max_hops: int = Field(3, ge=1, le=5, description="Maximum investigation depth")


class CognitiveQueryOutput(BaseModel):
    """Output from cognitive query processing."""

    session_id: str
    cycle_id: str
    query: str
    query_type: str
    agent_used: str
    response: str
    confidence: float
    evidence: Optional[List[Dict[str, Any]]] = None
    visualization_config: Optional[Dict[str, Any]] = None
    phases_completed: List[str]
    processing_time_ms: float
    worth_remembering: bool


class PhaseResult(BaseModel):
    """Result from a single cognitive phase."""

    phase_name: str
    completed: bool
    duration_ms: float
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# =============================================================================
# REFLECTOR TASK MANAGER
# =============================================================================


class ReflectorTaskManager:
    """
    Manages async reflector tasks with timeout, error tracking, and graceful shutdown.

    The reflector phase runs asynchronously after the response is sent to avoid
    blocking user responses. This manager provides:

    1. Task tracking - Prevents garbage collection of pending tasks
    2. Timeout handling - Prevents runaway tasks from consuming resources
    3. Error monitoring - Logs failures without affecting user responses
    4. Graceful shutdown - Waits for pending tasks during service shutdown
    5. Statistics - Exposes success/failure counts for observability

    Configuration via environment:
        E2I_REFLECTOR_TIMEOUT: Timeout in seconds (default: 30)

    Usage:
        manager = ReflectorTaskManager()
        await manager.submit(self._run_reflector(...), cycle_id="abc123")

        # At shutdown
        await manager.wait_all(timeout=60.0)
    """

    def __init__(self, timeout_seconds: Optional[float] = None):
        """
        Initialize the reflector task manager.

        Args:
            timeout_seconds: Timeout for each reflector task.
                            Defaults to E2I_REFLECTOR_TIMEOUT env var or 30.0.
        """
        if timeout_seconds is None:
            timeout_seconds = float(os.environ.get("E2I_REFLECTOR_TIMEOUT", "30.0"))

        self._timeout = timeout_seconds
        self._pending_tasks: Set[asyncio.Task] = set()
        self._success_count = 0
        self._failed_count = 0
        self._timeout_count = 0
        self._lock = asyncio.Lock()

    async def submit(
        self,
        coro: Coroutine[Any, Any, None],
        cycle_id: str,
    ) -> None:
        """
        Submit a reflector coroutine for async execution with tracking.

        The coroutine will be wrapped with:
        - Timeout handling (default 30s)
        - Error logging
        - Automatic cleanup on completion

        Args:
            coro: The reflector coroutine to execute
            cycle_id: Identifier for logging and tracking
        """

        async def wrapped_reflector():
            """Wrapper that handles timeout, errors, and cleanup."""
            try:
                await asyncio.wait_for(coro, timeout=self._timeout)

                async with self._lock:
                    self._success_count += 1

                logger.debug(f"Reflector completed successfully for cycle {cycle_id}")

            except asyncio.TimeoutError:
                async with self._lock:
                    self._timeout_count += 1
                    self._failed_count += 1

                logger.warning(f"Reflector timeout after {self._timeout}s for cycle {cycle_id}")

            except asyncio.CancelledError:
                logger.debug(f"Reflector cancelled for cycle {cycle_id}")
                raise  # Re-raise to properly handle cancellation

            except Exception as e:
                async with self._lock:
                    self._failed_count += 1

                logger.error(
                    f"Reflector failed for cycle {cycle_id}: {e}",
                    exc_info=True,
                )

            finally:
                # Remove from pending set
                current_task = asyncio.current_task()
                async with self._lock:
                    self._pending_tasks.discard(current_task)

        # Create and track the task
        task = asyncio.create_task(wrapped_reflector())

        async with self._lock:
            self._pending_tasks.add(task)

        # Add done callback as backup cleanup (in case finally doesn't run)
        def cleanup_callback(t: asyncio.Task):
            # Use non-blocking discard since we can't await in callback
            self._pending_tasks.discard(t)

        task.add_done_callback(cleanup_callback)

    async def wait_all(self, timeout: float = 60.0) -> Dict[str, int]:
        """
        Wait for all pending reflector tasks to complete.

        Used during graceful shutdown to ensure learning signals are persisted.

        Args:
            timeout: Maximum time to wait for all tasks

        Returns:
            Dict with counts of completed, cancelled, and still pending tasks
        """
        async with self._lock:
            pending = set(self._pending_tasks)

        if not pending:
            return {"completed": 0, "cancelled": 0, "pending": 0}

        logger.info(f"Waiting for {len(pending)} pending reflector tasks...")

        done, still_pending = await asyncio.wait(
            pending,
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED,
        )

        # Cancel any tasks that didn't complete
        cancelled_count = 0
        for task in still_pending:
            task.cancel()
            cancelled_count += 1
            try:
                await task
            except asyncio.CancelledError:
                pass

        result = {
            "completed": len(done),
            "cancelled": cancelled_count,
            "pending": 0,
        }

        logger.info(
            f"Reflector shutdown complete: {result['completed']} completed, "
            f"{result['cancelled']} cancelled"
        )

        return result

    @property
    def pending_count(self) -> int:
        """Number of currently pending reflector tasks."""
        return len(self._pending_tasks)

    @property
    def stats(self) -> Dict[str, Any]:
        """
        Get reflector task statistics.

        Returns:
            Dict with pending, succeeded, failed, and timeout counts
        """
        return {
            "pending": len(self._pending_tasks),
            "succeeded": self._success_count,
            "failed": self._failed_count,
            "timeouts": self._timeout_count,
            "total_processed": self._success_count + self._failed_count,
            "timeout_seconds": self._timeout,
        }


# =============================================================================
# COGNITIVE SERVICE
# =============================================================================


class CognitiveService:
    """
    Main service for executing cognitive cycles.

    Integrates:
    - Working memory (Redis) for session context
    - Episodic memory (Supabase) for historical retrieval
    - Procedural memory for learned patterns
    - Semantic memory (FalkorDB) for graph traversal
    - ReflectorTaskManager for async learning task management
    """

    def __init__(self):
        """Initialize the cognitive service."""
        self._working_memory: Optional[RedisWorkingMemory] = None
        self._reflector_manager = ReflectorTaskManager()

    async def get_working_memory(self) -> RedisWorkingMemory:
        """Get or create working memory instance."""
        if self._working_memory is None:
            self._working_memory = get_working_memory()
        return self._working_memory

    async def process_query(self, input: CognitiveQueryInput) -> CognitiveQueryOutput:
        """
        Execute a complete cognitive cycle for a user query.

        This implements the 4-phase cognitive workflow:
        1. Summarizer - Context compression and entity extraction
        2. Investigator - Multi-hop retrieval across memory types
        3. Agent - Synthesis and response generation
        4. Reflector - Learning and memory updates (async)

        Args:
            input: Cognitive query input parameters

        Returns:
            Cognitive query output with response and metadata
        """
        import uuid

        start_time = time.time()
        phases_completed = []

        # Generate IDs
        session_id = input.session_id or str(uuid.uuid4())
        cycle_id = str(uuid.uuid4())

        try:
            # Get working memory
            working_memory = await self.get_working_memory()

            # Create or retrieve session
            if not input.session_id:
                await working_memory.create_session(
                    user_id=input.user_id,
                    initial_context={"brand": input.brand, "region": input.region},
                )

            # Add user message to session
            await working_memory.add_message(
                session_id=session_id,
                role="user",
                content=input.query,
                metadata={"cycle_id": cycle_id},
            )

            # === PHASE 1: SUMMARIZER ===
            phase1_result = await self._run_summarizer(
                query=input.query, session_id=session_id, brand=input.brand, region=input.region
            )
            phases_completed.append("summarizer")

            # === PHASE 2: INVESTIGATOR ===
            phase2_result = await self._run_investigator(
                query=input.query,
                query_type=phase1_result.get("query_type", "general"),
                entities=phase1_result.get("entities", {}),
                brand=input.brand,
                region=input.region,
                max_hops=input.max_hops,
            )
            phases_completed.append("investigator")

            # === PHASE 3: AGENT ===
            phase3_result = await self._run_agent(
                query=input.query,
                query_type=phase1_result.get("query_type", "general"),
                evidence=phase2_result.get("evidence", []),
            )
            phases_completed.append("agent")

            # Add assistant response to session
            await working_memory.add_message(
                session_id=session_id,
                role="assistant",
                content=phase3_result.get("response", ""),
                metadata={
                    "cycle_id": cycle_id,
                    "agent_name": phase3_result.get("agent_used", "orchestrator"),
                    "confidence": phase3_result.get("confidence", 0.5),
                },
            )

            # Store evidence in session
            if input.include_evidence and phase2_result.get("evidence"):
                await working_memory.append_evidence(
                    session_id=session_id, evidence=phase2_result["evidence"]
                )

            phases_completed.append("agent_complete")

            # === PHASE 4: REFLECTOR (async with tracking) ===
            # Run reflector in background with timeout and error tracking
            await self._reflector_manager.submit(
                self._run_reflector(
                    session_id=session_id,
                    cycle_id=cycle_id,
                    query=input.query,
                    query_type=phase1_result.get("query_type", "general"),
                    response=phase3_result.get("response", ""),
                    confidence=phase3_result.get("confidence", 0.5),
                    evidence=phase2_result.get("evidence", []),
                    agent_used=phase3_result.get("agent_used", "orchestrator"),
                ),
                cycle_id=cycle_id,
            )
            phases_completed.append("reflector_started")

            processing_time = (time.time() - start_time) * 1000

            return CognitiveQueryOutput(
                session_id=session_id,
                cycle_id=cycle_id,
                query=input.query,
                query_type=phase1_result.get("query_type", "general"),
                agent_used=phase3_result.get("agent_used", "orchestrator"),
                response=phase3_result.get("response", ""),
                confidence=phase3_result.get("confidence", 0.5),
                evidence=phase2_result.get("evidence") if input.include_evidence else None,
                visualization_config=phase3_result.get("visualization_config"),
                phases_completed=phases_completed,
                processing_time_ms=processing_time,
                worth_remembering=phase3_result.get("confidence", 0.5) > 0.6,
            )

        except Exception as e:
            logger.error(f"Cognitive cycle failed: {e}", exc_info=True)
            processing_time = (time.time() - start_time) * 1000

            return CognitiveQueryOutput(
                session_id=session_id,
                cycle_id=cycle_id,
                query=input.query,
                query_type="error",
                agent_used="error_handler",
                response=f"I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                evidence=None,
                visualization_config=None,
                phases_completed=phases_completed,
                processing_time_ms=processing_time,
                worth_remembering=False,
            )

    # =========================================================================
    # PHASE IMPLEMENTATIONS
    # =========================================================================

    async def _run_summarizer(
        self, query: str, session_id: str, brand: Optional[str], region: Optional[str]
    ) -> Dict[str, Any]:
        """
        Phase 1: Summarizer

        - Detect query type (causal, prediction, optimization, etc.)
        - Extract entities (brands, regions, KPIs)
        - Prepare context for investigation
        """
        query_lower = query.lower()

        # Detect query type based on keywords
        query_type = "general"
        if any(kw in query_lower for kw in ["why", "cause", "impact", "effect", "because"]):
            query_type = "causal"
        elif any(kw in query_lower for kw in ["predict", "forecast", "will", "expect", "next"]):
            query_type = "prediction"
        elif any(
            kw in query_lower for kw in ["optimize", "improve", "best", "maximize", "allocate"]
        ):
            query_type = "optimization"
        elif any(kw in query_lower for kw in ["compare", "versus", "vs", "difference", "gap"]):
            query_type = "comparison"
        elif any(kw in query_lower for kw in ["trend", "change", "over time", "drift"]):
            query_type = "monitoring"

        # Extract entities
        entities: Dict[str, List[str]] = {"brands": [], "regions": [], "kpis": []}

        # Brand detection
        brand_keywords = {
            "kisqali": "Kisqali",
            "fabhalta": "Fabhalta",
            "remibrutinib": "Remibrutinib",
        }
        for keyword, brand_name in brand_keywords.items():
            if keyword in query_lower:
                entities["brands"].append(brand_name)

        # Add explicit brand/region if provided
        if brand and brand not in entities["brands"]:
            entities["brands"].append(brand)
        if region:
            entities["regions"].append(region)

        # Region detection
        region_keywords = ["northeast", "southeast", "midwest", "west", "north", "south"]
        for region_kw in region_keywords:
            if region_kw in query_lower:
                entities["regions"].append(region_kw)

        # KPI detection
        kpi_keywords = [
            "trx",
            "nrx",
            "scripts",
            "adoption",
            "conversion",
            "revenue",
            "market share",
        ]
        for kpi in kpi_keywords:
            if kpi in query_lower:
                entities["kpis"].append(kpi.upper())

        return {"query_type": query_type, "entities": entities, "context_ready": True}

    async def _run_investigator(
        self,
        query: str,
        query_type: str,
        entities: Dict[str, List[str]],
        brand: Optional[str],
        region: Optional[str],
        max_hops: int,
    ) -> Dict[str, Any]:
        """
        Phase 2: Investigator

        - Execute hybrid search across memory types
        - Build evidence trail
        - Find relevant patterns and historical context
        """
        # Build filters
        filters = {}
        if brand:
            filters["brand"] = brand
        if region:
            filters["region"] = region

        # Determine KPI for graph traversal
        kpi_name = entities.get("kpis", [None])[0] if entities.get("kpis") else None

        try:
            # Execute hybrid search
            results = await hybrid_search(
                query=query,
                k=10 * max_hops,  # More results for multi-hop
                entities=entities.get("brands", []) + entities.get("regions", []),
                kpi_name=kpi_name,
                filters=filters if filters else None,
            )

            # Convert to evidence format
            evidence = []
            for i, result in enumerate(results):
                evidence.append(
                    {
                        "hop": min(i // 3 + 1, max_hops),  # Distribute across hops
                        "source": result.source,
                        "content": result.content,
                        "score": result.score,
                        "method": result.retrieval_method,
                        "metadata": result.metadata,
                    }
                )

            return {
                "evidence": evidence,
                "hops_completed": min(len(evidence) // 3, max_hops),
                "total_results": len(evidence),
            }

        except Exception as e:
            logger.warning(f"Investigation phase failed: {e}")
            return {"evidence": [], "hops_completed": 0, "total_results": 0, "error": str(e)}

    async def _run_agent(
        self, query: str, query_type: str, evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Phase 3: Agent

        - Route to appropriate agent based on query type
        - Synthesize response from evidence
        - Calculate confidence score
        """
        # Map query type to agent
        agent_mapping = {
            "causal": "causal_impact",
            "prediction": "prediction_synthesizer",
            "optimization": "resource_optimizer",
            "comparison": "gap_analyzer",
            "monitoring": "drift_monitor",
            "general": "orchestrator",
        }
        agent_used = agent_mapping.get(query_type, "orchestrator")

        # Build response based on evidence
        if not evidence:
            response = (
                f"I don't have enough historical data to fully answer your question about {query_type} analysis. "
                "However, I can help you explore this topic further. "
                "Would you like me to set up monitoring for this metric or run a causal analysis?"
            )
            confidence = 0.3
        else:
            # Summarize top evidence
            top_evidence = sorted(evidence, key=lambda x: x.get("score", 0), reverse=True)[:5]
            evidence_summary = "\n".join(
                [
                    f"- [{e.get('source', 'unknown')}] {e.get('content', '')[:200]}"
                    for e in top_evidence
                ]
            )

            # Build response based on query type
            if query_type == "causal":
                response = (
                    f"Based on my analysis of causal relationships, I found the following insights:\n\n"
                    f"{evidence_summary}\n\n"
                    "These patterns suggest interconnected causal factors. "
                    "I can drill deeper into any specific relationship you're interested in."
                )
            elif query_type == "prediction":
                response = (
                    f"Based on historical patterns and trends:\n\n"
                    f"{evidence_summary}\n\n"
                    "These trends can inform forecasting models. "
                    "Would you like me to generate specific predictions?"
                )
            elif query_type == "optimization":
                response = (
                    f"Here are optimization opportunities I've identified:\n\n"
                    f"{evidence_summary}\n\n"
                    "I can help you prioritize these based on expected impact."
                )
            else:
                response = (
                    f"Here's what I found relevant to your query:\n\n"
                    f"{evidence_summary}\n\n"
                    "Let me know if you'd like to explore any aspect further."
                )

            # Calculate confidence from evidence scores
            avg_score = sum(e.get("score", 0) for e in top_evidence) / len(top_evidence)
            confidence = min(avg_score, 0.95)  # Cap at 95%

        # Visualization config based on query type
        viz_config = None
        if query_type in ["causal", "comparison"]:
            viz_config = {
                "chart_type": "sankey" if query_type == "causal" else "bar",
                "show_confidence": True,
            }
        elif query_type in ["prediction", "monitoring"]:
            viz_config = {"chart_type": "line", "show_trend": True}

        return {
            "agent_used": agent_used,
            "response": response,
            "confidence": confidence,
            "visualization_config": viz_config,
        }

    async def _run_reflector(
        self,
        session_id: str,
        cycle_id: str,
        query: str,
        query_type: str,
        response: str,
        confidence: float,
        evidence: List[Dict[str, Any]],
        agent_used: str,
    ) -> None:
        """
        Phase 4: Reflector (runs async after response)

        - Evaluate if interaction is worth remembering
        - Store episodic memory for significant interactions
        - Store to Graphiti knowledge graph for entity/relationship extraction
        - Record learning signals for DSPy optimization
        - Update procedural patterns
        """
        try:
            # Only remember high-value interactions
            # Threshold aligned with config: min_confidence_for_learning: 0.6
            if confidence < 0.6:
                logger.debug(f"Skipping reflection for low-confidence cycle: {cycle_id}")
                return

            # Store episodic memory
            memory_input = EpisodicMemoryInput(
                event_type="user_query",
                description=f"User asked: {query[:200]}",
                agent_name=agent_used,
                e2i_refs=E2IEntityReferences(),
                raw_content={
                    "query": query,
                    "query_type": query_type,
                    "response_preview": response[:500],
                    "confidence": confidence,
                    "evidence_count": len(evidence),
                },
            )

            await insert_episodic_memory_with_text(
                memory=memory_input, text_to_embed=query, session_id=session_id, cycle_id=cycle_id
            )

            # Store to Graphiti knowledge graph for entity extraction
            await self._store_to_graphiti(
                session_id=session_id,
                cycle_id=cycle_id,
                query=query,
                query_type=query_type,
                response=response,
                confidence=confidence,
                agent_used=agent_used,
            )

            # Record learning signal
            signal = LearningSignalInput(
                signal_type="outcome_success" if confidence > 0.7 else "outcome_partial",
                signal_value=confidence,
                applies_to_type="query",
                applies_to_id=cycle_id,
                rated_agent=agent_used,
                signal_details={
                    "query_type": query_type,
                    "evidence_count": len(evidence),
                    "is_training_example": confidence > 0.8,
                },
            )

            await record_learning_signal(signal=signal, session_id=session_id, cycle_id=cycle_id)

            logger.info(f"Reflection complete for cycle {cycle_id}")

        except Exception as e:
            logger.error(f"Reflector phase failed: {e}", exc_info=True)

    async def _store_to_graphiti(
        self,
        session_id: str,
        cycle_id: str,
        query: str,
        query_type: str,
        response: str,
        confidence: float,
        agent_used: str,
    ) -> None:
        """
        Store interaction to Graphiti knowledge graph.

        Graphiti automatically:
        - Extracts entities (HCP, Brand, Patient, KPI, etc.)
        - Discovers relationships between entities
        - Links to existing knowledge graph nodes
        - Tracks temporal validity
        """
        try:
            graphiti = await get_graphiti_service()

            # Build content for episode - combine query and response
            episode_content = (
                f"User Query ({query_type}): {query}\n\n"
                f"Agent Response ({agent_used}): {response[:1000]}"
            )

            # Add episode to Graphiti
            result = await graphiti.add_episode(
                content=episode_content,
                source=agent_used,
                session_id=session_id,
                metadata={
                    "cycle_id": cycle_id,
                    "query_type": query_type,
                    "confidence": confidence,
                    "timestamp": str(datetime.now()),
                },
            )

            logger.info(
                f"Graphiti episode stored: {result.episode_id} "
                f"(entities: {len(result.entities_extracted)}, "
                f"relationships: {len(result.relationships_extracted)})"
            )

        except Exception as e:
            # Don't fail the reflector if Graphiti fails
            logger.warning(f"Failed to store to Graphiti knowledge graph: {e}")

    # =========================================================================
    # SERVICE MANAGEMENT
    # =========================================================================

    async def shutdown(self, timeout: float = 60.0) -> Dict[str, Any]:
        """
        Gracefully shutdown the cognitive service.

        Waits for pending reflector tasks to complete before shutdown.
        This ensures learning signals are persisted.

        Args:
            timeout: Maximum time to wait for pending tasks

        Returns:
            Dict with shutdown statistics
        """
        logger.info("Initiating cognitive service shutdown...")

        reflector_result = await self._reflector_manager.wait_all(timeout=timeout)

        return {
            "reflector_tasks": reflector_result,
            "final_stats": self._reflector_manager.stats,
        }

    @property
    def reflector_stats(self) -> Dict[str, Any]:
        """
        Get reflector task statistics for monitoring.

        Returns:
            Dict with pending, succeeded, failed, and timeout counts
        """
        return cast(Dict[str, Any], self._reflector_manager.stats)


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_cognitive_service: Optional[CognitiveService] = None


def get_cognitive_service() -> CognitiveService:
    """Get or create the cognitive service singleton."""
    global _cognitive_service
    if _cognitive_service is None:
        _cognitive_service = CognitiveService()
    return _cognitive_service


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def process_cognitive_query(
    query: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    brand: Optional[str] = None,
    region: Optional[str] = None,
    include_evidence: bool = True,
) -> CognitiveQueryOutput:
    """
    Process a cognitive query with default settings.

    This is the main entry point for the cognitive workflow.
    """
    service = get_cognitive_service()

    input_data = CognitiveQueryInput(
        query=query,
        session_id=session_id,
        user_id=user_id,
        brand=brand,
        region=region,
        include_evidence=include_evidence,
        max_hops=3,
    )

    return await service.process_query(input_data)
