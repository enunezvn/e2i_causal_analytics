"""
Opik Integration for CopilotKit Chatbot.

This module provides Opik tracing utilities for the chatbot LangGraph workflow,
enabling observability of:
- Full workflow traces (init → load_context → classify_intent → retrieve_rag → generate → finalize)
- Per-node spans with timing and metrics
- LLM call tracking (tokens, latency, model)
- RAG retrieval metrics
- Intent classification (with DSPy confidence once integrated)

Usage:
    from src.api.routes.chatbot_tracer import (
        ChatbotOpikTracer,
        get_chatbot_tracer,
    )

    tracer = get_chatbot_tracer()

    # Trace a full chatbot workflow
    async with tracer.trace_workflow(
        query="What is Kisqali market share?",
        session_id="session_123",
    ) as trace:
        async with trace.trace_node("init") as node:
            # ... init logic
            node.log_init(is_new_conversation=True)
        async with trace.trace_node("generate") as node:
            response = await llm.ainvoke(messages)
            node.log_generate(input_tokens=150, output_tokens=300, model="claude-sonnet-4-20250514")
        trace.log_workflow_complete(status="success", total_tokens=450)

Author: E2I Causal Analytics Team
Version: 4.3.0
"""

import logging
import os
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

from uuid_utils import uuid7 as uuid7_func

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Feature flag for enabling/disabling tracing
CHATBOT_OPIK_TRACING_ENABLED = os.getenv("CHATBOT_OPIK_TRACING", "true").lower() == "true"


@dataclass
class NodeSpanContext:
    """Context for a chatbot workflow node span.

    Provides methods to log node-specific events and metrics.

    Attributes:
        trace_id: Parent trace identifier
        span_id: This span's identifier
        node_name: Name of the node (init, load_context, classify_intent, etc.)
        start_time: When the node started
        end_time: When the node ended (set on exit)
        duration_ms: Node duration in milliseconds
        metadata: Additional node metadata
        _opik_span: Reference to the Opik span context
    """

    trace_id: str
    span_id: str
    node_name: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _opik_span: Optional[Any] = None
    _parent_ctx: Optional["ChatbotTraceContext"] = None

    def log_init(
        self,
        is_new_conversation: bool,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log init node metrics.

        Args:
            is_new_conversation: Whether this is a new conversation
            session_id: Session identifier
            user_id: User identifier
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "is_new_conversation": is_new_conversation,
                "session_id": session_id,
                "user_id": user_id,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("is_new_conversation", is_new_conversation)
            if session_id:
                self._opik_span.set_attribute("session_id", session_id)
            self._opik_span.add_event(
                "init_complete",
                {
                    "is_new_conversation": is_new_conversation,
                },
            )

        logger.debug(f"[INIT] new_conversation={is_new_conversation}, session={session_id}")

    def log_context_load(
        self,
        previous_message_count: int,
        conversation_title: Optional[str] = None,
        brand_context: Optional[str] = None,
        region_context: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log load_context node metrics.

        Args:
            previous_message_count: Number of previous messages loaded
            conversation_title: Title of the conversation
            brand_context: Brand filter applied
            region_context: Region filter applied
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "previous_message_count": previous_message_count,
                "conversation_title": conversation_title,
                "brand_context": brand_context,
                "region_context": region_context,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("previous_message_count", previous_message_count)
            if brand_context:
                self._opik_span.set_attribute("brand_context", brand_context)
            if region_context:
                self._opik_span.set_attribute("region_context", region_context)
            self._opik_span.add_event(
                "context_loaded",
                {
                    "previous_message_count": previous_message_count,
                    "has_title": conversation_title is not None,
                },
            )

        logger.debug(
            f"[LOAD_CONTEXT] {previous_message_count} messages, "
            f"brand={brand_context}, region={region_context}"
        )

    def log_intent_classification(
        self,
        intent: str,
        confidence: float = 1.0,
        classification_method: str = "hardcoded",
        **kwargs: Any,
    ) -> None:
        """Log classify_intent node metrics.

        Args:
            intent: Classified intent type
            confidence: Classification confidence (0.0-1.0)
            classification_method: Method used (hardcoded, dspy)
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "intent": intent,
                "confidence": confidence,
                "classification_method": classification_method,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("intent", intent)
            self._opik_span.set_attribute("confidence", confidence)
            self._opik_span.set_attribute("classification_method", classification_method)
            self._opik_span.add_event(
                "intent_classified",
                {
                    "intent": intent,
                    "confidence": confidence,
                    "method": classification_method,
                },
            )

        logger.debug(
            f"[CLASSIFY_INTENT] intent={intent}, confidence={confidence:.2f}, "
            f"method={classification_method}"
        )

    def log_rag_retrieval(
        self,
        result_count: int,
        relevance_scores: Optional[List[float]] = None,
        kpi_filter: Optional[str] = None,
        brand_filter: Optional[str] = None,
        retrieval_method: str = "hybrid",
        **kwargs: Any,
    ) -> None:
        """Log retrieve_rag node metrics.

        Args:
            result_count: Number of RAG results retrieved
            relevance_scores: Score for each result
            kpi_filter: KPI filter applied
            brand_filter: Brand filter applied
            retrieval_method: Retrieval method (hybrid, semantic, sparse)
            **kwargs: Additional metrics
        """
        avg_score = 0.0
        if relevance_scores and len(relevance_scores) > 0:
            avg_score = sum(relevance_scores) / len(relevance_scores)

        self.metadata.update(
            {
                "result_count": result_count,
                "avg_relevance_score": avg_score,
                "relevance_scores": relevance_scores or [],
                "kpi_filter": kpi_filter,
                "brand_filter": brand_filter,
                "retrieval_method": retrieval_method,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("result_count", result_count)
            self._opik_span.set_attribute("avg_relevance_score", avg_score)
            self._opik_span.set_attribute("retrieval_method", retrieval_method)
            if kpi_filter:
                self._opik_span.set_attribute("kpi_filter", kpi_filter)
            if brand_filter:
                self._opik_span.set_attribute("brand_filter", brand_filter)
            self._opik_span.add_event(
                "rag_retrieved",
                {
                    "result_count": result_count,
                    "avg_score": avg_score,
                },
            )

        logger.debug(
            f"[RETRIEVE_RAG] {result_count} results, avg_score={avg_score:.3f}, "
            f"method={retrieval_method}"
        )

    def log_generate(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: Optional[str] = None,
        provider: str = "anthropic",
        tool_calls_count: int = 0,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> None:
        """Log generate node metrics.

        Args:
            input_tokens: Input token count
            output_tokens: Output token count
            model: Model identifier
            provider: LLM provider (anthropic, openai)
            tool_calls_count: Number of tool calls in response
            temperature: Temperature used
            **kwargs: Additional metrics
        """
        total_tokens = input_tokens + output_tokens

        self.metadata.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model": model,
                "provider": provider,
                "tool_calls_count": tool_calls_count,
                "temperature": temperature,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("input_tokens", input_tokens)
            self._opik_span.set_attribute("output_tokens", output_tokens)
            self._opik_span.set_attribute("total_tokens", total_tokens)
            if model:
                self._opik_span.set_attribute("model", model)
            self._opik_span.set_attribute("provider", provider)
            self._opik_span.set_attribute("tool_calls_count", tool_calls_count)
            self._opik_span.add_event(
                "generation_complete",
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "tool_calls": tool_calls_count,
                },
            )

        logger.debug(
            f"[GENERATE] tokens={total_tokens} (in={input_tokens}, out={output_tokens}), "
            f"model={model}, tools={tool_calls_count}"
        )

    def log_tool_execution(
        self,
        tool_name: str,
        success: bool,
        result_size: int = 0,
        error: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log tool execution metrics.

        Args:
            tool_name: Name of the tool executed
            success: Whether execution succeeded
            result_size: Size of the result
            error: Error message if failed
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "tool_name": tool_name,
                "tool_success": success,
                "result_size": result_size,
                "tool_error": error,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("tool_name", tool_name)
            self._opik_span.set_attribute("tool_success", success)
            self._opik_span.add_event(
                "tool_executed",
                {
                    "tool_name": tool_name,
                    "success": success,
                    "result_size": result_size,
                },
            )

        logger.debug(f"[TOOL] {tool_name}: success={success}, result_size={result_size}")

    def log_finalize(
        self,
        response_length: int,
        messages_persisted: bool,
        episodic_memory_saved: bool = False,
        significance_score: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Log finalize node metrics.

        Args:
            response_length: Length of final response
            messages_persisted: Whether messages were persisted to DB
            episodic_memory_saved: Whether saved to episodic memory
            significance_score: Significance score for episodic memory
            **kwargs: Additional metrics
        """
        self.metadata.update(
            {
                "response_length": response_length,
                "messages_persisted": messages_persisted,
                "episodic_memory_saved": episodic_memory_saved,
                "significance_score": significance_score,
                **kwargs,
            }
        )

        if self._opik_span:
            self._opik_span.set_attribute("response_length", response_length)
            self._opik_span.set_attribute("messages_persisted", messages_persisted)
            self._opik_span.set_attribute("episodic_memory_saved", episodic_memory_saved)
            self._opik_span.set_attribute("significance_score", significance_score)
            self._opik_span.add_event(
                "finalize_complete",
                {
                    "response_length": response_length,
                    "persisted": messages_persisted,
                    "episodic_saved": episodic_memory_saved,
                },
            )

        logger.debug(
            f"[FINALIZE] response_length={response_length}, "
            f"persisted={messages_persisted}, episodic={episodic_memory_saved}"
        )

    def set_output(self, output: Dict[str, Any]) -> None:
        """Set the output data for this node span."""
        if self._opik_span:
            self._opik_span.set_output(output)

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        """Log arbitrary metadata for this node span.

        Useful for cognitive RAG and other custom metrics.

        Args:
            metadata: Key-value pairs to log
        """
        self.metadata.update(metadata)

        if self._opik_span:
            for key, value in metadata.items():
                if value is not None:
                    self._opik_span.set_attribute(key, value)
            self._opik_span.add_event(
                "metadata_logged",
                {k: v for k, v in metadata.items() if v is not None},
            )

        logger.debug(f"[{self.node_name.upper()}] metadata: {list(metadata.keys())}")


@dataclass
class ChatbotTraceContext:
    """Context for a full chatbot workflow trace.

    Provides methods to create node spans and log overall metrics.

    Attributes:
        trace_id: Unique trace identifier
        span_id: Root span identifier
        query: User query being processed
        session_id: Chat session identifier
        start_time: When workflow started
        end_time: When workflow ended
        node_spans: Child spans for each node
        metadata: Additional trace metadata
        _opik_span: Reference to the Opik span
        _tracer: Reference to parent tracer
    """

    trace_id: str
    span_id: str
    query: str
    session_id: Optional[str] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    node_spans: Dict[str, NodeSpanContext] = field(default_factory=dict)
    node_durations: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _opik_span: Optional[Any] = None
    _tracer: Optional["ChatbotOpikTracer"] = None

    @asynccontextmanager
    async def trace_node(
        self,
        node_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing a specific node.

        Args:
            node_name: Name of the node (init, load_context, classify_intent, etc.)
            metadata: Additional node metadata

        Yields:
            NodeSpanContext for logging node events
        """
        span_id = str(uuid7_func())
        start_time = datetime.now(timezone.utc)

        node_ctx = NodeSpanContext(
            trace_id=self.trace_id,
            span_id=span_id,
            node_name=node_name,
            start_time=start_time,
            metadata=metadata or {},
            _parent_ctx=self,
        )

        try:
            # Create child span in Opik if parent is traced
            if self._opik_span and self._tracer and self._tracer.enabled:
                try:
                    from src.mlops.opik_connector import get_opik_connector

                    connector = get_opik_connector()
                    if connector.is_enabled:
                        async with connector.trace_agent(
                            agent_name="chatbot",
                            operation=node_name,
                            trace_id=self.trace_id,
                            parent_span_id=self.span_id,
                            metadata={
                                "node": node_name,
                                "node_index": self._get_node_index(node_name),
                                **(metadata or {}),
                            },
                            tags=["chatbot", node_name],
                        ) as span:
                            node_ctx._opik_span = span
                            yield node_ctx
                            # Set output on completion
                            span.set_output(node_ctx.metadata)
                        return
                except Exception as e:
                    logger.debug(f"Failed to create node span: {e}")

            # Fall through to non-traced version
            yield node_ctx

        finally:
            # Record duration
            end_time = datetime.now(timezone.utc)
            node_ctx.end_time = end_time
            node_ctx.duration_ms = (end_time - start_time).total_seconds() * 1000
            self.node_durations[node_name] = int(node_ctx.duration_ms)

            # Store in parent context
            self.node_spans[node_name] = node_ctx

            logger.debug(f"Node {node_name} completed in {node_ctx.duration_ms:.2f}ms")

    def _get_node_index(self, node_name: str) -> int:
        """Get numeric index for node ordering."""
        node_order = [
            "init",
            "load_context",
            "classify_intent",
            "retrieve_rag",
            "generate",
            "tools",
            "finalize",
        ]
        return node_order.index(node_name) if node_name in node_order else -1

    def log_workflow_complete(
        self,
        status: str,
        success: bool,
        total_duration_ms: Optional[int] = None,
        intent: Optional[str] = None,
        total_tokens: int = 0,
        tool_calls_count: int = 0,
        rag_result_count: int = 0,
        response_length: int = 0,
        errors: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Log completion of the full workflow.

        Args:
            status: Final status (success, partial, failed)
            success: Whether workflow succeeded
            total_duration_ms: Total duration in milliseconds
            intent: Classified intent
            total_tokens: Total tokens used
            tool_calls_count: Number of tool calls
            rag_result_count: Number of RAG results
            response_length: Length of response
            errors: Any errors encountered
            **kwargs: Additional metrics
        """
        if total_duration_ms is None:
            total_duration_ms = int(self.duration_ms or 0)

        output_data = {
            "status": status,
            "success": success,
            "total_duration_ms": total_duration_ms,
            "intent": intent,
            "total_tokens": total_tokens,
            "tool_calls_count": tool_calls_count,
            "rag_result_count": rag_result_count,
            "response_length": response_length,
            "node_durations": self.node_durations,
            "errors": errors or [],
            **kwargs,
        }

        if self._opik_span:
            # Set key attributes for filtering
            self._opik_span.set_attribute("status", status)
            self._opik_span.set_attribute("success", success)
            self._opik_span.set_attribute("total_duration_ms", total_duration_ms)
            if intent:
                self._opik_span.set_attribute("intent", intent)
            self._opik_span.set_attribute("total_tokens", total_tokens)
            self._opik_span.set_attribute("tool_calls_count", tool_calls_count)
            self._opik_span.set_attribute("response_length", response_length)

            # Set output data
            self._opik_span.set_output(output_data)

        logger.info(
            f"Chatbot workflow complete: status={status}, "
            f"intent={intent}, tokens={total_tokens}, "
            f"tools={tool_calls_count}, duration={total_duration_ms}ms"
        )


class ChatbotOpikTracer:
    """Opik tracer for CopilotKit chatbot LangGraph workflow.

    Provides observability into the chatbot pipeline with:
    - Root trace for full workflow
    - Child spans for each node
    - LLM call metrics (tokens, latency, model)
    - RAG retrieval metrics
    - Intent classification tracking

    Uses the shared OpikConnector for circuit breaker protection.

    Example:
        >>> tracer = ChatbotOpikTracer()
        >>> async with tracer.trace_workflow(query="...", session_id="...") as trace:
        ...     async with trace.trace_node("init") as node:
        ...         # ... init logic
        ...         node.log_init(is_new_conversation=True)
        ...     # ... other nodes
        ...     trace.log_workflow_complete(status="success", ...)
    """

    def __init__(
        self,
        project_name: str = "e2i-chatbot",
        enabled: bool = True,
        sample_rate: float = 1.0,
    ):
        """Initialize the Chatbot tracer.

        Args:
            project_name: Opik project name
            enabled: Whether tracing is enabled
            sample_rate: Sample rate (1.0 = trace all, 0.1 = 10%)
        """
        self.project_name = project_name
        # Respect both constructor param and feature flag
        self.enabled = enabled and CHATBOT_OPIK_TRACING_ENABLED
        self.sample_rate = sample_rate
        self._opik_connector = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of OpikConnector."""
        if self._initialized:
            return

        try:
            from src.mlops.opik_connector import get_opik_connector

            self._opik_connector = get_opik_connector()
            self._initialized = True
            logger.debug("ChatbotOpikTracer initialized")
        except ImportError:
            logger.warning("OpikConnector not available, tracing disabled")
            self._opik_connector = None
            self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize OpikConnector: {e}")
            self._opik_connector = None
            self._initialized = True

    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled and available."""
        self._ensure_initialized()
        return self.enabled and self._opik_connector is not None and self._opik_connector.is_enabled

    def _should_trace(self) -> bool:
        """Determine if this workflow should be traced."""
        return random.random() < self.sample_rate

    @asynccontextmanager
    async def trace_workflow(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        brand_context: Optional[str] = None,
        region_context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing a full chatbot workflow.

        Creates an Opik trace for the workflow and provides a context
        object for creating node spans and logging metrics.

        Args:
            query: The user query being processed
            session_id: Chat session identifier
            user_id: User identifier
            brand_context: Brand filter
            region_context: Region filter
            metadata: Additional trace metadata

        Yields:
            ChatbotTraceContext for node tracing and metric logging

        Example:
            async with tracer.trace_workflow(query, session_id) as trace:
                async with trace.trace_node("init") as node:
                    # ... init logic
                    node.log_init(...)
        """
        self._ensure_initialized()

        trace_id = str(uuid7_func())
        span_id = str(uuid7_func())
        start_time = datetime.now(timezone.utc)

        # Build metadata
        trace_metadata = {
            "query_length": len(query),
            "session_id": session_id,
            "user_id": user_id,
            "brand_context": brand_context,
            "region_context": region_context,
            **(metadata or {}),
        }

        # Create trace context
        trace_ctx = ChatbotTraceContext(
            trace_id=trace_id,
            span_id=span_id,
            query=query,
            session_id=session_id,
            start_time=start_time,
            metadata=trace_metadata,
            _tracer=self,
        )

        try:
            # Create Opik trace if enabled and sampled
            logger.debug(
                f"Chatbot tracing check: is_enabled={self.is_enabled}, "
                f"should_trace={self._should_trace()}, "
                f"connector={self._opik_connector is not None}"
            )
            if self.is_enabled and self._should_trace():
                try:
                    logger.info(f"Creating chatbot trace with id={trace_id}")
                    async with (
                        self._opik_connector.trace_agent(
                            agent_name="chatbot",
                            operation="workflow",
                            trace_id=trace_id,
                            metadata={
                                "pipeline": "init→load_context→classify_intent→retrieve_rag→generate→finalize",
                                "tier": 1,
                                **trace_metadata,
                            },
                            tags=["chatbot", "workflow", "copilotkit"],
                            input_data={
                                "query": query[:500],  # Truncate for Opik
                                "session_id": session_id,
                                "brand": brand_context,
                                "region": region_context,
                            },
                            force_new_trace=True,  # We're creating a new workflow trace with our own trace_id
                        ) as span
                    ):
                        trace_ctx._opik_span = span
                        logger.info(
                            f"Chatbot trace created: span_id={span.span_id[:12] if hasattr(span, 'span_id') else 'N/A'}..."
                        )
                        yield trace_ctx
                        return
                except Exception as e:
                    logger.warning(f"Opik tracing failed, continuing without: {e}")

            # Fall through to non-traced version
            yield trace_ctx

        except Exception as e:
            {"type": type(e).__name__, "message": str(e)}
            raise

        finally:
            # Record final timing
            end_time = datetime.now(timezone.utc)
            trace_ctx.end_time = end_time
            trace_ctx.duration_ms = (end_time - start_time).total_seconds() * 1000

            logger.debug(f"Chatbot workflow trace completed in {trace_ctx.duration_ms:.2f}ms")


def trace_chatbot_workflow(
    query_param: str = "query",
    session_id_param: str = "session_id",
    project_name: str = "e2i-chatbot",
) -> Callable[[F], F]:
    """Decorator to trace a chatbot workflow function.

    Use this decorator on workflow functions for automatic tracing.
    The decorated function receives a ChatbotTraceContext as the first argument.

    Args:
        query_param: Name of the query parameter in the decorated function
        session_id_param: Name of the session_id parameter
        project_name: Opik project name

    Returns:
        Decorated function

    Example:
        >>> @trace_chatbot_workflow()
        ... async def run_chatbot(trace: ChatbotTraceContext, query: str, session_id=None):
        ...     async with trace.trace_node("init") as node:
        ...         # ... init
        ...     # ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract query from kwargs or args
            query = kwargs.get(query_param, args[0] if args else "")
            session_id = kwargs.get(session_id_param)

            tracer = ChatbotOpikTracer(project_name=project_name)

            async with tracer.trace_workflow(
                query=query,
                session_id=session_id,
            ) as trace_ctx:
                # Pass context as first argument
                return await func(trace_ctx, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# Singleton tracer instance
_tracer_instance: Optional[ChatbotOpikTracer] = None


def get_chatbot_tracer(
    project_name: str = "e2i-chatbot",
    enabled: bool = True,
    sample_rate: float = 1.0,
) -> ChatbotOpikTracer:
    """Get the Chatbot Opik tracer singleton.

    Args:
        project_name: Opik project name
        enabled: Whether tracing is enabled
        sample_rate: Sample rate for tracing

    Returns:
        ChatbotOpikTracer instance
    """
    global _tracer_instance

    if _tracer_instance is None:
        _tracer_instance = ChatbotOpikTracer(
            project_name=project_name,
            enabled=enabled,
            sample_rate=sample_rate,
        )

    return _tracer_instance


def reset_chatbot_tracer() -> None:
    """Reset the tracer singleton (for testing)."""
    global _tracer_instance
    _tracer_instance = None


__all__ = [
    "NodeSpanContext",
    "ChatbotTraceContext",
    "ChatbotOpikTracer",
    "trace_chatbot_workflow",
    "get_chatbot_tracer",
    "reset_chatbot_tracer",
    "CHATBOT_OPIK_TRACING_ENABLED",
]
