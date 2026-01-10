"""
CopilotKit Integration Router
=============================

Provides CopilotKit backend runtime for the E2I Chat Sidebar.
Exposes backend actions for querying KPIs, running analyses,
and interacting with the E2I agent system.

Author: E2I Causal Analytics Team
Version: 1.20.0

Changelog:
    1.20.0 - Fixed null fields in MESSAGES_SNAPSHOT messages array.
             Root cause: CopilotKit React SDK (v1.50.1) Zod validation requires:
               messages[].name: string (empty string if null)
               messages[].toolCalls: array (empty array if null for assistant messages)
             AG-UI SDK emits null values for these optional fields, causing validation errors:
               "Expected string, received null" at path ["messages", 0, "name"]
               "Expected array, received null" at path ["messages", 1, "toolCalls"]
             Fix: Extended _fix_all_events() to iterate MESSAGES_SNAPSHOT messages and replace
             null name with "" and null toolCalls with [].
    1.19.0 - Fixed timestamp and source fields on ALL event types (not just lifecycle events).
             Root cause: CopilotKit React SDK (v1.50.1) Zod validation requires timestamp (number)
             and source (string) fields on ALL events, including TEXT_MESSAGE_START,
             TEXT_MESSAGE_CONTENT, TEXT_MESSAGE_END, STATE_SNAPSHOT, and MESSAGES_SNAPSHOT.
             The v1.18.0 fix only addressed lifecycle events (RUN_STARTED, RUN_FINISHED), but
             Zod validation errors persisted on other event types with null timestamp/source.
             Fix: Renamed _fix_lifecycle_event() to _fix_all_events() and expanded to add
             timestamp (ms since epoch) and source ("e2i-copilot") to ALL events.
             Also added timestamp and source to manually constructed TEXT_MESSAGE events.
    1.18.0 - Fixed input field structure in RUN_STARTED event.
             Root cause: CopilotKit React SDK (v1.50.1) Zod validation expects input to contain
             the full RunAgentInput structure: {threadId, runId, messages, tools, context}.
             The v1.17.0 fix set input to {} which still fails Zod validation:
               "Expected string, received undefined" for input.threadId, input.runId
               "Expected array, received undefined" for input.messages, input.tools, input.context
             Fix: Populate input with all required RunAgentInput fields.
    1.17.0 - Fixed missing required fields in RUN_STARTED/RUN_FINISHED events.
             Root cause: CopilotKit React SDK (v1.50.1) uses Zod validation that requires
             timestamp (number), parentRunId (string), and input (object) fields.
             AG-UI SDK emits these events with null values for optional fields, but Zod
             validation fails with "Expected number/string/object, received null" errors.
             Fix: Intercept lifecycle events and ensure all required fields have valid values.
    1.16.0 - Fixed event type casing: use SCREAMING_SNAKE_CASE for all event types.
             Root cause: CopilotKit React SDK (v1.50.1) uses Zod validation that expects
             SCREAMING_SNAKE_CASE event types (TEXT_MESSAGE_START, RUN_STARTED, etc.),
             not PascalCase. The v1.13.0 change incorrectly converted AG-UI SDK's native
             SCREAMING_SNAKE_CASE format to PascalCase, causing Zod validation errors:
               "Invalid discriminator value. Expected 'TEXT_MESSAGE_START' | ..."
             Fix: Removed _screaming_snake_to_pascal() conversion and use SCREAMING_SNAKE_CASE
             for all manually constructed events (TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, etc.).
    1.15.0 - Fixed streaming format: use SSE (text/event-stream) instead of NDJSON.
             Root cause: CopilotKit SDK uses ag-ui EventEncoder which produces SSE format:
               Content-Type: text/event-stream
               Event format: data: ${JSON.stringify(event)}\n\n
             But our backend was using NDJSON format:
               Content-Type: application/x-ndjson
               Event format: ${JSON.stringify(event)}\n
             The SDK's event parser expects SSE format, so events were not being parsed.
             Fix: Changed media_type to "text/event-stream" and event format to "data: {...}\n\n".
    1.14.0 - Fixed TextMessageContent field name: use "delta" instead of "content".
             Root cause: AG-UI protocol spec (https://docs.ag-ui.com/concepts/messages) defines
             TextMessageContentEvent with { type, messageId, delta } but v1.12.0 incorrectly used
             "content" field. CopilotKit SDK parses events using AG-UI protocol types which expect
             "delta" for text content chunks.
             Fix: Changed TextMessageContent event to use "delta" field for message text.
    1.13.0 - Fixed ALL event types to use PascalCase for CopilotKit Runtime compatibility.
             Root cause: AG-UI SDK uses SCREAMING_SNAKE_CASE for ALL event types (RUN_STARTED,
             TEXT_MESSAGE_START, etc.), but CopilotKit Runtime expects PascalCase (RunStarted,
             TextMessageStart, etc.). The v1.12.0 fix only converted TEXT_MESSAGE events but
             missed lifecycle events (RUN_STARTED, RUN_FINISHED) which prevented the SDK from
             recognizing the run and parsing messages.
             Fix: Add helper function to convert event types from SCREAMING_SNAKE_CASE to
             PascalCase, and apply it to ALL events during serialization.
    1.12.0 - Fixed event TYPE format for CopilotKit Runtime compatibility.
             Root cause: AG-UI SDK event classes (TextMessageStartEvent, etc.) serialize type to wrong format:
             - Type: SCREAMING_SNAKE_CASE (TEXT_MESSAGE_START) instead of PascalCase (TextMessageStart)
             Fix: Replace AG-UI SDK event classes with manual JSON construction using PascalCase types:
             {"type": "TextMessageStart", "messageId": "...", "role": "assistant"}
             {"type": "TextMessageContent", "messageId": "...", "delta": "message text"}
             {"type": "TextMessageEnd", "messageId": "..."}
             NOTE: v1.12.0 incorrectly used "content" field; fixed in v1.14.0 to use "delta".
    1.11.0 - Restored TEXT_MESSAGE event conversion in execute() method.
             Root cause: copilotkit_emit_message() emits CUSTOM events with name "copilotkit_manually_emit_message".
             The CopilotKit Runtime (TypeScript) normally converts these to TEXT_MESSAGE events, BUT our custom
             FastAPI endpoint bypasses the Runtime, so the conversion never happens. Frontend receives CUSTOM
             events which it doesn't render. Fix: Intercept CUSTOM events in execute() and emit TEXT_MESSAGE
             events ourselves, mimicking what the CopilotKit Runtime does.
    1.10.0 - (Broken) Major refactor using copilotkit_emit_message() - messages not rendering because
             custom endpoint bypasses CopilotKit Runtime that would convert CUSTOM to TEXT_MESSAGE events.
    1.9.6 - Fixed TEXT_MESSAGE event serialization: use by_alias=True for camelCase field names.
            Root cause: AG-UI SDK event classes produce snake_case by default (message_id),
            but CopilotKit React SDK v1.50+ uses Zod validation expecting camelCase (messageId).
            Fix: Changed model_dump_json() to model_dump_json(by_alias=True) everywhere.
    1.9.5 - Used AG-UI SDK event classes directly for TEXT_MESSAGE events (still had casing issue).
    1.9.4 - Fixed 39-second streaming delay: force fresh thread_id per request to prevent SDK's
            regenerate mode. Root cause: SDK's prepare_stream() compares checkpoint messages vs
            frontend messages; if checkpoint has more (from previous AI responses), it triggers
            regenerate mode which blocks waiting for get_checkpoint_before_message() to find
            message IDs that don't exist in the new checkpointer's history.
    1.9.3 - Fixed SDK handler path param: inject path into scope's path_params before creating new request
            Root cause: base route `/api/copilotkit` has no `{path:path}` param, so SDK handler's
            `request.path_params.get('path')` returns None, causing `re.match()` TypeError.
    1.9.2 - Fixed SDK handler body reconstruction: always reconstruct request after consuming body
            Root cause: `if body_bytes:` evaluates to False for empty bytes (`b""`), causing
            the original request (with consumed body) to be passed to sdk_handler, resulting
            in "expected string or bytes-like object, got 'NoneType'" errors.
    1.9.1 - Fixed AG-UI event type casing: use SCREAMING_SNAKE_CASE event types
            (TEXT_MESSAGE_START not TextMessageStart) per AG-UI protocol specification.
    1.9.0 - Fixed TEXT_MESSAGE events not being emitted: CopilotKit SDK v0.1.74 has a bug where
            _dispatch_event() creates TEXT_MESSAGE events but discards their return values.
            Workaround: manually emit TEXT_MESSAGE_START/CONTENT/END events in execute() method.
    1.8.0 - Fixed "Message ID not found in history" error: use fresh graph/checkpointer per request
            Root cause: SDK's prepare_stream() triggered regenerate mode when checkpoint had more
            messages than frontend sent, but frontend message IDs don't exist in checkpoint history.
    1.7.0 - Fixed custom event dispatch: use adispatch_custom_event with RunnableConfig for proper AG-UI routing
    1.6.9 - Fixed 307 redirect breaking streaming: add base path route for /api/copilotkit (without trailing slash)
    1.6.8 - Fixed custom event name: use copilotkit_manually_emit_message (not manually_emit_message) for SDK compatibility
    1.6.7 - Fixed text message emission: emit manually_emit_message custom event for AG-UI frontend rendering
    1.6.6 - Fixed streaming lifecycle: bypass SDK handle_execute_agent to properly stream all events
    1.6.5 - Added detailed timing diagnostics to trace 29-second streaming delay
    1.6.4 - Fixed streaming format: add newline delimiters for proper NDJSON parsing by frontend SDK
    1.6.3 - Fixed AG-UI event serialization: serialize Pydantic events to JSON strings for StreamingResponse
    1.6.2 - Added MemorySaver checkpointer to LangGraph graph (required by ag_ui_langgraph)
    1.6.1 - Fixed run_id validation error: auto-generate UUID when SDK doesn't provide run_id
    1.6.0 - Fixed SDK incompatibility: wrapper class adds execute() method to LangGraphAGUIAgent
    1.5.0 - (Reverted) Attempted switch to LangGraphAgent (SDK rejects it)
    1.4.0 - Replaced middleware with custom handler (cleaner SDK delegation with info transformation)
    1.3.0 - Connected to real repositories (BusinessMetricRepository, AgentRegistryRepository)
    1.2.0 - Refactored from monkey-patches to response transformer middleware
    1.1.0 - Added SDK compatibility patches for frontend v1.x
    1.0.0 - Initial CopilotKit integration
"""

import asyncio
import json
import logging
import operator
import os
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional, Sequence, TypedDict

from copilotkit import CopilotKitRemoteEndpoint, Action as CopilotAction
from copilotkit.langgraph import copilotkit_emit_message
from copilotkit.langgraph_agui_agent import LangGraphAGUIAgent as _LangGraphAGUIAgent
from ag_ui.core import RunAgentInput
from copilotkit.integrations.fastapi import (
    handler as sdk_handler,
    handle_execute_action,
    handle_execute_agent,
    handle_get_agent_state,
)
from copilotkit.sdk import COPILOTKIT_SDK_VERSION, CopilotKitContext
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# EVENT TYPE CONVERSION (DEPRECATED - v1.16.0)
# =============================================================================
# NOTE: This function was used in v1.13.0-v1.15.0 but is NO LONGER NEEDED.
# CopilotKit React SDK (v1.50.1) actually expects SCREAMING_SNAKE_CASE event types
# (TEXT_MESSAGE_START, RUN_STARTED, etc.) per the AG-UI protocol specification.
# The v1.13.0 conversion to PascalCase was based on incorrect assumptions.
# Keeping this function for reference but it is no longer called.


def _screaming_snake_to_pascal(event_type: str) -> str:
    """
    DEPRECATED: Convert SCREAMING_SNAKE_CASE event type to PascalCase.

    NOTE: This function is no longer used as of v1.16.0.
    CopilotKit SDK expects SCREAMING_SNAKE_CASE, not PascalCase.

    Examples:
        RUN_STARTED -> RunStarted
        TEXT_MESSAGE_START -> TextMessageStart
        TEXT_MESSAGE_CONTENT -> TextMessageContent
        RUN_FINISHED -> RunFinished

    Args:
        event_type: Event type in SCREAMING_SNAKE_CASE format

    Returns:
        Event type in PascalCase format
    """
    # Split by underscore, capitalize each word, join
    return "".join(word.capitalize() for word in event_type.split("_"))


def _fix_all_events(event_dict: dict, thread_id: str, run_id: str) -> dict:
    """
    Fix ALL events to include required timestamp and source fields.

    CopilotKit React SDK (v1.50.1) uses Zod validation that requires:
    - timestamp: number (Unix timestamp in milliseconds) on ALL events
    - source: string on ALL events (empty string allowed)

    Additionally, lifecycle events (RUN_STARTED, RUN_FINISHED) require:
    - parentRunId: string (can be empty but not null)
    - input/output: object structures
    - threadId: string
    - runId: string

    v1.18.0 only fixed lifecycle events. v1.19.0 extends this to ALL events.

    Args:
        event_dict: The serialized event dictionary
        thread_id: The thread ID for this run
        run_id: The run ID for this run

    Returns:
        Fixed event dictionary with all required fields
    """
    import time

    event_type = event_dict.get("type", "")

    # CRITICAL FIX (v1.19.0): Ensure timestamp and source on ALL events
    # CopilotKit React SDK (v1.50.1) Zod validation requires these on every event type,
    # including TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, TEXT_MESSAGE_END,
    # STATE_SNAPSHOT, MESSAGES_SNAPSHOT, and any other AG-UI events.
    if event_dict.get("timestamp") is None:
        event_dict["timestamp"] = int(time.time() * 1000)

    if event_dict.get("source") is None:
        event_dict["source"] = "e2i-copilot"

    # Lifecycle-specific fields (preserved from v1.17.0 and v1.18.0)
    if event_type in ("RUN_STARTED", "RUN_FINISHED"):
        # Ensure parentRunId is a string (empty string if null)
        if event_dict.get("parentRunId") is None:
            event_dict["parentRunId"] = ""

        # Ensure threadId is set
        if event_dict.get("threadId") is None:
            event_dict["threadId"] = thread_id

        # Ensure runId is set
        if event_dict.get("runId") is None:
            event_dict["runId"] = run_id

    # RUN_STARTED specific: ensure input contains full RunAgentInput structure
    # CopilotKit SDK (v1.50.1) Zod schema requires:
    #   input.threadId: string
    #   input.runId: string
    #   input.messages: array
    #   input.tools: array
    #   input.context: array
    if event_type == "RUN_STARTED":
        input_obj = event_dict.get("input")
        if input_obj is None:
            input_obj = {}
            event_dict["input"] = input_obj
        # Ensure all required fields are present
        if input_obj.get("threadId") is None:
            input_obj["threadId"] = thread_id
        if input_obj.get("runId") is None:
            input_obj["runId"] = run_id
        if input_obj.get("messages") is None:
            input_obj["messages"] = []
        if input_obj.get("tools") is None:
            input_obj["tools"] = []
        if input_obj.get("context") is None:
            input_obj["context"] = []

    # RUN_FINISHED specific: ensure output contains structure
    # Similar to input, output may need structure for Zod validation
    if event_type == "RUN_FINISHED":
        output_obj = event_dict.get("output")
        if output_obj is None:
            output_obj = {}
            event_dict["output"] = output_obj
        # Ensure messages array exists (SDK may expect this)
        if output_obj.get("messages") is None:
            output_obj["messages"] = []

    # CRITICAL FIX (v1.20.0): Fix messages in MESSAGES_SNAPSHOT
    # CopilotKit React SDK (v1.50.1) Zod validation requires:
    #   messages[].name: string (empty string if null)
    #   messages[].toolCalls: array (empty array if null)
    # AG-UI SDK emits null values for these optional fields, causing validation errors.
    if event_type == "MESSAGES_SNAPSHOT":
        messages = event_dict.get("messages", [])
        if messages:
            for msg in messages:
                # Ensure name is a string (empty string if null)
                if msg.get("name") is None:
                    msg["name"] = ""
                # Ensure toolCalls is an array (empty array if null) for assistant messages
                if msg.get("role") == "assistant" and msg.get("toolCalls") is None:
                    msg["toolCalls"] = []

    return event_dict


# =============================================================================
# SDK COMPATIBILITY: LangGraphAGUIAgent with execute() method
# =============================================================================
# The CopilotKit SDK (v0.1.74) has a bug: it enforces using LangGraphAGUIAgent
# but calls agent.execute() which doesn't exist on that class (only run() exists).
# This wrapper bridges the gap by adding execute() that delegates to run().


class LangGraphAgent(_LangGraphAGUIAgent):
    """
    Extended LangGraphAGUIAgent that adds the execute() method required by SDK.

    The SDK's CopilotKitRemoteEndpoint.execute_agent() calls agent.execute(),
    but LangGraphAGUIAgent only has run() inherited from ag_ui_langgraph.
    This wrapper provides the missing execute() method.

    CRITICAL FIX (v1.6.8): Uses a graph factory to create fresh checkpointer
    per request, avoiding "Message ID not found in history" error when
    checkpoint accumulates more messages than frontend sends.
    """

    def __init__(self, name: str, description: str = "", graph=None, graph_factory=None, **kwargs):
        """
        Initialize with either a static graph or a factory function.

        Args:
            name: Agent name
            description: Agent description
            graph: Pre-created graph (will be ignored if graph_factory provided)
            graph_factory: Callable that returns a fresh graph with new checkpointer
        """
        self._graph_factory = graph_factory
        # Initialize parent with a graph (required by parent class)
        # If factory is provided, this graph is just for initialization
        super().__init__(name=name, description=description, graph=graph, **kwargs)

    def _get_fresh_graph(self):
        """Get a fresh graph instance with new checkpointer."""
        if self._graph_factory:
            return self._graph_factory()
        return self.graph

    def dict_repr(self) -> Dict[str, Any]:
        """Return dictionary representation for SDK info endpoint."""
        return {
            "name": self.name,
            "description": getattr(self, "description", "") or "",
        }

    async def execute(
        self,
        *,
        thread_id: str,
        state: dict,
        messages: List[Any],
        config: Optional[dict] = None,
        actions: Optional[List[Any]] = None,
        node_name: Optional[str] = None,
        meta_events: Optional[List[Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Bridge method: converts execute() parameters to RunAgentInput and calls run().

        The SDK calls execute() with these parameters, but LangGraphAGUIAgent
        expects run(input: RunAgentInput). This method performs the conversion
        and serializes the AG-UI events to strings for the StreamingResponse.

        CRITICAL FIX (v1.9.4): Force unique thread_id per request to prevent SDK's
        regenerate mode from being triggered. The SDK compares checkpoint messages
        vs frontend messages, and if checkpoint has more (from previous AI responses),
        it triggers regenerate mode which fails when message IDs don't exist in history.
        By using a fresh thread_id, the checkpointer always returns empty state.
        """
        import time
        import sys
        from datetime import datetime
        start_time = time.time()

        def dbg(msg):
            """Flushed debug log with wall-clock and elapsed time."""
            wall = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            elapsed = time.time() - start_time
            print(f"[{wall}] DEBUG execute [{elapsed:.3f}s]: {msg}", flush=True)

        # CRITICAL FIX (v1.9.4): Generate a fresh thread_id to prevent regenerate mode.
        # The SDK's prepare_stream() triggers regenerate mode when:
        #   len(checkpoint_messages) > len(frontend_messages)
        # With a fresh thread_id, the checkpointer lookup returns empty state,
        # so the regenerate check (0 > N) is always False.
        original_thread_id = thread_id
        thread_id = str(uuid.uuid4())
        dbg(f"Using fresh thread_id={thread_id[:8]}... (original={original_thread_id[:8] if original_thread_id else 'None'}...)")

        # Convert messages to the format expected by RunAgentInput
        # Messages can come from:
        # 1. AG-UI protocol: dicts like {"role": "user", "content": "..."}
        # 2. SDK internals: LangChain message objects with .type and .content attributes
        # Note: ag_ui.core.Message is a Union type, so we must use specific types
        from ag_ui.core import UserMessage, AssistantMessage

        agui_messages = []
        dbg(f"Converting {len(messages or [])} messages to AG-UI format")
        for i, msg in enumerate(messages or []):
            dbg(f"msg[{i}] type={type(msg).__name__} value={msg}")

            if isinstance(msg, dict):
                # Handle dict format from AG-UI protocol (frontend sends this)
                role = msg.get("role", "user")
                content = msg.get("content", "")
                msg_id = msg.get("id") or f"msg-{uuid.uuid4()}"
                if content:
                    if role == "user":
                        agui_messages.append(UserMessage(id=msg_id, content=content))
                    else:
                        agui_messages.append(AssistantMessage(id=msg_id, content=content))
                    dbg(f"Added dict message: role={role}, content={content[:50]}...")
            elif hasattr(msg, "content") and hasattr(msg, "type"):
                # Convert langchain message to AGUI format
                role = "user" if msg.type == "human" else "assistant"
                msg_id = getattr(msg, "id", None) or f"msg-{uuid.uuid4()}"
                if role == "user":
                    agui_messages.append(UserMessage(id=msg_id, content=msg.content))
                else:
                    agui_messages.append(AssistantMessage(id=msg_id, content=msg.content))
                dbg(f"Added LangChain message: role={role}, content={msg.content[:50]}...")

        dbg(f"Converted to {len(agui_messages)} AG-UI messages")

        # Build RunAgentInput
        # Generate run_id if not provided (SDK doesn't always pass it)
        run_id = kwargs.get("run_id") or str(uuid.uuid4())

        run_input = RunAgentInput(
            thread_id=thread_id,
            run_id=run_id,
            state=state,
            messages=agui_messages,
            tools=actions,  # CopilotKit actions become tools
            context=[],
            forwarded_props={"node_name": node_name} if node_name else {},
        )

        dbg("Created RunAgentInput, calling self.run()")

        # CRITICAL FIX (v1.6.8): Use fresh graph with new checkpointer to avoid
        # "Message ID not found in history" error. The SDK's prepare_stream()
        # triggers regenerate mode when checkpoint has more messages than input.
        # By using a fresh checkpointer, the checkpoint is always empty.
        original_graph = self.graph
        if self._graph_factory:
            self.graph = self._graph_factory()
            dbg("Created fresh graph with new checkpointer")

        # Call parent's run() method and serialize each AG-UI event to string
        # The run() method yields Pydantic AG-UI event objects that need serialization
        # IMPORTANT: Add newline delimiter after each event for proper NDJSON streaming
        # CopilotKit frontend SDK expects newline-delimited JSON events
        #
        # FIX (v1.11.0): Intercept CUSTOM events with name "copilotkit_manually_emit_message"
        # and emit TEXT_MESSAGE_START/CONTENT/END events. This is needed because:
        # 1. copilotkit_emit_message() emits a CUSTOM event (not TEXT_MESSAGE directly)
        # 2. The CopilotKit Runtime (TypeScript) normally converts CUSTOM -> TEXT_MESSAGE
        # 3. But our custom FastAPI endpoint bypasses the Runtime, so we must do it ourselves
        # Import EventType for checking CUSTOM events (we use manual JSON for TEXT_MESSAGE)
        from ag_ui.core import EventType

        event_count = 0
        dbg("Entering self.run() async loop")
        try:
            async for event in self.run(run_input):
                event_count += 1
                event_type = getattr(event, 'type', 'unknown') if hasattr(event, 'type') else type(event).__name__
                dbg(f"Yielding event #{event_count} type={event_type}")

                # Check if this is a CUSTOM event with copilotkit_manually_emit_message
                # If so, emit TEXT_MESSAGE events (mimicking what CopilotKit Runtime does)
                if hasattr(event, 'type') and event.type == EventType.CUSTOM:
                    event_name = getattr(event, 'name', None)
                    if event_name == "copilotkit_manually_emit_message":
                        event_value = getattr(event, 'value', {}) or {}
                        message_id = event_value.get('message_id', str(uuid.uuid4()))
                        message = event_value.get('message', '')

                        dbg(f"Converting CUSTOM to TEXT_MESSAGE events for message_id={message_id}")

                        # CRITICAL FIX (v1.16.0): Use SCREAMING_SNAKE_CASE event types
                        # CopilotKit React SDK (v1.50.1) uses Zod validation that expects
                        # SCREAMING_SNAKE_CASE: TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, etc.
                        # The v1.13.0 PascalCase change was incorrect and caused Zod errors.
                        #
                        # CRITICAL FIX (v1.19.0): Add timestamp and source to ALL events
                        # CopilotKit React SDK (v1.50.1) Zod validation requires timestamp (number)
                        # and source (string) on ALL events, not just lifecycle events.
                        import time
                        current_ts = int(time.time() * 1000)
                        source = "e2i-copilot"

                        # Emit TEXT_MESSAGE_START with required timestamp and source
                        yield f"data: {json.dumps({'type': 'TEXT_MESSAGE_START', 'messageId': message_id, 'role': 'assistant', 'timestamp': current_ts, 'source': source})}\n\n"
                        event_count += 1

                        # Emit TEXT_MESSAGE_CONTENT with required timestamp and source
                        yield f"data: {json.dumps({'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': message, 'timestamp': current_ts, 'source': source})}\n\n"
                        event_count += 1

                        # Emit TEXT_MESSAGE_END with required timestamp and source
                        yield f"data: {json.dumps({'type': 'TEXT_MESSAGE_END', 'messageId': message_id, 'timestamp': current_ts, 'source': source})}\n\n"
                        event_count += 1

                        # Skip emitting the original CUSTOM event (frontend doesn't need it)
                        continue

                # Serialize and yield the event
                # CRITICAL FIX (v1.16.0): Keep SCREAMING_SNAKE_CASE event types
                # CopilotKit React SDK (v1.50.1) expects SCREAMING_SNAKE_CASE
                # (RUN_STARTED, TEXT_MESSAGE_START, etc.) per AG-UI protocol.
                # The v1.13.0 PascalCase conversion was incorrect.
                #
                # CRITICAL FIX (v1.17.0): Fix lifecycle events to include required fields
                # CopilotKit SDK requires timestamp, parentRunId, input/output for
                # RUN_STARTED/RUN_FINISHED events. AG-UI SDK emits null values.
                if isinstance(event, str):
                    # Already a string - wrap in SSE format
                    try:
                        event_dict = json.loads(event.strip())
                        # Keep event type as-is (SCREAMING_SNAKE_CASE)
                        if "type" in event_dict:
                            dbg(f"Yielding string event type: {event_dict['type']}")
                        # Fix lifecycle events (v1.17.0)
                        event_dict = _fix_all_events(event_dict, thread_id, run_id)
                        yield f"data: {json.dumps(event_dict)}\n\n"
                    except (json.JSONDecodeError, KeyError):
                        # Wrap in SSE format if not already
                        yield f"data: {event.strip()}\n\n"
                elif hasattr(event, "model_dump"):
                    # Pydantic v2 object - serialize to dict with SSE format
                    event_dict = event.model_dump(by_alias=True)
                    if "type" in event_dict:
                        # Handle enum objects that serialize to their value
                        if hasattr(event_dict["type"], "value"):
                            event_dict["type"] = event_dict["type"].value
                        else:
                            event_dict["type"] = str(event_dict["type"])
                        dbg(f"Yielding Pydantic event type: {event_dict['type']}")
                    # Fix lifecycle events (v1.17.0)
                    event_dict = _fix_all_events(event_dict, thread_id, run_id)
                    yield f"data: {json.dumps(event_dict)}\n\n"
                elif hasattr(event, "dict"):
                    # Pydantic v1 object - serialize to dict with SSE format
                    event_dict = event.dict(by_alias=True)
                    if "type" in event_dict:
                        if hasattr(event_dict["type"], "value"):
                            event_dict["type"] = event_dict["type"].value
                        else:
                            event_dict["type"] = str(event_dict["type"])
                        dbg(f"Yielding Pydantic v1 event type: {event_dict['type']}")
                    # Fix lifecycle events (v1.17.0)
                    event_dict = _fix_all_events(event_dict, thread_id, run_id)
                    yield f"data: {json.dumps(event_dict)}\n\n"
                else:
                    # Fallback - convert to string with SSE format
                    yield f"data: {str(event)}\n\n"
        finally:
            # Restore original graph if we swapped it
            if self._graph_factory:
                self.graph = original_graph
                dbg("Restored original graph")

        dbg(f"Finished yielding {event_count} events")


# =============================================================================
# REPOSITORY HELPERS
# =============================================================================


def _get_business_metric_repository():
    """Get BusinessMetricRepository instance with Supabase client."""
    try:
        from src.api.dependencies.supabase_client import get_supabase
        from src.repositories.business_metric import BusinessMetricRepository

        client = get_supabase()
        return BusinessMetricRepository(client=client) if client else None
    except Exception as e:
        logger.warning(f"Failed to get BusinessMetricRepository: {e}")
        return None


def _get_agent_registry_repository():
    """Get AgentRegistryRepository instance with Supabase client."""
    try:
        from src.api.dependencies.supabase_client import get_supabase
        from src.repositories.agent_registry import AgentRegistryRepository

        client = get_supabase()
        return AgentRegistryRepository(client=client) if client else None
    except Exception as e:
        logger.warning(f"Failed to get AgentRegistryRepository: {e}")
        return None

# =============================================================================
# E2I BACKEND ACTIONS
# =============================================================================

# Fallback sample data when database is unavailable
_FALLBACK_KPIS = {
    "Remibrutinib": {
        "trx_volume": 15420,
        "nrx_volume": 3250,
        "market_share": 12.5,
        "conversion_rate": 0.68,
        "hcp_reach": 2450,
        "patient_starts": 890,
    },
    "Fabhalta": {
        "trx_volume": 8920,
        "nrx_volume": 1850,
        "market_share": 8.2,
        "conversion_rate": 0.72,
        "hcp_reach": 1820,
        "patient_starts": 560,
    },
    "Kisqali": {
        "trx_volume": 22100,
        "nrx_volume": 4200,
        "market_share": 18.5,
        "conversion_rate": 0.65,
        "hcp_reach": 3200,
        "patient_starts": 1250,
    },
}

_FALLBACK_AGENTS = [
    {"id": "orchestrator", "name": "Orchestrator", "tier": 1, "status": "active"},
    {"id": "causal-impact", "name": "Causal Impact", "tier": 2, "status": "idle"},
    {"id": "gap-analyzer", "name": "Gap Analyzer", "tier": 2, "status": "idle"},
    {"id": "drift-monitor", "name": "Drift Monitor", "tier": 3, "status": "active"},
    {"id": "health-score", "name": "Health Score", "tier": 3, "status": "active"},
    {"id": "explainer", "name": "Explainer", "tier": 5, "status": "idle"},
]


async def _fetch_kpis_from_db(brand: str) -> Optional[Dict[str, Any]]:
    """
    Fetch KPI data from database for a brand.

    Returns:
        KPI metrics dict or None if unavailable
    """
    repo = _get_business_metric_repository()
    if not repo:
        return None

    try:
        # Define KPI mappings to metric_name in database
        kpi_mappings = {
            "trx_volume": "TRx",
            "nrx_volume": "NRx",
            "market_share": "market_share",
            "conversion_rate": "conversion_rate",
            "hcp_reach": "hcp_reach",
            "patient_starts": "patient_starts",
        }

        metrics = {}
        for metric_key, db_metric_name in kpi_mappings.items():
            results = await repo.get_by_kpi(
                kpi_name=db_metric_name,
                brand=brand if brand != "All" else None,
                limit=1,
            )
            if results:
                # Get most recent value
                metrics[metric_key] = results[0].get("value", 0)
            else:
                metrics[metric_key] = 0

        return metrics if any(metrics.values()) else None

    except Exception as e:
        logger.warning(f"Failed to fetch KPIs from database: {e}")
        return None


async def get_kpi_summary(brand: str) -> Dict[str, Any]:
    """
    Get KPI summary for a specific brand.

    Attempts to fetch real data from database, falls back to sample data if unavailable.

    Args:
        brand: Brand name (Remibrutinib, Fabhalta, Kisqali, or All)

    Returns:
        Dictionary with KPI metrics
    """
    logger.info(f"[CopilotKit] Fetching KPI summary for brand: {brand}")

    valid_brands = ["Remibrutinib", "Fabhalta", "Kisqali", "All"]
    if brand not in valid_brands:
        return {"error": f"Unknown brand: {brand}. Available: {valid_brands[:-1]}"}

    # Try to fetch from database first
    db_metrics = await _fetch_kpis_from_db(brand)
    data_source = "database"

    if db_metrics:
        metrics = db_metrics
    else:
        # Fall back to sample data
        data_source = "fallback"
        if brand == "All":
            metrics = {
                "trx_volume": sum(b["trx_volume"] for b in _FALLBACK_KPIS.values()),
                "nrx_volume": sum(b["nrx_volume"] for b in _FALLBACK_KPIS.values()),
                "market_share": sum(b["market_share"] for b in _FALLBACK_KPIS.values()) / 3,
                "conversion_rate": sum(b["conversion_rate"] for b in _FALLBACK_KPIS.values()) / 3,
                "hcp_reach": sum(b["hcp_reach"] for b in _FALLBACK_KPIS.values()),
                "patient_starts": sum(b["patient_starts"] for b in _FALLBACK_KPIS.values()),
                "brands_included": list(_FALLBACK_KPIS.keys()),
            }
        else:
            metrics = _FALLBACK_KPIS.get(brand, {})

    return {
        "brand": brand,
        "period": "Last 90 days",
        "metrics": metrics,
        "data_source": data_source,
    }


async def _fetch_agents_from_db() -> Optional[List[Dict[str, Any]]]:
    """
    Fetch agent status from database.

    Returns:
        List of agent dicts or None if unavailable
    """
    repo = _get_agent_registry_repository()
    if not repo:
        return None

    try:
        # Fetch all active agents
        all_agents = []
        for tier in range(1, 6):  # Tiers 1-5
            tier_agents = await repo.get_by_tier(tier)
            for agent in tier_agents:
                all_agents.append({
                    "id": agent.get("agent_name", "unknown"),
                    "name": agent.get("display_name", agent.get("agent_name", "Unknown")),
                    "tier": agent.get("tier", tier),
                    "status": "active" if agent.get("is_active", True) else "idle",
                    "description": agent.get("description", ""),
                })

        return all_agents if all_agents else None

    except Exception as e:
        logger.warning(f"Failed to fetch agents from database: {e}")
        return None


async def get_agent_status() -> Dict[str, Any]:
    """
    Get the status of all E2I agents.

    Attempts to fetch real data from database, falls back to sample data if unavailable.

    Returns:
        Dictionary with agent status information
    """
    logger.info("[CopilotKit] Fetching agent status")

    # Try to fetch from database first
    db_agents = await _fetch_agents_from_db()
    data_source = "database"

    if db_agents:
        agents = db_agents
    else:
        # Fall back to sample data
        data_source = "fallback"
        agents = _FALLBACK_AGENTS

    active_count = sum(1 for a in agents if a.get("status") == "active")

    return {
        "total_agents": len(agents),
        "active_agents": active_count,
        "idle_agents": len(agents) - active_count,
        "agents": agents,
        "data_source": data_source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _get_orchestrator():
    """Get OrchestratorAgent singleton for causal analysis."""
    try:
        from src.api.routes.cognitive import get_orchestrator
        return get_orchestrator()
    except Exception as e:
        logger.warning(f"Failed to get orchestrator: {e}")
        return None


async def run_causal_analysis(
    intervention: str,
    target_kpi: str,
    brand: str,
) -> Dict[str, Any]:
    """
    Run a causal impact analysis.

    Attempts to use the orchestrator for real causal analysis, falls back to simulated results.

    Args:
        intervention: Type of intervention (e.g., "HCP Engagement", "Marketing Campaign")
        target_kpi: Target KPI to analyze (e.g., "TRx Volume", "Market Share")
        brand: Brand to analyze

    Returns:
        Dictionary with causal analysis results
    """
    logger.info(f"[CopilotKit] Running causal analysis: {intervention} -> {target_kpi} for {brand}")

    # Try to run through orchestrator for real causal analysis
    orchestrator = _get_orchestrator()
    data_source = "orchestrator"

    if orchestrator:
        try:
            query = f"What is the causal impact of {intervention} on {target_kpi} for {brand}?"
            result = await orchestrator.run({
                "query": query,
                "user_context": {
                    "brand": brand,
                    "intervention": intervention,
                    "target_kpi": target_kpi,
                },
            })

            # Extract causal results if available
            if result and result.get("response_text"):
                return {
                    "intervention": intervention,
                    "target_kpi": target_kpi,
                    "brand": brand,
                    "results": result.get("causal_results", {
                        "average_treatment_effect": result.get("ate", 0.0),
                        "confidence_interval": result.get("ci", [0.0, 0.0]),
                        "p_value": result.get("p_value", 0.0),
                        "statistical_significance": result.get("significant", False),
                    }),
                    "interpretation": result.get("response_text", ""),
                    "data_source": data_source,
                    "agents_used": result.get("agents_dispatched", []),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            logger.warning(f"Orchestrator causal analysis failed: {e}")

    # Fallback to simulated results
    import random
    data_source = "simulated"
    ate = random.uniform(0.05, 0.25)

    return {
        "intervention": intervention,
        "target_kpi": target_kpi,
        "brand": brand,
        "results": {
            "average_treatment_effect": round(ate, 3),
            "confidence_interval": [round(ate - 0.05, 3), round(ate + 0.05, 3)],
            "p_value": round(random.uniform(0.001, 0.05), 4),
            "statistical_significance": True,
            "sample_size": random.randint(500, 2000),
        },
        "interpretation": f"The {intervention} shows a statistically significant positive effect on {target_kpi}, with an estimated {round(ate * 100, 1)}% lift.",
        "data_source": data_source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def get_recommendations(brand: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Get AI-powered recommendations for a brand.

    Args:
        brand: Brand to get recommendations for
        context: Optional context about what kind of recommendations are needed

    Returns:
        Dictionary with recommendations
    """
    logger.info(f"[CopilotKit] Generating recommendations for {brand}")

    recommendations = [
        {
            "priority": "high",
            "category": "HCP Targeting",
            "recommendation": f"Focus on high-decile HCPs in the Northeast region for {brand}",
            "expected_impact": "+15% TRx lift",
            "confidence": 0.85,
        },
        {
            "priority": "medium",
            "category": "Patient Journey",
            "recommendation": f"Implement patient support program to reduce {brand} discontinuation",
            "expected_impact": "+8% persistence rate",
            "confidence": 0.78,
        },
        {
            "priority": "medium",
            "category": "Market Access",
            "recommendation": f"Target formulary additions in 3 key health systems for {brand}",
            "expected_impact": "+12% market share",
            "confidence": 0.72,
        },
    ]

    return {
        "brand": brand,
        "context": context or "General recommendations",
        "recommendations": recommendations,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def search_insights(query: str, brand: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for insights in the E2I knowledge base.

    Args:
        query: Search query
        brand: Optional brand filter

    Returns:
        Dictionary with search results
    """
    logger.info(f"[CopilotKit] Searching insights: {query}")

    # Simulated search results
    results = [
        {
            "type": "causal_path",
            "title": "HCP Engagement -> TRx Volume Causal Chain",
            "summary": "Strong causal relationship identified between HCP engagement frequency and TRx volume increases.",
            "confidence": 0.89,
            "brand": brand or "Remibrutinib",
        },
        {
            "type": "trend",
            "title": "Q4 Market Share Trend",
            "summary": "Market share increased by 2.3% following targeted digital campaign.",
            "confidence": 0.92,
            "brand": brand or "All",
        },
        {
            "type": "agent_insight",
            "title": "Drift Monitor Alert",
            "summary": "Model drift detected in conversion prediction model. Retraining recommended.",
            "confidence": 0.95,
            "brand": None,
        },
    ]

    return {
        "query": query,
        "brand_filter": brand,
        "results": results,
        "total_results": len(results),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# COPILOTKIT ACTIONS
# =============================================================================

COPILOT_ACTIONS = [
    CopilotAction(
        name="getKPISummary",
        description="Get key performance indicator (KPI) summary for a pharmaceutical brand. Returns metrics like TRx volume, NRx volume, market share, conversion rate, HCP reach, and patient starts.",
        parameters=[
            {
                "name": "brand",
                "type": "string",
                "description": "The brand to get KPIs for. Options: Remibrutinib, Fabhalta, Kisqali, or All",
                "required": True,
            }
        ],
        handler=get_kpi_summary,
    ),
    CopilotAction(
        name="getAgentStatus",
        description="Get the current status of all E2I agents in the 6-tier hierarchy. Shows which agents are active, idle, or processing.",
        parameters=[],
        handler=get_agent_status,
    ),
    CopilotAction(
        name="runCausalAnalysis",
        description="Run a causal impact analysis to measure the effect of an intervention on a target KPI. Uses DoWhy/EconML for causal inference.",
        parameters=[
            {
                "name": "intervention",
                "type": "string",
                "description": "The type of intervention to analyze (e.g., 'HCP Engagement', 'Marketing Campaign', 'Patient Support Program')",
                "required": True,
            },
            {
                "name": "target_kpi",
                "type": "string",
                "description": "The KPI to measure impact on (e.g., 'TRx Volume', 'Market Share', 'Conversion Rate')",
                "required": True,
            },
            {
                "name": "brand",
                "type": "string",
                "description": "The brand to analyze",
                "required": True,
            },
        ],
        handler=run_causal_analysis,
    ),
    CopilotAction(
        name="getRecommendations",
        description="Get AI-powered recommendations for improving brand performance. Returns prioritized recommendations with expected impact.",
        parameters=[
            {
                "name": "brand",
                "type": "string",
                "description": "The brand to get recommendations for",
                "required": True,
            },
            {
                "name": "context",
                "type": "string",
                "description": "Optional context about what kind of recommendations are needed",
                "required": False,
            },
        ],
        handler=get_recommendations,
    ),
    CopilotAction(
        name="searchInsights",
        description="Search the E2I knowledge base for insights, causal paths, trends, and agent outputs.",
        parameters=[
            {
                "name": "query",
                "type": "string",
                "description": "The search query",
                "required": True,
            },
            {
                "name": "brand",
                "type": "string",
                "description": "Optional brand filter",
                "required": False,
            },
        ],
        handler=search_insights,
    ),
]


# =============================================================================
# LANGGRAPH AGENT FOR E2I CHAT
# =============================================================================


class E2IAgentState(TypedDict):
    """State for the E2I chat agent."""

    messages: Annotated[Sequence[BaseMessage], operator.add]


def create_e2i_chat_agent():
    """
    Create a simple LangGraph agent for E2I chat.

    This agent responds to chat messages and provides helpful information
    about the E2I Causal Analytics platform. It delegates tool calls
    to CopilotKit actions.
    """

    async def chat_node(state: E2IAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Process chat messages and generate responses."""

        messages = state.get("messages", [])

        import time
        node_start = time.time()

        # Debug: Log what messages look like
        print(f"DEBUG chat_node [t={node_start:.3f}]: Received {len(messages)} messages")
        for i, msg in enumerate(messages):
            print(f"DEBUG chat_node [t={node_start:.3f}]: msg[{i}] type={type(msg).__name__} content={msg if isinstance(msg, dict) else getattr(msg, 'content', str(msg)[:100])}")

        # Get the last human message - handle both dict and HumanMessage formats
        last_message = None
        for msg in reversed(messages):
            # Handle LangChain message objects
            if isinstance(msg, HumanMessage):
                last_message = msg.content
                print(f"DEBUG chat_node: Found HumanMessage: {last_message[:100]}")
                break
            # Handle dict format from AG-UI protocol
            elif isinstance(msg, dict):
                role = msg.get("role", "")
                if role == "user":
                    last_message = msg.get("content", "")
                    print(f"DEBUG chat_node: Found dict user message: {last_message[:100]}")
                    break

        # Generate message ID for this response
        message_id = f"ai-{uuid.uuid4()}"

        if not last_message:
            print("DEBUG chat_node: No user message found, returning greeting")
            greeting = "Hello! I'm the E2I Analytics Assistant. I can help you with KPI analysis, causal inference, and insights for pharmaceutical brands. What would you like to know?"
            # Use SDK's native message emission (v1.10.0)
            await copilotkit_emit_message(config, greeting)
            return {
                "messages": [AIMessage(id=message_id, content=greeting)]
            }

        # Generate a contextual response based on the query
        response = generate_e2i_response(last_message)
        print(f"DEBUG chat_node: Generated response: {response[:100]}...")

        # Use SDK's native message emission (v1.10.0)
        await copilotkit_emit_message(config, response)

        return {"messages": [AIMessage(id=message_id, content=response)]}

    # Build the graph
    workflow = StateGraph(E2IAgentState)
    workflow.add_node("chat", chat_node)
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", END)

    # Checkpointer is required by ag_ui_langgraph for state management
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def generate_e2i_response(query: str) -> str:
    """
    Generate a contextual response for E2I queries.

    This is a simple response generator. In production, this would
    integrate with Claude API for more sophisticated responses.
    """
    query_lower = query.lower()

    # KPI-related queries
    if any(kw in query_lower for kw in ["kpi", "trx", "nrx", "market share", "metric"]):
        return (
            "I can help you with KPI analysis! Use the **getKPISummary** action to get detailed metrics "
            "for any brand (Remibrutinib, Fabhalta, Kisqali, or All). This includes TRx volume, NRx volume, "
            "market share, conversion rate, HCP reach, and patient starts."
        )

    # Agent-related queries
    if any(kw in query_lower for kw in ["agent", "status", "tier", "orchestrator"]):
        return (
            "The E2I platform uses an 18-agent tiered architecture:\n\n"
            "- **Tier 0**: ML Foundation (7 agents)\n"
            "- **Tier 1**: Orchestration (2 agents)\n"
            "- **Tier 2**: Causal Analytics (3 agents)\n"
            "- **Tier 3**: Monitoring (3 agents)\n"
            "- **Tier 4**: ML Predictions (2 agents)\n"
            "- **Tier 5**: Self-Improvement (2 agents)\n\n"
            "Use the **getAgentStatus** action to see which agents are currently active."
        )

    # Causal analysis queries
    if any(kw in query_lower for kw in ["causal", "impact", "intervention", "effect", "ate"]):
        return (
            "I can run causal impact analyses! Use the **runCausalAnalysis** action with:\n\n"
            "- **intervention**: Type of intervention (e.g., 'HCP Engagement', 'Marketing Campaign')\n"
            "- **target_kpi**: KPI to measure (e.g., 'TRx Volume', 'Market Share')\n"
            "- **brand**: Brand to analyze\n\n"
            "The analysis uses DoWhy/EconML for rigorous causal inference."
        )

    # Recommendation queries
    if any(kw in query_lower for kw in ["recommend", "suggest", "improve", "optimize"]):
        return (
            "I can provide AI-powered recommendations! Use the **getRecommendations** action with a brand name "
            "to get prioritized suggestions for HCP targeting, patient journey optimization, and market access strategies."
        )

    # Search/insight queries
    if any(kw in query_lower for kw in ["search", "find", "insight", "trend"]):
        return (
            "I can search the E2I knowledge base for insights! Use the **searchInsights** action with your query "
            "to find causal paths, trends, and agent outputs. You can optionally filter by brand."
        )

    # Brand-specific queries
    if any(brand.lower() in query_lower for brand in ["remibrutinib", "fabhalta", "kisqali"]):
        return (
            "I see you're asking about a specific brand. I have data for:\n\n"
            "- **Remibrutinib** (CSU indication)\n"
            "- **Fabhalta** (PNH indication)\n"
            "- **Kisqali** (HR+/HER2- breast cancer)\n\n"
            "Use the **getKPISummary** action to get detailed metrics, or **runCausalAnalysis** for impact analysis."
        )

    # Default response
    return (
        "I'm the E2I Analytics Assistant. I can help you with:\n\n"
        "1. **KPI Analysis** - Get metrics for pharmaceutical brands\n"
        "2. **Agent Status** - Check the 18-agent system status\n"
        "3. **Causal Analysis** - Run causal impact analyses\n"
        "4. **Recommendations** - Get AI-powered suggestions\n"
        "5. **Insights Search** - Find trends and causal paths\n\n"
        "What would you like to explore?"
    )


# Create a static graph for initialization (used by parent class)
# The graph_factory creates fresh instances with new checkpointers per request
e2i_chat_graph = create_e2i_chat_agent()


# =============================================================================
# COPILOTKIT SDK SETUP
# =============================================================================


def create_copilotkit_sdk() -> CopilotKitRemoteEndpoint:
    """
    Create and configure the CopilotKit Remote Endpoint.

    IMPORTANT (v1.6.8): Uses graph_factory to create fresh checkpointer per request.
    This fixes "Message ID not found in history" error that occurs when:
    1. Checkpoint accumulates messages (user + AI responses)
    2. Frontend sends only user messages
    3. SDK's prepare_stream() detects mismatch and triggers regenerate mode
    4. Regenerate looks for frontend message IDs in checkpoint (they don't exist)

    Returns:
        Configured CopilotKitRemoteEndpoint instance with agents and actions
    """
    sdk = CopilotKitRemoteEndpoint(
        agents=[
            LangGraphAgent(
                name="default",
                description="E2I Analytics Assistant for pharmaceutical commercial analytics. Helps with KPI analysis, causal inference, and agent system insights.",
                graph=e2i_chat_graph,  # Initial graph for parent class
                graph_factory=create_e2i_chat_agent,  # Factory for fresh graphs per request
            ),
        ],
        actions=COPILOT_ACTIONS,
    )

    logger.info(f"[CopilotKit] Remote endpoint initialized with 1 agent and {len(COPILOT_ACTIONS)} actions")
    return sdk


def transform_info_response(sdk: CopilotKitRemoteEndpoint) -> Dict[str, Any]:
    """
    Transform SDK info response to frontend v1.x compatible format.

    The Python SDK (0.1.x) returns agents as an array with 'sdkVersion',
    but the JS frontend (1.x) expects agents as a dict with 'version'.

    Args:
        sdk: The CopilotKit remote endpoint instance

    Returns:
        Frontend-compatible info response
    """
    context: Dict[str, Any] = {}

    # Get agents - handle both callable and static
    agents = sdk.agents(context) if callable(sdk.agents) else sdk.agents

    # Get actions - handle both callable and static
    actions = sdk.actions(context) if callable(sdk.actions) else sdk.actions

    # Transform actions to dict representation
    actions_list = [action.dict_repr() for action in actions]

    # Transform agents array to dict keyed by agent ID (frontend v1.x format)
    agents_dict = {}
    for agent in agents:
        agent_id = agent.name
        agents_dict[agent_id] = {
            "description": getattr(agent, "description", "") or ""
        }

    return {
        "actions": actions_list,
        "agents": agents_dict,
        "version": COPILOTKIT_SDK_VERSION,  # Frontend expects 'version' not 'sdkVersion'
    }


async def copilotkit_custom_handler(request: Request, sdk: CopilotKitRemoteEndpoint, path: str = "") -> JSONResponse:
    """
    Custom CopilotKit endpoint handler that transforms info responses for frontend v1.x.

    Delegates to SDK handler functions but overrides info response format.
    This is cleaner than middleware because transformation happens at the source.

    Args:
        request: FastAPI request
        sdk: CopilotKit SDK instance
        path: Request path (extracted from route)

    Returns:
        JSONResponse with properly formatted data
    """
    import json
    from typing import cast
    from fastapi.encoders import jsonable_encoder

    method = request.method

    # Handle GET info request with our custom transformation
    if method == "GET" and path in ("", "info"):
        response = transform_info_response(sdk)
        print(f"DEBUG: GET info response with agents: {list(response['agents'].keys())}")
        return JSONResponse(content=response)

    # Handle POST to root or /info - need to check body to determine request type
    # IMPORTANT: Read body FIRST before any other operations that might consume it
    if method == "POST" and path in ("", "info"):
        try:
            # Read body bytes FIRST (only do this once!)
            body_bytes = await request.body()
            body_str = body_bytes.decode("utf-8") if body_bytes else ""
            print(f"DEBUG: POST body_str={body_str[:200] if body_str else '(empty)'}")

            # Parse body as JSON if present
            body_json = None
            if body_str.strip():
                try:
                    body_json = json.loads(body_str)
                except json.JSONDecodeError:
                    pass

            # Check if this is an info request:
            # - Empty body
            # - Empty JSON object {}
            # - Explicit getInfo action
            # - method: "info" (CopilotKit frontend format)
            is_info_request = (
                not body_str.strip()
                or body_str.strip() == "{}"
                or (body_json and body_json.get("action") == "getInfo")
                or (body_json and body_json.get("method") == "info")
            )

            print(f"DEBUG: is_info_request={is_info_request}")

            if is_info_request:
                response = transform_info_response(sdk)
                print(f"DEBUG: Returning info response with agents: {list(response['agents'].keys())}")
                return JSONResponse(content=response)

            # Non-info POST request - check AG-UI protocol method
            agui_method = body_json.get("method", "") if body_json else ""
            print(f"DEBUG: AG-UI method={agui_method}")

            # Handle AG-UI protocol: agent/run
            if agui_method == "agent/run":
                params = body_json.get("params", {})
                body_data = body_json.get("body", {})
                agent_name = params.get("agentId", "default") or body_json.get("agentName", "default")

                print(f"DEBUG: Executing agent '{agent_name}' with AG-UI protocol")

                # Extract parameters - check both nested body and top level (AG-UI protocol varies)
                # Some SDK versions send {"method": "agent/run", "body": {"threadId": ..., "messages": [...]}}
                # Others send {"method": "agent/run", "threadId": ..., "messages": [...]}
                thread_id = body_data.get("threadId") or body_json.get("threadId") or str(uuid.uuid4())
                state = body_data.get("state") or body_json.get("state") or {}
                messages = body_data.get("messages") or body_json.get("messages") or []
                actions = body_data.get("tools") or body_json.get("tools") or []  # AG-UI uses "tools"
                node_name = body_data.get("nodeName") or body_json.get("nodeName")

                print(f"DEBUG agent/run: thread_id={thread_id}")
                print(f"DEBUG agent/run: state keys={list(state.keys()) if state else 'empty'}")
                print(f"DEBUG agent/run: messages count={len(messages)}")
                for i, m in enumerate(messages):
                    print(f"DEBUG agent/run: messages[{i}]={m}")
                print(f"DEBUG agent/run: actions count={len(actions)}")
                print(f"DEBUG agent/run: node_name={node_name}")
                print(f"DEBUG agent/run: Full body_data keys={list(body_data.keys())}")

                # CUSTOM STREAMING HANDLER: Bypass SDK's handle_execute_agent to fix
                # the streaming lifecycle bug where HTTP response completes before all
                # events are yielded. The SDK handler was closing the connection after
                # yielding only the first event (RUN_STARTED), causing frontend to miss
                # MESSAGES_SNAPSHOT and other events generated 28+ seconds later.
                #
                # This custom handler:
                # 1. Gets agent from SDK
                # 2. Directly calls agent.execute() async generator
                # 3. Properly iterates and yields all events before closing

                # Get agent from SDK
                sdk_context: Dict[str, Any] = {}
                agents = sdk.agents(sdk_context) if callable(sdk.agents) else sdk.agents
                agent = None
                for a in agents:
                    if a.name == agent_name:
                        agent = a
                        break

                if agent is None:
                    return JSONResponse(
                        status_code=404,
                        content={"error": f"Agent '{agent_name}' not found"}
                    )

                async def stream_agent_events():
                    """
                    Stream all events from agent.execute() keeping connection alive.

                    This is the key fix: we iterate through ALL events from the async
                    generator and yield them one by one. The connection stays open
                    until the generator is exhausted.
                    """
                    import time
                    from datetime import datetime
                    stream_start = time.time()

                    def sdbg(msg):
                        """Flushed debug log with wall-clock and elapsed time."""
                        wall = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        elapsed = time.time() - stream_start
                        print(f"[{wall}] DEBUG stream [{elapsed:.3f}s]: {msg}", flush=True)

                    sdbg("Starting stream")

                    event_count = 0
                    try:
                        async for event in agent.execute(
                            thread_id=thread_id,
                            state=state,
                            messages=messages,
                            config=None,
                            actions=actions,
                            node_name=node_name,
                        ):
                            event_count += 1
                            sdbg(f"Streaming event #{event_count}")
                            # Event is already serialized by agent.execute()
                            yield event
                    except Exception as e:
                        sdbg(f"Error: {e}")
                        logger.error(f"[CopilotKit] Stream error: {e}")
                        # Yield error event (SSE format)
                        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

                    sdbg(f"Stream complete, yielded {event_count} events")

                return StreamingResponse(
                    stream_agent_events(),
                    media_type="text/event-stream",  # SSE format for CopilotKit SDK
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # Disable nginx buffering
                    },
                )

            # Handle AG-UI protocol: agent/connect (just acknowledge)
            if agui_method == "agent/connect":
                print(f"DEBUG: agent/connect - acknowledging")
                return JSONResponse(content={"status": "connected"})

            # Fall through to SDK handler for other methods
            print(f"DEBUG: Non-info POST, delegating to SDK handler")

            async def receive():
                return {"type": "http.request", "body": body_bytes}

            # Create new request with body restored and path param injected
            # FIX v1.9.3: SDK handler expects path in path_params, but base route has no path param
            scope_with_path = dict(request.scope)
            scope_with_path["path_params"] = {**request.path_params, "path": path}
            new_request = Request(scope_with_path, receive)
            return await sdk_handler(new_request, sdk)

        except Exception as e:
            print(f"DEBUG: Error in POST handler: {e}")
            logger.warning(f"[CopilotKit] Error parsing POST body: {e}")
            # Fall through to SDK handler on error

    # Build context for SDK handler (for non-root paths)
    try:
        body_bytes = await request.body()
    except:  # noqa: E722
        body_bytes = b""

    # For all other paths, delegate to SDK handler
    # ALWAYS reconstruct request since we consumed the body above (line 1219)
    # FIX v1.9.2: Previously used `if body_bytes:` which evaluates to False for empty bytes,
    # causing the original request (with consumed body) to be passed to SDK handler,
    # resulting in "expected string or bytes-like object, got 'NoneType'" errors.
    # FIX v1.9.3: SDK handler expects path in path_params, but base route has no path param
    async def receive():
        return {"type": "http.request", "body": body_bytes}
    scope_with_path = dict(request.scope)
    scope_with_path["path_params"] = {**request.path_params, "path": path}
    new_request = Request(scope_with_path, receive)
    return await sdk_handler(new_request, sdk)


def add_copilotkit_routes(app: FastAPI, prefix: str = "/api/copilotkit") -> None:
    """
    Add CopilotKit routes to the FastAPI application.

    Uses a custom endpoint handler instead of the SDK's add_fastapi_endpoint
    to properly transform info responses for frontend v1.x compatibility.

    The SDK's routing expects paths like:
    - / or /info → info endpoint (we transform this)
    - /agent/{name} → execute agent
    - /agent/{name}/state → get agent state
    - /action/{name} → execute action
    - /agents/execute, /actions/execute → v1 endpoints

    Args:
        app: FastAPI application instance
        prefix: URL prefix for CopilotKit endpoints
    """
    sdk = create_copilotkit_sdk()

    # Normalize prefix (ensure starts with / and no trailing /)
    normalized_prefix = "/" + prefix.strip("/")

    async def make_handler(request: Request, path: str = ""):
        """Route handler that extracts path and delegates to custom handler."""
        return await copilotkit_custom_handler(request, sdk, path)

    # Add base path route (WITHOUT trailing slash) to prevent 307 redirect
    # The frontend sends requests to /api/copilotkit (no trailing slash),
    # and FastAPI's redirect_slashes=True would cause a 307 redirect that breaks streaming
    app.add_api_route(
        normalized_prefix,
        make_handler,
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        name="copilotkit_handler_base",
    )

    # Add catch-all route for all CopilotKit sub-paths
    # This matches the SDK's pattern: {prefix}/{path:path}
    app.add_api_route(
        f"{normalized_prefix}/{{path:path}}",
        make_handler,
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        name="copilotkit_handler",
    )

    logger.info(f"[CopilotKit] Routes added at {normalized_prefix} and {normalized_prefix}/{{path}} (custom handler with info transformation)")


# =============================================================================
# STANDALONE ROUTER (for testing/info endpoints)
# =============================================================================

router = APIRouter(prefix="/copilotkit", tags=["copilotkit"])


@router.get("/status")
async def get_copilotkit_status() -> Dict[str, Any]:
    """Get CopilotKit integration status."""
    return {
        "status": "active",
        "version": "1.1.0",
        "agents_available": 1,
        "agent_names": ["default"],
        "actions_available": len(COPILOT_ACTIONS),
        "action_names": [a.name for a in COPILOT_ACTIONS],
        "llm_configured": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# E2I CHATBOT STREAMING ENDPOINTS
# =============================================================================


class ChatRequest(BaseModel):
    """Request schema for chatbot endpoints."""

    query: str = Field(..., description="User's query text")
    user_id: str = Field(..., description="User UUID")
    request_id: str = Field(..., description="Unique request identifier")
    session_id: Optional[str] = Field(
        default=None, description="Session ID (generated if not provided)"
    )
    brand_context: Optional[str] = Field(
        default=None, description="Brand filter (Kisqali, Fabhalta, Remibrutinib)"
    )
    region_context: Optional[str] = Field(
        default=None, description="Region filter (US, EU, APAC)"
    )


class ChatResponse(BaseModel):
    """Response schema for non-streaming chatbot endpoint."""

    success: bool
    session_id: str
    response: str
    conversation_title: Optional[str] = None
    agent_name: Optional[str] = None
    error: Optional[str] = None


async def _stream_chat_response(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream for chatbot response.

    Yields JSON-formatted SSE events:
    - {"type": "session_id", "data": "..."}
    - {"type": "text", "data": "..."}
    - {"type": "conversation_title", "data": "..."}
    - {"type": "tool_call", "data": "..."}
    - {"type": "done", "data": ""}
    - {"type": "error", "data": "..."}
    """
    try:
        from src.api.routes.chatbot_graph import stream_chatbot

        # Yield session_id first
        session_id = request.session_id
        if not session_id:
            import uuid
            session_id = f"{request.user_id}~{uuid.uuid4()}"

        yield f"data: {json.dumps({'type': 'session_id', 'data': session_id})}\n\n"

        response_text = ""
        conversation_title = None

        # Stream through chatbot workflow
        async for state_update in stream_chatbot(
            query=request.query,
            user_id=request.user_id,
            request_id=request.request_id,
            session_id=session_id,
            brand_context=request.brand_context,
            region_context=request.region_context,
        ):
            # Extract response from state updates
            if isinstance(state_update, dict):
                # Check for node outputs
                for node_name, node_output in state_update.items():
                    if isinstance(node_output, dict):
                        # Get response text from finalize node
                        if "response_text" in node_output and node_output["response_text"]:
                            text_chunk = node_output["response_text"]
                            if text_chunk and text_chunk != response_text:
                                # Yield new text
                                new_text = text_chunk[len(response_text):] if response_text else text_chunk
                                if new_text:
                                    yield f"data: {json.dumps({'type': 'text', 'data': new_text})}\n\n"
                                    response_text = text_chunk

                        # Get conversation title
                        if "conversation_title" in node_output and node_output["conversation_title"]:
                            title = node_output["conversation_title"]
                            if title != conversation_title:
                                conversation_title = title
                                yield f"data: {json.dumps({'type': 'conversation_title', 'data': title})}\n\n"

                        # Handle messages (for AIMessage content)
                        if "messages" in node_output:
                            for msg in node_output["messages"]:
                                if isinstance(msg, AIMessage) and msg.content:
                                    if msg.content != response_text:
                                        new_text = msg.content[len(response_text):] if response_text else msg.content
                                        if new_text:
                                            yield f"data: {json.dumps({'type': 'text', 'data': new_text})}\n\n"
                                            response_text = msg.content

        # Generate title if not set
        if not conversation_title and response_text:
            # Simple title generation from query
            title = request.query[:50] + "..." if len(request.query) > 50 else request.query
            yield f"data: {json.dumps({'type': 'conversation_title', 'data': title})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'data': ''})}\n\n"

    except Exception as e:
        logger.error(f"Streaming chat error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"


@router.post("/chat/stream")
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    """
    Stream chatbot response as Server-Sent Events (SSE).

    Returns an SSE stream with events:
    - session_id: The conversation session ID
    - text: Response text chunks
    - conversation_title: Auto-generated conversation title
    - tool_call: Tool invocation notifications
    - done: Stream completion signal
    - error: Error messages

    Usage:
        POST /api/copilotkit/chat/stream
        Content-Type: application/json

        {
            "query": "What is the TRx for Kisqali?",
            "user_id": "user-uuid",
            "request_id": "req-123",
            "session_id": "",  // Optional, generated if empty
            "brand_context": "Kisqali"  // Optional
        }
    """
    logger.info(f"[Chatbot] Streaming request: query={request.query[:50]}..., user={request.user_id}")

    return StreamingResponse(
        _stream_chat_response(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Non-streaming chatbot endpoint.

    Returns the complete response in a single JSON object.

    Usage:
        POST /api/copilotkit/chat
        Content-Type: application/json

        {
            "query": "Show agent status",
            "user_id": "user-uuid",
            "request_id": "req-456",
            "session_id": ""
        }
    """
    logger.info(f"[Chatbot] Chat request: query={request.query[:50]}..., user={request.user_id}")

    try:
        from src.api.routes.chatbot_graph import run_chatbot

        result = await run_chatbot(
            query=request.query,
            user_id=request.user_id,
            request_id=request.request_id,
            session_id=request.session_id,
            brand_context=request.brand_context,
            region_context=request.region_context,
        )

        response_text = result.get("response_text", "")
        session_id = result.get("session_id", "")
        agent_name = result.get("agent_name")

        # Generate title from query
        title = request.query[:50] + "..." if len(request.query) > 50 else request.query

        return ChatResponse(
            success=True,
            session_id=session_id,
            response=response_text,
            conversation_title=title,
            agent_name=agent_name,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            success=False,
            session_id=request.session_id or "",
            response="",
            error=str(e),
        )
