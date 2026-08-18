"""
CopilotKit Integration Router
=============================

Provides CopilotKit backend runtime for the E2I Chat Sidebar.
Exposes backend actions for querying KPIs, running analyses,
and interacting with the E2I agent system.

Author: E2I Causal Analytics Team
Version: 1.31.0

Changelog:
    1.31.0 - Fixed the frontend-action follow-up run 400 ("This model does not
             support assistant message prefill"). After the client executes a
             frontend action (e.g. renderKpiTrend), CopilotKit starts a follow-up
             run whose message list ends with the action result as a role-"tool"
             message so the model can narrate it. The execute() bridge flattened
             every non-user dict into a plain AssistantMessage and dropped
             content-less assistant tool-call turns, so the conversation reached
             Anthropic ending on an assistant message → 400 → the ⚠️ backend-error
             fallback (and the raw tool-result JSON echoed into the chat). Fix:
             _execute_bridge_agui_messages converts role-"tool" dicts (and
             LangChain ToolMessages) to AG-UI ToolMessages carrying tool_call_id,
             and preserves assistant toolCalls so tool_use/tool_result stay
             paired. Also corrected the Inline Charts prompt section: kpiId
             aliases now match the frontend alias map, and per-brand-only KPIs
             (nbrx, trx_share) require a brand argument.
    1.30.0 - Wired frontend actions (CopilotKit generative UI) into the chat agent —
             inline KPI trend charts now work end-to-end. The frontend has had the
             complete chart pipeline since renderKpiTrend shipped (useCopilotAction +
             Recharts KpiTrendChart + CustomAssistantMessage.generativeUI()), and the
             delivery chain (agent/run body tools → execute(actions=...) →
             RunAgentInput.tools → langgraph_default_merge_state →
             state["copilotkit"]["actions"]) was verified intact in the SDK source —
             but E2IAgentState had no "copilotkit" channel, so LangGraph dropped the
             key and chat_node bound only the backend tools: the model could never
             see the action. Fix: declare the channel, bind the riding actions
             (_frontend_action_schemas — accepts JSON-schema or CopilotKit
             parameter-array formats, skips backend-shadowing names), route
             frontend-action-only turns to END for client-side execution
             (_route_after_chat), strip frontend calls from mixed turns so ToolNode
             never sees an unknown tool (_strip_frontend_calls_when_mixed), and teach
             the system prompt when to call renderKpiTrend.
    1.29.0 - Fixed ghost empty-args tool calls minted by the streaming accumulator.
             Root cause: Anthropic streams messages as content blocks, so leading
             text shifts the first tool_use block to index 1. tool_call_chunks carry
             that provider index, but the parallel chunk.tool_calls entries carry NO
             index — the accumulator invented one from list position (0), the args
             merge then missed, and a duplicate entry shipped to ToolNode with {}
             args ("Error invoking tool 'kpi_calculate_tool' with kwargs {}" rendered
             raw in chat). Fix: _accumulate_tool_call_event treats tool_call_chunks
             as authoritative (chunk.tool_calls is a fallback merged by id) and
             _finalize_tool_calls collapses duplicate ids, preferring the entry that
             actually received args. Also: build_synthesis_prompt now forbids
             deriving baselines from overlapping windows and surfaces
             coverage_warning fields (2026-07-07 session review).
    1.28.0 - Fixed follow-up questions losing conversation context in tool-using turns.
             Root cause: synthesize_node is a separate LLM call from chat_node and its
             prompt contained only the current question + tool artifacts — no prior
             turns. chat_node (full history) picked the right tool args for follow-ups
             like "is that above baseline?", but the synthesizer then answered
             "I'm missing the preceding conversation". Fix: _extract_synthesis_history
             rebuilds the text-only transcript before the current question (capped at
             12 turns / 2000 chars each) and build_synthesis_prompt frames it ahead of
             the question with an explicit resolve-references instruction.
    1.27.0 - FIXED ROOT CAUSE of duplicate messages in tool-using queries.
             Problem: ag_ui_langgraph automatically emits TEXT_MESSAGE_* events when
             it detects LLM streaming from synthesize_node. Our code ALSO converted
             CUSTOM copilotkit_manually_emit_message events to TEXT_MESSAGE_* events.
             This created two TEXT_MESSAGE lifecycles for the same response.
             Fix: Detect when ag_ui_langgraph emits TEXT_MESSAGE_START (has rawEvent),
             set ag_ui_handling_stream=True, and skip CUSTOM event conversion when set.
             This ensures only ONE message lifecycle per user query.
    1.26.1 - Upgraded diagnostic logs to ERROR level for visibility in journalctl.
             INFO level logs were not appearing in systemd journal output.
    1.26.0 - Added comprehensive logging for duplicate message lifecycle detection.
             When TEXT_MESSAGE_START is emitted multiple times for the same user query,
             logs a WARNING with previous message IDs. This helps diagnose root cause
             of duplicate messages appearing in chat. Also logs all terminal events
             (RUN_FINISHED, RUN_ERROR) with streaming state context.
    1.25.1 - Strengthened tool_call detection to prevent race condition when
             streaming tool calls (name arrives before args in separate chunks).
             Now checks both response.tool_calls and accumulated_tool_calls.
    1.24.0 - Fixed multiple action bars appearing during chat response loading.
             Root cause: When LLM decides to use tools, chat_node was emitting
             content chunks immediately during streaming (e.g., "Let me look that up..."),
             creating TEXT_MESSAGE #1. Then synthesize_node emitted the actual response,
             creating TEXT_MESSAGE #2. Each message got its own action bar.
             Fix: Buffer content chunks during streaming instead of emitting immediately.
             After streaming completes, only emit buffered content if NO tool calls.
             If tool calls are present, don't emit - synthesize_node handles final response.
             This ensures only ONE action bar appears per user query.
    1.23.0 - Fixed action buttons appearing multiple times during streaming.
             Root cause: Each copilotkit_emit_message() call emitted complete
             TEXT_MESSAGE lifecycle (START/CONTENT/END) with new message_id,
             causing frontend to treat each chunk as a complete message.
             Fix: Track streaming state in execute() to emit:
             - TEXT_MESSAGE_START once at beginning (with stable message_id)
             - TEXT_MESSAGE_CONTENT for each chunk (same message_id)
             - TEXT_MESSAGE_END once when streaming ends or generator completes
             This ensures proper progress indicator display during streaming.
    1.22.0 - Implemented true token-by-token streaming for chat responses.
             Changed chat_node from ainvoke() to astream() for LLM calls.
             Each content chunk is now emitted via copilotkit_emit_message() immediately,
             enabling real-time text streaming in the CopilotKit frontend.
             Tool calls are still accumulated and handled after streaming completes.
             This enables users to see responses build up word-by-word instead of
             waiting for the complete response before display.
    1.21.5 - Fixed tool message ID being string "None" in MESSAGES_SNAPSHOT.
             Root cause: ToolMessage with id=None in Python serializes as "None" string.
             CopilotKit React SDK may fail validation on invalid ID format.
             Fix: For tool messages, use toolCallId with "tool-" prefix as the ID.
             For other messages, generate a proper UUID.
    1.21.4 - Fixed message content format and error event type.
             - Handle message as list of content blocks for copilotkit_manually_emit_message
             - Use RUN_ERROR event type instead of "error" (not valid AG-UI event type)
    1.21.3 - Fixed async/await issue: supabase-py is synchronous, not async.
             Created _persist_message_sync() helper to bypass broken async repository methods.
    1.21.2 - Fixed ChatbotMessageRepository instantiation: use `supabase_client` parameter
             instead of `client` to match BaseRepository.__init__() signature.
             Root cause: _get_chatbot_message_repository() was passing client=client but
             BaseRepository expects supabase_client parameter name.
             This caused messages not to persist despite session_id being available.
    1.21.1 - Fixed session_id not reaching chat_node() for message persistence.
             Root cause: AG-UI LangGraph's state management may not preserve custom
             fields from RunAgentInput.state when passing to graph nodes.
             Fix: Use Python contextvars to pass session_id across async boundaries.
             The context var is set in execute() and read in chat_node() as the
             primary source, with state and config.thread_id as fallbacks.
    1.21.0 - Added message persistence to Supabase chatbot_messages table.
             All user messages, assistant responses, tool calls, and synthesized responses
             are now persisted using ChatbotMessageRepository. This enables:
             - Feedback collection (P7.2) via message_id foreign key
             - Conversation history tracking
             - Agent performance analytics
             Session ID is derived from frontend thread_id for conversation continuity.
    1.20.1 - Fixed null error field in tool messages within MESSAGES_SNAPSHOT.
             Root cause: AG-UI SDK's AGUIToolMessage sets error=null by default.
             CopilotKit React SDK (v1.50.1) Zod validation expects error to be:
               - a string (for tool errors), OR
               - absent/undefined (for successful tool calls)
             When error is null, Zod throws:
               "Expected string, received null" at path ["messages", 2, "error"]
             Fix: Delete the error key entirely when it's null for tool messages.
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

import contextvars
import json
import logging
import operator
import os
import time
import uuid
from datetime import datetime, timezone
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    cast,
)

from ag_ui.core import AssistantMessage as AGUIAssistantMessage
from ag_ui.core import FunctionCall as AGUIFunctionCall
from ag_ui.core import RunAgentInput
from ag_ui.core import ToolCall as AGUIToolCall
from ag_ui.core import ToolMessage as AGUIToolMessage
from ag_ui.core import UserMessage as AGUIUserMessage
from copilotkit import Action as CopilotAction
from copilotkit import CopilotKitRemoteEndpoint
from copilotkit.integrations.fastapi import (
    handler as sdk_handler,
)
from copilotkit.langgraph import copilotkit_emit_message, copilotkit_emit_state
from copilotkit.langgraph_agui_agent import LangGraphAGUIAgent as _LangGraphAGUIAgent
from copilotkit.sdk import COPILOTKIT_SDK_VERSION
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from src.agents.factory import build_agent_roster_block
from src.agents.multi_faceted import is_multi_faceted_topic_count
from src.api.dependencies.auth import (
    TEST_USER,
    TESTING_MODE,
    AuthError,
    get_user_brands,
    is_cross_brand_admin,
    require_viewer,
    verify_supabase_token,
)
from src.api.middleware.tracing import get_request_id  # Phase 1 G08
from src.api.routes.chatbot_tools import E2I_CHATBOT_TOOLS, set_raw_user_query
from src.api.routes.synthesis_guard import (
    build_superlative_correction,
    find_superlative_contradictions,
)
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.api.utils.sse_keepalive import with_sse_keepalive
from src.kpi.synthetic_mode import kpi_include_synthetic, resolve_kpi_query_id
from src.utils.llm_attribution import (
    drain_run_usage,
    get_attribution,
    set_authenticated_user,
    set_chat_attribution,
)
from src.utils.llm_content import normalize_llm_content
from src.utils.llm_factory import MODEL_MAPPINGS, get_chat_llm, get_llm_provider
from src.utils.redaction import redact_query
from src.utils.tool_evidence import evidence_tool_count

logger = logging.getLogger(__name__)

# Context variable for passing session_id across async boundaries
# This is used because AG-UI LangGraph may not preserve custom state fields
_session_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "copilotkit_session_id", default=None
)

# Per-run discriminator for frontend_message_id stamping: the session key is
# the conversation threadId, so overlapping streams in the same conversation
# could otherwise cross-stamp identical-content assistant rows.
_run_id_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "copilotkit_run_id", default=None
)


def _latest_user_text(messages: Any) -> str:
    """Verbatim text of the most recent user message (dict or LC shapes).

    #1698: stashed via ``set_raw_user_query`` so tool-side honesty accounting
    sees the ask as the user typed it, not just the model's rewritten tool
    args. Non-string content (multimodal parts) yields "" — the side channel
    stays empty rather than guessing.
    """
    for msg in reversed(messages or []):
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                content = msg.get("content")
                return content if isinstance(content, str) else ""
        elif getattr(msg, "type", None) == "human":
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else ""
    return ""


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

    event_type = event_dict.get("type", "")

    # CRITICAL FIX (v1.19.0): Ensure timestamp and source on ALL events
    # CopilotKit React SDK (v1.50.1) Zod validation requires these on every event type,
    # including TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, TEXT_MESSAGE_END,
    # STATE_SNAPSHOT, MESSAGES_SNAPSHOT, and any other AG-UI events.
    if event_dict.get("timestamp") is None:
        event_dict["timestamp"] = int(time.time() * 1000)

    if event_dict.get("source") is None:
        event_dict["source"] = "e2i-copilot"

    # CRITICAL FIX (v1.24.0): Fix null parentMessageId on ANY event
    # ag_ui_langgraph emits parentMessageId=null on TEXT_MESSAGE_START and other events.
    # CopilotKit frontend Zod schema requires it to be a string, not null.
    # Apply universally since event type strings may vary (SCREAMING_SNAKE vs PascalCase).
    if "parentMessageId" in event_dict and event_dict["parentMessageId"] is None:
        event_dict["parentMessageId"] = ""

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
                # CRITICAL FIX (v1.20.1): Remove null error field from ANY message
                # CopilotKit React SDK (v1.50.1) Zod validation expects error to be:
                #   - a string (for errors), OR
                #   - absent/undefined (for successful operations)
                # AG-UI SDK sets error=null by default on tool messages, causing:
                #   "Expected string, received null" at path ["messages", N, "error"]
                # Fix: Delete the error key entirely when it's null on ANY message type
                if "error" in msg and msg["error"] is None:
                    del msg["error"]
                # CRITICAL FIX (v1.21.5): Fix tool message ID being string "None"
                # When ToolMessage has id=None in Python, it serializes as "None" string
                # which can cause frontend validation issues. Use toolCallId as fallback.
                if msg.get("id") in (None, "None", "null"):
                    # For tool messages, use toolCallId as the ID
                    if msg.get("role") == "tool" and msg.get("toolCallId"):
                        msg["id"] = f"tool-{msg['toolCallId']}"
                    else:
                        # Generate a new UUID for other messages with invalid ID
                        msg["id"] = f"msg-{uuid.uuid4()}"
                # CRITICAL FIX (v1.24.0): Fix null parentMessageId in snapshot messages
                if "parentMessageId" in msg and msg["parentMessageId"] is None:
                    msg["parentMessageId"] = ""

    return event_dict


# =============================================================================
# SDK COMPATIBILITY: LangGraphAGUIAgent with execute() method
# =============================================================================
# The CopilotKit SDK (v0.1.74) has a bug: it enforces using LangGraphAGUIAgent
# but calls agent.execute() which doesn't exist on that class (only run() exists).
# This wrapper bridges the gap by adding execute() that delegates to run().


def _coerce_agui_tool_call(raw: Any) -> Optional[AGUIToolCall]:
    """Parse one tool call from any of the shapes that reach the bridge.

    Accepts the OpenAI/AG-UI nested shape ({"id", "function": {"name",
    "arguments"}}) and the flat LangChain shape ({"id", "name", "args"}).
    ``arguments`` may already be a JSON string or still a dict.
    """
    if not isinstance(raw, dict):
        return None
    fn = raw["function"] if isinstance(raw.get("function"), dict) else raw
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = fn.get("arguments", fn.get("args", {}))
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments)
        except (TypeError, ValueError):
            return None
    call_id = raw.get("id") or f"call_{uuid.uuid4().hex[:8]}"
    return AGUIToolCall(id=str(call_id), function=AGUIFunctionCall(name=name, arguments=arguments))


def _execute_bridge_agui_messages(messages: Optional[List[Any]]) -> List[Any]:
    """Convert the raw messages reaching execute() into AG-UI messages.

    Sources: AG-UI protocol dicts from the frontend — including the follow-up
    run after a frontend action, whose list ends with the role-"tool" action
    result — and LangChain message objects from SDK internals.

    Assistant tool-call turns and their tool results must stay paired:
    Anthropic requires every tool_result to reference a tool_use, and the
    conversation to end on a user-role turn (tool results ride user turns).
    Flattening either side into plain assistant text ends the conversation on
    an assistant message → 400 "assistant message prefill". A tool result
    whose tool_call_id is missing degrades to a user turn for the same reason
    (a dangling tool_result also 400s).
    """
    converted: List[Any] = []
    for msg in messages or []:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            msg_id = msg.get("id") or f"msg-{uuid.uuid4()}"
            if role == "tool":
                if not isinstance(content, str):
                    try:
                        content = json.dumps(content)
                    except (TypeError, ValueError):
                        content = str(content)
                tool_call_id = msg.get("toolCallId") or msg.get("tool_call_id")
                if tool_call_id:
                    converted.append(
                        AGUIToolMessage(id=msg_id, content=content, tool_call_id=str(tool_call_id))
                    )
                elif content:
                    logger.warning(
                        "[execute] tool message %s has no toolCallId; degrading to user turn",
                        msg_id,
                    )
                    converted.append(AGUIUserMessage(id=msg_id, content=f"[tool result] {content}"))
            elif role == "user":
                if content:
                    converted.append(AGUIUserMessage(id=msg_id, content=content))
            else:
                # assistant — and, matching prior behavior, any unknown role
                raw_calls = msg.get("toolCalls") or msg.get("tool_calls") or []
                tool_calls = [tc for tc in map(_coerce_agui_tool_call, raw_calls) if tc]
                if content or tool_calls:
                    converted.append(
                        AGUIAssistantMessage(
                            id=msg_id,
                            content=content or None,
                            tool_calls=tool_calls or None,
                        )
                    )
        elif hasattr(msg, "content") and hasattr(msg, "type"):
            msg_id = getattr(msg, "id", None) or f"msg-{uuid.uuid4()}"
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            if msg.type == "tool":
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id:
                    converted.append(
                        AGUIToolMessage(id=msg_id, content=content, tool_call_id=str(tool_call_id))
                    )
                elif content:
                    logger.warning(
                        "[execute] LangChain ToolMessage %s has no tool_call_id; degrading",
                        msg_id,
                    )
                    converted.append(AGUIUserMessage(id=msg_id, content=f"[tool result] {content}"))
            elif msg.type == "human":
                converted.append(AGUIUserMessage(id=msg_id, content=content))
            else:
                lc_calls = getattr(msg, "tool_calls", None) or []
                tool_calls = [tc for tc in map(_coerce_agui_tool_call, lc_calls) if tc]
                if content or tool_calls:
                    converted.append(
                        AGUIAssistantMessage(
                            id=msg_id,
                            content=content or None,
                            tool_calls=tool_calls or None,
                        )
                    )
    return converted


# Name of the graph's ToolNode (see create_e2i_chat_agent). Used for graph
# construction and routing only.
#
# NOTE (#1636): the stream filter no longer keys on this constant. It was
# decoupled when `_is_tool_internal_llm_event` moved to an allow-list —
# tool-internal chat-model streams do not actually report this node name
# (nested graphs report their INNERMOST node, and `"tools"` was measured 0
# times across a 51-turn run), so keying on it caught nothing. See
# `_ANSWER_NODE_NAMES` for the filter's actual contract.
_TOOL_NODE_NAME = "tools"

#: The ONLY nodes whose chat-model streams are answer text. Everything else is
#: control plane (#1636).
#:
#: #1547 suppressed the literal node name ``"tools"``. That was too narrow twice
#: over. LangGraph's ``astream_events`` propagates callbacks into NESTED graphs
#: invoked inside a tool and reports the INNERMOST node name, so a tool-internal
#: call does not surface as ``"tools"`` at all — it surfaces under whatever the
#: nested graph called that node. The orchestrator's intent classifier arrived as
#: ``"classify"``, matched neither the suppressed name nor the allowed ones, and
#: fell through the fail-open branch into the answer stream (eval 2026-08-15 turn
#: 2.1: the classifier's raw JSON delivered as the FIRST assistant message).
#:
#: Measured over all 51 turns of that run, ``langgraph_node`` on
#: ``on_chat_model_*`` events takes exactly three values — ``chat`` (1372),
#: ``synthesize`` (814), ``classify`` (6) — and ``"tools"`` appears ZERO times,
#: confirming the old literal match had stopped catching anything.
#:
#: Allow-listing closes the class: any nested-graph node, present or future, is
#: suppressed without needing to be enumerated first. Add a node here ONLY if it
#: genuinely produces answer text.
_ANSWER_NODE_NAMES = frozenset({"chat", "synthesize"})

#: LangGraph's ``NS_SEP`` — it joins each nesting level of
#: ``metadata.langgraph_checkpoint_ns`` (``graph|subgraph|subsubgraph``), so a
#: namespace containing it belongs to an event raised INSIDE a nested graph.
#:
#: Duplicated as a literal rather than imported: upstream defines it in the
#: private ``langgraph._internal._constants``, and importing a private module
#: into a request path trades a loud test failure for a hard ImportError on
#: upgrade. ``TestLangGraphCoupling`` in
#: ``test_copilotkit_nested_graph_stream_leak_1641.py`` pins the two together.
_CHECKPOINT_NS_SEPARATOR = "|"


def _is_nested_graph_event(checkpoint_ns: Any) -> bool:
    """True when ``checkpoint_ns`` shows the event came from a NESTED graph (#1641).

    Fails open on anything that is not a string: an absent or malformed
    namespace must never be read as evidence, preserving #1547's property that
    a legitimate stream is not silenced on missing information.
    """
    if not isinstance(checkpoint_ns, str):
        return False
    return _CHECKPOINT_NS_SEPARATOR in checkpoint_ns


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

    @staticmethod
    def _is_tool_internal_llm_event(event: Any) -> bool:
        """True for chat-model callback events raised INSIDE the tools node (#1547).

        LangGraph's ``astream_events`` propagates callbacks into LLM calls made
        *inside* tools (async contextvars), and ag_ui_langgraph's
        ``_handle_single_event`` translates EVERY ``on_chat_model_stream`` event
        into a TEXT_MESSAGE lifecycle. So when ``tool_composer_tool`` ran, its
        decompose/plan phases' ``ainvoke`` generations streamed raw planner JSON
        into the delivered answer (eval 2026-08-11 turn 2.6: two blobs, ~3,000
        chars, the second truncated mid-generation, before any prose — measured
        in raw_agui.jsonl: the leaked TEXT_MESSAGE lifecycles carry
        ``rawEvent.metadata.langgraph_node == "tools"``, the legitimate ones
        ``"chat"`` / ``"synthesize"``).

        Tool-internal model streams are machinery, never answer text.

        #1636 WIDENED THIS. The original matched positively on ``"tools"`` alone,
        which failed twice: ``astream_events`` reports the INNERMOST node of a
        nested graph, so a tool-internal call surfaces under the nested graph's
        own node name (the orchestrator classifier arrived as ``"classify"``) and
        ``"tools"`` was measured 0 times across a 51-turn run. Enumerating
        offenders one at a time is how the same defect reached a third surface.

        #1641 ADDED THE DEPTH TEST. The allow-list above was the right move, but
        it keys on an identifier that is NOT UNIQUE ACROSS GRAPHS. The
        orchestrator graph (``src/agents/orchestrator/graph.py``) also ends
        ``dispatch -> synthesize -> END``, so when ``orchestrator_tool`` ran
        inside this graph's tools node its dispatch summary surfaced as
        ``langgraph_node == "synthesize"``, matched the allow-list, and was
        delivered as the FIRST assistant message — a generic "# Strategic
        Insights Summary" claiming "Both analyses returned null results" ahead of
        the real answer (eval 2026-08-15 turn A.9-followup: 3472 chars of
        template, then 2552 chars of answer opening "Straight answer:").

        A nested graph's stream is machinery whatever its innermost node is
        called, and ``langgraph_checkpoint_ns`` records that nesting directly.

        BOTH RULES ARE LOAD-BEARING — neither subsumes the other, and each catches
        leaks the other misses (measured, not assumed):

        * name-only misses NESTED answer names — this issue, 32 chat-model events
          at ``tools:<id>|synthesize:<id>`` in the 2026-08-15 run;
        * depth-only misses DEPTH-0 machinery — the #1547 leak carried
          ``langgraph_node == "tools"`` with ``langgraph_checkpoint_ns ==
          "tools"`` (no separator, 4063 chars on 2026-08-11 turn 2.6), because a
          direct LLM call inside the tools node enters no subgraph; likewise the
          separately-invoked explainer graph reports ``assemble`` / ``reason`` /
          ``generate`` at depth 0.

        ABSENT metadata still fails open at BOTH levels — #1547's safety property
        that a legitimate stream is never silenced on missing information is
        preserved deliberately, so a metadata regression degrades to noise rather
        than to a mute chatbot. This is safe against blanking a turn only because
        every answer the outer graph produces is raised at depth 0; verified over
        all 51 turns of the 2026-08-15 run, each has non-zero depth-0 answer text.
        """
        if not isinstance(event, dict):
            return False
        if not str(event.get("event", "")).startswith("on_chat_model_"):
            return False
        metadata = event.get("metadata") or {}
        node = metadata.get("langgraph_node")
        if node is None:
            # Fail open on missing information (#1547 contract, unchanged).
            return False
        if node not in _ANSWER_NODE_NAMES:
            return True
        # An answer NAME is not proof of an answer ORIGIN: only this graph's own
        # nodes produce answer text, and those are always at depth 0 (#1641).
        return _is_nested_graph_event(metadata.get("langgraph_checkpoint_ns"))

    async def _handle_single_event(self, event: Any, state: Any) -> AsyncGenerator[str, None]:
        """Drop tool-internal chat-model events BEFORE AG-UI translation (#1547).

        Filtering here (rather than in execute()) means no TEXT_MESSAGE
        lifecycle is ever created for a tool-internal stream, so
        ``messages_in_process`` bookkeeping cannot be corrupted for the real
        chat/synthesize streams.
        """
        if self._is_tool_internal_llm_event(event):
            return
        async for translated in super()._handle_single_event(event, state):
            yield translated

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
        from datetime import datetime

        start_time = time.time()

        def dbg(msg):
            """Debug log with wall-clock and elapsed time."""
            wall = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            elapsed = time.time() - start_time
            logger.debug(f"[{wall}] execute [{elapsed:.3f}s]: {msg}")

        # CRITICAL FIX (v1.9.4): Generate a fresh thread_id to prevent regenerate mode.
        # The SDK's prepare_stream() triggers regenerate mode when:
        #   len(checkpoint_messages) > len(frontend_messages)
        # With a fresh thread_id, the checkpointer lookup returns empty state,
        # so the regenerate check (0 > N) is always False.
        original_thread_id = thread_id
        thread_id = str(uuid.uuid4())
        dbg(
            f"Using fresh thread_id={thread_id[:8]}... (original={original_thread_id[:8] if original_thread_id else 'None'}...)"
        )

        # Convert messages to the format expected by RunAgentInput.
        # Messages can come from:
        # 1. AG-UI protocol: dicts like {"role": "user", "content": "..."} —
        #    including the follow-up run after a frontend action, whose list
        #    ends with the role-"tool" action result the model must narrate
        # 2. SDK internals: LangChain message objects with .type and .content
        dbg(f"Converting {len(messages or [])} messages to AG-UI format")
        agui_messages = _execute_bridge_agui_messages(messages)
        dbg(f"Converted to {len(agui_messages)} AG-UI messages")

        # Extract last user message ID for parentMessageId in response events
        last_user_msg_id = ""
        for msg in reversed(messages or []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_msg_id = msg.get("id", "")
                break
            elif hasattr(msg, "type") and msg.type == "human":
                last_user_msg_id = getattr(msg, "id", "") or ""
                break

        # #1698: stash the verbatim latest user message so orchestrator_tool
        # can thread it past the model's lossy query rewrite into the cohort
        # criteria accounting. Same fallback-channel rationale as
        # _session_id_context below: tools only ever see the model's args.
        set_raw_user_query(_latest_user_text(messages))

        # Build RunAgentInput
        # Generate run_id if not provided (SDK doesn't always pass it)
        run_id = kwargs.get("run_id") or str(uuid.uuid4())

        # Add session_id to state for message persistence
        # Use original_thread_id (from frontend) as the persistent session identifier
        state_with_session = dict(state) if state else {}
        persistent_session_id = original_thread_id or thread_id
        state_with_session["session_id"] = persistent_session_id
        # run_id rides state for the same reason session_id does: neither
        # channel is trusted alone (AG-UI may drop custom state fields; the
        # context var may be lost across execution boundaries). Nodes pass
        # state's run_id into _persist_message_sync as the fallback.
        state_with_session["run_id"] = run_id

        # CRITICAL (v1.21.1): Also set session_id in context var for reliable cross-async access
        # AG-UI LangGraph may not preserve custom state fields, so use contextvars as fallback
        _session_id_context.set(persistent_session_id)
        _run_id_context.set(run_id)
        # Attribute this run's LLM usage to the chat user/session (admin
        # observability, spec 2026-07-12). Both capture hooks read this
        # contextvar; the user_id is derived from the session prefix and the
        # anonymous UUID maps to NULL — attribution is honest-only.
        set_chat_attribution(persistent_session_id, run_id)
        dbg(f"Set session_id in state and context var: {persistent_session_id[:20]}...")

        run_input = RunAgentInput(
            thread_id=thread_id,
            run_id=run_id,
            state=state_with_session,
            messages=agui_messages,  # type: ignore[arg-type]
            tools=actions,  # type: ignore[arg-type]  # CopilotKit actions become tools
            context=[],
            forwarded_props={"node_name": node_name} if node_name else {},
        )

        dbg("Created RunAgentInput, calling self.run()")

        # CRITICAL FIX (v1.6.8): Use fresh graph with new checkpointer to avoid
        # "Message ID not found in history" error. The SDK's prepare_stream()
        # triggers regenerate mode when checkpoint has more messages than input.
        # By using a fresh checkpointer, the checkpoint is always empty.
        original_graph = self.graph  # type: ignore[has-type]
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

        # FIX (v1.23.0): Track streaming state for proper TEXT_MESSAGE lifecycle
        # Previously, each chunk emitted START/CONTENT/END, causing action buttons
        # to appear multiple times. Now we track state to emit:
        # - START once at beginning of streaming
        # - CONTENT for each chunk (with same message_id)
        # - END once when streaming ends (non-streaming event arrives)
        streaming_message_id = None
        streaming_started = False

        # FIX (v1.26.0): Track message lifecycle to prevent duplicates
        # The duplicate messages bug occurs when multiple TEXT_MESSAGE lifecycles
        # are created for a single user query (e.g., one empty message + one with content).
        # This happens when a terminal event resets streaming_started prematurely.
        # We track completed message IDs to detect this condition and log debug info.
        completed_message_ids: list[str] = []
        message_lifecycle_count = 0

        # FIX (v1.27.0): Track if ag_ui_langgraph is handling the LLM stream
        # When ag_ui_langgraph detects LLM streaming (e.g., from synthesize node),
        # it emits TEXT_MESSAGE_* events directly. In this case, we should NOT also
        # convert CUSTOM copilotkit_manually_emit_message events to TEXT_MESSAGE,
        # as this creates duplicate message lifecycles (the root cause of the duplicate
        # messages bug in tool-using queries).
        ag_ui_handling_stream = False

        # Track (frontend messageId -> accumulated text) for every outgoing
        # TEXT_MESSAGE lifecycle — custom-converted AND native ag_ui — so the
        # mapping can be stamped onto the persisted rows after the stream
        # (feedback resolution then has a stable key instead of content
        # heuristics; see _stamp_frontend_message_ids).
        text_message_deltas: Dict[str, List[str]] = {}
        completed_text_messages: Dict[str, str] = {}

        try:
            async for event in self.run(run_input):
                event_count += 1
                event_type = (
                    getattr(event, "type", "unknown")
                    if hasattr(event, "type")
                    else type(event).__name__
                )
                # FIX (v1.26.0): Log every event for comprehensive debugging (ERROR level to ensure visibility)
                logger.error(
                    f"[CopilotKit] Event #{event_count}: type={event_type}, streaming_started={streaming_started}"
                )
                dbg(f"Yielding event #{event_count} type={event_type}")

                # Check if this is a CUSTOM event with copilotkit_manually_emit_message
                # If so, emit TEXT_MESSAGE events (mimicking what CopilotKit Runtime does)
                is_streaming_event = (
                    hasattr(event, "type")
                    and event.type == EventType.CUSTOM
                    and getattr(event, "name", None) == "copilotkit_manually_emit_message"
                )

                # FIX (v1.24.0): Detect state events that should NOT end streaming
                # State emissions (copilotkit_emit_state) are informational and should
                # not interrupt the text message lifecycle
                is_state_event = (
                    hasattr(event, "type")
                    and event.type == EventType.CUSTOM
                    and getattr(event, "name", None) == "copilotkit_emit_state"
                )

                # FIX (v1.24.0): Detect terminal events that SHOULD end streaming
                # Only RUN_FINISHED and RUN_ERROR indicate the end of a response
                event_type_str = ""
                if hasattr(event, "type"):
                    if hasattr(event.type, "value"):
                        event_type_str = str(event.type.value).upper()
                    else:
                        event_type_str = str(event.type).upper()
                is_terminal_event = event_type_str in ("RUN_FINISHED", "RUN_ERROR")

                # FIX (v1.26.0): Log all terminal events for debugging duplicates (ERROR level to ensure visibility)
                if is_terminal_event:
                    logger.error(
                        f"[CopilotKit] Terminal event received: {event_type_str}, streaming_started={streaming_started}, event_count={event_count}"
                    )

                # FIX (v1.25.0): Filter out TOOL_CALL_RESULT events
                # These contain raw JSON tool results that should NOT be rendered as messages
                # The synthesize_node will emit the formatted human-readable response via TEXT_MESSAGE
                # Without this filter, users see duplicate messages:
                # 1. Raw JSON: {"success": true, "query_type": "kpi", ...}
                # 2. Synthesized response: "Here is a summary of the recent TRx performance..."
                is_tool_result_event = event_type_str == "TOOL_CALL_RESULT"

                if is_tool_result_event:
                    dbg("Skipping TOOL_CALL_RESULT event (raw tool results should not be rendered)")
                    continue

                # FIX (v1.27.0): Detect when ag_ui_langgraph emits TEXT_MESSAGE_START
                # ag_ui_langgraph automatically converts LLM streaming to TEXT_MESSAGE events
                # These have rawEvent containing "on_chat_model_stream" from LangGraph callback
                # When we detect this, set flag to skip our CUSTOM event conversion (prevents duplicates)
                if hasattr(event, "type") and event.type == EventType.TEXT_MESSAGE_START:
                    ag_ui_handling_stream = True
                    logger.error(
                        "[CopilotKit] ag_ui_langgraph TEXT_MESSAGE_START detected - will skip CUSTOM event conversion"
                    )

                # FIX (v1.27.0): Reset flag when ag_ui_langgraph's stream ends
                if hasattr(event, "type") and event.type == EventType.TEXT_MESSAGE_END:
                    if ag_ui_handling_stream:
                        logger.error(
                            "[CopilotKit] ag_ui_langgraph TEXT_MESSAGE_END detected - stream complete"
                        )
                        # Don't reset here - keep flag set for rest of this run to prevent any late CUSTOM events

                # FIX (v1.24.0): End streaming only on terminal events, not state events
                # Previously ended on ANY non-streaming event, causing multiple message lifecycles
                # when state was emitted between content chunks
                if (
                    streaming_started
                    and not is_streaming_event
                    and not is_state_event
                    and is_terminal_event
                ):
                    current_ts = int(time.time() * 1000)
                    source = "e2i-copilot"
                    yield f"data: {json.dumps({'type': 'TEXT_MESSAGE_END', 'messageId': streaming_message_id, 'timestamp': current_ts, 'source': source})}\n\n"
                    event_count += 1
                    _track_text_message_event(
                        {"type": "TEXT_MESSAGE_END", "messageId": streaming_message_id},
                        text_message_deltas,
                        completed_text_messages,
                    )
                    # FIX (v1.26.0): Track completed messages for duplicate detection (ERROR level to ensure visibility)
                    if streaming_message_id is not None:
                        completed_message_ids.append(streaming_message_id)
                    logger.error(
                        f"[CopilotKit] TEXT_MESSAGE_END: message_id={streaming_message_id}, lifecycle_count={message_lifecycle_count}, event_type={event_type_str}"
                    )
                    dbg(
                        f"Ended streaming on terminal event {event_type_str}, message_id={streaming_message_id}"
                    )
                    streaming_started = False
                    streaming_message_id = None

                if is_streaming_event:
                    # FIX (v1.27.0): Skip CUSTOM event conversion if ag_ui_langgraph is handling stream
                    # ag_ui_langgraph already converts LLM streaming to TEXT_MESSAGE events, so our
                    # CUSTOM -> TEXT_MESSAGE conversion would create duplicate message lifecycles
                    if ag_ui_handling_stream:
                        dbg(
                            "Skipping CUSTOM copilotkit_manually_emit_message - ag_ui_langgraph handling stream"
                        )
                        continue

                    event_value = getattr(event, "value", {}) or {}
                    raw_message = event_value.get("message", "")

                    # FIX (v1.21.4): Handle message as list of content blocks
                    # copilotkit_manually_emit_message sends message as list: [{'text': '...', 'type': 'text', 'index': 0}]
                    # But delta field expects a string, not a list
                    if isinstance(raw_message, list):
                        # Extract text from all content blocks
                        message = "".join(
                            block.get("text", "") if isinstance(block, dict) else str(block)
                            for block in raw_message
                        )
                    else:
                        message = str(raw_message) if raw_message else ""

                    # CRITICAL FIX (v1.16.0): Use SCREAMING_SNAKE_CASE event types
                    # CopilotKit React SDK (v1.50.1) uses Zod validation that expects
                    # SCREAMING_SNAKE_CASE: TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT, etc.
                    #
                    # CRITICAL FIX (v1.19.0): Add timestamp and source to ALL events
                    current_ts = int(time.time() * 1000)
                    source = "e2i-copilot"

                    # FIX (v1.23.0): Track streaming state for proper message lifecycle
                    if not streaming_started:
                        # First chunk - generate message_id and emit START
                        message_lifecycle_count += 1
                        streaming_message_id = str(uuid.uuid4())
                        streaming_started = True
                        # FIX (v1.26.0): Log message lifecycle for duplicate detection (ERROR level to ensure visibility)
                        logger.error(
                            f"[CopilotKit] TEXT_MESSAGE_START: message_id={streaming_message_id}, lifecycle_count={message_lifecycle_count}, completed_count={len(completed_message_ids)}"
                        )
                        if message_lifecycle_count > 1:
                            logger.warning(
                                f"[CopilotKit] DUPLICATE LIFECYCLE DETECTED: This is lifecycle #{message_lifecycle_count}, previous completed: {completed_message_ids}"
                            )
                        yield f"data: {json.dumps({'type': 'TEXT_MESSAGE_START', 'messageId': streaming_message_id, 'parentMessageId': last_user_msg_id, 'role': 'assistant', 'timestamp': current_ts, 'source': source})}\n\n"
                        event_count += 1
                        _track_text_message_event(
                            {"type": "TEXT_MESSAGE_START", "messageId": streaming_message_id},
                            text_message_deltas,
                            completed_text_messages,
                        )
                        dbg(f"Started streaming message_id={streaming_message_id}")

                    # Emit CONTENT for this chunk (using consistent message_id)
                    yield f"data: {json.dumps({'type': 'TEXT_MESSAGE_CONTENT', 'messageId': streaming_message_id, 'delta': message, 'timestamp': current_ts, 'source': source})}\n\n"
                    event_count += 1
                    _track_text_message_event(
                        {
                            "type": "TEXT_MESSAGE_CONTENT",
                            "messageId": streaming_message_id,
                            "delta": message,
                        },
                        text_message_deltas,
                        completed_text_messages,
                    )

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
                        _track_text_message_event(
                            event_dict, text_message_deltas, completed_text_messages
                        )
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
                    _track_text_message_event(
                        event_dict, text_message_deltas, completed_text_messages
                    )
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
                    _track_text_message_event(
                        event_dict, text_message_deltas, completed_text_messages
                    )
                    yield f"data: {json.dumps(event_dict)}\n\n"
                else:
                    # Fallback - convert to string with SSE format
                    yield f"data: {str(event)}\n\n"
        finally:
            # FIX (v1.23.0): Emit TEXT_MESSAGE_END if streaming was in progress
            # This handles the case where the generator ends without a non-streaming event
            if streaming_started and streaming_message_id:
                current_ts = int(time.time() * 1000)
                source = "e2i-copilot"
                # Note: We can't yield from finally, so we need a different approach
                # The streaming_started flag will be checked and END emitted before any other event
                dbg(f"Streaming was active at generator end (message_id={streaming_message_id})")

            # Restore original graph if we swapped it
            if self._graph_factory:
                self.graph = original_graph
                dbg("Restored original graph")

        # FIX (v1.23.0): Emit final TEXT_MESSAGE_END if streaming was still active
        # This ensures the message is properly terminated even if generator ends
        if streaming_started and streaming_message_id:
            current_ts = int(time.time() * 1000)
            source = "e2i-copilot"
            yield f"data: {json.dumps({'type': 'TEXT_MESSAGE_END', 'messageId': streaming_message_id, 'timestamp': current_ts, 'source': source})}\n\n"
            event_count += 1
            _track_text_message_event(
                {"type": "TEXT_MESSAGE_END", "messageId": streaming_message_id},
                text_message_deltas,
                completed_text_messages,
            )
            dbg(f"Ended streaming at generator completion, message_id={streaming_message_id}")

        # Stamp (frontend messageId -> persisted row) mappings now that the
        # stream is done — the graph nodes have finished persisting by the
        # time the terminal event flowed through. Best-effort; the response
        # already reached the client.
        if completed_text_messages and persistent_session_id:
            _stamp_frontend_message_ids(
                persistent_session_id, completed_text_messages, run_id=run_id
            )

        dbg(f"Finished yielding {event_count} events")


# =============================================================================
# REPOSITORY HELPERS
# =============================================================================


async def _get_agent_registry_repository():
    """Get AgentRegistryRepository instance with an ASYNC Supabase client.

    AgentRegistryRepository.get_by_tier -> BaseRepository.get_many awaits
    ``query.execute()``, so the repo requires the ASYNC client (the sync
    ``get_supabase()`` client raised ``TypeError`` on ``await execute()``).
    The base constructor takes ``supabase_client=`` (issue #821; the prior
    ``client=`` kwarg raised TypeError -> None -> silent sample-agent fallback).
    """
    try:
        from src.memory.services.factories import get_async_supabase_client
        from src.repositories.agent_registry import AgentRegistryRepository

        client = await get_async_supabase_client()
        return AgentRegistryRepository(supabase_client=client) if client else None
    except Exception as e:
        logger.warning(f"Failed to get AgentRegistryRepository: {e}")
        return None


def _get_chatbot_message_repository():
    """Get ChatbotMessageRepository instance with Supabase client."""
    try:
        from src.api.dependencies.supabase_client import get_supabase
        from src.repositories.chatbot_message import ChatbotMessageRepository

        client = get_supabase()
        return ChatbotMessageRepository(supabase_client=client) if client else None
    except Exception as e:
        logger.warning(f"Failed to get ChatbotMessageRepository: {e}")
        return None


def _get_chatbot_conversation_repository():
    """Get ChatbotConversationRepository instance with Supabase client."""
    try:
        from src.api.dependencies.supabase_client import get_supabase
        from src.repositories.chatbot_conversation import ChatbotConversationRepository

        client = get_supabase()
        return ChatbotConversationRepository(supabase_client=client) if client else None
    except Exception as e:
        logger.warning(f"Failed to get ChatbotConversationRepository: {e}")
        return None


# Anonymous user ID for conversations without authentication
_ANONYMOUS_USER_ID = "00000000-0000-0000-0000-000000000000"


def _persist_message_sync(
    session_id: str,
    role: str,
    content: str,
    agent_name: Optional[str] = None,
    metadata: Optional[dict] = None,
    run_id: Optional[str] = None,
) -> Optional[dict]:
    """
    Persist a message to the database using synchronous Supabase client.

    NOTE: This bypasses the async repository method because supabase-py
    uses synchronous HTTP calls internally, making 'await' incompatible.

    run_id follows the same two-channel resilience ladder as session_id
    (context var preferred, state-sourced parameter as fallback): a row
    persisted without metadata.run_id can never be stamped with its
    frontend_message_id, because _stamp_frontend_message_ids filters on
    metadata->>run_id.
    """
    try:
        from src.api.dependencies.supabase_client import get_supabase

        client = get_supabase()
        if not client:
            logger.warning("[CopilotKit] No Supabase client for message persistence")
            return None

        row_metadata = dict(metadata or {})
        run_id = _run_id_context.get() or run_id
        if run_id and "run_id" not in row_metadata:
            # Run discriminator so post-stream stamping can scope its
            # content match to this run's rows (see
            # _stamp_frontend_message_ids).
            row_metadata["run_id"] = run_id

        message_data: dict[str, Any] = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "agent_name": agent_name,
            "metadata": row_metadata,
        }

        if role == "assistant":
            # Drain the run's token accumulator into the row (read-and-reset:
            # sums across a session's assistant rows never double-count).
            # None when nothing was captured — honest NULL, never fabricated.
            drained = drain_run_usage()
            if drained:
                message_data["tokens_used"] = drained.input_tokens + drained.output_tokens
                message_data["model_used"] = drained.last_model

        result = client.table("chatbot_messages").insert(message_data).execute()

        if result.data:
            logger.debug(f"_persist_message_sync: inserted message id={result.data[0].get('id')}")
            return result.data[0]  # type: ignore[no-any-return]
        return None
    except Exception as e:
        logger.error(f"[CopilotKit] _persist_message_sync failed: {e}")
        return None


def _track_text_message_event(
    event: dict,
    deltas: Dict[str, List[str]],
    completed: Dict[str, str],
) -> None:
    """Observe a TEXT_MESSAGE_* lifecycle event and accumulate message content.

    The SSE translation layer is the only place that knows which messageId the
    frontend will see for an assistant message (the graph nodes persist rows
    without it — the id is minted here or inside ag_ui_langgraph, never in the
    node). Feeding every outgoing TEXT_MESSAGE_* event through this tracker
    rebuilds (messageId -> full text) so the mapping can be stamped onto the
    persisted row after the stream. Handles both camelCase (AG-UI wire alias)
    and snake_case keys. Observation must never break the stream: malformed
    events are ignored.
    """
    try:
        etype = event.get("type")
        message_id = event.get("messageId") or event.get("message_id")
        if not message_id:
            return
        key = str(message_id)
        if etype == "TEXT_MESSAGE_START":
            deltas.setdefault(key, [])
        elif etype == "TEXT_MESSAGE_CONTENT":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.setdefault(key, []).append(delta)
        elif etype == "TEXT_MESSAGE_END":
            parts = deltas.pop(key, None)
            if parts:
                completed[key] = "".join(parts)
    except Exception:  # noqa: BLE001 — observation-only, never propagate
        pass


def _stamp_frontend_message_ids(
    session_id: str, messages: Dict[str, str], run_id: Optional[str] = None
) -> None:
    """Stamp metadata.frontend_message_id onto the persisted assistant rows.

    Runs after an SSE stream completes. Write-time matching is deterministic:
    the row carrying each completed message's text was written moments ago by
    this very run, so the newest unstamped exact-content match IS that row —
    older identical-content rows (already stamped or from prior runs) rank
    later. When run_id is given the match is additionally scoped to rows
    persisted by this run (metadata.run_id, attached in
    _persist_message_sync): the session key is the conversation threadId, so
    two overlapping streams in the same conversation could otherwise
    cross-stamp identical-content rows. The stamp gives /copilotkit/feedback
    a stable resolution key (the client's message_uuid) instead of content
    heuristics. Best-effort: any failure leaves feedback on its
    content-matching fallback.
    """
    try:
        from src.api.dependencies.supabase_client import get_supabase

        client = get_supabase()
        if not client:
            return
        query = (
            client.table("chatbot_messages")
            .select("id, content, metadata")
            .eq("session_id", session_id)
            .eq("role", "assistant")
        )
        if run_id:
            query = query.eq("metadata->>run_id", run_id)
        result = query.order("created_at", desc=True).limit(20).execute()
        rows = result.data or []
        for message_id, content in messages.items():
            if not content:
                continue
            for row in rows:
                if (row.get("content") or "") != content:
                    continue
                meta = row.get("metadata") or {}
                if meta.get("frontend_message_id"):
                    # Already mapped (identical content stamped for another
                    # messageId) — try the next-newest matching row.
                    continue
                meta["frontend_message_id"] = message_id
                client.table("chatbot_messages").update({"metadata": meta}).eq(
                    "id", row["id"]
                ).execute()
                row["metadata"] = meta
                break
    except Exception as e:
        logger.debug(f"[CopilotKit] frontend_message_id stamping skipped: {e}")


def _tool_names_from(entries: Any) -> List[str]:
    """Extract tool names from a persisted tool_calls/tool_results list.

    Three shapes appear in chatbot_messages: orchestrator-flow rows store
    top-level tool_calls keyed "tool_name" (chatbot_graph finalize_node) and
    tool_results keyed "tool"/"name", sidebar rows store metadata
    tool_results keyed "tool" and tool_calls keyed "name". Non-list input
    and nameless entries yield nothing.
    """
    if not isinstance(entries, list):
        return []
    names: List[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("tool") or entry.get("name") or entry.get("tool_name")
        if name:
            names.append(str(name))
    return names


def _classify_query_type(query: str) -> str:
    """
    Classify a user query into a query type for analytics.

    Returns one of the valid chatbot_query_type enum values:
    - kpi_inquiry
    - causal_analysis
    - agent_status
    - recommendation
    - experiment
    - prediction
    - drift_alert
    - general
    - multi_faceted

    Multi-faceted detection is delegated to
    ``src.agents.multi_faceted.is_multi_faceted_topic_count`` (issue #295)
    so the 5 topic-keyword groups have a single source of truth and a
    future change is observable in SSOT tests.
    """
    query_lower = query.lower()

    # Multi-faceted detection — delegated to SSOT (issue #295). The 5
    # topic-keyword groups (KPI/causal/predict/experiment/drift) live in
    # ``src/agents/multi_faceted.py::TOPIC_COUNT_KEYWORD_GROUPS``.
    if is_multi_faceted_topic_count(query):
        return "multi_faceted"

    # Single topic detection
    if any(
        kw in query_lower
        for kw in ["kpi", "trx", "nrx", "market share", "metric", "performance", "volume"]
    ):
        return "kpi_inquiry"
    if any(
        kw in query_lower for kw in ["causal", "impact", "effect", "intervention", "why", "cause"]
    ):
        return "causal_analysis"
    if any(kw in query_lower for kw in ["agent", "status", "tier", "health", "system"]):
        return "agent_status"
    if any(kw in query_lower for kw in ["recommend", "suggest", "should", "advice", "optimize"]):
        return "recommendation"
    if any(kw in query_lower for kw in ["experiment", "test", "ab test", "a/b", "trial"]):
        return "experiment"
    if any(kw in query_lower for kw in ["predict", "forecast", "future", "projection"]):
        return "prediction"
    if any(kw in query_lower for kw in ["drift", "shift", "degradation", "alert"]):
        return "drift_alert"

    return "general"


def _record_analytics_sync(
    session_id: str,
    query_type: str,
    response_time_ms: Optional[int] = None,
    tools_invoked: Optional[List[str]] = None,
    primary_agent: str = "copilotkit",
    error_occurred: bool = False,
    error_type: Optional[str] = None,
    orchestrator_used: bool = False,
    tool_composer_used: bool = False,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    """
    Record analytics to the database using synchronous Supabase client.

    This tracks usage patterns, performance, and tool usage for the chatbot.
    Used for monitoring, capacity planning, and optimization (P7.1).
    """
    try:
        from src.api.dependencies.supabase_client import get_supabase

        client = get_supabase()
        if not client:
            logger.debug("[CopilotKit] No Supabase client for analytics recording")
            return None

        analytics_data = {
            "session_id": session_id,
            "query_type": query_type,
            "response_time_ms": response_time_ms,
            "tools_invoked": tools_invoked or [],
            "primary_agent": primary_agent,
            "error_occurred": error_occurred,
            "error_type": error_type,
            "orchestrator_used": orchestrator_used,
            "tool_composer_used": tool_composer_used,
            "metadata": metadata or {},
            "response_completed_at": datetime.now(timezone.utc).isoformat()
            if response_time_ms
            else None,
        }

        # Remove None values for cleaner insert
        analytics_data = {k: v for k, v in analytics_data.items() if v is not None}

        result = client.table("chatbot_analytics").insert(analytics_data).execute()

        if result.data:
            logger.debug(f"[CopilotKit] Recorded analytics id={result.data[0].get('id')}")
            return result.data[0]  # type: ignore[no-any-return]
        return None
    except Exception as e:
        # Analytics should never block the main flow
        logger.debug(f"[CopilotKit] Analytics recording failed (non-critical): {e}")
        return None


def _evidence_tool_count(tool_results: List[Dict[str, Any]]) -> int:
    """Count tool results that actually carry evidence (#1257).

    E2I tools fail closed with a ``{"success": false, ...}`` JSON envelope
    that still becomes a ToolMessage — counting those toward the grounding
    bonus graded an all-errored turn 0.8 (the copilot surface maximum), and
    those rows persist as top-reward training examples. Only results not
    positively marked failed count; payloads without the envelope (or
    unparseable ones) ARE the evidence, not an error marker, so they count.

    The semantics live in ``src.utils.tool_evidence`` — extracted for #1458
    so chat_bridge's ``tool_grounded`` gate applies the exact same rule
    without importing this module's heavy SDK surface.
    """
    return evidence_tool_count(tool_results)


def _grade_copilot_turn(response: str, tool_count: int, synthesis_error: bool = False) -> float:
    """Grade one completed copilot turn on observable outcome quality (#1240).

    Mirrors the cognitive workflow's ``agent_reward`` derivation
    (``cognitive_rag_dspy._collect_training_signals``) with tool results as
    the evidence-board analog: base 0.5 for producing a synthesized response
    (keeps completed turns inside the GEPA fuel band, reward >= 0.5),
    + 0.2 * min(1, tool_results/4) evidence grounding, + 0.1 substantive
    length. A failed synthesis (raw tool-dump fallback served) forfeits the
    base — the response exists but is not a synthesis — leaving only the
    observable components. No response at all is 0.0.

    Calibration (codex-1240 M2, iter-2): the composite is intentionally NOT
    rescaled to [0, 1]. The copilot surface has no evidence-board/
    visualization analog, so its honest ceiling is 0.8 while the cognitive
    path reaches 1.0 — but every hard consumer of the shared
    ``rating_1to5 = 1 + 4*reward`` mapping is bottom-anchored
    (``avg_rating < 3.0`` pooled gate, per-agent ``num < 3`` negative
    counting, ``< 2.0`` severity), and the raw scale already agrees with the
    cognitive path there: 0.5 = acceptable synthesis = rating 3.0 on BOTH
    surfaces. A linear rescale (``/0.8``) would inflate exactly that consumed
    region — a mediocre ungrounded copilot turn (raw 0.5) would outscore an
    identical-quality cognitive turn 3.5 vs 3.0 and mask low-rating
    patterns in mixed pools. The unreachable top band (0.8–1.0) is the
    honest statement that this surface cannot exhibit those two extra
    quality axes; raw reward is preserved in signal metadata and the
    ``source_path`` marker below drives the source-aware aggregation
    (#1251: ``feedback_learner.rating_utils.rating_surface`` groups rating
    pools per surface so mixed pools can't mask a low surface).
    """
    if not response:
        return 0.0
    reward = 0.0 if synthesis_error else 0.5
    reward += 0.2 * min(1.0, tool_count / 4.0)
    reward += 0.1 if len(response) >= 200 else 0.0
    return reward


async def _collect_copilot_learning_signal(
    query: str,
    response: str,
    tool_names: List[str],
    conversation_id: Optional[str],
    synthesis_error: bool = False,
    evidence_tool_count: Optional[int] = None,
) -> None:
    """Persist one honestly graded learning signal for a copilot turn (#1240).

    Writes the same ``learning_signals`` row shape the cognitive workflow's
    Phase-4 Reflector produces (``signal_details`` carrying
    type/query/response/reward/metadata + ``domain_signal='dspy_signal'``,
    ``is_synthetic=false`` via DB default) — the ONLY substrate
    ``LearningSignalsFeedbackStore`` feeds the Tier-5 feedback learner. Only
    the ``agent`` (synthesis) component is graded: the copilot path runs no
    summarizer/investigator, and fabricating signals for components that
    never executed would be dishonest attribution. ``routed_agents`` credits
    ``copilotkit`` (the surface actually graded); the tools invoked are kept
    in metadata for finer-grained future grading.

    Best-effort: any failure is logged and swallowed — signal collection
    must never unwind into the chat response path.
    """
    try:
        # #1260: the surface marker is the shared constant the classifier
        # (rating_surface) keys on — a one-sided rename would silently
        # reclassify copilot rows into the cognitive pool. Lazy import per
        # this function's idiom (the learner package init is heavy).
        from src.agents.feedback_learner.rating_utils import COPILOT_SURFACE
        from src.memory import procedural_memory
        from src.memory.procedural_memory import LearningSignalInput

        # #1257: grade on EVIDENCE-BEARING tool results only — errored tools
        # (fail-closed envelopes) invoked but produced no grounding. The full
        # invocation list stays in metadata.tools_invoked for observability.
        evidence_count = len(tool_names) if evidence_tool_count is None else evidence_tool_count
        reward = _grade_copilot_turn(
            response, tool_count=evidence_count, synthesis_error=synthesis_error
        )
        metadata: Dict[str, Any] = {
            # agent attribution (NOT the surface marker — same string today,
            # different meaning: this names the responder, source_path below
            # routes the reward-surface split)
            "routed_agents": ["copilotkit"],
            "tools_invoked": list(tool_names),
            "evidence_tool_count": evidence_count,
            "conversation_id": conversation_id,
            "source_path": COPILOT_SURFACE,
        }
        if synthesis_error:
            metadata["synthesis_error"] = True
        signal = LearningSignalInput(
            signal_type="rating",
            signal_value=reward,
            is_training_example=True,
            training_input=query,
            training_output=response[:500],
            signal_details={
                "type": "agent",
                "query": query,
                "response": response[:500],
                "reward": reward,
                "feedback": None,
                "metadata": metadata,
                "domain_signal": "dspy_signal",
            },
        )
        # Late-bound module attribute so tests (and future writers) can patch
        # record_learning_signal at its home module.
        await procedural_memory.record_learning_signal(signal=signal, cycle_id=None)
        logger.debug(
            f"[CopilotKit] Collected learning signal (reward={reward:.2f}, "
            f"tools={len(tool_names)}, conversation={conversation_id})"
        )
    except Exception as e:  # noqa: BLE001 - best-effort by design
        logger.warning(f"[CopilotKit] Learning-signal collection failed (non-critical): {e}")


async def _ensure_conversation_exists(session_id: str) -> bool:
    """
    Ensure a conversation record exists for the given session_id.
    Creates one attributed to the JWT-verified owner if it doesn't exist
    (falls back to the anon sentinel only when genuinely unattributed).

    Args:
        session_id: The session/thread ID (must be a valid UUID)

    Returns:
        True if conversation exists or was created, False on failure
    """
    try:
        logger.debug(f"_ensure_conversation_exists: Starting for session_id={session_id[:20]}...")
        conv_repo = _get_chatbot_conversation_repository()
        logger.debug(f"_ensure_conversation_exists: conv_repo={conv_repo is not None}")
        if not conv_repo:
            logger.warning("[CopilotKit] Could not get conversation repository")
            return False

        # Check if conversation already exists (don't use .single() which throws on no results)
        # Use direct query with limit instead (Supabase client is synchronous, no await)
        try:
            logger.debug("_ensure_conversation_exists: Checking existing...")
            if conv_repo.client:
                check_result = (
                    conv_repo.client.table("chatbot_conversations")
                    .select("session_id")
                    .eq("session_id", session_id)
                    .limit(1)
                    .execute()
                )
                existing = check_result.data if check_result.data else None
                logger.debug(f"_ensure_conversation_exists: existing={existing}")
                if existing:
                    return True
            else:
                logger.debug("_ensure_conversation_exists: No client!")
                return False
        except Exception as check_err:
            logger.debug(f"_ensure_conversation_exists: Check error: {check_err}")
            # Continue to try creating if check fails

        # Create new conversation for this session (use 'general' query_type from enum)
        # NOTE: Using direct synchronous Supabase call since supabase-py is synchronous
        logger.debug("_ensure_conversation_exists: Creating new conversation...")
        # #1405: attribute the conversation to the JWT-verified owner (stashed in the
        # attribution contextvar by the auth gate). The anon sentinel is only a fallback
        # for a genuinely unattributed request — never a fabricated id.
        # chatbot_messages/chatbot_message_feedback.computed_user_id inherit this via
        # migration 123's trigger, so message-owner == conversation-owner.
        attr = get_attribution()
        owner_id = (attr.user_id if attr else None) or _ANONYMOUS_USER_ID

        def _insert_conversation(uid: str) -> bool:
            conversation_data = {
                "session_id": session_id,
                "user_id": uid,
                "title": "CopilotKit Conversation",
                "query_type": "general",
                "metadata": {"source": "copilotkit", "created_automatically": True},
            }
            result = (
                conv_repo.client.table("chatbot_conversations").insert(conversation_data).execute()
            )
            logger.debug(f"_ensure_conversation_exists: create result data={result.data}")
            return bool(result.data)

        try:
            if _insert_conversation(owner_id):
                logger.info(
                    f"[CopilotKit] Created conversation for session_id={session_id[:20]}..."
                )
                return True
            logger.warning(
                f"[CopilotKit] Failed to create conversation for session_id={session_id}"
            )
            return False
        except Exception as create_err:
            # #1405 HIGH: a real owner_id with no chatbot_user_profiles row FK-fails
            # (23503). Never silently drop the whole session — the exact failure this
            # migration fixes — so retry with the always-provisioned anon owner. Honest
            # fallback attribution; the conversation (and its messages/feedback) persists.
            if owner_id != _ANONYMOUS_USER_ID:
                logger.warning(
                    f"[CopilotKit] conversation create failed for owner={owner_id} "
                    f"({create_err}); retrying as anon for session_id={session_id[:20]}..."
                )
                try:
                    if _insert_conversation(_ANONYMOUS_USER_ID):
                        return True
                except Exception as retry_err:
                    logger.debug(f"_ensure_conversation_exists: anon retry error: {retry_err}")
            else:
                logger.debug(f"_ensure_conversation_exists: Create error: {create_err}")
            return False
    except Exception as e:
        logger.error(f"[CopilotKit] Error ensuring conversation exists: {e}")
        return False


# =============================================================================
# E2I BACKEND ACTIONS
# =============================================================================

# Fallback sample data when database is unavailable
# Real KPI substrate for the Home landing tiles. Each metric field maps to a
# vetted allowlisted query in `public.kpi_query_registry` (migrations 044 + 063),
# run via the `kpi_query` RPC against REAL tables (treatment_events). This replaces
# the former synthetic `business_metrics` rollup AND the hardcoded `_FALLBACK_KPIS`
# sample (an intentional sample fallback, commits 96e2ca24e / 2662a2c04 -- now
# superseded: showing fabricated/synthetic values as real on the landing page is
# the exact harm the H1 fix removes).
#
# field -> (kpi_query registry id, json result key, brand-scoped?, defined-for-"All"?)
# `defined_for_all=False` marks an inherently per-brand metric (TRx share filters
# `brand = $1` with no NULL guard, unlike the other queries' `$1 IS NULL OR ...`):
# for the aggregate "All" view it is N/A, so we return honest None instead of the
# misleading 0 that a NULL brand would yield from the share SQL.
_KPI_SUMMARY_QUERIES: Dict[str, tuple] = {
    "trx_volume": ("business_impact_trx", "trx", True, True),
    "nrx_volume": ("business_impact_nrx", "nrx", True, True),
    "market_share": ("business_impact_trx_share", "share", True, False),
    "conversion_rate": ("business_impact_conversion_rate", "conversion_rate", False, True),
    "hcp_reach": ("business_impact_hcp_reach", "hcp_reach", True, True),
    "patient_starts": ("business_impact_nbrx", "nbrx", True, True),
}


# Moved to the measure-basis SSOT (#1640): the KPI history/segmented/nowcast
# routes need the same derivation, and importing this route module to get it
# is not an option (orchestrator/RAG stacks, ~30s).
from src.kpi.measure_basis import query_substrates_cached  # noqa: E402


def _kpi_summary_measure_bases(
    client: Any = "__default__", brand: str | None = None, region: str | None = None
) -> dict[str, dict]:
    """Per-tile substrate for the Home KPI summary, derived from the SQL (#1640).

    Two earlier shapes were wrong in instructive ways. The first declared one
    block-level object with a ``not_substrate`` key and no ``substrate``, which
    kept the label and dropped the contract. The second mapped each tile to a
    registry KPI id by hand -- and got ``hcp_reach`` wrong, labelling an
    event-ledger ``COUNT(DISTINCT hcp_id) FROM treatment_events`` as WS3-BI-004
    HCP Coverage (a fraction over ``hcp_profiles``). A confident WRONG substrate
    is worse than none, and no registry KPI corresponds to that tile at all.

    The tiles run QUERIES, so the substrate comes from the registry SQL those
    queries are made of. Nothing here is hand-written, and an unreadable
    registry yields no basis rather than a guess.
    """
    if client is None:
        return {}
    # The RESOLVED query id, not the base: a region filter routes each tile to a
    # `*_region` variant, and those read more tables (measured:
    # business_impact_hcp_reach reads {treatment_events} while
    # business_impact_hcp_reach_region reads {patient_journeys,
    # treatment_events}). Deriving from the base would understate the substrate
    # under a region filter — the hcp_reach defect one level down.
    brand_param = None if brand in (None, "All") else brand
    resolved: dict[str, str] = {
        field: _kpi_summary_query(spec[0], brand_param, region, spec[2])[0]
        for field, spec in _KPI_SUMMARY_QUERIES.items()
    }
    by_query = query_substrates_cached(tuple(sorted(set(resolved.values()))))
    if not by_query:
        return {}
    bases: dict[str, dict] = {}
    for field, spec in _KPI_SUMMARY_QUERIES.items():
        query_id = resolved[field]
        tables = by_query.get(query_id)
        if tables:
            bases[field] = {
                "substrate": tables,
                # The field both prompts name as the only comparability key
                # (#1640). For a live computed tile it equals the SQL's tables.
                "comparison_key": tables,
                "query_id": query_id,
                "computed": True,
                "runtime_confirmed": True,
                "declared_sources": tables,
                "measure": f"computed on demand from {', '.join(tables)}",
                "note": (
                    "Computed on demand from the operational substrate via the kpi_query "
                    "registry — NOT read from the stored business_metrics table. Only "
                    "compare with a figure whose substrate matches."
                ),
            }
    return bases


def _kpi_summary_query(
    base_query_id: str, brand_param: Optional[str], region: Optional[str], brand_scoped: bool
) -> tuple[str, list]:
    """Resolve ``(query_id, params)`` for one KPI-summary metric, region-aware.

    When a ``region`` is selected, route to the additive ``*_region`` variant
    (migration 077). Those variants are deliberately NOT in
    ``SYNTHETIC_TWINNED_QUERY_IDS`` (which mirrors migration 066 and is
    drift-checked), so ``resolve_kpi_query_id`` will not auto-swap them — we
    append ``_include_synthetic`` here under the showcase flag instead, matching
    the calculator's ``_region_variant`` behavior. The region-variant param order
    is ``[brand, region]`` for brand-scoped metrics and ``[region]`` otherwise
    (mirrors the migration's ``$1``/``$2`` positions). With no region selected,
    falls back to the existing base query + ``resolve_kpi_query_id`` twin path.
    """
    if region:
        qid = f"{base_query_id}_region"
        if kpi_include_synthetic():
            qid = f"{qid}_include_synthetic"
        params = [brand_param, region] if brand_scoped else [region]
        return qid, params
    return resolve_kpi_query_id(base_query_id), ([brand_param] if brand_scoped else [])


_FALLBACK_AGENTS = [
    {"id": "orchestrator", "name": "Orchestrator", "tier": 1, "status": "active"},
    {"id": "causal-impact", "name": "Causal Impact", "tier": 2, "status": "idle"},
    {"id": "gap-analyzer", "name": "Gap Analyzer", "tier": 2, "status": "idle"},
    {"id": "drift-monitor", "name": "Drift Monitor", "tier": 3, "status": "active"},
    {"id": "health-score", "name": "Health Score", "tier": 3, "status": "active"},
    {"id": "explainer", "name": "Explainer", "tier": 5, "status": "idle"},
]


def _coerce_metric(value: Any) -> Optional[float]:
    """Coerce a kpi_query JSON value to a number, or None when absent."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    # COUNT-based metrics arrive as integral floats; keep ints integral so the
    # FE renders "HCPs Reached" / "TRx" as whole numbers, not "0.0".
    return int(num) if num.is_integer() else num


def _fetch_data_through(client: Any) -> Optional[str]:
    """Latest prescription event_date in treatment_events (the data-coverage end),
    via the vetted kpi_query allowlist. None when unavailable. Lets the FE render a
    DYNAMIC "data through <date>" label on empty tiles instead of a bare 0."""
    if client is None:
        return None
    try:
        response = client.rpc(
            "kpi_query",
            {"query_id": resolve_kpi_query_id("business_impact_data_through"), "params": []},
        ).execute()
        rows = response.data or []
        value = rows[0].get("data_through") if rows else None
        return str(value) if value is not None else None
    except Exception as e:
        logger.warning(f"[CopilotKit] KPI summary data_through query failed: {e}")
        return None


async def get_kpi_summary(brand: str, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the REAL KPI summary for a brand for the Home landing tiles.

    Reads the vetted allowlisted KPI queries (treatment_events via the `kpi_query`
    RPC) -- the same real substrate the KPI grid uses -- NOT the synthetic
    `business_metrics` table and NOT a hardcoded sample. When the source data is
    stale/empty the values are honest zeros with ``data_source="database"``; when
    the DB is unreachable or every query fails the result is fail-closed with
    ``data_source="unavailable"`` and ``None`` metrics. Values are NEVER fabricated.

    DEMO/REVIEW (``E2I_KPI_INCLUDE_SYNTHETIC``): on a synthetic-gold instance the
    production gate (migration 066 default-excludes ``is_synthetic=true`` rows)
    leaves these tiles honest-empty. With the flag set, the underlying queries
    swap to their ``_include_synthetic`` twins so the tiles render from the
    synthetic data, and ``data_source`` becomes ``"synthetic"`` so the FE labels
    them as such -- still never fabricated, just explicitly synthetic-sourced.

    Args:
        brand: Brand name (Remibrutinib, Fabhalta, Kisqali, or All)
        region: Optional geographic region (northeast/south/midwest/west, matched
            case-insensitively). When set, each metric routes to its region-scoped
            query variant (migration 077) so the tiles re-scope by region.

    Returns:
        ``{brand, period, metrics, data_source}`` -- always this shape, even on
        an unknown brand (honest ``data_source="unavailable"``).
    """
    logger.info(f"[CopilotKit] Fetching real KPI summary for brand: {brand}, region: {region}")

    metric_fields = list(_KPI_SUMMARY_QUERIES.keys())
    valid_brands = ["Remibrutinib", "Fabhalta", "Kisqali", "All"]

    if brand not in valid_brands:
        return {
            "brand": brand,
            "period": "Last 30 days of data",
            "metrics": dict.fromkeys(metric_fields),
            "data_source": "unavailable",
            "data_through": None,
            "error": f"Unknown brand: {brand}. Available: {valid_brands[:-1]}",
        }

    from src.api.dependencies.supabase_client import get_supabase

    client = get_supabase()
    metrics: Dict[str, Any] = {}
    any_ok = False

    if client is not None:
        brand_param = None if brand == "All" else brand
        for field, (
            query_id,
            result_key,
            brand_scoped,
            defined_for_all,
        ) in _KPI_SUMMARY_QUERIES.items():
            if brand == "All" and not defined_for_all:
                # Inherently per-brand (e.g. TRx share) -> N/A for the aggregate
                # view; honest None rather than a misleading 0.
                metrics[field] = None
                continue
            resolved_query_id, params = _kpi_summary_query(
                query_id, brand_param, region, brand_scoped
            )
            try:
                response = client.rpc(
                    "kpi_query",
                    {"query_id": resolved_query_id, "params": params},
                ).execute()
                rows = response.data or []
                metrics[field] = _coerce_metric(rows[0].get(result_key)) if rows else None
                any_ok = True
            except Exception as e:  # fail-closed for this metric, never fabricate
                logger.warning(f"[CopilotKit] KPI summary query {query_id} failed: {e}")
                metrics[field] = None
    else:
        logger.warning("[CopilotKit] No Supabase client; KPI summary unavailable")
        metrics = dict.fromkeys(metric_fields)

    return {
        "brand": brand,
        # Frontier-anchored (migration 089): the window ends at data_through,
        # not wall-clock now — "of data" keeps the tile label honest.
        "period": "Last 30 days of data",
        "metrics": metrics,
        # #1640: per-tile substrate, derived from the registry. Every figure
        # here is COMPUTED from the operational substrate via the kpi_query
        # registry — none is a stored business_metrics row — so a tile must not
        # be read as a check on, or a correction to, a business_metrics value
        # under the same name (measured ~73x apart for TRx).
        "measure_basis": _kpi_summary_measure_bases(client, brand=brand, region=region),
        # When the E2I_KPI_INCLUDE_SYNTHETIC demo flag is on, the figures are
        # computed over synthetic-gold rows (the _include_synthetic twins) rather
        # than real-world data -> surface "synthetic" so the FE badges them
        # honestly (Home QuickStatTile shows a "synthetic data" chip), never
        # passing synthetic figures off as production "database" values.
        "data_source": (
            "synthetic"
            if (any_ok and kpi_include_synthetic())
            else ("database" if any_ok else "unavailable")
        ),
        # Data-coverage end (latest treatment_events prescription date) so the FE
        # renders a 0/null tile as "No recent activity -- data through <date>" with
        # a DYNAMIC date, not a bare 0. None when unavailable.
        "data_through": _fetch_data_through(client),
    }


async def _fetch_agents_from_db() -> Optional[List[Dict[str, Any]]]:
    """
    Fetch agent status from database.

    Returns:
        List of agent dicts or None if unavailable
    """
    repo = await _get_agent_registry_repository()
    if not repo:
        return None

    try:
        from src.repositories.agent_registry import tier_number_for_category

        # Fetch all active agents in one schema-clean query. The real schema
        # stores the tier as the ``agent_tier`` text category (NOT an int
        # ``tier`` column), so the numeric tier is derived from that category
        # rather than looping a phantom int tier (issue #825).
        active_agents = await repo.get_active_agents()
        all_agents = [
            {
                "id": agent.get("agent_name", "unknown"),
                "name": agent.get("display_name", agent.get("agent_name", "Unknown")),
                "tier": tier_number_for_category(agent.get("agent_tier")),
                "status": "active" if agent.get("is_active", True) else "idle",
                "description": agent.get("description", ""),
            }
            for agent in active_agents
        ]

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

    Delegates to the orchestrator for real causal analysis. If the orchestrator
    is unavailable, raises, or returns an empty response, this function
    fail-closes with a structured error envelope rather than fabricating
    statistics. See GitHub issue #418 for context on the prior RNG-fallback bug.

    A dev-mode placeholder path is available behind
    ``E2I_ENABLE_SIMULATED_FALLBACK=1`` (defaults OFF). When enabled it returns
    pinned zeros with ``data_source="dev_mock"`` — it never returns RNG values,
    so even in dev mode the response is unambiguous about its provenance.

    Args:
        intervention: Type of intervention (e.g., "HCP Engagement", "Marketing Campaign")
        target_kpi: Target KPI to analyze (e.g., "TRx Volume", "Market Share")
        brand: Brand to analyze

    Returns:
        Dictionary with causal analysis results. On failure, returns an error
        envelope ``{"success": False, "error": ..., "data_source": "unavailable", ...}``
        suitable for CopilotKit chat surfaces to render to the user.
    """
    logger.info(f"[CopilotKit] Running causal analysis: {intervention} -> {target_kpi} for {brand}")

    common_context = {
        "intervention": intervention,
        "target_kpi": target_kpi,
        "brand": brand,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Try to run through orchestrator for real causal analysis
    orchestrator = _get_orchestrator()

    if orchestrator:
        try:
            query = f"What is the causal impact of {intervention} on {target_kpi} for {brand}?"
            result = await orchestrator.run(
                {
                    "query": query,
                    "user_context": {
                        "brand": brand,
                        "intervention": intervention,
                        "target_kpi": target_kpi,
                    },
                }
            )

            # Extract causal results if available. Three cases:
            #   1) Orchestrator returned explicit causal_results dict → pass through.
            #   2) Orchestrator returned ate/ci/p_value top-level → build dict from real fields.
            #   3) Orchestrator returned response_text only → return interpretation
            #      with results=None (F-001 iter-1 HIGH-2: no fabricated zero defaults).
            if result and result.get("response_text"):
                causal_results = result.get("causal_results")
                if causal_results is None:
                    # Look for top-level real fields; only build dict if at
                    # least one numeric field is actually present.
                    has_real_fields = any(
                        result.get(k) is not None for k in ("ate", "ci", "p_value", "significant")
                    )
                    if has_real_fields:
                        causal_results = {
                            "average_treatment_effect": result.get("ate"),
                            "confidence_interval": result.get("ci"),
                            "p_value": result.get("p_value"),
                            "statistical_significance": result.get("significant"),
                        }
                    # else: causal_results stays None — interpretation-only response.

                return {
                    **common_context,
                    "results": causal_results,
                    "interpretation": result.get("response_text", ""),
                    "data_source": "orchestrator",
                    "agents_used": result.get("agents_dispatched", []),
                }

            upstream_error = "Causal orchestrator returned an empty response (no response_text)"
            logger.warning(f"[CopilotKit] {upstream_error}")
        except Exception as e:  # noqa: BLE001 — broad catch is intentional; details surfaced below
            upstream_error = f"Causal orchestrator failed: {type(e).__name__}: {e}"
            logger.warning(f"[CopilotKit] {upstream_error}")
    else:
        upstream_error = "Causal orchestrator is not initialized"
        logger.warning(f"[CopilotKit] {upstream_error}")

    # Optional dev-mode placeholder path (default OFF). Returns pinned zeros
    # with explicit data_source="dev_mock" so callers cannot confuse it for
    # real data. NEVER call RNG primitives here (would re-introduce F-001).
    if os.getenv("E2I_ENABLE_SIMULATED_FALLBACK", "0").lower() in ("1", "true", "yes"):
        logger.warning(
            "[CopilotKit] E2I_ENABLE_SIMULATED_FALLBACK enabled — returning dev_mock placeholder"
        )
        return {
            **common_context,
            "success": True,
            "results": {
                "average_treatment_effect": 0.0,
                "confidence_interval": [0.0, 0.0],
                "p_value": 0.0,
                "statistical_significance": False,
                "sample_size": 0,
            },
            "interpretation": (
                "Dev-mode placeholder response. Real causal analysis is unavailable; "
                "these values are pinned zeros and must not be used for decisions."
            ),
            "data_source": "dev_mock",
            "upstream_error": upstream_error,
        }

    # Default path: fail closed with a structured error envelope.
    return {
        **common_context,
        "success": False,
        "error": (
            "Causal analysis service is currently unavailable. Please try again later "
            f"({upstream_error})."
        ),
        "data_source": "unavailable",
    }


# Disclaimer surfaced on every placeholder response so the chat UI / caller
# cannot mistake scaffolded sample data for real AI-generated analysis.
_PLACEHOLDER_DISCLAIMER = (
    "PLACEHOLDER / NOT YET IMPLEMENTED — these are illustrative sample values, "
    "NOT real analysis. Do not use for decisions."
)


def _placeholder_actions_enabled() -> bool:
    """Whether the scaffolded placeholder copilot actions may return sample data.

    These actions (``getRecommendations``, ``searchInsights``) are scaffolded
    placeholders awaiting a real backend (intent: the actions are a requested
    feature, but the data layer is not wired yet — see commit 2662a2c0 which
    added them as part of the initial CopilotKit integration). Their hardcoded
    pharma numbers (``+15% TRx lift``, ``confidence 0.85`` ...) are plausible
    enough to be mistaken for real causal output.

    Mirrors the ``E2I_ENABLE_SIMULATED_FALLBACK`` pattern already used by
    ``run_causal_analysis``: DEFAULT OFF so the production chat fails closed
    rather than presenting fabricated advice as real analysis. When explicitly
    enabled for dev/demo, the responses carry an unmistakable provenance marker
    (``data_source="placeholder"`` + ``is_placeholder`` + ``disclaimer``).
    """
    return os.getenv("E2I_ENABLE_PLACEHOLDER_ACTIONS", "0").lower() in ("1", "true", "yes")


async def get_recommendations(brand: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Get AI-powered recommendations for a brand.

    SCAFFOLDED PLACEHOLDER (no real backend yet). Fails closed by default so
    the chat never presents fabricated recommendations as real AI analysis.
    Set ``E2I_ENABLE_PLACEHOLDER_ACTIONS=1`` to surface clearly-marked sample
    data for dev/demo. Once a real recommendations service exists, wire it here.

    Args:
        brand: Brand to get recommendations for
        context: Optional context about what kind of recommendations are needed

    Returns:
        Dictionary with recommendations (or a fail-closed envelope by default)
    """
    logger.info(f"[CopilotKit] get_recommendations requested for {brand}")

    if not _placeholder_actions_enabled():
        # Fail closed: do NOT return fabricated recommendations in production.
        logger.warning(
            "[CopilotKit] get_recommendations is a placeholder and "
            "E2I_ENABLE_PLACEHOLDER_ACTIONS is OFF — returning not_implemented"
        )
        return {
            "brand": brand,
            "context": context or "General recommendations",
            "success": False,
            "recommendations": [],
            "data_source": "not_implemented",
            "error": (
                "Recommendations are not yet available (no recommendations "
                "service is wired). This feature is under development."
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    logger.warning(
        "[CopilotKit] E2I_ENABLE_PLACEHOLDER_ACTIONS enabled — "
        "returning PLACEHOLDER recommendations (sample data, not real analysis)"
    )
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
        # Provenance markers: the UI MUST surface these so users cannot mistake
        # placeholder advice for real AI-generated analysis.
        "data_source": "placeholder",
        "is_placeholder": True,
        "disclaimer": _PLACEHOLDER_DISCLAIMER,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def search_insights(query: str, brand: Optional[str] = None) -> Dict[str, Any]:
    """
    Search for insights in the E2I knowledge base.

    SCAFFOLDED PLACEHOLDER (no real knowledge-base search yet). Fails closed by
    default so the chat never presents fabricated insights as real results. Set
    ``E2I_ENABLE_PLACEHOLDER_ACTIONS=1`` to surface clearly-marked sample data
    for dev/demo. Once a real insights/search backend exists, wire it here.

    Args:
        query: Search query
        brand: Optional brand filter

    Returns:
        Dictionary with search results (or a fail-closed envelope by default)
    """
    logger.info(f"[CopilotKit] search_insights requested: {redact_query(query)}")

    if not _placeholder_actions_enabled():
        # Fail closed: do NOT return fabricated insights in production.
        logger.warning(
            "[CopilotKit] search_insights is a placeholder and "
            "E2I_ENABLE_PLACEHOLDER_ACTIONS is OFF — returning not_implemented"
        )
        return {
            "query": query,
            "brand_filter": brand,
            "success": False,
            "results": [],
            "total_results": 0,
            "data_source": "not_implemented",
            "error": (
                "Insights search is not yet available (no knowledge-base "
                "search service is wired). This feature is under development."
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    logger.warning(
        "[CopilotKit] E2I_ENABLE_PLACEHOLDER_ACTIONS enabled — "
        "returning PLACEHOLDER insights (sample data, not real results)"
    )
    # Sample (placeholder) search results — illustrative only.
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
        # Provenance markers: the UI MUST surface these so users cannot mistake
        # placeholder insights for real search results.
        "data_source": "placeholder",
        "is_placeholder": True,
        "disclaimer": _PLACEHOLDER_DISCLAIMER,
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


# Caps for the synthesis transcript: keep the most recent prior turns and
# truncate long messages so the synthesis prompt cannot grow unbounded.
_SYNTHESIS_HISTORY_MAX_MESSAGES = 12
_SYNTHESIS_HISTORY_MAX_CHARS = 2000


def _extract_synthesis_history(messages: Sequence[Any]) -> List[Dict[str, str]]:
    """Prior conversation turns as ``[{role, content}]`` for the synthesizer.

    synthesize_node is a SEPARATE LLM call from chat_node: chat_node sees the
    full message history (so it picks the right tool args for follow-ups),
    but the synthesizer previously saw only the current question + tool
    artifacts — it answered follow-ups like "is that above baseline?" with
    "I'm missing the preceding conversation". This rebuilds the text-only
    transcript BEFORE the current question; tool messages and tool-call stub
    AIMessages (empty content) are plumbing, not conversation.
    """
    turns: List[Dict[str, str]] = []
    for msg in messages:
        role: Optional[str] = None
        content: Any = None
        if isinstance(msg, HumanMessage):
            role, content = "user", msg.content
        elif isinstance(msg, AIMessage):
            role, content = "assistant", msg.content
        elif isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
            role, content = msg["role"], msg.get("content")
        if role is None:
            continue
        if isinstance(content, list):
            # Anthropic content blocks
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        if not isinstance(content, str) or not content.strip():
            continue
        turns.append({"role": role, "content": content.strip()[:_SYNTHESIS_HISTORY_MAX_CHARS]})

    # Everything from the last user turn onward is the CURRENT question, not history
    last_user_idx = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["role"] == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return []
    return turns[:last_user_idx][-_SYNTHESIS_HISTORY_MAX_MESSAGES:]


# System prompt for the CopilotKit chat agent
def _accumulate_tool_call_event(accumulated: list[dict[str, Any]], chunk: Any) -> None:
    """Merge one streamed chunk's tool-call data into ``accumulated``.

    ``tool_call_chunks`` is the AUTHORITATIVE channel: it carries the provider's
    index (Anthropic: content-block index; OpenAI: tool-call ordinal) plus
    name/id on the start chunk, then args deltas. The parallel
    ``chunk.tool_calls`` entries are a best-effort parse of the SAME event and
    carry NO index — reconciling the two by list position minted a ghost entry
    whose args never arrived whenever leading text shifted Anthropic's block
    indices; the ghost reached ToolNode as ``{}`` and its validation error
    rendered raw in the chat (2026-07-07 session review). ``chunk.tool_calls``
    is therefore only a fallback for chunks that carry no tool_call_chunks,
    merged by id so re-emissions never duplicate.
    """
    tool_call_chunks = getattr(chunk, "tool_call_chunks", None) or []
    if tool_call_chunks:
        for tc_chunk in tool_call_chunks:
            tc_index = tc_chunk.get("index")
            tc_index = 0 if tc_index is None else tc_index
            tc_id = tc_chunk.get("id")
            tc_name = tc_chunk.get("name")
            tc_args = tc_chunk.get("args", "")

            existing = None
            for entry in accumulated:
                if entry.get("index") == tc_index:
                    existing = entry
                    break
            if existing is None:
                existing = {
                    "index": tc_index,
                    "id": tc_id,
                    "name": tc_name or "",
                    "args": {},
                    "args_str": "",
                }
                accumulated.append(existing)
            if tc_id and not existing.get("id"):
                existing["id"] = tc_id
            if tc_name and not existing.get("name"):
                existing["name"] = tc_name
            if tc_args:
                existing["args_str"] = existing.get("args_str", "") + tc_args
        return

    for tc in getattr(chunk, "tool_calls", None) or []:
        if not tc.get("name") and not tc.get("id"):
            continue
        tc_id = tc.get("id")
        existing = None
        if tc_id:
            for entry in accumulated:
                if entry.get("id") == tc_id:
                    existing = entry
                    break
        if existing is not None:
            if tc.get("name") and not existing.get("name"):
                existing["name"] = tc["name"]
            if tc.get("args") and not existing.get("args_str"):
                existing["args"] = tc["args"]
            continue
        accumulated.append(
            {
                # Negative ordinal keyspace: a fallback entry must never collide
                # with a provider chunk index in a mixed stream.
                "index": -1 - len(accumulated),
                "id": tc_id,
                "name": tc.get("name") or "",
                "args": tc.get("args") or {},
                "args_str": "",
            }
        )


def _finalize_tool_calls(accumulated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse accumulated stream entries into executable tool calls.

    Prefers the streamed ``args_str`` (accumulated from chunks) over the
    often-empty parsed ``args``, repairs braces for args that started mid-JSON,
    drops nameless entries, and collapses duplicate ids keeping the entry that
    actually received args — defense-in-depth so a ghost duplicate can never
    reach ToolNode with ``{}`` args. Dedup is strictly by id: two DISTINCT
    calls to the same tool (different ids) both survive, even arg-less ones.
    """
    import json as json_mod

    parsed: list[dict[str, Any]] = []
    for tc in accumulated:
        if not tc.get("name"):
            continue
        args_str = tc.get("args_str", "")
        args = tc.get("args", {})
        if args_str:
            # The args_str might be missing outer braces if it started mid-JSON
            args_str_stripped = args_str.strip()
            if args_str_stripped and not args_str_stripped.startswith("{"):
                args_str_stripped = "{" + args_str_stripped
            if args_str_stripped and not args_str_stripped.endswith("}"):
                args_str_stripped = args_str_stripped + "}"
            try:
                args = json_mod.loads(args_str_stripped) if args_str_stripped else {}
            except json_mod.JSONDecodeError as e:
                logger.error(f"[CopilotKit] Failed to parse args_str for {tc.get('name')}: {e}")
                logger.error(f"[CopilotKit] Raw args_str: {args_str[:500]}")
                # Fall back to original args if parsing fails
                if isinstance(args, str):
                    try:
                        args = json_mod.loads(args) if args else {}
                    except (json_mod.JSONDecodeError, ValueError):
                        args = {}
        elif isinstance(args, str):
            try:
                args = json_mod.loads(args) if args else {}
            except json_mod.JSONDecodeError:
                args = {}
        parsed.append({"id": tc.get("id") or str(uuid.uuid4()), "name": tc["name"], "args": args})

    deduped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for call in parsed:
        cid = call["id"]
        if cid in deduped:
            if not deduped[cid]["args"] and call["args"]:
                deduped[cid] = call
            continue
        deduped[cid] = call
        order.append(cid)
    return [deduped[cid] for cid in order]


# ---------------------------------------------------------------------------
# Frontend actions (CopilotKit generative UI) — v1.30.0
#
# The frontend registers actions via useCopilotAction (renderKpiTrend,
# navigateTo, …) and the CopilotKit runtime forwards them as ``tools`` in every
# agent/run body. Our execute() bridge maps them into RunAgentInput.tools, and
# LangGraphAGUIAgent.langgraph_default_merge_state injects them into graph
# input as state["copilotkit"]["actions"]. The chain then BROKE here:
# E2IAgentState had no ``copilotkit`` channel (LangGraph drops unknown input
# keys) and chat_node bound only the backend E2I_CHATBOT_TOOLS — so the model
# could never see, let alone call, the chart action. These helpers close that
# last mile: convert the riding actions into bind-able tool schemas, route
# frontend-action calls to END (the client executes the handler and renders
# the generative UI — ToolNode has no implementation for them), and drop
# frontend calls from prompt-discouraged mixed turns so ToolNode never sees an
# unknown tool name.
# ---------------------------------------------------------------------------

#: CopilotKit action parameter types → JSON-schema types (the frontend may
#: send parameters in CopilotKit's array-of-parameters format rather than a
#: prebuilt JSON schema; both shapes are accepted by _frontend_action_schemas).
_CK_PARAM_TYPES = {
    "string": "string",
    "number": "number",
    "boolean": "boolean",
    "object": "object",
}


def _frontend_action_names(state: Mapping[str, Any]) -> set[str]:
    """Names of the frontend actions riding this run's state (empty set when
    the surface forwarded none)."""
    copilotkit_state = state.get("copilotkit") or {}
    actions = copilotkit_state.get("actions") or []
    return {action["name"] for action in actions if isinstance(action, dict) and action.get("name")}


def _ck_param_array_to_json_schema(parameters: list) -> dict[str, Any]:
    """Convert CopilotKit's array-of-parameters action format to JSON schema."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in parameters:
        if not isinstance(param, dict) or not param.get("name"):
            continue
        raw_type = str(param.get("type") or "string")
        if raw_type.endswith("[]"):
            schema: dict[str, Any] = {
                "type": "array",
                "items": {"type": _CK_PARAM_TYPES.get(raw_type[:-2], "string")},
            }
        else:
            schema = {"type": _CK_PARAM_TYPES.get(raw_type, "string")}
        if param.get("description"):
            schema["description"] = param["description"]
        properties[param["name"]] = schema
        if param.get("required"):
            required.append(param["name"])
    json_schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        json_schema["required"] = required
    return json_schema


def _frontend_action_schemas(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bind-able OpenAI-style tool schemas for the frontend actions in state.

    Skips malformed entries, duplicate names, and names shadowing a backend
    tool (backend implementations win — a frontend action must never hijack
    kpi_calculate_tool etc.). Parameters are accepted either as a JSON-schema
    object or as CopilotKit's array-of-parameters format; anything else
    degrades to an empty-object schema.
    """
    copilotkit_state = state.get("copilotkit") or {}
    raw_actions = copilotkit_state.get("actions") or []
    backend_names = {tool.name for tool in E2I_CHATBOT_TOOLS}
    schemas: list[dict[str, Any]] = []
    seen: set[str] = set()
    for action in raw_actions:
        if not isinstance(action, dict):
            continue
        name = action.get("name")
        if not name or name in backend_names or name in seen:
            continue
        seen.add(name)
        parameters = action.get("parameters")
        if isinstance(parameters, dict) and parameters.get("type") == "object":
            json_schema = parameters
        elif isinstance(parameters, list):
            json_schema = _ck_param_array_to_json_schema(parameters)
        else:
            json_schema = {"type": "object", "properties": {}}
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": action.get("description") or "",
                    "parameters": json_schema,
                },
            }
        )
    return schemas


def _strip_frontend_calls_when_mixed(
    tool_calls: list[dict[str, Any]], frontend_names: set[str]
) -> list[dict[str, Any]]:
    """In a mixed backend+frontend turn keep only the backend calls.

    The prompt tells the model to call frontend actions on their own; if it
    mixes anyway, ToolNode would choke on the unknown frontend name. Dropping
    the frontend call (and keeping the data flow) is the graceful degradation
    — the user can re-ask for the chart. Homogeneous turns pass through
    unchanged.
    """
    if not frontend_names or not tool_calls:
        return tool_calls
    backend_calls = [tc for tc in tool_calls if tc.get("name") not in frontend_names]
    if backend_calls and len(backend_calls) < len(tool_calls):
        dropped = [tc.get("name") for tc in tool_calls if tc.get("name") in frontend_names]
        logger.warning(
            f"[CopilotKit] Mixed tool turn: dropping frontend action call(s) {dropped} "
            f"so ToolNode only receives backend tools"
        )
        return backend_calls
    return tool_calls


def _is_frontend_only_turn(call_names: Sequence[Optional[str]], frontend_names: set[str]) -> bool:
    """True when every tool call in the turn is a frontend (generative-UI) action.

    Shared by the two places that must agree about it: ``_route_after_chat``,
    which ENDs such a turn for client-side execution, and ``chat_node``, which
    records its analytics before returning — because a turn that ends there has
    no downstream node left to record it. If these two conditions ever drifted
    apart we would either record turns that did not end, or (worse) keep the
    blind spot for turns that did.
    """
    if not frontend_names or not call_names:
        return False
    return all(name in frontend_names for name in call_names)


def _route_after_chat(state: Mapping[str, Any]) -> str:
    """Post-chat routing: backend tool calls → "tools"; everything else ends.

    A turn whose calls are ALL frontend actions must END the run: the client
    executes the action handler and renders the generative UI (e.g. the
    KpiTrendChart), then reports the result back in a follow-up run. Sending
    those calls to ToolNode would fail — there is no backend implementation.
    """
    messages = state.get("messages", [])
    if not messages:
        return "end"
    last_message = messages[-1]
    if not (isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None)):
        return "end"
    call_names = [tc.get("name") for tc in last_message.tool_calls]
    frontend_names = _frontend_action_names(state)
    if _is_frontend_only_turn(call_names, frontend_names):
        logger.info(
            f"[CopilotKit] Frontend action call(s) {call_names} — ending run for "
            f"client-side execution (generative UI)"
        )
        return "end"
    logger.info(f"[CopilotKit] Claude requested {len(call_names)} tool call(s)")
    return "tools"


def build_synthesis_prompt(
    original_query: str,
    tool_calls: list[dict],
    tool_results: list[dict],
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Frame the prior conversation + the user's question + the tool calls (with args)
    + the tool results so the synthesizer answers the ACTUAL question, resolves
    follow-up references ("is that above baseline?") against earlier turns, names the
    brand/period it used, and is honest about any window limitation. Fixes the 'asks
    for a brand it already used' and 'missing the preceding conversation' bugs."""
    import json as _json

    calls = _json.dumps(tool_calls, indent=2, default=str)
    results = _json.dumps(tool_results, indent=2, default=str)
    history_block = ""
    if history:
        transcript = "\n".join(
            f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
            for turn in history
        )
        history_block = (
            "Conversation so far (earlier turns; the question below may refer to values "
            "established here):\n" + transcript + "\n\n"
        )
    return (
        history_block + "User question:\n" + (original_query or "(none)") + "\n\n"
        "Tool calls the assistant made (note the brand/window/args already chosen):\n"
        + calls
        + "\n\n"
        "Tool results:\n" + results + "\n\n"
        "Write a concise, direct answer to the user's question. Use the specific data values. "
        "State which brand and time period the figure covers (from the tool args/results). "
        "If a result's window_status is 'not_applicable' or 'default' while the user asked for a "
        "specific period, say plainly that the figure covers the engine's reporting window, not the "
        "requested one. Do NOT ask the user to re-specify a brand or period they already provided. "
        "If the question refers to something from the conversation ('that', 'it', 'compared to "
        "before'), resolve the reference from the conversation above — do NOT claim missing "
        "context when the referenced value appears there. "
        "BASELINE MATH: never derive a baseline or trend by comparing a shorter window against a "
        "longer window that overlaps it (e.g. last 30 days vs last 90 days — the 90-day figure "
        "CONTAINS those same 30 days); a valid baseline is a prior non-overlapping period of the "
        "same length. If no non-overlapping comparison figure is available in the tool results, "
        "say so instead of manufacturing one. If any tool result carries a coverage_warning, "
        "quote it and do not draw trend conclusions from that figure. "
        "PROSE MUST MATCH YOUR OWN TABLE (#1550): every comparative claim in your prose — "
        "rankings, 'largest'/'smallest'/'top' superlatives, signs and directions, dollar "
        "impacts, averages — MUST be re-derived from the values printed in the SAME answer, "
        "not paraphrased from memory. State each number's sign and direction exactly as "
        "printed: +0.092 is positive (an increase), never 'declined by 0.09', and its dollar "
        "impact keeps the same sign. When ranking items, order them by the printed values and "
        "make the stated rank match that order. Name exactly ONE 'largest'/'smallest' item per "
        "axis, the one the printed n or value actually supports — never two different ones in "
        "the same answer. An average or total must be computed from the tabled values "
        "(round, don't drift). Before finishing, re-check every comparative sentence against "
        "the table beside it; on any conflict the TABLE is correct — rewrite the sentence. "
        "PERIOD GRAIN (#1552): every dated figure covers a reporting period (bucket) of some "
        "width — the tool results label the grain when known (e.g. 'calendar month 2026-08 "
        "(August 2026, monthly grain)'). Give each source period its OWN row and label: NEVER "
        "merge adjacent periods into one label such as 'Jun/Jul 2026' unless the source itself "
        "labels them as a single bucket — two periods with identical values stay two rows. "
        "Never mix periods of different widths in one table without labeling each row's period "
        "width. Describe a figure as partial, month-to-date (MTD), or pro-rated ONLY when the "
        "tool results say so (an as-of/data-through date inside the period) — then label it, "
        "e.g. 'Aug 2026 (through 2026-08-10)'; never invent a partial-period reading. When two "
        "figures differ mainly because their periods differ in width or completeness, say "
        "exactly that — a period-width difference is an explained artifact, NEVER an "
        "'unexplained discontinuity', a data-quality break, or a real decline — and do not "
        "infer a trend across periods of unequal width. "
        "FOOTNOTES: never place a footnote marker (*, †) on a table value unless the footnote "
        "text itself appears immediately below that table; with no footnote text, omit the "
        "marker."
    )


E2I_COPILOT_SYSTEM_PROMPT = """You are the E2I Analytics Assistant, an intelligent AI specialized in pharmaceutical commercial analytics for Novartis brands.

## Your Expertise

You help users with:
1. **KPI Analysis** - TRx, NRx, market share, conversion rates, patient starts
2. **Causal Analysis** - Understanding WHY metrics change and what drives performance
3. **Agent System** - Information about the tiered agent architecture (roster below)
4. **Recommendations** - AI-powered suggestions for HCP targeting and market access
5. **Insights Search** - Finding trends, causal paths, and historical patterns
6. **Cohort Profiles** - Aggregate HCP/patient cohort profiles (counts and breakdowns by specialty, tier, severity) served through the orchestrator's cohort_profiler agent

## Brands You Support

- **Kisqali** - HR+/HER2- breast cancer
- **Fabhalta** - PNH (Paroxysmal Nocturnal Hemoglobinuria)
- **Remibrutinib** - CSU (Chronic Spontaneous Urticaria)

## Guidelines

1. **Data-Driven Responses**: ALWAYS use the available tools to fetch real data before answering
2. **Source Attribution**: Cite the data source when presenting metrics or insights
3. **Commercial Focus, Clinically Grounded**: This is pharmaceutical COMMERCIAL analytics — KPIs, causal drivers, market access, HCP targeting. You MAY surface a brand's FDA-label indications, mechanism of action, pivotal endpoints, and real-world evidence (via `clinical_context_tool`) as factual, source-attributed context to GROUND and TAILOR commercial/causal/strategic insight — e.g. on-label HCP targeting, competitive density within an indication, how the label boundary shapes causal drivers. Do NOT provide individualized prescribing guidance or medical advice; for a patient-specific clinical decision, point users to the official Prescribing Information / Medical Information.
4. **Causal Clarity**: When discussing causation, be clear about confidence levels
5. **Actionable Insights**: Provide recommendations that can drive business decisions
6. **Honest Windows**: If a requested time window isn't supported for a metric, say so plainly and report the window actually used — never imply a figure covers a different period, and never ask for a brand or period the user already gave. When the user names a period ("this month"), PASS it to the tool; if you did not pass a window, the engine default was used — say that, and NEVER explain the gap as a platform limitation ("this metric doesn't use calendar-month windows") unless a tool result states it. Attribute window metadata (`data_through`, `reporting_window`) only to the KPI whose payload carries it, and never assert a window is absent unless that payload actually lacks it.
7. **Grounded Provenance**: claims about where information came from must be grounded in a tool payload from THIS conversation. Never describe synthetic/modeled data as "real" — when a payload carries `data_source: "synthetic"` or `evidence_is_synthetic: true`, disclose that once, plainly. Never attribute specific findings, quotes, or author names to a retrieved source beyond the fields the payload contains (a PubMed record returning only pmid/title/journal/doi does not license "per <author>," or clinical detail "from the literature pulled just now" — say plainly when detail comes from general knowledge rather than the retrieved record). Never invent method labels (e.g. "DoWhy/EconML-style") the payload doesn't state.
8. **Units Stay As Declared**: an effect size arrives with its declared scale (e.g. `cohens_d`). Report it in that unit — never relabel a standardized effect as percentage points (or any other unit), and never do arithmetic on a relabeled unit to derive new figures.
9. **Negatives About This Conversation**: before asserting something was not discussed, asked, or computed earlier ("we haven't run a segment ranking"), check the conversation AND the tools already called this session — a prior turn's tool results count as available even when its visible answer only summarized them.
10. **Negatives About The Platform**: an empty or narrow query result licenses only a query-scoped negative — "that query returned no rows", "I didn't find X via Y" — NEVER a platform-level one ("no blended health score exists", "no ROC-AUC data is available for <brand>", "there is no direct query tool for X"). Before asserting the platform lacks a capability or metric, check your own toolbox for a tool that serves it, and check this session for a turn that already served it (a champion AUC presented two turns ago refutes "none is available" now). If you cannot verify the absence at the platform level, present it as absence of evidence from the one query you ran — not absence of the thing.

## Tool Usage - CRITICAL

You MUST use tools proactively when users ask about data:
- Use `e2i_data_query_tool` for KPI metrics, causal chains, agent analyses (an agent's ACTIVITY LOG — runs, confidences, timestamps; NOT the system health score), triggers
- Use kpi_calculate_tool to COMPUTE a KPI value for a brand/period (NRx, TRx, NBRx, market share, conversion rate, ROI). Pass the brand and any time window the user names, and state which brand and window your answer covers.
- BREAKDOWN GUIDANCE: For NRx/TRx/NBRx/TRx-share/conversion-rate patient-segment breakdowns, call `kpi_calculate_tool` once per bucket of ONE axis and present the results as a table. Axes: `segment` ∈ {low_severity, medium_severity, high_severity}; `therapy_line` ∈ {0,1,2,3}; and FOR REMIBRUTINIB ONLY (volume KPIs and share, NOT conversion rate) `biologic` ∈ {naive, experienced} and `ige_tier` ∈ {low, medium, high}. A volume axis's buckets sum to the head-line KPI, so the breakdown reconciles with the total (rates/shares don't sum — their numerators/denominators do). When the user names a period ("last year", "Q1 2025"), ALWAYS pass `window` too — it composes with `segment`/`therapy_line` for TRx/NRx/NBRx, TRx share, and conversion rate. Use exactly one axis per breakdown — they are mutually exclusive.
- HONESTY GUARD: Clinical context is background framing only — never present a clinical sub-population as a data breakdown unless a tool actually returned per-bucket values. Biologic status (`biologic`) and IgE tier (`ige_tier`) are REAL breakdown axes for REMIBRUTINIB ONLY; for Kisqali/Fabhalta `kpi_calculate_tool` returns an error because the data is NULL by design — say it is unavailable for that brand and do NOT fabricate a split or guess. The real breakdown axes are severity tier (`segment`), line-of-therapy (`therapy_line`), biologic status and IgE tier (Remibrutinib only), plus geographic `region`. TRx Share is the brand's share of the TRACKED PORTFOLIO's prescriptions (Fabhalta + Kisqali + Remibrutinib, cross-indication) — NOT market share vs external competitors; competitor brands (e.g. Xolair, Dupixent) are not in the data model, so NEVER attribute the share complement to named competitors.
- ROI DISPERSION (#1532): the ROI headline is a pooled point estimate and carries NO interval — never invent one. When the response includes `temporal_variability_band`, present each slice's band as "the range of its monthly ROI values over the past 12 months" with its `n` — it measures recent temporal variability, NOT a confidence interval and NOT uncertainty about the current value; for slices with `band_suppressed: true`, state only the n and that the band is suppressed.
- SCALE GUARD (#1640): every KPI figure arrives with a `measure_basis` naming the substrate it was computed from. TWO FIGURES ARE COMPARABLE ONLY IF THEIR `measure_basis.comparison_key` MATCHES — that field, not `substrate`. They differ on purpose: a materialized history series is READ from `kpi_history` (its `substrate`) but RESTS on whatever the backfill drew it from (`materialized_from`, reduced to tables in `comparison_key`). Comparing on `substrate` would call a TRx history and an ROI history comparable because both are read from the same table, and would wrongly fence ROI history against stored ROI when both rest on `business_metrics`. If `measure_basis.mixed_sources` is true the series spans more than one substrate and is comparable with NOTHING — say so rather than comparing it. `kpi_calculate_tool` computes volume KPIs from the `treatment_events` ledger; `e2i_data_query_tool(query_type='kpi')` returns stored `business_metrics` rows, whose `value` is a MODELED market-scale level — measured, the national business_metrics TRx total is ~73x the trailing-30-day event count for the same brand. They are different quantities sharing a name. NEVER present one as a check, correction, total, or share-of for the other; never divide or sum across them; and never call the gap a discontinuity or a data error. If an answer needs both, give each its own row with its substrate stated, and say plainly that they measure different things. When a figure carries no `measure_basis`, treat it as NOT comparable rather than assuming it agrees.
- Use `causal_analysis_tool` for understanding metric drivers
- Use `clinical_context_tool` to fetch a brand's REAL FDA-label indications, mechanism of action, pivotal endpoints, and competitor landscape (OpenFDA / ChEMBL / ClinicalTrials.gov / PubMed) — call it for ANY label / indication / approved-use / mechanism / on-off-label / competitive-landscape question, then frame the answer commercially instead of deflecting.
- Use `document_retrieval_tool` for searching the knowledge base
- Use `agent_routing_tool` to get agent status and information
- Use `predict_hcp_segment_likelihood_tool` for "which HCP segments / specialties / regions are most likely to increase (or start / adopt) <brand> prescriptions?" — it scores the brand's promoted HCP-adoption champion over the real HCP cohort and returns a per-segment ranking of predicted adoption propensity with real n and standard errors. Pass the `brand` (required — do NOT guess one; the single brand a conversation is about is user-provided, use it and say so; if the conversation names multiple brands and the ask doesn't select one, run the ranking once per candidate brand or ask which one; only if no brand appears anywhere in the conversation, ask which brand) and `segment_by` ('specialty' by default, or 'geographic_region' for a region-phrased ask). The score is CURRENT adoption propensity (the platform's "likelihood to prescribe"), NOT a horizon-specific increase — if the user names a horizon ("next quarter") say the ranking is horizon-agnostic. Present the top segments as a table (segment, mean propensity, n) and flag any `low_confidence` (thin) segments. If it fails closed (no promoted champion), say so plainly — do NOT substitute a regional TRx trend as if it answered the segment question. Also use it for model-quality asks about a brand's adoption champion ("what's the ROC-AUC / calibration for <brand>?"): its payload carries the promoted champion's metadata (`model_name`, `holdout_auc`, `n_scored`) — agent-activity logs via `e2i_data_query_tool` hold NO model metrics, and a 0-row log query is not evidence that no metric exists.
- Use `orchestrator_tool` for complex multi-agent workflows — and for COHORT/PROFILE asks (#1562): aggregate HCP or patient cohort profiles ("profile the HCP cohort", "patient cohort for <brand>") ARE a supported analysis — route them through `orchestrator_tool`, which reaches the cohort_profiler agent and returns aggregate counts and breakdowns (e.g. by specialty, priority tier, severity). What the platform genuinely does NOT serve is the individual level: named individual HCP/patient identities, per-person rosters, or list exports — when a roster/export is asked for, state that limit in one sentence, then run (or offer) the aggregate cohort profile instead of declining the whole ask. Cohort asks follow the same brand rules as every tool: a single conversation brand is user-provided — use it and say so; if no brand appears anywhere in the conversation, ask which brand (an all-brands profile is a valid option to offer) — do NOT guess one. When dispatching a cohort ask, pass the user's stated inclusion criteria (age bounds, dates, regions, thresholds) through in the query VERBATIM — even criteria you believe the data model doesn't carry: the cohort layer itself decides servability and honestly reports any criterion it can't apply; never pre-filter the ask, and never present your own rewrite's omission as a platform limitation. Also use it for SYSTEM-HEALTH asks ("what is the current system health score?"): it reaches the health_score audit, whose payload carries the composite — `overall_health_score`, `health_grade`, `health_summary` (e.g. "Grade: A, Score: 99.4/100") with measured provenance. `e2i_data_query_tool(query_type='agent_analysis', agent_name='health_score')` reads only that agent's activity log; an average of per-run log confidences is NOT the health score — never present one as a proxy for it when the audit is one dispatch away.
- Use `tool_composer_tool` for multi-faceted queries

DO NOT just describe what tools can do - actually CALL them to get data!

## Nearest Supported Analysis — run it, don't ask (#1549)

When the user's ask does not map 1:1 to a tool but you can YOURSELF name a supported analysis that answers the nearest useful version of it, RUN that analysis in the same turn with stated default parameters. Naming the analysis and then asking permission ("I could show likelihood-to-prescribe ranked by specialty — tell me the brand and axis and I'll pull it") is the failure mode this rule forbids: if your draft reply names a runnable analysis and ends on a question, call the tool instead.
- Defaults, each stated inline as an assumption: brand = the brand this conversation is about (a single brand named anywhere in the conversation counts as user-provided — using it is not guessing); axis = the one the ask leans toward (specialty for "which doctors / oncologists", geographic_region for region-phrased asks); window = the tool's default reporting window when none is named.
- Multiple brands in play — ambiguous is not absent: when the conversation has discussed more than one brand and neither the ask nor the current topic clearly selects one, do not default — never silently pick one brand. Run the supported per-brand comparison when one exists (one call per candidate brand, presented side by side); otherwise ask ONE crisp question naming the candidate brands. A single conversation brand stays a default (rule above); no brand anywhere is case 1 below.
- Lead with the data: state the assumption in one short sentence, present the results, then offer the other slice (the other axis, brand, or window) as the follow-up — the offer comes after the data, not instead of it. Do not open with a paragraph of clarifying questions you then answer yourself.

Ask-ending (a reply with no tool call that ends on a question) is reserved for exactly two cases:
1. Genuinely undefined referent: NEITHER the ask NOR the conversation pins the entity or metric (e.g. a cold "why did it drop?" opening a session with no prior brand or metric) — ask 1-3 crisp clarifying questions.
2. Capability refusal with nothing adjacent: the request is outside the platform's supported analyses (e.g. building or exporting named individual HCP/patient rosters or lists, clinical cohort criteria the data model doesn't carry such as diagnosis year, individualized prescribing guidance) AND no supported analysis answers a nearby version of it — decline honestly and say what would be needed. When a nearby supported analysis DOES exist (e.g. the aggregate cohort profile via `orchestrator_tool` for a per-HCP roster ask, or the segment likelihood-to-prescribe ranking), state the capability limit in one sentence, then run that analysis under the rule above. Aggregate cohort profiling itself is NOT a refusal case — it is supported (see `orchestrator_tool` above).

## Inline Charts (generative UI)

When the user asks to chart / plot / graph / visualize a KPI's trend over time AND a `renderKpiTrend` tool is available, call `renderKpiTrend` — the UI renders a real line chart from stored KPI history directly inside your reply. Valid kpiId values: `trx`, `nrx`, `nbrx`, `trx_share` (aka `market_share`), `conversion_rate`, `roi`, or a registry code such as `WS3-BI-005` / `BR-001`. `nbrx` and `trx_share` are tracked per brand only — pass `brand` (Remibrutinib, Fabhalta, or Kisqali) for them or the chart will be empty; other KPIs accept an optional brand. History is monthly: requests for finer windows (e.g. "last 90 days") chart the stored monthly series — say so rather than apologizing. For `trx`/`nrx`/`nbrx` the trend can also be split by patient axis: pass `compareBy: "severity"` (or `"lot"`) to render ONE comparison chart with a line per severity tier (low/medium/high) or per line of therapy (0-3 prior lines) — for "compare TRx across segments" requests make a SINGLE call with `compareBy`, never one call per tier. To chart just one tier, pass `segment` ('low'/'medium'/'high') or `therapyLine` ('0'-'3') instead. Segment/LOT splits exist ONLY for trx/nrx/nbrx — `trx_share`, `conversion_rate`, and `roi` have no per-tier series; say so if asked. Call it on its OWN, never combined with other tool calls in the same turn (combined turns drop the chart). If `renderKpiTrend` is not in your tool list, answer with data from the other tools and say inline charts aren't available in this surface — do NOT describe a chart you cannot render.

## Response Format

- Be concise but comprehensive
- Use bullet points for lists
- Highlight key metrics with **bold**
- Include actual data values from tool results
- Offer at most one genuinely useful follow-up, only when it adds value.

## Agent Roster

{agent_roster}

When asked which agents exist, answer from THIS roster. These are agents, not
tools — do not substitute tool names (`kpi_calculate_tool`, `causal_analysis_tool`,
…) for agent names, and do not call `agent_routing_tool` to find out: that routes
a single query to an agent, it is not a directory.
"""


# #1638: the roster is INTERPOLATED from src.agents.factory, never transcribed.
# Turn 5.2 ("what agents are available") answered with tool names because this
# prompt carried no roster at all — only a hardcoded architecture phrase whose
# count had gone stale against the registry. A
# hand-written list would have fixed that turn and rotted at the next agent.
#
# factory imports only logging/os/typing at module scope (agents are loaded lazily
# by module/class name), so this costs nothing at import time.
E2I_COPILOT_SYSTEM_PROMPT = E2I_COPILOT_SYSTEM_PROMPT.replace(
    "{agent_roster}", build_agent_roster_block()
)


class E2IAgentState(TypedDict, total=False):
    """
    State for the E2I chat agent.

    Includes observable fields for CoAgent bidirectional state sync with frontend.
    These fields are emitted via copilotkit_emit_state() for real-time UI updates.
    """

    # Core message state
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_id: str  # Persistent session ID for message storage (optional)
    run_id: str  # Per-run stamping discriminator; rides state as the fallback channel to _run_id_context, mirroring session_id's redundancy

    # Observable state for CoAgent sync (Phase 1 of state sync implementation)
    current_node: str  # Current processing node: "chat", "tools", "synthesize", "idle"
    progress_steps: List[str]  # Progress steps: ["Processing query...", "Calling tools..."]
    progress_percent: int  # Progress percentage: 0-100
    tools_executing: List[str]  # Tools currently executing: ["orchestrator_tool", ...]
    agent_status: str  # Agent status: "processing", "waiting", "complete", "error"
    error_message: Optional[str]  # Error message if agent_status is "error"

    # CoAgents channel injected by LangGraphAGUIAgent.langgraph_default_merge_state:
    # {"actions": [<frontend useCopilotAction schemas>], "context": [...]}. This key
    # MUST be declared here — LangGraph drops input keys that aren't state channels,
    # and without it chat_node can never see (or bind) the frontend actions that
    # power generative UI like the inline KPI trend chart (v1.30.0).
    copilotkit: Dict[str, Any]


def create_e2i_chat_agent(
    chat_llm_tier: Literal["fast", "standard", "reasoning"] = "standard",
    chat_llm_reasoning_effort: str = "medium",
):
    """
    Create a LangGraph agent for E2I chat with Claude and tool calling.

    This agent:
    1. Uses Claude with bound E2I tools for data-driven responses
    2. Automatically executes tools when Claude invokes them
    3. Streams responses back to CopilotKit frontend

    Args:
        chat_llm_tier: llm_factory tier for the CHAT leg only (#1475). The
            chat leg is the tool-selection round-trip; the synthesize leg
            (the user-facing prose author) always stays on "standard".
            Defaults preserve the AG-UI brain byte-for-byte — only the
            #1336 conversational bridge passes "fast", on measured evidence
            (2026-08-04, same box/key/prompt/tools): the chat leg emitted
            0 content chars and only a tool call on every bridged probe,
            haiku-4-5 selected the identical tool with equivalent args on
            3/3 real bridged queries at 1.17-1.33s vs sonnet-5's 3.12-5.79s.
        chat_llm_reasoning_effort: reasoning effort for the chat leg.
            The bridge passes "none" (thinking measured immaterial for tool
            selection: medium 2.9-3.2s vs none 2.5-2.6s on sonnet; ignored
            entirely by models without a thinking control surface).

    Returns:
        Compiled LangGraph with chat → tools → chat loop
    """

    def should_continue(state: E2IAgentState) -> str:
        """Determine if we should continue to tools or end.

        Delegates to module-level _route_after_chat (v1.30.0): frontend-action
        calls end the run for client-side execution (generative UI); backend
        tool calls go to ToolNode as before.
        """
        return _route_after_chat(state)

    async def chat_node(state: E2IAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Process chat messages using Claude with bound tools."""
        from datetime import datetime

        node_start = time.time()

        # Log chat_node invocation
        logger.debug(f"[CopilotKit] chat_node CALLED at {datetime.now()}")
        logger.debug(
            f"[CopilotKit] chat_node state keys: {list(state.keys()) if state else 'None'}"
        )

        # CoAgent State Sync: Emit initial state for real-time UI progress
        state["current_node"] = "chat"
        state["progress_steps"] = ["Processing your query..."]
        state["progress_percent"] = 25
        state["agent_status"] = "processing"
        state["tools_executing"] = []
        state["error_message"] = None
        try:
            await copilotkit_emit_state(config, state)
        except Exception as e:
            logger.debug(f"[CopilotKit] State emission skipped (not in CoAgent context): {e}")

        messages = state.get("messages", [])

        # Get session_id with priority: context var (most reliable) > state > config
        # Context var is set in execute() and persists across async boundaries
        session_id = _session_id_context.get()
        session_id_source = "context_var" if session_id else None

        if not session_id:
            session_id = state.get("session_id")
            session_id_source = "state" if session_id else None

        if not session_id:
            configurable = config.get("configurable", {}) if config else {}
            session_id = configurable.get("thread_id") if configurable else None
            session_id_source = "config.thread_id" if session_id else None

        logger.info(
            f"[CopilotKit] chat_node: {len(messages)} messages, session_id={session_id[:20] if session_id else 'None'}... (source={session_id_source})"
        )

        # Get the last human message
        last_human_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_message = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "user":
                last_human_message = msg.get("content", "")
                break

        # Persist user message to database
        user_message_id = None
        logger.debug(
            f"Persistence check: last_human_message={last_human_message[:50] if last_human_message else 'None'}..., session_id={session_id[:20] if session_id else 'None'}... (source={session_id_source})"
        )
        if last_human_message and session_id:
            try:
                # Ensure conversation exists before inserting messages (FK constraint)
                conv_exists = await _ensure_conversation_exists(session_id)
                logger.debug(f"Conversation exists: {conv_exists}")
                if conv_exists:
                    # Use synchronous helper to persist message (supabase-py is sync)
                    result = _persist_message_sync(
                        session_id=session_id,
                        role="user",
                        content=last_human_message,  # type: ignore[arg-type]
                        metadata={"source": "copilotkit"},
                        run_id=state.get("run_id"),
                    )
                    if result:
                        user_message_id = result.get("id")
                        logger.debug(f"[CopilotKit] Persisted user message id={user_message_id}")
            except Exception as e:
                logger.error(f"[CopilotKit] Failed to persist user message: {e}", exc_info=True)

        # If no user message, return greeting
        if not last_human_message:
            greeting = "Hello! I'm the E2I Analytics Assistant. I can help you with KPI analysis, causal inference, and insights for pharmaceutical brands. What would you like to know?"
            await copilotkit_emit_message(config, greeting)
            # Persist greeting message (ensure conversation exists first)
            if session_id:
                try:
                    conv_exists = await _ensure_conversation_exists(session_id)
                    if conv_exists:
                        _persist_message_sync(
                            session_id=session_id,
                            role="assistant",
                            content=greeting,
                            agent_name="copilotkit",
                            metadata={"source": "copilotkit", "type": "greeting"},
                            run_id=state.get("run_id"),
                        )
                except Exception as e:
                    logger.warning(f"[CopilotKit] Failed to persist greeting: {e}")
            return {"messages": [AIMessage(content=greeting)]}

        # Get LLM via factory — use Anthropic for reliable tool calling
        try:
            # 8192 + effort: sonnet-5's adaptive thinking counts against
            # max_tokens; at 2048 a thinking pass can consume the entire budget
            # and stream zero text (2026-07-20 frozen-at-75% incident).
            # #1475: tier is a graph-construction parameter — "standard" for
            # the AG-UI brain, "fast" for the bridge's tool-selection leg.
            llm = get_chat_llm(
                model_tier=chat_llm_tier,
                max_tokens=8192,
                temperature=0.3,
                provider="anthropic",
                reasoning_effort=chat_llm_reasoning_effort,
            )
            provider = "anthropic"
            # #1257 provenance class: configured_model metadata must name the
            # model that actually ran this leg, never a hardcoded 'standard'.
            chat_leg_model = f"{provider}:{MODEL_MAPPINGS[provider][chat_llm_tier]}"
            logger.info(f"[CopilotKit] Using {provider} LLM for chat")

            # Bind E2I tools to LLM with tool_choice="auto" to encourage tool use.
            # v1.30.0: also bind the frontend actions riding state (generative UI —
            # renderKpiTrend etc.); without this the model can never call them.
            frontend_action_schemas = _frontend_action_schemas(state)
            if frontend_action_schemas:
                bound_action_names = [s["function"]["name"] for s in frontend_action_schemas]
                logger.info(
                    f"[CopilotKit] Binding {len(frontend_action_schemas)} frontend "
                    f"action(s): {bound_action_names}"
                )
            llm_with_tools = llm.bind_tools(
                [*E2I_CHATBOT_TOOLS, *frontend_action_schemas], tool_choice="auto"
            )

            # Build messages for LLM
            system_msg = SystemMessage(content=E2I_COPILOT_SYSTEM_PROMPT)
            llm_messages: list[BaseMessage] = [system_msg]

            # Add conversation history (convert to LangChain format if needed)
            for msg in messages:
                if isinstance(msg, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
                    llm_messages.append(msg)
                elif isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        llm_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        llm_messages.append(AIMessage(content=content))

            logger.info(
                f"[CopilotKit] Invoking {provider} LLM with {len(llm_messages)} messages and {len(E2I_CHATBOT_TOOLS)} tools bound"
            )

            # STREAMING IMPLEMENTATION (v1.24.0)
            # FIX: Don't emit content during streaming if tool calls may follow.
            # Previously (v1.22.0), we emitted each chunk immediately, but if the LLM
            # decides to call tools, this creates a partial message that gets its own
            # action bar, and then synthesize_node creates another message with ANOTHER
            # action bar. Root cause of "multiple action bars" bug.
            #
            # New approach:
            # 1. Accumulate ALL content and tool_calls during streaming
            # 2. After streaming completes, check if tool_calls exist
            # 3. Only emit content if NO tool calls (direct response)
            # 4. If tool calls exist, don't emit - synthesize_node will handle final response
            full_content = ""
            accumulated_tool_calls: list[dict[str, Any]] = []
            content_chunks = []  # Buffer chunks for potential later emission
            response = None

            logger.debug(
                f"[CopilotKit] Starting streaming LLM response with {len(llm_messages)} messages"
            )
            async for chunk in llm_with_tools.astream(llm_messages):
                # Accumulate content chunks (DON'T emit yet - wait to check for tool calls)
                if hasattr(chunk, "content") and chunk.content:
                    # Anthropic returns content as list of blocks, OpenAI returns str
                    chunk_text = chunk.content
                    if isinstance(chunk_text, list):
                        chunk_text = "".join(
                            block.get("text", "") if isinstance(block, dict) else str(block)
                            for block in chunk_text
                        )
                    full_content += chunk_text
                    content_chunks.append(chunk_text)
                    logger.debug(f"[CopilotKit] Accumulated chunk: {len(chunk_text)} chars")

                # Accumulate tool calls (they may come in chunks) — tool_call_chunks
                # is authoritative, chunk.tool_calls is a merge-by-id fallback; see
                # _accumulate_tool_call_event for why (v1.29.0 ghost-call fix).
                _accumulate_tool_call_event(accumulated_tool_calls, chunk)

                # Keep last chunk as final response
                response = chunk

            logger.debug(
                f"[CopilotKit] Streaming complete: {len(full_content)} chars, {len(accumulated_tool_calls)} tool calls"
            )

            # Build final AIMessage with accumulated content and tool calls
            if accumulated_tool_calls or full_content:
                logger.debug(
                    f"[CopilotKit] Accumulated tool calls before parsing: {accumulated_tool_calls}"
                )

                # Parse accumulated entries into executable calls (args_str preferred,
                # nameless dropped, duplicate ids collapsed — v1.29.0 ghost-call fix).
                parsed_tool_calls = _finalize_tool_calls(accumulated_tool_calls)

                # v1.30.0: a prompt-discouraged mixed backend+frontend turn keeps
                # only the backend calls — ToolNode has no frontend implementations.
                parsed_tool_calls = _strip_frontend_calls_when_mixed(
                    parsed_tool_calls, _frontend_action_names(state)
                )

                response = AIMessage(
                    content=full_content,
                    tool_calls=parsed_tool_calls if parsed_tool_calls else [],
                )

            # FIX (v1.25.1): Strengthen tool_call detection to prevent race condition
            # Previously only checked response.tool_calls, but due to streaming this can be
            # empty [] even when accumulated_tool_calls has entries (name comes before args).
            # Now we also check accumulated_tool_calls directly as a fallback.
            has_tool_calls = (getattr(response, "tool_calls", None) and response.tool_calls) or any(  # type: ignore[union-attr]
                tc.get("name") or tc.get("id") for tc in accumulated_tool_calls
            )

            # If response has tool calls, return without additional emit (tools node will handle)
            if has_tool_calls:
                # Get tool names from response.tool_calls if available, else from accumulated
                tool_names = (
                    [tc["name"] for tc in response.tool_calls if tc.get("name")]  # type: ignore[union-attr]
                    if response.tool_calls  # type: ignore[union-attr]
                    else [
                        tc.get("name", "unknown")
                        for tc in accumulated_tool_calls
                        if tc.get("name") or tc.get("id")
                    ]
                )
                logger.info(f"[CopilotKit] Claude invoked tools: {tool_names}")

                # CoAgent State Sync: Emit tools executing state
                state["current_node"] = "tools"
                state["progress_steps"].append(f"Executing {len(tool_names)} tool(s)...")
                state["progress_percent"] = 50
                state["tools_executing"] = tool_names
                try:
                    await copilotkit_emit_state(config, state)
                except Exception as e:
                    logger.debug(f"[CopilotKit] State emission skipped: {e}")

                # Persist tool call request (for tracking)
                if session_id:
                    try:
                        _persist_message_sync(
                            session_id=session_id,
                            role="assistant",
                            agent_name="copilotkit",
                            content="",  # No content yet, tools will respond
                            metadata={
                                "source": "copilotkit",
                                "type": "tool_request",
                                "tool_calls": [
                                    {"name": tc["name"], "args": tc.get("args", {})}
                                    for tc in response.tool_calls  # type: ignore[union-attr]
                                ],
                                "configured_model": chat_leg_model,
                            },
                            run_id=state.get("run_id"),
                        )
                    except Exception as e:
                        logger.warning(f"[CopilotKit] Failed to persist tool call: {e}")

                # Analytics for a generative-UI turn.
                #
                # This node returns early for EVERY tool turn, before the
                # direct-response _record_analytics_sync below. A backend-tool
                # turn is still recorded later, by synthesize_node once the
                # results come back. A turn whose calls are ALL frontend
                # actions has no such second act: _route_after_chat sends it
                # straight to END for client-side execution, so nothing
                # downstream ever records it and the turn is invisible in
                # chat_analytics — no row, not even an empty one.
                #
                # That blind spot is the whole reason "which chart action do
                # users actually get?" could not be answered from the data.
                # Recording here closes it, and `tools_invoked` carries the
                # action name so the split between the chart actions is
                # readable straight off the existing column.
                if session_id and _is_frontend_only_turn(tool_names, _frontend_action_names(state)):
                    _record_analytics_sync(
                        session_id=session_id,
                        query_type=_classify_query_type(last_human_message or ""),  # type: ignore[arg-type]
                        response_time_ms=int((time.time() - node_start) * 1000),
                        tools_invoked=tool_names,
                        primary_agent="copilotkit",
                        metadata={
                            "configured_model": chat_leg_model,
                            # Distinguishes this from both a direct answer
                            # (direct_response) and a backend-tool turn
                            # (tools_used), which are the other two shapes a
                            # chat_analytics row can take.
                            "generative_ui": True,
                            "tool_count": len(tool_names),
                        },
                    )

                return {"messages": [response]}

            # FIX (v1.24.0): NOW emit buffered content since we confirmed no tool calls
            # This is a direct text response, so stream the accumulated chunks
            if full_content and content_chunks:
                logger.debug(
                    f"[CopilotKit] Emitting {len(content_chunks)} buffered chunks (no tool calls)"
                )
                for chunk_content in content_chunks:
                    await copilotkit_emit_message(config, chunk_content)
                logger.debug("[CopilotKit] Finished emitting buffered content")

            if full_content:
                # Persist assistant response to database
                if session_id:
                    try:
                        elapsed_ms = int((time.time() - node_start) * 1000)
                        persisted = _persist_message_sync(
                            session_id=session_id,
                            role="assistant",
                            content=response.content,  # type: ignore[union-attr]
                            agent_name="copilotkit",
                            metadata={
                                "source": "copilotkit",
                                "configured_model": chat_leg_model,
                                "latency_ms": elapsed_ms,
                            },
                            run_id=state.get("run_id"),
                        )
                        # #1405: only claim persistence when a row actually came back —
                        # the old unconditional info log overstated persistence on a None result.
                        if persisted:
                            logger.debug(
                                f"[CopilotKit] Persisted assistant message id={persisted.get('id')}"
                            )
                        else:
                            logger.warning(
                                "[CopilotKit] Assistant message not persisted (no row returned)"
                            )
                    except Exception as e:
                        logger.warning(f"[CopilotKit] Failed to persist assistant message: {e}")

            elapsed = time.time() - node_start
            elapsed_ms = int(elapsed * 1000)
            logger.info(f"[CopilotKit] chat_node completed in {elapsed:.2f}s")

            # Record analytics (P7.1)
            if session_id:
                _record_analytics_sync(
                    session_id=session_id,
                    query_type=_classify_query_type(last_human_message or ""),  # type: ignore[arg-type]
                    response_time_ms=elapsed_ms,
                    tools_invoked=[],
                    primary_agent="copilotkit",
                    metadata={
                        "configured_model": chat_leg_model,
                        "direct_response": True,
                    },
                )

            # #1240: a completed direct-answer turn is graded learning
            # substrate for the Tier-5 feedback learner (no tools invoked).
            if full_content:
                await _collect_copilot_learning_signal(
                    query=last_human_message or "",  # type: ignore[arg-type]
                    response=full_content,
                    tool_names=[],
                    conversation_id=session_id,
                )

            # CoAgent State Sync: Emit completion state for direct responses
            state["current_node"] = "idle"
            state["progress_steps"].append("Response complete")
            state["progress_percent"] = 100
            state["agent_status"] = "complete"
            try:
                await copilotkit_emit_state(config, state)
            except Exception as e:
                logger.debug(f"[CopilotKit] State emission skipped: {e}")

            return {"messages": [response]}

        except Exception as e:
            logger.error(f"[CopilotKit] LLM invocation failed: {e}", exc_info=True)
            fallback = generate_e2i_response(last_human_message)  # type: ignore[arg-type]
            await copilotkit_emit_message(config, fallback)
            # Emit a TERMINAL state so the UI progress does not freeze. The chat
            # node set progress_percent=25 ("Processing your query...") on entry;
            # without this, an LLM failure left it stuck at 25% forever (the
            # success path resets to 100, the error path used to skip it). A
            # response WAS delivered (the fallback), so this is "complete", with
            # the underlying error recorded for support.
            state["current_node"] = "idle"
            state["progress_percent"] = 100
            state["agent_status"] = "complete"
            state["error_message"] = str(e)
            try:
                await copilotkit_emit_state(config, state)
            except Exception as emit_err:
                logger.debug(f"[CopilotKit] Terminal state emission skipped: {emit_err}")
            # Persist fallback response
            if session_id:
                try:
                    _persist_message_sync(
                        session_id=session_id,
                        role="assistant",
                        content=fallback,
                        agent_name="copilotkit",
                        metadata={"source": "copilotkit", "type": "fallback", "error": str(e)},
                        run_id=state.get("run_id"),
                    )
                except Exception as persist_err:
                    logger.warning(f"[CopilotKit] Failed to persist fallback: {persist_err}")
            return {"messages": [AIMessage(content=fallback)]}

    async def synthesize_node(state: E2IAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Synthesize tool results into a final response."""

        node_start = time.time()

        # CoAgent State Sync: Emit synthesizing state
        state["current_node"] = "synthesize"
        if "progress_steps" not in state or not state.get("progress_steps"):
            state["progress_steps"] = []
        state["progress_steps"].append("Synthesizing tool results...")
        state["progress_percent"] = 75
        state["agent_status"] = "processing"
        try:
            await copilotkit_emit_state(config, state)
        except Exception as e:
            logger.debug(f"[CopilotKit] State emission skipped: {e}")

        messages = state.get("messages", [])
        session_id = state.get("session_id")  # For message persistence

        # Get tool results from messages
        tool_results = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_results.append({"tool": msg.name, "result": msg.content})

        if not tool_results:
            return {"messages": []}

        # Collect the assistant's tool-CALL args (from the AIMessage(s) that
        # requested the tools). The synthesizer needs the brand/window/args the
        # assistant already chose so it answers the actual question instead of
        # re-asking for a brand the user already gave.
        tool_calls: list[dict] = []
        for msg in messages:
            tc = getattr(msg, "tool_calls", None)
            if tc:
                for c in tc:
                    tool_calls.append(
                        {
                            "name": c.get("name")
                            if isinstance(c, dict)
                            else getattr(c, "name", None),
                            "args": c.get("args")
                            if isinstance(c, dict)
                            else getattr(c, "args", None),
                        }
                    )

        # Extract original query for analytics classification
        original_query = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                original_query = msg.content if isinstance(msg.content, str) else str(msg.content)

        # Persist tool results
        if session_id and tool_results:
            try:
                _persist_message_sync(
                    session_id=session_id,
                    role="tool",
                    content=json.dumps(tool_results, default=str),
                    metadata={
                        "source": "copilotkit",
                        "type": "tool_results",
                        "tool_results": tool_results,
                    },
                    run_id=state.get("run_id"),
                )
            except Exception as e:
                logger.warning(f"[CopilotKit] Failed to persist tool results: {e}")

        # Get LLM via factory to synthesize tool results — use Anthropic for consistency
        try:
            # 8192 + effort: sonnet-5's adaptive thinking counts against
            # max_tokens; at 2048 a thinking pass can consume the entire budget
            # and stream zero text (2026-07-20 frozen-at-75% incident).
            llm = get_chat_llm(
                model_tier="standard",
                max_tokens=8192,
                temperature=0.3,
                provider="anthropic",
                reasoning_effort="medium",
            )
            provider = "anthropic"
            logger.info(f"[CopilotKit] Using {provider} LLM for synthesis")

            # Ask LLM to synthesize the results — frame the prior conversation,
            # the user's question, and the tool-call args (brand/window) so the
            # synthesizer answers the actual question, resolves follow-up
            # references, and never re-asks for a brand it already used.
            synthesis_prompt = build_synthesis_prompt(
                original_query,
                tool_calls,
                tool_results,
                history=_extract_synthesis_history(messages),
            )

            # STREAMING (v1.22.0): Stream synthesis response token-by-token
            full_content = ""
            logger.debug("[CopilotKit] Starting streaming synthesis response")
            async for chunk in llm.astream(
                [
                    SystemMessage(content=E2I_COPILOT_SYSTEM_PROMPT),
                    HumanMessage(content=synthesis_prompt),
                ]
            ):
                if hasattr(chunk, "content") and chunk.content:
                    # Anthropic returns content as list of blocks, OpenAI returns str
                    chunk_text = chunk.content
                    if isinstance(chunk_text, list):
                        chunk_text = "".join(
                            block.get("text", "") if isinstance(block, dict) else str(block)
                            for block in chunk_text
                        )
                    full_content += chunk_text
                    await copilotkit_emit_message(config, chunk_text)
                    logger.debug(f"[CopilotKit] Streamed synthesis chunk: {len(chunk_text)} chars")

            logger.debug(f"[CopilotKit] Synthesis streaming complete: {len(full_content)} chars")

            if not full_content.strip():
                # Fail LOUD: a 0-char synthesis was once persisted as success
                # and froze the chat at 75% (2026-07-20 incident — thinking
                # consumed the whole max_tokens budget before any text). Route
                # to the tool-dump fallback below instead.
                raise RuntimeError(
                    "synthesis stream produced no text "
                    "(model thinking may have consumed the max_tokens budget)"
                )

            # #1691: the synthesis already streamed, so a superlative that
            # contradicts the answer's own table cannot be rewritten away —
            # append a deterministic correction note instead (visible-tier
            # findings only; the rest are logged for monitoring).
            guard_findings = find_superlative_contradictions(full_content)
            if guard_findings:
                logger.warning(
                    "[CopilotKit] #1691 superlative guard: %s",
                    [
                        f"{f.keyword} {f.number_text} vs {f.column_header} "
                        f"[{f.column_min:g}..{f.column_max:g}] visible={f.visible}"
                        for f in guard_findings
                    ],
                )
                guard_note = build_superlative_correction(full_content)
                if guard_note:
                    await copilotkit_emit_message(config, guard_note)
                    full_content += guard_note

            response = AIMessage(content=full_content)

            # Persist synthesized response
            if session_id:
                try:
                    elapsed_ms = int((time.time() - node_start) * 1000)
                    _persist_message_sync(
                        session_id=session_id,
                        role="assistant",
                        content=response.content,  # type: ignore[arg-type]
                        agent_name="copilotkit",
                        metadata={
                            "source": "copilotkit",
                            "type": "synthesis",
                            "configured_model": f"{provider}:{MODEL_MAPPINGS[provider]['standard']}",
                            "tool_results": tool_results,
                            "latency_ms": elapsed_ms,
                        },
                        run_id=state.get("run_id"),
                    )
                    logger.info("[CopilotKit] Persisted synthesized message")
                except Exception as e:
                    logger.warning(f"[CopilotKit] Failed to persist synthesized message: {e}")

                # Record analytics (P7.1) for tool-using queries
                _record_analytics_sync(
                    session_id=session_id,
                    query_type=_classify_query_type(original_query),
                    response_time_ms=elapsed_ms,
                    tools_invoked=[tr["tool"] for tr in tool_results],  # type: ignore[misc]
                    primary_agent="copilotkit",
                    metadata={
                        "configured_model": f"{provider}:{MODEL_MAPPINGS[provider]['standard']}",
                        "tools_used": True,
                        "tool_count": len(tool_results),
                    },
                )

            # #1240: a completed tool-grounded turn is graded learning
            # substrate for the Tier-5 feedback learner.
            await _collect_copilot_learning_signal(
                query=original_query,
                response=full_content,
                tool_names=[tr["tool"] for tr in tool_results],  # type: ignore[misc]
                conversation_id=session_id,
                evidence_tool_count=_evidence_tool_count(tool_results),
            )

            # CoAgent State Sync: Emit completion state
            state["current_node"] = "idle"
            state["progress_steps"].append("Response complete")
            state["progress_percent"] = 100
            state["agent_status"] = "complete"
            state["tools_executing"] = []
            try:
                await copilotkit_emit_state(config, state)
            except Exception as e:
                logger.debug(f"[CopilotKit] State emission skipped: {e}")

            return {"messages": [response]}

        except Exception as e:
            logger.error(f"[CopilotKit] Synthesis failed: {e}")
            result_text = "\n\n".join([f"**{tr['tool']}**: {tr['result']}" for tr in tool_results])
            await copilotkit_emit_message(config, result_text)
            # Terminal state: the fallback IS a delivered response — without
            # this the progress UI stays frozen at 75% even though text
            # arrived (chat_node's except path got the same fix for its 25%
            # stage).
            state["current_node"] = "idle"
            state["progress_percent"] = 100
            state["agent_status"] = "complete"
            state["error_message"] = str(e)
            try:
                await copilotkit_emit_state(config, state)
            except Exception as emit_err:
                logger.debug(f"[CopilotKit] Terminal state emission skipped: {emit_err}")
            # Persist error fallback
            if session_id:
                try:
                    _persist_message_sync(
                        session_id=session_id,
                        role="assistant",
                        content=result_text,
                        agent_name="copilotkit",
                        metadata={
                            "source": "copilotkit",
                            "type": "synthesis_error",
                            "error": str(e),
                            "tool_results": tool_results,
                        },
                        run_id=state.get("run_id"),
                    )
                except Exception as persist_err:
                    logger.warning(f"[CopilotKit] Failed to persist error fallback: {persist_err}")

                # Record analytics (P7.1) for failed synthesis
                elapsed_ms = int((time.time() - node_start) * 1000)
                _record_analytics_sync(
                    session_id=session_id,
                    query_type=_classify_query_type(original_query),
                    response_time_ms=elapsed_ms,
                    tools_invoked=[tr["tool"] for tr in tool_results],  # type: ignore[misc]
                    primary_agent="copilotkit",
                    error_occurred=True,
                    error_type="synthesis_error",
                    metadata={
                        "error": str(e),
                        "tools_used": True,
                        "tool_count": len(tool_results),
                    },
                )

            # #1240: grade the degraded turn too — skipping failures would
            # select only good turns into the learning substrate. The raw
            # tool-dump fallback forfeits the synthesis base reward.
            await _collect_copilot_learning_signal(
                query=original_query,
                response=result_text,
                tool_names=[tr["tool"] for tr in tool_results],  # type: ignore[misc]
                conversation_id=session_id,
                synthesis_error=True,
                evidence_tool_count=_evidence_tool_count(tool_results),
            )
            return {"messages": [AIMessage(content=result_text)]}

    # Build the graph with tool calling support
    workflow = StateGraph(E2IAgentState)

    # Add nodes. "chat" and "synthesize" are the ONLY answer-producing nodes;
    # `_ANSWER_NODE_NAMES` allow-lists them so every other chat-model stream
    # (this graph's tools node, and any nested graph reached through it) is
    # suppressed before AG-UI translation (#1636). Adding an answer-producing
    # node here without adding it there would silently mute it — the assertion
    # in test_copilotkit_classifier_stream_leak_1636.py fails loudly if the two
    # drift apart.
    workflow.add_node("chat", chat_node)
    workflow.add_node(_TOOL_NODE_NAME, ToolNode(E2I_CHATBOT_TOOLS))
    workflow.add_node("synthesize", synthesize_node)

    # Set entry point
    workflow.set_entry_point("chat")

    # Add conditional edge: chat → tools or end
    workflow.add_conditional_edges(
        "chat",
        should_continue,
        {
            "tools": _TOOL_NODE_NAME,
            "end": END,
        },
    )

    # After tools, synthesize the results
    workflow.add_edge(_TOOL_NODE_NAME, "synthesize")
    workflow.add_edge("synthesize", END)

    # Checkpointer is required by ag_ui_langgraph for state management
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def generate_e2i_response(query: str) -> str:
    """Honest fallback used ONLY when the LLM chat path raises.

    The chat node (``chat_node``) drives the real Anthropic-backed LLM, which
    fetches live data through the bound tools and synthesizes the answer. This
    function is reached ONLY in the ``except`` arm of that call — i.e. the model
    invocation actually failed (provider outage, auth/model error, timeout).

    It deliberately does NOT fabricate a data answer or echo a generic
    capability list dressed up as a reply. The previous implementation
    keyword-matched the query and returned canned text like "Use the
    **getKPISummary** action…", which (a) referenced internal action names the
    user can't invoke, (b) read like a confident real answer, and (c) silently
    masked the underlying failure — exactly the "not optimal" response users hit
    when the configured LLM model 404'd. Honesty over plausibility: return a
    short "temporarily unavailable" message that names the situation and invites
    a retry, fabricating nothing. ``query`` is intentionally unused — there is no
    honest query-specific answer to give once the reasoning backend is down.
    """
    return (
        "⚠️ I couldn't complete that request just now — the analytics assistant "
        "hit a temporary error reaching its data and reasoning backend, so I don't "
        "have a reliable answer to give you yet. Please try again in a moment. If "
        "this keeps happening, the assistant's LLM connection likely needs "
        "attention (it isn't a problem with your question)."
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

    logger.info(
        f"[CopilotKit] Remote endpoint initialized with 1 agent and {len(COPILOT_ACTIONS)} actions"
    )
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
        agents_dict[agent_id] = {"description": getattr(agent, "description", "") or ""}

    return {
        "actions": actions_list,
        "agents": agents_dict,
        "version": COPILOTKIT_SDK_VERSION,  # Frontend expects 'version' not 'sdkVersion'
    }


async def _require_auth_for_copilotkit_execution(request: Request) -> Dict[str, Any]:
    """Validate JWT for CopilotKit execution paths.

    Mirrors ``src.api.dependencies.auth.require_auth`` but callable
    directly from inside ``copilotkit_custom_handler``. It is invoked
    from TWO sites in that handler, together covering every execution
    surface:

      * the POST-to-root branch (``path in ("", "info")``), where the
        public ``/api/copilotkit`` + ``/info`` allowlist entries permit
        unauthenticated DISCOVERY requests (empty body / ``{}`` /
        ``{"method":"info"}``) — this gate runs only after the
        discovery shapes have returned, so it covers execution BODIES
        (``agent/run``, ``action/run``, ``agent/connect``); and
      * the sub-path fallthrough (#1432), where execution/state URL
        paths (``agent/{name}``, ``agent/{name}/state``,
        ``action/{name}``, ``agents/execute``, ...) reach the SDK
        handler — gated there before delegation so the "anything else
        before the SDK handler" contract holds for those paths too.

    Closes #399 codex iter-0 H1+H2: path-based middleware allowlist
    alone is insufficient because the CopilotKit JSON-RPC protocol
    mixes discovery and execution under the same paths via the
    ``method`` field in the request body. Body-aware auth lives where
    the body is already being inspected — inside the handler. #1432
    extends the same in-handler gate to the ``agent/{name}`` fallthrough
    (defense-in-depth behind the middleware allowlist, which already
    marks those sub-paths non-public).

    Args:
        request: FastAPI request; ``Authorization`` header is read.

    Returns:
        User dict (mirror of ``require_auth`` return shape).

    Raises:
        AuthError: If no token, malformed Authorization header, or
            invalid/expired JWT. The caller is expected to catch this
            and convert to a 401 JSONResponse (see usage in
            ``copilotkit_custom_handler``).
    """
    if TESTING_MODE:
        request.state.user = TEST_USER
        set_authenticated_user(TEST_USER.get("id"))
        return TEST_USER

    auth_header = request.headers.get("Authorization", "")
    parts = auth_header.split() if auth_header else []
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthError("Missing or invalid Authorization header for execution endpoint")

    token = parts[1]
    user = await verify_supabase_token(token)
    if user is None:
        raise AuthError("Invalid or expired token")

    request.state.user = user
    # CopilotKit threadIds are bare UUIDs, so the adapter's session-prefix
    # derivation can never attribute chat usage; stash the verified identity
    # for set_chat_attribution's fallback (contextvar rides the request task
    # into the streaming/adapter context, same channel as _session_id_context).
    set_authenticated_user(user.get("id"))
    return user


#: Anything ``copilotkit_custom_handler`` or the delegated ``sdk_handler`` may
#: return. Bound as a TypeVar so ``_bound_silent_window`` is transparent to the
#: caller's declared return type instead of widening it to ``Any``.
_SDKResponse = TypeVar("_SDKResponse", bound=Response)


def _bound_silent_window(response: _SDKResponse) -> _SDKResponse:
    """Bound the SILENT windows of a streaming response the SDK handler built.

    TWO independent windows of silence, two independent fixes, same seam:

    * **#1669 — the window nginx kills.** Nothing in ``ag_ui_langgraph``
      heartbeats a quiet graph, so a long tool call produces a silent stretch
      with no ceiling of its own, against ``proxy_read_timeout 300s``. Bounded
      by wrapping the body in ``with_sse_keepalive``.
    * **#1673 — the window the USER sees.** nginx delivered the entire turn in a
      single flush at the very end: measured on production ``TTFB 8.754s`` of an
      ``8.756s`` turn (ttfb/total = 0.9998, 2 chunks), against ``TTFB 0.263s`` of
      a ``7.709s`` turn (81 chunks) direct to the app. The answer was correct and
      the status was 200 — token streaming was produced here and destroyed at the
      proxy. Bounded by ``X-Accel-Buffering: no``.

    They are orthogonal: the keepalive resets nginx's upstream read timer whether
    or not the response is buffered, and the header changes nothing about how
    long the graph stays quiet. Dropping either while keeping the other looks
    correct in isolation, which is why
    ``tests/unit/test_api/test_agui_accel_buffering_1673.py`` asserts both
    survive together.

    WHY ``X-Accel-Buffering`` FIXES A GZIP PROBLEM
    ----------------------------------------------
    #1673 attributed the buffering to ``proxy_buffering on``. Measurement says
    otherwise — same production URL, same nginx, same ``location /api/``,
    differing only in ``Accept-Encoding``::

        gzip       TTFB 8.754s / total 8.756s   ratio 0.9998    2 chunks
        identity   TTFB 0.683s / total 9.218s   ratio 0.0741   99 chunks

    nginx forwards proxy buffers as they fill; it is the **gzip filter** that
    holds the whole turn, because it emits only when its deflate buffer fills or
    a flush marker arrives, and ``proxy_buffering on`` produces no flush markers.
    The stream is gzip-eligible only because ``handle_execute_agent`` labels it
    ``media_type="application/json"``, which is in the live ``gzip_types`` — the
    two responses this module builds itself say ``text/event-stream``, which is
    not, and escape by accident rather than by design.

    That the header nevertheless defeats gzip buffering is the single
    load-bearing assumption of this fix, so it was measured rather than reasoned:
    on a replica nginx carrying the production gzip and ``proxy_buffer*``
    directives verbatim, ``application/json`` + gzip + this header streamed in
    **61 chunks at ttfb/total 0.0071 while still ``Content-Encoding: gzip``**,
    against 1 chunk at 0.9999 without it. Disabling proxy buffering makes nginx
    flag each upstream read with ``flush``, and the gzip filter honours flush
    with a ``Z_SYNC_FLUSH``.

    WHY THIS IS NOT AN NGINX-CONFIG CHANGE
    --------------------------------------
    #1673 suggested ``location /copilotkit/`` in ``docker/nginx/host-nginx.conf``.
    That block is dead — it proxies to ``127.0.0.1:8000/copilotkit/``, a prefix
    this app does not serve, and authenticated probes return ``404
    EndpointNotFoundError`` on every path through it. The live surface is
    ``location /api/``: the shipped frontend bundle bakes ``apiUrl:"/api"`` and
    builds its runtime URL as ``${apiUrl}/copilotkit/``, and
    ``scripts/demos/copilot_agui_runner.py`` defaults to
    ``--api-base https://eznomics.site/api``. ``location /api/`` carries the whole
    REST API, so disabling gzip or buffering there to fix one streaming endpoint
    would de-optimise every JSON response the platform sends. The nginx config
    also reaches production only via a manual root ``cp`` + ``nginx -t`` +
    ``systemctl reload`` (``docs/runbooks/frontend-serving-flip.md``) — no
    workflow ships it, ``/etc/nginx/sites-enabled/e2i-analytics`` is a regular
    file rather than a symlink, and ``sites-available`` currently holds a
    different, older copy. This header ships with the ordinary image deploy.

    The AG-UI surface has two entry points and only one of them constructs its
    own ``StreamingResponse`` in this module:

    * ``POST /api/copilotkit`` with body ``{"method": "agent/run"}`` -> the
      custom ``stream_agent_events`` branch above, wrapped inline at its
      ``StreamingResponse(...)`` call;
    * ``POST /api/copilotkit/agent/{name}`` -> delegated to the third-party
      ``sdk_handler``, which builds ``StreamingResponse(events,
      media_type="application/json")`` itself
      (``copilotkit/integrations/fastapi.py::handle_execute_agent``). We cannot
      pass a wrapper into a response we do not construct, so the body iterator
      is replaced on the way out.

    Both carry the SAME bytes: ``copilotkit/sdk.py::execute_agent`` returns this
    module's ``LangGraphAgent.execute`` unchanged, and ``execute`` yields
    ``data: {json}\\n\\n`` records. The SDK's ``application/json`` media type is
    therefore a mislabel, not a different protocol.

    That the keepalive is safe on the mislabelled branch was VERIFIED against
    the real client, not inferred from the SSE spec.
    ``@copilotkit/react-core@1.51.2`` reads this stream via
    ``@ag-ui/client@0.0.42``, whose ``transformHttpEventStream`` routes any
    content-type OTHER than ``application/vnd.ag-ui.event+proto`` to its SSE
    parser, and whose SSE parser collects only lines matching ``startsWith("data:
    ")`` and calls ``JSON.parse`` only when at least one such line exists. A
    comment-only record is dropped without error. Driving that parser directly
    with the keepalive interleaved, leading, trailing, and split across chunk
    boundaries produced event sequences identical to baseline. See
    ``test_keepalive_frame_is_an_ignorable_sse_comment`` for the full citation.

    Keyed on the response TYPE, not on the path: ``sdk_handler`` also returns
    ``JSONResponse`` (info, agent state, errors) and those must pass through
    untouched — an ordinary JSON body has nothing to gain from either fix and
    genuinely benefits from the proxy buffering the header would disable.

    Why this is not left to ``#1659``'s AST guard: that guard can only see
    literal ``StreamingResponse(...)`` calls, and this one is built inside a
    third-party package. ``tests/unit/test_api/test_agui_stream_health_1667_1669.py``
    covers it behaviourally instead, by draining the real response body.
    ``tests/integration/api/test_agui_edge_buffering_1673.py`` goes one further
    and asserts ``TTFB << total`` through a real nginx, because the buffering
    property does not exist in-process: an ASGI call has no wire.

    No ``type: ignore`` here, deliberately. ``body_iterator`` is an
    ``AsyncIterable[str | bytes | memoryview]`` and ``with_sse_keepalive`` is
    bounded by that same union (``sse_keepalive.SSEFrame``), so the assignment
    type-checks on its own. It previously needed a suppression only because the
    wrapper's TypeVar was over-restricted to ``str`` — a suppression would have
    hidden the mismatch rather than resolved it (#1672 CI).
    """
    if isinstance(response, StreamingResponse):
        response.body_iterator = with_sse_keepalive(response.body_iterator)
        # Mutated in place rather than passed to a constructor for the same
        # reason as the body iterator: this response was built inside
        # ``copilotkit/integrations/fastapi.py``, not here.
        response.headers["X-Accel-Buffering"] = "no"
    return response


async def copilotkit_custom_handler(
    request: Request, sdk: CopilotKitRemoteEndpoint, path: str = ""
) -> JSONResponse | StreamingResponse:
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

    method = request.method

    # Handle GET info request with our custom transformation
    if method == "GET" and path in ("", "info"):
        response = transform_info_response(sdk)
        logger.debug(f"GET info response with agents: {list(response['agents'].keys())}")
        return JSONResponse(content=response)

    # #399 iter-1 H1+H2 closure: the JWTAuthMiddleware allowlist permits
    # ``/api/copilotkit``, ``/api/copilotkit/status``, ``/api/copilotkit/info``
    # for unauthenticated SDK discovery. But the CopilotKit JSON-RPC
    # protocol routes EXECUTION (agent/run, action/run, agent/connect,
    # SDK fallback) via the SAME paths using the request body's
    # ``method`` field. So path-based allowlist alone cannot tell
    # discovery from execution — the handler must.
    #
    # The ``is_info_request`` branch below catches pure-discovery POSTs
    # (empty / {} / {"action":"getInfo"} / {"method":"info"}) and returns
    # before reaching any execution branch. Everything else — agent/run
    # at L2631, agent/connect at L2748, the SDK fallback at L2753 — is
    # execution-shaped and must be auth-gated.
    #
    # The auth check is wired AFTER the discovery branch (so legitimate
    # unauthenticated discovery still works) but BEFORE any execution
    # branch. Mirrors ``require_auth`` semantics: testing-mode bypass,
    # Bearer-token extraction, JWT verification via ``verify_supabase_token``.

    # Handle POST to root or /info - need to check body to determine request type
    # IMPORTANT: Read body FIRST before any other operations that might consume it
    if method == "POST" and path in ("", "info"):
        try:
            # Read body bytes FIRST (only do this once!)
            body_bytes = await request.body()
            body_str = body_bytes.decode("utf-8") if body_bytes else ""
            logger.debug(f"POST body preview: {body_str[:100] if body_str else '(empty)'}...")

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

            logger.debug(f"is_info_request={is_info_request}")

            if is_info_request:
                response = transform_info_response(sdk)
                logger.debug(
                    f"Returning info response with agents: {list(response['agents'].keys())}"
                )
                return JSONResponse(content=response)

            # #399 iter-1: this is an execution-shaped POST to a
            # middleware-public path. Verify JWT before reaching agent/run,
            # agent/connect, or the SDK fallback handler.
            try:
                await _require_auth_for_copilotkit_execution(request)
            except AuthError as auth_exc:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Authentication required for CopilotKit execution endpoints",
                        "detail": str(auth_exc.detail) if auth_exc.detail else str(auth_exc),
                    },
                )

            # Non-info POST request - check AG-UI protocol method
            agui_method = body_json.get("method", "") if body_json else ""
            logger.debug(f"AG-UI method={agui_method}")

            # Handle AG-UI protocol: agent/run
            if agui_method == "agent/run":
                assert body_json is not None
                params = body_json.get("params", {})
                body_data = body_json.get("body", {})
                agent_name = params.get("agentId", "default") or body_json.get(
                    "agentName", "default"
                )

                logger.debug(f"Executing agent '{agent_name}' with AG-UI protocol")

                # Extract parameters - check both nested body and top level (AG-UI protocol varies)
                # Some SDK versions send {"method": "agent/run", "body": {"threadId": ..., "messages": [...]}}
                # Others send {"method": "agent/run", "threadId": ..., "messages": [...]}
                thread_id = (
                    body_data.get("threadId") or body_json.get("threadId") or str(uuid.uuid4())
                )
                state = body_data.get("state") or body_json.get("state") or {}
                messages = body_data.get("messages") or body_json.get("messages") or []
                actions = (
                    body_data.get("tools") or body_json.get("tools") or []
                )  # AG-UI uses "tools"
                node_name = body_data.get("nodeName") or body_json.get("nodeName")

                logger.debug(
                    f"agent/run: thread_id={thread_id[:8]}..., messages={len(messages)}, actions={len(actions)}, node={node_name}"
                )

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
                        status_code=404, content={"error": f"Agent '{agent_name}' not found"}
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
                        """Debug log with wall-clock and elapsed time."""
                        wall = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        elapsed = time.time() - stream_start
                        logger.debug(f"[{wall}] stream [{elapsed:.3f}s]: {msg}")

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
                        import traceback

                        tb_str = traceback.format_exc()
                        sdbg(f"Error: {e}")
                        sdbg(f"Traceback:\n{tb_str}")
                        logger.error(f"[CopilotKit] Stream error: {e}")
                        logger.error(f"[CopilotKit] Stream traceback:\n{tb_str}")
                        # FIX (v1.21.4): Use RUN_ERROR event type (AG-UI protocol)
                        # "error" is not a valid AG-UI event type - causes ZodError on frontend
                        error_event = {
                            "type": "RUN_ERROR",
                            "message": str(e),
                            "code": "STREAM_ERROR",
                        }
                        yield f"data: {json.dumps(error_event)}\n\n"

                    sdbg(f"Stream complete, yielded {event_count} events")

                return StreamingResponse(
                    with_sse_keepalive(stream_agent_events()),
                    media_type="text/event-stream",  # SSE format for CopilotKit SDK
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",  # Disable nginx buffering
                    },
                )

            # Handle AG-UI protocol: agent/connect (just acknowledge)
            if agui_method == "agent/connect":
                logger.debug("agent/connect - acknowledging")
                return JSONResponse(content={"status": "connected"})

            # Fall through to SDK handler for other methods
            logger.debug("Non-info POST, delegating to SDK handler")

            async def receive():
                return {"type": "http.request", "body": body_bytes}

            # Create new request with body restored and path param injected
            # FIX v1.9.3: SDK handler expects path in path_params, but base route has no path param
            scope_with_path = dict(request.scope)
            scope_with_path["path_params"] = {**request.path_params, "path": path}
            new_request = Request(scope_with_path, receive)
            return _bound_silent_window(await sdk_handler(new_request, sdk))  # type: ignore[no-any-return]

        except Exception as e:
            logger.warning(f"[CopilotKit] Error parsing POST body: {e}")
            # Fall through to SDK handler on error

    # #1432: sub-path execution/state endpoints (agent/{name},
    # agent/{name}/state, action/{name}, agents/execute, actions/execute) reach
    # the third-party SDK handler through THIS fallthrough — the POST-to-root
    # gate above only covers ``path in ("", "info")``. The JWTAuthMiddleware
    # allowlist already marks every such sub-path non-public (see
    # ``src/api/middleware/auth_middleware.py`` PUBLIC_PATHS), but that
    # middleware is the ONLY gate today; mirror the root's body-aware auth here
    # so the handler itself enforces auth — defense-in-depth if the allowlist
    # regresses or ``is_auth_enabled()`` is False (misconfig → middleware warns
    # and passes through), and it makes
    # ``_require_auth_for_copilotkit_execution``'s "anything else before the SDK
    # handler" contract actually true. The only public discovery surfaces — GET
    # "" / "info" (handled above) and the static GET /status route (served by
    # the standalone router, never this handler) — do not reach here, so every
    # non-OPTIONS request that does is execution- or state-shaped. OPTIONS (CORS
    # preflight) is exempt, matching the middleware.
    #
    # FAIL-SAFE guard: skip re-auth ONLY when identity is already known-good. A
    # successful ``_require_auth_for_copilotkit_execution`` sets
    # ``request.state.user`` (both the TESTING_MODE and JWT branches), while
    # every failure raises BEFORE any assignment — so ``request.state.user is
    # None`` reliably means "identity not yet established". Guarding on it
    # avoids the duplicate Supabase round-trip when the root-POST branch already
    # authenticated and then fell through here on a later exception, WITHOUT
    # ever letting an unauthenticated sub-path pass: unknown identity ⇒ gate runs.
    if method != "OPTIONS" and getattr(request.state, "user", None) is None:
        try:
            await _require_auth_for_copilotkit_execution(request)
        except AuthError as auth_exc:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authentication required for CopilotKit execution endpoints",
                    "detail": str(auth_exc.detail) if auth_exc.detail else str(auth_exc),
                },
            )

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
    async def receive_fallback():  # noqa: E303
        return {"type": "http.request", "body": body_bytes}

    scope_with_path = dict(request.scope)
    scope_with_path["path_params"] = {**request.path_params, "path": path}
    new_request = Request(scope_with_path, receive_fallback)
    return _bound_silent_window(await sdk_handler(new_request, sdk))  # type: ignore[no-any-return]


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
    # Note: include_in_schema=False excludes dynamic CopilotKit routes from OpenAPI
    # to avoid duplicate operation ID errors during TypeScript type generation
    app.add_api_route(
        normalized_prefix,
        make_handler,
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        name="copilotkit_handler_base",
        include_in_schema=False,
    )

    # Add catch-all route for all CopilotKit sub-paths
    # This matches the SDK's pattern: {prefix}/{path:path}
    app.add_api_route(
        f"{normalized_prefix}/{{path:path}}",
        make_handler,
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        name="copilotkit_handler",
        include_in_schema=False,
    )

    logger.info(
        f"[CopilotKit] Routes added at {normalized_prefix} and {normalized_prefix}/{{path}} (custom handler with info transformation)"
    )


# =============================================================================
# STANDALONE ROUTER (for testing/info endpoints)
# =============================================================================

router = APIRouter(
    prefix="/copilotkit",
    tags=["copilotkit"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


@router.get("/status", summary="Get CopilotKit status", operation_id="get_copilotkit_status")
async def get_copilotkit_status() -> Dict[str, Any]:
    """Get CopilotKit integration status."""
    provider = get_llm_provider()
    return {
        "status": "active",
        "version": "1.1.0",
        "agents_available": 1,
        "agent_names": ["default"],
        "actions_available": len(COPILOT_ACTIONS),
        "action_names": [a.name for a in COPILOT_ACTIONS],
        "llm_provider": provider,
        "llm_model": MODEL_MAPPINGS[provider]["standard"],
        "llm_configured": bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/kpis/summary",
    summary="Get the real business_metrics KPI rollup for a brand",
    operation_id="get_kpi_summary_rest",
)
async def kpi_summary_endpoint(
    brand: str = Query("All"),
    region: Optional[str] = Query(None, description="Geographic region filter"),
) -> Dict[str, Any]:
    """REST exposure of the real business_metrics KPI rollup.

    Thin wrapper over the existing :func:`get_kpi_summary` (also registered as a
    CopilotAction) so the Home QUICK_STATS bar can read Total TRx (MTD) and
    HCPs Reached directly. Returns ``{brand, period, metrics, data_source}``;
    ``data_source`` is ``"database"`` for real values, ``"fallback"`` otherwise.
    When ``region`` is supplied the metrics re-scope to that region (migration
    077 variants).

    Open (no auth) to match the sibling ``GET /copilotkit/status`` so the
    dashboard read works without the auth envelope.
    """
    return await get_kpi_summary(brand, region=region)


# =============================================================================
# E2I CHATBOT STREAMING ENDPOINTS
# =============================================================================


class ChatRequest(BaseModel):
    """Request schema for chatbot endpoints.

    Note: request_id is optional - if not provided, it will be extracted from
    the X-Request-ID header via TracingMiddleware (Phase 1 G08).
    """

    query: str = Field(..., description="User's query text")
    user_id: str = Field(
        ...,
        description=(
            "NON-AUTHORITATIVE. Retained for backward compatibility only. The "
            "server derives the caller's identity from the authenticated token; "
            "this value is ignored for identity and, if it does not match the "
            "token identity, the request is rejected (403)."
        ),
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request identifier (auto-extracted from X-Request-ID header if not provided)",
    )
    session_id: Optional[str] = Field(
        default=None, description="Session ID (generated if not provided)"
    )
    brand_context: Optional[str] = Field(
        default=None, description="Brand filter (Kisqali, Fabhalta, Remibrutinib)"
    )
    region_context: Optional[str] = Field(default=None, description="Region filter (US, EU, APAC)")


class ChatResponse(BaseModel):
    """Response schema for non-streaming chatbot endpoint."""

    success: bool
    session_id: str
    response: str
    conversation_title: Optional[str] = None
    agent_name: Optional[str] = None
    error: Optional[str] = None

    # Dispatch observability (Phase 1 System Evaluation)
    orchestrator_used: bool = False
    agents_dispatched: List[str] = []
    routed_agent: Optional[str] = None
    response_confidence: Optional[float] = None
    execution_time_ms: Optional[float] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    # Phase 4: Decision rationale for agent routing transparency
    routing_rationale: Optional[str] = None
    # 4-stage ClassificationPipeline observability (ORCHESTRATOR_CLASSIFIER_MODE
    # shadow/active; None when off or when the orchestrator was not consulted).
    routing_pattern: Optional[str] = None
    # #1582: "pipeline" | "legacy" — which subsystem produced this turn's
    # dispatch plan. routing_pattern above is the PIPELINE's decision and is
    # emitted in shadow mode too, where legacy routing is what answered; this
    # names the difference so an abstaining pattern beside a real
    # agents_dispatched no longer reads as a routing regression.
    routing_authority: Optional[str] = None
    classification_latency_ms: Optional[float] = None
    used_llm_layer: Optional[bool] = None


# Generic, client-safe error message for the chat endpoints. Internal
# exception detail is logged server-side, never returned to the caller
# (Finding 3 — info disclosure).
_GENERIC_CHAT_ERROR = "An internal error occurred while processing your request. Please try again."

# #1561: honest envelope for a stream that COMPLETED without producing any
# text. Distinct from _GENERIC_CHAT_ERROR (the exception path): here nothing
# raised — the graph ran and yielded zero characters (measured 2026-08-12
# turn 5.1: HTTP 200, 27 s, generate node ran ~19 s, zero chars, clean
# retry). The message explains the fault and invites retry; it must never
# read as an analytical answer.
_EMPTY_STREAM_FALLBACK = (
    "I wasn't able to produce a response for this question — the analysis ran "
    "but returned no text. This is a transient platform issue, not an answer; "
    "please try asking again."
)


def _resolve_chat_identity(authenticated_user: Dict[str, Any], body_user_id: Optional[str]) -> str:
    """Resolve the authoritative chat identity from the authenticated token.

    Finding 1 [HIGH IDOR]: ``ChatRequest.user_id`` was a required request-body
    field used as the caller's identity (session ownership, message persistence,
    cross-user memory). A caller could pass any ``user_id`` and impersonate
    another user. The body value is therefore NON-AUTHORITATIVE — the identity
    is always taken from the authenticated token (``require_viewer`` →
    ``user["id"]``).

    For backward compatibility the body may still carry ``user_id``; if it is
    present and disagrees with the token identity it is treated as an
    impersonation attempt and rejected with 403 (skipped in testing mode, which
    deliberately bypasses real auth).

    Args:
        authenticated_user: The user dict from ``require_viewer``.
        body_user_id: The (optional, non-authoritative) ``user_id`` from the body.

    Returns:
        The authoritative user id to use for all downstream calls.

    Raises:
        HTTPException: 403 if a mismatching body ``user_id`` is supplied
            (production only).
    """
    token_user_id = (authenticated_user or {}).get("id")
    if not token_user_id:
        # Should not happen behind require_viewer, but fail closed if it does.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user identity is missing.",
        )

    if body_user_id and body_user_id != token_user_id and not TESTING_MODE:
        logger.warning(
            "[Chatbot] Rejected user_id mismatch (possible impersonation): "
            "body user_id does not match authenticated identity"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Request user_id does not match the authenticated user.",
        )

    return str(token_user_id)


def _resolve_chat_brand(authenticated_user: Dict[str, Any], requested_brand: Optional[str]) -> str:
    """Validate a chat ``brand_context`` against the caller's brand grants (H1 / #694).

    A causal analysis stamps its ``brand_context`` onto any causal finding it
    writes to the FalkorDB graph; ``GET /memory/semantic/paths`` then surfaces
    that finding to the brand's scoped viewers. So an un-grant-checked
    ``brand_context`` is a cross-tenant WRITE-poisoning vector: a viewer granted
    only Brand-A could stamp a finding ``brand=Brand-B`` and have it appear in
    Brand-B's scoped reads. We therefore require the requested brand to be within
    the caller's grants.

    Returns the validated brand (or ``""`` when none was requested — an unbranded
    finding stays admin-only). A cross-brand admin (or ``'all'`` grant) may use
    any brand. Skipped under ``TESTING_MODE`` (which bypasses real auth).

    Raises:
        HTTPException: 403 if a scoped caller requests a brand outside their grants.
    """
    if not requested_brand or TESTING_MODE:
        return requested_brand or ""
    if is_cross_brand_admin(authenticated_user):
        return requested_brand
    if requested_brand in set(get_user_brands(authenticated_user)):
        return requested_brand
    logger.warning(
        "[Chatbot] Rejected brand_context outside caller's grants "
        "(possible cross-tenant write-poisoning attempt)."
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="brand_context is outside your brand grants.",
    )


async def _stream_chat_response(
    request: ChatRequest, authenticated_user_id: str
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream for chatbot response.

    Yields JSON-formatted SSE events:
    - {"type": "session_id", "data": "..."}
    - {"type": "text", "data": "..."}
    - {"type": "conversation_title", "data": "..."}
    - {"type": "tool_call", "data": "..."}
    - {"type": "dispatch_info", "data": {...}} - Dispatch observability
    - {"type": "done", "data": ""}
    - {"type": "error", "data": "..."}
    """

    start_time = time.time()

    try:
        from src.api.routes.chatbot_graph import LATENCY_SPAN_KEY, stream_chatbot

        # Yield session_id first. Identity is the AUTHENTICATED user id
        # (Finding 1 — never trust request.user_id for identity).
        session_id = request.session_id
        if not session_id:
            import uuid

            session_id = f"{authenticated_user_id}~{uuid.uuid4()}"

        yield f"data: {json.dumps({'type': 'session_id', 'data': session_id})}\n\n"

        response_text = ""
        conversation_title = None

        # Track dispatch observability (Phase 1 System Evaluation + Phase 4 rationale)
        dispatch_info = {
            "orchestrator_used": False,
            "agents_dispatched": [],
            "routed_agent": None,
            "response_confidence": None,
            "intent": None,
            "intent_confidence": None,
            "routing_rationale": None,
            # 4-stage classifier observability
            "routing_pattern": None,
            "routing_authority": None,
            "classification_latency_ms": None,
            "used_llm_layer": None,
            # #1454: per-request latency span (from the __latency_span__ item)
            "node_wall_ms": None,
            "graph_total_ms": None,
            "untimed_overhead_ms": None,
            "first_request_in_worker": None,
            # #1454: which worker served this request — pairs with the pid on
            # the startup-warm completion log line.
            "worker_pid": None,
            # #1475: orchestrator-internal attribution (same span item)
            "orchestrator_stage_ms": None,
            "orchestrator_run_ms": None,
            "orchestrator_untimed_ms": None,
            # #1484: retrieve_rag chain-internal attribution (same span item)
            "rag_stage_ms": None,
            "rag_meta": None,
            # #1561: True when the zero-char guard emitted the fallback
            # envelope — makes recurrence countable without scraping logs.
            "empty_response_fallback": False,
        }

        # Stream through chatbot workflow
        async for state_update in stream_chatbot(
            query=request.query,
            user_id=authenticated_user_id,
            request_id=request.request_id or "unknown",
            session_id=session_id,
            brand_context=request.brand_context or "",
            region_context=request.region_context or "",
        ):
            # Extract response from state updates
            if isinstance(state_update, dict):
                # #1454: synthetic latency-span item — observability only,
                # surfaced via dispatch_info, never rendered as answer text.
                span_payload = state_update.get(LATENCY_SPAN_KEY)
                if span_payload is not None:
                    dispatch_info["node_wall_ms"] = span_payload.get("node_wall_ms")
                    dispatch_info["graph_total_ms"] = span_payload.get("graph_total_ms")
                    dispatch_info["untimed_overhead_ms"] = span_payload.get("untimed_overhead_ms")
                    dispatch_info["first_request_in_worker"] = span_payload.get(
                        "first_request_in_worker"
                    )
                    dispatch_info["worker_pid"] = span_payload.get("worker_pid")
                    # #1475: orchestrator-internal attribution
                    dispatch_info["orchestrator_stage_ms"] = span_payload.get(
                        "orchestrator_stage_ms"
                    )
                    dispatch_info["orchestrator_run_ms"] = span_payload.get("orchestrator_run_ms")
                    dispatch_info["orchestrator_untimed_ms"] = span_payload.get(
                        "orchestrator_untimed_ms"
                    )
                    # #1484: retrieve_rag chain-internal attribution
                    dispatch_info["rag_stage_ms"] = span_payload.get("rag_stage_ms")
                    dispatch_info["rag_meta"] = span_payload.get("rag_meta")
                    continue

                # Check for node outputs
                for _node_name, node_output in state_update.items():
                    if isinstance(node_output, dict):
                        # Get response text from finalize node
                        if "response_text" in node_output and node_output["response_text"]:
                            text_chunk = node_output["response_text"]
                            if text_chunk and text_chunk != response_text:
                                # Yield new text
                                new_text = (
                                    text_chunk[len(response_text) :]
                                    if response_text
                                    else text_chunk
                                )
                                if new_text:
                                    yield f"data: {json.dumps({'type': 'text', 'data': new_text})}\n\n"
                                    response_text = text_chunk

                        # Get conversation title
                        if (
                            "conversation_title" in node_output
                            and node_output["conversation_title"]
                        ):
                            title = node_output["conversation_title"]
                            if title != conversation_title:
                                conversation_title = title
                                yield f"data: {json.dumps({'type': 'conversation_title', 'data': title})}\n\n"

                        # Handle messages (for AIMessage content).
                        # NEVER stream load_context's messages (#1442): that node
                        # returns conversation HISTORY restored for downstream
                        # context, not new output. Emitting its AIMessages here
                        # re-streams the prior assistant turn (and mis-slices the
                        # real answer). This guard closes the replay on every path
                        # — including the Redis-down/cross-worker fallback where
                        # load_context still re-loads DB history into `messages`.
                        if "messages" in node_output and _node_name != "load_context":
                            for msg in node_output["messages"]:
                                if isinstance(msg, AIMessage) and msg.content:
                                    # AIMessage.content is str | list of content
                                    # blocks (#1350); the diff below needs str.
                                    content_text = normalize_llm_content(msg.content)
                                    if content_text and content_text != response_text:
                                        new_text = (
                                            content_text[len(response_text) :]
                                            if response_text
                                            else content_text
                                        )
                                        if new_text:
                                            yield f"data: {json.dumps({'type': 'text', 'data': new_text})}\n\n"
                                            response_text = content_text

                        # Track dispatch observability fields
                        if "orchestrator_used" in node_output:
                            dispatch_info["orchestrator_used"] = node_output["orchestrator_used"]
                        if "agents_dispatched" in node_output:
                            dispatch_info["agents_dispatched"] = node_output["agents_dispatched"]
                        if "routed_agent" in node_output:
                            dispatch_info["routed_agent"] = node_output["routed_agent"]
                        if "response_confidence" in node_output:
                            dispatch_info["response_confidence"] = node_output[
                                "response_confidence"
                            ]
                        if "intent" in node_output:
                            dispatch_info["intent"] = node_output["intent"]
                        if "intent_confidence" in node_output:
                            dispatch_info["intent_confidence"] = node_output["intent_confidence"]
                        # Phase 4: Decision rationale for transparency
                        if "routing_rationale" in node_output:
                            dispatch_info["routing_rationale"] = node_output["routing_rationale"]
                        # 4-stage classifier observability
                        if "routing_pattern" in node_output:
                            dispatch_info["routing_pattern"] = node_output["routing_pattern"]
                        if "routing_authority" in node_output:
                            dispatch_info["routing_authority"] = node_output["routing_authority"]
                        if "classification_latency_ms" in node_output:
                            dispatch_info["classification_latency_ms"] = node_output[
                                "classification_latency_ms"
                            ]
                        if "used_llm_layer" in node_output:
                            dispatch_info["used_llm_layer"] = node_output["used_llm_layer"]

        # #1561: a zero-char completion must never close as a silent empty
        # 200. Guard the TOTAL response at stream end — individual empty
        # chunks are benign (wave-1 finding on the AG-UI surface); the defect
        # is the whole turn producing nothing while returning HTTP 200.
        #
        # #1336 bridge interaction: the conversational bridge fires INSIDE
        # orchestrator_node on complete orchestrator failure, and its answer
        # (or the #883 fail-closed summary) reaches this writer as ordinary
        # response_text. Guard and bridge are therefore mutually exclusive by
        # construction: bridge-authored text makes the guard inert, and an
        # empty total means the bridge either never triggered (the
        # successful-agents-but-empty-synthesis gap falls between the two
        # mechanisms) or itself returned None. No masking, no double-wrap.
        if not response_text.strip():
            dispatch_info["empty_response_fallback"] = True
            logger.error(
                "[#1561] zero-char /chat/stream completion (request_id=%s, "
                "session_id=%s, %.0f ms) — emitting fallback envelope",
                request.request_id,
                session_id,
                (time.time() - start_time) * 1000,
            )
            yield f"data: {json.dumps({'type': 'text', 'data': _EMPTY_STREAM_FALLBACK})}\n\n"

        # Generate title if not set
        if not conversation_title and response_text:
            # Simple title generation from query
            title = request.query[:50] + "..." if len(request.query) > 50 else request.query
            yield f"data: {json.dumps({'type': 'conversation_title', 'data': title})}\n\n"

        # Add execution time to dispatch info
        dispatch_info["execution_time_ms"] = round((time.time() - start_time) * 1000, 2)

        # Yield dispatch observability before done (Phase 1 System Evaluation)
        yield f"data: {json.dumps({'type': 'dispatch_info', 'data': dispatch_info})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'data': ''})}\n\n"

    except Exception as e:
        execution_time_ms = round((time.time() - start_time) * 1000, 2)
        # Finding 3: log internal detail server-side, return a generic message.
        logger.error(f"Streaming chat error after {execution_time_ms}ms: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'data': _GENERIC_CHAT_ERROR})}\n\n"


@router.post("/chat/stream", summary="Stream chatbot response", operation_id="stream_chat")
async def stream_chat(
    chat_request: ChatRequest,
    request: Request,
    _user: dict = Depends(require_viewer),
) -> StreamingResponse:
    """
    Stream chatbot response as Server-Sent Events (SSE).

    Returns an SSE stream with events:
    - session_id: The conversation session ID
    - text: Response text chunks
    - conversation_title: Auto-generated conversation title
    - tool_call: Tool invocation notifications
    - done: Stream completion signal
    - error: Error messages

    Note: request_id is optional. If not provided, it's extracted from the
    X-Request-ID header via TracingMiddleware (Phase 1 G08).

    Usage:
        POST /api/copilotkit/chat/stream
        Content-Type: application/json
        X-Request-ID: optional-tracking-id  // Optional, auto-generated if not set

        {
            "query": "What is the TRx for Kisqali?",
            "user_id": "user-uuid",
            "request_id": "req-123",  // Optional, falls back to X-Request-ID header
            "session_id": "",  // Optional, generated if empty
            "brand_context": "Kisqali"  // Optional
        }
    """
    # Finding 1: derive identity from the authenticated token, never the body.
    authenticated_user_id = _resolve_chat_identity(_user, chat_request.user_id)
    # H1 (#694): a brand_context outside the caller's grants would let them poison
    # another tenant's scoped causal-graph view via store_causal_path -> reject.
    chat_request.brand_context = _resolve_chat_brand(_user, chat_request.brand_context)

    # Phase 1 G08: Use middleware request_id if not provided in body
    effective_request_id = chat_request.request_id or get_request_id() or "unknown"

    logger.info(
        f"[Chatbot] Streaming request: query={redact_query(chat_request.query)}, "
        f"user={authenticated_user_id}, request_id={effective_request_id}"
    )

    # Update the request with the effective request_id
    chat_request.request_id = effective_request_id

    # #1659: every frame below originates from a LangGraph node-completion
    # update, and the orchestrator is ONE node that ainvokes a nested graph — so
    # without a keepalive this body is silent for the whole turn. Measured on
    # prod 2026-08-16: one frame at 860.9 ms, then 34 395.7 ms of nothing (the
    # sum of every node's wall time, 6 ms apart), on a turn that dispatched
    # heterogeneous_optimizer. nginx's proxy_read_timeout bounds exactly that
    # gap, so the pre-fix constraint was "total turn wall time < 300 s" — which
    # the 420 s heterogeneous_optimizer budget in router.py blows on its own.
    # The wrapper makes the bounded quantity a constant instead.
    return StreamingResponse(
        with_sse_keepalive(_stream_chat_response(chat_request, authenticated_user_id)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": effective_request_id,  # Include in response for correlation
        },
    )


@router.post("/chat", summary="Non-streaming chatbot", operation_id="chat")
async def chat(
    chat_request: ChatRequest,
    request: Request,
    _user: dict = Depends(require_viewer),
) -> ChatResponse:
    """
    Non-streaming chatbot endpoint.

    Returns the complete response in a single JSON object with dispatch observability.

    Usage:
        POST /api/copilotkit/chat
        Content-Type: application/json

        {
            "query": "Show agent status",
            "user_id": "user-uuid",
            "request_id": "req-456",
            "session_id": ""
        }

    Response includes dispatch observability fields:
        - orchestrator_used: Whether the orchestrator processed this query
        - agents_dispatched: List of agents that were dispatched
        - routed_agent: Primary agent routed for this query
        - response_confidence: Orchestrator response confidence (0.0-1.0)
        - execution_time_ms: Total execution time in milliseconds
        - intent: Classified intent type
        - intent_confidence: Intent classification confidence (0.0-1.0)
        - routing_rationale: Explanation for why this agent was selected (Phase 4)

    Note: request_id is optional. If not provided, it's extracted from the
    X-Request-ID header via TracingMiddleware (Phase 1 G08).
    """

    # Finding 1: derive identity from the authenticated token, never the body.
    # (Outside the try/except so a 403 propagates instead of being swallowed
    # into a 200 error body.)
    authenticated_user_id = _resolve_chat_identity(_user, chat_request.user_id)
    # H1 (#694): a brand_context outside the caller's grants would let them poison
    # another tenant's scoped causal-graph view via store_causal_path -> reject.
    chat_request.brand_context = _resolve_chat_brand(_user, chat_request.brand_context)

    # Phase 1 G08: Use middleware request_id if not provided in body
    effective_request_id = chat_request.request_id or get_request_id() or "unknown"
    chat_request.request_id = effective_request_id

    logger.info(
        f"[Chatbot] Chat request: query={redact_query(chat_request.query)}, "
        f"user={authenticated_user_id}, request_id={effective_request_id}"
    )

    # Start timing for execution_time_ms
    start_time = time.time()

    try:
        from src.api.routes.chatbot_graph import run_chatbot

        result = await run_chatbot(
            query=chat_request.query,
            user_id=authenticated_user_id,
            request_id=chat_request.request_id or "unknown",
            session_id=chat_request.session_id or "",
            brand_context=chat_request.brand_context or "",
            region_context=chat_request.region_context or "",
        )

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Extract basic response fields
        response_text = result.get("response_text", "")
        session_id = result.get("session_id", "")
        agent_name = result.get("agent_name")

        # Extract dispatch observability fields (Phase 1 System Evaluation + Phase 4)
        orchestrator_used = result.get("orchestrator_used", False)
        agents_dispatched = result.get("agents_dispatched", [])
        routed_agent = result.get("routed_agent")
        response_confidence = result.get("response_confidence")
        intent = result.get("intent")
        intent_confidence = result.get("intent_confidence")
        # Phase 4: Decision rationale for transparency
        routing_rationale = result.get("routing_rationale")
        # 4-stage classifier observability
        routing_pattern = result.get("routing_pattern")
        routing_authority = result.get("routing_authority")
        classification_latency_ms = result.get("classification_latency_ms")
        used_llm_layer = result.get("used_llm_layer")

        # Generate title from query
        title = (
            chat_request.query[:50] + "..." if len(chat_request.query) > 50 else chat_request.query
        )

        logger.info(
            f"[Chatbot] Response: orchestrator={orchestrator_used}, "
            f"agents={agents_dispatched}, intent={intent}, "
            f"rationale={routing_rationale[:50] if routing_rationale else None}..., "
            f"time_ms={execution_time_ms:.1f}"
        )

        return ChatResponse(
            success=True,
            session_id=session_id,
            response=response_text,
            conversation_title=title,
            agent_name=agent_name,
            # Dispatch observability
            orchestrator_used=orchestrator_used,
            agents_dispatched=agents_dispatched,
            routed_agent=routed_agent,
            response_confidence=response_confidence,
            execution_time_ms=round(execution_time_ms, 2),
            intent=intent,
            intent_confidence=intent_confidence,
            # Phase 4: Decision rationale
            routing_rationale=routing_rationale,
            # 4-stage classifier observability
            routing_pattern=routing_pattern,
            routing_authority=routing_authority,
            classification_latency_ms=classification_latency_ms,
            used_llm_layer=used_llm_layer,
        )

    except Exception as e:
        # Calculate execution time even on error
        execution_time_ms = (time.time() - start_time) * 1000
        # Finding 3: log internal detail server-side, return a generic message
        # to the client (no exception text / connection strings leaked).
        logger.error(f"Chat error after {execution_time_ms:.1f}ms: {e}", exc_info=True)
        return ChatResponse(
            success=False,
            session_id=chat_request.session_id or "",
            response="",
            error=_GENERIC_CHAT_ERROR,
            execution_time_ms=round(execution_time_ms, 2),
        )


# =============================================================================
# FEEDBACK ENDPOINTS
# =============================================================================


class FeedbackRequest(BaseModel):
    """Request schema for submitting message feedback.

    Two ways to identify the rated message:

    - ``message_id`` — the ``chatbot_messages.id`` DB key, for surfaces that
      render persisted history and genuinely know it.
    - ``session_id`` + ``message_uuid``/``response_preview`` — for the live
      CopilotKit stream, which only knows its own AG-UI message uuid (never
      the DB id). The server resolves the DB row by the stamped
      ``metadata.frontend_message_id`` key first, then by content matching
      (``session_id`` IS the CopilotKit threadId — that is what message
      persistence stores as the session key).
    """

    message_id: Optional[int] = Field(
        default=None,
        description="DB id of the message being rated (omit to resolve by "
        "session_id + message_uuid/response_preview)",
    )
    message_uuid: Optional[str] = Field(
        default=None,
        description="Client-side AG-UI message uuid; primary resolution key "
        "(the SSE layer stamps it onto the persisted row as "
        "metadata.frontend_message_id), also stored in feedback metadata",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session ID (CopilotKit threadId); required when message_id is omitted",
    )
    rating: str = Field(..., description="Rating: 'thumbs_up' or 'thumbs_down'")
    comment: Optional[str] = Field(default=None, description="Optional user comment")
    query_text: Optional[str] = Field(
        default=None, description="The user query that led to this response"
    )
    response_preview: Optional[str] = Field(
        default=None, description="First 500 chars of the response"
    )
    response_text: Optional[str] = Field(
        default=None,
        max_length=20000,
        description="Full response text, used for exact-match message resolution "
        "(response_preview stays the stored 500-char excerpt)",
    )
    agent_name: Optional[str] = Field(
        default=None,
        description="Fallback-only hint: the persisted message row is the "
        "authority on which agent responded",
    )
    tools_used: Optional[List[str]] = Field(
        default=None,
        description="Fallback-only hint: the persisted row's tool_results/"
        "tool_calls (top-level columns or metadata) win whenever present",
    )


class FeedbackResponse(BaseModel):
    """Response schema for feedback endpoints."""

    success: bool
    feedback_id: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None


@router.post("/feedback", summary="Submit message feedback", operation_id="submit_feedback")
async def submit_feedback(
    request: FeedbackRequest,
    _user: dict = Depends(require_viewer),
) -> FeedbackResponse:
    """
    Submit feedback (thumbs up/down) for a chatbot message.

    This endpoint allows users to rate assistant responses for quality
    improvement and prompt optimization.

    Usage (explicit DB id — persisted-history surfaces):
        POST /api/copilotkit/feedback
        {"message_id": 123, "rating": "thumbs_up", ...}

    Usage (live CopilotKit stream — no DB id available client-side):
        POST /api/copilotkit/feedback
        {
            "session_id": "<copilotkit threadId>",
            "response_text": "<full response text>",
            "response_preview": "The TRx performance...",
            "message_uuid": "<ag-ui message uuid>",
            "rating": "thumbs_up"
        }
        The server resolves the DB message by the stamped frontend_message_id
        key (message_uuid) first, then by exact full-content match
        (response_text) against recent assistant messages in the session,
        finally by response-prefix matching (response_preview).
        agent_name/tools_used are derived server-side from the matched row.
    """
    logger.info(
        f"[Feedback] Received feedback: message_id={request.message_id}, "
        f"message_uuid={request.message_uuid}, rating={request.rating}, "
        f"session={request.session_id[:20] if request.session_id else 'to-be-looked-up'}..."
    )

    # Validate rating
    if request.rating not in ("thumbs_up", "thumbs_down"):
        return FeedbackResponse(
            success=False,
            error=f"Invalid rating: {request.rating}. Must be 'thumbs_up' or 'thumbs_down'",
        )

    try:
        import os

        from supabase import create_client

        from src.memory.services.factories import get_async_supabase_client
        from src.repositories import get_chatbot_feedback_repository

        # Use service key client to bypass RLS for message lookup
        service_url = os.environ.get("SUPABASE_URL")
        service_key = os.environ.get("SUPABASE_SERVICE_KEY")

        if not service_url or not service_key:
            return FeedbackResponse(
                success=False,
                error="Server configuration error: missing Supabase credentials",
            )

        # Resolve the rated message row. Two paths:
        #  (a) explicit DB message_id — surfaces that render persisted history;
        #  (b) session_id + response_preview — the live CopilotKit stream only
        #      knows its AG-UI uuid, which is unrelated to the DB id. Match the
        #      response prefix against recent assistant rows in the session.
        #      (The old client fabricated an id via parseInt(uuid)||Date.now(),
        #      which either failed this lookup or collided with a real row from
        #      a DIFFERENT session — silently mis-attributed feedback.)
        # Using service key client to bypass RLS policies.
        session_id = None
        resolved_message_id = request.message_id
        matched_row: Optional[dict] = None
        lookup_error = None
        try:
            service_client = create_client(service_url, service_key)
            if resolved_message_id is not None:
                message_result = (
                    service_client.table("chatbot_messages")
                    .select("id, session_id, agent_name, metadata, tool_calls, tool_results")
                    .eq("id", resolved_message_id)
                    .limit(1)
                    .execute()
                )

                if message_result.data and len(message_result.data) > 0:
                    matched_row = cast(Dict[str, Any], message_result.data[0])
                    session_id = matched_row.get("session_id")
                    logger.info(
                        f"[Feedback] Found message {resolved_message_id} with session_id={session_id}"
                    )
                else:
                    lookup_error = f"Message {resolved_message_id} not found in database"
                    logger.warning(f"[Feedback] {lookup_error}")
            elif request.session_id and (
                (request.message_uuid or "").strip()
                or (request.response_preview or "").strip()
                or (request.response_text or "").strip()
            ):
                preview = (request.response_preview or request.response_text or "")[:500]
                full_text = request.response_text or None
                # Pass 0 — stable key. The SSE translation layer stamps each
                # assistant row with the frontend-visible messageId after the
                # stream (metadata.frontend_message_id, see
                # _stamp_frontend_message_ids), so the client's message_uuid
                # resolves the row directly, no content heuristics.
                if request.message_uuid:
                    try:
                        uuid_result = (
                            service_client.table("chatbot_messages")
                            .select(
                                "id, session_id, content, agent_name, metadata, "
                                "tool_calls, tool_results"
                            )
                            .eq("session_id", request.session_id)
                            .eq("role", "assistant")
                            .eq("metadata->>frontend_message_id", request.message_uuid)
                            .order("created_at", desc=True)
                            .limit(1)
                            .execute()
                        )
                        if uuid_result.data:
                            matched_row = cast(Dict[str, Any], uuid_result.data[0])
                    except Exception as pass0_error:  # noqa: BLE001
                        # A pass-0 failure must degrade to content matching,
                        # not abort resolution (the enclosing try also covers
                        # passes 1-2).
                        logger.debug(
                            f"[Feedback] stamped-id lookup failed, "
                            f"falling back to content: {pass0_error}"
                        )
                if matched_row is None and (full_text or preview):
                    candidates = (
                        service_client.table("chatbot_messages")
                        .select(
                            "id, session_id, content, agent_name, metadata, "
                            "tool_calls, tool_results"
                        )
                        .eq("session_id", request.session_id)
                        .eq("role", "assistant")
                        .order("created_at", desc=True)
                        .limit(20)
                        .execute()
                    )
                    rows = cast(List[Dict[str, Any]], candidates.data or [])
                    # Content fallback for rows the stamp didn't reach (e.g.
                    # persistence raced the stamping sweep, or pre-deploy rows).
                    # Pass 1 — exact full-content equality (identical duplicate
                    # responses are interchangeable for feedback, newest wins).
                    if full_text:
                        for row in rows:
                            if (row.get("content") or "") == full_text:
                                matched_row = row
                                break
                    # Pass 2 — exact-prefix match only: a fuzzy/newest-row
                    # fallback could attach the rating to the wrong message.
                    if matched_row is None:
                        for row in rows:
                            content = row.get("content") or ""
                            if content[: len(preview)] == preview:
                                matched_row = row
                                break
                if matched_row is not None:
                    resolved_message_id = matched_row.get("id")
                    session_id = matched_row.get("session_id")
                    logger.info(
                        f"[Feedback] Resolved message {resolved_message_id} "
                        f"in session {str(request.session_id)[:20]}..."
                    )
                else:
                    lookup_error = (
                        "No persisted assistant message in this session matches the "
                        "rated response (it may not have been persisted yet)"
                    )
                    logger.warning(f"[Feedback] {lookup_error}")
            else:
                lookup_error = (
                    "Either message_id or (session_id + response_preview) is required "
                    "to identify the rated message"
                )
        except Exception as lookup_err:
            lookup_error = f"Error looking up message: {lookup_err}"
            logger.warning(f"[Feedback] {lookup_error}")

        if not session_id or resolved_message_id is None:
            return FeedbackResponse(
                success=False,
                error=lookup_error or f"Could not find session for message_id {request.message_id}",
            )

        # The persisted message row is the authority on attribution (trust
        # boundary — the old client hardcoded agent_name='copilotkit' on every
        # live thumb). Sidebar-graph rows genuinely say 'copilotkit' (that
        # pipeline IS the responder — its graph has no routed agent), while
        # orchestrator-flow rows carry the real routed agent, so deriving from
        # the row keeps both honest. Client-supplied values are fallback-only
        # for rows persisted without attribution.
        row_meta = (matched_row or {}).get("metadata") or {}
        resolved_agent_name = (matched_row or {}).get("agent_name") or request.agent_name
        # Row data beats the client hint whenever it exists: orchestrator-flow
        # rows store tools in top-level tool_results/tool_calls columns
        # (chatbot_graph persistence), sidebar synthesis rows in
        # metadata.tool_results. request.tools_used is fallback-only, for rows
        # persisted without tool data.
        resolved_tools = None
        for source in (
            (matched_row or {}).get("tool_results"),
            (matched_row or {}).get("tool_calls"),
            row_meta.get("tool_results"),
            row_meta.get("tool_calls"),
        ):
            tool_names = _tool_names_from(source)
            if tool_names:
                resolved_tools = tool_names
                break
        if not resolved_tools:
            resolved_tools = request.tools_used or None

        client = await get_async_supabase_client()
        repo = get_chatbot_feedback_repository(supabase_client=client)

        result = await repo.add_feedback(
            message_id=resolved_message_id,
            session_id=session_id,  # type: ignore[arg-type]
            rating=request.rating,  # type: ignore[arg-type]
            comment=request.comment,
            query_text=request.query_text,
            response_preview=request.response_preview,
            agent_name=resolved_agent_name,
            tools_used=resolved_tools,
            metadata=({"message_uuid": request.message_uuid} if request.message_uuid else None),
        )

        if result:
            logger.info(f"[Feedback] Saved feedback ID={result.get('id')}")
            return FeedbackResponse(
                success=True,
                feedback_id=result.get("id"),
                message="Feedback submitted successfully",
            )
        else:
            return FeedbackResponse(
                success=False,
                error="Failed to save feedback - no result returned",
            )

    except Exception as e:
        logger.error(f"[Feedback] Error submitting feedback: {e}")
        return FeedbackResponse(
            success=False,
            error=str(e),
        )


@router.get("/feedback/stats", summary="Get feedback statistics", operation_id="get_feedback_stats")
async def get_feedback_stats(
    agent_name: Optional[str] = None,
    days: int = Query(default=30, ge=1, le=90, description="Days of history"),
    _user: dict = Depends(require_viewer),
) -> Dict[str, Any]:
    """
    Get feedback statistics for analytics.

    Usage:
        GET /api/copilotkit/feedback/stats?agent_name=tool_composer&days=30
    """
    try:
        from src.memory.services.factories import get_async_supabase_client
        from src.repositories import get_chatbot_feedback_repository

        client = await get_async_supabase_client()
        repo = get_chatbot_feedback_repository(supabase_client=client)

        # Get agent-level stats
        agent_stats = await repo.get_agent_stats(agent_name=agent_name, days=days)

        # Get summary
        summary = await repo.get_feedback_summary(days=days)

        return {
            "success": True,
            "summary": summary,
            "agent_stats": agent_stats,
        }

    except Exception as e:
        logger.error(f"[Feedback] Error getting stats: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# ============================================================================
# Analytics Endpoints (P7.1)
# ============================================================================


@router.get("/analytics/usage", summary="Get usage analytics", operation_id="get_usage_analytics")
async def get_usage_analytics(
    days: int = Query(default=7, ge=1, le=90, description="Days of history"),
    _user: dict = Depends(require_viewer),
) -> Dict[str, Any]:
    """
    Get chatbot usage analytics summary.

    Usage:
        GET /api/copilotkit/analytics/usage?days=7

    Returns usage statistics including:
    - Total queries
    - Average response time
    - Query type distribution
    - Tool usage patterns
    """
    try:
        from datetime import datetime, timedelta, timezone

        from src.repositories import get_chatbot_analytics_repository

        repo = get_chatbot_analytics_repository()

        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        end_date = datetime.now(timezone.utc)

        # Get usage summary
        summary = await repo.get_usage_summary(start_date=start_date, end_date=end_date)

        # Get query type distribution
        query_types = await repo.get_query_type_distribution(days=days)

        # Get tool usage stats
        tool_usage = await repo.get_tool_usage_stats(days=days)

        return {
            "success": True,
            "period_days": days,
            "summary": summary,
            "query_types": query_types,
            "tool_usage": tool_usage,
        }

    except Exception as e:
        logger.error(f"[Analytics] Error getting usage stats: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@router.get("/analytics/agents", summary="Get agent analytics", operation_id="get_agent_analytics")
async def get_agent_analytics(
    agent_name: Optional[str] = None,
    days: int = Query(default=30, ge=1, le=90, description="Days of history"),
    _user: dict = Depends(require_viewer),
) -> Dict[str, Any]:
    """
    Get agent performance analytics.

    Usage:
        GET /api/copilotkit/analytics/agents?agent_name=tool_composer&days=30

    Returns performance metrics by agent:
    - Query count
    - Average response time
    - Error rate
    - Tool invocation patterns
    """
    try:
        from src.repositories import get_chatbot_analytics_repository

        repo = get_chatbot_analytics_repository()

        # Get agent performance metrics
        agent_stats = await repo.get_agent_performance(agent_name=agent_name, days=days)

        return {
            "success": True,
            "agent_name": agent_name,
            "period_days": days,
            "agent_stats": agent_stats,
        }

    except Exception as e:
        logger.error(f"[Analytics] Error getting agent stats: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@router.get("/analytics/errors", summary="Get error analytics", operation_id="get_error_analytics")
async def get_error_analytics(
    limit: int = Query(default=20, ge=1, le=200, description="Max errors to return"),
    _user: dict = Depends(require_viewer),
) -> Dict[str, Any]:
    """
    Get recent error analytics for debugging.

    Usage:
        GET /api/copilotkit/analytics/errors?limit=20

    Returns recent errors with:
    - Error type
    - Error message
    - Session context
    - Timestamp
    """
    try:
        from src.repositories import get_chatbot_analytics_repository

        repo = get_chatbot_analytics_repository()

        errors = await repo.get_recent_errors(limit=limit)

        return {
            "success": True,
            "count": len(errors),
            "errors": errors,
        }

    except Exception as e:
        logger.error(f"[Analytics] Error getting error stats: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@router.get(
    "/analytics/hourly", summary="Get hourly usage pattern", operation_id="get_hourly_pattern"
)
async def get_hourly_pattern(
    days: int = Query(default=7, ge=1, le=90, description="Days of history"),
    _user: dict = Depends(require_viewer),
) -> Dict[str, Any]:
    """
    Get hourly usage pattern for capacity planning.

    Usage:
        GET /api/copilotkit/analytics/hourly?days=7

    Returns usage distribution by hour of day.
    """
    try:
        from src.repositories import get_chatbot_analytics_repository

        repo = get_chatbot_analytics_repository()

        hourly_pattern = await repo.get_hourly_pattern(days=days)

        return {
            "success": True,
            "period_days": days,
            "hourly_pattern": hourly_pattern,
        }

    except Exception as e:
        logger.error(f"[Analytics] Error getting hourly pattern: {e}")
        return {
            "success": False,
            "error": str(e),
        }
