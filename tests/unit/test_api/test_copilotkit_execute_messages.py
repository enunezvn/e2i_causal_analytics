"""Unit coverage for the execute() bridge message conversion — 2026-07-07.

Why this exists: the frontend-action roundtrip has two legs. Leg 1 (the model
calls ``renderKpiTrend``, the run ends, the client executes the handler and
renders the chart) shipped in v1.30.0 and works. Leg 2 broke: CopilotKit then
starts a follow-up run whose message list ends with the action RESULT as a
role-"tool" message so the model can narrate the chart. The execute() bridge
flattened every non-user dict into a plain AssistantMessage and silently
dropped content-less assistant tool-call messages, so the conversation Claude
received ended with an assistant message — the Anthropic API rejects that with
``400 This model does not support assistant message prefill``. Live symptom:
the raw tool-result JSON echoed into the chat, followed by the ⚠️
"temporary error reaching its data and reasoning backend" fallback
(session_1783439832521_8ochpvd, req:969dda13).

Contract under test (``_execute_bridge_agui_messages``):
- dict role "tool" → AG-UI ToolMessage carrying tool_call_id
  (accepts both camelCase ``toolCallId`` and snake_case ``tool_call_id``)
- dict role "assistant" with toolCalls → AssistantMessage PRESERVING
  tool_calls, even when content is empty (previously dropped)
- LangChain ToolMessage / AIMessage(tool_calls) get the same treatment
- a tool result missing its tool_call_id degrades to a UserMessage — a
  dangling tool_result 400s at Anthropic just like a trailing assistant turn
- plain user/assistant conversions and empty-content skipping are unchanged

The SDK layer downstream (``ag_ui_langgraph.utils.agui_messages_to_langchain``)
already converts AG-UI ToolMessage/AssistantMessage.tool_calls correctly; the
bridge was the only lossy hop.
"""

import json

from ag_ui.core import AssistantMessage, UserMessage
from ag_ui.core import ToolMessage as AGUIToolMessage
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.api.routes.copilotkit import _execute_bridge_agui_messages

# ---------------------------------------------------------------------------
# Existing conversions must not regress
# ---------------------------------------------------------------------------


def test_user_dict_converts_to_user_message():
    out = _execute_bridge_agui_messages([{"id": "m1", "role": "user", "content": "hi"}])
    assert len(out) == 1
    assert isinstance(out[0], UserMessage)
    assert out[0].content == "hi"
    assert out[0].id == "m1"


def test_assistant_dict_with_content_converts():
    out = _execute_bridge_agui_messages([{"id": "m1", "role": "assistant", "content": "hello"}])
    assert len(out) == 1
    assert isinstance(out[0], AssistantMessage)
    assert out[0].content == "hello"


def test_empty_content_dict_without_tool_calls_is_dropped():
    out = _execute_bridge_agui_messages(
        [
            {"id": "m1", "role": "user", "content": ""},
            {"id": "m2", "role": "assistant", "content": ""},
        ]
    )
    assert out == []


def test_langchain_human_converts_to_user():
    out = _execute_bridge_agui_messages([HumanMessage(content="hi", id="m1")])
    assert len(out) == 1
    assert isinstance(out[0], UserMessage)
    assert out[0].content == "hi"


# ---------------------------------------------------------------------------
# Tool-result messages (the prefill-400 fix)
# ---------------------------------------------------------------------------


def test_tool_dict_becomes_agui_tool_message():
    out = _execute_bridge_agui_messages(
        [
            {
                "id": "t1",
                "role": "tool",
                "content": '{"kpi_id":"WS3-BI-006","count":35}',
                "toolCallId": "call_1",
            }
        ]
    )
    assert len(out) == 1
    assert isinstance(out[0], AGUIToolMessage)
    assert out[0].tool_call_id == "call_1"
    assert out[0].content == '{"kpi_id":"WS3-BI-006","count":35}'


def test_tool_dict_snake_case_tool_call_id():
    out = _execute_bridge_agui_messages(
        [{"id": "t1", "role": "tool", "content": "ok", "tool_call_id": "call_2"}]
    )
    assert len(out) == 1
    assert isinstance(out[0], AGUIToolMessage)
    assert out[0].tool_call_id == "call_2"


def test_tool_dict_non_string_content_is_json_serialized():
    """CopilotKit action handlers can return objects; AG-UI ToolMessage.content
    must be a string."""
    out = _execute_bridge_agui_messages(
        [
            {
                "id": "t1",
                "role": "tool",
                "content": {"kpi_id": "WS3-BI-006", "points": []},
                "toolCallId": "call_1",
            }
        ]
    )
    assert isinstance(out[0], AGUIToolMessage)
    assert json.loads(out[0].content) == {"kpi_id": "WS3-BI-006", "points": []}


def test_tool_dict_missing_tool_call_id_degrades_to_user_message():
    """A ToolMessage without a pairable tool_call_id would be a dangling
    tool_result — Anthropic 400s. Degrade to a user message so the model still
    sees the data and the conversation stays valid."""
    out = _execute_bridge_agui_messages([{"id": "t1", "role": "tool", "content": "result"}])
    assert len(out) == 1
    assert isinstance(out[0], UserMessage)
    assert "result" in out[0].content


def test_langchain_tool_message_converts():
    out = _execute_bridge_agui_messages(
        [ToolMessage(content="result", tool_call_id="call_9", id="t1")]
    )
    assert len(out) == 1
    assert isinstance(out[0], AGUIToolMessage)
    assert out[0].tool_call_id == "call_9"
    assert out[0].content == "result"


# ---------------------------------------------------------------------------
# Assistant tool-call messages must be preserved (they pair with the results)
# ---------------------------------------------------------------------------


def test_assistant_dict_with_tool_calls_no_content_is_preserved():
    out = _execute_bridge_agui_messages(
        [
            {
                "id": "m1",
                "role": "assistant",
                "content": "",
                "toolCalls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "renderKpiTrend",
                            "arguments": '{"kpiId": "nrx"}',
                        },
                    }
                ],
            }
        ]
    )
    assert len(out) == 1
    assert isinstance(out[0], AssistantMessage)
    assert out[0].tool_calls is not None and len(out[0].tool_calls) == 1
    assert out[0].tool_calls[0].id == "call_1"
    assert out[0].tool_calls[0].function.name == "renderKpiTrend"
    assert json.loads(out[0].tool_calls[0].function.arguments) == {"kpiId": "nrx"}


def test_assistant_dict_snake_case_tool_calls_and_dict_arguments():
    out = _execute_bridge_agui_messages(
        [
            {
                "id": "m1",
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "renderKpiTrend", "arguments": {"kpiId": "trx"}},
                    }
                ],
            }
        ]
    )
    assert len(out) == 1
    assert isinstance(out[0], AssistantMessage)
    assert out[0].tool_calls is not None
    assert json.loads(out[0].tool_calls[0].function.arguments) == {"kpiId": "trx"}


def test_assistant_dict_flat_tool_call_shape():
    """LangChain-style dumped shape: {"id", "name", "args"} without the nested
    "function" wrapper."""
    out = _execute_bridge_agui_messages(
        [
            {
                "id": "m1",
                "role": "assistant",
                "toolCalls": [{"id": "call_1", "name": "renderKpiTrend", "args": {"kpiId": "roi"}}],
            }
        ]
    )
    assert len(out) == 1
    assert isinstance(out[0], AssistantMessage)
    assert out[0].tool_calls is not None
    assert out[0].tool_calls[0].function.name == "renderKpiTrend"
    assert json.loads(out[0].tool_calls[0].function.arguments) == {"kpiId": "roi"}


def test_assistant_dict_malformed_tool_calls_without_content_is_dropped():
    """Unparseable tool calls with no content carry no signal — keep the old
    skip behavior rather than emitting an empty assistant turn."""
    out = _execute_bridge_agui_messages(
        [{"id": "m1", "role": "assistant", "content": "", "toolCalls": [{"bogus": True}]}]
    )
    assert out == []


def test_langchain_ai_message_with_tool_calls_preserved():
    out = _execute_bridge_agui_messages(
        [
            AIMessage(
                content="",
                id="m1",
                tool_calls=[{"name": "renderKpiTrend", "args": {"kpiId": "nrx"}, "id": "call_1"}],
            )
        ]
    )
    assert len(out) == 1
    assert isinstance(out[0], AssistantMessage)
    assert out[0].tool_calls is not None
    assert out[0].tool_calls[0].id == "call_1"
    assert out[0].tool_calls[0].function.name == "renderKpiTrend"
    assert json.loads(out[0].tool_calls[0].function.arguments) == {"kpiId": "nrx"}


# ---------------------------------------------------------------------------
# The full follow-up-run shape that produced the live 400
# ---------------------------------------------------------------------------


def test_tool_roundtrip_conversation_ends_with_tool_message():
    """[user, assistant(toolCalls), tool] must survive conversion intact.
    Previously: the assistant tool-call turn was dropped and the tool result
    became a trailing AssistantMessage → Anthropic prefill 400."""
    out = _execute_bridge_agui_messages(
        [
            {"id": "m1", "role": "user", "content": "plot NRX trends for the last 90 days"},
            {
                "id": "m2",
                "role": "assistant",
                "content": "",
                "toolCalls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "renderKpiTrend", "arguments": '{"kpiId": "nrx"}'},
                    }
                ],
            },
            {
                "id": "m3",
                "role": "tool",
                "content": '{"kpi_id":"WS3-BI-006","count":35,"points":[]}',
                "toolCallId": "call_1",
            },
        ]
    )
    assert [type(m) for m in out] == [UserMessage, AssistantMessage, AGUIToolMessage]
    assert out[1].tool_calls is not None
    assert out[1].tool_calls[0].id == "call_1"
    assert out[2].tool_call_id == "call_1"
