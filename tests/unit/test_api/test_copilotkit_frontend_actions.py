"""Unit coverage for chat frontend-action (generative UI) wiring — 2026-07-07.

Why this exists: the frontend has had a complete inline-chart pipeline since
`renderKpiTrend` shipped (CopilotKit action + Recharts `KpiTrendChart` +
`CustomAssistantMessage.generativeUI()`), but the chat agent could never call
it. The delivery chain was verified end-to-end in the installed SDK source:

  useCopilotAction → agent/run body ``tools`` → execute(actions=...) →
  RunAgentInput(tools=...) → LangGraphAGUIAgent.langgraph_default_merge_state
  → state["copilotkit"]["actions"]

…and then BROKE at the graph boundary: ``E2IAgentState`` had no ``copilotkit``
channel, so LangGraph dropped the key before chat_node ever saw it, and
chat_node bound only the backend ``E2I_CHATBOT_TOOLS``.

Contract under test:
- ``E2IAgentState`` declares the ``copilotkit`` channel (schema-drop guard).
- ``_frontend_action_schemas``: converts the actions riding state into
  bind-able OpenAI-style tool schemas; tolerates BOTH parameter shapes
  (JSON-schema object and CopilotKit parameter-array); skips malformed
  entries, duplicates, and names shadowing backend tools.
- ``_route_after_chat``: frontend-action-only tool calls END the run (the
  client executes the action and renders the generative UI); backend calls
  still go to the tools node.
- ``_strip_frontend_calls_when_mixed``: ToolNode only knows backend tools, so
  in the prompt-discouraged mixed case the frontend calls are dropped rather
  than crashing ToolNode.
- The system prompt teaches the model when to call ``renderKpiTrend``.
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.api.routes.copilotkit import (
    E2I_COPILOT_SYSTEM_PROMPT,
    E2IAgentState,
    _frontend_action_names,
    _frontend_action_schemas,
    _is_frontend_only_turn,
    _route_after_chat,
    _strip_frontend_calls_when_mixed,
)

RENDER_KPI_TREND = {
    "name": "renderKpiTrend",
    "description": "Render an inline line chart of a KPI's monthly historical trend.",
    "parameters": {
        "type": "object",
        "properties": {
            "kpiId": {"type": "string", "description": "KPI identifier to chart"},
            "title": {"type": "string", "description": "Optional chart title"},
        },
        "required": ["kpiId"],
    },
}


def _state(actions=None, messages=None) -> dict:
    state: dict = {"messages": messages or []}
    if actions is not None:
        state["copilotkit"] = {"actions": actions, "context": []}
    return state


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


def test_state_schema_carries_copilotkit_channel():
    """Without this channel LangGraph silently drops the injected actions —
    the exact break that made charts impossible."""
    assert "copilotkit" in E2IAgentState.__annotations__


# ---------------------------------------------------------------------------
# _frontend_action_schemas
# ---------------------------------------------------------------------------


def test_action_with_json_schema_parameters_binds_as_is():
    schemas = _frontend_action_schemas(_state(actions=[RENDER_KPI_TREND]))
    assert len(schemas) == 1
    fn = schemas[0]["function"]
    assert schemas[0]["type"] == "function"
    assert fn["name"] == "renderKpiTrend"
    assert fn["parameters"]["properties"]["kpiId"]["type"] == "string"
    assert fn["parameters"]["required"] == ["kpiId"]


def test_action_with_copilotkit_parameter_array_converts_to_json_schema():
    """CopilotKit's native action format carries parameters as an ARRAY of
    {name,type,description,required} — must convert to a JSON-schema object."""
    action = {
        "name": "renderKpiTrend",
        "description": "chart a KPI",
        "parameters": [
            {"name": "kpiId", "type": "string", "description": "KPI id", "required": True},
            {"name": "title", "type": "string", "required": False},
        ],
    }
    schemas = _frontend_action_schemas(_state(actions=[action]))
    params = schemas[0]["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["properties"]) == {"kpiId", "title"}
    assert params["required"] == ["kpiId"]


def test_missing_or_malformed_parameters_become_empty_object_schema():
    schemas = _frontend_action_schemas(
        _state(actions=[{"name": "toggleChat", "description": "toggle", "parameters": None}])
    )
    assert schemas[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_skips_backend_shadow_malformed_and_duplicate_entries():
    actions = [
        {"name": "kpi_calculate_tool", "parameters": {}},  # shadows a backend tool
        "not-a-dict",  # malformed
        {"description": "nameless"},  # malformed
        RENDER_KPI_TREND,
        {**RENDER_KPI_TREND},  # duplicate name
    ]
    schemas = _frontend_action_schemas(_state(actions=actions))
    assert [s["function"]["name"] for s in schemas] == ["renderKpiTrend"]


def test_no_copilotkit_state_yields_no_schemas():
    assert _frontend_action_schemas(_state()) == []
    assert _frontend_action_names(_state()) == set()


# ---------------------------------------------------------------------------
# _route_after_chat
# ---------------------------------------------------------------------------


def _ai_with_calls(*names: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": n, "args": {"kpiId": "trx"}, "id": f"toolu_{i}"} for i, n in enumerate(names)
        ],
    )


def test_frontend_only_tool_calls_end_the_run():
    """The client executes the action and renders the chart — the run must END,
    not hit ToolNode (which doesn't know renderKpiTrend and would error)."""
    state = _state(actions=[RENDER_KPI_TREND], messages=[_ai_with_calls("renderKpiTrend")])
    assert _route_after_chat(state) == "end"


def test_backend_tool_calls_still_route_to_tools():
    state = _state(actions=[RENDER_KPI_TREND], messages=[_ai_with_calls("kpi_calculate_tool")])
    assert _route_after_chat(state) == "tools"


def test_backend_calls_route_to_tools_even_without_actions_in_state():
    state = _state(messages=[_ai_with_calls("kpi_calculate_tool")])
    assert _route_after_chat(state) == "tools"


def test_no_tool_calls_ends():
    state = _state(actions=[RENDER_KPI_TREND], messages=[AIMessage(content="hi")])
    assert _route_after_chat(state) == "end"
    assert _route_after_chat(_state(messages=[HumanMessage(content="hi")])) == "end"
    assert _route_after_chat(_state(messages=[])) == "end"


# ---------------------------------------------------------------------------
# _strip_frontend_calls_when_mixed
# ---------------------------------------------------------------------------


def test_mixed_calls_keep_backend_drop_frontend():
    calls = [
        {"name": "kpi_calculate_tool", "args": {"kpi_name": "TRx"}, "id": "t1"},
        {"name": "renderKpiTrend", "args": {"kpiId": "trx"}, "id": "t2"},
    ]
    kept = _strip_frontend_calls_when_mixed(calls, {"renderKpiTrend"})
    assert [c["name"] for c in kept] == ["kpi_calculate_tool"]


def test_homogeneous_calls_pass_through_unchanged():
    frontend_only = [{"name": "renderKpiTrend", "args": {}, "id": "t1"}]
    backend_only = [{"name": "kpi_calculate_tool", "args": {}, "id": "t1"}]
    assert _strip_frontend_calls_when_mixed(frontend_only, {"renderKpiTrend"}) == frontend_only
    assert _strip_frontend_calls_when_mixed(backend_only, {"renderKpiTrend"}) == backend_only
    assert _strip_frontend_calls_when_mixed(backend_only, set()) == backend_only


# ---------------------------------------------------------------------------
# System prompt guidance
# ---------------------------------------------------------------------------


def test_prompt_teaches_render_kpi_trend():
    assert "renderKpiTrend" in E2I_COPILOT_SYSTEM_PROMPT
    low = E2I_COPILOT_SYSTEM_PROMPT.lower()
    assert "chart" in low
    # Must warn against combining the chart action with other tool calls
    # (mixed turns lose the chart — see _strip_frontend_calls_when_mixed).
    assert "own" in low or "combin" in low


def test_prompt_chart_ids_match_frontend_alias_map():
    """The kpiIds the prompt advertises must be ids the frontend alias map
    resolves to real kpi_history series. v1.30.0 advertised bare ids that
    don't exist in the substrate ('plot NRX trends' → count:0, empty chart).
    Per-brand-only KPIs (nbrx, trx_share) must teach the model to pass brand.
    """
    low = E2I_COPILOT_SYSTEM_PROMPT.lower()
    for kpi_id in ("trx", "nrx", "nbrx", "trx_share", "roi"):
        assert kpi_id in low, f"prompt must advertise supported kpiId {kpi_id!r}"
    # Registry codes pass through the alias map — teach at least one example.
    assert "ws3-bi-" in low
    # The per-brand-only KPIs return honest-empty without a brand argument.
    assert "per brand only" in low


# ---------------------------------------------------------------------------
# Generative-UI turn analytics (#1383 follow-up)
# ---------------------------------------------------------------------------
#
# Why this exists: chat_node returns early for EVERY tool turn, before the
# direct-response analytics call. A backend-tool turn is still recorded later,
# by synthesize_node once results come back. A frontend-action-only turn has no
# second act — _route_after_chat sends it straight to END for client-side
# execution — so nothing downstream recorded it and the turn produced NO
# chat_analytics row at all. That blind spot is why "which chart action do
# users actually get?" was unanswerable from the data.
#
# The recording condition and the routing condition MUST agree: record a turn
# that does not end and the row is a lie; miss a turn that does and the blind
# spot survives. Both now call _is_frontend_only_turn, and these tests pin it.


def test_frontend_only_turn_is_recognised():
    assert _is_frontend_only_turn(["renderChart"], {"renderChart", "renderKpiTrend"})
    assert _is_frontend_only_turn(
        ["renderChart", "renderKpiTrend"], {"renderChart", "renderKpiTrend"}
    )


def test_backend_turn_is_not_a_generative_ui_turn():
    # A backend turn routes to the tools node and is recorded by synthesize_node;
    # recording it here too would double-count it.
    assert not _is_frontend_only_turn(["kpi_calculate_tool"], {"renderChart"})


def test_mixed_turn_is_not_a_generative_ui_turn():
    # Mixed turns keep their backend calls (see _strip_frontend_calls_when_mixed)
    # and therefore still reach synthesize_node.
    assert not _is_frontend_only_turn(["kpi_calculate_tool", "renderChart"], {"renderChart"})


def test_no_calls_and_no_actions_are_not_generative_ui_turns():
    # A direct text answer falls through to the existing direct_response row;
    # an empty action registry means nothing can be a frontend call.
    assert not _is_frontend_only_turn([], {"renderChart"})
    assert not _is_frontend_only_turn(["renderChart"], set())


def test_routing_and_recording_share_one_condition():
    # The guard against the two drifting apart: every turn _route_after_chat
    # ends as frontend-only must be exactly the set _is_frontend_only_turn
    # accepts, since a recorded-but-not-ended turn would be a false row.
    state = {
        "copilotkit": {"actions": [{"name": "renderChart", "description": "d", "parameters": []}]}
    }
    frontend_names = _frontend_action_names(state)

    for call_names, expect_end in [
        (["renderChart"], True),
        (["kpi_calculate_tool"], False),
        (["kpi_calculate_tool", "renderChart"], False),
    ]:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": n, "args": {}, "id": f"c{i}"} for i, n in enumerate(call_names)],
        )
        routed_end = _route_after_chat({**state, "messages": [msg]}) == "end"
        assert routed_end is expect_end
        assert _is_frontend_only_turn(call_names, frontend_names) is expect_end
