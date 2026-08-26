"""AG-UI readables (``useCopilotReadable``) reach the chat graph — 2026-08-26.

Measured defect (trace ``session_1787762049084_4psbqsx``, live wire capture on
eznomics.site/predictive-analytics): the browser's ``agent/run`` body carries
every ``useCopilotReadable`` value in ``body.context`` (react-core 1.51.2) —
filters, preferences, ``{currentPath}``, agent roster — and the backend threw
them away: the route handler never read ``context`` and ``execute()``
hard-coded ``RunAgentInput(context=[])``. The 2026-08-19 review wrote
"readables never leave the browser" into three code comments after reading the
``RUN_STARTED`` echo, which shows the backend's ZEROED value, not the request.
Positive control: the agent quoted the date-range filter (CoAgent state
channel) while answering "I don't have access to your current page path".

Contract under test (backend half):

1. ``_coerce_agui_context`` turns the raw wire list into ``Context``-shaped
   dicts and DROPS junk — a malformed readable must never 500 the run through
   ``RunAgentInput`` validation.
2. ``execute()`` forwards ``context`` into ``RunAgentInput`` (the SDK's
   ``langgraph_default_merge_state`` then lands it at
   ``state["copilotkit"]["context"]`` as ``Context`` pydantic objects).
3. ``_readables_context_note`` renders ``state["copilotkit"]["context"]`` as
   a system-prompt suffix — accepts pydantic ``Context`` objects AND dicts,
   caps size, skips junk, and is "" when nothing usable (prompt stays
   byte-identical for readable-less runs).
4. chat_node folds the note into its system message; ``build_synthesis_prompt``
   accepts ``readables`` so a tool-calling turn keeps the on-screen context.
5. The route handler reads ``context`` from the agent/run body (source pin —
   the handler is a closure, the established pattern for this route).
"""

from __future__ import annotations

import inspect
import json

import pytest
from ag_ui.core import Context, RunAgentInput

from src.api.routes.copilotkit import (
    _READABLE_ITEM_MAX_CHARS,
    _READABLES_TOTAL_MAX_CHARS,
    _coerce_agui_context,
    _readables_context_note,
    build_synthesis_prompt,
    create_e2i_chat_agent,
)

pytestmark = pytest.mark.unit


WIRE_CONTEXT = [
    {
        "description": "Current page path and navigation context",
        "value": '{"currentPath":"/predictive-analytics"}',
    },
    {
        "description": "Predictive Analytics: scored holdout cohort on screen",
        "value": json.dumps(
            {
                "model_name": "initiation_remibrutinib_goldstd_lr_v1",
                "n_scored": 1234,
                "distribution": {"bin_edges": [0, 0.5, 1], "bin_counts": [1000, 234]},
                "top_rows": [{"rank": 1, "entity_id": "scvpt_1", "probability": 0.97}],
            }
        ),
    },
]


# ---------------------------------------------------------------------------
# 1. _coerce_agui_context
# ---------------------------------------------------------------------------


class TestCoerceAguiContext:
    def test_wire_items_pass_through(self):
        out = _coerce_agui_context(WIRE_CONTEXT)
        assert out == WIRE_CONTEXT
        # Must be RunAgentInput-valid — this is the exact shape that was 500-risk.
        RunAgentInput(
            thread_id="t",
            run_id="r",
            state={},
            messages=[],
            tools=[],
            context=out,
            forwarded_props={},
        )

    def test_junk_is_dropped_not_fatal(self):
        raw = [
            "not-a-dict",
            {"description": "no value"},
            {"value": "no description"},
            {"description": 42, "value": "x"},
            None,
            WIRE_CONTEXT[0],
        ]
        assert _coerce_agui_context(raw) == [WIRE_CONTEXT[0]]

    def test_non_string_values_are_json_stringified(self):
        """Older clients (and hand-rolled callers) may send the raw object."""
        out = _coerce_agui_context([{"description": "d", "value": {"a": 1}}])
        assert out == [{"description": "d", "value": '{"a": 1}'}]

    def test_absent_or_non_list_is_empty(self):
        assert _coerce_agui_context(None) == []
        assert _coerce_agui_context({}) == []
        assert _coerce_agui_context("garbage") == []


# ---------------------------------------------------------------------------
# 2. execute() forwards context into RunAgentInput
# ---------------------------------------------------------------------------


class TestExecuteForwardsContext:
    def test_execute_accepts_context_and_no_longer_hardcodes_empty(self):
        from src.api.routes.copilotkit import LangGraphAgent

        sig = inspect.signature(LangGraphAgent.execute)
        assert "context" in sig.parameters, "execute() has no context parameter"
        src = inspect.getsource(LangGraphAgent.execute)
        assert "context=[]" not in src, "execute() still zeroes the readables"
        assert "_coerce_agui_context(" in src

    def test_route_handler_reads_context_from_agent_run_body(self):
        from src.api.routes import copilotkit as module

        src = inspect.getsource(module)
        handler = src.split('if agui_method == "agent/run":', 1)[1]
        assert 'body_data.get("context")' in handler, "agent/run handler ignores body.context"
        # First "agent.execute(" in the handler is a comment; anchor on the call.
        execute_call = handler.split("async for event in agent.execute(", 1)[1].split("):", 1)[0]
        assert "context=context" in execute_call, "handler does not pass context to execute()"


# ---------------------------------------------------------------------------
# 3. _readables_context_note
# ---------------------------------------------------------------------------


class TestReadablesContextNote:
    def test_renders_dict_items(self):
        note = _readables_context_note({"context": WIRE_CONTEXT})
        assert "/predictive-analytics" in note
        assert "initiation_remibrutinib_goldstd_lr_v1" in note
        assert "Current page path" in note

    def test_renders_pydantic_context_objects(self):
        """The SDK's merge_state stores ag_ui Context models, not dicts."""
        items = [Context(**c) for c in WIRE_CONTEXT]
        note = _readables_context_note({"context": items})
        assert "/predictive-analytics" in note
        assert "n_scored" in note

    def test_instructs_model_to_use_on_screen_data(self):
        """The whole point: when the user says 'the data is on the GUI', the
        model must know this block IS the GUI and answer from it."""
        note = _readables_context_note({"context": WIRE_CONTEXT}).lower()
        assert "on screen" in note or "on-screen" in note or "looking at" in note
        assert "answer" in note

    def test_absent_or_empty_renders_nothing(self):
        assert _readables_context_note(None) == ""
        assert _readables_context_note({}) == ""
        assert _readables_context_note({"context": []}) == ""
        assert _readables_context_note({"context": None}) == ""
        assert _readables_context_note("garbage") == ""
        assert _readables_context_note({"actions": [{"name": "x"}]}) == ""

    def test_junk_items_are_skipped_not_fatal(self):
        note = _readables_context_note(
            {"context": ["junk", None, {"description": "only"}, WIRE_CONTEXT[0]]}
        )
        assert "/predictive-analytics" in note
        assert "junk" not in note

    def test_empty_value_items_are_skipped(self):
        assert _readables_context_note({"context": [{"description": "d", "value": ""}]}) == ""

    def test_long_item_is_truncated_with_marker(self):
        big = {"description": "big", "value": "x" * (_READABLE_ITEM_MAX_CHARS + 500)}
        note = _readables_context_note({"context": [big]})
        assert "truncated" in note
        assert "x" * (_READABLE_ITEM_MAX_CHARS + 1) not in note

    def test_total_budget_is_bounded(self):
        many = [
            {"description": f"item {i}", "value": "y" * (_READABLE_ITEM_MAX_CHARS - 10)}
            for i in range(50)
        ]
        note = _readables_context_note({"context": many})
        assert len(note) <= _READABLES_TOTAL_MAX_CHARS + 1000  # header + marker slack
        assert "omitted" in note


# ---------------------------------------------------------------------------
# 4. Node coupling + synthesis prompt
# ---------------------------------------------------------------------------


class TestNodeCoupling:
    def test_chat_node_folds_readables_into_system_prompt(self):
        src = inspect.getsource(create_e2i_chat_agent)
        assert "_readables_context_note(" in src, (
            "chat_node does not fold state['copilotkit']['context'] into its system message"
        )

    def test_synthesize_node_passes_readables_to_prompt_builder(self):
        src = inspect.getsource(create_e2i_chat_agent)
        # Nested call args carry their own ")" — window on the closing "\n            )".
        call = src.split("build_synthesis_prompt(", 1)[1].split("\n            )", 1)[0]
        assert 'readables=state.get("copilotkit")' in call


class TestSynthesisPromptReadables:
    ARGS = (
        "how many are above 90%?",
        [{"name": "t", "args": {}}],
        [{"tool": "t", "result": "{}"}],
    )

    def test_readables_section_present(self):
        p = build_synthesis_prompt(*self.ARGS, readables={"context": WIRE_CONTEXT})
        assert "/predictive-analytics" in p
        assert "n_scored" in p

    def test_no_readables_keeps_prompt_byte_identical(self):
        base = build_synthesis_prompt(*self.ARGS)
        assert base == build_synthesis_prompt(*self.ARGS, readables=None)
        assert base == build_synthesis_prompt(*self.ARGS, readables={})
        assert base == build_synthesis_prompt(*self.ARGS, readables={"context": []})
