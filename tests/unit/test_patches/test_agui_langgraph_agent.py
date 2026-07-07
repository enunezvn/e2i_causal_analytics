"""Regression tests for the vendored ag-ui-langgraph patch.

These run against the installed ``ag_ui_langgraph`` package, which pip
path-installs from ``patches/ag-ui-langgraph`` (the same source the docker
image bakes in), so they guard the vendored copy we actually ship.
"""

from ag_ui_langgraph.agent import LangGraphAgent


def _bare_agent() -> LangGraphAgent:
    # __init__ needs a compiled graph; the method under test only touches
    # ``messages_in_process``, so bypass construction.
    agent = object.__new__(LangGraphAgent)
    agent.messages_in_process = {}
    return agent


class TestSetMessageInProgress:
    def test_updates_existing_entry(self):
        agent = _bare_agent()
        agent.messages_in_process["run-1"] = {"id": "m1"}
        agent.set_message_in_progress("run-1", {"tool_call_id": "t1"})
        assert agent.messages_in_process["run-1"] == {"id": "m1", "tool_call_id": "t1"}

    def test_missing_key_starts_fresh(self):
        agent = _bare_agent()
        agent.set_message_in_progress("run-1", {"id": "m1"})
        assert agent.messages_in_process["run-1"] == {"id": "m1"}

    def test_none_end_marker_does_not_crash(self):
        """TEXT_MESSAGE_END stores ``None`` under the run id as an "ended"
        marker; a subsequent TOOL_CALL_START in the same run (model streams
        an ack text, then calls a tool) must not raise
        ``TypeError: 'NoneType' object is not a mapping`` — live incident
        2026-07-07, sidebar gap-analysis run hung at "Working..."."""
        agent = _bare_agent()
        agent.messages_in_process["run-1"] = None
        agent.set_message_in_progress(
            "run-1", {"id": "m2", "tool_call_id": "t1", "tool_call_name": "orchestrator"}
        )
        assert agent.messages_in_process["run-1"]["id"] == "m2"
