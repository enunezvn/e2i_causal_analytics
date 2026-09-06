"""_readables_context_note wording: readables can be JSON or prose page summaries (2026-09-05)."""

from src.api.routes.copilotkit import _readables_context_note


def test_note_renders_prose_summary_and_says_how_to_treat_it():
    state = {
        "copilotkit": {
            "context": [
                {
                    "description": "Summary of the data currently visible on the page",
                    "value": "Home dashboard. Brand filter: Kisqali; region: All US.",
                }
            ]
        }
    }
    note = _readables_context_note(state["copilotkit"])
    assert "Home dashboard. Brand filter: Kisqali" in note
    assert "values are JSON or short prose summaries" in note
    assert "not a data table" in note


def test_note_is_empty_without_readables():
    assert _readables_context_note({"context": []}) == ""
    assert _readables_context_note(None) == ""
