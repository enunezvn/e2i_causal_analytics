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


def test_note_without_prose_is_byte_identical_to_the_pre_summary_wording():
    """Pages that publish no summary must keep the exact prompt they had before
    the summary readable existed (JSON readables only)."""
    state = {
        "context": [
            {
                "description": "Active dashboard filters",
                "value": '{"brand": "Kisqali", "region": "All US"}',
            },
            {"description": "Current route", "value": '"/kpi-dictionary"'},
        ]
    }
    note = _readables_context_note(state)
    assert "values are JSON):\n" in note
    assert "prose" not in note
    assert "not a data table" not in note
    assert note.endswith("Call tools only for data that is not on screen.")


def test_note_mixed_json_and_prose_uses_the_prose_wording():
    state = {
        "context": [
            {"description": "Active dashboard filters", "value": '{"brand": "Kisqali"}'},
            {
                "description": "Summary of the data currently visible on the page",
                "value": "Feature importance for Kisqali: top drivers are age and prior therapy.",
            },
        ]
    }
    note = _readables_context_note(state)
    assert "values are JSON or short prose summaries" in note
    assert "not a data table" in note
    assert "top drivers are age" in note
