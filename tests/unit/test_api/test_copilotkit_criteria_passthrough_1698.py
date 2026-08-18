"""#1698 copilotkit-side surfaces: the raw-ask side channel and the prompt rule.

The tool-chain legs (orchestrator_tool -> resolver -> agent merge) are covered
in tests/unit/test_agents/test_orchestrator/test_cohort_criteria_passthrough_1698.py;
this file pins the two surfaces that live in copilotkit.py: the helper that
extracts the verbatim latest user message for ``set_raw_user_query``, and the
prompt sentence telling the model not to pre-filter cohort criteria (the 2.1
rewrite dropped "adults over 18" / "diagnosed in 2024" before dispatch).
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT, _latest_user_text

ORIGINAL = (
    "Build a patient cohort for Remibrutinib CSU with inclusion criteria "
    "for adults over 18 diagnosed in 2024"
)


def test_latest_user_text_dict_shape():
    messages = [
        {"role": "user", "content": "earlier ask", "id": "m1"},
        {"role": "assistant", "content": "earlier answer", "id": "m2"},
        {"role": "user", "content": ORIGINAL, "id": "m3"},
    ]
    assert _latest_user_text(messages) == ORIGINAL


def test_latest_user_text_langchain_shape():
    messages = [
        HumanMessage(content="earlier ask"),
        AIMessage(content="earlier answer"),
        HumanMessage(content=ORIGINAL),
    ]
    assert _latest_user_text(messages) == ORIGINAL


def test_latest_user_text_empty_and_nonstring():
    assert _latest_user_text([]) == ""
    assert _latest_user_text(None) == ""
    # Multimodal/parts content yields "" rather than a guess.
    assert _latest_user_text([{"role": "user", "content": [{"type": "text"}]}]) == ""


def test_prompt_forbids_prefiltering_cohort_criteria():
    assert "through in the query VERBATIM" in E2I_COPILOT_SYSTEM_PROMPT
    assert (
        "never present your own rewrite's omission as a platform limitation"
        in E2I_COPILOT_SYSTEM_PROMPT
    )
