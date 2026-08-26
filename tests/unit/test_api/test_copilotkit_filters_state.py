"""Dashboard-filter state channel for the AG-UI chat brain (2026-08-19 review).

Measured defect: the UI's brand filter had NO channel to the chat graph — the
filter was set to Remibrutinib at 09:44:45 (KPI summary log) yet the 09:51
chat turn still asked "which brand?". The frontend sent brand only via the
CopilotChat ``instructions`` prop (not on the wire) and ``useCopilotReadable``
(on the wire as ``context`` — but the route hard-coded ``context=[]``, see
test_copilotkit_readables_context.py for the 2026-08-26 correction); the chat
node built a static system prompt.

Fix contract (this file pins the backend half):

1. ``E2IAgentState`` declares a ``filters`` channel — LangGraph DROPS input
   keys that are not state channels (the ``copilotkit`` channel comment
   documents the same trap for frontend actions), so without the declaration
   the frontend's filters can never reach the nodes.
2. ``_filters_context_note`` renders the active filters as a system-prompt
   suffix: use-don't-ask semantics, explicit user wording always wins,
   robust to malformed/absent frontend state, and silent (returns "") when
   there is nothing usable — the prompt must stay byte-identical for
   filter-less runs (wave-16 prompt tuning is freshly certified).
3. ``build_synthesis_prompt`` accepts ``filters`` and frames them so the
   synthesizer resolves brand/period from the dashboard instead of re-asking.
4. Both nodes are closures in ``create_e2i_chat_agent`` — coupling is pinned
   by source inspection, the established pattern for this factory.
"""

import inspect

import pytest

from src.api.routes.copilotkit import (
    E2IAgentState,
    _filters_context_note,
    build_synthesis_prompt,
)

pytestmark = pytest.mark.unit


FULL_FILTERS = {
    "brand": "Remibrutinib",
    "territory": "west",
    "dateRange": {"start": "2026-05-21", "end": "2026-08-19"},
    "hcpSegment": "high-prescribers",
}


class TestStateChannel:
    def test_filters_is_a_declared_state_channel(self):
        """LangGraph drops undeclared input keys — without this channel the
        frontend's filters silently vanish before any node runs."""
        assert "filters" in E2IAgentState.__annotations__


class TestFiltersContextNote:
    def test_full_filters_render_brand_and_range(self):
        note = _filters_context_note(FULL_FILTERS)
        assert "Remibrutinib" in note
        assert "2026-05-21" in note and "2026-08-19" in note
        assert "west" in note
        assert "high-prescribers" in note

    def test_use_dont_ask_semantics_are_stated(self):
        """The note must instruct the model to RESOLVE unspecified brand/period
        from the filters instead of asking a clarification — that is the whole
        defect ("which brand?" with the filter set)."""
        note = _filters_context_note(FULL_FILTERS).lower()
        assert "filter" in note
        assert "instead of asking" in note or "without asking" in note

    def test_explicit_user_wording_wins(self):
        """A user naming a different brand mid-chat must override the filter —
        the note has to say so, or the filter would trap the conversation."""
        note = _filters_context_note(FULL_FILTERS).lower()
        assert "user" in note and (
            "wins" in note or "overrides" in note or "takes precedence" in note
        )

    def test_brand_all_is_not_a_brand_constraint(self):
        """brand='All' means no single brand is selected — advertising it as a
        brand would ground every ambiguous query in a fake brand."""
        note = _filters_context_note({**FULL_FILTERS, "brand": "All"})
        assert "brand=All" not in note and "brand: All" not in note

    def test_null_fields_are_omitted(self):
        note = _filters_context_note({"brand": "Kisqali", "territory": None, "hcpSegment": None})
        assert "Kisqali" in note
        assert "None" not in note

    def test_absent_or_malformed_filters_render_nothing(self):
        """Filter-less runs must keep the prompt byte-identical."""
        assert _filters_context_note(None) == ""
        assert _filters_context_note({}) == ""
        assert _filters_context_note("Remibrutinib") == ""  # non-dict garbage
        assert _filters_context_note({"brand": None, "territory": None}) == ""

    def test_non_string_scalars_are_skipped_not_fatal(self):
        note = _filters_context_note({"brand": {"nested": "junk"}, "territory": "east"})
        assert "east" in note
        assert "nested" not in note

    # #1753 — the Home Region selector re-scopes the KPI dashboard, but the
    # note folded brand/dateRange/territory/hcpSegment only: even once the
    # frontend ships region in the CoAgent filters, agent runs stayed blind
    # to it. Same skip-the-sentinel semantics as brand='All'.

    def test_region_renders_when_set(self):
        note = _filters_context_note({**FULL_FILTERS, "region": "West"})
        assert "region=West" in note

    def test_region_all_us_is_not_a_region_constraint(self):
        """region='All US' means no region is selected — advertising it would
        ground every ambiguous query in a fake geographic scope."""
        note = _filters_context_note({**FULL_FILTERS, "region": "All US"})
        assert "region=" not in note

    def test_region_alone_renders(self):
        note = _filters_context_note({"region": "Midwest"})
        assert "region=Midwest" in note

    def test_resolve_instruction_names_region(self):
        """The use-don't-ask sentence enumerates the resolvable fields; region
        must be in that contract or the model has no license to resolve it."""
        note = _filters_context_note({**FULL_FILTERS, "region": "West"})
        resolve_sentence = note.split("When the user's message", 1)[1]
        assert "region" in resolve_sentence


class TestSynthesisPromptFilters:
    def test_filters_section_present_with_brand(self):
        p = build_synthesis_prompt(
            "what is the TRx shortfall?",
            [{"name": "kpi_calculate_tool", "args": {"kpi_name": "TRx"}}],
            [{"tool": "kpi_calculate_tool", "result": '{"value": 4184}'}],
            filters=FULL_FILTERS,
        )
        assert "Remibrutinib" in p
        assert "filter" in p.lower()

    def test_no_filters_keeps_prompt_byte_identical(self):
        args = (
            "q",
            [{"name": "t", "args": {}}],
            [{"tool": "t", "result": "{}"}],
        )
        assert build_synthesis_prompt(*args) == build_synthesis_prompt(*args, filters=None)
        assert build_synthesis_prompt(*args) == build_synthesis_prompt(*args, filters={})


class TestNodeCoupling:
    def _factory_source(self) -> str:
        from src.api.routes.copilotkit import create_e2i_chat_agent

        return inspect.getsource(create_e2i_chat_agent)

    def test_chat_node_folds_filters_into_system_prompt(self):
        assert "_filters_context_note(" in self._factory_source(), (
            "chat_node no longer folds state filters into its system message"
        )

    def test_synthesize_node_passes_filters_to_prompt_builder(self):
        src = self._factory_source()
        assert "filters=" in src.split("build_synthesis_prompt(", 1)[1].split(")")[0] or (
            "filters=state.get" in src
        ), "synthesize_node no longer passes state filters to build_synthesis_prompt"
