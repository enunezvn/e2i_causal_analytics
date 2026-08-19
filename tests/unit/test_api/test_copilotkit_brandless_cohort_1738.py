"""#1738: brand-less cohort-profile asks must dispatch ALL-BRANDS, not ask-first.

Measured route regression (post1730 full eval, sha 3e62df821): eval turns
4.1/4.2/4.3 — brand-less cohort asks ("Create a cohort of HCPs who are
oncologists in the northeast", ...) — ALL reverted from real cohort_profiler
dispatches (post1708 baseline: 3/3 ``orchestrator_tool`` TOOL_CALL_START,
answers headed "all brands") to zero-tool brand clarifications.

Live probes (2026-08-19, deploy 5ea9ed0a9, same wire shape as the eval
runner, same 4.1 question text, judged from TOOL_CALL_START/ARGS only):

- (a) ``state.filters.brand="Kisqali"``  -> 3/3 dispatch, ``brand:"Kisqali"``
  in the tool args (the #1724 seam works and must keep working);
- (b) filters ABSENT (``state: {}``, the runner/eval default) -> 1/3 dispatch;
- (c) filters present-but-empty (``{"brand": null}``)          -> 1/3 dispatch.

(b) == (c) statistically AND structurally: ``_filters_context_note`` renders
"" for both (pinned by test_copilotkit_filters_state.py), so the prompts are
byte-identical — the issue's hypothesis that #1724's fold *absence-handling*
is the trigger is REFUTED. The trigger is the prompt itself: the #1562 cohort
clause ("if no brand appears anywhere in the conversation, ask which brand")
put the brand-less cohort route on a knife edge, and the wave-16/17 prompt
additions tipped the sampled equilibrium from 3/3 dispatch to clarify.

#1738's acceptance supersedes #1562's ask-first coexistence note: brand-less
cohort-profile asks dispatch CROSS-BRAND in the same turn (the baseline
behavior the platform's own evals grade as correct — gold for 4.1-4.3 is
SINGLE_AGENT cohort_profiler). The protection the old clause actually
provided — never SILENTLY GUESS a single brand — survives verbatim in the
new wording: all-brands is the stated default, a single-brand guess stays
forbidden, and the dashboard-filter / conversation-brand resolution paths
are untouched.
"""

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT, _filters_context_note

LOW = E2I_COPILOT_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# The ask-first cohort clause is gone (the #1738 regression trigger)
# ---------------------------------------------------------------------------


def test_cohort_clause_no_longer_instructs_ask_first():
    """The exact #1562 sub-clause that routed brand-less cohort asks to a
    clarification wall must not survive. Scoped by its unique parenthetical:
    `predict_hcp_segment_likelihood_tool`'s ask-which-brand text is CORRECT
    (that tool genuinely requires a brand) and stays."""
    assert (
        "if no brand appears anywhere in the conversation, ask which brand "
        "(an all-brands profile is a valid option to offer)" not in E2I_COPILOT_SYSTEM_PROMPT
    )


def test_brandless_cohort_asks_dispatch_all_brands_in_the_same_turn():
    """The replacement directive: no brand anywhere + no dashboard brand
    filter -> dispatch the all-brands aggregate profile, same turn."""
    assert "dispatch the all-brands cohort profile in the same turn" in LOW


def test_brand_is_optional_for_cohort_profiles_never_a_prerequisite():
    """Root framing: a cohort ask defined by specialty/region/volume criteria
    is complete without a brand — brand is a filter, not a gate."""
    assert "never a prerequisite" in LOW


def test_all_brands_scope_is_stated_inline_not_silently_assumed():
    """The old clause's real protection, kept: the answer names its scope
    ('all brands') and a single-brand guess stays forbidden."""
    assert "state 'all brands' as the scope inline" in LOW
    assert "never silently guess a single brand" in LOW


# ---------------------------------------------------------------------------
# The ask-ending case-1 rule no longer swallows brand-less cohort asks
# ---------------------------------------------------------------------------


def test_case1_carves_out_cohort_profile_asks():
    """'no brand anywhere is case 1 below' (the #1549 ask-ending default)
    must carve out cohort-profile asks, or the prompt self-contradicts and
    the route stays a coin flip."""
    assert "no brand anywhere is case 1 below" in LOW
    assert "except aggregate cohort-profile asks" in LOW


# ---------------------------------------------------------------------------
# The #1724 seam is untouched (probe (a) must keep working after deploy)
# ---------------------------------------------------------------------------


def test_dashboard_brand_filter_still_resolves_and_note_text_unchanged():
    """Probe (a): filters.brand set -> the fold renders the resolve-don't-ask
    note verbatim. Value-pinned so a wording drift here can't silently change
    the certified #1724 behavior."""
    note = _filters_context_note({"brand": "Kisqali"})
    assert "brand=Kisqali" in note
    assert "resolve it from these filters instead of asking" in note


def test_absent_and_empty_filters_stay_byte_identical():
    """Probes (b)/(c): the eval runner's ``state: {}`` (filters key absent)
    and the frontend's present-but-empty ``{"brand": null}`` must keep
    producing the identical (empty) prompt suffix — the measured basis for
    refuting the fold-absence hypothesis, pinned here as the #1738
    acceptance guarantee ('absent and empty filters behave IDENTICALLY')."""
    assert _filters_context_note(None) == ""
    assert _filters_context_note({}) == ""
    assert _filters_context_note({"brand": None}) == ""
