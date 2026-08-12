"""System-prompt pins for the #1549 run-nearest-supported-analysis directive.

Measured defect (2026-08-12 post-#1546 re-measure, wave2_premeasure): turns
4.1/4.2/4.3 invoke ZERO tools, name the exact runnable supported analysis
themselves ("I could show likelihood-to-prescribe for Kisqali ranked by
specialty — tell me brand and axis and I'll pull it"), then ask-end instead
of running it. The residual is attitudinal — the chat LLM's own decision —
so the fix seam is the AG-UI system prompt: a decision rule that says RUN
the nearest supported analysis in the same turn with stated defaults, and
reserves ask-ending for (a) genuinely undefined referents (the A.5 cold
"why did it drop?" control) and (b) capability refusals with no adjacent
supported analysis (the 2.1 clinical cohort-building control).

These tests pin the directive's load-bearing markers AND the pre-existing
clarify/honesty language the directive must not weaken (#1407's clarify
gate lives on the /chat brain in chatbot_graph.py — this surface's honesty
language is the Guidelines + HONESTY GUARD + per-tool no-guess rules).
"""

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT

LOW = E2I_COPILOT_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# The new #1549 decision rule
# ---------------------------------------------------------------------------


def test_prompt_directs_running_nearest_supported_analysis_same_turn():
    """The core rule: when the ask doesn't map 1:1 to a tool but a supported
    analysis answers the nearest useful version, RUN it in the same turn."""
    assert "nearest" in LOW
    assert "same turn" in LOW
    # Naming-the-analysis-then-asking-permission is called out as the
    # failure mode the rule forbids.
    assert "failure mode" in LOW


def test_prompt_defines_session_referent_brand_as_user_provided():
    """A brand named anywhere in the conversation is user-provided, not a
    guess — proceeding on it (stated as an assumption) is required, which is
    the 5.5 acceptance and the 4.1-class default-brand rule."""
    assert "counts as user-provided" in LOW


def test_prompt_requires_stating_assumptions_and_offering_other_slices_after():
    """Defaults must be stated as assumptions, and other slices offered
    AFTER the data — not instead of it (5.5's leading-clarify residual)."""
    assert "assumption" in LOW
    assert "after" in LOW and "not instead of" in LOW


def test_prompt_reserves_ask_ending_for_undefined_referent():
    """A.5 control: a cold ask with NO entity/metric anywhere in the ask or
    conversation must still clarify — the rule names that case explicitly."""
    assert "genuinely undefined" in LOW


def test_prompt_reserves_ask_ending_for_capability_refusal():
    """2.1 control: requests outside supported analyses with nothing
    adjacent still get an honest decline."""
    assert "no supported analysis answers a nearby version" in LOW


# ---------------------------------------------------------------------------
# Pre-existing clarify/honesty language must NOT be weakened by the directive
# ---------------------------------------------------------------------------


def test_prompt_keeps_honesty_guard():
    assert "HONESTY GUARD" in E2I_COPILOT_SYSTEM_PROMPT
    assert "do NOT fabricate" in E2I_COPILOT_SYSTEM_PROMPT


def test_prompt_keeps_no_guess_brand_rule_for_segment_tool():
    """The predict tool's no-guess rule stays: a brand is either named
    somewhere in the conversation (then use it) or genuinely absent (then
    ask). Guessing is never allowed."""
    assert "do NOT guess one" in E2I_COPILOT_SYSTEM_PROMPT


def test_prompt_keeps_honest_windows_guideline():
    assert "never ask for a brand or period the user already gave" in LOW
