"""System-prompt pins for the #1562 cohort-path guidance.

Measured defect (2026-08-12 wave-2 both-surface sweep): the AG-UI brain told
turn 4.1 "individual HCP identities/lists aren't something this platform
surfaces — that's outside what the analytics layer supports" and turn 2.1
"I'm not able to build or export a patient-level cohort list — that's
outside what this platform supports", while the /chat brain served the SAME
asks the SAME day via cohort_profiler (4.1: 3,423 HCPs / 36,734 TRx by
specialty and tier; 2.1: 2,676 new-Rx Remibrutinib patients by severity,
refusing only the unservable diagnosis-year criterion).

The disclaimer is right about the individual level and wrong about cohort
profiling: aggregate cohort profiles are a shipped capability reachable
from this brain via ``orchestrator_tool`` (pattern selector maps
``Domain.COHORT_DEFINITION`` to cohort_profiler; the dispatcher grounds its
input from chat context and never fails closed). The prompt just never
advertised the path.

These tests pin (a) the new cohort-path guidance, (b) the disclaimer being
scoped to what is genuinely unsupported (named individual rosters/exports,
unservable clinical criteria), and (c) the #1549 honesty rules surviving
verbatim — the clarify rules and the accurate capability description must
coexist (issue acceptance: a zero-brand fresh-session cohort ask still asks
which brand rather than guessing).
"""

from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT

LOW = E2I_COPILOT_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# (a) The new cohort-path guidance
# ---------------------------------------------------------------------------


def test_prompt_routes_cohort_asks_through_orchestrator_tool():
    """Cohort/profile asks are served, not declined: the prompt must name
    cohort_profiler as reachable through orchestrator_tool."""
    assert "cohort_profiler" in LOW
    assert "aggregate hcp or patient cohort profiles" in LOW
    assert "are a supported analysis" in LOW


def test_prompt_says_aggregate_cohort_profiling_is_not_a_refusal():
    """The over-abstention 4.1/2.1 answers claimed the capability doesn't
    exist — the prompt must state the opposite explicitly."""
    assert "not a refusal case" in LOW


def test_prompt_offers_aggregate_profile_when_roster_is_refused():
    """A roster ask gets the one-sentence limit plus the aggregate profile
    run/offer — never a whole-ask decline (issue acceptance: serve it, or
    accurately describe and offer it)."""
    assert "aggregate cohort profile" in LOW
    assert "instead of declining the whole ask" in LOW


def test_cohort_asks_keep_the_brand_clarify_rule():
    """Issue acceptance: the accurate capability description and the
    zero-brand clarify COEXIST — a fresh-session cohort ask with no brand
    still asks which brand (all-brands offered as an option, not silently
    assumed)."""
    assert "ask which brand" in LOW
    assert "an all-brands profile" in LOW


# ---------------------------------------------------------------------------
# (b) Disclaimer scoped to what is genuinely unsupported
# ---------------------------------------------------------------------------


def test_refusal_example_names_individual_rosters_not_cohort_lists():
    """The genuinely-unsupported surface is the INDIVIDUAL level."""
    assert "named individual" in LOW
    assert "rosters" in LOW


def test_stale_overbroad_refusal_examples_are_gone():
    """The old case-2 example ('per-HCP/patient-level cohort lists',
    'clinical cohort definitions') taught the brain that cohort profiling
    itself is a refusal — /chat serves criteria-bound aggregate cohort
    profiles, refusing only unservable criteria, so the blanket wording
    must not survive."""
    assert "per-hcp/patient-level cohort lists" not in LOW
    assert "clinical cohort definitions" not in LOW


# ---------------------------------------------------------------------------
# (c) #1549 honesty rules survive verbatim (PR #1559 must not regress)
# ---------------------------------------------------------------------------


def test_zero_brand_do_not_guess_rule_survives_verbatim():
    """The #1549 zero-brand rule: brand is required and never guessed."""
    assert "do NOT guess one" in E2I_COPILOT_SYSTEM_PROMPT


def test_single_brand_session_referent_rule_survives():
    """The #1549 session-referent rule: a single brand named anywhere in
    the conversation counts as user-provided."""
    assert "counts as user-provided" in LOW


def test_ask_ending_reservations_survive():
    """Both #1549 ask-ending cases keep their load-bearing markers."""
    assert "genuinely undefined" in LOW
    assert "no supported analysis answers a nearby version" in LOW


def test_honesty_guard_survives():
    assert "HONESTY GUARD" in E2I_COPILOT_SYSTEM_PROMPT
    assert "do NOT fabricate" in E2I_COPILOT_SYSTEM_PROMPT
