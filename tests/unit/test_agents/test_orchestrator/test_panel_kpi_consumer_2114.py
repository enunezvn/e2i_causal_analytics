"""The panel vocabulary ANSWERS through the real consumer path, and #1475 stays armed.

Entry-gate first: these phrasings are the ones that pass KPI_VALUE_LOOKUP_RE. The bare
aliases ("NRx panel") are correct for the RESOLVER tests and wrong here — they have no
interrogative lead-in, so they never reach the guards and would assert nothing.

The seam is the REAL one: _kpi_lookup_evidence does `from src.api.routes.kpi import
get_kpi_calculator` INSIDE the function and calls `.calculate(kpi.id, context=context)`
(dispatcher.py). Patch that name. Asserting only "evidence is not None" would let a
resolver-correct/consumer-wrong path pass, and asserting only "evidence is None" on the
refusal cases would let an unrelated calculator exception masquerade as guard enforcement
— the engine call is wrapped in `except Exception: return None` (codex r19-02).
"""

from typing import Any, Dict, List

import pytest

from src.agents.orchestrator.nodes.dispatcher import _kpi_lookup_evidence
from src.kpi.models import KPIResult, KPIStatus

#: The REAL result model, not a hand-rolled stand-in.
#:
#: The plan's draft used a 4-field dataclass (`value`, `error`, `window_status`,
#: `metadata`). `_kpi_lookup_evidence` reads SEVEN attributes off this object --
#: measured: value, status, error, metadata, window_requested, window_applied,
#: window_status -- so the stub raised `AttributeError: '_Result' object has no
#: attribute 'status'` at dispatcher.py:2369 and all five answer cases failed for a
#: reason that had nothing to do with the panel vocabulary they were written to test.
#:
#: Binding to `KPIResult` instead of listing the seven means the double cannot drift
#: from the contract: a field added to the real model arrives here automatically, and
#: one removed fails loudly. Same lesson as the r6 wrong-layer fake -- a double is only
#: evidence while it is faithful to the thing it stands in for.


class _RecordingCalculator:
    """⚠ `calls` RECORDS IDENTITY; `contexts` RECORDS WHAT WAS ASKED FOR.

    Until r12 this double accepted `context` and threw it away, so every battery row
    could assert WHICH KPI answered and none could see that the QUALIFIER had been
    dropped — seven rows were answering with `{}` under an assertion that looked
    thorough. A double that discards an argument cannot witness anything about it, and
    that is the same family as the fake that ignores its arguments.

    `calls` keeps its list-of-ids shape deliberately: dozens of tests assert
    `calculator.calls == [id]`, and widening it would have rewritten them all to fix a
    blindness that a second attribute closes.
    """

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.contexts: List[Dict[str, Any]] = []

    def calculate(self, kpi_id: str, context: Dict[str, Any]) -> KPIResult:
        self.calls.append(kpi_id)
        self.contexts.append(dict(context or {}))
        return KPIResult(
            kpi_id=kpi_id,
            value=123.0,
            status=KPIStatus.INFORMATIONAL,
            metadata={"context": {"data_through": "2026-08-31"}},
        )


@pytest.fixture
def calculator(monkeypatch) -> _RecordingCalculator:
    rec = _RecordingCalculator()
    monkeypatch.setattr("src.api.routes.kpi.get_kpi_calculator", lambda: rec)
    return rec


@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("What is NRx panel for Kisqali?", "WS3-BI-012"),
        ("What is the TRx panel for Kisqali?", "WS3-BI-011"),
        ("What is panel NBRx for Kisqali?", "WS3-BI-013"),
        ("What is TRx share panel for Kisqali?", "WS3-BI-014"),
        ("show me the NRx panel", "WS3-BI-012"),
    ],
)
def test_a_panel_lookup_resolves_and_answers(query, expected_id, calculator):
    """Branch 1 of the Step 4 triage: a legitimate panel lookup must RESOLVE AND ANSWER.
    The calculator must RECEIVE the expected id, and the evidence must carry its value."""
    evidence = _kpi_lookup_evidence({"query": query})
    assert calculator.calls == [expected_id], (query, calculator.calls)
    assert evidence, f"{query!r} produced no evidence — it did not answer"
    rendered = " ".join(str(e) for e in evidence)
    assert "123" in rendered, (query, rendered[:200])


@pytest.mark.parametrize(
    "query,why",
    [
        (
            "What is the cost of NRx panel?",
            "governing 'of' head — the KPI is a modifier, not the asked value",
        ),
        ("What are NBRx panel drivers?", "causal right-head — a bare value does not answer it"),
    ],
)
def test_the_1475_guards_are_still_armed_over_panel_phrasings(query, why, calculator):
    """Branch 2: a cost / drivers phrasing must KEEP its refusal, and must be refused BEFORE
    the engine is consulted. Zero calculator calls is the assertion that distinguishes a guard
    refusal from an engine failure swallowed by the fail-closed except."""
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered; {why}"
    assert calculator.calls == [], f"{query!r} reached the engine; the guard did not refuse it"


# --- codex r8 MEDIUM: a REPEATED panel mention falsely refused -----------------------------
# Task 11 introduced a SECOND vocabulary (`_PANEL_MEMBER_ALIASES`) and taught only the
# RESOLVER about it. The multi-KPI veto masks the first matched span, then rescans with the
# STRICT vocabulary — which does not know "panel nrx" — so a second "NRx panel" had its
# embedded "nrx" read as the canonical WS3-BI-006 and the ask was refused as two metrics.
# Measured pre-fix, all FOUR members (the review named three):
#     "...NRx panel ... the NRx panel?"        -> (012, 006) REFUSED
#     "...NBRx panel ... the NBRx panel?"      -> (013, 007) REFUSED
#     "...TRx share panel ... the TRx share panel?" -> (014, 008) REFUSED
#     "...TRx panel ... the TRx panel?"        -> (011, 005) REFUSED
#
# THE TRAP: the broken case and a LEGITIMATE two-KPI refusal are INDISTINGUISHABLE by the
# scanner's output — both return (012, 006):
#     "What is NRx panel for Kisqali, the NRx panel?"  must ANSWER
#     "What is NRx panel and NRx for Kisqali?"         must REFUSE
# The distinguishing fact is POSITIONAL: whether that "nrx" lies INSIDE an occurrence owned
# by the resolved KPI. So both classes are pinned here; a fix that only makes the first class
# pass is half a fix, and one keyed on "the second hit is a canonical embedded in a panel
# phrase" would break the legitimate veto.


@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("What is NRx panel for Kisqali, the NRx panel?", "WS3-BI-012"),
        ("What is NBRx panel for Kisqali, the NBRx panel?", "WS3-BI-013"),
        ("What is TRx share panel for Kisqali, the TRx share panel?", "WS3-BI-014"),
        ("What is TRx panel for Kisqali, the TRx panel?", "WS3-BI-011"),
        # The SAME KPI named by TWO different aliases — the residual case the
        # review's ownership prototype could not fix, because "panel nrx" was
        # absent from the veto's vocabulary and masking cannot mask what it
        # cannot see.
        ("What is panel NRx and NRx panel for Kisqali?", "WS3-BI-012"),
    ],
)
def test_a_repeated_panel_mention_still_answers(query, expected_id, calculator):
    """A repeated mention of the SAME KPI binds — the established #1475 contract
    (test_explainer_evidence_binding_1475.py pins it for the canonical KPIs)."""
    evidence = _kpi_lookup_evidence({"query": query})
    assert calculator.calls == [expected_id], (query, calculator.calls)
    assert evidence, f"{query!r} produced no evidence — it did not answer"


def test_the_canonical_repeat_control_still_answers(calculator):
    """POSITIVE CONTROL on the pre-existing behaviour this must not disturb: the
    canonical repeat already bound before Task 11 and must still bind."""
    assert _kpi_lookup_evidence({"query": "What is TRx for Kisqali, the TRx?"}) is not None
    assert calculator.calls == ["WS3-BI-005"]


@pytest.mark.parametrize(
    "query",
    [
        "What is TRx and NRx for Kisqali?",
        "What is NRx panel and NRx for Kisqali?",
        "What is TRx and TRx panel for Kisqali?",
        "What is TRx panel and NRx panel for Kisqali?",
        "What is TRx share and TRx share panel for Kisqali?",
    ],
)
def test_two_genuinely_distinct_metrics_still_refuse(query, calculator):
    """The other half of the boundary. These name TWO different KPIs; one value
    presented as the whole answer is a wrong answer, so the veto must still fire
    and must fire BEFORE the engine is consulted."""
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered"
    assert calculator.calls == [], f"{query!r} reached the engine"


# --- codex r9: 11a's masking FAILS OPEN in two ways -----------------------------------------
# 11a decided everything from the FIRST mention and then erased ALL mentions. Masking is
# DESTRUCTIVE, and it destroyed the evidence the later checks depend on. Both defects are the
# same shape one level down — a PER-OCCURRENCE fact gone by the time it is needed:
#
#   M1  WHO OWNS this occurrence.   "trx share" (008) was masked out of the MIDDLE of
#       "trx share panel" (014), leaving an orphan "panel" the scanner cannot recognise, so a
#       genuine two-KPI ask ANSWERED with one number.
#         masked: 'what is the                      and           panel?'
#   M2  WHAT GOVERNS this occurrence. The #1475 head checks run on the FIRST match only; every
#       later mention was then masked, so a later "cost of" / "drivers" context was gone before
#       anything could see it.
#
# WORSE IN KIND than the false refusal 11a fixed: that failed CLOSED (a refusal the user did
# not deserve); these fail OPEN — a plausible value returned for a question never asked.


@pytest.mark.parametrize(
    "query",
    [
        # M1: the resolved KPI's phrase sits INSIDE a longer phrase owned by another KPI
        "What is the share of Kisqali TRx and TRx share panel?",
        "What is the share of Kisqali TRx and the TRx share panel please?",
    ],
)
def test_masking_must_not_steal_an_occurrence_owned_by_another_kpi(query, calculator):
    """Two genuinely distinct KPIs (008 and 014). Ownership of an occurrence belongs to the
    LONGEST vocabulary phrase covering it, not to whichever KPI resolved first."""
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered — veto lost"
    assert calculator.calls == [], f"{query!r} reached the engine"


@pytest.mark.parametrize(
    "query",
    [
        # M2: a LATER mention of the SAME KPI sits in a disqualifying context
        "What is NRx panel and the cost of NRx panel?",
        "What is NRx panel and NRx panel drivers?",
    ],
)
def test_a_later_mention_in_a_guarded_context_is_not_a_redundant_repeat(query, calculator):
    """A repeated mention binds only when the repeat is genuinely redundant. "cost of X" and
    "X drivers" are not: the ask contains a sub-ask a bare value does not answer, so the
    #1475 guards must see EVERY owned occurrence, not just the first."""
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered"
    assert calculator.calls == [], f"{query!r} reached the engine"


# --- asking what property ALL the rows above share, and breaking it ------------------------
# The r9 lesson was that a 12-case matrix looked exhaustive while every row shared a hidden
# structural property: single-owner phrases, and repeats in the SAME governing context. The
# rows above fix those two, but they acquired properties of their own — in every cross-owner
# case the longer phrase came SECOND, in every guarded case the guard came on the LATER
# mention, and every case had exactly TWO owned mentions. These break all three. They passed
# first time; they are here so the next change cannot quietly reintroduce a positional or
# arity assumption.


@pytest.mark.parametrize(
    "query,why",
    [
        (
            "What is TRx share panel and the share of Kisqali TRx?",
            "cross-owner, LONGER phrase FIRST — ownership must not depend on order",
        ),
        (
            "What is the cost of NRx panel and NRx panel?",
            "the guard is on the FIRST mention, a clean repeat follows",
        ),
        (
            "What is NRx panel and NRx panel and the cost of NRx panel?",
            "THREE owned mentions, the third guarded — arity must not matter",
        ),
        (
            "What is panel NRx and the share of Kisqali TRx?",
            "cross-owner where neither phrase is the resolved KPI's longest",
        ),
        (
            "What is NRx panel and the cost of TRx share panel?",
            "the guarded later mention belongs to a DIFFERENT owner",
        ),
    ],
)
def test_the_boundary_holds_regardless_of_position_arity_or_owner(query, why, calculator):
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered; {why}"
    assert calculator.calls == [], f"{query!r} reached the engine; {why}"


def test_three_clean_owned_mentions_still_answer(calculator):
    """The ANSWER side of the arity probe: redundancy is redundancy however often
    it repeats, so long as no occurrence carries a disqualifying context."""
    query = "What is NRx panel and NRx panel and the NRx panel?"
    assert _kpi_lookup_evidence({"query": query}) is not None, query
    assert calculator.calls == ["WS3-BI-012"]


# --- r9 MEDIUM-1, the CAUSAL consumer: "Add regressions through both consumers" -------------
# 11b remediated `_kpi_lookup_evidence` and wired `owned_mask` into `_causal_path_evidence`,
# but nothing exercised the causal site. These close that.
#
# The seam is the REAL one: `_causal_path_evidence` does
# `from src.repositories import causal_path as causal_path_repo` INSIDE the function and
# calls `causal_path_repo.search_paths_for_outcome_sync(kpi.name, ...)`. Patching that
# attribute records WHICH KPI was bound as the outcome. The masking and the directional
# selection are NOT re-implemented here — a hand-rolled replica would be a probe faithful to
# the wrong layer, which has already cost this lane one round.
#
# Asserting "evidence is not None" would be a test named for a behaviour it never exercises:
# the finding is about WHICH outcome gets bound, so the assertion is the id.

_PATH_ROW = {
    "start_node": "HCP Visits",
    "end_node": "TRx",
    "brand": "Kisqali",
    "confidence_level": 0.9,
    "path_id": "p1",
    "is_synthetic": False,
}


@pytest.fixture
def causal_registry(monkeypatch) -> List[str]:
    """Record the outcome term handed to the causal-path registry."""
    bound: List[str] = []

    def _fake(outcome_term, **kwargs):
        bound.append(outcome_term)
        return [_PATH_ROW]

    monkeypatch.setattr(
        "src.repositories.causal_path.search_paths_for_outcome_sync",
        _fake,
    )
    return bound


def _causal_kpi_id(query: str):
    from src.agents.orchestrator.nodes.dispatcher import _causal_path_evidence

    evidence = _causal_path_evidence({"query": query})
    return evidence[0]["kpi_id"] if evidence else None


def test_the_directed_causal_outcome_survives_ownership_masking(causal_registry):
    """r9 MEDIUM-1's own causal example. "impact of X on Y" makes Y the outcome, and Y here
    is the panel share (014). On 11a the ownership bug masked "trx share" out of
    "trx share panel", the second mention was lost, and the outcome fell back to the
    TRx-share KPI (008) — a DIFFERENT unit silently substituted before causal selection.
    MEASURED on 11a: 008. On the Task-11 base and on 11b: 014."""
    assert _causal_kpi_id("impact of share of Kisqali TRx on TRx share panel") == "WS3-BI-014"
    assert causal_registry == ["Observed Rx Events - Patient Panel TRx Share (TRx Share Panel)"]


def test_a_cross_owner_causal_ask_without_direction_fails_closed(causal_registry):
    """No directional grammar, two distinct KPIs: a singleton path answer chosen by alias
    order answers neither, so the `else: return None` branch must fire. On 11a this ANSWERED
    with 008 — fail-OPEN on the causal route. MEASURED on 11a: 008; base and 11b: None."""
    assert _causal_kpi_id("what drives the share of Kisqali TRx and TRx share panel") is None
    assert causal_registry == [], "the registry was consulted; the ask should have failed closed"


def test_a_single_panel_causal_ask_binds_the_panel_kpi(causal_registry):
    """POSITIVE CONTROL. Both assertions above are about losing or refusing an outcome; they
    are worth nothing unless a plain panel causal ask still binds the panel KPI."""
    assert _causal_kpi_id("what drives TRx share panel") == "WS3-BI-014"
    assert causal_registry == ["Observed Rx Events - Patient Panel TRx Share (TRx Share Panel)"]


def test_the_mirror_direction_is_a_documented_PIN_not_a_fix(causal_registry):
    """The direction-reversed mirror returns None, and it does so on the Task-11 base, on 11a
    and on 11b alike — so this is PRE-EXISTING behaviour, pinned, NOT something 11b fixed.

    Why it refuses: `recognize_kpi_span` binds "share of kisqali trx" (008) as the FIRST
    match; that mention is neither `of`-headed by a causal word nor right-headed, so
    `causally_headed` is False, and the query-level causal patterns do not match "impact of
    … on …" either — the entry gate declines before any ownership logic runs. Recorded as a
    known gap in the causal ENTRY gate, unrelated to #2114's masking."""
    assert _causal_kpi_id("impact of TRx share panel on the share of Kisqali TRx") is None
    assert causal_registry == []


# --- codex r10 MEDIUM: the causal path masks later UNSUPPORTED governing heads -------------
# M2's defect, on the causal route. 11b closed it for `_kpi_lookup_evidence` via
# `masked_or_refusal` (head-checks every owned occurrence); `_causal_path_evidence` still
# called `owned_mask`, which does no head checks at all — so a later "cost of NRx panel"
# was masked away as a redundant repeat and the registry was asked for WS3-BI-012.
#
# codex called this pre-existing. A three-way measurement says otherwise:
#
#   "what drives NRx panel and the cost of NRx panel?"
#       33e85a13e (Task 11 base)  None, 0 calls      <- refused
#       1a6ee9956 (11a)           012, 1 call        <- 11a INTRODUCED it
#       61a03b3ad (HEAD)          012, 1 call        <- 11b did not close the causal half
#
# In-range and lane-caused. The base column also refuses "NRx panel and NRx panel", which is
# the over-refusal 11a legitimately fixed — so the whole story is: base over-refuses the
# family, 11a fixes the false refusals AND opens the guarded ones, 11b closes the value half.
#
# The head set DIFFERS by path and the value guard must NOT be reused: causal ACCEPTS a
# causal right-head ("NRx panel drivers" is the ask), and refuses only a governing of-head
# outside _CAUSAL_OF_HEADS ("cost of").


@pytest.mark.parametrize(
    "query",
    [
        "what drives NRx panel and the cost of NRx panel?",
        "what drives NRx panel drivers and the cost of NRx panel?",
        "what drives the cost of NRx panel?",
    ],
)
def test_a_causal_ask_with_an_unsupported_head_on_any_occurrence_fails_closed(
    query, causal_registry
):
    """`cost of X` names a head the registry does not model. Refusing only when it lands on
    the FIRST mention leaves the later one to be masked away as a redundant repeat."""
    assert _causal_kpi_id(query) is None, f"{query!r} bound an outcome"
    assert causal_registry == [], f"{query!r} consulted the registry; that is not a refusal"


@pytest.mark.parametrize(
    "query",
    [
        "what drives NRx panel and NRx panel?",
        "what drives NRx panel and NRx panel drivers?",
        "what drives NRx panel?",
    ],
)
def test_a_causal_repeat_and_a_causal_right_head_still_bind(query, causal_registry):
    """The other half of the boundary, and the reason the VALUE guard cannot be reused here:
    `drivers` is a causal right-head and IS the ask on this path, while the value path refuses
    it. A repeated driver mention stays redundant."""
    assert _causal_kpi_id(query) == "WS3-BI-012", query
    assert causal_registry == ["Observed Rx Events - Patient Panel NRx (NRx Panel)"]


# --- what property do ALL the causal rows share? ------------------------------------------
# Third time this genus has hidden a live defect in this lane. The 11c causal rows were every
# one of them cross-owner-or-single: none was a same-KPI repeat with a guarded LATER
# occurrence, which is exactly why r10's defect survived 11c. The rows above fix that, then
# acquire properties of their own — the guard always LATER, always exactly TWO occurrences,
# always the head "cost", always NRx panel. These break all four.
#
# MEASURED against 11c (61a03b3ad), because "passed first time" says nothing about the base:
# FOUR of the five are FIXES, not pins —
#     three occurrences, third guarded            RED on 11c
#     a different of-head ("accuracy of")         RED on 11c
#     a different panel KPI (TRx share panel)     RED on 11c
#     a CANONICAL KPI (TRx), no panel alias       RED on 11c   <- not panel-specific at all
# and exactly ONE is a genuine pin:
#     the guard on the FIRST occurrence           green on 11c (the pre-existing
#                                                 first-mention check already caught it)
# The canonical-KPI row is the one worth noticing: this was never a panel-vocabulary defect,
# it was a per-occurrence head defect that the panel aliases merely made reachable.


@pytest.mark.parametrize(
    "query,why",
    [
        ("what drives the cost of NRx panel and NRx panel?", "guard on the FIRST occurrence"),
        (
            "what drives NRx panel and NRx panel and the cost of NRx panel?",
            "THREE occurrences, the third guarded — arity must not matter",
        ),
        (
            "what drives NRx panel and the accuracy of NRx panel?",
            "a different unsupported of-head than 'cost'",
        ),
        (
            "what drives TRx share panel and the cost of TRx share panel?",
            "a different panel KPI",
        ),
        ("what drives TRx and the cost of TRx?", "a CANONICAL KPI, no panel alias involved"),
        (
            "what drives NRx and the accuracy of NRx?",
            "CANONICAL + a second of-head — discriminates provenance the same way",
        ),
    ],
)
def test_the_causal_boundary_holds_regardless_of_position_arity_head_or_kpi(
    query, why, causal_registry
):
    """READ THE PROVENANCE NOTE BELOW BEFORE TREATING THESE AS LANE REGRESSIONS.

    The two CANONICAL rows ("what drives TRx and the cost of TRx?", "what drives NRx and
    the accuracy of NRx?") pin a **PRE-EXISTING** defect — broken at the lane base
    `f73476bfc`, i.e. live on main before #2114 touched anything, and still live there.
    They are NOT lane regressions. The panel rows are.
    """
    assert _causal_kpi_id(query) is None, f"{query!r} bound an outcome; {why}"
    assert causal_registry == [], f"{query!r} consulted the registry; {why}"


# --- PROVENANCE CORRECTION (appended; nothing above is rewritten) --------------------------
# `eff8c5825`'s commit body says of codex r10:
#     "CODEX CALLED THIS PRE-EXISTING ("This predates the range"). IT IS NOT."
# THAT IS HALF WRONG, and the half it gets wrong is the one that matters for scope.
#
# Measured through the real `_causal_path_evidence` at FIVE commits, `src.__file__`
# asserted, sources swapped with `git show` and restored by `cp` with `sha256sum -c`:
#
#                                            f73476bfc  33e85a13e  1a6ee9956  61a03b3ad  eff8c5825
#                                            LANE BASE  Task 11    11a        11c        11d
#   "what drives TRx and the cost of TRx?"     005 ***    005 ***    005 ***    005 ***    None OK
#   "what drives NRx and the accuracy of NRx?" 006 ***    006 ***    006 ***    006 ***    None OK
#   "what drives NRx panel and the cost of
#                          NRx panel?"         None       None       012 ***    012 ***    None OK
#
# TWO HALVES, DIFFERENT PROVENANCE:
#   * CANONICAL half — PRE-EXISTING. Broken at the lane base, before this lane existed;
#     LIVE ON MAIN TODAY. codex's "predates the range" was RIGHT about this half.
#   * PANEL half — LANE-INTRODUCED. Refused at base and at Task 11, opened by 11a.
#     codex was wrong about this half.
#
# HOW BOTH OF US GOT IT WRONG, which is the reusable part: codex generalised from ONE case
# to "predates the range"; the dispatcher generalised from ONE case to "lane-introduced".
# Same error, opposite conclusions. PROVENANCE NEEDS THE CASE THAT DISCRIMINATES — here the
# CANONICAL form, which contains no panel vocabulary at all — AND a commit old enough to
# predate the suspected cause. Two commits inside the lane could not have shown this; the
# lane base could.
#
# The canonical fix STAYS. Reverting it would leave an identical fail-open live in the same
# function while shipping the panel fix — strictly worse. That 11d also closes a pre-existing
# production fail-open OUTSIDE #2114's stated scope (same class as #2131) is a scope question
# for the owner, and belongs in the PR body rather than being absorbed silently here.


def test_three_clean_causal_occurrences_still_bind(causal_registry):
    """The ANSWER side of the arity probe on the causal path."""
    assert _causal_kpi_id("what drives NRx panel and NRx panel and NRx panel?") == "WS3-BI-012"
    assert causal_registry == ["Observed Rx Events - Patient Panel NRx (NRx Panel)"]


# --- codex r11 MEDIUM: unsupported RIGHT-heads bypass causal refusal -----------------------
# The causal path checks governing OF-heads on every owned occurrence and NEVER checks
# right-heads at all. "what drives NRx panel cost?" has no of-head, so nothing refuses it and
# the registry is asked for the NRx-panel outcome despite the unsupported "cost" sub-ask.
#
# TWO CORRECTIONS to the review, both measured, both making it bigger and older:
#  1. NOT a per-occurrence gap. A SINGLE mention fails open too ("what drives NRx panel
#     cost?" -> 012, 1 call). There is no check to extend on every occurrence; there is a
#     check to ADD. codex's "on every occurrence" framing would have missed the first one.
#  2. PRE-EXISTING against ACTUAL origin/main (bd3c7fb3d), not merely the lane base:
#         "what drives TRx cost?"        main 005, 1 call   HEAD 005, 1 call
#         "what drives NRx panel cost?"  main None, 0 calls  HEAD 012, 1 call
#     The lane does not cause it; the lane NEWLY EXPOSES it for the four panel KPIs, which is
#     what made it in-scope. Fixed under OWNER DECISION #10.
#
# ⚠ THE DESIGN HAZARD. The of-head check is a fail-closed ALLOWLIST and that shape CANNOT be
# copied here: `_kpi_right_head` returns the next token whatever it is, so an allowlist of
# causal heads alone would refuse every ordinary "for Kisqali" / "in Q3" ask. The
# discriminator is STRUCTURAL and was established by enumeration, not guessed —
#   legitimate right-heads are CLOSED-CLASS function words (for, in, by, across, among, at,
#   with, from, and, or, than, this, last, next, the, when, where, ...), causal heads, a
#   period/number token, or end-of-string;
#   the defect's right-heads are OPEN-CLASS nouns naming another quantity (cost, accuracy,
#   price, forecast, target, volume, trend, uplift, benchmark).
# A denylist of nouns would be a labeling fix — nouns are an OPEN class and cannot be
# enumerated. Function words are a CLOSED class and can be. That asymmetry is the whole
# justification for the shape of this fix.


@pytest.mark.parametrize(
    "query",
    [
        "what drives NRx panel cost?",  # SINGLE mention — the review's framing missed this
        "what drives NRx panel and NRx panel cost?",
        "what drives TRx cost?",  # canonical — pre-existing on main
        "what drives NRx panel accuracy?",
        "what drives TRx price?",
        "what drives NRx panel forecast?",
        "what drives TRx target?",
    ],
)
def test_an_unsupported_right_head_compound_fails_closed_on_the_causal_path(query, causal_registry):
    """ "NRx panel cost" names a quantity the registry does not model. Binding NRx panel's
    drivers answers a different question, so it must refuse BEFORE the repository call."""
    assert _causal_kpi_id(query) is None, f"{query!r} bound an outcome"
    assert causal_registry == [], f"{query!r} consulted the registry; that is not a refusal"


@pytest.mark.parametrize(
    "query,expected_id",
    [
        # prepositions and scope — the class this fix most risks over-refusing
        ("what drives NRx panel for Kisqali?", "WS3-BI-012"),
        ("what drives NRx panel in the west region?", "WS3-BI-012"),
        ("what drives TRx by severity?", "WS3-BI-005"),
        ("what drives NRx panel across brands?", "WS3-BI-012"),
        ("what drives TRx among new patients?", "WS3-BI-005"),
        ("what drives NRx panel at the HCP level?", "WS3-BI-012"),
        ("what drives TRx with high adherence?", "WS3-BI-005"),
        ("what drives NRx panel from Q1?", "WS3-BI-012"),
        ("what drives TRx to date?", "WS3-BI-005"),
        ("what drives NRx panel on the panel?", "WS3-BI-012"),
        ("what drives TRx per brand?", "WS3-BI-005"),
        ("what drives NRx panel within the cohort?", "WS3-BI-012"),
        ("what drives TRx under the new plan?", "WS3-BI-005"),
        ("what drives NRx panel between Q1 and Q2?", "WS3-BI-012"),
        # temporal
        ("what drives NRx panel in Q3?", "WS3-BI-012"),
        ("what drives TRx last quarter?", "WS3-BI-005"),
        ("what drives NRx panel this year?", "WS3-BI-012"),
        ("what drives TRx over time?", "WS3-BI-005"),
        ("what drives NRx panel during the launch?", "WS3-BI-012"),
        ("what drives TRx since January?", "WS3-BI-005"),
        ("what drives NRx panel after launch?", "WS3-BI-012"),
        ("what drives TRx before launch?", "WS3-BI-005"),
        ("what drives NRx panel next quarter?", "WS3-BI-012"),
        ("what drives TRx recently?", "WS3-BI-005"),
        # comparison / coordination / subordination
        # NOTE the tail: "versus NRx" would name a SECOND KPI and is correctly refused by
        # the two-metric veto — my first draft asserted it should bind, and red-first caught
        # the bad EXPECTATION rather than a defect. The right-head under test is "versus".
        ("what drives NRx panel versus last quarter?", "WS3-BI-012"),
        ("what drives NRx panel vs the prior period?", "WS3-BI-012"),
        ("what drives NRx panel when adherence is low?", "WS3-BI-012"),
        ("what drives NRx panel where coverage is high?", "WS3-BI-012"),
        ("what drives NRx panel if adherence drops?", "WS3-BI-012"),
        # causal right-heads — accepted on THIS path, refused on the value path
        ("what drives NRx panel drivers?", "WS3-BI-012"),
        ("what drives TRx determinants?", "WS3-BI-005"),
        ("what drives NRx panel predictors?", "WS3-BI-012"),
        # end of string
        ("what drives NRx panel?", "WS3-BI-012"),
        ("what drives TRx", "WS3-BI-005"),
    ],
)
def test_ordinary_right_heads_must_keep_binding(query, expected_id, causal_registry):
    """THE OVER-REFUSAL BATTERY, and it is the point of the task. A fix that refuses
    "what drives NRx panel for Kisqali?" is worse than the defect it closes — the same
    mirror failure as 11a's "mask more", which destroyed five legitimate refusals."""
    assert _causal_kpi_id(query) == expected_id, query
    assert causal_registry, f"{query!r} never reached the registry"


# --- what property do ALL the r11 rows share? ----------------------------------------------
# codex found r11 by asking this of OUR fixtures ("coincidentally restricted to unsupported
# of-heads"). Asking it of the new rows: every unsupported one is a SINGLE-WORD noun, and
# every accepted one is a bare function word. These break both. All seven passed first time,
# and the teeth run confirms they are PINS not fixes — the 7 reds on 9bd77796c were the
# `..._fails_closed_on_the_causal_path` rows only.
#
# --- CORRECTION appended 2026-09-16; the paragraph above stands as written -----------------
# "the teeth run confirms they are PINS not fixes" is FALSE, and the reason is worth keeping:
# THE TEETH RUN PREDATED THESE ROWS. It collected 86 tests; this file collects 93. The seven
# property-breakers below were written AFTER that run and were never in it, so it could not
# confirm anything about them. A measurement cited for rows it never collected is a number
# attached to the wrong tree — the same defect as "I ran the gates, then edited a file", in
# the other artifact. Arithmetic was the tell, not a re-run: the commit body's "7 failed /
# 79 passed" sums to 86 standing beside a reported panel-consumer count of 93.
#
# RE-MEASURED on 9bd77796c — `git show` swap of `kpi_mentions.py` alone, positive control that
# the swap was observed (both 11e symbols absent on base, present again after restore), `cp`
# restore with `sha256sum -c` OK:
#
#     11 failed / 82 passed
#      7  test_an_unsupported_right_head_compound_fails_closed_on_the_causal_path  (every row)
#      4  test_unsupported_right_heads_refuse_in_any_surface_form                  (every row)
#      0  test_period_and_causal_right_heads_still_bind, and the period-limit test
#
# So the four `..._in_any_surface_form` rows are FIXES, not pins: the base binds WS3-BI-012 on
# every one of them, and this fix closes those surface forms too. Only the three binding rows
# — the two `..._still_bind` rows and the period-limit test — are true pins. The teeth are
# STRONGER than the commit body claimed, not weaker. The defect is in the reporting.


@pytest.mark.parametrize(
    "query,why",
    [
        ("what drives NRx panel unit cost?", "multi-word compound, not a single noun"),
        ("what drives NRx panel's cost?", "possessive form"),
        ("what drives NRx panel run rate?", "a metric noun that reads temporal-ish"),
        ("what drives NRx panel cost-per-script?", "hyphenated compound"),
    ],
)
def test_unsupported_right_heads_refuse_in_any_surface_form(query, why, causal_registry):
    assert _causal_kpi_id(query) is None, f"{query!r} bound an outcome; {why}"
    assert causal_registry == [], f"{query!r} consulted the registry; {why}"


@pytest.mark.parametrize(
    "query,expected_id,why",
    [
        ("what drives NRx panel 2026?", "WS3-BI-012", "a bare number is scope, not a quantity"),
        ("what drives NRx panel causes?", "WS3-BI-012", "plural causal head"),
    ],
)
def test_period_and_causal_right_heads_still_bind(query, expected_id, why, causal_registry):
    assert _causal_kpi_id(query) == expected_id, f"{query!r}; {why}"
    assert causal_registry, f"{query!r} never reached the registry; {why}"


def test_a_period_right_head_no_longer_licenses_the_noun_behind_it(causal_registry):
    """11f REVERSES the limit 11e pinned here one commit earlier, and the reversal is the
    point of the test.

    11e pinned "what drives NRx panel q3 performance?" as BINDING, on the reasoning that
    refusing it would also refuse "what drives NRx panel q3?" and that over-refusal was the
    worse failure. Both halves were wrong. The second is wrong on this lane's own severity
    ordering (fail-open outranks fail-closed). The FIRST is wrong on the facts: the two cases
    are separable, because a period token can be CONSUMED and the decision deferred to what
    follows it — "q3" then end-of-string binds, "q3" then an open-class noun refuses.

    And the limit was never one row. I pinned the mildest instance available — "performance"
    is a vague noun — so the pin never tested a sharp one. Measured on 1ef6cdf3c, the same
    branch bound "month cost", "quarter forecast", "year target" and "2026 revenue": the
    defect's OWN nouns, reachable through any period token. The pin was concealing a hole of
    the same class as the defect the commit it shipped in was closing."""
    assert _causal_kpi_id("what drives NRx panel q3 performance?") is None
    assert causal_registry == []


# --- 11f: the period branch must LOOP, not accept ------------------------------------------
# 11e accepted a period right-head unconditionally, whatever stood behind it. The fix walks
# the tail instead: consume period tokens, and consume a determiner ONLY when a period token
# follows it, then decide on the first token that is neither.
#
# ⚠ THE WALK MUST NOT CONSUME PREPOSITIONS, and that is the whole difference between this fix
# and an over-refusing one. A preposition OPENS A SCOPE PHRASE whose object is an ordinary
# noun — "for Kisqali", "in the west region", "by severity". Consuming it and then testing its
# object would refuse the entire battery below. A determiner does not open a phrase; it
# modifies the period noun ("last quarter", "this year"), so the compound head is still ahead.


@pytest.mark.parametrize(
    "query",
    [
        # a SHARP noun behind one period token — the defect's own vocabulary
        "what drives NRx panel month cost?",
        "what drives TRx quarter forecast?",
        "what drives NRx panel year target?",
        "what drives NRx panel 2026 revenue?",
        "what drives TRx q3 accuracy?",
        # CHAINED period tokens — a single-step recursion would accept these
        "what drives NRx panel q3 2026 cost?",
        "what drives TRx h1 2026 forecast?",
        # a determiner AHEAD of the period token — the other chained shape
        "what drives NRx panel last quarter cost?",
        "what drives TRx this year forecast?",
        "what drives NRx panel next quarter target?",
        "what drives NRx panel the q3 uplift?",
    ],
)
def test_a_noun_behind_a_period_chain_still_fails_closed(query, causal_registry):
    """The period token defers the decision; it does not license what stands behind it."""
    assert _causal_kpi_id(query) is None, f"{query!r} bound an outcome"
    assert causal_registry == [], f"{query!r} consulted the registry; that is not a refusal"


@pytest.mark.parametrize(
    "query,expected_id,why",
    [
        ("what drives NRx panel q3?", "WS3-BI-012", "period then end-of-string"),
        ("what drives NRx panel q3 2026?", "WS3-BI-012", "period chain then end-of-string"),
        ("what drives TRx h1 2026?", "WS3-BI-005", "period chain then end-of-string"),
        ("what drives NRx panel q3 and q4?", "WS3-BI-012", "period then a conjunction"),
        ("what drives NRx panel q3 drivers?", "WS3-BI-012", "period then a causal head"),
        ("what drives NRx panel month over month?", "WS3-BI-012", "period then a preposition"),
        ("what drives NRx panel 2026 vs 2025?", "WS3-BI-012", "period then a comparison"),
        ("what drives NRx panel last quarter?", "WS3-BI-012", "determiner then period, then EOS"),
        ("what drives TRx this year?", "WS3-BI-005", "determiner then period, then EOS"),
        ("what drives NRx panel in Q3 2026?", "WS3-BI-012", "preposition — never reaches the walk"),
        # "what drives NRx panel this brand?" was pinned HERE as binding in 11f, on the
        # reasoning that a determiner not before a period token ends the walk. r12 showed
        # that rule also bound "last two quarters cost". The row now lives — reversed, as a
        # refusal — in test_a_determiner_does_not_license_the_noun_behind_it_on_the_causal_path.
    ],
)
def test_the_period_walk_does_not_over_refuse(query, expected_id, why, causal_registry):
    """THE OVER-REFUSAL GUARD FOR THE WALK ITSELF. Every row here binds on 1ef6cdf3c too, so
    it constrains the fix rather than describing it. The last two are the ones that fail if
    the walk consumes function words indiscriminately."""
    assert _causal_kpi_id(query) == expected_id, f"{query!r}; {why}"
    assert causal_registry, f"{query!r} never reached the registry; {why}"


def test_every_period_modifier_is_already_a_binding_token():
    """Every period modifier is also a function word — pinned, not left to inspection.

    ⚠ TITLE CORRECTED (r12). This was called "THE WALK MAY ONLY EVER REFUSE MORE, NEVER BIND
    MORE", which is not what it proves and is no longer true of the walk at all: `_scope_span`
    BINDS tails that used to refuse ("TRx Kisqali" refused under 11e, binds now, by design).
    What the subset actually buys is local and worth keeping: stepping over a determiner into
    an end-of-string cannot bind a tail the single-token rule would have refused, because the
    determiner was already a binding token on its own. It proves nothing about the grammar
    being complete, and nothing about the helper as a whole. A test named for a property it
    does not test is the family member this file has caught twice before.

    CORRECTION appended to 6a5eab03d, whose body says "RED-FIRST: 12 failed / 104 passed".
    12 + 104 = 116; this file collects 117. That run predated THIS test, so it is the same
    defect 1ef6cdf3c corrected one commit earlier — a measurement bound to the tree it ran
    against, cited for a later one — caught the same way, by arithmetic across two numbers
    rather than by a re-run. Re-measured on 1ef6cdf3c (`cp` + `sha256sum -c`, positive
    control: `_PERIOD_MODIFIERS` and `_tail_changes_the_quantity` absent on base, the 11e
    helper `_right_head_changes_the_quantity` present):

        13 failed / 104 passed
         1  test_a_period_right_head_no_longer_licenses_the_noun_behind_it  (the inverted pin)
        11  test_a_noun_behind_a_period_chain_still_fails_closed            (every row)
         1  test_every_period_modifier_is_already_a_binding_token           (this test)

    Said precisely, because it matters: THIS row's red is not a behavioural red. It fails on
    base because `_PERIOD_MODIFIERS` does not exist there — it is a structural pin of the new
    design, so it could not have been red for the reason the other twelve are. The twelve are
    the behavioural teeth.

    The single-token rule bound on any function word. The walk steps over a modifier when a
    period token follows, and could therefore run off the end and BIND something that used to
    refuse — but only if a modifier were not already a binding token. This subset relation is
    what forecloses that, so it is an invariant of the design and not a tidy coincidence."""
    from src.agents.orchestrator.nodes.kpi_mentions import (
        _PERIOD_MODIFIERS,
        _RIGHT_HEAD_FUNCTION_WORDS,
    )

    assert _PERIOD_MODIFIERS <= _RIGHT_HEAD_FUNCTION_WORDS, sorted(
        _PERIOD_MODIFIERS - _RIGHT_HEAD_FUNCTION_WORDS
    )


def test_a_preposition_behind_a_period_token_still_binds(causal_registry):
    """A MEASURED RESIDUAL LIMIT, pinned sharply this time rather than by its mildest case.

    "time" is a period token, so "what drives NRx panel time to fill?" consumes it, meets the
    preposition "to" and BINDS — although "time to fill" is a metric name, not a scope. The
    walk stops at prepositions by design (see the battery), so closing this would mean
    deciding that a preposition behind a period token reads differently from one directly
    after the mention, and that would also refuse "what drives TRx time to date?".

    Stated as what it is: a narrower hole of the same class, left open deliberately, with the
    sharp case named rather than a vague one. Unlike 11e's pin, this one is NOT load-bearing
    for any battery row — it can be closed later without reversing anything here."""
    assert _causal_kpi_id("what drives NRx panel time to fill?") == "WS3-BI-012"
    assert causal_registry


# --- 11g: the VALUE path's right-head gap (#2139), and an 11e regression it exposed ---------
# ENUMERATED FIRST, through the real `_kpi_lookup_evidence`, before a line was designed.
#
# ⚠ THE CAUSAL RULE CANNOT BE TRANSPLANTED HERE, and the enumeration is what proved it. On the
# causal path a bare open-class noun after the mention names ANOTHER QUANTITY. On the value
# path it is routinely SCOPE — "What is TRx Kisqali?", "What is NRx panel west region?" both
# bind today and must keep binding. A closed-class allowlist would refuse every one of them.
#
# ⚠ AND "forecast" REFUSING IS NOT EVIDENCE OF A HEAD GUARD. `KPI_VALUE_LOOKUP_PATTERN`
# (intent_classifier.py:536) opens with a whole-query negative lookahead —
#     (?s)\A(?!.*(?:predict|expect|forecast|project|likelihood|probabilit|what will))
# — so those queries die at the ENTRY GATE and never reach `masked_or_refusal`. Reading that
# refusal as a working right-head rule would be a check that cannot fail for the reason you
# care about. For the same reason NO refusal test below may rely on the entry gate: every one
# uses a lead-in the gate admits ("what is", "show me"), so the refusal is THIS fix's doing.
#
# THE DISCRIMINATOR IS "DOES THE PLATFORM RESOLVE THIS INTO SCOPE AT ALL", and the measurement
# that makes refusing the unresolved ones safe is what reaches the calculator:
#
#     What is NRx panel for Kisqali?        -> context {'brand': 'Kisqali'}
#     What is NRx panel in the west region? -> context {'region': 'west'}
#     What is NRx panel oncology?           -> context {}      <- qualifier SILENTLY DROPPED
#     What is TRx patients?                 -> context {}      <- same
#     What is NRx panel cost?               -> context {}      <- same
#
# "oncology" and "patients" are NOT legitimate answers being protected: the platform ignores
# the qualifier and returns the national figure, exactly as it does for "cost". Refusing them
# closes a second fail-open rather than costing an answer. Scope membership comes from the
# resolver the value path itself consults (`brand_from_text` / `region_from_text` /
# `PATIENT_AXES`), never a copied word list, so it cannot drift from the registry.


@pytest.mark.parametrize(
    "query",
    [
        # open-class QUANTITY nouns — the #2139 defect proper
        "What is NRx panel cost?",
        "What is TRx cost?",
        "What is NRx panel accuracy?",
        "What is TRx price?",
        "What is NRx panel target?",
        "What is TRx trend?",
        "What is NRx panel uplift?",
        "What is TRx benchmark?",
        "What is NRx panel variance?",
        "What is TRx performance?",
        # the 11f period class, on this path too
        "What is NRx panel month cost?",
        "What is TRx q3 performance?",
        "What is NRx panel last quarter cost?",
        # surface forms of the compound
        "What is NRx panel's cost?",
        "What is NRx panel unit cost?",
        "What is NRx panel cost-per-script?",
        # a resolvable scope token followed by a quantity — scope does not license the noun
        "What is TRx Kisqali cost?",
        "What is NRx panel west cost?",
        # an alternative admitted lead-in, so the gate is not doing the work
        "show me the NRx panel cost",
    ],
)
def test_an_unsupported_right_head_fails_closed_on_the_value_path(query, calculator):
    """The calculator must never be consulted: a figure computed for the KPI is not an answer
    to a question about its cost, and returning one is the fail-open #2139 describes."""
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered"
    assert calculator.calls == [], f"{query!r} called the calculator; that is not a refusal"


@pytest.mark.parametrize(
    "query",
    [
        "What is NRx panel oncology?",
        "What is TRx patients?",
        "What is NRx panel HCPs?",
        "What is TRx specialty?",
    ],
)
def test_a_qualifier_the_platform_cannot_resolve_fails_closed(query, calculator):
    """DECLARED SCOPE BEYOND #2139's STATED DEFECT: these are dropped-scope fail-opens, not
    right-head ones. Same shape, adjacent cause. Measured: each reaches the calculator with
    context {} today — the qualifier is discarded and a national figure returned — so they
    are indistinguishable from "cost" to any honest discriminator, and keeping them binding
    would mean whitelisting tokens we have measured to be ignored."""
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered"
    assert calculator.calls == [], f"{query!r} called the calculator"


@pytest.mark.parametrize(
    "query,expected_id,scope,why",
    [
        # ⭐ THE TRANSPLANT'S OWN DISPROOF — pinned so nobody reintroduces the causal rule here
        ("What is TRx Kisqali?", "WS3-BI-005", ("brand",), "bare BRAND is scope on this path"),
        ("What is NRx panel west region?", "WS3-BI-012", ("region",), "bare REGION is scope"),
        ("What is TRx northeast?", "WS3-BI-005", ("region",), "census region, bare"),
        ("What is TRx new england?", "WS3-BI-005", ("region",), "two-token region phrase"),
        # NOTE: no patient-axis row here. An earlier draft asserted "What is NRx panel
        # segment?" binds "as a served PATIENT_AXES axis" — measured FALSE, see
        # test_a_free_text_patient_axis_is_dropped_scope_not_scope below.
        # prepositional scope — ⚠ MOST OF THESE ARRIVE WITH NOTHING BOUND, see the docstring
        ("What is NRx panel for Kisqali?", "WS3-BI-012", ("brand",), "preposition"),
        ("What is TRx by severity?", "WS3-BI-005", (), "preposition, qualifier DROPPED"),
        ("What is NRx panel in the west region?", "WS3-BI-012", ("region",), "preposition"),
        ("What is TRx per brand?", "WS3-BI-005", (), "preposition, qualifier DROPPED"),
        ("What is NRx panel across brands?", "WS3-BI-012", (), "preposition, DROPPED"),
        ("What is TRx with high adherence?", "WS3-BI-005", (), "preposition, DROPPED"),
        ("What is NRx panel within the cohort?", "WS3-BI-012", (), "preposition, DROPPED"),
        ("What is TRx among new patients?", "WS3-BI-005", (), "preposition, DROPPED"),
        ("What is NRx panel at the HCP level?", "WS3-BI-012", (), "preposition, DROPPED"),
        ("What is TRx by segment?", "WS3-BI-005", (), "r12 asked for this row: also DROPPED"),
        # temporal
        ("What is NRx panel in Q3?", "WS3-BI-012", (), "preposition then period, DROPPED"),
        ("What is TRx last quarter?", "WS3-BI-005", ("window",), "determiner then period"),
        ("What is NRx panel this year?", "WS3-BI-012", ("window",), "determiner then period"),
        ("What is TRx since January?", "WS3-BI-005", (), "preposition then month, DROPPED"),
        ("What is NRx panel q3?", "WS3-BI-012", (), "bare period token, DROPPED"),
        ("What is TRx q3 2026?", "WS3-BI-005", ("window",), "period chain"),
        # comparison / end of string
        ("What is NRx panel versus last quarter?", "WS3-BI-012", ("window",), "comparison"),
        ("What is TRx vs the prior period?", "WS3-BI-005", (), "comparison, DROPPED"),
        ("What is TRx?", "WS3-BI-005", (), "end of string, nothing to bind"),
        ("show me the NRx panel", "WS3-BI-012", (), "another admitted lead-in"),
    ],
)
def test_the_value_path_over_refusal_battery(query, expected_id, scope, why, calculator):
    """THE OVER-REFUSAL BATTERY FOR THIS PATH — every row binds on 6d321cbcf as well as
    after, so it guards the fix rather than describing it.

    ⚠ WHAT A BINDING ROW PROVES, STATED EXACTLY: that THE WALK DOES NOT REFUSE IT. It does
    NOT prove the answer is scoped. Until r12 this file's double discarded `context`, so
    these rows could not tell the difference, and ten of them are answering with NOTHING
    BOUND — "by severity", "per brand", "across brands", "within the cohort", "among new
    patients", "at the HCP level", "in Q3", "since January", "q3", "vs the prior period".
    The `scope` column is the MEASURED context, so those rows are now WITNESSES OF #2141
    rather than silent passes: when #2141 is fixed they will fail deliberately and whoever
    fixes it will see exactly which asks change.

    ⚠ AND IT EXPOSES A LIMIT IN THE LANE'S OWN STORY, which belongs here in plain words.
    This lane refuses "What is TRx patients?" on the ground that it reaches the calculator
    with nothing bound — yet "What is TRx among new patients?" binds with nothing bound too.
    Same dropped qualifier, same empty context, opposite verdict; the only difference is a
    preposition. That is DELIBERATE and it is not a discriminator we claim to have: the
    guard closes the BARE dropped-qualifier hole and leaves the PREPOSITIONAL one to #2141,
    because refusing prepositional scope would refuse most legitimately-scoped asks there
    are (owner decision #13). The honest statement of the rule is "a bare noun after the
    mention must resolve", not "an ask must be scoped to answer".
    """
    evidence = _kpi_lookup_evidence({"query": query})
    assert calculator.calls == [expected_id], (query, why, calculator.calls)
    assert evidence, f"{query!r} produced no evidence; {why}"
    assert tuple(sorted(calculator.contexts[0])) == scope, (query, why, calculator.contexts)


@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("what drives TRx Kisqali?", "WS3-BI-005"),
        ("what drives NRx panel Kisqali?", "WS3-BI-012"),
        ("what drives NRx panel west?", "WS3-BI-012"),
        ("what drives TRx northeast?", "WS3-BI-005"),
        # no patient-axis row: free text does not bind an axis on either path
    ],
)
def test_bare_scope_binds_on_the_causal_path_too(query, expected_id, causal_registry):
    """AN 11e REGRESSION, LANE-CAUSED, found by enumerating the OTHER path — and measured at
    three commits with a control proving which implementation was loaded:

        9bd77796c  no right-head check     "what drives TRx Kisqali?" -> 005, 1 call
        1ef6cdf3c  11e single-token        -> None, 0 calls   <- INTRODUCED HERE
        6d321cbcf  11f walk                -> None, 0 calls   <- inherited, not caused

    11e's allowlist admitted only function words and period tokens, so a bare brand or region
    read as an open-class noun and a perfectly ordinary scoped causal ask started refusing.
    My 11e and 11f batteries could not see it: every scoped row in both used a PREPOSITION.
    Same genus as the fixture blindness those commits each found in their predecessor."""
    assert _causal_kpi_id(query) == expected_id, query
    assert causal_registry, f"{query!r} never reached the registry"


def test_scope_does_not_license_a_quantity_behind_it_on_the_causal_path(causal_registry):
    """The scope token is consumed and the decision DEFERRED, exactly as a period token is —
    it does not license whatever follows. Without this the 11e repair would reopen r11."""
    assert _causal_kpi_id("what drives TRx Kisqali cost?") is None
    assert causal_registry == []


# --- what property do ALL the 11g rows share? ----------------------------------------------
# Every binding row above was ONE scope token, sitting DIRECTLY after the mention, resolved by
# the brand-or-region route. Three properties, each breakable. Asking this question before
# declaring done has found a live defect six rounds running — and this round it ran AHEAD of
# the code, disproving the transplant before a line was written.


@pytest.mark.parametrize(
    "query,expected_id,why",
    [
        ("What is TRx urticaria?", "WS3-BI-005", "brand via the INDICATION route, not a name"),
        ("What is TRx Kisqali west?", "WS3-BI-005", "two scope tokens in a row"),
        ("What is NRx panel q3 kisqali?", "WS3-BI-012", "scope AFTER a period token"),
    ],
)
def test_scope_binds_in_shapes_the_new_rows_did_not_cover(query, expected_id, why, calculator):
    assert _kpi_lookup_evidence({"query": query}), f"{query!r} refused; {why}"
    assert calculator.calls == [expected_id], (query, why, calculator.calls)


@pytest.mark.parametrize(
    "query,why",
    [
        ("What is TRx west region cost?", "scope + scope-noun does not license a quantity"),
        ("What is NRx panel kisqali accuracy?", "scope does not license a quantity"),
        ("What is TRx new england cost?", "a two-token region does not license a quantity"),
        ("What is NRx panel level?", "a scope NOUN with no resolved scope before it"),
    ],
)
def test_scope_defers_the_decision_it_does_not_license_what_follows(query, why, calculator):
    """`_SCOPE_NOUNS` is reachable only immediately after a resolver-confirmed token, so it
    cannot open a hole on its own: "level" alone still refuses, as an unresolved qualifier."""
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered; {why}"
    assert calculator.calls == [], f"{query!r} called the calculator; {why}"


@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("What is NRx panel for Kisqali in the west region?", "WS3-BI-012"),
        ("What is TRx for Kisqali, given that access issues ate into field time?", "WS3-BI-005"),
        ("show me the NRx panel for Kisqali across the northeast", "WS3-BI-012"),
    ],
)
def test_a_preposition_ends_the_walk_before_any_scope_lookahead(query, expected_id, calculator):
    """CAUGHT BY GATE B, NOT BY THIS FILE — `test_explainer_evidence_binding_1475.py`'s
    "What drives TRx for Kisqali, given that access issues ate into field time?" went RED.

    `_scope_span`'s two-token window matched "for kisqali" as a unit, so the walk stepped
    PAST the preposition and judged the prose behind it, refusing on "given". Every
    prepositional row in my own battery ended immediately after the brand ("for Kisqali?"),
    so the walk hit end-of-string and bound — the RIGHT answer for the WRONG reason, which is
    precisely what a fixture set blind through its choice of inputs produces. These rows put
    prose behind the brand so the ordering is pinned, not incidental."""
    assert _kpi_lookup_evidence({"query": query}), f"{query!r} refused"
    assert calculator.calls == [expected_id], (query, calculator.calls)


def test_a_preposition_ends_the_walk_on_the_causal_path_too(causal_registry):
    """The #1475 row's own shape, pinned here as well so this file can catch it next time."""
    q = "what drives TRx for Kisqali, given that access issues ate into field time?"
    assert _causal_kpi_id(q) == "WS3-BI-005"
    assert causal_registry


@pytest.mark.parametrize(
    "query",
    [
        "What is NRx panel segment?",
        "What is TRx therapy line?",
        "What is NRx panel biologic?",
        "What is TRx ige tier?",
    ],
)
def test_a_free_text_patient_axis_is_dropped_scope_not_scope(query, calculator):
    """11h, correcting 11g: `f3663f2d6` had `PATIENT_AXES` inside `_scope_span`, so a bare
    axis token bound as though free text could scope by it. IT CANNOT.

    `_extract_brand_region` (dispatcher.py:216-226) asks `brand_from_text` and
    `region_from_text` and NOTHING ELSE, so brand and region are the whole of what query
    text can bind. The four patient axes are a separate channel it never feeds — and
    "region" is that second text channel, NOT one of the four axes, which is why
    "in the west region" arrives as {'region': 'west'} while an axis arrives as nothing.

    Still live in this tree as the positive control, needing no source swap to show it:

        What is NRx panel by segment?  ->  binds, calculator context {}

    It binds at the preposition so the walk never judges it, and the axis is dropped all
    the same. So an admitted axis token is indistinguishable from "oncology" — allowlisting
    it whitelists a token measured to be ignored, which is precisely what 11g refuses
    "oncology" for. I inferred "served axis" meant "bindable from free text" instead of
    measuring it, and built the #2141 defect into the fix for #2139."""
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered"
    assert calculator.calls == [], f"{query!r} called the calculator"


def test_a_free_text_patient_axis_is_dropped_scope_on_the_causal_path_too(causal_registry):
    """The causal half, asserted through its OWN consumer.

    A first draft of the test above carried this row inside the same parametrize and
    asserted `value is None or causal is None`. That `or` CANNOT FAIL for the reason it
    exists: `_causal_kpi_id` returns None for every "What is ..." row, so the value half
    was never witnessed. Same shape as the vacuous assertion deleted in 10f, written by me
    again four tasks later — which is why each path is now asserted through its own."""
    assert _causal_kpi_id("what drives NRx panel segment?") is None
    assert causal_registry == []


# --- r12 HIGH-b: generic scope nouns recreated the axis allowlist 11h removed --------------
# `_SCOPE_NOUNS` was a flat set consumable whenever `after_scope` was true, and `after_scope`
# was never reset. Two defects in one:
#   (i) it chained — "Kisqali tier cohort axis" consumed three unserved dimension nouns;
#  (ii) "cohort"/"tier"/"axis" (and "market"/"area"/"territory") name dimensions NEITHER
#       resolver binds. Probed: brand_from_text and region_from_text return None for every
#       one of them. They are PATIENT_AXES under a new name — exactly what 11h removed one
#       commit earlier, readmitted through a different door. "territory-level detail" is in
#       the capability catalogue's NEVER_BLOCK list, so "west territory" names something no
#       tool serves at all.
# The appositive is now KEYED TO THE DIMENSION THAT ACTUALLY BOUND, taken from which resolver
# returned non-None — never a second word list — and consumable ONCE, immediately.


@pytest.mark.parametrize(
    "query,why",
    [
        ("What is NRx panel Kisqali tier?", "'tier' binds on neither resolver"),
        ("What is NRx panel Kisqali cohort?", "'cohort' binds on neither resolver"),
        ("What is TRx Kisqali tier cohort axis?", "three unserved nouns chained"),
        ("What is NRx panel west region cohort tier?", "chained behind a real region"),
        ("What is NRx panel Kisqali market?", "'market' is not a served dimension"),
        ("What is NRx panel west territory?", "territory detail is in NEVER_BLOCK"),
        ("What is NRx panel Kisqali region?", "BRAND bound; 'region' is the other dimension"),
        ("What is NRx panel west brand?", "REGION bound; 'brand' is the other dimension"),
        ("What is NRx panel west region region?", "the appositive is consumable ONCE"),
    ],
)
def test_a_scope_noun_does_not_chain_or_cross_dimensions(query, why, calculator):
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered; {why}"
    assert calculator.calls == [], f"{query!r} called the calculator; {why}"


@pytest.mark.parametrize(
    "query,expected_id,why",
    [
        ("What is NRx panel Kisqali brand?", "WS3-BI-012", "brand bound, brand appositive"),
        ("What is NRx panel west region?", "WS3-BI-012", "region bound, region appositive"),
        ("What is TRx Kisqali?", "WS3-BI-005", "bare brand, no appositive"),
        ("What is NRx panel for Kisqali?", "WS3-BI-012", "preposition ends the walk"),
    ],
)
def test_the_matching_appositive_still_binds(query, expected_id, why, calculator):
    assert _kpi_lookup_evidence({"query": query}), f"{query!r} refused; {why}"
    assert calculator.calls == [expected_id], (query, why, calculator.calls)


# --- r12 MEDIUM: the two-token scope window swallowed the #2139 defect noun ----------------
# `_scope_span` tried `f"{token} {tokens[index+1]}"`, and `brand_from_text` matches a brand
# ANYWHERE in the string it is given. So "cost kisqali" resolved to Kisqali and the pair was
# consumed as one scope span — "What is NRx panel cost Kisqali?" ANSWERED, with the very noun
# #2139 exists to refuse sitting inside the "scope". Substring extraction was being read as
# whole-span membership. Every prior test put the brand FIRST, which is why it survived.
#
# ⚠ THE OBVIOUS GUARD IS WRONG, AND MEASURING IT IS WHAT SHOWED THAT. "Reject the pair when
# the second token resolves alone" would break real phrases:
#
#     'south west'  whole->south      second 'west'->west     <- second DOES resolve alone
#     'mid west'    whole->midwest    second 'west'->west     <- and to a DIFFERENT region
#     'new england' whole->northeast  second 'england'->None
#     'cost kisqali' whole->Kisqali   second 'kisqali'->Kisqali  <- identical: the defect
#
# The operational meaning of "genuine multi-word phrase" is that THE FIRST TOKEN CHANGES THE
# RESOLUTION. Accept the pair only when the pair resolves to something the second token alone
# does not — which admits "south west" and "mid west" and rejects "cost kisqali".


@pytest.mark.parametrize(
    "query,why",
    [
        ("What is NRx panel cost Kisqali?", "the brand matched inside the pair, not as it"),
        ("What is TRx accuracy Kisqali?", "same, canonical KPI"),
        ("What is TRx price Fabhalta?", "same, a different brand"),
        ("What is TRx target west?", "region form: 'west' alone resolves identically"),
    ],
)
def test_a_quantity_noun_before_a_brand_is_not_a_scope_span(query, why, calculator):
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered; {why}"
    assert calculator.calls == [], f"{query!r} called the calculator; {why}"


@pytest.mark.parametrize(
    "query,expected_id,why",
    [
        ("What is TRx new england?", "WS3-BI-005", "second token resolves to nothing"),
        ("What is TRx north east?", "WS3-BI-005", "second token resolves to nothing"),
        ("What is TRx mid west?", "WS3-BI-005", "second RESOLVES, but to a different region"),
        ("What is TRx south west?", "WS3-BI-005", "second RESOLVES, but to a different region"),
    ],
)
def test_a_genuine_multi_word_region_phrase_still_binds(query, expected_id, why, calculator):
    """The two rows that make this a real test are 'mid west' and 'south west': they are the
    ones the obvious guard would have broken, and they are green on base as well as after."""
    assert _kpi_lookup_evidence({"query": query}), f"{query!r} refused; {why}"
    assert calculator.calls == [expected_id], (query, why, calculator.calls)


def test_a_trailing_phrase_word_after_a_resolved_region_still_over_refuses(calculator):
    """A MEASURED PRE-EXISTING OVER-REFUSAL, found while pinning the phrase rows above, and
    pinned rather than quietly fixed because the fix belongs to a different mechanism.

    "What is TRx west coast?" REFUSES, although `region_scan` binds the whole query to
    'west'. The single-token branch consumes "west" first, so the pair branch never sees
    "west coast", and "coast" is then judged alone and refuses.

    PROVENANCE, measured with a per-commit control (not inferred from the diff):

        f3663f2d6  (11g, flat _SCOPE_NOUNS)  REFUSES
        9340bcd15  (11h+correction)          REFUSES
        9259cf5e2  (r12 HIGH-b, keyed)       REFUSES

    So it arrived with `_scope_span` in 11g and is NOT a regression of the keying change.
    It cannot be closed by the r12-MEDIUM rule above: "west coast" and "Kisqali tier" are
    STRUCTURALLY IDENTICAL to these resolvers — first token resolves, pair resolves to the
    same value — and one must bind while the other must refuse. Separating them needs the
    resolver's own multi-word phrase vocabulary (`_FREE_TEXT_REGION_PHRASES`), which is a
    different mechanism from anything in this commit. Fail-closed, so the lesser evil under
    this lane's ordering; reported for its own decision rather than absorbed here."""
    assert _kpi_lookup_evidence({"query": "What is TRx west coast?"}) is None
    assert calculator.calls == []


# --- r12 MEDIUM: a determiner licensed everything behind it --------------------------------
# Traced before fixing: 'last' is in BOTH _RIGHT_HEAD_FUNCTION_WORDS and _PERIOD_MODIFIERS.
# Its follower 'two' is not a period token, so the modifier branch was skipped and control
# reached the function-word `return False` — the determiner ENDED the walk and bound, with
# 'cost' never examined. 'the' did the same one token earlier.
#
# A determiner now never ends the walk; it is consumed and the decision deferred to what
# follows. Prepositions still end it — that distinction is the whole design, and it is why
# "for Kisqali" cannot be judged by its object.
#
# ⚠ THIS REVERSES AN 11f PIN, deliberately: "this brand" now REFUSES on both paths. Measured
# reason, not taste — it binds with calculator context {} (`brand_from_text('this brand')` is
# None), so it is a dropped qualifier, exactly what #2141 covers. Contrast the row that must
# keep binding: "last quarter" arrives with a real window,
#     context={'window': {'start': '2026-04-01T00:00:00+00:00', 'end': '2026-07-01T...'}}
# Temporal scope IS served; "this brand" is not. The determiner rule separates them.


@pytest.mark.parametrize(
    "query,why",
    [
        ("What is TRx last two quarters cost?", "'last' bound before 'cost' was seen"),
        ("What is TRx the last two quarters cost?", "'the' did it one token earlier"),
        ("What is NRx panel the cost?", "bare determiner then the defect noun"),
        ("What is NRx panel this brand?", "dropped qualifier: context {} (reverses an 11f pin)"),
        ("What is TRx this segment cost?", "determiner, unserved axis, quantity"),
    ],
)
def test_a_determiner_does_not_license_the_noun_behind_it(query, why, calculator):
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered; {why}"
    assert calculator.calls == [], f"{query!r} called the calculator; {why}"


def test_a_determiner_does_not_license_the_noun_behind_it_on_the_causal_path(causal_registry):
    """The causal half of the same reversal, asserted through its own consumer."""
    assert _causal_kpi_id("what drives NRx panel this brand?") is None
    assert causal_registry == []


@pytest.mark.parametrize(
    "query,expected_id,why",
    [
        ("What is TRx last quarter?", "WS3-BI-005", "determiner then period, then EOS"),
        ("What is NRx panel this year?", "WS3-BI-012", "determiner then period"),
        ("What is NRx panel next quarter?", "WS3-BI-012", "determiner then period"),
        ("What is TRx vs the prior period?", "WS3-BI-005", "comparison ends the walk first"),
        ("What is NRx panel for Kisqali?", "WS3-BI-012", "preposition still ends the walk"),
        ("What is NRx panel in the west region?", "WS3-BI-012", "preposition, then scope"),
        ("What is TRx recently?", "WS3-BI-005", "adverb, not a determiner"),
    ],
)
def test_temporal_determiners_and_prepositions_still_bind(query, expected_id, why, calculator):
    assert _kpi_lookup_evidence({"query": query}), f"{query!r} refused; {why}"
    assert calculator.calls == [expected_id], (query, why, calculator.calls)


@pytest.mark.parametrize(
    "query,expected_id,why",
    [
        (
            "What is the TRx, the total prescriptions, for Kisqali?",
            "WS3-BI-005",
            "appositive restatement of the SAME metric",
        ),
        ("What is NRx panel, the panel count, for Kisqali?", "WS3-BI-012", "same shape"),
    ],
)
def test_a_clause_boundary_ends_the_compound(query, expected_id, why, calculator):
    """CAUGHT BY GATE B AGAIN, and by #1475's suite rather than this file: the determiner fix
    consumed "the" in "the TRx, the total prescriptions, ..." and then judged "total" — a word
    from this KPI's OWN registry name — as a foreign quantity, failing closed on a legitimate
    restatement. A compound head cannot span a clause boundary, so the tail is cut at the
    first one. Pinned here so this module's own tests catch it next time; the second row is
    the same shape on a panel KPI, which #1475 does not cover."""
    assert _kpi_lookup_evidence({"query": query}), f"{query!r} refused; {why}"
    assert calculator.calls == [expected_id], (query, why, calculator.calls)


@pytest.mark.parametrize(
    "query,why",
    [
        ("What is NRx panel cost, for Kisqali?", "the defect noun is BEFORE the comma"),
        ("What is TRx accuracy; for Kisqali?", "semicolon likewise"),
    ],
)
def test_a_clause_boundary_does_not_rescue_a_compound_before_it(query, why, calculator):
    """The cut must not become an escape hatch: a quantity noun sitting BEFORE the boundary
    is still judged. Without this row the fix above would be a hole rather than a limit."""
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered; {why}"
    assert calculator.calls == [], f"{query!r} called the calculator; {why}"


# --- r12 MEDIUM: the lane over-refused a brand alias the lane itself added -----------------
# `_TAIL_TOKEN_RE` was `[\w'-]+`, which splits "hr+" into "hr" — and `brand_from_text('hr')`
# is None while `brand_from_text('hr+')` is 'Kisqali'. Commit b09a3271d IN THIS LANE exists
# precisely to let HR+ ground Kisqali (#2114), so the walk was refusing input its own
# discriminator says must bind. Self-inflicted, and invisible to every test because they all
# used plain alphabetic brands.
#
# MEASURED, correcting half the brief: '/' needs no handling. Normalisation turns it into a
# space long before the walk sees it —
#     "What is NRx panel HR+/HER2-?"  ->  normalized tail ' hr+ her2 ?'
# so only '+' has to survive tokenisation.


def test_a_brand_alias_with_punctuation_still_binds(calculator):
    assert _kpi_lookup_evidence({"query": "What is NRx panel HR+?"}), "HR+ refused"
    assert calculator.calls == ["WS3-BI-012"], calculator.calls


@pytest.mark.parametrize(
    "query,why",
    [
        ("What is TRx triple negative?", "brand_from_text('triple negative') is None"),
        ("What is NRx panel HR+ cost?", "the alias does not license a quantity behind it"),
    ],
)
def test_the_alias_fix_does_not_open_a_hole(query, why, calculator):
    assert _kpi_lookup_evidence({"query": query}) is None, f"{query!r} answered; {why}"
    assert calculator.calls == [], f"{query!r} called the calculator; {why}"


def test_a_split_compound_alias_binds_because_both_halves_resolve(calculator):
    """I PREDICTED THIS WOULD REFUSE AND THE TEST DISPROVED ME — recorded because the wrong
    prediction is the useful part.

    I reasoned from the tokenisation alone: "HR+/HER2-" normalises to 'hr+ her2', 'hr+'
    resolves so the single-token branch consumes it, and 'her2' would then be judged alone
    and refuse — the same shape as the "west coast" limit. Measured instead:

        brand_from_text('hr+')  = 'Kisqali'
        brand_from_text('her2') = 'Kisqali'   <- an INDICATION alias, INDICATION_TO_BRAND

    Both halves resolve independently, so both are consumed and the ask binds. The fix is
    worth more than I claimed for it. Reading the token rule told me about the tokeniser and
    nothing about the resolver behind it, which is the layer confusion this lane keeps
    finding — here caught by red-first rather than shipped in a docstring."""
    assert _kpi_lookup_evidence({"query": "What is NRx panel HR+/HER2-?"})
    assert calculator.calls == ["WS3-BI-012"], calculator.calls
