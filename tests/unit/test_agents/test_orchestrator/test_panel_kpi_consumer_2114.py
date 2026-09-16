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
    def __init__(self) -> None:
        self.calls: List[str] = []

    def calculate(self, kpi_id: str, context: Dict[str, Any]) -> KPIResult:
        self.calls.append(kpi_id)
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
    ],
)
def test_the_causal_boundary_holds_regardless_of_position_arity_head_or_kpi(
    query, why, causal_registry
):
    assert _causal_kpi_id(query) is None, f"{query!r} bound an outcome; {why}"
    assert causal_registry == [], f"{query!r} consulted the registry; {why}"


def test_three_clean_causal_occurrences_still_bind(causal_registry):
    """The ANSWER side of the arity probe on the causal path."""
    assert _causal_kpi_id("what drives NRx panel and NRx panel and NRx panel?") == "WS3-BI-012"
    assert causal_registry == ["Observed Rx Events - Patient Panel NRx (NRx Panel)"]
