"""#1775 — ground a causal scenario in the clinical context that bears on it.

#1763 made the clinical panel follow the analysis for ``drug_therapy`` and
``clinical_covariate`` treatments. For ``commercial`` levers it shipped an honest
REFUSAL instead: "Open Targets and the FDA label describe the therapy and its
indication, not this lever." On the ``patient_journeys`` dataset 5 of the 10
selectable treatments are commercial, so half of every analysis an analyst can run
got a panel that declined to connect itself to the question being asked.

Declining to make a claim ABOUT the lever was right and is preserved. Declining to
GROUND the analysis was not. "Does copay support improve 180-day persistence?" has
obvious clinical content bearing on it: what the label says drives discontinuation
(monitoring burden, dose interruption, the dosing schedule) and what a patient
switches to when they stop. That is the clinical backdrop the commercial effect is
being isolated against — it is confounding structure, not a regulatory claim.

Nothing here is generated. Label considerations are verbatim label text selected by
the outcome under analysis (see ``label_considerations``); the competitive framing
is composed from the curated competitor map. Both are labelled with their source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from src.services.clinical_context.brand_map import (
    BrandClinicalProfile,
    TreatmentContext,
    outcome_framing_for,
)
from src.services.clinical_context.label_considerations import LabelConsideration

# Outcomes that ask "does the patient STAY on therapy". What matters is the burden
# of remaining on it: what must be monitored, when treatment is interrupted or
# reduced, and how demanding the schedule is.
# Kept in step with brand_map._OUTCOME_FRAMING — an outcome missing here is grounded
# on NOTHING while claiming the label could not be read (codex iter-1 HIGH).
_PERSISTENCE_OUTCOMES = frozenset(
    {"persistent_180d", "discontinued_180d", "adherent_180d", "low_gap_180d"}
)
# Outcomes that ask "does the patient START" — or, at HCP grain, does the prescriber
# start prescribing. What matters is what gates the first dose.
_INITIATION_OUTCOMES = frozenset({"treatment_initiated", "adopted"})

PERSISTENCE_THEME = "persistence"
INITIATION_THEME = "initiation"

# Verbatim label phrasing that marks a consideration as bearing on STAYING on
# therapy. Matched against the label's own words — no clinical inference is made
# about a consideration the label did not phrase this way.
_PERSISTENCE_CUES = (
    "monitor",
    "interrupt",
    "discontinu",
    "dose reduction",
    "dose modification",
    "withhold",
    "tolerability",
    "days off treatment",
    "permanently discontinue",
)
# Phrasing that marks a consideration as a gate on STARTING therapy.
_INITIATION_CUES = (
    "before initiating",
    "prior to initiation",
    "prior to starting",
    "before the first dose",
    "before treatment",
    "vaccinat",
    "for initiation",
    "contraindicated",
    "before initiation",
)


@dataclass(frozen=True)
class AnalysisGrounding:
    """The clinical context that bears on ONE (treatment -> outcome) analysis."""

    label_considerations: Tuple[LabelConsideration, ...] = field(default_factory=tuple)
    competitive_context: Optional[str] = None
    note: str = ""
    outcome_theme: str = ""


def _theme_for(outcome: str) -> str:
    if outcome in _PERSISTENCE_OUTCOMES:
        return PERSISTENCE_THEME
    if outcome in _INITIATION_OUTCOMES:
        return INITIATION_THEME
    return ""


# Sentence boundary, good enough for label prose. Abbreviations inside a bullet
# ("e.g.", "i.e.") would over-split, which costs nothing here: an over-split sentence
# still carries its own cue or does not.
_SENTENCE = re.compile(r"(?<=\.)\s+")


def _earns_theme(consideration: LabelConsideration, theme: str) -> bool:
    """True when some SENTENCE carries the theme's cue without being scoped away.

    Matching cues over the whole item let a cue trapped inside an initiation clause
    earn the persistence claim: "Monitor ECGs and electrolytes prior to initiation."
    was reported as a factor bearing on STAYING on therapy when it establishes only a
    pre-initiation gate (codex iter-11 HIGH).

    The blanket rule — an item that mentions initiation is not persistence — was
    measured against the live ribociclib label and rejected: all three items it would
    have dropped carry BOTH a gate and ongoing monitoring ("Perform CBC before
    initiating therapy. Monitor CBC every 2 weeks ..."), and they are real persistence
    grounding. Scoping per sentence keeps those and drops the gate-only case.
    """
    cues = _PERSISTENCE_CUES if theme == PERSISTENCE_THEME else _INITIATION_CUES
    other = _INITIATION_CUES if theme == PERSISTENCE_THEME else ()
    for sentence in _SENTENCE.split(f"{consideration.title}. {consideration.detail}"):
        lowered = sentence.lower()
        if any(cue in lowered for cue in cues) and not any(cue in lowered for cue in other):
            return True
    return False


def _select(
    considerations: Sequence[LabelConsideration], theme: str
) -> Tuple[LabelConsideration, ...]:
    """Considerations whose OWN WORDS bear on the theme under analysis.

    An unrecognised outcome selects nothing rather than showing the whole label:
    an unfiltered dump under an "evidence for this analysis" heading is the
    borrowed-relevance failure #1763 was filed about.
    """
    if not theme:
        return ()
    return tuple(c for c in considerations if _earns_theme(c, theme))


def _competitive_context(profile: BrandClinicalProfile, theme: str) -> Optional[str]:
    competitors = profile.competitor_map.get(profile.disease.lower()) or []
    if not competitors:
        return None
    listed = ", ".join(competitors)
    if theme == PERSISTENCE_THEME:
        return (
            f"A patient who stops {profile.drug_name} in {profile.disease_search_term} has "
            f"alternatives within the same class: {listed}. A switch to one of these is a "
            f"competing risk for this outcome rather than a simple failure to persist, and "
            f"it is confounding structure for any effect estimated here."
        )
    if theme == INITIATION_THEME:
        return (
            f"At initiation in {profile.disease_search_term}, {profile.drug_name} is chosen "
            f"against the same-class alternatives {listed}. Which therapy a patient starts "
            f"is confounding structure for any effect estimated here."
        )
    # No theme means we never established what the analysis asks, so there is nothing
    # to say bears on it. This used to fall through to "Same-class alternatives in
    # X: ..." which the panel renders under the heading "What bears on this analysis"
    # — asserting relevance to an outcome the code explicitly declined to map, which
    # is the borrowed-relevance complaint #1763 was filed about (codex iter-8 HIGH).
    # `_select` already returns nothing here; this makes the two consistent. The
    # honest note still renders and says what was and was not established.
    return None


def _note(
    profile: BrandClinicalProfile,
    treatment_context: TreatmentContext,
    outcome: str,
    theme: str,
    selected: Sequence[LabelConsideration],
    available: Sequence[LabelConsideration],
    label_source: str,
) -> str:
    outcome_phrase = outcome_framing_for(outcome)
    # A two-way ternary made an outcome we have NO theme for fall through to the
    # "starting therapy" story — asserting something about the analysis we never
    # established (codex iter-1 HIGH).
    theme_phrase = {
        PERSISTENCE_THEME: "staying on therapy",
        INITIATION_THEME: "starting therapy",
    }.get(theme, "")
    parts: list[str] = []

    if not theme:
        # NONE of the branches below fit an outcome we never mapped, and it used to
        # fall into the one for "the label was read", claiming "none of its
        # highlighted factors are phrased around <outcome>". That is a statement about
        # the LABEL, made when `_select` never evaluated relevance at all because
        # `_theme_for` produced no theme — false outright whenever a relevant factor
        # is sitting in the input (codex iter-12 HIGH). The gap is in our mapping.
        # An unmapped outcome has no curated phrasing, so `outcome_framing_for` hands
        # back the raw key — "trx_volume" rendered as "Trx_volume" to an analyst.
        # Naming the identifier is right (it is what the analysis actually selected);
        # showing it with its underscore is just us leaking a dict key into prose.
        readable = outcome_phrase.replace("_", " ")
        parts.append(
            f"{readable.capitalize()} is not one we have mapped to a clinical "
            f"question, so no label factors were selected for it. That is a gap in our "
            f"mapping, not a statement about what the {profile.drug_name} label contains."
        )
    elif selected:
        parts.append(
            f"Label factors bearing on {theme_phrase}, selected from the prescribing "
            f"information for {profile.drug_name} by relevance to {outcome_phrase}. This is "
            f"a filtered view, not the complete safety profile — each item cites the label "
            f"section it came from."
        )
    elif label_source == "openfda":
        # The label WAS read; it simply carries no highlighted factor phrased around
        # this outcome. "We checked and there is none" is a different claim from "we
        # could not check", and conflating them is the #1767 defect in a new place.
        # The label WAS read. Whether it carried no parseable Highlights at all, or
        # carried some that are phrased around a different question, "we checked and
        # there is none" is a different claim from "we could not check". Conflating
        # them is the #1767 defect, and it survived my first fix for it here
        # (codex iter-1 HIGH).
        detail = (
            f"none of its highlighted factors are phrased around {outcome_phrase}"
            if available
            else "it carries no highlighted factors we can read"
        )
        parts.append(
            f"The prescribing information for {profile.drug_name} was read, but {detail}; "
            f"the full prescribing information may still bear on it."
        )
    else:
        parts.append(
            f"The FDA label for {profile.drug_name} could not be read for factors bearing "
            f"on {outcome_phrase}, so what is missing here is unknown, not absent."
        )

    if treatment_context.kind == "commercial":
        # The #1763 boundary, kept exactly: the label is silent on the lever. What
        # changed is that we no longer stop at saying so.
        boundary = (
            f"{treatment_context.label} is a commercial access lever and the label says "
            f"nothing about it; none of the above is a claim that the label speaks to "
            f"{treatment_context.label.lower()}."
        )
        # The backdrop sentence tells the reader HOW TO READ THE ESTIMATE, so it may
        # only be said once we know what the outcome asks. A `.get(theme, default)`
        # kept asserting "the clinical picture that has nothing to do with access" for
        # an outcome `_theme_for` had explicitly declined to map — the same
        # borrowed-relevance defect as the competitive fallthrough, one function over
        # (codex iter-10 HIGH). No theme now means the boundary is stated and nothing
        # more is claimed.
        backdrop = {
            PERSISTENCE_THEME: "the reasons a patient stops that have nothing to do with access",
            INITIATION_THEME: (
                "the clinical requirements for starting that have nothing to do with access"
            ),
        }.get(theme)
        if backdrop:
            boundary += (
                f" It is the clinical backdrop the lever operates against — {backdrop}, "
                f"which an estimate of this lever has to be read alongside."
            )
        parts.append(boundary)
    return " ".join(parts)


def ground_analysis(
    profile: BrandClinicalProfile,
    *,
    outcome: str,
    treatment_context: Optional[TreatmentContext],
    label_considerations: Sequence[LabelConsideration],
    label_source: str = "static_fallback",
) -> AnalysisGrounding:
    """Clinical grounding for one (treatment -> outcome) analysis.

    Returns an empty grounding when there is no curated treatment framing: without
    a scenario there is nothing to ground, and guessing one is what #1763 was about.
    """
    if treatment_context is None:
        return AnalysisGrounding()
    theme = _theme_for(outcome)
    selected = _select(label_considerations, theme)
    return AnalysisGrounding(
        label_considerations=selected,
        competitive_context=_competitive_context(profile, theme),
        note=_note(
            profile,
            treatment_context,
            outcome,
            theme,
            selected,
            label_considerations,
            label_source,
        ),
        outcome_theme=theme,
    )
