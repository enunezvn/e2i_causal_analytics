"""#1775 — grounding a causal scenario in the clinical context that bears on it.

#1763 made the context follow the analysis for drug-therapy and covariate
treatments. For COMMERCIAL levers it shipped an honest refusal instead — "Open
Targets and the FDA label describe the therapy, not this lever" — and on the
patient_journeys dataset 5 of the 10 selectable treatments are commercial. Half of
all analyses therefore got a panel that declined to connect itself to the question.

Refusing to make a claim ABOUT the lever was right. Refusing to ground the analysis
was not: a copay-support persistence question has obvious clinical content bearing
on it — what the label says drives discontinuation, and what a patient switches to.
"""

from __future__ import annotations

import pytest

from src.services.clinical_context.analysis_grounding import ground_analysis
from src.services.clinical_context.brand_map import resolve_brand_profile, treatment_context_for
from src.services.clinical_context.label_considerations import (
    DOSAGE_SECTION,
    WARNINGS_SECTION,
    LabelConsideration,
)

_MONITORING = LabelConsideration(
    title="QT Interval Prolongation",
    detail="Monitor electrocardiograms (ECGs) and electrolytes prior to initiation.",
    section=WARNINGS_SECTION,
    references="2.2 , 5.3",
)
_SCHEDULE = LabelConsideration(
    title="Advanced or Metastatic Breast Cancer Recommended starting dose",
    detail="600 mg orally once daily for 21 consecutive days followed by 7 days off treatment.",
    section=DOSAGE_SECTION,
    references="2.1",
)
_INTERRUPTION = LabelConsideration(
    title="Dosage and administration",
    detail="Dose interruption, reduction, and/or discontinuation may be required based on "
    "individual safety and tolerability.",
    section=DOSAGE_SECTION,
    references="2.2",
)
_EMBRYO = LabelConsideration(
    title="Embryo-Fetal Toxicity",
    detail="Can cause fetal harm. Advise females of reproductive potential to use effective "
    "contraception during therapy.",
    section=WARNINGS_SECTION,
    references="5.7",
)
_ALL = (_MONITORING, _SCHEDULE, _INTERRUPTION, _EMBRYO)


def _ground(
    treatment,
    outcome="persistent_180d",
    considerations=_ALL,
    brand="Kisqali",
    label_source="openfda",
):
    """Defaults to a label that WAS read. Provenance is load-bearing: an empty
    consideration list means "checked, none" under openfda and "could not check"
    under static_fallback, and those must never share a sentence."""
    return ground_analysis(
        resolve_brand_profile(brand),
        outcome=outcome,
        treatment_context=treatment_context_for(brand, treatment),
        label_considerations=considerations,
        label_source=label_source,
    )


@pytest.mark.unit
def test_a_commercial_lever_is_GROUNDED_not_refused():
    """THE #1775 REGRESSION. copay_support -> persistent_180d used to receive no
    clinical content at all for the treatment side."""
    g = _ground("copay_support")
    assert g.label_considerations, "a commercial analysis must still be grounded"
    titles = [c.title for c in g.label_considerations]
    assert "QT Interval Prolongation" in titles
    assert g.competitive_context
    assert g.note


@pytest.mark.unit
def test_grounding_never_claims_the_label_speaks_to_the_lever():
    """The honesty boundary that #1763 was right about, preserved. The label says
    nothing about copay assistance and must not appear to."""
    g = _ground("copay_support")
    low = g.note.lower()
    assert "copay" in low  # the lever is named...
    # ...but never as something the label or the regulator speaks to.
    assert "label does not" in low or "not a claim" in low or "says nothing about" in low


@pytest.mark.unit
def test_considerations_are_selected_by_the_OUTCOME_being_analysed():
    """A persistence question wants what drives stopping — monitoring burden, dose
    interruption, the dosing schedule. Embryo-fetal toxicity is real and important
    and has nothing to do with 180-day persistence."""
    g = _ground("copay_support", outcome="persistent_180d")
    titles = [c.title for c in g.label_considerations]
    assert "Dosage and administration" in titles  # dose interruption
    assert "Embryo-Fetal Toxicity" not in titles


@pytest.mark.unit
def test_the_outcome_filter_is_disclosed_so_it_cannot_read_as_the_whole_label():
    """Showing a filtered subset without saying so invites the reader to treat it as
    the complete safety picture."""
    g = _ground("copay_support")
    assert "persistence" in g.note.lower() or "staying on" in g.note.lower()
    assert "not the complete" in g.note.lower() or "full prescribing" in g.note.lower()


@pytest.mark.unit
def test_an_initiation_outcome_selects_pre_initiation_gates_instead():
    g = _ground("copay_support", outcome="treatment_initiated")
    titles = [c.title for c in g.label_considerations]
    assert "QT Interval Prolongation" in titles  # "prior to initiation"
    assert "Advanced or Metastatic Breast Cancer Recommended starting dose" not in titles


@pytest.mark.unit
def test_the_competitive_context_frames_switching_as_a_competing_risk():
    """A bare list of two competitor names does not ground anything. For a
    persistence question the competitor set IS the alternative a patient switches
    to, which is a competing risk for the outcome and a confounder for the estimate."""
    g = _ground("copay_support")
    ctx = g.competitive_context or ""
    assert "Ibrance (palbociclib)" in ctx and "Verzenio (abemaciclib)" in ctx
    assert "switch" in ctx.lower()


@pytest.mark.unit
def test_a_drug_therapy_analysis_is_grounded_too():
    """Grounding is about the SCENARIO, not about rescuing commercial levers."""
    g = _ground("treatment_arm")
    assert g.label_considerations
    assert g.competitive_context


@pytest.mark.unit
def test_no_label_considerations_yields_an_honest_empty_grounding():
    """openFDA down, or a label with no parseable Highlights: say nothing rather
    than invent a clinical consideration."""
    g = _ground("copay_support", considerations=(), label_source="static_fallback")
    assert g.label_considerations == ()
    assert "could not" in g.note.lower() or "no label" in g.note.lower()


@pytest.mark.unit
def test_an_uncurated_treatment_is_not_grounded_against_an_invented_scenario():
    g = ground_analysis(
        resolve_brand_profile("Kisqali"),
        outcome="persistent_180d",
        treatment_context=None,
        label_considerations=_ALL,
    )
    assert g.label_considerations == ()
    assert g.competitive_context is None


# --- found by running the REAL providers against the LIVE labels ---------------


@pytest.mark.unit
def test_the_note_does_not_talk_about_stopping_on_an_initiation_outcome():
    """Found live on Fabhalta psp_enrolled -> treatment_initiated: the commercial
    clause was hardcoded to "the reasons a patient stops", which is the wrong
    question entirely for an initiation outcome."""
    g = _ground("copay_support", outcome="treatment_initiated")
    low = g.note.lower()
    assert "reasons a patient stops" not in low
    assert "start" in low


@pytest.mark.unit
def test_a_label_that_read_fine_but_matched_nothing_is_not_reported_as_unreadable():
    """Found live: the note said "The FDA label ... could not be read", when the
    label had been read perfectly and simply carried no Highlights bullet bearing on
    that outcome. That is the outage-vs-absence conflation of #1767 recurring here —
    'we could not check' and 'we checked and there is none' are different claims."""
    # _EMBRYO matches NEITHER theme. Using a persistence item here would match the
    # initiation cue "prior to initiation" inside its own text and pass vacuously.
    g = _ground("copay_support", outcome="treatment_initiated", considerations=(_EMBRYO,))
    low = g.note.lower()
    assert "could not be read" not in low
    assert "none of" in low or "no highlighted" in low


@pytest.mark.unit
def test_no_considerations_at_all_still_reads_as_could_not_check():
    """The other side of the same split: nothing came back from the label at all."""
    g = _ground("copay_support", considerations=(), label_source="static_fallback")
    assert "could not be read" in g.note.lower()


@pytest.mark.unit
def test_a_boxed_warning_is_available_as_a_consideration():
    """Found live on Fabhalta: its initiation gate — vaccinate before the first dose
    — lives in the BOXED WARNING, which is prose rather than Highlights bullets and
    so produced no consideration at all. It reaches the panel whole elsewhere, but
    it was invisible to grounding."""
    from src.services.clinical_context.label_considerations import (
        BOXED_WARNING_SECTION,
        boxed_warning_consideration,
    )

    item = boxed_warning_consideration(
        "WARNING: SERIOUS INFECTIONS CAUSED BY ENCAPSULATED BACTERIA FABHALTA increases the "
        "risk of serious infections. Complete or update vaccinations against encapsulated "
        "bacteria at least 2 weeks prior to initiation."
    )
    assert item is not None
    assert item.section == BOXED_WARNING_SECTION
    assert item.source == "openfda"
    g = _ground("copay_support", outcome="treatment_initiated", considerations=(item,))
    assert [c.section for c in g.label_considerations] == [BOXED_WARNING_SECTION]
    assert boxed_warning_consideration(None) is None
    assert boxed_warning_consideration("   ") is None


# --- codex iter-1 -------------------------------------------------------------


@pytest.mark.unit
def test_every_curated_outcome_gets_a_theme():
    """codex HIGH. `low_gap_180d` (gap-free refill adherence) and `adopted`
    (prescriber adoption) were omitted, so an adherence analysis — the one most
    directly about staying on therapy — was grounded on NOTHING while claiming the
    label could not be read."""
    from src.services.clinical_context.analysis_grounding import _theme_for

    for outcome in ("persistent_180d", "discontinued_180d", "adherent_180d", "low_gap_180d"):
        assert _theme_for(outcome) == "persistence", outcome
    for outcome in ("treatment_initiated", "adopted"):
        assert _theme_for(outcome) == "initiation", outcome


@pytest.mark.unit
def test_an_adherence_outcome_is_grounded_like_a_persistence_one():
    g = _ground("copay_support", outcome="low_gap_180d")
    assert [c.title for c in g.label_considerations]
    assert "staying on therapy" in g.note.lower()


@pytest.mark.unit
def test_an_unrecognised_outcome_does_not_get_the_starting_therapy_story():
    """codex HIGH. The theme phrase was a two-way ternary, so an outcome we have no
    theme for fell through to 'starting therapy' — asserting a story about the
    analysis that we never established."""
    g = _ground("copay_support", outcome="some_unmapped_outcome")
    low = g.note.lower()
    assert "starting therapy" not in low
    assert "staying on therapy" not in low
    # The commercial backdrop clause defaulted to the initiation wording too.
    assert "requirements for starting" not in low
    assert "reasons a patient stops" not in low


@pytest.mark.unit
def test_a_label_read_with_no_parseable_items_is_not_reported_as_unreadable():
    """codex HIGH — the #1767 conflation surviving my own fix for it. An empty
    consideration list meant 'could not be read', but the provider returns
    source='openfda' with an empty tuple when the label WAS read and simply carried
    no parseable Highlights."""
    g = ground_analysis(
        resolve_brand_profile("Kisqali"),
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "copay_support"),
        label_considerations=(),
        label_source="openfda",
    )
    assert "could not be read" not in g.note.lower()
    assert "no highlighted" in g.note.lower() or "carries no" in g.note.lower()


@pytest.mark.unit
def test_an_unreachable_label_still_reads_as_could_not_check():
    g = ground_analysis(
        resolve_brand_profile("Kisqali"),
        outcome="persistent_180d",
        treatment_context=treatment_context_for("Kisqali", "copay_support"),
        label_considerations=(),
        label_source="static_fallback",
    )
    assert "could not be read" in g.note.lower()


@pytest.mark.unit
def test_a_bare_clinical_reduction_does_not_count_as_a_persistence_factor():
    """codex MEDIUM. The cue list contained bare 'reduction', which matches any
    clinical reduction ('reduction in tumour size') rather than DOSE reduction."""
    item = LabelConsideration(
        title="Efficacy",
        detail="A reduction in circulating tumour cells was observed.",
        section=WARNINGS_SECTION,
        references="5.9",
    )
    g = _ground("copay_support", considerations=(item,))
    assert g.label_considerations == ()


@pytest.mark.unit
def test_an_unmapped_outcome_gets_no_competitive_claim_either():
    """codex iter-8 HIGH. `_select` correctly returns nothing for an outcome we never
    mapped, but `_competitive_context` fell through to a generic "Same-class
    alternatives in X: ..." — and the panel renders that under the heading "What
    bears on this analysis". Asserting that competitors bear on an outcome the code
    explicitly declined to map is borrowed relevance, which is the exact complaint
    #1763 was filed about.

    No theme means no grounding CLAIMS at all. The honest note still renders, and
    says what was and was not established.
    """
    profile = resolve_brand_profile("Kisqali")
    grounding = ground_analysis(
        profile,
        outcome="some_unmapped_outcome",
        treatment_context=treatment_context_for("Kisqali", "copay_support"),
        label_considerations=(),
        label_source="openfda",
    )
    assert grounding.outcome_theme == ""
    assert grounding.label_considerations == ()
    assert grounding.competitive_context is None, grounding.competitive_context
    # the disclosure survives — it is the only thing left to render
    assert grounding.note
