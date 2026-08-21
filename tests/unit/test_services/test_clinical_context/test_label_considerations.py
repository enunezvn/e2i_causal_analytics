"""#1775 — label considerations, parsed VERBATIM from the FDA label Highlights.

The panel must ground a causal scenario in what the label actually says that
bears on the OUTCOME being analysed (why a patient stops, what gates starting).
CLAUDE.md forbids plausible-but-fake values in production paths, so every item's
DETAIL is verbatim label text — no summarisation, no LLM, nothing invented. Titles
and references carry a weaker guarantee (a title may be the section's plain name, and
the boxed warning has no cross-reference of its own); the module docstring states it
per field. This header used to make the blanket claim, which was false for the boxed
warning emitter its sibling test file covers.

Fixtures are trimmed but otherwise VERBATIM excerpts captured from the live
openFDA labels on 2026-08-21.
"""

from __future__ import annotations

import pytest

from src.services.clinical_context.label_considerations import (
    CONTRAINDICATIONS_SECTION,
    DOSAGE_SECTION,
    SECTION_DISPLAY,
    WARNINGS_SECTION,
    parse_label_considerations,
)

# Verbatim from the live ribociclib (KISQALI) label, section 5.
_KISQALI_WARNINGS = (
    "5 WARNINGS AND PRECAUTIONS Interstitial Lung Disease (ILD)/Pneumonitis: Severe, "
    "life threatening, or fatal ILD/pneumonitis can occur with KISQALI treatment. "
    "Monitor for pulmonary symptoms indicative of ILD/pneumonitis. Interrupt and "
    "evaluate patients with new or worsening respiratory symptoms suspected to be due "
    "to ILD/pneumonitis. Permanently discontinue KISQALI in patients with recurrent "
    "symptomatic or severe ILD/pneumonitis. ( 2.2 , 5.1 ) "
    "QT Interval Prolongation: KISQALI has been shown to prolong the QT interval in a "
    "concentration-dependent manner. Monitor electrocardiograms (ECGs) and electrolytes "
    "prior to initiation of treatment with KISQALI. ( 2.2 , 5.3 , 7.1 , 7.4 ) "
    "Neutropenia: Perform complete blood count (CBC) before initiating therapy with "
    "KISQALI. Monitor CBC every 2 weeks for the first 2 cycles. ( 2.2 , 5.6 ) "
    "5.1 Interstitial Lung Disease/Pneumonitis Severe, life-threatening, or fatal "
    "interstitial lung disease (ILD) and/or pneumonitis can occur in patients treated "
    "with KISQALI and other CDK 4/6 inhibitors. In patients with early breast cancer"
)

# Verbatim from the live iptacopan (FABHALTA) label, section 5.
_FABHALTA_WARNINGS = (
    "5 WARNINGS AND PRECAUTIONS Monitoring of PNH Manifestations After FABHALTA "
    "Discontinuation: Monitor for signs of hemolysis after discontinuation. ( 5.3 ) "
    "Hyperlipidemia: Monitor serum lipid parameters periodically during treatment and "
    "initiate cholesterol-lowering medication, if indicated. ( 5.4 )"
)

# Verbatim from the live remibrutinib (RHAPSIDO) label, section 5.
_RHAPSIDO_WARNINGS = (
    "5 WARNINGS AND PRECAUTIONS Risk of Bleeding: Monitor for signs and symptoms of "
    "bleeding. Interrupt treatment with RHAPSIDO if bleeding is observed or pre- and "
    "post-surgery. ( 5.1 ) "
    "Live Attenuated Vaccines: Avoid live or live-attenuated vaccines in patients "
    "receiving RHAPSIDO. ( 5.2 )"
)

# Verbatim from the live ribociclib label, section 2 — the persistence-relevant one.
_KISQALI_DOSAGE = (
    "2 DOSAGE AND ADMINISTRATION KISQALI tablets are taken orally with or without food "
    "in combination with an aromatase inhibitor or fulvestrant. ( 2 ) "
    "Early Breast Cancer Recommended starting dose: 400 mg orally (two 200 mg tablets) "
    "taken once daily with or without food for 21 consecutive days followed by 7 days "
    "off treatment. ( 2.1 ) "
    "Dose interruption, reduction, and/or discontinuation may be required based on "
    "individual safety and tolerability. ( 2.2 )"
)


@pytest.mark.unit
def test_items_carry_title_detail_and_the_label_cross_reference():
    items = parse_label_considerations(_KISQALI_WARNINGS, WARNINGS_SECTION)
    titles = [i.title for i in items]
    assert "QT Interval Prolongation" in titles
    qt = next(i for i in items if i.title == "QT Interval Prolongation")
    assert qt.references == "2.2 , 5.3 , 7.1 , 7.4"
    assert qt.section == WARNINGS_SECTION
    assert qt.source == "openfda"
    # Verbatim, not paraphrased.
    assert "Monitor electrocardiograms (ECGs) and electrolytes" in qt.detail


@pytest.mark.unit
def test_the_section_header_does_not_eat_the_first_letter_of_the_first_item():
    """Regression from the prototype: a greedy all-caps header pattern consumed the
    leading capital of the following word, yielding 'onitoring of PNH ...' and
    'isk of Bleeding'. A truncated title is a fabricated title."""
    fab = parse_label_considerations(_FABHALTA_WARNINGS, WARNINGS_SECTION)
    assert fab[0].title == "Monitoring of PNH Manifestations After FABHALTA Discontinuation"
    rhap = parse_label_considerations(_RHAPSIDO_WARNINGS, WARNINGS_SECTION)
    assert rhap[0].title == "Risk of Bleeding"


@pytest.mark.unit
def test_the_section_header_never_leaks_into_an_item_body():
    """The other half of the same bug: when the header is followed by an ALL-CAPS
    brand token or a digit, the heuristic found no boundary and left
    '2 DOSAGE AND ADMINISTRATION KISQALI tablets ...' in the first item."""
    items = parse_label_considerations(_KISQALI_DOSAGE, DOSAGE_SECTION)
    assert items, "expected at least one dosage item"
    for item in items:
        assert "DOSAGE AND ADMINISTRATION" not in item.detail
        assert not item.detail.startswith("2 ")


@pytest.mark.unit
def test_single_level_cross_references_are_matched():
    """'( 2 )' is as valid a reference as '( 2.1 )'. Missing it merged two items."""
    items = parse_label_considerations(_KISQALI_DOSAGE, DOSAGE_SECTION)
    assert any(i.references == "2" for i in items), [i.references for i in items]


@pytest.mark.unit
def test_the_dosing_schedule_survives_parsing():
    """The 3-weeks-on / 1-week-off schedule is the single most persistence-relevant
    fact on the Kisqali label."""
    items = parse_label_considerations(_KISQALI_DOSAGE, DOSAGE_SECTION)
    joined = " ".join(i.detail for i in items)
    assert "21 consecutive days followed by 7 days off treatment" in joined
    assert "Dose interruption, reduction, and/or discontinuation" in joined


@pytest.mark.unit
def test_full_text_subsections_are_excluded_from_the_highlights_items():
    """Everything from '5.1 Interstitial Lung Disease/Pneumonitis' on is the full
    prescribing text, not a Highlights bullet. Including it would produce
    multi-thousand-character 'considerations'."""
    items = parse_label_considerations(_KISQALI_WARNINGS, WARNINGS_SECTION)
    assert len(items) == 3, [i.title for i in items]
    for item in items:
        assert "In patients with early breast cancer" not in item.detail


@pytest.mark.unit
def test_empty_or_missing_text_is_handled():
    assert parse_label_considerations("", WARNINGS_SECTION) == ()
    assert parse_label_considerations(None, WARNINGS_SECTION) == ()  # type: ignore[arg-type]


# --- codex iter-1 -------------------------------------------------------------


@pytest.mark.unit
def test_a_prose_parenthetical_does_not_terminate_an_item():
    """codex HIGH. Any numeric parenthetical used to end an item, so a bullet whose
    own prose contains one was cut in half and the prose number was misattributed as
    a label cross-reference — truncated verbatim text plus an invented citation.

    Measured against the live labels: real Highlights references are always spaced
    ('( 2.2 , 5.1 )'), while inline/prose numbers are not ('(5.1)'). Losing a bullet
    is honest under-reporting; truncating one and citing a made-up section is not.
    """
    text = (
        "5 WARNINGS AND PRECAUTIONS Hepatotoxicity: Assess patients (2) weeks after dose "
        "and monitor liver function. ( 5.1 )"
    )
    items = parse_label_considerations(text, WARNINGS_SECTION)
    assert len(items) == 1, [(i.title, i.references) for i in items]
    assert items[0].references == "5.1"
    assert "Assess patients (2) weeks after dose and monitor liver function." in items[0].detail


@pytest.mark.unit
def test_the_section_header_is_stripped_despite_irregular_whitespace():
    """codex MEDIUM. The header was matched with literal single spaces, so
    '5  WARNINGS  AND\\nPRECAUTIONS' would leak into the first title."""
    text = "5  WARNINGS   AND\nPRECAUTIONS Risk of Bleeding: Monitor for bleeding. ( 5.1 )"
    items = parse_label_considerations(text, WARNINGS_SECTION)
    assert items[0].title == "Risk of Bleeding"
    assert "WARNINGS" not in items[0].detail


@pytest.mark.unit
def test_a_none_only_contraindications_section_yields_no_item():
    """codex LOW. The previous assertion was `x == () or all(...)`, which passes even
    if the parser emits a bogus 'Contraindications: None' item."""
    assert (
        parse_label_considerations(
            "4 CONTRAINDICATIONS None. None. ( 4 )", CONTRAINDICATIONS_SECTION
        )
        == ()
    )


@pytest.mark.unit
def test_an_unspaced_terminal_reference_does_not_merge_two_bullets():
    """codex iter-2 HIGH. Requiring spaced references did not FAIL CLOSED: an
    unspaced terminal ref was simply not a boundary, so the bullet it ended was
    swallowed into the NEXT one and given the next one's citation. That fabricates a
    label item out of two, under a reference it never carried — worse than dropping
    it. A terminal reference is one followed by end-of-region or a new capitalised
    bullet; inline prose numbers are followed by lowercase."""
    text = "5 WARNINGS AND PRECAUTIONS A: Keep this. (5.1) B: Monitor next. ( 5.2 )"
    items = parse_label_considerations(text, WARNINGS_SECTION)
    assert [(i.title, i.references) for i in items] == [("A", "5.1"), ("B", "5.2")]
    assert items[0].detail == "Keep this."
    assert items[1].detail == "Monitor next."


@pytest.mark.unit
def test_inline_prose_numbers_are_still_not_boundaries():
    """The iter-1 protection must survive the iter-2 fix."""
    text = (
        "5 WARNINGS AND PRECAUTIONS Hepatotoxicity: Assess patients (2) weeks after dose "
        "and monitor liver function. ( 5.1 )"
    )
    items = parse_label_considerations(text, WARNINGS_SECTION)
    assert len(items) == 1, [(i.title, i.references) for i in items]
    assert items[0].references == "5.1"
    assert "(2) weeks after dose" in items[0].detail


@pytest.mark.unit
def test_a_prose_number_followed_by_a_capital_is_never_emitted_as_a_citation():
    """codex iter-3 HIGH, and its own iter-6 LOW: the assertions did not check what
    the name claimed.

    The parser DOES treat '(1)' as a boundary here — the shape is identical to a real
    terminal reference — and rejects it only because '1' does not name section 5. The
    pending text is then dropped, so the bullet loses its lead and is re-titled by its
    section. The old test asserted just that the suffix survived, which was true
    whether or not the truncation happened, so it could not fail.

    Pinning the ACTUAL behaviour, which is the safe resolution of an undecidable
    input: the prose number is never rendered as a citation, and no text is ever
    attributed to it. Truncation is under-reporting; misattribution is not.
    """
    text = (
        "5 WARNINGS AND PRECAUTIONS A: Assess patients (1) Patients received therapy "
        "and monitor labs. ( 5.1 )"
    )
    items = parse_label_considerations(text, WARNINGS_SECTION)
    assert len(items) == 1, [(i.title, i.detail, i.references) for i in items]
    assert items[0].references == "5.1", "the prose '(1)' must never become a citation"
    assert "Patients received therapy" in items[0].detail
    # What the old assertions hid: the lead IS dropped, and the item is honestly
    # re-titled by its section rather than keeping a title it no longer leads.
    assert items[0].title == "Warnings and precautions"
    assert "Assess patients" not in items[0].detail


@pytest.mark.unit
def test_a_section_that_opens_straight_into_its_full_text_yields_nothing():
    """codex iter-6 HIGH. `_SUBSECTION` CONSUMED the character before the subsection
    number, so it could not match at position 0. A section carrying no Highlights
    summary at all found no cutoff, and its entire prescribing text — thousands of
    characters — was emitted as one 'consideration'. An honest nothing is the only
    correct answer: there are no Highlights bullets to report."""
    text = "5 WARNINGS AND PRECAUTIONS 5.1 Serious Infections Monitor patients closely. ( 5.1 )"
    assert parse_label_considerations(text, WARNINGS_SECTION) == ()


@pytest.mark.unit
def test_full_prescribing_text_cannot_leak_in_when_the_boundary_is_unspaced():
    """codex iter-3 HIGH. `_SUBSECTION` required whitespace before the full-text
    header, so '(5.1)5.1 Full Text Begins' hid the boundary and prescribing text was
    pulled into a consideration under the wrong citation."""
    text = "5 WARNINGS AND PRECAUTIONS A: Keep this. (5.1)5.1 Full Text Begins Monitor full text. ( 5.2 )"
    items = parse_label_considerations(text, WARNINGS_SECTION)
    for item in items:
        assert "Full Text Begins" not in item.detail
        assert "Monitor full text" not in item.detail


@pytest.mark.unit
def test_a_citation_that_does_not_name_this_section_is_dropped():
    """Defence in depth against a fabricated citation, independent of the regex.

    Measured across the live labels for all three brands: EVERY Highlights bullet in
    section N cites section N, usually alongside others ('2.2 , 5.1' in section 5).
    A parsed 'reference' that never names its own section is not a cross-reference we
    established, so the item is dropped rather than rendered with it.
    """
    text = "5 WARNINGS AND PRECAUTIONS Bogus: Something happened. ( 9.9 )"
    assert parse_label_considerations(text, WARNINGS_SECTION) == ()


@pytest.mark.unit
def test_every_rendered_detail_is_verbatim_from_the_source_section():
    """The promise of this module in one assertion: whatever reaches the panel can be
    found in the label text we were given."""
    import re as _re

    for text, section in (
        (_KISQALI_WARNINGS, WARNINGS_SECTION),
        (_FABHALTA_WARNINGS, WARNINGS_SECTION),
        (_RHAPSIDO_WARNINGS, WARNINGS_SECTION),
        (_KISQALI_DOSAGE, DOSAGE_SECTION),
    ):
        haystack = " ".join(text.split())
        items = parse_label_considerations(text, section)
        assert items, f"expected items for {section}"
        for item in items:
            assert " ".join(item.detail.split()) in haystack, item.detail
            if item.title not in SECTION_DISPLAY.values():
                assert " ".join(item.title.split()) in haystack, item.title
            for ref in _re.split(r"\s*,\s*", item.references):
                assert ref in haystack, ref


@pytest.mark.unit
def test_a_rejected_citation_drops_its_bullet_instead_of_merging_it_forward():
    """codex iter-4 HIGH — the ROOT of this whole family.

    Every validation rule added created a new merge opportunity, because the parser
    was a lazy scan that simply kept going when a citation was rejected: the bullet
    it belonged to got absorbed into the NEXT one and rendered under the next one's
    reference. Here bullet A ends in ')' rather than '.', so its own '( 5.1 )' is
    rejected — and A's text then appeared inside B, cited only to 5.2.

    Dropping an un-attributable bullet is honest under-reporting. Carrying its words
    forward under someone else's citation is fabrication.
    """
    text = (
        "5 WARNINGS AND PRECAUTIONS "
        "A: Avoid use in patients taking strong CYP3A inhibitors (including ketoconazole) ( 5.1 ) "
        "B: Monitor next. ( 5.2 )"
    )
    items = parse_label_considerations(text, WARNINGS_SECTION)
    assert [(i.title, i.references) for i in items] == [("B", "5.2")]
    assert items[0].detail == "Monitor next."
    for item in items:
        assert "ketoconazole" not in item.detail
        assert "5.1" not in item.detail


@pytest.mark.unit
def test_a_bullet_whose_reference_names_another_section_is_dropped_not_merged():
    """The section-number invariant must drop, never absorb, for the same reason."""
    text = (
        "5 WARNINGS AND PRECAUTIONS Bogus: Something happened. ( 9.9 ) "
        "Real: Monitor liver function. ( 5.5 )"
    )
    items = parse_label_considerations(text, WARNINGS_SECTION)
    assert [(i.title, i.references) for i in items] == [("Real", "5.5")]
    assert "Something happened" not in items[0].detail


@pytest.mark.unit
def test_a_terminal_reference_flush_against_the_next_title_does_not_merge():
    """codex iter-5 HIGH. The comment above `_BOUNDARY_AFTER` said position decides,
    not spacing — and then the pattern required whitespace. So '(5.1)B:' was read as
    prose, the cursor never advanced, and BOTH bullets were emitted as one under
    B's '5.2'. The section-number invariant cannot catch this one: 5.1 does name
    section 5, it just does not belong to the text it was attached to.
    """
    text = "5 WARNINGS AND PRECAUTIONS A: Keep this. (5.1)B: Monitor next. ( 5.2 )"
    items = parse_label_considerations(text, WARNINGS_SECTION)
    assert [(i.title, i.references) for i in items] == [("A", "5.1"), ("B", "5.2")]
    assert items[0].detail == "Keep this."
    assert items[1].detail == "Monitor next."


@pytest.mark.unit
def test_a_body_that_swallowed_this_sections_own_reference_is_dropped():
    """Fail-closed, independent of how good the boundary heuristic is.

    Five rounds of findings were one defect wearing different spacing: some shape we
    had not imagined was not recognised as a boundary, so a bullet merged forward
    under a citation it never carried. Rather than keep guessing spacings, assert the
    INVARIANT — a body that still contains a reference naming its own section has
    swallowed a boundary by construction, whatever the spacing was, and cannot be
    attributed to the citation at its end.

    Here the flush-lowercase shape '(5.1)and' is not a bullet title, so no positional
    rule sees a boundary; the invariant drops it anyway.
    """
    text = "5 WARNINGS AND PRECAUTIONS A: Keep this. (5.1)and monitor more. ( 5.2 )"
    items = parse_label_considerations(text, WARNINGS_SECTION)
    for item in items:
        assert "Keep this." not in item.detail, (item.title, item.detail, item.references)
        assert "5.1" not in item.detail


@pytest.mark.unit
def test_bullet_glyphs_delimit_highlights_items():
    """A REAL merge on a REAL label, found by widening the empirical sample.

    Palbociclib's Highlights bullets are delimited by U+2022, so every terminal
    reference is followed by ' • Title' rather than ' Title'. The boundary rule
    required a capital, saw prose, and merged ALL THREE warnings into one item titled
    '• Neutropenia' carrying the THIRD one's citation ('5.3 , 8.1 , 8.3') — a
    fabricated label item, on a marketed oncology drug, of exactly the kind this
    module exists to prevent.

    The self-reference invariant did catch it and drop it, which is the honest
    failure. But three real warnings vanishing is a poor answer when the delimiter is
    unambiguous: a bullet glyph is a bullet boundary, not a shape to guess at.
    """
    text = (
        "5 WARNINGS AND PRECAUTIONS "
        "• Neutropenia: Monitor complete blood count prior to start of therapy. ( 2.2 , 5.1 ) "
        "• Interstitial Lung Disease: Interrupt immediately if suspected. ( 5.2 ) "
        "• Embryo-Fetal Toxicity: Can cause fetal harm. ( 5.3 , 8.1 , 8.3 )"
    )
    items = parse_label_considerations(text, WARNINGS_SECTION)
    assert [(i.title, i.references) for i in items] == [
        ("Neutropenia", "2.2 , 5.1"),
        ("Interstitial Lung Disease", "5.2"),
        ("Embryo-Fetal Toxicity", "5.3 , 8.1 , 8.3"),
    ], [(i.title, i.references) for i in items]
    # the glyph is a delimiter, not part of the clinical text
    for item in items:
        assert "•" not in item.title and "•" not in item.detail
    assert items[0].detail == "Monitor complete blood count prior to start of therapy."


@pytest.mark.unit
def test_a_section_header_numbered_with_a_period_is_still_stripped():
    """Found on the live everolimus label. The strip required '<digits><space>', so
    '2. DOSAGE AND ADMINISTRATION' kept its header and rendered it as clinical text:
    the first dosing item read '2. DOSAGE AND ADMINISTRATION Do not combine ...'.
    A section heading presented as label guidance is invented content."""
    text = "2. DOSAGE AND ADMINISTRATION Do not combine the two forms. ( 2.1 )"
    items = parse_label_considerations(text, DOSAGE_SECTION)
    assert len(items) == 1, [(i.title, i.detail) for i in items]
    assert items[0].detail == "Do not combine the two forms."
    assert "DOSAGE AND ADMINISTRATION" not in items[0].detail
    assert "DOSAGE AND ADMINISTRATION" not in items[0].title


@pytest.mark.unit
def test_the_undecidable_inline_reference_is_bounded_to_mis_segmentation():
    """codex iter-7 HIGH + MEDIUM, characterised rather than "fixed".

    An inline reference naming its own section, after a sentence end and before a
    capital, is byte-identical to a real terminal citation — olaparib's live label
    writes exactly this shape as a genuine one. So this input IS split into two
    considerations, and no rule can tell it apart without dropping olaparib's real
    warnings, which is what requiring the spaced form was measured to do.

    What this test pins is the BOUND, which is the part that must never regress: the
    damage stays mis-segmentation. Every rendered detail is a contiguous verbatim run
    of the section, and no item ever carries words from a bullet on the far side of
    another item's citation. A split is recoverable by an analyst opening the PI; a
    merge that puts one warning's words under another warning's citation is not.

    codex's MEDIUM was the load-bearing half: the verbatim + adjacency audit CANNOT
    catch this, because the mis-split detail is verbatim and its reference genuinely
    is adjacent. The module docstring no longer claims otherwise.
    """
    section_text = (
        "A: Monitor renal function. (5.1) Patients with renal impairment should "
        "interrupt therapy. ( 5.2 )"
    )
    items = parse_label_considerations(
        f"5 WARNINGS AND PRECAUTIONS {section_text}", WARNINGS_SECTION
    )
    assert items, "the bullet must not vanish"
    for item in items:
        assert item.detail in section_text, f"not a contiguous verbatim run: {item.detail!r}"
    # No merge: no item may hold text from both sides of the inner reference.
    for item in items:
        assert not (
            "Monitor renal function" in item.detail and "interrupt therapy" in item.detail
        ), (item.title, item.references, item.detail)


@pytest.mark.unit
def test_a_range_form_reference_terminates_its_bullet_instead_of_merging():
    """codex iter-8 HIGH. `_CANDIDATE` matched comma lists but not ranges, so a
    terminal "( 5.1-5.3 )" was invisible BOTH as a boundary and to the self-reference
    invariant that backstops it — the one combination that still produced a MERGE,
    which is the failure class this module refuses. Bullet A's words came out inside
    bullet B under B's citation.

    Not found on any of the 20 live labels (0 range references across 60 sections),
    but widening what counts as a candidate is additive: it can only make a reference
    visible, never hide one, which is why it is safe where the spaced-form rule was
    not.
    """
    text = "5 WARNINGS AND PRECAUTIONS A: Keep this. ( 5.1-5.3 ) B: Monitor next. ( 5.4 )"
    items = parse_label_considerations(text, WARNINGS_SECTION)
    assert [(i.title, i.references) for i in items] == [("A", "5.1-5.3"), ("B", "5.4")], [
        (i.title, i.references, i.detail) for i in items
    ]
    for item in items:
        assert not ("Keep this" in item.detail and "Monitor next" in item.detail)


@pytest.mark.unit
def test_a_bullet_whose_period_follows_its_citation_is_still_read():
    """codex iter-9 MEDIUM, and it is a whole CONVENTION we were blind to.

    Our three curated brands write '... regularly thereafter. ( 5.1 )'. Plenty of
    labels write '... regularly thereafter ( 5.1 ).' with the sentence period AFTER
    the citation — and for those, `_BOUNDARY_AFTER` saw '. •' and read prose while
    the ends-with-a-period rule rejected the body, so the section parsed to NOTHING.

    Measured across 28 live labels: 10 sections returned zero items purely because of
    this, including every section of ivosidenib and atorvastatin and both of
    spironolactone's. The text below is spironolactone's real Highlights.
    """
    text = (
        "5 WARNINGS AND PRECAUTIONS "
        "• Hyperkalemia: Monitor serum potassium within one week of initiation and "
        "regularly thereafter ( 5.1 ). "
        "• Hypotension and Worsening Renal Function: Monitor volume status and renal "
        "function periodically ( 5.2 )."
    )
    items = parse_label_considerations(text, WARNINGS_SECTION)
    assert [(i.title, i.references) for i in items] == [
        ("Hyperkalemia", "5.1"),
        ("Hypotension and Worsening Renal Function", "5.2"),
    ], [(i.title, i.references, i.detail) for i in items]
    assert items[0].detail == (
        "Monitor serum potassium within one week of initiation and regularly thereafter"
    )


@pytest.mark.unit
def test_an_unrecognised_reference_separator_drops_the_bullet_instead_of_merging():
    """codex iter-9 HIGH, and the finding is really about MY BACKSTOP.

    `_swallowed_a_boundary` was described as independent defence in depth, but it
    located internal references with `_CANDIDATE` — the very pattern that failed to
    recognise the form. A backstop sharing the blind spot of the thing it backstops
    is not a backstop, so every separator `_CANDIDATE` missed ('; ', ' and ', '/',
    square brackets, '5.1.1') produced a MERGE.

    It now scans with a deliberately LOOSE pattern of its own: any bracketed group
    containing a number that names this section. Loose is the correct bias there
    because the guard can only ever DROP, never emit.
    """
    for terminal in ("( 5.1; 5.3 )", "( 5.1 and 5.3 )", "( 5.1/5.3 )", "( 5.1.1 )", "[5.1]"):
        text = f"5 WARNINGS AND PRECAUTIONS A: Keep this. {terminal} B: Monitor next. ( 5.4 )"
        items = parse_label_considerations(text, WARNINGS_SECTION)
        for item in items:
            assert not ("Keep this" in item.detail and "Monitor next" in item.detail), (
                terminal,
                item.title,
                item.references,
                item.detail,
            )


@pytest.mark.unit
def test_a_parenthetical_of_prose_is_not_mistaken_for_a_reference_group():
    """Measured cost of making the backstop loose, caught on a real label.

    Alpelisib's dosing bullet says "Pediatric patients (2 to less than 18 years of
    age): 50 mg ...". That is an AGE RANGE, but it is a bracketed group containing
    "2", which names the dosage section — so the loose guard read it as a swallowed
    boundary and dropped a real, correctly-parsed bullet.

    "Loose is safe because it can only drop" was wrong: dropping real label content is
    a cost, not a free failure. The guard now fires only on groups holding NOTHING but
    reference numbers and separators, which still covers every separator codex found
    ('; ', ' and ', '/', brackets, '5.1.1') while leaving prose alone.
    """
    text = (
        "2 DOSAGE AND ADMINISTRATION Recommended Dose: Pediatric patients "
        "(2 to less than 18 years of age): 50 mg taken orally once daily with food. ( 2.1 )"
    )
    items = parse_label_considerations(text, DOSAGE_SECTION)
    assert [(i.title, i.references) for i in items] == [("Recommended Dose", "2.1")], [
        (i.title, i.references, i.detail) for i in items
    ]
    assert "2 to less than 18 years of age" in items[0].detail


@pytest.mark.unit
def test_word_separated_references_and_footnote_markers():
    """codex iter-11, two HIGH in one shape: what counts as a separator, and what
    counts as a bullet.

    `_is_reference_group` stripped only "and", so "( 5.1 or 5.2 )" read as prose, the
    guard stayed silent, and the bullets merged. And `*` was in `_BULLET_GLYPHS` on no
    evidence at all — the live survey found U+2022 and nothing else — so a statistical
    footnote "*P<0.05 versus control." was treated as the start of a bullet and its
    text became the next bullet's TITLE.
    """
    merged = parse_label_considerations(
        "5 WARNINGS AND PRECAUTIONS A: Keep this. ( 5.1 or 5.2 ) B: Monitor next. ( 5.4 )",
        WARNINGS_SECTION,
    )
    for item in merged:
        assert not ("Keep this" in item.detail and "Monitor next" in item.detail), (
            item.title,
            item.references,
            item.detail,
        )

    # codex iter-12 HIGH: this half USED to be "for item in footnoted: assert
    # 'P<0.05' not in item.title" — and the fixture emits ZERO items, so it passed
    # without ever exercising the thing it named. Vacuous in the exact shape this
    # file keeps getting caught by: asserting an absence over an empty collection.
    footnoted = parse_label_considerations(
        "5 WARNINGS AND PRECAUTIONS A: In Study 1 (N=123), response was assessed. (5.1) "
        "*P<0.05 versus control. B: Monitor next. ( 5.2 )",
        WARNINGS_SECTION,
    )
    # POSITIVE CONTROL first: the same text WITHOUT the footnote must parse, or the
    # assertions below prove nothing about footnotes.
    clean = parse_label_considerations(
        "5 WARNINGS AND PRECAUTIONS A: In Study 1 (N=123), response was assessed. (5.1) "
        "B: Monitor next. ( 5.2 )",
        WARNINGS_SECTION,
    )
    assert [(i.title, i.references) for i in clean] == [("A", "5.1"), ("B", "5.2")], [
        (i.title, i.references) for i in clean
    ]
    # With the footnote present the honest outcome is that BOTH bullets are dropped:
    # '*' is no longer a delimiter, so '(5.1)' is not a boundary, the bullets merge,
    # and the self-reference guard drops the merge. Stated plainly rather than left
    # to an assertion that cannot fail. Corrupting B's title with footnote text —
    # which is what recognising '*' would do — is the outcome being refused.
    assert footnoted == (), [(i.title, i.references, i.detail) for i in footnoted]


@pytest.mark.unit
def test_a_see_prefixed_reference_drops_rather_than_merges():
    """codex iter-12 HIGH. Real SPL writes "(see 5.1)". `_CANDIDATE` cannot match it
    (the word), and `_is_reference_group` rejected it as prose (alphabetic), so the
    guard stayed silent and the bullets MERGED — the one failure this module refuses.

    "see" joins "and"/"or" as a word the guard looks past. Note the asymmetry that is
    deliberate: the guard sees through it, `_CANDIDATE` still does not, so the bullet
    DROPS rather than being attributed to a reference whose form we have never
    verified against a real label. Under-reporting over invention.
    """
    items = parse_label_considerations(
        "5 WARNINGS AND PRECAUTIONS A: Keep this. (see 5.1) B: Monitor next. ( 5.4 )",
        WARNINGS_SECTION,
    )
    for item in items:
        assert not ("Keep this" in item.detail and "Monitor next" in item.detail), (
            item.title,
            item.references,
            item.detail,
        )


@pytest.mark.unit
def test_word_separated_references_drop_rather_than_merge_or_invent():
    """The full arc of one decision, because the middle of it shipped.

    iter-11: adding "or" to the GUARD stopped "( 5.1 or 5.2 )" from merging, and the
    section then parsed to nothing.
    iter-12 (codex MEDIUM): I called that under-reporting and widened `_CANDIDATE` so
    the reference would be READ.
    iter-13 (codex HIGH): that widening read "occurred in 5 or 6 patients ( 5 or 6 )"
    as a bullet citing "label 5 or 6".

    The measurement that settles it, which belonged at the start: word-separated
    TERMINAL references occur ZERO times across 28 live labels / 82 sections. I had
    added support for a form no label has been observed to use, on the strength of a
    constructed example, and it admitted prose.

    So the honest behaviour is the one pinned here: such a group never becomes a
    citation, and the guard still looks past its words so it can never cause a MERGE.
    The bullets are lost. Losing a bullet whose reference form we cannot verify beats
    inventing a citation for it.
    """
    items = parse_label_considerations(
        "5 WARNINGS AND PRECAUTIONS A: Keep this. ( 5.1 or 5.2 ) B: Monitor next. ( 5.4 )",
        WARNINGS_SECTION,
    )
    for item in items:
        assert not ("Keep this" in item.detail and "Monitor next" in item.detail), (
            item.title,
            item.references,
            item.detail,
        )
    assert "5.1 or 5.2" not in [i.references for i in items]


@pytest.mark.unit
def test_a_bracketed_self_reference_with_prose_in_it_still_drops():
    """codex iter-13 HIGH. "(5.1, Table 1)" carries a self-reference AND a word, so
    `_CANDIDATE` skipped it and `_is_reference_group` rejected it as prose — the last
    combination that still MERGED.

    The guard now fires on either shape: a group that is nothing but references, OR a
    group containing a DOTTED "N.M" whose section is this one. Measured against 28
    live labels, the second rule is free: all five bracketed groups carrying both a
    dotted number and words are prose — "(0.1 mL of 150 mg/mL solution)", "(eGFR
    below 30 mL/min/1.73 m 2 )" — and not one of their dotted numbers names its own
    section. A bare integer stays ambiguous with prose quantities, which is why it
    still requires the pure-reference form; that is what spares alpelisib's
    "(2 to less than 18 years of age)".
    """
    items = parse_label_considerations(
        "5 WARNINGS AND PRECAUTIONS A: Keep this. (5.1, Table 1) B: Monitor next. ( 5.4 )",
        WARNINGS_SECTION,
    )
    for item in items:
        assert not ("Keep this" in item.detail and "Monitor next" in item.detail), (
            item.title,
            item.references,
            item.detail,
        )
    # and the real-label prose shapes must still parse
    kept = parse_label_considerations(
        "2 DOSAGE AND ADMINISTRATION Dose: Inject 0.1 mL of solution "
        "(0.1 mL of 150 mg/mL solution) once daily. ( 2.1 )",
        DOSAGE_SECTION,
    )
    assert [(i.title, i.references) for i in kept] == [("Dose", "2.1")], [
        (i.title, i.references, i.detail) for i in kept
    ]


@pytest.mark.unit
def test_prose_numbers_joined_by_a_word_are_not_a_citation():
    """codex iter-13 HIGH, and this one was MY REGRESSION from the round before.

    Iteration 12 widened `_CANDIDATE` to accept "and"/"or" separators so that
    "( 5.1 or 5.2 )" would be READ rather than merely not merged. It then read
    "Adverse reactions occurred in 5 or 6 patients ( 5 or 6 )" as a bullet citing
    "label 5 or 6".

    The measurement I should have run first: word-separated TERMINAL references occur
    ZERO times across 28 live labels / 82 sections. I added support for a form I have
    never observed and it admitted prose. Reverted — the guard still looks past
    "and"/"or" so such a group cannot cause a MERGE, but nothing is ever attributed
    to a reference form no label has been seen to use.
    """
    items = parse_label_considerations(
        "5 WARNINGS AND PRECAUTIONS Risk: Adverse reactions occurred in 5 or 6 patients "
        "( 5 or 6 ).",
        WARNINGS_SECTION,
    )
    assert [i.references for i in items] == [], [(i.title, i.references) for i in items]
