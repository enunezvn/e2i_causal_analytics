"""#1775 — label considerations, parsed VERBATIM from the FDA label Highlights.

The panel must ground a causal scenario in what the label actually says that
bears on the OUTCOME being analysed (why a patient stops, what gates starting).
CLAUDE.md forbids plausible-but-fake values in production paths, so every item
here is verbatim label text carrying its own cross-reference — no summarisation,
no LLM, nothing invented.

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
def test_a_prose_number_followed_by_a_capital_is_not_a_citation():
    """codex iter-3 HIGH. The positional rule alone still accepted a prose
    parenthetical that happened to be followed by a capitalised word, truncating the
    bullet and citing '(1)' as its label section."""
    text = (
        "5 WARNINGS AND PRECAUTIONS A: Assess patients (1) Patients received therapy "
        "and monitor labs. ( 5.1 )"
    )
    items = parse_label_considerations(text, WARNINGS_SECTION)
    assert len(items) == 1, [(i.title, i.detail, i.references) for i in items]
    assert items[0].references == "5.1"
    assert "Patients received therapy" in items[0].detail


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
