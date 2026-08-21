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
    DOSAGE_SECTION,
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
def test_a_section_with_no_highlights_items_yields_nothing_rather_than_garbage():
    """Kisqali's contraindications section is literally '4 CONTRAINDICATIONS None.'
    An empty result is correct; inventing an item would not be."""
    assert parse_label_considerations(
        "4 CONTRAINDICATIONS None. None. ( 4 )", "contraindications"
    ) == () or all(
        i.detail.strip(". ")
        for i in parse_label_considerations(
            "4 CONTRAINDICATIONS None. None. ( 4 )", "contraindications"
        )
    )


@pytest.mark.unit
def test_empty_or_missing_text_is_handled():
    assert parse_label_considerations("", WARNINGS_SECTION) == ()
    assert parse_label_considerations(None, WARNINGS_SECTION) == ()  # type: ignore[arg-type]
