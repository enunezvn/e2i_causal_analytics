"""#1691: the deterministic superlative-vs-table guard, pinned to the real defect texts.

Every fixture under tests/fixtures/synthesis_guard/ is a VERBATIM chat response
from the two 2026-08-18 51-turn eval runs (morning baseline + post-#1690 rerun).
The positive fixtures are the measured #1691 instances; the negative fixtures
are the exact responses that produced false positives during guard development
— each suppression rule in synthesis_guard.py exists because one of these
fired wrongly. Keep asserting the MEASURED numbers (0.198 vs 0.267, 0.231 vs
0.224, 30 vs 18), never just "some finding exists": the historical texts are
the instrument that proves the guard both fires and stays quiet correctly.
"""

from pathlib import Path

import pytest

from src.api.routes.synthesis_guard import (
    build_superlative_correction,
    find_superlative_contradictions,
)

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "synthesis_guard"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text()


def _visible(text: str):
    return [f for f in find_superlative_contradictions(text) if f.visible]


# ---------------------------------------------------------------------------
# The three measured #1691 instances MUST be detected, on the exact numbers.
# ---------------------------------------------------------------------------


def test_morning_5_7_false_largest_effect_size_is_detected():
    """'largest 0.198' printed directly under a 0.267 row (the canonical case)."""
    findings = _visible(_load("morning_5_7_false_largest.md"))
    assert len(findings) == 1
    f = findings[0]
    assert f.keyword == "largest"
    assert f.value == pytest.approx(0.198)
    assert f.column_header == "Effect Size"
    assert f.column_max == pytest.approx(0.267)


def test_morning_5_6_false_lowest_heterogeneity_is_detected():
    """'the lowest (0.231)' with 0.224 in the same Heterogeneity Score column."""
    findings = _visible(_load("morning_5_6_false_lowest.md"))
    assert len(findings) == 1
    f = findings[0]
    assert f.keyword == "lowest"
    assert f.value == pytest.approx(0.231)
    assert f.column_min == pytest.approx(0.224)


def test_rerun_6_4_false_fastest_lag_is_detected():
    """'fastest-materializing (30-day lag)' over a table whose Lag column holds 18."""
    findings = _visible(_load("rerun_6_4_false_fastest.md"))
    assert len(findings) == 1
    f = findings[0]
    assert f.keyword == "fastest"
    assert f.value == pytest.approx(30)
    assert f.column_header == "Lag"
    assert f.column_min == pytest.approx(18)


def test_correction_note_names_the_contradiction():
    note = build_superlative_correction(_load("morning_5_7_false_largest.md"))
    assert "0.198" in note
    assert "0.267" in note
    assert "Effect Size" in note
    assert "table" in note.lower()


# ---------------------------------------------------------------------------
# Clean real responses MUST stay silent — each of these produced a false
# positive during development; the rule that fixed it is named in the test.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture",
    [
        # Correct '4th-strongest of 6' ordinal over a full six-value list.
        "rerun_5_7_correct_ordinal.md",
        # 'X (+0.41) is the largest, Y (+0.27) second' — depth-0 comma split
        # must stop the +0.27 from pairing with 'largest'.
        "rerun_2_4_parallel_ranking.md",
        # 'Largest negative driver: … -0.073' — a negative column-min
        # satisfies a max-superlative.
        "morning_6_1_negative_driver.md",
        # 'largest share (1,238)' is true among buckets; the 2,341 Total row
        # must be excluded from the column.
        "morning_2_2_total_row.md",
        # "isn't the region's strongest lever" — negated claims are skipped.
        "rerun_6_2_negated.md",
        # '(Kisqali, 2026-07-28, lowest)' paren annotations pair BACKWARD;
        # the real defect here (wrong-row attribution at the column max) is a
        # documented non-goal, so the whole text must be silent.
        "morning_5_1_row_attribution_miss.md",
        # Correct claims throughout (champion-restore turn).
        "rerun_6_1_correct_claims.md",
    ],
)
def test_clean_real_responses_are_silent(fixture):
    assert _visible(_load(fixture)) == []


def test_scoped_claim_is_log_only_never_visible():
    """'largest absolute volume opportunity … on propensity' narrows to a row
    subset the column check cannot see: allowed at log tier, never visible."""
    text = _load("rerun_2_5_scoped_and_annotations.md")
    assert _visible(text) == []
    assert build_superlative_correction(text) == ""


def test_rerun_3_3_same_paren_cross_quantity_annotation_is_silent():
    """#1701 (the guard's first post-sweep FP): '(highest propensity, n=1,016)'
    — the superlative names its own quantity ("propensity", whose 58.4% IS the
    column max) while n= labels a different axis of the same row. The prose is
    TRUE; no visible finding and no correction note may be produced."""
    text = _load("rerun_3_3_same_paren_annotation.md")
    assert _visible(text) == []
    assert build_superlative_correction(text) == ""


# ---------------------------------------------------------------------------
# Focused synthetic cases: one per suppression/detection rule, so each rule
# survives independently of the fixture texts.
# ---------------------------------------------------------------------------

_MINI_TABLE = "| Driver | Effect |\n|---|---|\n| A | 0.10 |\n| B | 0.20 |\n| C | 0.30 |\n"


def test_interior_value_with_max_word_fires():
    text = _MINI_TABLE + "\nDriver B has the largest effect (0.20) here.\n"
    findings = _visible(text)
    assert len(findings) == 1
    assert findings[0].value == pytest.approx(0.20)
    assert findings[0].column_max == pytest.approx(0.30)


def test_true_extremum_is_silent():
    text = _MINI_TABLE + "\nDriver C has the largest effect (0.30) here.\n"
    assert find_superlative_contradictions(text) == []


def test_ordinal_prefix_is_skipped():
    text = _MINI_TABLE + "\nDriver B is the second-largest effect (0.20).\n"
    assert find_superlative_contradictions(text) == []


def test_number_absent_from_every_table_is_skipped():
    text = _MINI_TABLE + "\nThe largest gap is 0.55 between segments.\n"
    assert find_superlative_contradictions(text) == []


def test_no_table_means_no_findings():
    assert find_superlative_contradictions("The largest value is 0.20.") == []
    assert build_superlative_correction("The largest value is 0.20.") == ""


def test_restrictive_scope_demotes_to_log_only():
    text = _MINI_TABLE + "\nB is the largest effect (0.20) among rep-attributable drivers.\n"
    findings = find_superlative_contradictions(text)
    assert len(findings) == 1
    assert findings[0].visible is False
    assert build_superlative_correction(text) == ""


def test_deictic_scope_stays_visible():
    text = _MINI_TABLE + "\nB carries the largest effect (0.20) among these three.\n"
    findings = _visible(text)
    assert len(findings) == 1


# #1701 focused cases: same-parenthetical cross-quantity annotation, and the
# single-letter-header demotion on backward pairs.

_SEGMENT_TABLE = (
    "| Segment | Propensity | n |\n"
    "|---|---|---|\n"
    "| A | 58.4% | 1,016 |\n"
    "| B | 51.1% | 478 |\n"
    "| C | 34.9% | 1,662 |\n"
    "| D | 34.5% | 156 |\n"
)


def test_same_paren_cross_quantity_annotation_does_not_pair():
    """#1701: inside one parenthetical, 'highest' names "propensity" while the
    number is introduced by its own label 'n=' — the annotation names its own
    axis and must not pair with the superlative (not even at log tier)."""
    text = _SEGMENT_TABLE + "\nSegment A leads (highest propensity, n=1,016) here.\n"
    assert find_superlative_contradictions(text) == []


def test_same_paren_matching_label_still_pairs():
    """Counter-control for the #1701 rule: when the label IS the quantity the
    superlative names ('highest n=478'), the pair must survive and fire."""
    text = _SEGMENT_TABLE + "\nSegment B stands out (highest n=478) here.\n"
    findings = _visible(text)
    assert len(findings) == 1
    assert findings[0].value == pytest.approx(478)
    assert findings[0].column_max == pytest.approx(1662)


def test_backward_pair_single_letter_header_is_log_only():
    """#1701 mechanism 2: a single-letter header ('n') can never satisfy the
    backward visibility gate — such pairs stay log-only, never user-visible."""
    text = _SEGMENT_TABLE + "\nSegment B sits at 478 (highest) in this table.\n"
    findings = find_superlative_contradictions(text)
    assert len(findings) == 1
    assert findings[0].column_header == "n"
    assert findings[0].visible is False
    assert build_superlative_correction(text) == ""
