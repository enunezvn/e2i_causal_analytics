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


# ---------------------------------------------------------------------------
# #1717: the em-dash blind spot found by positive control in the 2026-08-19
# full eval (turn 4.1). '**X number N** — the largest ... (out of M total ...)'
# severed N from the superlative, then bound "largest" to the no-column total
# M and silently dropped the claim — a FALSE version of the shape shipped
# uncorrected. full_4_1_true_emdash.md is the VERBATIM 4.1 response; the
# falsified fixture is the n3n4 grader's controlled mutation (claim rewritten
# to name neurology at 27, the HCPs column MINIMUM, table unchanged).
# ---------------------------------------------------------------------------


def test_full_4_1_falsified_emdash_split_is_detected():
    """#1717 red-first: the grader's falsification of 4.1 — column minimum 27
    claimed as 'largest' in the em-dash split shape — must fire a visible
    finding (pre-fix the guard returned 0 findings on this exact text)."""
    findings = _visible(_load("full_4_1_falsified_emdash.md"))
    assert len(findings) == 1
    f = findings[0]
    assert f.keyword == "largest"
    assert f.value == pytest.approx(27)
    assert f.column_header == "HCPs"
    assert f.column_max == pytest.approx(256)


def test_full_4_1_true_emdash_text_is_silent():
    """The verbatim 4.1 response: 256 IS the HCPs column max, so the same
    shape carrying a true claim stays silent at every tier."""
    text = _load("full_4_1_true_emdash.md")
    assert find_superlative_contradictions(text) == []
    assert build_superlative_correction(text) == ""


# Focused synthetic cases for the two #1717 mechanisms, independent of the
# fixture texts: (a) a dash right after a closing-bold subject must not sever
# it from the following superlative clause; (b) a forward pair bound to a
# number appearing in no table column falls back to the preceding bolded
# number instead of dropping the claim.


@pytest.mark.parametrize("dash", ["—", "–", "-"])
def test_bold_subject_dash_superlative_falsehood_fires(dash):
    text = (
        _MINI_TABLE + f"\n**Driver A stands at 0.10 units** {dash} the largest effect "
        "in the set (out of 3 total drivers).\n"
    )
    findings = _visible(text)
    assert len(findings) == 1
    assert findings[0].value == pytest.approx(0.10)
    assert findings[0].column_max == pytest.approx(0.30)


@pytest.mark.parametrize("dash", ["—", "–", "-"])
def test_bold_subject_dash_superlative_true_claim_is_silent(dash):
    text = (
        _MINI_TABLE + f"\n**Driver C stands at 0.30 units** {dash} the largest effect "
        "in the set (out of 3 total drivers).\n"
    )
    assert find_superlative_contradictions(text) == []


# The #1717 sweep measured two new FPs caused by the dash unsplitting — both
# TRUE claims whose superlative names an INVERSE quantity of the paired
# column ("largest engagement shortfall — 63.0% achievement" and "the largest
# target miss" beside 73.9%, each the Achievement column MINIMUM). Such
# claims are direction-ambiguous: only a strictly interior value fires.


@pytest.mark.parametrize(
    "fixture",
    [
        # perf 1.7: '**Northeast has the largest engagement shortfall** —
        # 63.0% achievement' — 63.0 IS the Achievement column minimum.
        "perf_1_7_inverse_shortfall.md",
        # perf 6.5: '**… 73.9% (August)** — the largest target miss of any
        # region/period shown' — 73.9 IS the Achievement column minimum.
        "perf_6_5_inverse_target_miss.md",
    ],
)
def test_inverse_quantity_corpus_texts_stay_silent(fixture):
    """No visible tier, no note, and the inverse 'largest' claims absent at
    EVERY tier. (1.7 also carries a pre-existing log-only 'highest engagement
    ROI … among the three regions shown' finding — the documented
    restrictive-scope demotion, identical pre/post #1717 — so all-tier
    emptiness is asserted per-keyword, not globally.)"""
    text = _load(fixture)
    findings = find_superlative_contradictions(text)
    assert all(f.keyword != "largest" for f in findings)
    assert _visible(text) == []
    assert build_superlative_correction(text) == ""


_ACHIEVEMENT_TABLE = (
    "| Region | Achievement |\n|---|---|\n| NE | 63.0% |\n| MW | 71.8% |\n| S | 107.6% |\n"
)


def test_inverse_noun_at_column_minimum_is_silent():
    text = (
        _ACHIEVEMENT_TABLE
        + "\n**NE has the largest engagement shortfall** — 63.0% achievement against target.\n"
    )
    assert find_superlative_contradictions(text) == []


def test_inverse_noun_interior_value_fires_with_span_note():
    """An interior value contradicts EVERY reading of 'largest shortfall', so
    the claim still fires — and the note states the span, not a direction."""
    text = (
        _ACHIEVEMENT_TABLE
        + "\n**MW has the largest engagement shortfall** — 71.8% achievement against target.\n"
    )
    findings = _visible(text)
    assert len(findings) == 1
    assert findings[0].value == pytest.approx(71.8)
    assert findings[0].inverse_axis is True
    note = build_superlative_correction(text)
    assert "spans" in note
    assert "largest value is actually" not in note
