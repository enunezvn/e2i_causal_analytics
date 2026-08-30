"""#1850: ``_strip_segment_suffix`` must be qualifier-aware.

Since #1835 (PR #1838) ``render_action`` emits ``… in {segment_value} (benchmark-driven |
top-decile target | restore prior performance)`` for vs_benchmark / vs_potential /
temporal gaps, so the segment is no longer the LAST thing in the action and the
``$``-anchored strip was a no-op: the live Fabhalta brief read "in south within south"
three times and the real segment name survived as a prose alias the token-index
attribution check cannot see.

The fixtures here are RENDERED by ``render_action`` (and the trailing-form template set
is derived from the template tables themselves) so the two modules cannot drift apart
again without this file going red.
"""

from __future__ import annotations

import re

import pytest

from src.agents.gap_analyzer.action_templates import (
    BRAND_TEMPLATES,
    GAP_TYPE_SUFFIXES,
    NEUTRAL_TEMPLATES,
    render_action,
)
from src.insights.executive_brief import (
    _lm_opportunity_line,
    _strip_segment_suffix,
    build_grounding,
)

SEGMENT = "south"
_SEG_TOKEN_RE = re.compile(r"\{SEG_(\d+)\}")

# Every gap_type the templates know a suffix for, plus the two no-suffix paths
# (vs_target has no entry; an unknown gap_type falls open to no suffix).
GAP_TYPES = tuple(GAP_TYPE_SUFFIXES) + ("vs_target", "unknown_gap_type")


def _trailing_forms() -> list[tuple[str, str | None, str]]:
    """(metric, brand, difficulty) for every template whose prose ENDS with
    ``in {segment_value}`` — the only shape the strip owns (mid-sentence
    mentions are deliberately left alone, see the pre-existing test)."""
    forms: list[tuple[str, str | None, str]] = []
    for brand, table in (("Fabhalta", BRAND_TEMPLATES), (None, NEUTRAL_TEMPLATES)):
        for metric, by_difficulty in table.items():
            for difficulty, template in by_difficulty.items():
                if template.endswith("in {segment_value}"):
                    forms.append((metric, brand, difficulty))
    return forms


TRAILING_FORMS = _trailing_forms()


def _render(metric: str, brand: str | None, difficulty: str, gap_type: str) -> str:
    return render_action(
        metric=metric,
        difficulty=difficulty,
        segment="region",
        segment_value=SEGMENT,
        gap_type=gap_type,
        brand=brand,
    )


def _ids() -> list[str]:
    return [f"{m}-{b or 'neutral'}-{d}" for m, b, d in TRAILING_FORMS]


# ---- Positive controls: the derived fixture set is not vacuous -------------------


def test_trailing_forms_cover_the_live_evidence_shapes():
    # The two shapes measured live on 2026-08-30 (Fabhalta / Kisqali #1 actions):
    # both are market_share brand templates; if a template rewrite ever moves the
    # segment mid-sentence this positive control fails instead of the suite going
    # silently vacuous.
    assert TRAILING_FORMS, "no template ends with 'in {segment_value}' — fixture is vacuous"
    assert ("market_share", "Fabhalta", "medium") in TRAILING_FORMS
    assert ("market_share", "Fabhalta", "high") in TRAILING_FORMS
    assert len(TRAILING_FORMS) >= 10


def test_every_suffix_gap_type_renders_a_parenthesised_qualifier():
    # The strip's qualifier branch assumes "(…)" at end-of-string; pin the
    # template contract it relies on.
    for gap_type, suffix in GAP_TYPE_SUFFIXES.items():
        assert suffix.startswith(" (") and suffix.endswith(")"), (gap_type, suffix)
        assert _render("market_share", "Fabhalta", "medium", gap_type).endswith(suffix)


# ---- (a)+(b): rendered trailing forms x every gap_type ---------------------------


@pytest.mark.parametrize("gap_type", GAP_TYPES)
@pytest.mark.parametrize(("metric", "brand", "difficulty"), TRAILING_FORMS, ids=_ids())
def test_strip_removes_segment_and_keeps_qualifier(metric, brand, difficulty, gap_type):
    action = _render(metric, brand, difficulty, gap_type)
    assert SEGMENT in action.lower()  # the rendered fixture really names the segment
    stripped = _strip_segment_suffix(action, SEGMENT)
    # (a) the segment is gone — no prose alias survives into the LM-facing line.
    assert SEGMENT not in stripped.lower(), stripped
    suffix = GAP_TYPE_SUFFIXES.get(gap_type, "")
    if suffix:
        # (b) the gap-type qualifier is preserved, attached to the action.
        assert stripped.endswith(suffix.strip()), stripped
        assert not stripped.endswith("  " + suffix.strip()), stripped  # single space
    else:
        assert not stripped.endswith(")"), stripped
    # The prose before "in <segment>" is untouched.
    assert stripped.startswith(action.split(f" in {SEGMENT}")[0]), stripped
    assert stripped == stripped.strip()


# ---- (c): the LM line names the segment exactly once, as a token ------------------


@pytest.mark.parametrize("gap_type", GAP_TYPES)
def test_lm_line_has_one_seg_token_and_no_literal_segment(gap_type):
    action = _render("market_share", "Fabhalta", "medium", gap_type)
    line = _lm_opportunity_line(
        1,
        {
            "recommended_action": action,
            "segment_value": SEGMENT,
            "gap_metric": "market_share",
            "implementation_difficulty": "medium",
        },
    )
    assert _SEG_TOKEN_RE.findall(line) == ["1"], line
    assert SEGMENT not in line.lower(), line
    suffix = GAP_TYPE_SUFFIXES.get(gap_type, "")
    expected_tail = f"among hematologists/internists{suffix} in {{SEG_1}} — "
    assert expected_tail in line, line


def test_build_grounding_lm_opportunities_keep_qualifier_before_seg_token():
    # End-to-end through build_grounding for the live #1 Fabhalta action.
    action = _render("market_share", "Fabhalta", "medium", "vs_benchmark")
    g = build_grounding(
        brand="Fabhalta",
        total_addressable_value=5_400_000.0,
        quick_wins_count=1,
        steady_plays_count=0,
        strategic_bets_count=0,
        suppressed_count=0,
        opportunities=[
            {
                "rank": 1,
                "recommended_action": action,
                "expected_roi": 21.72,
                "revenue_impact": 2_271_986.0,
                "gap_metric": "market_share",
                "gap_percentage": 16.7,
                "segment_value": SEGMENT,
                "implementation_difficulty": "medium",
            }
        ],
    )
    lm = g["lm_opportunities"]
    assert "(benchmark-driven) in {SEG_1}" in lm, lm
    assert "in south" not in lm.lower(), lm
    assert g["injection"]["{SEG_1}"] == SEGMENT


# ---- (d): guards unchanged — short segment, case-insensitive, "the", whitespace ----


def test_short_segment_guard_still_applies_with_qualifier():
    action = "Expand coverage in NY (benchmark-driven)"
    assert _strip_segment_suffix(action, "NY") == action


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (
            "Boost engagement in South (benchmark-driven)",
            "Boost engagement (benchmark-driven)",
        ),
        (
            "Boost engagement in the South (top-decile target)",
            "Boost engagement (top-decile target)",
        ),
        (
            "Boost engagement in SOUTH   (restore prior performance)   ",
            "Boost engagement (restore prior performance)",
        ),
        # A qualifier with nested parenthesised text survives intact.
        (
            "Boost engagement in south (restore prior (Q2) performance)",
            "Boost engagement (restore prior (Q2) performance)",
        ),
        # Pre-#1835 shapes keep working exactly as before.
        ("Boost engagement in the south", "Boost engagement"),
        ("Boost engagement in south  ", "Boost engagement"),
    ],
)
def test_case_article_and_whitespace_variants(action, expected):
    assert _strip_segment_suffix(action, "south") == expected


# ---- (e): actions that do not end with "in <segment> [(…)]" are untouched --------


@pytest.mark.parametrize(
    "action",
    [
        "Deploy field triggers",
        "Deploy field triggers (benchmark-driven)",
        "Invest in south channels now",
        "Invest in south channels now (benchmark-driven)",
        # A trailing qualifier alone, segment mid-sentence: not the strip's shape.
        "Launch targeted sampling campaign in south to drive TRx growth (benchmark-driven)",
        # Segment followed by prose then a paren: not end-of-string qualifier.
        "Boost engagement in south (benchmark-driven) next quarter",
        # A different segment.
        "Boost engagement in west (benchmark-driven)",
    ],
)
def test_non_matching_actions_untouched(action):
    assert _strip_segment_suffix(action, "south") == action
