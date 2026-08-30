"""#1854 — no rendered action may keep the segment value after the brief's strip.

#1851 made ``_strip_segment_suffix`` qualifier-aware, but by design it only owns
the TRAILING ``in [the] <segment> [(qualifier)]`` shape. 19 of the 36
``render_action`` template shapes named the segment MID-SENTENCE (e.g.
``"{segment_value} to close the TRx gap"``, ``"for the {segment_value}
segment"``), so the literal segment survived into the LM-facing opportunity
line: the brief narrated it twice ("…program in west … in west") and handed the
LM a prose alias the ``{SEG_n}`` token-index attribution check cannot see
(lane-1850 disproof: 165/360 rendered actions leaking, unchanged by #1851).

The fix moves every segment mention to the trailing strippable position — the
templates end with ``in {segment_value}`` and the gap-type suffix appends after,
producing exactly the ``… in <seg> (qualifier)`` shape the strip owns. This file
enumerates metric x difficulty x gap_type x brand x segment value EXHAUSTIVELY
(both template tables, both fail-open paths, multi-word and regex-metacharacter
values) so a future template that re-introduces a mid-sentence mention goes red
here before it reaches the brief.
"""

from __future__ import annotations

import pytest

from src.agents.gap_analyzer.action_templates import (
    ACTION_METRICS,
    GAP_TYPE_SUFFIXES,
    render_action,
)
from src.insights.executive_brief import _strip_segment_suffix

# Every metric with a dedicated template row plus one that falls to _default.
METRICS = tuple(ACTION_METRICS) + ("unknown_metric",)
DIFFICULTIES = ("low", "medium", "high")
# The three suffixed gap types plus the no-suffix fall-open path.
GAP_TYPES = tuple(GAP_TYPE_SUFFIXES) + ("unknown_gap_type",)
# The three real brands (BRAND_TEMPLATES), None and an unknown brand (both
# resolve to the NEUTRAL_TEMPLATES fail-open path — the neutral actions flow
# through the same brief seam).
BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali", None, "competitor")
# Live-observed single-word values, a value containing another value ("west" in
# "midwest"), a multi-word value, and a regex-metacharacter value (the strip
# must re.escape).
SEGMENT_VALUES = ("west", "midwest", "New York", "east+west")


@pytest.mark.parametrize("segment_value", SEGMENT_VALUES)
@pytest.mark.parametrize("brand", BRANDS, ids=[str(b) for b in BRANDS])
@pytest.mark.parametrize("gap_type", GAP_TYPES)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("metric", METRICS)
def test_no_segment_mention_survives_the_strip(
    metric: str,
    difficulty: str,
    gap_type: str,
    brand: str | None,
    segment_value: str,
) -> None:
    action = render_action(
        metric=metric,
        difficulty=difficulty,
        segment="region",
        segment_value=segment_value,
        gap_type=gap_type,
        brand=brand,
    )
    # Positive control: the API-facing action itself still NAMES the segment —
    # the #1854 fix is position, not deletion (GET /gaps/opportunities keeps
    # showing where to act).
    assert segment_value.lower() in action.lower(), action

    stripped = _strip_segment_suffix(action, segment_value)

    # The #1854 guarantee: after the strip, no occurrence of the segment value
    # remains — no prose alias reaches the LM beside the {SEG_n} token.
    assert segment_value.lower() not in stripped.lower(), (
        f"segment {segment_value!r} survives the strip: {stripped!r} (rendered: {action!r})"
    )

    # The gap-type qualifier survives the strip, attached to the action.
    suffix = GAP_TYPE_SUFFIXES.get(gap_type, "")
    if suffix:
        assert stripped.endswith(suffix.strip()), stripped
    else:
        assert not stripped.endswith(")"), stripped
