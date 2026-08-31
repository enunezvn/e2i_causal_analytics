"""#1862: same-valued ``{SEG_n}`` RANGE phrasings must collapse at injection.

#1856 collapsed enumeration runs joined by commas/"and"/"or", but the LM also
writes ranges — "from {SEG_1} to {SEG_2}", "from {SEG_1} through {SEG_3}" —
and those separators split the run, so the endpoints injected the same value
uncollapsed ("from midwest to midwest", "from south through south (three
initiatives)", both live on prod 2026-08-31). The LM cannot avoid this: it
never sees the values, so it cannot know a range's endpoints are equal.

Distinct-valued ranges are healthy prose ("from west to midwest") and must
survive untouched — the collapse already requires all-identical values.
"""

from types import SimpleNamespace

import src.insights.executive_brief as eb
from src.insights.executive_brief import build_grounding, generate_insight


def _opportunity(rank, segment_value, **overrides):
    o = {
        "rank": rank,
        "recommended_action": f"Deploy field triggers wave {rank}",
        "expected_roi": 4.0 - rank * 0.5,
        "revenue_impact": 1_000_000.0 / rank,
        "gap_metric": "trx",
        "gap_percentage": 40.0 - rank,
        "segment_value": segment_value,
        "implementation_difficulty": "medium",
    }
    o.update(overrides)
    return o


def _grounding(segments):
    return build_grounding(
        brand="Kisqali",
        total_addressable_value=15_000_000.0,
        quick_wins_count=0,
        steady_plays_count=1,
        strategic_bets_count=3,
        suppressed_count=0,
        opportunities=[_opportunity(i, seg) for i, seg in enumerate(segments, start=1)],
    )


def _pred(interpretation, takeaways=()):
    return SimpleNamespace(interpretation=interpretation, key_takeaways=list(takeaways))


_INJ = {
    "{SEG_1}": "midwest",
    "{SEG_2}": "midwest",
    "{SEG_3}": "midwest",
    "{SEG_4}": "west",
}


# ---- the collapse helper on token text --------------------------------------------


def test_same_value_to_range_collapses():
    assert (
        eb._collapse_same_value_seg_runs("Sequence investment from {SEG_1} to {SEG_2}.", _INJ)
        == "Sequence investment from {SEG_1} (two initiatives)."
    )


def test_same_value_through_range_collapses():
    assert (
        eb._collapse_same_value_seg_runs("expand from {SEG_1} through {SEG_3} first", _INJ)
        == "expand from {SEG_1} (two initiatives) first"
    )


def test_range_endpoint_adjoining_comma_run_collapses_as_one_run():
    # The live Fabhalta shape: "from {SEG_1} through {SEG_2}, {SEG_3}" must be
    # ONE run — post-#1856 only the comma half collapsed, leaving
    # "from south through south (three initiatives)".
    assert (
        eb._collapse_same_value_seg_runs("from {SEG_1} through {SEG_2}, {SEG_3} now", _INJ)
        == "from {SEG_1} (three initiatives) now"
    )


def test_distinct_value_range_is_untouched():
    text = "from {SEG_1} to {SEG_4}"
    assert eb._collapse_same_value_seg_runs(text, _INJ) == text


def test_stuttered_same_index_range_collapses_without_count():
    assert eb._collapse_same_value_seg_runs("from {SEG_1} to {SEG_1}", _INJ) == "from {SEG_1}"


def test_prose_to_before_single_token_is_untouched():
    # "to" preceding a LONE token is ordinary prose, not a range — no
    # token-sep-token run exists, so nothing may change.
    text = "extend positioning to {SEG_1} next quarter"
    assert eb._collapse_same_value_seg_runs(text, _INJ) == text


# ---- end-to-end through generate_insight ------------------------------------------


def test_issue_1862_kisqali_takeaway_range_reads_collapsed(monkeypatch):
    g = _grounding(["midwest", "midwest", "midwest"])
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred(
            "Lead with {SEG_1} at {ROI_1}.",
            ["Sequence investment from {SEG_1} to {SEG_2}, then broaden coverage."],
        ),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is False
    assert "from midwest (two initiatives), then broaden" in out["key_takeaways"][0]
    assert "midwest to midwest" not in out["key_takeaways"][0]


def test_distinct_range_survives_end_to_end(monkeypatch):
    g = _grounding(["midwest", "midwest", "midwest", "west"])
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred("Shift resources from {SEG_1} to {SEG_4} in rank order."),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is False
    assert "from midwest to west in rank order" in out["insight"]
