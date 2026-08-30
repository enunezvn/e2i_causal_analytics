"""#1856: same-valued ``{SEG_n}`` enumeration runs must collapse at injection.

By design (#1850/#1854) the LM sees segment identity only through per-
opportunity ``{SEG_n}`` tokens — never the values — so when several ranked
opportunities share one ``segment_value`` the LM has no way to know that
"across {SEG_2}, {SEG_3}, and {SEG_4}" will expand to the same word three
times ("across south, south, and south", live on prod post-#1854). Prompt
guidance cannot fix a fact the model cannot observe; the deterministic seam
is the server-side substitution step, which collapses a same-valued run to
one mention plus a count of the distinct opportunities it stands for.

Attribution stays intact: ``_placeholder_violation`` validates the RAW token
text before any collapse, and every surviving mention still names the exact
value each collapsed opportunity carries.
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
        brand="Fabhalta",
        total_addressable_value=5_000_000.0,
        quick_wins_count=2,
        steady_plays_count=1,
        strategic_bets_count=1,
        suppressed_count=0,
        opportunities=[_opportunity(i, seg) for i, seg in enumerate(segments, start=1)],
    )


def _pred(interpretation, takeaways=()):
    return SimpleNamespace(interpretation=interpretation, key_takeaways=list(takeaways))


_INJ = {
    "{SEG_1}": "south",
    "{SEG_2}": "south",
    "{SEG_3}": "south",
    "{SEG_4}": "south",
    "{SEG_5}": "west",
}


# ---- the collapse helper on token text --------------------------------------------


def test_three_token_same_value_oxford_run_collapses_with_count():
    assert eb._collapse_same_value_seg_runs(
        "Use a staged sequence across {SEG_2}, {SEG_3}, and {SEG_4} rather than "
        "treating every opportunity as equally ready.",
        _INJ,
    ) == (
        "Use a staged sequence across {SEG_2} (three initiatives) rather than "
        "treating every opportunity as equally ready."
    )


def test_two_token_and_run_collapses():
    assert (
        eb._collapse_same_value_seg_runs("Sequence penetration across {SEG_1} and {SEG_2}.", _INJ)
        == "Sequence penetration across {SEG_1} (two initiatives)."
    )


def test_no_oxford_comma_run_collapses():
    assert (
        eb._collapse_same_value_seg_runs("in {SEG_1}, {SEG_2} and {SEG_3} now", _INJ)
        == "in {SEG_1} (three initiatives) now"
    )


def test_or_separated_run_collapses():
    assert (
        eb._collapse_same_value_seg_runs("either {SEG_1} or {SEG_2}", _INJ)
        == "either {SEG_1} (two initiatives)"
    )


def test_distinct_values_run_is_untouched():
    # Healthy enumerations over different segments are GOOD prose — the
    # collapse must never fire on them.
    text = "across {SEG_1}, {SEG_2}, and {SEG_5}"
    assert eb._collapse_same_value_seg_runs(text, _INJ) == text


def test_mixed_value_run_is_untouched():
    # Partial doubling ("south, south, and west") is left alone: the collapse
    # owns only the all-identical class #1856 certified live.
    text = "across {SEG_3}, {SEG_4}, and {SEG_5}"
    assert eb._collapse_same_value_seg_runs(text, _INJ) == text


def test_repeated_same_index_collapses_without_count():
    # {SEG_1} twice is ONE opportunity stuttered, not two — a "(two
    # initiatives)" annotation would be a fabricated count.
    assert eb._collapse_same_value_seg_runs("in {SEG_1} and {SEG_1}", _INJ) == "in {SEG_1}"


def test_count_is_distinct_indices_not_run_length():
    assert (
        eb._collapse_same_value_seg_runs("{SEG_1}, {SEG_2}, and {SEG_1}", _INJ)
        == "{SEG_1} (two initiatives)"
    )


def test_unresolved_token_leaves_run_untouched():
    text = "across {SEG_1} and {SEG_9}"
    assert eb._collapse_same_value_seg_runs(text, _INJ) == text


def test_single_token_is_untouched():
    text = "concentrate on {SEG_1} first"
    assert eb._collapse_same_value_seg_runs(text, _INJ) == text


def test_four_and_five_way_runs_spell_the_count():
    assert (
        eb._collapse_same_value_seg_runs("{SEG_1}, {SEG_2}, {SEG_3}, and {SEG_4}", _INJ)
        == "{SEG_1} (four initiatives)"
    )


# ---- end-to-end through generate_insight ------------------------------------------


def test_issue_1856_fabhalta_takeaway_reads_collapsed(monkeypatch):
    g = _grounding(["south", "south", "south", "south"])
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred(
            "Lead with {SEG_1} at {ROI_1}.",
            [
                "Use a staged sequence across {SEG_2}, {SEG_3}, and {SEG_4} rather "
                "than treating every opportunity as equally ready for investment."
            ],
        ),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is False
    assert "across south (three initiatives) rather than" in out["key_takeaways"][0]
    assert "south, south" not in out["key_takeaways"][0]


def test_issue_1856_kisqali_interpretation_collapses_two_token_run(monkeypatch):
    g = _grounding(["midwest", "midwest", "midwest"])
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred(
            "Sequence penetration across {SEG_1} and {SEG_2} before broadening "
            "into competitive positioning."
        ),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is False
    assert "across midwest (two initiatives) before broadening" in out["insight"]
    assert "midwest and midwest" not in out["insight"]


def test_distinct_segment_enumeration_survives_end_to_end(monkeypatch):
    g = _grounding(["northeast", "south", "west"])
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred("Move across {SEG_1}, {SEG_2}, and {SEG_3} in rank order."),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is False
    assert "across northeast, south, and west in rank order" in out["insight"]


def test_validation_sees_raw_text_not_collapsed_text(monkeypatch):
    # "{SEG_1}, {SEG_2} … {ROI_2}" is compliant (metrics {2} ⊆ segs {1,2}) but
    # its collapsed form mentions only {SEG_1}. If the collapse ran BEFORE
    # _placeholder_violation this sample would be wrongly rejected as a
    # cross-index pairing — it must be accepted, and injected collapsed.
    g = _grounding(["south", "south"])
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred("Target {SEG_1} and {SEG_2} to capture {ROI_2} returns."),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is False
    assert "Target south (two initiatives) to capture 3.0x returns." == out["insight"]


def test_collapsed_output_carries_no_tokens_or_stray_digits(monkeypatch):
    g = _grounding(["south", "south", "south"])
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred("Invest across {SEG_1}, {SEG_2}, and {SEG_3} at {ROI_1}."),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is False
    assert "{" not in out["insight"]
    assert "Invest across south (three initiatives) at 3.5x." == out["insight"]
