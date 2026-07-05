from src.insights.executive_brief import _fallback, build_grounding, generate_insight


def _opportunity(**overrides):
    o = {
        "rank": 1,
        "recommended_action": "Deploy field triggers to lapsed writers",
        "expected_roi": 3.2,
        "revenue_impact": 1_200_000.0,
        "gap_metric": "trx",
        "gap_percentage": 42.0,
        "segment_value": "Northeast",
        "implementation_difficulty": "medium",
    }
    o.update(overrides)
    return o


def _grounding(**overrides):
    kwargs = {
        "brand": "Remibrutinib",
        "total_addressable_value": 5_000_000.0,
        "quick_wins_count": 2,
        "steady_plays_count": 1,
        "strategic_bets_count": 1,
        "suppressed_count": 3,
        "opportunities": [
            _opportunity(),
            _opportunity(
                rank=2,
                recommended_action="Rebalance speaker programs toward high-decile HCPs",
                expected_roi=2.1,
                revenue_impact=300_000.0,
                gap_metric="nbrx",
                gap_percentage=18.0,
                segment_value="South",
                implementation_difficulty="low",
            ),
        ],
    }
    kwargs.update(overrides)
    return build_grounding(**kwargs)


def test_build_grounding_derives_scope_opportunities_and_chips():
    g = _grounding()
    assert "Remibrutinib" in g["scope"]
    assert "$5.0M" in g["scope"]
    assert "2 quick win(s)" in g["scope"]
    assert "3.2x ROI" in g["opportunities"]
    assert "$1.2M revenue impact" in g["opportunities"]
    assert "42% TRX gap in Northeast" in g["opportunities"]
    assert "medium effort" in g["opportunities"]
    assert "3 low-value opportunities were suppressed" in g["caveats"]
    assert any(c["label"] == "Brand" and c["value"] == "Remibrutinib" for c in g["grounding"])
    assert any(c["label"] == "Addressable value" and c["value"] == "$5.0M" for c in g["grounding"])
    assert any(c["label"] == "Top ROI" and c["value"] == "3.2x" for c in g["grounding"])
    assert any(c["label"] == "Suppressed" and c["value"] == "3" for c in g["grounding"])
    assert g["has_signal"] is True


def test_opportunities_are_ordered_by_rank_and_capped_at_five():
    opps = [_opportunity(rank=r, recommended_action=f"Action {r}") for r in (4, 2, 6, 1, 3, 5)]
    g = _grounding(opportunities=opps)
    text = g["opportunities"]
    assert text.index("Action 1") < text.index("Action 2") < text.index("Action 3")
    assert "Action 6" not in text  # rank 6 falls outside the top-5 cap


def test_all_suppressed_is_real_signal_not_silence():
    # Mirrors the T6 gap-analyzer honest narrative: everything below break-even
    # is a "don't invest now" brief, not an empty state.
    g = _grounding(opportunities=[], suppressed_count=4)
    assert g["has_signal"] is True
    assert "below the break-even threshold" in g["opportunities"]
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "break-even" in out["insight"]


def test_no_signal_yields_honest_run_a_gap_analysis_fallback():
    g = _grounding(opportunities=[], suppressed_count=0)
    assert g["has_signal"] is False
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "run a gap analysis" in out["insight"].lower()
    assert out["key_takeaways"] == []


def test_generate_insight_fallback_surfaces_real_figures():
    # No LM configured in the test env -> deterministic factual fallback built
    # verbatim from the grounded figures, never fabrication.
    g = _grounding()
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "$5.0M" in out["insight"]
    assert "3.2x ROI" in out["insight"]
    assert "validate them before committing budget" in out["insight"]


def test_fallback_states_suppression_caveat():
    out = _fallback(_grounding(suppressed_count=1))
    assert "1 low-value opportunity was suppressed" in out["insight"]


def test_verbose_free_text_is_bounded():
    g = _grounding(
        opportunities=[
            _opportunity(
                recommended_action="x" * 500,
                segment_value="y" * 200,
            )
        ]
    )
    # One verbose opportunity must not blow up the prompt: action capped at
    # 160, segment at 60 (mirrors the frontend brief-request bounds).
    assert "x" * 161 not in g["opportunities"]
    assert "y" * 61 not in g["opportunities"]
