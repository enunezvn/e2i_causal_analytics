from types import SimpleNamespace

from src.insights.executive_brief import (
    _count_claims,
    _fallback,
    _is_grounded,
    _numeric_claims,
    build_grounding,
    generate_insight,
)


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


def test_numeric_claims_canonicalize_across_formats():
    # The guard compares VALUES, not spellings: $5.0M == $5,000,000 == $5 million.
    assert _numeric_claims("$5.0M at 4.0x closing a 42% gap") == {
        ("money", 5_000_000.0),
        ("mult", 4.0),
        ("pct", 42.0),
    }
    assert _numeric_claims("$5,000,000 and $5 million") == {("money", 5_000_000.0)}
    # Plain counts are not money/pct/multiple claims — they are validated by
    # the SEPARATE labelled-count extractor below (codex PR-5 round 4).
    assert _numeric_claims("plain counts like 2 quick wins carry no unit") == set()


def test_count_claims_extract_labelled_portfolio_counts():
    assert _count_claims("2 quick wins, 1 steady play, 1 strategic bet") == {
        ("count:quick_win", 2.0),
        ("count:steady_play", 1.0),
        ("count:strategic_bet", 1.0),
    }
    # Words may sit between the number and its label, in either direction.
    assert _count_claims("3 low-value opportunities were suppressed") == {("count:suppressed", 3.0)}
    assert _count_claims("we suppressed 42 opportunities") == {("count:suppressed", 42.0)}
    # Digits act as barriers: one count never borrows another's label.
    assert ("count:steady_play", 2.0) not in _count_claims("2 quick win(s), 1 steady play(s)")
    # Unlabelled counts stay outside the vocabulary (prompt-governed).
    assert _count_claims("the top 3 opportunities") == set()


def test_is_grounded_rejects_invented_portfolio_counts():
    # codex PR-5 round 4 HIGH (its own repro): invented counts carried no
    # money/pct/mult claim, so the guard skipped the sentence entirely.
    sources = [
        "Kisqali / $5.0M / mix: 2 quick win(s), 1 steady play(s), 1 strategic bet(s)",
        "3 low-value opportunities were suppressed (below break-even).",
    ]
    assert (
        _is_grounded("There are 99 quick wins and 42 suppressed opportunities.", sources) is False
    )
    # Correct restatements pass, even mixing scope-counts with the caveat
    # count in one sentence (counts are one portfolio-level set, not
    # per-opportunity pairings).
    assert _is_grounded("The mix holds 2 quick wins with 3 suppressed.", sources) is True


def test_is_grounded_accepts_reformatted_and_rejects_invented_figures():
    sources = ["Kisqali / total addressable opportunity value $5.0M", "3.2x ROI, 42% TRX gap"]
    # Reformatted values from ONE unit pass; sentences with no figures pass.
    assert _is_grounded("The $5,000,000 opportunity awaits.", sources) is True
    assert _is_grounded("A 3.2x play on the 42% gap.", sources) is True
    assert _is_grounded("No figures here at all.", sources) is True
    # Invented values fail regardless of unit.
    assert _is_grounded("Expect roughly $7.5M upside", sources) is False
    assert _is_grounded("a 55% gap", sources) is False
    assert _is_grounded("at 5x returns", sources) is False


def test_is_grounded_rejects_cross_unit_pairing_within_a_sentence():
    # codex PR-5 round 2 HIGH: global value-membership would accept an LM
    # sentence that pairs unit A's dollar value with unit B's ROI — every
    # number "appears somewhere". The pairing unit is the sentence: its claims
    # must come from a SINGLE source unit.
    sources = ["Kisqali / total addressable opportunity value $5.0M", "3.2x ROI, 42% TRX gap"]
    assert _is_grounded("The $5,000,000 opportunity at 3.2x.", sources) is False
    # The same figures split across sentences each trace to one unit — fine.
    assert _is_grounded("The $5,000,000 opportunity awaits. It leads at 3.2x.", sources) is True


def test_generate_insight_rejects_llm_output_with_ungrounded_figures(monkeypatch):
    # codex PR-5 round 1 HIGH: the prompt alone must not be the only defense.
    # An LM response inventing a dollar value falls back to the factual summary.
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: SimpleNamespace(
            interpretation="Invest now: the portfolio is worth $9.9M at 8x returns.",
            key_takeaways=["Fund it"],
        ),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "$9.9M" not in out["insight"]


def test_build_grounding_binds_segment_and_metric_tokens_per_unit():
    g = _grounding()
    # scope + caveats own no attribute tokens; each opportunity owns its own.
    assert g["source_tokens"][0] == set()
    assert g["source_tokens"][1] == set()
    assert g["source_tokens"][2] == {"northeast", "trx"}
    assert g["source_tokens"][3] == {"south", "nbrx"}


def test_is_grounded_rejects_another_units_segment_in_a_numeric_sentence():
    sources = ["A — 3.2x ROI, $1.2M impact in Northeast.", "B — 2.1x ROI, $300K impact in South."]
    tokens = [{"northeast", "trx"}, {"south", "nbrx"}]
    # Figures from unit A re-attributed to unit B's segment: reject.
    assert _is_grounded("Capture $1.2M at 3.2x in South.", sources, tokens) is False
    # Same figures with their OWN segment: pass.
    assert _is_grounded("Capture $1.2M at 3.2x in Northeast.", sources, tokens) is True
    # Non-numeric prose may name any segment freely.
    assert _is_grounded("Both Northeast and South matter strategically.", sources, tokens) is True


def test_generate_insight_rejects_fabricated_portfolio_counts(monkeypatch):
    # codex PR-5 round 4 HIGH: fabricated breadth ("99 quick wins") must fall
    # back even though it carries no money/pct/multiple claim.
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: SimpleNamespace(
            interpretation="There are 99 quick wins and 42 suppressed opportunities.",
            key_takeaways=[],
        ),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "99" not in out["insight"]


def test_generate_insight_passes_correct_count_restatement(monkeypatch):
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: SimpleNamespace(
            interpretation=(
                "The portfolio holds 2 quick wins, 1 steady play, and 1 strategic "
                "bet, with 3 opportunities suppressed below break-even."
            ),
            key_takeaways=[],
        ),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is False


def test_generate_insight_rejects_segment_swapped_attribution(monkeypatch):
    # codex PR-5 round 3 HIGH: fully-grounded figures re-attributed to another
    # opportunity's segment is the same false-recommendation class as swapped
    # numbers. Must fall back.
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: SimpleNamespace(
            interpretation="Deploy field triggers in South for $1.2M at 3.2x ROI.",
            key_takeaways=[],
        ),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is True


def test_generate_insight_rejects_swapped_figure_pairing(monkeypatch):
    # codex PR-5 round 2 HIGH: attributing opportunity 2's $300K to
    # opportunity 1's action/ROI is a FALSE quantified recommendation even
    # though every number is individually grounded. Must fall back.
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: SimpleNamespace(
            interpretation="Deploy field triggers to capture $300K at 3.2x ROI.",
            key_takeaways=[],
        ),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is True


def test_generate_insight_rejects_ungrounded_takeaway_even_with_grounded_body(monkeypatch):
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: SimpleNamespace(
            interpretation="Prioritize the $1.2M Northeast TRX gap at 3.2x ROI.",
            key_takeaways=["Expect $4.4M incremental revenue"],
        ),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert all("$4.4M" not in t for t in out["key_takeaways"])


def test_generate_insight_passes_grounded_llm_output_through(monkeypatch):
    # Guardrail against over-gating: a faithful distillation (every figure
    # present in the grounding, even reformatted) renders as the real insight.
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: SimpleNamespace(
            interpretation=(
                "Deploy field triggers first: $1,200,000 at stake at 3.2x ROI "
                "closing the 42% TRX gap; then the 18% NBRX gap at 2.1x."
            ),
            key_takeaways=["Northeast first ($1.2M, 3.2x)", "South second ($300K)"],
        ),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is False
    assert "$1,200,000" in out["insight"]
    assert out["key_takeaways"] == ["Northeast first ($1.2M, 3.2x)", "South second ($300K)"]


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
