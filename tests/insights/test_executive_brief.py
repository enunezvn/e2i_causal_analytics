from types import SimpleNamespace

import pytest

from src.insights.executive_brief import (
    ExecutiveBriefInsightSignature,
    _fallback,
    _inject,
    _placeholder_violation,
    _strip_segment_suffix,
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


_CLINICAL = (
    "Clinical setting for Remibrutinib: BTK inhibitor, indicated for chronic "
    "spontaneous urticaria. Key competitors (curated reference): Xolair; Dupixent."
)


@pytest.mark.skipif(ExecutiveBriefInsightSignature is None, reason="DSPy not installed in this env")
def test_signature_mandates_one_clinical_sentence_when_available_and_omits_otherwise():
    # The user's directive (2026-07-12): the brief must VISIBLY reflect clinical
    # context, not merely have it fed as optional LM color. The instruction must
    # REQUIRE exactly one clinical-setting sentence when a setting is provided,
    # and FORBID inventing one when the brand has no clinical context.
    instr = (ExecutiveBriefInsightSignature.instructions or "").lower()
    assert "clinical" in instr
    assert "exactly one sentence" in instr  # a mandate, not "you may"
    assert "omit" in instr  # never fabricated when none is available


def test_fallback_surfaces_clinical_context_when_available():
    # The deterministic (no-LM) path must ALSO carry the clinical setting, so the
    # brief reflects it even when the LM is unavailable/guard-rejected.
    g = _grounding(clinical_context=_CLINICAL)
    assert g["has_clinical_context"] is True
    out = _fallback(g)
    assert "BTK inhibitor" in out["insight"]
    assert any(c["label"] == "Clinical context" for c in out["grounding"])


# ---- Placeholder inputs + injection map ------------------------------------------


def test_lm_inputs_carry_tokens_not_figures():
    # The LM must never see a real figure: every number in the LM-facing
    # inputs is a placeholder token the server later substitutes.
    g = _grounding()
    lm_text = " ".join([g["lm_scope"], g["lm_opportunities"], g["lm_caveats"]])
    for real in ("$5.0M", "3.2", "$1.2M", "42%", "2.1", "$300K", "18%"):
        assert real not in lm_text
    for token in ("{TOTAL}", "{QUICK}", "{ROI_1}", "{IMPACT_1}", "{GAP_1}", "{SEG_1}"):
        assert token in lm_text
    assert "42% TRX gap" not in g["lm_opportunities"]
    assert "{GAP_1} TRX gap" in g["lm_opportunities"]
    assert "{SUPPRESSED} low-value opportunities were suppressed" in g["lm_caveats"]


def test_injection_map_binds_tokens_to_grounded_values():
    g = _grounding()
    inj = g["injection"]
    assert inj["{TOTAL}"] == "$5.0M"
    assert inj["{QUICK}"] == "2"
    assert inj["{STEADY}"] == "1"
    assert inj["{BETS}"] == "1"
    assert inj["{SUPPRESSED}"] == "3"
    assert inj["{ROI_1}"] == "3.2x"
    assert inj["{IMPACT_1}"] == "$1.2M"
    assert inj["{GAP_1}"] == "42%"
    assert inj["{SEG_1}"] == "Northeast"
    assert inj["{ROI_2}"] == "2.1x"
    assert inj["{SEG_2}"] == "South"


def test_injection_token_index_is_position_not_feed_rank():
    # Duplicate feed ranks must not collide two opportunities onto one token.
    opps = [
        _opportunity(rank=7, segment_value="Alpha"),
        _opportunity(rank=7, segment_value="Beta", expected_roi=1.5),
    ]
    g = _grounding(opportunities=opps)
    assert g["injection"]["{SEG_1}"] == "Alpha"
    assert g["injection"]["{SEG_2}"] == "Beta"
    assert "{ROI_2}" in g["lm_opportunities"]


def test_suppressed_token_absent_when_nothing_suppressed():
    g = _grounding(suppressed_count=0)
    assert "{SUPPRESSED}" not in g["injection"]
    assert "{SUPPRESSED}" not in g["lm_caveats"]


def test_strip_segment_suffix_removes_trailing_segment_alias():
    # The LM line appends "in {SEG_n}" itself; a real segment name left in the
    # action prose would read twice and bypass the token-index check.
    assert (
        _strip_segment_suffix("Boost the HCP engagement program in the west", "west")
        == "Boost the HCP engagement program"
    )
    assert _strip_segment_suffix("Boost engagement in West", "west") == "Boost engagement"
    # Mid-sentence mentions and non-matching tails are left alone.
    assert _strip_segment_suffix("Invest in west channels now", "west") == (
        "Invest in west channels now"
    )
    assert _strip_segment_suffix("Deploy field triggers", "west") == "Deploy field triggers"


def test_lm_opportunity_line_places_segment_token():
    g = _grounding(
        opportunities=[
            _opportunity(
                recommended_action="Boost the HCP engagement program in the Northeast",
            )
        ]
    )
    line = g["lm_opportunities"]
    assert "in the Northeast" not in line
    assert "Boost the HCP engagement program in {SEG_1}" in line


# ---- Placeholder contract validation ---------------------------------------------

_VOCAB = {"{TOTAL}", "{QUICK}", "{ROI_1}", "{IMPACT_1}", "{SEG_1}", "{ROI_2}", "{SEG_2}"}


def test_placeholder_violation_passes_clean_token_prose():
    assert (
        _placeholder_violation(
            "Lead with {SEG_1}: {ROI_1} ROI on {IMPACT_1}, within a {TOTAL} portfolio.",
            _VOCAB,
        )
        is None
    )
    # Multi-opportunity comparisons are the point of the design — segment
    # tokens grouped in one sentence are fine, as are figure-free sentences.
    assert _placeholder_violation("Then expand from {SEG_1} into {SEG_2}.", _VOCAB) is None
    assert _placeholder_violation("No figures here at all.", _VOCAB) is None
    # Metric tokens with no segment token in the sentence are unambiguous by
    # construction (the token itself names the rank).
    assert _placeholder_violation("Returns run from {ROI_2} up to {ROI_1}.", _VOCAB) is None


def test_placeholder_violation_rejects_leaked_digits():
    assert _placeholder_violation("Expect a 6.2x return on {IMPACT_1}.", _VOCAB) is not None
    assert _placeholder_violation("Roughly $500K of upside.", _VOCAB) is not None
    # Malformed/lowercased tokens leave their index digits behind — trapped by
    # the same digit check.
    assert _placeholder_violation("Lead with {roi_1} returns.", _VOCAB) is not None
    assert _placeholder_violation("Lead with {ROI_1 returns.", _VOCAB) is not None


def test_placeholder_violation_rejects_non_decimal_numeric_glyphs():
    # codex PR-1153 round 1 HIGH: \d only matches Unicode DECIMAL digits, so
    # circled/superscript/Roman/fraction glyphs would render as figures.
    assert _placeholder_violation("Lead with {SEG_1} at ②x ROI on {IMPACT_1}.", _VOCAB) is not None
    assert _placeholder_violation("A ²x return awaits.", _VOCAB) is not None
    assert _placeholder_violation("Phase Ⅲ expansion follows.", _VOCAB) is not None
    assert _placeholder_violation("Capture ½ the market.", _VOCAB) is not None


def test_placeholder_violation_rejects_unknown_tokens():
    assert _placeholder_violation("Expect {ROI_9} returns.", _VOCAB) is not None
    assert _placeholder_violation("A {MADEUP} figure.", _VOCAB) is not None


def test_placeholder_violation_rejects_cross_index_seg_metric_pairing():
    # "{SEG_2} yields {ROI_1}" re-attributes rank 1's figure to rank 2's
    # segment even though both values are real — exact index arithmetic, no
    # English parsing.
    assert _placeholder_violation("Then {SEG_2} yields {ROI_1}.", _VOCAB) is not None
    assert _placeholder_violation("Then {SEG_2} yields {ROI_2}.", _VOCAB) is None
    # The pairing unit is the sentence: the same tokens in separate sentences
    # claim no relationship.
    assert _placeholder_violation("Start with {SEG_2}. Best return is {ROI_1}.", _VOCAB) is None
    # A comparison naming both segments may cite both figures.
    assert _placeholder_violation("{SEG_1} at {ROI_1} outpaces {SEG_2} at {ROI_2}.", _VOCAB) is None


def test_inject_substitutes_all_tokens_in_one_pass():
    inj = {"{ROI_1}": "3.2x", "{SEG_1}": "Northeast", "{TOTAL}": "$5.0M"}
    assert (
        _inject("Lead with {SEG_1} at {ROI_1} inside {TOTAL}.", inj)
        == "Lead with Northeast at 3.2x inside $5.0M."
    )


# ---- generate_insight: contract enforcement end-to-end ----------------------------


def _pred(interpretation, takeaways=()):
    return SimpleNamespace(interpretation=interpretation, key_takeaways=list(takeaways))


def test_generate_insight_injects_real_values_into_compliant_output(monkeypatch):
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred(
            "Lead with {SEG_1}: {ROI_1} ROI on {IMPACT_1} closing a {GAP_1} gap. "
            "Then {SEG_2} follows at {ROI_2}.",
            ["{SEG_1} first ({IMPACT_1} at {ROI_1})", "Portfolio totals {TOTAL}"],
        ),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is False
    assert "Lead with Northeast: 3.2x ROI on $1.2M closing a 42% gap." in out["insight"]
    assert "Then South follows at 2.1x." in out["insight"]
    assert out["key_takeaways"] == ["Northeast first ($1.2M at 3.2x)", "Portfolio totals $5.0M"]
    assert "{" not in out["insight"]


def test_generate_insight_rejects_leaked_digits_and_falls_back(monkeypatch):
    # A digit outside a token is by definition not server-injected — the exact
    # fabrication class the old numeric guard patrolled with English parsing.
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred("Invest now: the portfolio is worth $9.9M at 8x returns."),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "$9.9M" not in out["insight"]


def test_generate_insight_rejects_unknown_tokens(monkeypatch):
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred("Expect {ROI_9} returns on {IMPACT_1}."),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "{ROI_9}" not in out["insight"]


def test_generate_insight_rejects_cross_index_attribution(monkeypatch):
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred("Capture {IMPACT_1} at {ROI_1} in {SEG_2}."),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is True


def test_generate_insight_rejects_violating_takeaway_even_with_clean_body(monkeypatch):
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred(
            "Lead with {SEG_1} at {ROI_1}.",
            ["Expect $4.4M incremental revenue"],
        ),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert all("$4.4M" not in t for t in out["key_takeaways"])


def test_generate_insight_retries_once_with_a_fresh_sample(monkeypatch):
    # One rejected draw triggers exactly one fresh-sample retry; the compliant
    # second draw renders. lm_cache must be False on every attempt or the
    # in-memory DSPy cache would replay the rejected completion.
    g = _grounding()
    calls = []

    def _fake(sig, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _pred("A leaked 8x figure.")
        return _pred("Lead with {SEG_1} at {ROI_1}.")

    monkeypatch.setattr("src.insights.executive_brief.run_signature", _fake)
    out = generate_insight(g)
    assert out["is_fallback"] is False
    assert out["insight"] == "Lead with Northeast at 3.2x."
    assert len(calls) == 2
    assert all(k["lm_cache"] is False for k in calls)


def test_generate_insight_falls_back_after_two_bad_samples(monkeypatch):
    g = _grounding()
    calls = []

    def _fake(sig, **kwargs):
        calls.append(kwargs)
        return _pred("Always an 8x leak.")

    monkeypatch.setattr("src.insights.executive_brief.run_signature", _fake)
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert len(calls) == 2


def test_generate_insight_falls_back_without_retry_when_lm_unavailable(monkeypatch):
    # None means the LM itself failed (no key / provider error) — a second
    # immediate call would fail the same way.
    g = _grounding()
    calls = []

    def _fake(sig, **kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr("src.insights.executive_brief.run_signature", _fake)
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert len(calls) == 1


def test_generate_insight_treats_empty_interpretation_as_violation(monkeypatch):
    g = _grounding()
    monkeypatch.setattr(
        "src.insights.executive_brief.run_signature",
        lambda *a, **k: _pred(""),
    )
    out = generate_insight(g)
    assert out["is_fallback"] is True


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
    assert "x" * 161 not in g["lm_opportunities"]
    assert "y" * 61 not in g["injection"]["{SEG_1}"]


# ---- Causal-registry levers (commercial grain, 2026-07-07) ----------------------
CAUSAL_LEVERS = [
    "rep detailing frequency → TRx volume",
    "patient persistence → TRx volume",
    "copay support program → ROI",
]


def test_build_grounding_carries_digit_free_causal_levers():
    """The brief's placeholder guard fails closed on ANY numeric character in
    LM output, so the lever context the LM sees must be digit-free by
    construction — names only, no effects, no confidences."""
    g = _grounding(causal_drivers=CAUSAL_LEVERS)
    assert "rep detailing frequency" in g["lm_causal_context"]
    assert not any(ch.isnumeric() for ch in g["lm_causal_context"]), g["lm_causal_context"]
    # Display variant matches (no separate figure-bearing channel to drift).
    assert g["causal_context"] == g["lm_causal_context"]
    assert any(c["label"] == "Causal levers" for c in g["grounding"])


def test_build_grounding_filters_digit_bearing_lever_defensively():
    g = _grounding(causal_drivers=["persistent_180d → trx_volume", *CAUSAL_LEVERS])
    assert "180d" not in g["lm_causal_context"]
    assert not any(ch.isnumeric() for ch in g["lm_causal_context"])


def test_build_grounding_without_levers_says_none_and_no_chip():
    g = _grounding()
    assert "no modeled causal levers" in g["lm_causal_context"].lower()
    assert not any(c["label"] == "Causal levers" for c in g["grounding"])


def test_fallback_appends_causal_levers_when_present():
    g = _grounding(causal_drivers=CAUSAL_LEVERS)
    out = generate_insight(g)  # LM off in tests -> deterministic fallback
    assert out["is_fallback"] is True
    assert "rep detailing frequency" in out["insight"]


# ---------------------------------------------------------------------------
# clinical_context (2026-07-12: commercial outputs don't happen in a clinical
# vacuum — the brief cites the brand's clinical setting qualitatively)
# ---------------------------------------------------------------------------

CLINICAL_CONTEXT = (
    "Clinical setting for Remibrutinib (remibrutinib): Bruton tyrosine kinase "
    "(BTK) inhibitor, indicated for chronic spontaneous urticaria. Key "
    "competitors (curated reference): Xolair (omalizumab). Context from public "
    "biomedical and regulatory sources; reference qualitatively."
)


def test_build_grounding_carries_digit_free_clinical_context():
    """Same contract as the causal levers: the placeholder guard fails closed
    on ANY numeric character in LM output, so the clinical setting the LM sees
    must be digit-free by construction, identical on both channels."""
    g = _grounding(clinical_context=CLINICAL_CONTEXT)
    assert "BTK" in g["lm_clinical_context"]
    assert not any(ch.isnumeric() for ch in g["lm_clinical_context"]), g["lm_clinical_context"]
    assert g["clinical_context"] == g["lm_clinical_context"]
    assert any(c["label"] == "Clinical context" for c in g["grounding"])


def test_build_grounding_drops_digit_bearing_clinical_context_defensively():
    g = _grounding(clinical_context="Indicated for patients 12 years of age and older.")
    assert "12" not in g["lm_clinical_context"]
    assert "no clinical context" in g["lm_clinical_context"].lower()
    assert not any(c["label"] == "Clinical context" for c in g["grounding"])


def test_build_grounding_without_clinical_context_says_none():
    g = _grounding()
    assert "no clinical context" in g["lm_clinical_context"].lower()
    assert not any(ch.isnumeric() for ch in g["lm_clinical_context"])


def test_fallback_appends_clinical_context_when_present(monkeypatch):
    monkeypatch.setattr("src.insights.executive_brief.run_signature", lambda *a, **k: None)
    g = _grounding(clinical_context=CLINICAL_CONTEXT)
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "BTK" in out["insight"]


def test_generate_insight_passes_clinical_context_to_the_lm(monkeypatch):
    captured = {}

    def _capture(sig, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr("src.insights.executive_brief.run_signature", _capture)
    g = _grounding(clinical_context=CLINICAL_CONTEXT)
    generate_insight(g)
    assert captured.get("clinical_context") == CLINICAL_CONTEXT
