"""#1863: the gap metric must read as prose, never as a raw enum token.

``gap_metric`` arrives as a KPI name ("market_share", "trx"); uppercasing it
verbatim put "closing a 21% MARKET_SHARE gap" in front of the LM — which
echoed it into executive prose live on every brand (2026-08-31). Humanize it
before either path sees it: industry casing for Rx counts (TRx/NRx/NBRx),
hyphenated lowercase otherwise. A digit-bearing metric name is omitted from
the LM-facing line only — the placeholder guard fails closed on any digit in
LM output (same rationale as the causal-lever filter), so feeding one would
silently poison every sample into fallback; the deterministic fallback line
has no such guard and keeps it.
"""

from src.insights.executive_brief import _humanize_metric, build_grounding


def _opportunity(**overrides):
    o = {
        "rank": 1,
        "recommended_action": "Deploy triggered engagement plays",
        "expected_roi": 3.2,
        "revenue_impact": 1_200_000.0,
        "gap_metric": "market_share",
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
        "suppressed_count": 0,
        "opportunities": [_opportunity()],
    }
    kwargs.update(overrides)
    return build_grounding(**kwargs)


def test_humanize_metric_casing_table():
    assert _humanize_metric("market_share") == "market-share"
    assert _humanize_metric("conversion_rate") == "conversion-rate"
    assert _humanize_metric("trx") == "TRx"
    assert _humanize_metric("nrx") == "NRx"
    assert _humanize_metric("nbrx") == "NBRx"
    assert _humanize_metric("TRX") == "TRx"  # case-insensitive lookup
    assert _humanize_metric("engagement") == "engagement"
    assert _humanize_metric("") == "—"


def test_fallback_line_reads_humanized_metric():
    g = _grounding()
    assert "42% market-share gap in Northeast" in g["opportunities"]
    assert "MARKET_SHARE" not in g["opportunities"]


def test_lm_line_reads_humanized_metric():
    g = _grounding()
    assert "{GAP_1} market-share gap" in g["lm_opportunities"]
    assert "MARKET_SHARE" not in g["lm_opportunities"]


def test_trx_keeps_industry_casing_in_both_paths():
    g = _grounding(opportunities=[_opportunity(gap_metric="trx")])
    assert "42% TRx gap in Northeast" in g["opportunities"]
    assert "{GAP_1} TRx gap" in g["lm_opportunities"]


def test_digit_bearing_metric_is_omitted_from_lm_line_but_kept_in_fallback():
    g = _grounding(opportunities=[_opportunity(gap_metric="persistence_180d")])
    # LM line: no digits may enter the prompt vocabulary via the metric — the
    # guard would reject every echo. The gap keeps its placeholder and reads
    # as a plain "gap".
    assert "persistence-180d" not in g["lm_opportunities"]
    assert "{GAP_1} gap" in g["lm_opportunities"]
    # Deterministic fallback line: guard-free, keeps the humanized name.
    assert "42% persistence-180d gap in Northeast" in g["opportunities"]
