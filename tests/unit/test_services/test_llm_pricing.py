"""llm_pricing: read-time cost from tokens x static rates; unknown models
return None (surfaced as 'unpriced'), NEVER a default cost."""

from src.services.llm_pricing import (
    PRICING_VERSION,
    cost_usd,
    normalize_model,
    resolve_pricing_key,
)


def test_known_model_cost_per_million():
    # claude-sonnet-4-6: $3/1M in, $15/1M out
    assert cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000) == 18.0


def test_small_call_cost():
    # 1000 in + 500 out on gpt-4o-mini: 1000*0.15/1M + 500*0.60/1M
    assert cost_usd("gpt-4o-mini", 1000, 500) == (1000 * 0.15 + 500 * 0.60) / 1_000_000


def test_dated_variant_resolves_to_base_key():
    assert resolve_pricing_key("claude-haiku-4-5-20251001") == "claude-haiku-4-5"


def test_provider_prefixes_stripped():
    assert normalize_model("anthropic/claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert normalize_model("openai/gpt-4o") == "gpt-4o"
    # copilotkit metadata format uses a colon
    assert normalize_model("anthropic:claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_gpt4o_mini_does_not_match_gpt4o():
    assert resolve_pricing_key("gpt-4o-mini") == "gpt-4o-mini"
    assert resolve_pricing_key("gpt-4o") == "gpt-4o"


def test_model_refresh_2026_07_18_entries():
    # claude-sonnet-5: $3/1M in, $15/1M out (list price, not the intro rate)
    assert cost_usd("claude-sonnet-5", 1_000_000, 1_000_000) == 18.0
    # gpt-5.6 family: sol $5/$30, terra $2.50/$15, luna $1/$6
    assert cost_usd("gpt-5.6-sol", 1_000_000, 1_000_000) == 35.0
    assert cost_usd("gpt-5.6-terra", 1_000_000, 1_000_000) == 17.5
    assert cost_usd("gpt-5.6-luna", 1_000_000, 1_000_000) == 7.0
    assert cost_usd("claude-opus-4-8", 1_000_000, 1_000_000) == 30.0


def test_gpt56_variants_resolve_distinctly():
    assert resolve_pricing_key("gpt-5.6-terra") == "gpt-5.6-terra"
    assert resolve_pricing_key("gpt-5.6-luna") == "gpt-5.6-luna"
    # Dated snapshots (as recorded by llm_usage_events) resolve to the base key.
    assert resolve_pricing_key("gpt-5.6-terra-2026-06-01") == "gpt-5.6-terra"
    assert resolve_pricing_key("claude-sonnet-5-20260601") == "claude-sonnet-5"


def test_superseded_models_stay_priced_for_historical_rows():
    # Read-time pricing: rows recorded before the refresh must still resolve.
    assert cost_usd("gpt-4o-2024-08-06", 1000, 500) is not None
    assert cost_usd("claude-sonnet-4-6", 1000, 500) is not None
    assert cost_usd("claude-opus-4-5-20251101", 1000, 500) is not None


def test_unknown_model_returns_none_not_zero():
    assert cost_usd("mistral-large", 1000, 1000) is None
    assert resolve_pricing_key("mistral-large") is None


def test_zero_tokens_known_model_is_zero_cost():
    assert cost_usd("gpt-4o", 0, 0) == 0.0


def test_pricing_version_is_a_date_string():
    assert len(PRICING_VERSION) == 10  # YYYY-MM-DD
