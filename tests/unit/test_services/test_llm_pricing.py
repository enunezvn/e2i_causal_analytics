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


def test_unknown_model_returns_none_not_zero():
    assert cost_usd("mistral-large", 1000, 1000) is None
    assert resolve_pricing_key("mistral-large") is None


def test_zero_tokens_known_model_is_zero_cost():
    assert cost_usd("gpt-4o", 0, 0) == 0.0


def test_pricing_version_is_a_date_string():
    assert len(PRICING_VERSION) == 10  # YYYY-MM-DD
