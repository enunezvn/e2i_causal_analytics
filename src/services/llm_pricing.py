"""Read-time LLM pricing (admin observability, spec 2026-07-12).

No cost is ever baked into llm_usage_events rows — consumers call cost_usd()
at read time so pricing corrections apply retroactively. Unknown models
return None (rendered as 'unpriced'), never a silent default.

Rate-table pattern borrowed from src/mlops/agent_cost_tracker.MODEL_PRICING
(which stays untouched — it is an unwired per-agent budget scaffold on a
different axis).
"""

from typing import Optional

# Bump whenever rates change; surfaced in the API payload for provenance.
PRICING_VERSION = "2026-07-12"

# USD per 1M tokens: normalized model key -> (input_rate, output_rate).
# Keys are prefixes: resolve_pricing_key strips provider prefixes and picks
# the LONGEST matching key, so dated variants (claude-haiku-4-5-20251001)
# resolve and gpt-4o-mini never falls into gpt-4o.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-5": (5.00, 25.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

_KEYS_LONGEST_FIRST = sorted(MODEL_PRICING, key=len, reverse=True)

_PROVIDER_PREFIXES = ("anthropic", "openai", "azure")


def normalize_model(model: str) -> str:
    """Strip provider prefixes: 'anthropic/x', 'openai:y' -> bare model id."""
    m = (model or "").strip().lower()
    for sep in ("/", ":"):
        if sep in m:
            head, tail = m.split(sep, 1)
            if head in _PROVIDER_PREFIXES:
                m = tail
    return m


def resolve_pricing_key(model: str) -> Optional[str]:
    m = normalize_model(model)
    for key in _KEYS_LONGEST_FIRST:
        if m == key or m.startswith(key + "-"):
            return key
    return None


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """USD cost, or None for unknown models — callers surface those as
    'unpriced' rather than costing them at a fabricated rate."""
    key = resolve_pricing_key(model)
    if key is None:
        return None
    in_rate, out_rate = MODEL_PRICING[key]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
