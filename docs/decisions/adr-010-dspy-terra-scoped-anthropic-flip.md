# ADR-010: DSPy default → `openai/gpt-5.6-terra`; scoped Anthropic flip for the factory lanes

**Date**: 2026-07-18 | **Status**: Accepted | **Implemented by**: PRs #1275, #1278

## Context

The DSPy reasoning path (orchestrator/RAG/insights — the platform's highest-volume LLM surface, ~650 calls/week) still defaulted to 2024-era `gpt-4o`. After the refresh moved it to `gpt-5.6-terra` (PR #1275), live probing found terra had a provider-side transient 401 flake (~15–30% in the affected window). Measured impact split by lane:

- **DSPy lane**: unaffected in practice — dspy/litellm's default `num_retries=3` absorbed every observed flake (10/10 pass while raw no-retry calls failed 3/10 in the same window).
- **LangChain OpenAI lane** (`get_chat_llm` standard/reasoning): **exposed** — the OpenAI SDK does not retry 401.

An interleaved prod-container disproof through the real factory constructors measured: claude-sonnet-5 13/13 and claude-haiku-4-5 10/10 clean vs terra 7/10 (three live 401s in-window); latency penalty for realistic synthesis calls was negligible. The copilot chat path was already pinned `provider="anthropic"` with zero errors over the same period.

## Decision

1. **DSPy default**: `get_default_dspy_model()` → `openai/{LLM_MODEL}`, falling back to `openai/gpt-5.6-terra`. The GEPA-optimized prompts were tuned on OpenAI models — the DSPy lane does **not** move providers on vibes; any future move requires the golden-set A/B eval (plan exists, PROPOSED).
2. **Scoped provider flip** for everything else: `LLM_PROVIDER=anthropic`, so the factory lanes run ChatAnthropic (`claude-sonnet-5` standard/reasoning, `claude-haiku-4-5` fast).
3. **The pin that makes the flip safe**: `DSPY_LM_MODEL=openai/gpt-5.6-terra` is set explicitly and honored first. Without it, the flip would drag DSPy onto `anthropic/{ANTHROPIC_MODEL}` (= opus-4-8 on the droplet) where litellm's temperature default would 400.
4. **Env plumbing**: compose `x-common-env` forwards `LLM_PROVIDER` / `LLM_MODEL` / `DSPY_LM_MODEL` into api + workers (PR #1278). `ANTHROPIC_MODEL` is deliberately NOT forwarded — the host pins opus-4-8 for interactive use; in-code defaults are current.

## Consequences

- (+) The exposed LangChain lane is off the flaky model entirely; the flaky-but-retried DSPy lane keeps its GEPA-tuned model.
- (+) Cost roughly at parity (sonnet-5 $3/$15 vs terra $2.50/$15 per Mtok; haiku ≈ luna).
- (+) `llm_usage_events` rows now split by design: factory → anthropic rows, DSPy phases → terra rows.
- (−) Two providers are both load-bearing; both API keys are required in production.
- (−) The DSPy lane's provider question is only deferred — deciding it properly needs the golden-set A/B.

## References

- ADR-009 — the tier system this builds on
- `docs/LLM_CONFIGURATION.md`
