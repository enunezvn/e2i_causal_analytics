# ADR-009: Central LLM factory with model tiers; July 2026 model refresh

**Date**: 2026-07-18 | **Status**: Accepted | **Implemented by**: PRs #1274, #1276

## Context

Model IDs were pinned ad hoc across dozens of source and config files. When providers retired model generations, those pins became dead IDs that returned live 404s in production — the exact failure class PR #1276 swept. Two further constraints arrived with the 2025/2026 model generations: several models (claude-sonnet-5, claude-opus-4-8, all gpt-5*) reject the `temperature` parameter with HTTP 400, and gpt-5.x models default to spending reasoning tokens, which starves small `max_tokens` responses on latency-sensitive paths.

## Decision

1. **Single source of truth**: `src/utils/llm_factory.py` `MODEL_MAPPINGS` maps each provider to three tiers — `fast`, `standard`, `reasoning`:
   - Anthropic: `claude-haiku-4-5` (fast), `claude-sonnet-5` (standard/reasoning)
   - OpenAI: `gpt-5.6-luna` (fast), `gpt-5.6-terra` (standard/reasoning)
2. **Env override**: `LLM_MODEL` overrides the OpenAI workhorse model without a code change (must be forwarded by the compose `x-common-env` whitelist — see Consequences).
3. **Parameter compatibility is the factory's job**: `_supports_temperature()` drops `temperature` for models that reject it; the fast tier pins `reasoning_effort="none"` so gpt-5.x fast calls don't burn their token budget on reasoning.
4. **Pricing versioned alongside**: `src/services/llm_pricing.py` carries a dated `PRICING_VERSION` so every live model ID resolves to a cost for `llm_usage_events` rows.
5. **Dead-ID sweep policy**: scattered pins outside the factory are swapped to *active, temperature-tolerant* IDs (`claude-sonnet-4-6`, `claude-haiku-4-5`) as pure 404-proofing. Latest-generation upgrades happen only in the factory — call sites that pass `temperature` must not be pointed at models that reject it.

## Consequences

- (+) Model upgrades are a one-file change; dead-ID 404s are structurally prevented at the call sites that matter.
- (+) Per-call cost attribution works for every live ID (no unknown-model fallbacks).
- (−) Changing a src default requires sweeping default-coupled test assertions (CI caught `test_graphiti_config.py`).
- (−) An env-read in src is dead in production until `docker/docker-compose.yml` `x-common-env` forwards it — this bit `LLM_MODEL`/`LLM_PROVIDER` (third instance of the never-forwarded-env gap; fixed in PR #1278, see ADR-010).

## References

- `docs/LLM_CONFIGURATION.md` — operator-facing configuration guide
- ADR-010 — provider split that builds on these tiers
