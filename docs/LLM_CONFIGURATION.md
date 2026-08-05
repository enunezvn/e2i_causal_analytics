# LLM Configuration

**Status**: Living reference | **Last verified against code**: 2026-07-18

How the platform selects, configures, and meters LLMs. The **code is the source
of truth** — every default below was transcribed from the files cited:

- `src/utils/llm_factory.py` — LangChain factory (tiers, provider default, overrides)
- `src/optimization/dspy_lm.py` — DSPy/litellm model resolution
- `src/services/llm_pricing.py` — read-time pricing for admin observability
- `src/utils/llm_usage_callback.py` — per-call usage capture

---

## TL;DR

- **The default provider is OpenAI**, not Anthropic (`llm_factory.get_llm_provider()`
  reads `LLM_PROVIDER`, default `"openai"`). A fresh deployment REQUIRES
  `OPENAI_API_KEY`; `ANTHROPIC_API_KEY` is only needed when `LLM_PROVIDER=anthropic`.
- LangChain callers get models through a **three-tier factory** (`fast` /
  `standard` / `reasoning`), never by hardcoding model IDs.
- DSPy paths (chatbot signatures, feedback-loop optimizer) resolve their model
  separately via `src/optimization/dspy_lm.py`, defaulting to
  `openai/gpt-5.6-terra`.
- Every factory-built model records token usage into `llm_usage_events`
  (migration 104); the `/admin` Observability tab prices it at read time.

---

## 1. The tier factory (`src/utils/llm_factory.py`)

Callers ask for a *tier*, and the factory maps tier x provider to a model:

| Tier | Used for | OpenAI (default provider) | Anthropic |
|------|----------|---------------------------|-----------|
| `fast` | classification, routing | `gpt-5.6-luna` (with `reasoning_effort="none"`) | `claude-haiku-4-5-20251001` |
| `standard` | general chat, synthesis | `gpt-5.6-terra` | `claude-sonnet-5` |
| `reasoning` | complex analysis | `gpt-5.6-terra` | `claude-sonnet-5` |

Entry points: `get_chat_llm(model_tier=...)`, plus convenience wrappers
`get_fast_llm()` (max_tokens 256, temperature 0.0, timeout 5s,
`reasoning_effort="none"`), `get_standard_llm()` (max_tokens 2048), and
`get_reasoning_llm()` (max_tokens 8192, timeout 120s).

Every model ID in the mapping was verified callable on the deployment's actual
API keys before being mapped (model refresh 2026-07-18, PRs #1274–#1276). The
previous Anthropic IDs were retired upstream and returned HTTP 404, which
silently degraded the chatbot to canned keyword responses — the failure class
this factory + verification discipline exists to prevent.

### Temperature handling

Some current models reject a non-default `temperature` (Claude Sonnet 5 /
Opus 4.8 return HTTP 400; gpt-5.x tolerates it inconsistently). The factory
**silently drops `temperature`** for models matching
`_TEMPERATURE_UNSUPPORTED_PREFIXES` (`claude-sonnet-5`, `claude-opus-4-8`,
`claude-fable-5`, `gpt-5`), so existing callers that pass one keep working
across model upgrades.

### Reasoning effort

For OpenAI gpt-5.x models, `reasoning_effort` (`"none"`/`"low"`/`"medium"`/
`"high"`) is forwarded to the API. The `fast` tier pins
`reasoning_effort="none"` — without it, gpt-5.x default reasoning can consume a
small `max_tokens` budget entirely and return empty content. Note that
reasoning/thinking tokens count against `max_tokens` on gpt-5.x and Claude
5-family models.

---

## 2. Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `LLM_PROVIDER` | `openai` | Selects the provider for the factory AND the DSPy path (`openai` \| `anthropic`) |
| `OPENAI_API_KEY` | — | **Required under the default provider** |
| `ANTHROPIC_API_KEY` | — | Required only when `LLM_PROVIDER=anthropic` |
| `LLM_MODEL` | unset | Pins the OpenAI **standard/reasoning** model without a code change (the `fast` tier is unaffected). Also feeds the DSPy default. |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | Anthropic model for the DSPy/chat paths that read it (`dspy_lm.py`, `chatbot_graph.py`, `causal_rag.py`) — independent of the factory's hardcoded mapping |
| `DSPY_LM_MODEL` | unset | Explicit DSPy/litellm model, used verbatim; must be provider-prefixed (e.g. `openai/gpt-5.6-terra`) |

Precedence example (DSPy): `DSPY_LM_MODEL` > provider branch
(`anthropic/{ANTHROPIC_MODEL}` when `LLM_PROVIDER=anthropic`) >
`openai/{LLM_MODEL or gpt-5.6-terra}`.

**Gotcha**: setting `LLM_PROVIDER=anthropic` alone also flips DSPy onto
`anthropic/{ANTHROPIC_MODEL}`. If you want Anthropic for the LangChain lanes
but keep DSPy on OpenAI, set `DSPY_LM_MODEL=openai/gpt-5.6-terra` explicitly.

---

## 3. The DSPy lane (`src/optimization/dspy_lm.py`)

DSPy talks to providers through **litellm**, so its model string carries a
`<provider>/<model>` prefix — it does NOT go through the LangChain factory.
`ensure_dspy_configured()` is idempotent, checks the provider-appropriate API
key (`dspy_provider_api_key_present()`), and configures `dspy.LM` with the
resolved model. Both the chatbot DSPy route (`src/api/routes/chatbot_dspy.py`)
and the feedback-loop optimizer share this config.

litellm's built-in retries (`num_retries`) absorb transient provider-side
errors on this lane; the plain LangChain/OpenAI-SDK lane does not retry
auth-class errors, so intermittent provider 401 flakes (observed on
`gpt-5.6-terra`, 2026-07-18) surface there first. If a chat lane starts
throwing intermittent 401s with a valid key, suspect the provider before the
config.

---

## 4. Usage metering and pricing

- **Capture**: the factory attaches `UsageRecorderCallback` at construction
  time, so every `invoke`/`astream` on a factory-built model records input/
  output tokens into the `llm_usage_events` table (migration 104), attributed
  to the authenticated user via a JWT contextvar.
- **Pricing is read-time**: no cost is stored in `llm_usage_events`. Consumers
  call `src/services/llm_pricing.cost_usd()`, which resolves the model against
  `MODEL_PRICING` (longest-prefix match, provider prefixes stripped). Unknown
  models render as **"unpriced"**, never a silent default. Superseded models
  stay listed so historical rows keep pricing. `PRICING_VERSION` (bumped on
  each rate change) is surfaced in the API payload for provenance.
- **Surface**: the `/admin` page's **Observability** tab
  (`/api/admin/observability/llm-usage`) aggregates usage and cost per model /
  user / day.

When a new model ID enters `MODEL_MAPPINGS` (or arrives via `LLM_MODEL`), add
its rate to `MODEL_PRICING` in the same change — otherwise its usage shows as
unpriced in the admin view.

---

## 5. Other LLM consumers (fixed models, out of factory scope)

- **RAGAS fixture regression** (`.github/workflows/ragas-evaluation.yml`):
  gpt-4o as judge; manual-only in CI (throughput-bound on the CI OpenAI key —
  see issue #504). Scores a static fixture, so it detects judge-stack drift on
  frozen input, not production RAG quality (#1485).
- **RAGAS real-pipeline gate** (`scripts/run_real_pipeline_ragas.py`): the same
  frozen gpt-4o judge, applied to answers the live pipeline actually generated
  over contexts it actually retrieved. On-demand only.
- **Adaptive-validity audit evaluator**: defaults to
  `anthropic/claude-haiku-4-5-20251001` via `ADAPTIVE_VALIDITY_EVALUATOR_MODEL`
  (off by default; see `.env.example` §11).
- **Embeddings**: RAG embeddings are configured separately from the chat
  factory (see `src/rag/`).

---

## Cross-reference

- `.env.example` §3 — the copy-paste starting point for keys and overrides
- `DEPLOYMENT.md` — environment table for a running stack
- `docs/data/07-SUPPORTING-SCHEMAS.md` — `llm_usage_events` schema context
- Migration 104 (`llm_usage_events`), spec 2026-07-12 (admin observability)
