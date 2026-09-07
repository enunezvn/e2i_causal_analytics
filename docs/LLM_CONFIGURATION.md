# LLM Configuration

**Status**: Living reference | **Last verified against code**: 2026-09-07,
`src/utils/llm_factory.py` @ `d2be4a3d5` (re-pin the sha when you re-verify:
`git log -1 --format=%h -- src/utils/llm_factory.py`)

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
  `OPENAI_API_KEY`. `ANTHROPIC_API_KEY` is required when `LLM_PROVIDER=anthropic`
  and additionally gates (fail-open, both off the factory) the nightly
  routing-label judge (`src/tasks/routing_label_tasks.py`) and the Layer-4
  adaptive-validity evaluator (`src/data/causal_role_evaluator.py`).
- **As deployed** (droplet `.env`, ADR-010) the two lanes are split on purpose:
  `LLM_PROVIDER=anthropic` **and** `DSPY_LM_MODEL=openai/gpt-5.6-terra`, so
  production needs **both** keys. "`LLM_PROVIDER=anthropic`, therefore no OpenAI
  key" is wrong twice over — the DSPy pin and the RAG embeddings (§5) both need
  OpenAI. Verify: `grep -E '^(LLM_PROVIDER|DSPY_LM_MODEL)=' .env`.
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

`reasoning_effort` (`"none"`/`"low"`/`"medium"`/`"high"`) is a single knob that
the factory translates per provider:

- **OpenAI gpt-5.x** (`model.startswith("gpt-5")`): forwarded verbatim as the
  `reasoning_effort` request field.
- **Claude 5-family** (since #1299): models matching
  `_ADAPTIVE_THINKING_PREFIXES` — `claude-sonnet-5`, `claude-opus-4-8`,
  `claude-fable-5` — get `thinking={"type": "disabled"}` for `"none"`, and
  otherwise `thinking={"type": "adaptive"}` plus
  `model_kwargs={"output_config": {"effort": <level>}}` to bound the thinking
  appetite. These models **reject** `thinking.type=enabled`/`budget_tokens`;
  adaptive + `output_config.effort` is their control surface.
- Any other model: `reasoning_effort` is silently ignored.

The `fast` tier pins `reasoning_effort="none"` — without it, gpt-5.x default
reasoning can consume a small `max_tokens` budget entirely and return empty
content. Reasoning/thinking tokens count against `max_tokens` on both families,
which is why the synthesis lanes that build their own model
(`src/api/routes/copilotkit.py`, `src/agents/experiment_designer/nodes/design_reasoning.py`,
`src/memory/graphiti_service.py`) pass `max_tokens=8192` rather than the
2048 `standard` default: at 2048 a Claude 5 thinking pass consumed the whole
budget and streamed zero text (2026-07-20 copilot frozen-at-75% incident).

Measure: `grep -n '_ADAPTIVE_THINKING_PREFIXES' -A 6 src/utils/llm_factory.py`;
`sed -n '198,206p' src/utils/llm_factory.py`.

---

## 2. Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `LLM_PROVIDER` | `openai` | Selects the provider for the factory AND the DSPy path (`openai` \| `anthropic`) |
| `OPENAI_API_KEY` | — | **Required under the default provider** |
| `ANTHROPIC_API_KEY` | — | Required only when `LLM_PROVIDER=anthropic` |
| `LLM_MODEL` | unset | Pins the OpenAI **standard/reasoning** model without a code change (the `fast` tier is unaffected). Also feeds the DSPy default. |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | **Model choice: the DSPy lane only** (`dspy_lm.py`, and only when `LLM_PROVIDER=anthropic` with `DSPY_LM_MODEL` unset). `chatbot_graph.py:1970` reads it as a **telemetry label only**, with a different default (`claude-sonnet-4-6`) than the model `get_chat_llm()` actually built — a known logging defect, not a model selector. `causal_rag.py` does not read it directly; it resolves through `dspy_lm.get_default_dspy_model()`. **Not forwarded into containers** (see below) |
| `DSPY_LM_MODEL` | unset | Explicit DSPy/litellm model, used verbatim; must be provider-prefixed (e.g. `openai/gpt-5.6-terra`) |

Precedence example (DSPy): `DSPY_LM_MODEL` > provider branch
(`anthropic/{ANTHROPIC_MODEL}` when `LLM_PROVIDER=anthropic`) >
`openai/{LLM_MODEL or gpt-5.6-terra}`.

**Gotcha**: setting `LLM_PROVIDER=anthropic` alone also flips DSPy onto
`anthropic/{ANTHROPIC_MODEL}`. If you want Anthropic for the LangChain lanes
but keep DSPy on OpenAI, set `DSPY_LM_MODEL=openai/gpt-5.6-terra` explicitly.

### Compose forwards a whitelist, not your `.env`

Docker Compose forwards **no** `.env` file wholesale. The `x-common-env` anchor
in `docker/docker-compose.yml` is a **whitelist** of 39 host variables; a value
set in the host `.env` reaches `api`, `worker_*` and `scheduler` only if that
anchor names it. Anything else is a silent no-op inside the containers — the
in-code default governs and nothing warns you.

```bash
sed -n '/^x-common-env:/,/^x-common-worker:/p' docker/docker-compose.yml \
  | grep -o '\${[A-Z_0-9]*' | tr -d '${' | sort -u
```

Of the variables in the table above, `LLM_PROVIDER`, `LLM_MODEL`,
`DSPY_LM_MODEL`, `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are forwarded.
**`ANTHROPIC_MODEL` is deliberately not** (`grep -c 'ANTHROPIC_MODEL:'
docker/docker-compose.yml` → 0): its in-code defaults resolve to current models,
while the host `.env` pins an opus-tier id meant for interactive use.

**LLM-lane runtime knobs that compose *does* forward** (all optional, shown at
their compose default; the full operator table lives in DEPLOYMENT.md and
`.env.example` §12; the trailing column is the **issue** each knob's introducing
commit cites, not a PR number):

| Variable | Compose default | Purpose | Issue |
|----------|-----------------|---------|-------|
| `CHATBOT_STARTUP_WARM_ENABLED` | `true` | per-worker warm of DSPy LM config, retrieval clients, registry, classifier | #1454 |
| `CHATBOT_STARTUP_WARM_LLM_ENABLED` | `true` | the warm's 2 synthetic-LLM legs (~2 small completions per worker per boot) | #1475 |
| `CHATBOT_RAG_LLM_TIMEOUT_S` | `20` | fail-open ceiling per `retrieve_rag` chain LLM call (rewrite/score/hop-decider) | #1484 |
| `CHATBOT_RAG_DRY_HOP_LIMIT` | `2` | consecutive zero-new-keep hops before the loop stops; `0` = run to max | #1484 |
| `CHATBOT_RAG_REWRITE_COT` | *(empty)* | empty = Predict-only rewriter; `true` restores ChainOfThought | #1518 |
| `CHATBOT_RAG_SKIP_EMPTY_DECIDER` | *(empty)* | empty/`true` = skip the decider call when hop-1 kept nothing | #1518 |
| `DSPY_RAG_RECORDS_PATH` | *(empty)* | GEPA records file, **resolved inside the container** — a host path skips forever while looking configured | #1486 |
| `DSPY_RAG_MAX_METRIC_CALLS` | *(empty)* | judge-call budget; empty on purpose, the in-code default (40) is the SSOT | #1486 |
| `DSPY_RAG_DB_FEEDSTOCK_ENABLED` | *(empty)* | live-traffic feedstock; the only way the cycle runs unattended (fail-closed parse) | #1489 |
| `DSPY_RAG_DB_LOOKBACK_DAYS` | *(empty)* | read window in days (in-code default 30) | #1489 |
| `CHATBOT_OPT_DRAIN_ENABLED` | *(empty)* | chatbot optimization queue drainer gate; unset = drain skipped | #1515 |
| `CHATBOT_OPT_DRAIN_MAX_PER_CYCLE` | *(empty)* | GEPA executions per cycle (in-code default 1) | #1515 |
| `CHATBOT_OPT_STALE_HOURS` | *(empty)* | in-code default 168 | #1515 |
| `CHATBOT_OPT_ZOMBIE_HOURS` | *(empty)* | in-code default 12 | #1515 |
| `CHATBOT_OPT_MIN_SIGNALS` | *(empty)* | producer minimum (in-code default 50) | #1515 |
| `ROUTING_LABEL_MIN_NEW_ROWS` | `10` | skip the nightly labeling cycle below this many unlabeled `classification_logs` rows | #1341 |
| `ROUTING_LABEL_JUDGE_CAP` | `50` | max LLM-judge calls per run (token-spend bound) | #1341 |
| `ROUTING_LABEL_JUDGE_MODEL` | `claude-haiku-4-5-20251001` | judge model; needs `ANTHROPIC_API_KEY`, fail-open when absent | #1341 |
| `ROUTING_LABEL_LOOKBACK_DAYS` | `30` | how far back to look for unlabeled rows | #1341 |

`ROUTING_LABEL_*` is documented for the chat lane in `docs/api/chat.md` §7.

---

## 3. The DSPy lane (`src/optimization/dspy_lm.py`)

DSPy talks to providers through **litellm**, so its model string carries a
`<provider>/<model>` prefix — it does NOT go through the LangChain factory.
`ensure_dspy_configured()` is idempotent, checks the provider-appropriate API
key (`dspy_provider_api_key_present()`), and configures `dspy.LM` with the
resolved model. **Eight modules** call it (`grep -rl ensure_dspy_configured src/`
minus the definition): `src/api/routes/chatbot_dspy.py`,
`src/api/chatbot_warmup.py`, `src/insights/common.py`,
`src/tasks/dspy_optimization_tasks.py`,
`src/agents/heterogeneous_optimizer/dspy_integration.py`, and three
feedback-learner modules (`nodes/pattern_analyzer.py`, `optimization_runner.py`,
`recipient_optimizer.py`).

litellm's built-in retries absorb transient provider-side errors on this lane —
we do not pass `num_retries`, so dspy 3.1.0's `dspy.LM` default of **3** applies
(`grep -n 'num_retries: int' .venv/lib/python3.12/site-packages/dspy/clients/lm.py`); the plain LangChain/OpenAI-SDK lane does not retry
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

**Do not confuse this with `src/mlops/agent_cost_tracker.MODEL_PRICING`.** That
is a second, unrelated rate table: it is an unwired scaffold (its `AgentCostTracker`,
`calculate_cost` and `CostRecord` have zero callers outside `src/mlops/__init__.py`'s
re-export) and its ids are stale (`claude-3-5-sonnet-*`, `gpt-4`, `gpt-3.5-turbo`
— none of the four current factory ids appear), and unlike `llm_pricing` it falls
back to a `DEFAULT_PRICING` rather than reporting "unpriced". The authoritative
table for anything user-facing is `src/services/llm_pricing.MODEL_PRICING`.

---

## 5. Other LLM consumers (fixed models, out of factory scope)

- **RAGAS fixture regression** (`.github/workflows/ragas-evaluation.yml`):
  gpt-4o as judge; manual-only in CI (throughput-bound on the CI OpenAI key —
  see issue #504). Scores a static fixture, so it detects judge-stack drift on
  frozen input, not production RAG quality (#1485).
- **RAGAS real-pipeline gate** (`scripts/run_real_pipeline_ragas.py`): the same
  frozen gpt-4o judge, applied to answers the live pipeline actually generated
  over contexts it actually retrieved. On-demand only.
- **Adaptive-validity audit evaluator** (`src/data/causal_role_evaluator.py`):
  `DEFAULT_EVALUATOR_MODEL = "anthropic/claude-haiku-4-5-20251001"`, overridable
  via `ADAPTIVE_VALIDITY_EVALUATOR_MODEL`, gated off by default behind
  `ADAPTIVE_VALIDITY_EVALUATOR_ENABLED`. **Caveat:** neither var is forwarded by
  compose (`grep -c ADAPTIVE_VALIDITY_EVALUATOR_MODEL docker/docker-compose.yml`
  → 0), so both are host-side only — setting them in the droplet `.env` is inert
  inside the containers and changing them there needs a compose change. Only
  `ADAPTIVE_VALIDITY_ARTIFACTS_DIR` is forwarded. It also fail-opens when
  `ANTHROPIC_API_KEY` is absent.
- **Nightly routing-label judge** (`src/tasks/routing_label_tasks.py`):
  `ROUTING_LABEL_JUDGE_MODEL`, default `claude-haiku-4-5-20251001`, direct
  Anthropic SDK. `ANTHROPIC_API_KEY`-gated and **fail-open** — with no key the
  judge is simply disabled for that run, which is why the key is needed on the
  droplet even when it is not the factory provider.
- **Fast-tier suggestion pills** (`POST /api/chat/suggestions`, #1900): goes
  through the factory (`get_fast_llm(max_tokens=600, timeout=8)`), so it *is*
  metered — listed here because it is the highest-volume fast-tier consumer and
  a provider flip changes its model.
- **Direct-SDK consumers that bypass the factory and therefore its metering**
  — these construct `anthropic.Anthropic` / `anthropic.AsyncAnthropic`
  themselves, so their tokens never reach `llm_usage_events` and never appear in
  the `/admin` Observability tab. All three hardcode `claude-sonnet-4-6`, which
  is **not** in the factory's `MODEL_MAPPINGS`:
  `src/rag/insight_enricher.py`, `src/rag/query_optimizer.py`, and
  `AnthropicLLMService` in `src/memory/services/factories.py` (the memory /
  Graphiti services lane). They also ignore `LLM_PROVIDER`: they stay on
  Anthropic regardless.
- **Embeddings**: RAG embeddings are **OpenAI `text-embedding-3-small`
  (1536-dim) regardless of `LLM_PROVIDER`** — `src/rag/config.py` reads
  `EMBEDDING_MODEL` with that default and has no Anthropic branch. This is the
  second reason `OPENAI_API_KEY` is required even on an Anthropic deployment.

---

## Cross-reference

- `.env.example` §3 — the copy-paste starting point for keys and overrides;
  §12 for the compose-forwarded runtime knobs and the "NOT forwarded" list
- `docs/decisions/adr-010-dspy-terra-scoped-anthropic-flip.md` — why the droplet
  runs `LLM_PROVIDER=anthropic` with the DSPy lane pinned to OpenAI
- `DEPLOYMENT.md` — environment table for a running stack
- `docs/data/07-SUPPORTING-SCHEMAS.md` — `llm_usage_events` schema context
- Migration 104 (`llm_usage_events`), spec 2026-07-12 (admin observability)
