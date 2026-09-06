# Copilot suggestion pills: capability catalog, validator, and page summaries as readables

**Date:** 2026-09-05
**Status:** approved by the user 2026-09-05; implemented on branch claude/copilot-pill-capability-catalog (PR opened 2026-09-06)
**Evidence:** `docs/demos/results/2026-09-05_pill_suggestions_review/`

## 1. Problem

The chat sidebar's suggestion pills come from `POST /api/chat/suggestions`
(`src/api/routes/chat.py`): one fast-tier LLM call over the page path, brand
filter, an optional on-screen `page_context` summary, and the recent
transcript. Measured live on 2026-09-05 (46 calls, 92 pills), graded against
what the chat agent's bound tools and generative-UI actions can deliver:

| Slice | OK | PARTIAL | NO |
|---|---|---|---|
| All pills | 25% | 33% | 42% |
| Pages that publish `page_context` | 20% | 17% | 63% |
| Pages without `page_context` | 38% | 35% | 27% |
| Per-turn follow-ups | 25% | 25% | 50% |

Root causes, in order of weight:

1. **The prompt's capability description is one prose sentence** with no
   negative list and no axis or dimension constraints. The model fills the gap
   with plausible analytics the platform does not offer.
2. **Context asymmetry.** Eight pages publish a `pageChatSummary` through
   `usePageChatContext`, but that text reaches only the pill endpoint. The chat
   agent's on-screen context comes from `useCopilotReadable` (filters, path,
   agent list, preferences; PredictiveAnalytics alone adds data readables). So
   the pill generator proposes questions about SHAP features, optimizer
   territories, CATE by segment and gap sizes that the agent can neither see
   nor recompute.
3. **Causal-registry outcomes read as computable KPIs** (`persistent_180d`
   "rate", "trend", "by region"). The registry has 14 outcome nodes and no
   time, region or segment dimension.
4. **No output validation** beyond JSON shape.

A scratch prototype with a code-derived capability catalog (same
`get_fast_llm`, positive control reproduced live pills verbatim) moved the
overall grade to OK 76% / PARTIAL 15% / NO 9%. Six of the eight residual NO
pills were cause 3. The prototype also reduced pill diversity on the 12 pages
that publish no `page_context` ("Portfolio TRx trend" led on 9 of 12).

## 2. Scope (user decision 2026-09-05)

Both parts:

- **A.** Code-interpolated capability catalog in the pill prompt, plus
  per-route hints for pages that publish no summary.
- **B.** Deterministic post-generation validator with static-pill top-up.
- **C.** Publish each page's `pageChatSummary` to the chat agent as a
  CopilotKit readable, so the agent sees the same on-screen summary the pill
  generator sees.

Non-goals (explicitly out):

- New read tools for SHAP, optimizer, gap or HTE results. The agent will be
  able to *read and discuss* the on-screen summary, not recompute or drill
  into it.
- A second LLM self-check pass over the pills.
- Re-using the catalog in the agent's own system prompt
  (`E2I_COPILOT_SYSTEM_PROMPT`). Possible follow-up once the builder exists.
- Adding `usePageChatContext` publishing to more pages. Route hints cover
  them cheaply; per-page summaries can follow one page at a time.

## 3. Design

### 3.1 Part A: capability catalog (backend)

New module `src/services/chat_capability_catalog.py` with one public
coroutine, `get_capability_catalog() -> CapabilityCatalog`, and a pure
renderer `render_catalog_block(catalog) -> str`. The pill prompt in `chat.py`
becomes a template with two placeholders, `{capability_catalog}` and
`{route_hint}`, filled per request. This follows the #1638 pattern
(`build_agent_roster_block()` interpolated into the agent prompt): lists are
derived from code, never transcribed.

`CapabilityCatalog` is a frozen dataclass with these fields and sources:

| Field | Source | Notes |
|---|---|---|
| `kpis` | `src.kpi.registry.get_registry().get_all()` | id, display name, workstream, `brand` for 45 KPIs. Loads YAML only; no DB. |
| `trend_kpi_ids` | `KPIHistoryRepository.get_coverage()` (view `v_kpi_history_coverage`) | KPI ids with a materialized monthly series, and whether each has a global scope or per-brand only. One small read. |
| `axis_kpi_ids` | `src.kpi.segmented_history.SEGMENTED_KPI_QUERY_FAMILIES` keys | The three Rx-volume KPIs that accept severity / therapy-line splits. |
| `causal_outcomes` | `CausalPathRepository.get_distinct_outcomes(include_synthetic=kpi_include_synthetic())` | 14 names in prod (both synthetic flags are `true` in `.env`). |
| `agent_roster` | `src.agents.factory.build_agent_roster_block()` | Already prompt-ready. |
| `loaded_at`, `degraded` | cache bookkeeping | `degraded` lists which DB-backed fields failed to load. |

The rendered block has fixed sections A through H, mirroring the validated
prototype prompt (`prototype_v2_system_prompt.txt`):

- **A. KPI values.** Every registry KPI by display name, grouped by
  workstream. Breakdown axes stated once: severity tier, therapy line 0-3, US
  census region, biologic and IgE tier (Remibrutinib only); axes are mutually
  exclusive; region does not compose with an explicit time window for share,
  conversion and trigger KPIs. The axis vocabulary is checked by a unit test
  against `inspect.signature(kpi_calculate_tool.coroutine)` parameter names so the
  prose cannot name an axis the tool does not accept.
- **B. Charts.** Monthly trend lines for the KPIs in `trend_kpi_ids`
  (per-brand-only ones marked), severity or therapy-line comparison lines for
  `axis_kpi_ids`, and a current-value chart for any other registry KPI
  (`renderChart`).
- **C. Causal drivers.** "Causal drivers, paths and treatment effects for
  these registry OUTCOMES: ...". Followed by the sentence the prototype was
  missing: outcomes are registry nodes, not KPIs; they cannot be computed,
  trended, charted or broken down by region or segment. Registry has no time,
  region or segment dimension.
- **D. Segments.** Rank HCP segments by predicted likelihood to prescribe, by
  specialty or geographic region, for one brand
  (`predict_hcp_segment_likelihood_tool`); KPI breakdowns by the axes in A.
- **E. Clinical context.** Label indications, mechanism, pivotal endpoints and
  real-world evidence for the three brands.
- **F. Platform.** `agent_roster` verbatim; agent status.
- **G. Documents.** Internal document retrieval.
- **H. Dashboard actions.** Navigate, set brand, region and date-range
  filters.
- **NEVER PROPOSE.** Emails, exports, CRM or field actions, external or
  competitor data, writing to systems, model retraining, per-HCP or
  per-patient predictions, SHAP or feature-importance recomputation,
  territory or optimizer allocations, gap-size recomputation, uplift by
  segment. Also: treating a section-C outcome as a KPI.

**On-screen rule (changes because of Part C).** The prototype told the model
that `page_content` is invisible to the assistant. After Part C the assistant
sees the same summary. The rule becomes: pills may ask the assistant to read,
compare, rank or explain values that appear literally in `page_content`; they
must not ask for anything beyond those values that needs a tool the assistant
lacks (another segment's SHAP, features beyond the ones shown, per-territory
detail not shown). This rule is the single assumption the redesign adds over
the measured prototype, so it is re-measured before implementation (section
5, step 1).

**Route hints.** A module-level `ROUTE_HINTS: dict[str, str]` maps each app
route (from `frontend/src/router`, auth routes excluded) to one sentence
describing what the page shows and which catalog sections fit it, e.g.
`"/kpi-dictionary": "KPI registry definitions; pills should ask for the value,
definition or trend of specific KPIs."` Used only when `page_content` is
empty. Unknown paths get an empty hint and the prompt falls back to path and
brand as today. The prompt also asks for four pills that cover at least two
different catalog sections, which addresses the diversity regression directly
rather than through the hint alone.

**Caching.** The catalog is built lazily on first request and cached
in-process with a 10-minute TTL (60 s while degraded). The refresh is
single-flight: concurrent cold or expired callers await the one build in
flight, so a failing DB is not hammered and a slower degraded build cannot
overwrite a faster good one; a build orphaned by the test-only `reset()`
serves its own waiters but stays out of the cache. On refresh failure the
last good catalog is kept and the failing fields are recorded in `degraded`. If the very first
build fails for a DB-backed field, the section renders an honest one-line
fallback ("outcome list unavailable; propose at most one causal-driver pill
and name no outcome") rather than an invented list. Registry and roster never
fail (pure code). No startup hook: CI's unit job runs `TestClient(app)`
lifespans on a thread with a 30 s timeout, and a lazy cache avoids adding
work there.

**Prompt size and latency.** The prototype's catalog added roughly 1.5k
input tokens; measured latency was 2.0 to 3.3 s against a baseline of 2.4 to
5.4 s, inside the route's 8 s timeout. `max_tokens` rises from 500 to 600 to
match the prototype (pill messages are slightly longer when they name a
brand and an axis). The system message is now a stable prefix across
requests, which suits provider prompt caching if the fast-tier client ever
enables it; not part of this change.

### 3.2 Part B: validator and top-up

`chat_capability_catalog.filter_unsupported_pills(pills, catalog) ->
(kept, dropped)` runs after `_parse_suggestions`. It is deterministic and
narrow, tuned only to the pill families graded NO in the baseline sample:

1. **Outcome-as-KPI.** A message that names a `causal_outcomes` entry and
   also asks for a value, rate, breakdown or region, without also asking
   for drivers, causes or paths; or one where a trend, time-series or
   chart word (trend, over time, monthly, quarterly, weekly, chart, plot,
   graph) attaches to the outcome itself - within a few words of it with no
   causal word between - whatever causal clause follows, because a causal
   tail does not make a section-C outcome trendable (#1906). A time word
   that modifies a driver ("does monthly copay support drive
   persistent_180d?") is a section-C ask and stays. The outcome names come
   from the cached catalog, not a hardcoded list.
2. **Off-platform asks.** Word-boundary patterns for SHAP, feature
   importance, territory detail (which is how optimizer allocations were
   phrased in the sample), individual-HCP or individual-patient prediction
   (aggregate HCP-segment likelihood by specialty or region is served and
   stays), gap-size recomputation, uplift or CATE by segment, email, export,
   CRM, competitor share.
3. **On-screen reads are kept.** Because Part C gives the agent the same
   summary, a pill that only reads, ranks or compares SHAP, gap, CATE or
   prediction values that are literally on screen is answerable and the
   four artefact rules yield to it — unless the text also asks to
   recompute, validate, extend, explain *why* (describing what a chart
   shows is a read), or trend the artefact
   (the extends-list mirrors the prompt's forbidden verbs and the rules'
   own trend and axis vocabulary). Registry KPI names the rules collide
   with (Geographic Consistency Gap, SHAP Coverage, Conditional ATE (CATE)
   as a current-value ask or chart) are exempt; trend and segment forms of
   CATE still drop.

Each drop is logged at INFO with the rule name and the pill title, so the
production drop rate is measurable from logs. The route returns the survivors
(one to four). If none survive it returns 502 exactly as today, and the
frontend falls back to static pills.

Frontend top-up in `ConversationSuggestions` (`E2IChatSidebar.tsx`): when the
endpoint returns fewer than four pills, append pills from
`buildChatSuggestions(pathname, brand)` whose title is not already present,
up to four. The endpoint never has to know about the static pills, and the
pane keeps its four-pill shape.

### 3.3 Part C: page summaries as a readable (frontend + one wording tweak)

In `CopilotHooksInner` (`E2ICopilotProvider.tsx`), add a fifth
`useCopilotReadable`:

- `description`: "Summary of the data currently visible on the page, as
  published by the page itself (prose; same text used to ground suggestion
  pills)".
- `value`: `context.pageChatContext` (the exact string the pill endpoint
  receives, so the two channels cannot diverge).
- `available`: `'enabled'` when the summary is non-empty, `'disabled'`
  otherwise. `useCopilotReadable` 1.51.2 supports this option; the hook
  stays inside `CopilotHooksInner`, which is only mounted when CopilotKit is
  enabled (the hook throws outside a `<CopilotKit>` provider).

The backend already renders readables into the agent prompt through
`_readables_context_note` in `copilotkit.py`: strings pass through, items are
capped at 12,000 chars and the note at 32,000, and page summaries are at most
4,000 chars. One wording change, applied only when a prose readable is
present (the page summary is the only readable that bypasses the SDK's JSON
encoding, so the note detects prose by value): the note then says "values
are JSON or short prose summaries" and the reading instruction adds "an
on-screen summary is a description of what the page shows, not a data table;
cite it as on-screen context and do not present its figures as tool
results". Pages that publish no summary keep the pre-existing note
byte-identical. The existing instruction to answer from on-screen context
first and call tools only for what is not on screen stays.

PredictiveAnalytics publishes both data readables and a summary; both will
appear in the note. This is redundant but within budget and harmless.

### 3.4 Data flow after the change

```
Page (8 publishers)  --pageChatSummary-->  E2IContext.pageChatContext
     |                                            |
     |                    +-----------------------+--------------------+
     |                    v                                            v
     |   ConversationSuggestions (opener mode)         CopilotHooksInner readable #5
     |   POST /chat/suggestions {page, brand,          AG-UI request context[]
     |        page_context}                                     |
     |            |                                             v
     |            v                               copilotkit.py _readables_context_note
     |   chat.py: prompt = TEMPLATE.format(               -> ON-SCREEN APP CONTEXT
     |      capability_catalog=render(catalog),
     |      route_hint=ROUTE_HINTS.get(page, ""))
     |   -> get_fast_llm().ainvoke -> _parse_suggestions
     |   -> filter_unsupported_pills -> 1..4 pills (502 if 0)
     |            |
     |            v
     |   frontend tops up to 4 with buildChatSuggestions
```

Per-turn follow-ups take the same backend path without `page_context` and
benefit from the catalog and validator equally.

## 4. Error handling

| Failure | Behaviour |
|---|---|
| DB unavailable on first catalog build | Section renders honest fallback line; request proceeds; `degraded` logged once per refresh attempt. |
| DB unavailable on refresh | Last good catalog kept and rendered; `degraded` names the failed fields, so the next retry is after the degraded TTL (60 s), not the full TTL. |
| LLM timeout or error | 502 as today; frontend static pills. |
| All pills dropped by validator | 502 with detail "no supported pills"; frontend static pills. |
| Validator false positive | Pill lost, replaced by a static pill; INFO log carries rule and title so false positives can be found and the pattern narrowed. |
| Readable value null | `available: 'disabled'`; the note omits it; prompt is byte-identical to today on pages that publish nothing. |

## 5. Testing and certification

1. **Prompt re-measurement before code (cheapest disproof).** Re-run the
   scratch prototype (`proto_v2.py`) with the final catalog text, the
   revised on-screen rule, the outcome-not-KPI sentence and route hints,
   over the same 46 request shapes. Same rubric. Gate: NO at or below 10%
   overall and on the `page_context` pages; at least three distinct lead
   pills across the 12 no-context pages. If the revised on-screen rule
   raises NO on the `page_context` pages, tighten it before implementing.
2. **Backend unit tests** (`tests/api/test_chat_capability_catalog.py`):
   registry-derived KPI names render; axis vocabulary is a subset of
   `kpi_calculate_tool` parameters; outcomes and coverage are injected
   through fakes; degraded rendering when a fake repo raises; TTL refresh
   keeps the last good catalog; `ROUTE_HINTS` keys are normalized paths;
   validator table test using a fixture of NO and OK pills taken from
   `live_pills_baseline_2026-09-05.json` (every baseline NO in the two
   covered families is dropped, every baseline OK is kept).
3. **Route tests** (extend `tests/api/test_chat_suggestions.py`): the prompt
   sent to the fake LLM contains catalog markers and the route hint; dropped
   pills reduce the count; all-dropped returns 502; existing brand-rule and
   parsing tests keep passing.
4. **Frontend tests**: `E2ICopilotProvider.test.tsx` gains a case that the
   fifth readable carries `pageChatContext` when set and is disabled when
   null; `E2IChatSidebar.suggestions.test.tsx` gains a top-up case (two
   adaptive pills become four, no duplicate titles).
5. **Live certification after deploy**: re-run `probe_pills.py` against the
   deployed endpoint and grade; confirm INFO drop logs appear; on
   `/feature-importance` ask the assistant "what is on this page?" through
   the sidebar and confirm the reply cites the published summary and does
   not claim to have computed SHAP values. Hold the cert until the last
   deploy run is terminal, and grep lazy-chunk bundles by their own hash.

**Gate 5.1 result (2026-09-06, worktree dfc361eb3, real catalog + claude-haiku-4-5
via `get_fast_llm`, 23 scenarios, 92 pills, two blind graders reconciled):**
NO 3.4% on kept pills / 8.7% on all generated (gate ≤ 10%); NO 3.3% kept /
9.4% generated on the `page_context` pages (gate ≤ 10%); 12 distinct lead
pills across the 12 no-context openers (gate ≥ 3); 0 parse failures. PASS on
the first iteration. The validator dropped 5 pills, all graded NO (precision
5/5) and missed 3 (responder-cohort extension, live experiment status, a
non-journey outcome used as a rate). Kept-pill OK/PARTIAL/NO = 75/9/3 versus
baseline 23/30/39 and prototype v2 70/14/8 on 92. Evidence:
`docs/demos/results/2026-09-05_pill_suggestions_review/` — `proto_v3.py`,
`proto_v3.log`, `prototype_v3_pills.json`, `prototype_v3_system_prompt.txt`,
`v3_grades.md`.

## 6. Files

| File | Change |
|---|---|
| `src/services/chat_capability_catalog.py` | New: dataclass, builder, TTL cache, renderer, `ROUTE_HINTS`, `filter_unsupported_pills`. |
| `src/api/routes/chat.py` | Prompt becomes a template; fill catalog and hint; run validator; log drops; `max_tokens=600`. Module docstring updated. |
| `src/api/routes/copilotkit.py` | `_readables_context_note` wording for prose summaries. |
| `frontend/src/providers/E2ICopilotProvider.tsx` | Fifth readable in `CopilotHooksInner`; comment count "4 readables" becomes 5. |
| `frontend/src/components/chat/E2IChatSidebar.tsx` | Top-up to four pills; architecture doc comment updated. |
| `tests/api/test_chat_capability_catalog.py` | New. |
| `tests/api/test_chat_suggestions.py` | Extended. |
| `frontend/src/providers/E2ICopilotProvider.test.tsx`, `frontend/src/components/chat/E2IChatSidebar.suggestions.test.tsx` | Extended. |
| `frontend/src/providers/copilotReadableConverters.ts`, `frontend/src/providers/copilotReadableConverters.test.tsx` | New (added in review): rest-args readable converter for CopilotKit's single-argument `convert(value)` runtime call; pure-function, SDK-fence and real-hook contract tests. |
| `tests/api/test_copilotkit_readables_note.py` | New: `_readables_context_note` wording for prose summaries. |
| `.github/workflows/backend-tests.yml` | Unit Tests lane collects the three `tests/api/` files above (`test_chat_suggestions.py` was never run in CI before). |
| `docs/demos/results/2026-09-05_pill_suggestions_review/` | Add the re-measurement output from step 5.1. |

## 7. Risks and open points

- **Validator over-blocking.** Kept narrow to two measured families; logs
  make false positives visible. Widening the denylist is a separate decision.
  Known fail-safe residual (2026-09-06): the validator matches on title plus
  message, so a pill whose short title uses the bare token "CATE" while its
  message asks an answerable current-value question is still dropped by the
  bare-CATE alternative; closing it needs a wider subgroup alternative
  first ("for high-severity patients" currently relies on the bare token).
  The 92-pill gate run had no such title and drop precision 5/5.
- **Catalog drift.** Lists come from code and data. The only prose that can
  rot is the axis and composition rules in section A and the `ROUTE_HINTS`
  map; the axis test guards the former, and a route rename shows up as an
  unknown path falling back to today's behaviour.
- **Readable size.** Summaries are capped at 4,000 chars by the publishers
  and the endpoint; the note's per-item cap is 12,000. No new truncation.
- **Agent honesty.** Seeing the summary lets the agent answer "what does
  this page show", which is the intent. It could also tempt the agent to
  present summary figures as computed results. The wording tweak in 3.3
  addresses this; the live cert checks it.
