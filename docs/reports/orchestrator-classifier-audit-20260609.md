# Orchestrator Routing-Engine Audit — the "4‑stage classifier"

**Date:** 2026-06-09
**Scope:** `src/agents/orchestrator/` — the decision engine that routes every query
**Method:** Read all source + **real, offline, no‑mock** execution of both classifiers on a 13‑query pharma battery (cheapest‑disproof), then a 12‑agent adversarially‑verified workflow (6 per‑dimension finders + 6 skeptics) grounded in that probe output.
**Status:** Audit only — **no code changed, no PR, no deploy.**
**Repro:** `docs/reports/orchestrator-classifier-audit-20260609-repro/` (`probe_classifiers.py`, `probe_output.txt`)

---

## 0. Headline — the premise is inverted

The task names "the Orchestrator 4‑stage classifier, the decision engine that routes every query. Four deterministic stages run before any LLM is called."

**That module routes nothing.** The 4‑stage `ClassificationPipeline` in `src/agents/orchestrator/classifier/` has **zero production consumers and zero tests** — it is not wired into `create_orchestrator_graph()`. The engine that *actually* routes every query is a **different** module: `nodes/intent_classifier.py` (`IntentClassifierNode`) + `nodes/router.py` (`RouterNode`), and its real shape is **one regex stage, then an LLM fallback** — not four deterministic stages.

| | **`classifier/` — the "4‑stage classifier"** (audited target) | **Live engine** (what graph.py wires) |
|---|---|---|
| Wired into the orchestrator graph? | **No** — orphaned | **Yes** — `classify`→`classify_intent`, `route`→`route_to_agents` |
| Shape | FeatureExtractor → DomainMapper → DependencyDetector(+LLM) → PatternSelector | `IntentClassifierNode` (regex; conf≥0.8 short‑circuit, else 1 Haiku call) + `RouterNode` |
| "4 deterministic stages before any LLM"? | Accurate as a *design*, but unwired | **False** — it's 1 regex stage **then** an LLM |
| Tests | **0** | `test_intent_classifier*.py`, `test_multi_faceted_ssot.py`, etc. |
| Real behavior on 13‑query battery | **2 crash, 6 refused, every query forced multi‑domain** | 9/13 routed to the correct single agent; 2 real misroutes |

Why the confusion is reasonable: `docs/ONBOARDING.md:276` and `docs/Archive/E2I_USER_GUIDE.md:468` **both assert** the orchestrator uses a "4‑stage classifier." Those docs describe the *planned v4.2 architecture* that was **built but never integrated** — the integration was a **P0 task left unchecked** (`docs/Archive/tool_composer_component_update_list.md:137`), and the doc even calls a method (`classify_query()`) **that does not exist** on the class (it's `classify()`). The docs drifted; the code never caught up.

**Net:** the live router works reasonably (and has 2 fixable misroutes); the named "4‑stage classifier" is a **latent landmine** — if anyone completes the still‑documented P0 wiring without fixing it, real pharma queries **crash** or get **refused**.

---

## 1. Architecture ground truth (verified)

Live path, from `src/agents/orchestrator/graph.py:90-103`:

```
[audit_init] → [classify] → [rag_context] → [route] → [dispatch] → [synthesize] → END
                   │                            │
       classify_intent (IntentClassifierNode)   route_to_agents (RouterNode)
       nodes/intent_classifier.py               nodes/router.py
```

- `OrchestratorAgent.run()` (`agent.py:56`) builds this graph. `OrchestratorAgent.classify_intent()`/`route_query()` (`agent.py:292-329`) also use `IntentClassifierNode`+`RouterNode`.
- Whole‑repo grep for `ClassificationPipeline | FeatureExtractor | DomainMapper | PatternSelector | DependencyDetector` returns hits **only** inside `classifier/` and its `__init__.py`. Nothing in `src/api/`, chatbot routes, factories, or tasks imports it.
- Git: `classifier/` was added in the initial platform commit (`3e1c70cf`, 2025‑12‑20); the last *functional* change was `e9836741` (2026‑01‑16, cohort routing). Everything since is ruff/mypy/format. So it is **built‑and‑maintained but never wired** — not vestigial copy‑paste.

---

## 2. Reasoning before rules — what is the unwired pipeline, and what should happen to it?

Per the project's REASON‑BEFORE‑RULES directive, the orphaned pipeline is **not** an automatic "zero consumers → delete":

1. **What is it trying to do?** Be a cheaper, more structured router than `IntentClassifierNode`: deterministic feature/domain/dependency analysis that escalates to an LLM only for genuinely complex queries and sends dependent multi‑domain queries to `tool_composer`.
2. **Why does it exist in this shape?** It is the **v4.2 design** (`docs/Archive/integration/v4.2_integration.md`). Integration into `agent.py` was specced **P0** but the checkbox is unchecked and the call‑site (`classify_query()`) was never written.
3. **Is it harming production now?** **No** — it is unreachable. Its defects are **HARMFUL_IF_WIRED** (latent), not HARMFUL_NOW. The real present‑day harms are (a) **doc drift** that makes operators believe it's live, and (b) a **maintenance + landmine** cost.
4. **What does the user want?** To trust the engine that routes queries. The deliverable is the truth about which engine is live, the real defects in **both**, and a reasoned path — *not* a reflexive delete.

**Recommendation menu (choose one):**

- **Option A — Deprecate/quarantine (recommended now).** The live `IntentClassifierNode`+`RouterNode` already work; the 4‑stage pipeline as‑is would *regress* routing (crashes + refusals). Mark `classifier/` `EXPERIMENTAL / NOT WIRED` at the package docstring, and **fix the docs** (§Finding A2) so nobody believes it's live or wires it by mistake. Lowest risk, removes the landmine and the confusion.
- **Option B — Wire it after a real fix‑pack.** Only if the team wants the Tool‑Composer‑routing benefits. Requires fixing **B1–B5 at minimum** (crash, threshold, base‑floor, LLM stub, escalation), adding the missing `agent.py` integration + tests, and shadow‑running against live traffic before switching. This is a project, not a patch.
- **Option C — Delete `classifier/`.** Defensible (0 consumers, 0 tests) and simplest, but discards a non‑trivial design the team twice invested in (cohort routing was added in Jan 2026). Only if the team has decided the v4.2 router direction is dead.

Either A or C also **must** include the doc correction. Do **not** wire it without the fix‑pack.

---

## 3. Findings

Severity = defect seriousness. **Harm class** = *when* it bites: `HARMFUL_NOW` (affects the live router today) · `HARMFUL_IF_WIRED` (latent in the orphaned pipeline) · `CONFUSION_DOC_DRIFT`. All findings below survived adversarial verification against the real source + probe.

### A. Architecture & documentation

| ID | Sev | Harm | Location | Finding |
|---|---|---|---|---|
| **A1** | HIGH | IF_WIRED | `classifier/*` | `ClassificationPipeline` is unwired — **0 consumers, 0 tests**. Not in `graph.py`. The live router is `IntentClassifierNode`+`RouterNode`. |
| **A2** | HIGH | DOC_DRIFT | `docs/ONBOARDING.md:276`, `docs/Archive/E2I_USER_GUIDE.md:468` | Both docs state the orchestrator routes via a "4‑stage classifier." False — those describe the unwired v4.2 design. This is the root of the task's inverted premise. **Fix:** correct to "IntentClassifierNode (regex) + RouterNode; 4‑stage pipeline is experimental/not wired." |
| **A3** | MED | DOC_DRIFT | `pipeline.py:58` vs `docs/Archive/tool_composer_component_update_list.md:148`, `v4.2_IMPLEMENTATION_TODO.md:524` | Integration was **P0 but unchecked**; the spec calls `classify_query()` which **doesn't exist** (impl is `classify()`) — hard proof the wiring was never done. |

### B. 4‑stage pipeline defects — `HARMFUL_IF_WIRED` (latent landmines)

These do not affect production today (the pipeline is unreachable). They **would** activate the moment the P0 integration is completed. Proven by real execution.

| ID | Sev | Location | Finding & evidence |
|---|---|---|---|
| **B1** | **CRITICAL** | `feature_extractor.py:76` (`TIME_PATTERNS`), `:292`, `schemas.py:74` | **Crash.** Regex `\b(last\|this\|next)\s+(week\|month\|quarter\|year)\b` has **two capture groups** → `re.findall` returns a **tuple** `('next','quarter')` → `TemporalFeatures.time_references: list[str]` raises `ValidationError`. **Any** query with last/this/next + week/month/quarter/year crashes the whole pipeline. Probe: 2/13 crashed (`"Predict next quarter's TRx…"`, `"…dropped last month."`). **Fix:** non‑capturing groups `(?:…)`. |
| **B2** | **CRITICAL** | `pattern_selector.py:44,81-92` vs `domain_mapper.py:63-76` | **Half of clean queries refused.** Rule 1 demands top confidence ≥ `MIN_CONFIDENCE=0.5`, but `DomainMapper` tops out ~**0.40–0.46** on ordinary single‑keyword queries → `CLARIFICATION_NEEDED` with empty `target_agents`. Probe: **6/13** refused, incl. a clean `"What was the impact of the Q3 Kisqali campaign…"`. **Fix:** recalibrate `MIN_CONFIDENCE` (~0.35) *and/or* the base scores (B3) together. |
| **B3** | **HIGH** | `domain_mapper.py:63-69,79`; `pattern_selector.py:123-142,162-187` | **MONITORING(base 0.4)+EXPLANATION(base 0.3) ≥ threshold 0.3 on EVERY query** — confirmed on `""`, `"hello"`, `"asdf qwerty zzz"` (all → primary `MONITORING`). Consequences: every query is forced multi‑domain → `PARALLEL_DELEGATION` **always appends spurious `drift_monitor`+`explainer`**; Rule 3 `SINGLE_AGENT` is effectively **dead code**. **Fix:** drop MONITORING/EXPLANATION base scores or require a keyword match. |
| **B4** | **HIGH** | `dependency_detector.py:277-307` (`:294` model) | **Stage‑3 LLM is a discarded stub.** `_detect_with_llm` calls `claude-3-5-haiku-20241022` then **throws the response away** and returns `[]` ("…return empty list as placeholder"). If wired it burns a token+latency per escalated query for nothing, and dependency detection never benefits from the LLM. Also hardcodes the model, bypassing `llm_factory`/provider abstraction. **Fix:** implement parsing, or remove the call and gate `_needs_llm_analysis` off. |
| **B5** | **HIGH** | `pipeline.py:163-189` | **`_should_use_llm` always fires.** Because `domain_count ≥ 2` always (B3), `MULTI_DOMAIN_THRESHOLD=2` makes nearly every query escalate to the Stage‑3 LLM — i.e. straight into the B4 stub. The "deterministic before any LLM" intent is undermined twice over. |
| **B6** | MED | `feature_extractor.py:342-343,130` | **Substring keyword matching** (`kw in query_lower`) → false positives (`"mean"` ⊂ `"meaningful"`). And `"A/B"` is stored **uppercase** but the query is lowercased → it can **never** match `"a/b test"`. **Fix:** word‑boundary regex; lowercase the keyword sets. |
| **B7** | MED | `dependency_detector.py:211,144-147,177-193` | Reference‑pronoun detection is substring‑based (`"it"` ⊂ `"activities"`, `"them"` ⊂ `"treatment"`) → false dependencies; `_decompose_query` misses `"then"` conjunctions; `_infer_domains_for_part` is a **second, divergent** domain matcher duplicating `DomainMapper`. |
| **B8** | MED | `feature_extractor.py:189-211,252-254`; `domain_mapper.py:179-181` | `ENTITY_PATTERNS` conflates `segment`/`cohort`; clause count filters by char‑length (>5) not meaning; GAP_ANALYSIS & HETEROGENEITY both key off `exploration_keywords` and interfere. |
| **B9** | MED | `pipeline.py:123-161` | `classify_sync` docstring claims "cannot use LLM layer" (false — it calls `classify()` which can) and uses deprecated `asyncio.get_event_loop()` without 3.10+ safety. |
| **B10** | LOW | `dependency_detector.py:204` | Dead statement: `query.lower()` result is discarded. |

### C. Live‑engine defects — `HARMFUL_NOW` (these route real traffic)

These affect production today and should be prioritized over the B‑series.

| ID | Sev | Location | Finding & evidence |
|---|---|---|---|
| **C1** | **HIGH** | `nodes/intent_classifier.py:102-108` | **Past‑tense causal miss.** `"what drove the Kisqali uplift?"` → **`general` → `explainer`** (my probe), while `"what caused…"`→`causal_effect` (0.87) and `"what drives…"`→`causal_effect` (0.93). The pattern token `driv` matches `driv‑es/‑ing` but **not `drove`** — the most common retrospective‑causal phrasing. **Fix:** add `drove`/`drshowed`‑style past tense or use `driv\|drove\|drv`; broaden `r"why.*(increase\|decrease\|change\|drop\|rise)"` likewise. |
| **C2** | **HIGH** | `src/agents/multi_faceted.py:51-57`; consumed by `intent_classifier.py:166` | **Multi‑part queries route to a single agent.** `MULTI_FACETED_PATTERNS` only fire on exact `and (also\|then\|additionally\|furthermore)` or `compare … (vs\|versus\|against\|to) … and`. Natural phrasings — `"…, and which … then design…"` and `"compare X and Y, then recommend…"` — **miss** (verified), so a genuinely 3‑part dependent query → single `heterogeneous_optimizer`, **never reaching `tool_composer`**. (The chatbot facet‑scorer `is_multi_faceted_facet_score` also returns 0 facets on the same query.) **Fix:** broaden the conjunction regex (`and\s+\w+.*,?\s*then`, `compare .* and .*,?\s*then`). |
| **C3** | MED | `nodes/router.py:163-176,199,213-217` | `requires_multi_agent` is computed but only honored for **3 hardcoded, directional** `MULTI_AGENT_PATTERNS` pairs; any other multi‑agent query (e.g. `(segment_analysis, experiment_design)`) **silently degrades to single‑agent** with no warning. **Fix:** make lookup order‑insensitive, add observed pairs, log on miss. |
| **C4** | MED | `nodes/intent_classifier.py:281` | `requires_multi_agent` gate needs `secondary[0] > 0.8` (strict) — conservative by design; secondary intents at 0.6–0.8 never trigger multi‑agent. Acceptable, but document/tune. |
| **C5** | LOW | `nodes/intent_classifier.py:323`; `src/utils/llm_factory.py:51` | **Verify** the project‑wide fast‑model id `claude-haiku-4-20250414`. It does not match the known Haiku 4.5 id `claude-haiku-4-5-20251001`; if it is not a valid deployed id, the (rarely‑hit, conf<0.8) live LLM fallback silently degrades to `general`. Scoped as an open item, not a confirmed bug — needs a faithful check against the deployment's model access. |

---

## 4. Real probe evidence (no mocks)

`PYTHONPATH=. .venv/bin/python docs/reports/orchestrator-classifier-audit-20260609-repro/probe_classifiers.py`

**4‑stage `ClassificationPipeline` (offline, deterministic) — 13‑query battery:**

| Query (label) | Result | Top domain/conf | Note |
|---|---|---|---|
| causal | `CLARIFICATION_NEEDED` | CAUSAL 0.40 | refused (B2) |
| gap | `CLARIFICATION_NEEDED` | MONITORING 0.40 | primary is MONITORING (B3) |
| segment | `CLARIFICATION_NEEDED` | MONITORING 0.40 | refused (B2/B3) |
| experiment | `CLARIFICATION_NEEDED` | EXPERIMENTATION 0.46 | refused (B2) |
| prediction | **CRASH** | — | "next quarter" (B1) |
| monitoring | `PARALLEL_DELEGATION` | MONITORING 0.58 | + spurious `causal_impact`,`explainer` (B3) |
| explanation | **CRASH** | — | "last month" (B1) |
| cohort | `PARALLEL_DELEGATION` | COHORT 0.62 | + spurious `drift_monitor`,`explainer` (B3) |
| multi‑dependent | `TOOL_COMPOSER` | EXPERIMENTATION 0.56 | (correct pattern, by luck of scores) |
| ambiguous | `PARALLEL_DELEGATION` | EXPLANATION 0.51 | `explainer`,`drift_monitor` |
| exploration | `PARALLEL_DELEGATION` | EXPLANATION 0.51 | 4 agents incl. spurious |
| comparison | `CLARIFICATION_NEEDED` | MONITORING 0.40 | refused (B2) |
| greeting `"hello"` | `CLARIFICATION_NEEDED` | MONITORING 0.40 | base‑floor only (B3) |

Outcome: **CLARIFICATION_NEEDED 6/13, CRASH 2/13**, MONITORING & EXPLANATION present in **13/13** (incl. `""`, `"hello"`, `"asdf qwerty zzz"`).

**Live engine — same battery:** 9/13 → correct single agent deterministically (causal→`causal_impact`, gap→`gap_analyzer`, segment→`heterogeneous_optimizer`, experiment→`experiment_designer`, prediction→`prediction_synthesizer`, monitoring→`drift_monitor`, cohort→`cohort_constructor`). Misroutes: multi‑dependent → single `heterogeneous_optimizer` (C2); exploration/comparison/greeting → `general`→`explainer` (conf 0.5 → would hit the LLM fallback in prod). Separately, `"what drove…"` → `general` (C1).

---

## 5. Prioritized actions

1. **Fix the docs (A2/A3)** — correct `ONBOARDING.md:276` and `E2I_USER_GUIDE.md:468`; this is what created the inverted premise. Cheap, high value.
2. **Fix the live‑engine misroutes (C1, C2, C3)** — these affect real traffic now. C1 ("what drove…") and C2 (multi‑part → single agent) are one‑line/one‑regex changes each, with the battery as a ready test.
3. **Decide the orphan's fate (Option A/B/C)** — recommend **A (quarantine + doc fix)** unless the team commits to the B fix‑pack. Do **not** wire `classifier/` without fixing B1–B5.
4. If wiring is ever pursued, treat **B1 (crash), B2 (threshold), B3 (base‑floor), B4 (LLM stub), B5 (escalation)** as blocking, and add the missing `agent.py` integration + a real test suite (there is none today).

---

## 6. Method & confidence

- **No mocks.** Both engines executed against real code; the 4‑stage pipeline ran fully offline/deterministic (its rule‑based path), the live engine's deterministic regex layer + `RouterNode` ran directly. The crash, the refusals, the base‑floor, and the live misroutes are **observed**, not theorized.
- **Adversarial verification.** 6 per‑dimension finders + 6 skeptics (12 agents) re‑checked every finding against the real source/probe; REFUTED items were dropped, severities corrected. The two highest‑stakes live‑engine claims (C1, C2) were additionally re‑confirmed by the dispatcher's own probe.
- **Honesty note:** C5 (model id) is the one item I did **not** fully disprove — flagged as "verify," not asserted. Several verifiers initially tagged B1–B3 as `HARMFUL_NOW`; corrected here to `HARMFUL_IF_WIRED` because the pipeline is unreachable in production. The defects are real and severe **as code**, but they harm only upon wiring.

---

## 7. Remediation — live-engine multi-part routing (2026-06-09)

Fixed **C2** (multi-part queries collapsing to a single agent) and **C3** (the `requires_multi_agent` secondary intent being silently dropped) on the live engine. Branch `feat/orchestrator-multipart-routing`; TDD red-first; tests in `tests/unit/test_agents/test_orchestrator/test_multipart_tool_composer_routing.py` (34 cases, no mocks); **362 orchestrator unit tests green**. The design was hardened across **7 adversarial Codex (gpt-5.5) rounds → ACCEPT** (see commit history `6181a3d1..487ba2ef`).

**Final design principle (the discriminator):** route to `tool_composer` **only** when a dependency marker joins **≥2 distinct *recognised* analytical intents** — exactly when `tool_composer`'s sub-question decomposition (`tool_composer/decomposer.py`, which owns the real LLM decomposition + dependency DAG) is useful. Everything else stays single-/parallel-agent. This favours **precision** for the expensive (180 s SLA) `tool_composer`.

| Fix | File | What it does |
|---|---|---|
| **1 — router safety net** | `nodes/router.py` | When `requires_multi_agent` but the (primary, secondary) pair isn't a hard-coded `MULTI_AGENT_PATTERNS` parallel pair: parallel-delegate **primary + top real-domain secondary** instead of silently dropping the secondary (C3). `multi_faceted` as a *secondary* is skipped; a promoted `multi_faceted` *primary* → `tool_composer`. The `MULTI_AGENT_PATTERNS` lookup is **order-insensitive** so a reversed pair keeps its canonical priorities. |
| **2 — classifier pipeline promotion** | `nodes/intent_classifier.py` + `multi_faceted.has_sequential_composition` | A sequential/dependency marker (`then`, `after that`, `based on that/this/these`, `using the … results`, …) joining **≥2 distinct MAPPED strong intents** promotes the primary intent to `multi_faceted` (→ `tool_composer`). **Defers** to a `PARALLEL_INTENT_PAIRS` pair when there are exactly 2 such intents (an incidental leading preamble shouldn't override a deliberate parallel route). |
| **3 — broaden detection** | `multi_faceted.has_sequential_composition` | Broadened the dependency-marker regex (anaphoric `this/these`, `using the <modifier> results`). **No new SSOT `MULTI_FACETED_PATTERNS`** and **no LLM backstop**: a bare `then <verb>` pattern over-routed single asks, and an LLM-escalation backstop only ever fired on single-intent queries — both were rejected in review. `MULTI_FACETED_PATTERNS` stays at the original 4. |

**Consequence (intended):** a multi-part query whose sub-asks the intent regexes don't recognise (only 1 mapped intent) routes to the best single agent rather than over-routing to `tool_composer`.

**Pre-existing, intentionally left as-is:** the original R1 SSOT pattern matches bare `"and then"` as `multi_faceted` (locked by `test_multi_faceted_ssot.py::test_pattern_matches_canonical_phrase[and then]`); on a degenerate repeated single ask this over-routes, but it is a deliberate product contract, not introduced here (REASON-BEFORE-RULES).

**Still open (separate follow-ups, flagged so they are not lost):**
- **C1** — `"what drove …"` (past tense) still misroutes to `general`/explainer (causal token `driv` ≠ `drove`). One-line regex fix; recommended fast follow-up.
- **C5** — verify the project-wide fast-model id `claude-haiku-4-20250414`.
- **A1/B-series** — the orphaned 4-stage `classifier/` package remains unwired (Option A/B/C decision open). This change improves the **live** engine and does **not** wire the orphan.
