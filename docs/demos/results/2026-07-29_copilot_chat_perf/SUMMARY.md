# Copilot Chat Performance Test — 2026-07-29

Recorded run of the full question suite from `docs/demos/COPILOT_CHAT_DEMO_SCENARIOS.md`
(6 narratives + Appendix A robustness = 51 scripted turns; A.7 interrupt is UI-only).

## The two-brain finding (read this first)

The platform has **two distinct chat brains**, and the demo doc's expectations span both:

| Surface | Endpoint | Graph | Behavior |
|---|---|---|---|
| Real copilot UI | `POST /api/copilotkit/agent/default` (CopilotKit protocol) | `chat_node` + `E2I_CHATBOT_TOOLS` + `synthesize_node` (`copilotkit.py`) | Real grounded answers via tools (`kpi_calculate_tool`, …); orchestrator only when the model elects `orchestrator_tool` |
| Scripted chat API | `POST /api/copilotkit/chat/stream` | `classify → orchestrator → generate` (`chatbot_graph.py`) | Orchestrator always dispatches; most agents **fail closed** on conversational queries (#883 discipline: no structured inputs → no fabricated analysis) |

Consequences for this test:

- **Answers record** (`transcripts` / `raw_agui.jsonl`): from the AG-UI surface — this is what a demo audience sees.
- **Routing record** (`raw_shadow.jsonl`, `classification_logs`): from `/chat/stream` — the deterministic
  instrument for the 4-stage ClassificationPipeline wired in PR #1330 (shadow mode).
- The `/chat/stream` answers themselves are mostly fail-closed error summaries. That is a
  pre-existing property of that surface (dispatcher fail-closed merged 2026-06-12), not a regression.

## Run inventory

| Label | Surface | Questions | Mode | File |
|---|---|---|---|---|
| shadow | /chat/stream | 51 | ORCHESTRATOR_CLASSIFIER_MODE=shadow | `measurements_shadow.csv`, `raw_shadow.jsonl` |
| agui | agent/default | 51 | shadow (orchestrator only via orchestrator_tool) | `measurements_agui.csv`, `raw_agui.jsonl`, `transcripts.md` |
| active | /chat/stream | 10 subset | ORCHESTRATOR_CLASSIFIER_MODE=active | `measurements_active.csv`, `raw_active.jsonl` |
| ui | browser (chrome-devtools) | 9 beats | shadow | `ui_pass.md`, `screenshots/` |

`measurements.csv` is the merged Appendix B master: AG-UI answers/timings per question with the
shadow pipeline's routing fields and the active-run dispatch joined alongside.

## Shadow pass — classifier pipeline results (FINAL)

51/51 turns recorded; 47 produced classifications (4 turns crashed pre-classification, see defect D1).

- **Pattern distribution**: CLARIFICATION_NEEDED 25, PARALLEL_DELEGATION 10, SINGLE_AGENT 9,
  TOOL_COMPOSER 1 (+6 blank: 4 errored turns, 2 simple-intent turns that skip the orchestrator).
- **CLARIFICATION_NEEDED = 25/45 (56%)**: the rule-based (LLM-layer-disabled) pipeline abstains on a
  majority of realistic demo questions — confidence lands below the 0.5 active-mode dispatch floor.
  In active mode all of these fall back to legacy routing (safe by design; measured, not assumed).
- **Classifier overhead**: median 0.72 ms, p100 6.97 ms across 45 turns — effectively free, as designed.
- **classification_logs**: 47 rows written by the fire-and-forget writer, 1:1 with classified turns.
- Turn totals on this surface: median 28.4 s, max 287.9 s (6.2 hit the 300 s client timeout budget's
  neighborhood; A.10 completed a real 2-agent parallel run in 117.5 s).

## Defects observed (all pre-existing; none introduced by PR #1330)

- **D1 — simple intents crash /chat/stream**: `greeting` / `help` / `agent_status` questions
  (5.1, 5.2, A.1, A.2a) die with `KeyError: 'low_severity, medium_severity, high_severity'` in
  `E2I_CHATBOT_SYSTEM_PROMPT.format` (generate_node direct-answer path). The two long-standing
  "pre-existing failures" in `tests/integration/test_chatbot_graph.py::TestGenerateNode` are this
  exact bug — the tests were right; prod is broken on that path.
- **D2 — orchestrator LLM intent classify fails JSON parse**: `LLM classification failed: Expecting
  value: line 1 column 1` on every turn → always falls back to regex rules.
- **D3 — RAG embeddings broken on /chat/stream**: `'OpenAIEmbeddingClient' object has no attribute
  'embed'` → hybrid retriever returns 0 results every turn.
- **D4 — session bookkeeping on /chat/stream**: `chatbot_messages` FK violation (`session_id` not in
  `chatbot_conversations`) + audit-chain init fails on `user~uuid` composite ids → scripted-surface
  turns are not persisted to conversation history.
- **D5 — fail-closed answer UX**: most /chat/stream conversational queries return "I was unable to
  complete the analysis..." (dispatcher fail-closed; by design per #883, but the doc's demo answers
  are unreachable on this surface).

## AG-UI pass — the answers record (FINAL)

**51/51 questions answered, zero failures** — including the 4 questions that crash `/chat/stream`
(defect D1) and every T4 composite. Answers are tool-grounded with source citations
(`kpi_calculate_tool` ×15, `e2i_data_query_tool` ×13, `causal_analysis_tool` ×9,
`clinical_context_tool` ×3, `orchestrator_tool` ×2, others ×2). Full Q&A in `transcripts.md`.

- Timings (client): ttfb median 11.2 s (min 4.6 / max 81.3); first_progress median 7.7 s;
  total median 12.7 s. Server-side (`chatbot_analytics`, 53 turns): p50 8.4 s, max 48.7 s.
- `orchestrator_tool` fired on 2.6 and 3.4 — and the PR #1330 shadow classifier logged those
  turns from inside the real UI brain (3 `classification_logs` rows), confirming the
  instrumentation covers the production surface whenever the orchestrator actually runs.
- No hallucination observed on scope edges: 6.1 explicitly declined to attribute share vs
  external competitors absent from the data model; 1.4 declined to quantify a "drop" without
  a clean baseline and offered to pull one.
- `answer_correct` in `measurements.csv` is a heuristic (error-free + >200 chars grounded
  answer) spot-checked against ~10 transcripts; all spot-checks were on-topic and grounded.

## Tier pass/fail vs demo-doc latency budgets (AG-UI pass)

| Tier | Budget | Result | Notes |
|---|---|---|---|
| T1 | < 3 s | **0/2 pass** (8 s, 15 s) | Every turn includes an LLM round-trip + tool call; a 3 s budget is not achievable on this architecture. Recommend re-baselining T1 to < 10 s. |
| T2 | < 8 s | **4/12 pass** (overs mostly 9–18 s; one 52 s outlier 4.6) | Same structural floor; median T2 turn is 8.9 s. |
| T3 | < 40 s | **15/16 pass** | Only 3.6 over (81 s). First-progress < 5 s met on ~25% of turns (median 7.7 s). |
| T3–T4 | < 40 s | 0/1 (3.4 = 75 s) | orchestrator_tool turn — full experiment design. |
| T4 | < 90 s | **6/6 pass** | Max 46 s (A.10 composite). |
| T5 | warm faster than cold | **FAIL** — A.8 cold 4.9 s vs warm 6.2 s | No response/session cache on this path; verbatim repeat re-executes tools. |
| T6 | no hallucination | **PASS** (see above) | Honest scope-limiting and baseline refusals observed. |

## Active-mode subset & routing agreement (FINAL)

10 routing-sensitive questions re-run on `/chat/stream` with `ORCHESTRATOR_CLASSIFIER_MODE=active`
(then reverted to shadow — verified). The pipeline took routing authority exactly where designed:

| Q | Pipeline pattern (conf) | Shadow (legacy) dispatch | Active dispatch | Outcome |
|---|---|---|---|---|
| 1.4 | SINGLE_AGENT (0.510) | causal_impact→explainer | **explainer** | **pipeline engaged** |
| 6.2 | TOOL_COMPOSER (0.527) | drift_monitor, gap_analyzer | **tool_composer** | **pipeline engaged** |
| 1.5 | CLARIFICATION (0.000) | causal_impact→explainer | same | abstained → legacy |
| 1.6 | PARALLEL (0.472 < 0.5) | het_optimizer, gap_analyzer | same | abstained (low conf) |
| 2.5 | CLARIFICATION (0.000) | gap_analyzer | same | abstained |
| 3.4 | CLARIFICATION (0.000) | experiment_designer | same | abstained |
| 4.3 | CLARIFICATION (0.000) | het_optimizer, gap_analyzer | same | abstained |
| 5.7 | PARALLEL (0.469 < 0.5) | explainer | same | abstained (low conf) |
| 6.1 | PARALLEL (0.475 < 0.5) | tool_composer | same | abstained (low conf) |
| 6.4 | CLARIFICATION (0.000) | explainer | same | abstained |

**2/10 engaged, 8/10 abstained to legacy — zero unsafe dispatches.** Verified end-to-end by a
local repro (direct nodes + full compiled graph, both dispatch `[explainer]` for 1.4 in active
mode). The 56% overall CLARIFICATION rate (shadow full pass) means active mode today changes
routing rarely; raising engagement requires the stage-3 LLM layer and richer rules (follow-up).

## UI pass (FINAL — see ui_pass.md)

All demo beats pass: opener pills, contextual per-turn pill refresh (`POST /api/chat/suggestions`),
live progress renderer (25%→75% with step text), visible 5-tool parallel decomposition on 6.1,
A.7 interrupt (both turns complete; episodic-memory recap on the repeat), "Why this agent?"
routing-transparency chip. Two view-layer defects: a captured instance of the known od9uob3
silent-dead-turn (run's POST aborted client-side, no error shown — first capture with the
aborted request in hand), and a cosmetic stale "Working…" header on completed progress cards.

## Cost & DB footprint

898 LLM calls, 1,855,649 input + 199,943 output tokens (2026-07-29 03:20–04:55 UTC window,
all passes incl. UI). 65 `chatbot_training_signals` rows; 60 `classification_logs` rows
(47 shadow full + 10 active + 3 AG-UI orchestrator_tool). Mode verified back on `shadow`,
prod `.env` byte-identical to pre-run state.

## Follow-ups proposed

1. **Fix D1** (simple-intent crash on `/chat/stream`): un-skip the 2 `TestGenerateNode` tests and
   fix `E2I_CHATBOT_SYSTEM_PROMPT.format` brace escaping — the tests were right all along.
2. **D2/D3**: orchestrator LLM intent-classify JSON parse failure; `OpenAIEmbeddingClient.embed`
   attribute error (RAG returns 0 results on every chat turn).
3. **Classifier rule quality**: 56% CLARIFICATION abstention — stage-3 LLM layer (async impl) +
   rule tuning, measured against this run's `classification_logs` baseline.
4. **Re-baseline T1/T2 budgets** in the demo doc to match the architecture (every turn ≥ one
   LLM round-trip).
5. **T5 caching**: verbatim repeats re-execute tools; consider a short-TTL session answer cache.
6. **UI**: silent-dead-turn error surfacing + stale "Working…" header cleanup (od9uob3 family).
