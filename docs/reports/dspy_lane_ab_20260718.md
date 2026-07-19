# DSPy-lane provider A/B — terra vs sonnet-5 vs haiku-4-5 (2026-07-18/19)

Executes `.claude/plans/dspy_lane_anthropic_flip_plan.md` (Phases 1–3).
Baseline: `openai/gpt-5.6-terra` (current `DSPY_LM_MODEL`). Candidates:
`anthropic/claude-sonnet-5`, `anthropic/claude-haiku-4-5-20251001`.
All measurements are REAL calls through production code paths in the
`e2i_api` container (deployed main d1cec7f7) with production keys, data,
and retrieval — no mocks anywhere.

Harness: `scripts/run_dspy_lane_ab.py` (+ `src/optimization/dspy_lane_ab.py`),
golden set `tests/fixtures/dspy_lane_golden_queries.json` (30 real production
queries from `chatbot_training_signals`, 5 disproof queries, 5 synthetic
intent-coverage queries; acceptable-set hand labels; case-sensitive scoring
because production compares intent strings exactly).

## 1. Signature-level A/B (n=40 queries × 2 signatures × 3 models = 240 calls)

Both real intent surfaces riding `DSPY_LM_MODEL`: cognitive RAG
`IntentClassificationSignature` (6 intents) and chatbot
`ChatbotIntentClassificationSignature` (9 intents). Models interleaved
per query; `cache=False`; zero parse failures and zero provider errors
across all 240 calls.

| model | cognitive acc (n=40) | chatbot acc (n=38) | cognitive p50/p95 | chatbot p50/p95 |
|---|---|---|---|---|
| terra (baseline) | 0.850 | 0.947 | 1.48s / 2.06s | 1.19s / 2.12s |
| sonnet-5 | **0.875** | 0.947 | 4.05s / 6.08s | 2.63s / 4.61s |
| haiku-4-5 | 0.800 | 0.921 | 1.93s / 2.77s | 1.79s / 2.82s |

- sonnet-5 *beats* the baseline on cognitive intent accuracy and ties on
  chatbot accuracy, at ~2.7× the per-call latency.
- haiku-4-5 lands exactly at the −5pp accuracy margin on cognitive (0.800 vs
  0.850) and inside it on chatbot (0.921 vs 0.947), at ~1.3–1.5× latency.
- Strict and lenient accuracy are identical for every model: all three emit
  exact-case intent labels (the temperature/casing failure modes hypothesized
  pre-disproof did not materialize).

## 2. End-to-end RAG replays (n=10 queries × 3 models)

Full `CausalRAG.cognitive_search` (the `/api/cognitive/rag` path: ~19 LLM
calls/turn, real memory backends + retrieval), fresh process per model with
per-process `DSPY_LM_MODEL` — mechanically identical to how the flip itself
works. Two interleaved half-batches per model to spread provider-load drift.

RESULTS PENDING (runs in progress).

## 3. RAGAS judge (frozen gpt-4o) on the replay answers

Each candidate's REAL generated answers + actually-retrieved contexts from
section 2, judged by the production `RAGASEvaluator`. Faithfulness averaged
only over samples that retrieved ≥1 context (zero-context faithfulness is an
artifact); answer-relevancy over all samples.

RESULTS PENDING.

## 4. Insight panels side-by-side (5 panels × 3 models)

`scripts/run_dspy_lane_insights_ab.py`: home_kpi, executive_brief,
knowledge_graph, feedback_learning, experiments — all with fully
server-derived groundings (the originally proposed hte / causal_discovery /
treatment_effect panels take frontend-posted figures or expired persisted
analyses; generating them would have required fabricating inputs). Route
Redis cache deliberately bypassed (its key ignores the LM).

RESULTS PENDING.

## 5. feedback_learner_pattern GEPA module — honest limits

A real GEPA-metric re-evaluation is IMPOSSIBLE on current data: zero
persisted `feedback_learner` signals meet the production converter's own
gold bar (`reward >= 0.5`; max recorded reward is 0.42, n=135). Fabricating
gold examples would be mock data. Instead a functional smoke runs the
deployed optimized module (`gepa_v1_feedback_learner_pattern_20260608`) on
the real recorded inputs that carry patterns, checking structured-output
validity per LM plus a type-overlap diagnostic (explicitly NOT a gate).

RESULTS PENDING.

## 6. Pre-registered gates (plan §5) — verdict

RESULTS PENDING.

## 7. Side effects & cleanup

E2e replays exercise the real Reflector phase. Replay-written rows are
tagged `dspy-ab-20260718-<model>-<query_id>` in their conversation ids;
counts checked and cleaned post-run. (Signal-buffer flush thresholds meant
zero `dspy_agent_training_signals` rows had landed as of the mid-run check.)
