# DSPy-lane provider A/B — terra vs sonnet-5 vs haiku-4-5 (2026-07-18/19)

Executes `.claude/plans/dspy_lane_anthropic_flip_plan.md` (Phases 1–3).
Baseline: `openai/gpt-5.6-terra` (current `DSPY_LM_MODEL`). Candidates:
`anthropic/claude-sonnet-5`, `anthropic/claude-haiku-4-5-20251001`.
All measurements are REAL calls through production code paths in the
`e2i_api` container with production keys, data, and retrieval — no mocks
anywhere. (Deployed main moved d1cec7f7 → 71de56c0 mid-run when the
PR #1280 docs deploy converged — docs-only diff, identical backend src;
see §7 for the container-recreation collision this caused.)

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

| model | n | errors | p50 | p95 | mean hops | mean evidence calls | mean answer chars |
|---|---|---|---|---|---|---|---|
| terra (baseline) | 10 | 0 | 39.6s | 69.3s | 3.0 | 1.4 | 2228 |
| sonnet-5 | 10 | 0 | 88.8s | 110.0s | 3.0 | 2.8 | 1863 |
| haiku-4-5 | 10 | 0 | 53.2s | 86.8s | 3.0 | 3.0 | 1763 |

- Latency gate (p50 ≤ 1.5× baseline = 59.4s): **sonnet-5 FAILS at 2.24×**
  (exactly as the pre-plan D3 disproof predicted); **haiku passes at 1.34×**.
- Zero errors and zero provider error classes across all 30 replays → the
  no-new-error-class gate passes for both candidates.
- The Anthropic models drive ~2× more evidence-tool calls per turn (2.8–3.0
  vs 1.4) — that is why haiku's e2e latency exceeds baseline despite being
  faster per call at signature level. Behavioral, not a failure.
- E2e detected-intent-in-golden-set (diagnostic, judged on the full pipeline's
  intent phase): terra 10/10, sonnet-5 9/10, haiku 7/10.

## 3. RAGAS judge (frozen gpt-4o) on the replay answers

Each candidate's REAL generated answers + actually-retrieved contexts from
section 2, judged by the production `RAGASEvaluator`. Faithfulness averaged
only over samples that retrieved ≥1 context (zero-context faithfulness is an
artifact); answer-relevancy over all samples.

| model | faithfulness | n w/ contexts | answer relevancy (n=10) |
|---|---|---|---|
| terra (baseline) | 0.690 | 3 | 0.401 |
| sonnet-5 | 0.122 | 7 | 0.465 |
| haiku-4-5 | 0.339 | 8 | 0.529 |

- **Both candidates fail the faithfulness gate** (floor = 0.690 − 0.05 =
  0.640). Both pass answer-relevancy (floor 0.351) — in fact both beat the
  baseline on it.
- **Denominator caveat, checked**: faithfulness averages over different
  sample subsets (terra retrieved contexts in only 3/10 replays vs 7–8/10
  for the Anthropic models — retrieval frequency itself rides the LM's hop
  decisions). Robustness re-aggregation on the common context-bearing subset
  (syn-1, ts-12, ts-346; n=3 for all models): terra 0.690, haiku 0.523,
  sonnet-5 0.000. Same verdict — haiku still misses the floor, sonnet
  collapses. The confound does not change the gate outcome.
- Pattern worth noting: sonnet-5 scored 0.0 faithfulness on 6 of its 7
  context-bearing samples (its answers assert claims the judge cannot ground
  in the retrieved contexts), while writing the second-longest answers. Terra
  retrieves less but stays grounded in what it retrieves.

## 4. Insight panels side-by-side (5 panels × 3 models)

`scripts/run_dspy_lane_insights_ab.py`: home_kpi, executive_brief,
knowledge_graph, feedback_learning, experiments — all with fully
server-derived groundings (the originally proposed hte / causal_discovery /
treatment_effect panels take frontend-posted figures or expired persisted
analyses; generating them would have required fabricating inputs). Route
Redis cache deliberately bypassed (its key ignores the LM).

All 15 panel×model runs succeeded with substantive output (no errors, no
empty insights):

| model | median latency | total (5 panels) | insight length range |
|---|---|---|---|
| terra (baseline) | 6.3s | 37.1s | 425–1640 chars |
| sonnet-5 | 20.3s | 110.1s | 906–1764 chars |
| haiku-4-5 | 13.9s | 60.8s | 951–2400 chars |

Both Anthropic models write longer panel insights; both are 2–3× slower per
panel. Insight panels are generated server-side (not interactive), so panel
latency is a soft factor, not a gate. Full side-by-side texts:
`insights_ab.log` (scratchpad artifact, RESULTS_JSON block).

## 5. feedback_learner_pattern GEPA module — honest limits

A real GEPA-metric re-evaluation is IMPOSSIBLE on current data: zero
persisted `feedback_learner` signals meet the production converter's own
gold bar (`reward >= 0.5`; max recorded reward is 0.42, n=135). Fabricating
gold examples would be mock data. Instead a functional smoke runs the
deployed optimized module (`gepa_v1_feedback_learner_pattern_20260608`) on
the real recorded inputs that carry patterns, checking structured-output
validity per LM plus a type-overlap diagnostic (explicitly NOT a gate).

Results (6 real cases per model, 18 runs total, 0 errors):

| model | cases | errors | all structured | mean diag. score | latency (min–max, mean) |
|-------|-------|--------|----------------|------------------|-------------------------|
| openai/gpt-5.6-terra | 6 | 0 | no (2 empty-pattern outputs) | 0.700 | 1.9–10.2s, 6.6s |
| anthropic/claude-sonnet-5 | 6 | 0 | no (2 cases with pattern dicts missing `type`/`pattern_type`) | 0.700 | 7.8–34.5s, 24.5s |
| anthropic/claude-haiku-4-5-20251001 | 6 | 0 | **yes** (6/6) | 0.700 | 3.6–22.1s, 11.4s |

Reading the results honestly:

- **The diagnostic score is non-discriminative here**: every one of the 18
  runs scored exactly 0.700, including terra's two empty-pattern outputs.
  The GEPA metric as invoked on these sub-bar inputs does not separate the
  models; the informative columns are structure and latency. This reinforces
  that the smoke is a functional check, NOT a gate — it played no role in
  the §6 verdict.
- Structure: haiku was the only model producing well-formed typed pattern
  dicts on all 6 cases; terra returned empty pattern lists on the 2 smallest
  inputs; sonnet emitted patterns on all cases but 2 had dicts missing the
  `type`/`pattern_type` key.
- Latency ordering (terra < haiku < sonnet, roughly 1× / 1.7× / 3.7×)
  matches the e2e replay ordering in §3.

Three incidents hit while running the smoke, all recorded as findings in §7:

1. The GEPA artifact is host-only (untracked, never in CI-built images) — the
   smoke had to stage it into container `/tmp` via stdin pipe (docker cp
   cannot write into a tmpfs mount).
2. The artifact's saved `module_state` embeds the RETIRED
   `anthropic/claude-sonnet-4-20250514` as a per-predictor `lm` pin, which
   overrides `dspy.context(lm=...)` and 404s — the smoke strips `pred.lm`
   after load; the shipped artifact is unusable as-is even if it did ship.
3. The GEPA metric returns a plain dict (not a score object) — score
   extraction must branch on `isinstance(score, dict)`.

## 6. Pre-registered gates (plan §5) — verdict

**Both candidates FAIL → data-driven NO-FLIP. `DSPY_LM_MODEL` stays
`openai/gpt-5.6-terra`.**

### anthropic/claude-sonnet-5 — FAIL (2 gates)

| gate | result | detail |
|------|--------|--------|
| parse_failure[cognitive_rag] | PASS | 0.000 vs baseline 0.000 |
| accuracy[cognitive_rag] | PASS | 0.875 vs baseline 0.850 (margin 0.05) |
| parse_failure[chatbot] | PASS | 0.000 vs baseline 0.000 |
| accuracy[chatbot] | PASS | 0.947 vs baseline 0.947 (margin 0.05) |
| ragas[faithfulness] | **FAIL** | 0.122 vs baseline 0.690 (margin 0.05) |
| ragas[answer_relevancy] | PASS | 0.465 vs baseline 0.401 (margin 0.05) |
| ragas[completeness] | PASS | judged/requested replays 10/10 both sides |
| no_new_error_class | PASS | none |
| e2e_latency_p50 | **FAIL** | 88.8s vs limit 59.4s (1.5× baseline 39.6s) |

### anthropic/claude-haiku-4-5-20251001 — FAIL (1 gate)

| gate | result | detail |
|------|--------|--------|
| parse_failure[cognitive_rag] | PASS | 0.000 vs baseline 0.000 |
| accuracy[cognitive_rag] | PASS | 0.800 vs baseline 0.850 (margin 0.05, boundary) |
| parse_failure[chatbot] | PASS | 0.000 vs baseline 0.000 |
| accuracy[chatbot] | PASS | 0.921 vs baseline 0.947 (margin 0.05) |
| ragas[faithfulness] | **FAIL** | 0.339 vs baseline 0.690 (margin 0.05) |
| ragas[answer_relevancy] | PASS | 0.529 vs baseline 0.401 (margin 0.05) |
| ragas[completeness] | PASS | judged/requested replays 10/10 both sides |
| no_new_error_class | PASS | none |
| e2e_latency_p50 | PASS | 53.2s vs limit 59.4s (1.5× baseline 39.6s) |

Haiku came closest — it fails only faithfulness, and the §3 robustness
re-aggregation confirms that failure isn't a denominator artifact (0.523 vs
0.640 floor on the common subset). The plan's Phase 4 flip is therefore NOT
executed; the pre-registered rollback/no-flip branch applies. If a flip is
ever revisited, the data points at haiku + a grounding-behavior fix (its
extra evidence calls retrieve more but its answers ground less), not at
sonnet-5 (unfixable 2.24× latency at equal cost).

## 7. Side effects, incidents & cleanup

- **Episodic writes structurally rejected (zero contamination).** The replay
  conversation ids (`dspy-ab-20260718-<model>-<query_id>`) are not UUIDs, so
  every `episodic_memories` insert attempted by `store_episode` fails with
  Postgres 22P02 and is caught/logged — the replay itself continues
  unaffected (the store happens after answer generation and never lands in
  the run's error field). Whether a run attempts the store at all is
  LM-dependent (only haiku triggered it in the first batch) — noted as a
  behavioral diagnostic, not a gate input. Consequence: no episodic cleanup
  needed; only signal-buffer tables checked post-run.
- **Deploy-convergence collision (not OOM).** The `Deploy to Production` run
  fired by PR #1280's merge (23:33Z, docs-only) converged at 00:08Z mid-run:
  `docker compose up -d` recreated `e2i_api`, gracefully stopping the old
  container — which SIGKILLed haiku's in-flight half-a exec (exit 137) — and
  the three half-b launches hit the ~25s recreation gap ("container is not
  running"). Host and container memory were healthy throughout (container
  1.7GiB/5GiB after restart). The four lost runs were re-run behind a guard
  (container healthy + in-container mem < 3GiB + zero in-flight deploy
  workflow runs). Rerunning the same conversation ids is clean: the LangGraph
  checkpointer is in-process `MemorySaver`, and the only cross-process
  session store is the episodic backend whose writes are rejected (above).
- **Comparability across the restart**: the rebuilt image carries an
  identical backend `src/` (docs-only merge); Redis/FalkorDB/Supabase never
  restarted, so retrieval backends stayed warm for all runs.
- **DISCOVERED PROD GAP: the GEPA-optimized `feedback_learner_pattern`
  module has never shipped in any CI-built image.** The §5 smoke first
  failed with `FileNotFoundError: No saved modules for agent:
  feedback_learner_pattern` — the only artifact
  (`optimized_modules/feedback_learner_pattern/gepa_v1_..._20260608_235145.json`)
  lives untracked on the droplet host, `optimized_modules/` appears in no
  Dockerfile COPY, compose mount, or git tree, and CI builds images from the
  git checkout. Production's `pattern_analyzer._load_optimized_pattern_module`
  catches the miss and **silently falls back to the un-optimized module**
  (DEBUG-level log only) — so the fallback, not the tuned module, has been
  serving every feedback_learner cycle in every deployed container. The
  smoke was re-run with the artifact staged into the container's ephemeral
  `/tmp` (tmpfs; prod probes only `./optimized_modules`, so prod behavior
  untouched). Fix options (user decision, out of A/B scope): commit the
  artifact to git so images carry it, or mount `optimized_modules/` in
  compose, or accept the fallback and delete the artifact.
- **Signal-buffer sweep (final)**: zero rows in
  `dspy_agent_training_signals` and `chatbot_training_signals` reference the
  replay prefix. The cognitive Reflector DID write **96 learner-visible
  `learning_signals` rows** (`signal_type=rating`,
  `is_training_example=True`) during the 30 replays. All 96 were neutralized
  in place — `is_synthetic=True, is_training_example=False` (learner queries
  filter `is_synthetic=False`); hard deletion was withheld (destructive-op
  guardrail) and left to the user:
  `delete from learning_signals where is_synthetic and signal_details::text like '%dspy-ab-20260718%'`.
