# Routing benchmark (#1337 Step 0)

A 337-query benchmark for the chat routing layer, plus a two-axis gold stage.
Step 0 of #1337 measures routing-classifier candidates (the 4-stage pipeline, a
single-LLM-call classifier, and legacy `INTENT_TO_AGENTS` routing) against a
**gold** routing label per query. This directory holds the corpus, the gold
labels, and the answer-quality evaluation of the live chatbot surface.

The one rule that governs everything here: **gold is decided by contract
ownership, not by what any classifier can currently reach.** The verified
14-agent contract registry `data/agent_contracts.json` is the single source of
truth. Classifier coverage gaps, fail-closed input resolvers, dispatch-budget
timeouts, and known crashes (e.g. the `tool_composer` #1350 crash) are findings
the benchmark *exposes* — they never constrain a gold label.

## Pipeline

```
pool ─┐
      ├─(demo 51 + historical 115)──► query_pool.jsonl
perturbations ─(haiku variants of demo)──► perturbations.jsonl
authored ─(hand-authored probe cells)──► authored_queries.jsonl
                                    │
                        assemble_benchmark.py
                                    ▼
                      benchmark_queries.jsonl  (337 rows, bench-NNNN)
                                    │
              empirical pass (22 disputed TOOL_COMPOSER queries)
        forced-route dispatch + live AG-UI ──► review/empirical_results/
                                    │
                    human review (gen_judge_review.py)
             review/tool_composer_corrections.md ─► proposed_verdicts.json (22 finals)
                                    │
                          ┌─────────┴─────────┐
                    gold_judge.py       build_answer_quality.py
                    (AXIS 1: routing)   (AXIS 2: answer quality)
                          ▼                     ▼
          benchmark_queries_gold.jsonl   answer_quality_22.json
                          └────────► GOLD_STAGE_RESULTS.md ◄────────┘
```

1. **pool** — `build_query_pool.py` → `data/query_pool.jsonl`. Two free sources:
   real `chatbot_messages` traffic (fragments, typos, pronoun follow-ups, each
   carrying its prior-turn context) + the 51 doc-authored demo questions
   (tier/intent-annotated).
2. **perturbations** — `gen_perturbations.py` → `data/perturbations.jsonl`. Haiku
   variants of the demo questions (paraphrase always, plus one rotating
   typo/fragment/pronoun_followup) so the set isn't biased toward clean,
   rule-classifier-friendly English. Each variant keeps `parent_query_id` +
   `perturbation_type`.
3. **authored** — `data/authored_queries.jsonl`. Hand-authored probe cells that
   carry the author's own gold in `authored_gold_pattern`/`authored_gold_agents`
   (e.g. the `unmapped:resource_optimizer` cell, genuinely-ambiguous
   CLARIFICATION cells, and multi-domain-dependent TOOL_COMPOSER cells).
4. **assemble** — `assemble_benchmark.py` → `data/benchmark_queries.jsonl`. Merges
   the three sources in stable order, dedupes on normalized text, assigns
   `bench-NNNN` ids (source ids kept as `source_query_id`).
5. **empirical pass** — `empirical_pass.py` (forced-route dispatch through the real
   RouterNode/DispatcherNode) + `scripts/demos/copilot_agui_runner.py` (live AG-UI
   answers) over the 22 disputed TOOL_COMPOSER-correction queries → artifacts in
   `review/empirical_results/`. A fail-closed `NeedsStructuredInput` result is
   evidence, not a harness bug.
6. **human review** — `gen_judge_review.py` renders `review/tool_composer_corrections.md`
   (contract reference + proposed verdicts + empirical evidence per query); the
   2026-07-29 interactive review recorded the 22 ratified `final` verdicts in
   `review/proposed_verdicts.json`.
7. **gold stage** — this increment:
   - `gold_judge.py` (**axis 1**) labels all 337 rows → `data/benchmark_queries_gold.jsonl`.
   - `build_answer_quality.py` (**axis 2**) grades the 22 live answers →
     `review/answer_quality_22.json`.
   - Results + methodology in `GOLD_STAGE_RESULTS.md`; protocol in `GOLD_STAGE_PROTOCOL.md`.

## Artifacts

### `data/`
- **`agent_contracts.json`** — the verified 14-agent contract registry (SSOT). `agents`
  (covers/does_not_cover per agent), `routing_patterns`, `composition_ruling`,
  `intent_to_agents`, `classifier_domain_to_agent`, `router_reachable_agents`,
  `not_chat_routable`. Never distill contracts from memory — read this.
- **`query_pool.jsonl`** — demo + historical pool (stage 1).
- **`perturbations.jsonl`** — haiku demo variants (stage 2).
- **`authored_queries.jsonl`** — authored probe cells with author gold (stage 3).
- **`benchmark_queries.jsonl`** — the assembled 337-row benchmark (stage 4). Fields:
  `text`, `source` (historical/perturbation/authored/demo), `is_followup`, `context`,
  `demo_meta`, `perturbation_type`, `parent_query_id`, `authored_gold_pattern`/`_agents`,
  `query_id`, `source_query_id`. `gold_pattern`/`gold_agents` are null here.
- **`benchmark_queries_gold.jsonl`** — the gold-labeled output (stage 7, axis 1): every
  input row + `gold_pattern`, `gold_agents`, `gold_confidence`, `gold_rationale`,
  `gold_source` (`human-ratified` / `authored-ground-truth` / `llm-judge`), and the
  independent `judge_pattern`/`judge_agents`/`judge_confidence` cross-check.

### `review/`
- **`proposed_verdicts.json`** — the 22 human-ratified verdicts (`final` per query),
  authoritative calibration anchors.
- **`tool_composer_corrections.md`** — the human-review worksheet (contract reference +
  proposed verdicts + empirical evidence). Contains the five ambiguity resolutions.
- **`empirical_manifest.json`**, **`empirical_live_questions.json`** — the 22-query manifest
  and the emitted live-questions file for the AG-UI runner.
- **`empirical_results/`** — forced-route dispatch records (`forced_qNN_<agent>.json`), the
  live AG-UI answers (`raw_empirical22.jsonl`), classifier measurement
  (`classifier_measurement_22.json`), and the gold judge's resumable checkpoint
  (`gold_progress.jsonl`).
- **`gold_low_confidence.md`** — the 52 llm-judged rows below the 0.6 confidence floor,
  for human follow-up.
- **`answer_quality_22.json`** — axis-2 per-question, three-layer grades + DB-verified claims.

### Root
- **`GOLD_STAGE_PROTOCOL.md`** — the governing protocol (axis 1 + axis 2 three layers).
- **`GOLD_STAGE_RESULTS.md`** — the gold-stage results (both axes, distributions,
  disagreements, cost).

## Re-running

```bash
# axis 1 — routing gold (idempotent/resumable; ~$10 one-time on claude-sonnet-5)
.venv/bin/python scripts/benchmarks/routing/gold_judge.py
.venv/bin/python scripts/benchmarks/routing/gold_judge.py --assemble-only   # rebuild from checkpoint

# axis 2 — answer quality (no LLM; DB-backed human grades)
.venv/bin/python scripts/benchmarks/routing/build_answer_quality.py
```

Env: the gold judge reads `ANTHROPIC_API_KEY` from `.env` (never printed/committed);
`GOLD_JUDGE_MODEL` / `GOLD_JUDGE_CONCURRENCY` override the defaults. Axis-2 DB spot-checks
use `docker exec supabase-db psql` with READ-ONLY SELECTs only.
