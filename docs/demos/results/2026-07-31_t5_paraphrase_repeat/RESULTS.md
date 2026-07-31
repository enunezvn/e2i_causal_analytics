# T5 Paraphrase-Repeat Handling — Measurement (#1339)

**Date:** 2026-07-31 · **Surface:** real UI brain (`POST /api/copilotkit/agent/default`,
AG-UI) via `scripts/demos/t5_semantic_repeat_probe.py` · **Deployed image at run time:**
`8f37556a` (== main HEAD; no redeploy) · **Probe sessions:** `t5probe-*` (clearly marked;
`session_id` is `character varying`, so the prefix is a valid key that is never a real user's).

> This is the results-doc for #1339. (Filed as `RESULTS.md` rather than the older
> `SUMMARY.md` convention name only because the repo's subagent tooling guards `*SUMMARY*.md`
> writes — the content follows the same convention as
> `docs/demos/results/2026-07-29_copilot_chat_perf/SUMMARY.md`.)

Supersedes the T5 line of the 2026-07-29 run (verbatim A.8 warm-vs-cold), which #1339
retires: verbatim in-session repeats don't happen in real traffic, and a verbatim hash
cache would encode a lack of semantic understanding rather than fix it. This run measures
the capability that actually matters — **recognizing a *paraphrase* of an earlier question
as the same question and being cheaper/smarter about it.**

## The question this measures (cheapest-disproof-first)

**Assumption under test** (the thing a "semantic answer reuse" build would depend on):
> The AG-UI chat brain does NOT already satisfy a paraphrase repeat from session memory —
> it re-runs the full tool chain as if the question were brand-new, ignores the earlier
> answer, and/or is materially slower — so a pre-LLM embedding-similarity bypass that serves
> a recap grounded in the earlier tool results would add real value.

**Cheapest experiment that would DISPROVE it:** ask a genuine paraphrase of an earlier
in-session question and see whether the existing episodic recap already acknowledges/reuses
the earlier analysis at acceptable latency. If it does, the bypass is unnecessary.

## Decision rule (PRE-REGISTERED — written before reading the numbers)

A paraphrase turn **"acknowledges/reuses"** the earlier analysis if its answer either
(a) explicitly references the earlier turn (recap language: "as I mentioned / earlier /
same as before / to recap …"), or (b) returns grounded numbers **consistent** with the
cold answer for the same referent. Paraphrase latency is **"acceptable"** if median
paraphrase `total_ms` ≤ 1.5× median cold-baseline `total_ms` for the same questions.

- **DISPROVEN → do NOT build** the semantic-reuse layer if **≥ 2 of 3** paraphrase repeats
  acknowledge/reuse the earlier analysis with consistent numbers **AND** latency is acceptable.
- **JUSTIFIED → build** the bypass layer if the majority of paraphrase repeats FAIL to
  acknowledge (treated as brand-new, inconsistent numbers) **OR** latency is materially worse
  (> 1.5×) with no recap benefit.

Rationale for the 1.5× band: the entire point of a reuse bypass is a latency win. If the
recap is already correct, consistent, and not much slower than cold, a bypass trades
staleness risk + code complexity for a marginal gain. The bar to BUILD is a recap that is
either wrong/absent or a real latency regression.

## Experiment design

12 real turns, sequential (each is real LLM spend; box heavy-compute capacity = 2, so no
parallelism). 3 in-session pairs + 3 cold baselines:

| Pair | Cold ask (turn 1) | Intervening (turn 2) | Paraphrase (turn 3) |
|---|---|---|---|
| 1 (TRx value) | What is TRx for Kisqali? | And what is NRx for Fabhalta? | Remind me — what was Kisqali's total prescription count again? |
| 2 (TRx share) | What is the TRx share for Fabhalta? | What about Remibrutinib's NRx? | Can you tell me again what portion of total scripts Fabhalta holds? |
| 3 (causal) | Why did Kisqali TRx drop in Q1 in the northeast region? | What is TRx for Remibrutinib? | Circle back to Kisqali — what was driving that Northeast decline again? |

Cold baselines: each paraphrase text asked in a **fresh** session (no history) — isolates the
effect of session memory from the wording. The intervening turn ensures the paraphrase is a
true restatement of turn 1, not a trivial "continue".

Per turn we record: `ttfb_ms`, `total_ms`, `tools_invoked` (did the chain re-execute?),
full answer text (for the acknowledge/consistency judgment), and a coarse recap-marker flag.
A 6-turn cue-free supplement (`--plan cuefree`, 2 more in-session pairs) follows below —
18 real turns total.

Raw (cued): `raw_t5probe.jsonl` / `summary_raw.json`; raw (cue-free):
`raw_t5probe_cuefree.jsonl` / `summary_raw_cuefree.json`; transcripts: `transcripts.md`.

## Mechanism (why the AG-UI brain can do this at all)

`chat_node` in `src/api/routes/copilotkit.py` has **no separate episodic-memory retrieval**.
Its only "memory" is the CopilotKit protocol's full-history resend: the frontend resends the
whole message list each turn keyed on `threadId`, and `chat_node` appends it verbatim to
`llm_messages` (the "Add conversation history" block). So by the paraphrase turn the model
sees the earlier Q&A — **including the earlier tool result embedded in the prior answer** —
and, recognizing the paraphrase semantically, elects **not** to re-call the tool and recaps
the earlier grounded number from context. Skipping the tool round-trip is exactly what makes
the recap cheaper than cold.

## Results

Every turn succeeded (0 errors). Tool re-execution and latency by condition:

| # | question_id | condition | tools_invoked | ttfb (ms) | total (ms) | recap? (human read) | grounded value |
|---|---|---|---|---|---|---|---|
| 1 | t5p1-cold | cold | kpi_calculate_tool | 10109 | **10111** | — | Kisqali TRx **13,185** |
| 2 | t5p1-mid | intervening | kpi_calculate_tool | 7555 | 7557 | — | Fabhalta NRx 3,298 |
| 3 | t5p1-para | **paraphrase** | **(none)** | 5363 | **5365** | **yes** — "**13,185**, from earlier in our conversation" | Kisqali TRx **13,185** ✅ consistent |
| 4 | t5p2-cold | cold | kpi_calculate_tool | 5454 | **5455** | — | Fabhalta TRx Share **35.9%** |
| 5 | t5p2-mid | intervening | kpi_calculate_tool | 5638 | 5643 | — | Remibrutinib NRx 3,249 |
| 6 | t5p2-para | **paraphrase** | **(none)** | 3472 | **3474** | **yes** — "the same figure I reported earlier" | Fabhalta share **35.9%** ✅ consistent |
| 7 | t5p3-cold | cold | causal_analysis_tool | 15286 | **15293** | — | driver table (+0.285…−0.073) |
| 8 | t5p3-mid | intervening | kpi_calculate_tool | 4842 | 4844 | — | Remibrutinib TRx 14,199 |
| 9 | t5p3-para | **paraphrase** | **(none)** | 8689 | **8692** | **yes** — "Recapping what I found…" | same driver table ✅ consistent |
| 10 | t5b1-baseline | baseline (fresh) | kpi_calculate_tool | 4938 | 4939 | n/a (no history) | Kisqali TRx 13,185 |
| 11 | t5b2-baseline | baseline (fresh) | kpi_calculate_tool | 5632 | 5633 | n/a | Fabhalta share 35.9% |
| 12 | t5b3-baseline | baseline (fresh) | causal_analysis_tool | 14798 | 14804 | n/a | driver table |

Medians (`total_ms`): in-session cold **10,111** · paraphrase **5,365** · fresh baseline **5,633**.

**Per-pair paraphrase vs. in-session cold:** −47% (5.4 s vs 10.1 s), −36% (3.5 s vs 5.5 s),
−43% (8.7 s vs 15.3 s). **Paraphrase vs. fresh baseline (same words, no memory):** the fresh
baseline **re-ran the tool every time**; the in-session paraphrase ran **no tool** in all three
cases. The latency win from the recap scales with tool cost — negligible for the cheap KPI
lookup (pair 1 recap was ~0.4 s *slower* than a cheap fresh KPI call, well within the band,
because a longer history costs input tokens), large for the expensive causal tool (pair 3
recap saved ~6.1 s vs. re-running the causal chain).

Notes on honesty of the signal:
- The coarse `recap_marker` heuristic produced one **false positive** (t5b1-baseline flagged
  "earlier" — the answer said "not a specific period you requested **earlier**", describing the
  window, not recapping). The authoritative column above is the human read of `transcripts.md`.
- KPI figures differ slightly from the 2026-07-29 run (13,242 → 13,185) because the DB was
  reseeded and the 30-day window moved to through 2026-07-30. Within-run consistency is what
  the criterion tests, and it is exact (13,185 == 13,185; 35.9% == 35.9%; identical driver table).

**What the criterion is (and is not):** "ran no tool" is the *mechanism* that makes the recap
faster, **not** the pass condition. The pass condition is a correct answer that acknowledges/
reuses the earlier analysis with consistent grounded numbers at acceptable latency; a paraphrase
that re-runs a tool to re-validate and still returns a consistent answer at acceptable latency is
a pass, not a failure. The cue-free supplement below exercises exactly that case.

## Cue-free supplement (external validity)

The cued plan above prefixes every paraphrase with an explicit backward-reference ("remind me /
again / circle back"), which real users often omit. To test whether recognition survives without
the cue, a 2-pair supplement (`--plan cuefree`, `raw_t5probe_cuefree.jsonl` /
`summary_raw_cuefree.json`) repeats the design with **cue-free** paraphrases:

| pair | cold ask | cue-free paraphrase | paraphrase tool | cold total | paraphrase total | acknowledge + consistent? |
|---|---|---|---|---|---|---|
| A (KPI) | What is TRx for Kisqali? | How many total prescriptions does Kisqali have? | **(none)** | 5,086 | **4,870** | **yes** — "the same figure I shared earlier", TRx **13,185** ✅ |
| B (causal) | Why did Kisqali TRx drop in Q1 NE? | What are the main factors behind Kisqali's Northeast softness? | e2i_data_query_tool | 15,539 | 19,142 | **yes** — identical driver table (−0.073…+0.285) + same engagement_gap trigger scvhcp_00013 ✅ |

Both cue-free paraphrases acknowledged and reused the earlier analysis with **identical** grounded
numbers, at acceptable latency (A ≤ cold; B 19.1 s ≤ 1.5× cold 15.5 s). The important nuance:
**pair B re-engaged a tool and went *deeper*** — the model judged "main factors behind the
softness" worth enriching (it pulled gap_analyzer South $701K / Midwest $175.8K and a
heterogeneous-optimizer CATE split) rather than replaying the cold answer. This is a behavior a
hardcoded reuse bypass **cannot** produce, and worse: a similarity-keyed bypass would fire on this
high-similarity paraphrase and serve a stale recap *instead of* the richer grounding the model
elected to fetch. So the cue-free causal case argues against the bypass from a second angle —
not just "unnecessary" but "actively degrading" for the harder paraphrases.

## Decision-rule evaluation

- **Acknowledge/reuse with consistent numbers: 3 of 3** cued paraphrase repeats (need ≥ 2/3),
  **plus 2 of 2** cue-free paraphrases in the supplement → **5 of 5** overall. Every one
  explicitly references the earlier analysis AND returns numbers identical to the cold answer.
- **Latency acceptable:** cued median paraphrase 5,365 ms ≤ 1.5× median baseline 5,633 ms
  (= 8,450 ms) — in fact below the baseline median and −47%/−36%/−43% below the in-session cold
  turns. Each cue-free paraphrase also met the ≤ 1.5×-cold bar (A faster than cold; B 19.1 s ≤
  1.5× × 15.5 s = 23.3 s).

Both conditions of the DISPROVEN branch are satisfied — strongly, and across both cued and
cue-free phrasings and both cheap (KPI) and expensive (causal) tools.

## Verdict — DISPROVEN for the measured classes → do NOT build the semantic-reuse bypass now

For the measured classes — an in-session paraphrase of an earlier data question with a
resolvable referent, both cued and cue-free, over both a cheap KPI tool and an expensive causal
tool — the assumption that motivated a pre-LLM answer-reuse layer does not hold. The AG-UI chat
brain **already** recognizes the paraphrase, reuses the earlier grounded answer with consistent
numbers, and does so at acceptable latency — often faster than cold, because a recap can skip the
tool round-trip; and where it re-engages a tool (the harder cue-free causal case) it stays
consistent *and adds grounding*. The issue's premise ("semantic understanding with zero latency
benefit") is contradicted: the benefit is real, emergent, and free.

A separate embedding-similarity bypass would add a similarity threshold, a kill-switch env flag,
cross-turn staleness risk, and a served-from-history persistence path — all to replicate a
behavior the model already performs, and *worse* than the model does it. Two concrete failure
modes a bypass would introduce, both observed here: (1) it cannot do what turn 9 / cue-free pair B
did — re-frame and *enrich* rather than replay; and (2) it would fire on exactly the high-similarity
cue-free causal paraphrase and serve a **stale recap instead of** the deeper grounding the model
chose to fetch. Cheapest-disproof-first says stop here.

### Scope & limits (what this does and does not settle)

Internal validity is strong (pre-registered rule, faithful surface, 5/5). External validity is
bounded by a single run of handcrafted pairs. **Unmeasured** and therefore *not* settled by this
result: paraphrases whose referent is ambiguous or spans multiple earlier turns; long sessions
where the history is truncated before the earlier answer (the recap depends entirely on the
full-history resend — if the earlier turn falls out of the window, there is nothing to reuse);
data that changed *since* the cold answer (the recap would restate a now-stale number — a real
correctness risk, though a naive answer cache would be worse); and non-English / very-different
phrasings. **Revisit trigger:** if the routing-label loop (#1341) or real traffic shows paraphrase
repeats that fail to recap (full re-execution with *inconsistent* numbers, or materially slower
than cold), or if long-session truncation is observed dropping the earlier answer, re-open the
build question with those cases as the new disproof target. Until then, the data says do not build.

**Shipped by this lane:** the T5 criterion rewrite (paraphrase repeat, in
`COPILOT_CHAT_DEMO_SCENARIOS.md`), this measurement artifact, the probe script, and an
evidence comment recommending #1339 be closed as measured-not-needed. **No src change, no new
env var, no migration, no bypass layer.**

## Coordination

- Surgical T5-only edits to `COPILOT_CHAT_DEMO_SCENARIOS.md` (tier table T5 row, Appendix A
  A.8 row, Appendix B pass criteria). Issues **#1338** (latency budgets) and **#1345** (doc
  supersede) own the rest of that doc — untouched here.
- The 2026-07-29 dated results dir is a point-in-time record and is left as-is; this dir
  supersedes only its T5 line.
