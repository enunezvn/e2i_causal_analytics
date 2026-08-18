# 51-Turn Copilot Chat Eval — 2026-08-18 post-#1690 rerun

**Verdict: 37 PASS / 14 PARTIAL / 0 FAIL** (morning baseline same day: 37 / 13 / 1).
The one FAIL is gone (3.3, champion outage → PASS). Per-subset: sessions 1–2 = 9/5/0,
sessions 3–4 = 7/6/0, sessions 5–6 = 10/2/0, appendix = 11/1/0.

## Why this run exists

The morning 08-18 run carried two environment confounds: the #1690 Monday
champion-demotion (3 hcp_adoption production champions demoted by the weekly
retrain UPSERT) and the #1688 ChEMBL/EBI outage. Both were resolved before this
rerun (champions re-promoted 15:09 UTC via the #1384 script, PR #1692 makes it
permanent; ChEMBL recovered upstream). **Same container image both runs**
(`ghcr.io/enunezvn/e2i-api:1a17b73e7…` — #1692 touched only scripts/tests, no
deploy fired), so every behavioral difference is environment or LLM/orchestrator
variance, never code.

Preconditions measured at run start (~15:46 UTC): all 3
`hcp_adoption_*_goldstd_lr_v1` rows `production|is_champion|promoted_at
2026-08-18 15:09:46`; ChEMBL molecule + mechanism endpoints 200; API healthy.
Runner: 51/51 OK, 0 failed, wall Σtotal 1,158.5 s (morning 1,113.4 s). Grading:
4 parallel graders (n1n2 / n3n4 / n5n6 / appendix), every numeric claim traced
to `on_tool_end` payloads in the raw event stream — never the CSV
`agents_dispatched`/`tools_invoked` columns, which all four graders again
confirmed record only the first tool per turn.

## Environment fixes: verified live, to the digit

- **#1690 champion restore — VERIFIED in 5 turns.** 1.6 and 2.5 serve
  `hcp_adoption_kisqali_goldstd_lr_v1` (AUC 0.7907) and
  `hcp_adoption_remibrutinib_goldstd_lr_v1` (AUC 0.7945); 3.3 serves
  `hcp_adoption_fabhalta_goldstd_lr_v1` (AUC 0.811, n_scored 5000, all 7
  segments) — FAIL → PASS; 6.1 recovers the full 7-specialty ranking; 5.3
  (unlisted in the brief) now returns the real champion + AUC where morning had
  "0 results". Every restored figure matches the 08-15 pre-outage run to the
  digit — the restore returned the same models, not retrained substitutes.
  Correction to the morning attribution: **6.2 was never a champion-outage
  turn** — raw events show it never calls `predict_hcp_segment_likelihood_tool`
  in either run.
- **ChEMBL — VERIFIED live in-payload** on 3.1, 6.4, A.9-followup
  (`mechanism.source == "chembl"`; `static_fallback` occurs zero times).

## What reproduces (the step-3 fix list)

The defining result: **the three defect families the rerun was meant to test all
reproduced as families while every individual morning instance cleared.** The
per-turn incidence is stochastic; the classes persist on the same image.

1. **#1691 prose-vs-table (summary layer) — REPRODUCES, moved turns.**
   Morning's 3 instances: 5.1 ("highest … at 3 each", table/payload say 2 —
   caught only on re-grade), 5.6 ("lowest 0.231" over a 0.224 row), 5.7
   ("largest 0.198" under a 0.267 row). None recurs — but 5.7 is the only
   *genuine* clean (explicit "4th-strongest of 6" ordinal verified exact);
   5.1/5.6 took different tool routes and never printed a comparable table
   (clean by absence). Meanwhile the rerun added **new instances in new
   turns**: 6.4 ("fastest-materializing, 30-day lag" for a driver whose own
   table row and the very next sentence say 18 days — a 67% margin), 2.5
   (adjacent rows annotated "Largest cohort by n" (1,016) and "Largest n
   overall" (1,662)), and 3.6 (empty `feasibility_warnings` list converted into
   "This design IS feasible" for a 3,876-cluster design against a 5,000-HCP
   universe). That is 6+ instances in 6 different turns across two runs. The
   `build_synthesis_prompt` #1550 rule ("PROSE MUST MATCH YOUR OWN TABLE")
   already prohibits exactly this and demonstrably does not stop it — the fix
   needs a different mechanism (deterministic post-check, or the rule is
   missing from the code path these turns actually take — verify path first).

2. **Provenance family — 1.5 cleared, class reproduces in two new forms.**
   Morning 1.5 (no-tool turn upgrading synthetic → "real patient-journey data",
   invented "(DoWhy/EconML-style)" label) is gone: the rerun 1.5 calls the tool
   and grounds everything. But (a) **the inverse appeared**: 0 of 14 session-1/2
   turns say "synthetic" anywhere while every kpi/causal payload carries
   `data_source: "synthetic"` (morning: 5 of 14 disclosed it) — omission, not
   misstatement, but a demo viewer is never told; and (b) **3.1 fabricated a
   citation**: attributed a flow-cytometry/GPI-anchor claim + author surname to
   "the FDA label and clinical literature pulled just now" when the PubMed
   payload carries only pmid/title/journal/doi — textbook-correct parametric
   recall dressed as a sourced retrieval. The 1.5 provenance rule remains
   warranted, generalized: claims of what was "pulled/retrieved" must be
   payload-grounded; no provenance upgrades, no invented method/source labels.

3. **Window honesty — 4.6 cleared, class reproduces at 4.5, inverted.**
   Morning: 4.6 asserted a window the payload didn't carry. Rerun: 4.6 reads
   the metadata per-KPI correctly (fixed), but 4.5 — which morning got right —
   passed no window and then told the user "neither metric uses a
   calendar-month window", **a platform-capability claim disproven by
   morning's own payload on the same image** (both KPIs applied calendar-August
   with `window_status: "applied"`). The model attributed its own omission to
   the product. Rule warranted: never assert a window/capability is absent
   unless the payload says so.

4. **Session-memory negatives — reproduces in a new form (5.5, 4th straight
   clarify).** Morning 5.5 claimed no prior turn named a brand (5.3/5.4 both
   did); rerun 5.5 claims "we haven't run a segment ranking in this
   conversation" while 5.3 in the same session called the propensity tool and
   received the full ranking two turns earlier. Both are a turn asserting a
   negative about its own session history that its own tool history refutes.
   (Scenario doc line ~458 confirmed stale for 5.5.)

5. **#1693 region-drop — NOT exercised, NOT fixed.** Rerun 4.1 ran zero tools
   (asked which brand first, honoring the prompt contract morning broke) so the
   cohort_profiler region-drop was never reached — routing luck, not repair.
   The answer still *offers* "an aggregate cohort profile for oncologist HCPs
   in the Northeast", advertising exactly the scoping the profiler silently
   discards. The code bug stands as filed.

## New backend defect found by two graders independently

**`causal_analysis_tool` echoes its `region` parameter without applying it.**
Cross-run positive control: region="Northeast" (1.4), region="northeast" (2.4),
and region=null (1.5, morning 2.4) return byte-identical path_id sets with
identical effect sizes (e.g. the same six `scp_c…` ids). Because the payload
echoes the requested region back, the chat layer *cannot* catch it — 1.4, 2.4,
2.6, A.9-seed and A.10 all assert regional causal specificity that does not
exist, while morning's region=null runs correctly disclaimed it. Same
user-visible failure as #1693, different mechanism: cohort_profiler silently
DROPS an unparsable criterion; causal_analysis_tool silently ECHOES an
unapplied one. → filed as its own issue (see below).

## Regressions this run (LLM/orchestrator variance, same image)

- **2.1 misroute (largest regression):** orchestrator dispatched
  heterogeneous_optimizer + gap_analyzer, never cohort_profiler (gold routing:
  SINGLE_AGENT → cohort_profiler); returned a gap-analysis stub instead of
  morning's 2,341-patient profile. Handled honestly (quotes the failure,
  fabricates nothing) → PARTIAL not FAIL. Also cost 2.2/2.3 their independent
  cross-check and breached T3 latency (32.7 s).
- **1.2 transposed digit:** prose says uniform-expected share 30.61% where the
  payload says 32.61% — and the error flips the conclusion (reads "above pace"
  when the true comparison is below).
- **Experiment-design chain 3.4→3.5→3.6 compounds a unit mislabel:** the
  payload declares `cohens_d`; 3.4 relabels d=0.02 as "+2 percentage points on
  a 3.92% baseline"; 3.5 does arithmetic on the mislabel and emits "~5.92%" —
  the subset's only fully ungrounded number across both runs; 3.6 asserts
  "3,876 territories/clusters" (payload gives no unit; its own t-test basis
  implies individuals) and presents 546 days as "6mo + 3mo total" when 546 is
  the accrual function. Same family as #1691, expressed in units instead of
  superlatives.
- **4.3, 4.5** worse per above; **3.1** worse on the fabricated citation.

The unifying pattern (n3n4): this run is more *assertive*, and assertiveness
cuts both ways — every improvement names something morning left vague (failed
agent, per-KPI windows, missing trigger node, missing brand); every regression
names something the payload does not say (units, derived rates, feasibility,
capabilities).

## Latency

Σtotal 1,158.5 s vs 1,113.4 s (+4%). Appendix −10.5%. First-progress passes
everywhere (≤0.8 s vs 15 s budget). T3 breaches: 2.4 33.7 s (improved from
41.6), 2.1 32.7 s (misroute), 1.4 30.4 s (3 tools). 5.6 recovered its T2
budget (17.5 s vs 22.4). Demo-feel watch item: 6.2/6.4 TTFB 28–32 s (tool-call
counts grew 5→9 and 4→5), inside T4's 90 s.

## Standing items (unchanged, not regressions)

- A.8-warm still re-runs kpi_calculate_tool (no round-trip skip); warm faster
  than cold this run is LLM variance, not a cache.
- Harness: A.8-warm sends a verbatim repeat (not the doc's paraphrase);
  A.9-followup re-sends the seed question (doesn't probe anaphora as designed).
- window_coverage pacing signal unused on A.9-seed/A.10.
- Regression guards still green: #1636 (no classifier JSON), #1549 (no planner
  JSON/code fences), #1442 (no replayed prior-assistant text), #1638 roster
  22/22, SCALE fence #1640.

## Grader verdict on the morning run's open question

Re-run narratives "after #1690 is settled" — done. The champion and ChEMBL
confounds are fully cleared and accounted for. What remains PARTIAL is now
almost entirely the summary-layer honesty families above plus routing/planning
variance; retrieval grounding is otherwise excellent (n5n6: zero fabricated
values across 12 turns, several reconciled to the cent).
