# 30-Minute Strategic Demo — "Causal analysis that supports a decision"

**Purpose.** Answer the challenge *"why the E2I platform as designed, rather than
simply integrating causal ML (DoWhy / EconML) without the agentic harness?"* with
a single cohesive story, in 30 minutes, in front of a decision-making audience.

**Spine.** One brand — **Remibrutinib (BTK inhibitor, CSU launch)** — walked from
*what is happening* → *why* → *can I believe it* → *for whom* → *what should we
do and is it worth it* → *test before we spend* → *did it work, and what did the
system learn*. Every beat ends in a decision a brand lead actually makes.

**Evidence discipline.** Every "expected answer" below is transcribed from a
*recorded* run of the platform (`docs/demos/results/…`, cited per beat) or from
the code (`file:line`), never authored. The data behind those runs is
**synthetic** (`data_source: "synthetic"` rides on every KPI / causal payload) —
say so out loud in the first minute; it is a strength of the story (§6,
objection 2), not a weakness to hide. Numbers vary run to run (LLM variance,
data refresh, and the estimator's own seed); the *shape* of each answer is what
is stable and what this script asserts.

Companion material: the 51-turn question suite and routing rubric in
[`COPILOT_CHAT_DEMO_SCENARIOS_V2.md`](COPILOT_CHAT_DEMO_SCENARIOS_V2.md); the
platform explains its own method at `/documentation` ("How E2I Works").

---

## 1. The challenge, and the one-sentence answer

> "A data scientist can `pip install dowhy econml`, fit an ATE in an afternoon,
> and put the number in a slide. What does the 22-agent harness add?"

**Answer:** *Bare causal ML produces a number. The harness produces a
**decision-grade, governed, remembered** number — and it produces it for the
person who has to make the decision, not only for the person who can write the
estimator.*

Concretely, the harness is the ~80 % of the job that sits *around* the
estimator, which a notebook integration silently leaves to humans to do by
hand, inconsistently, or not at all:

| # | What the harness does automatically | What a bare DoWhy/EconML integration leaves to you | Where the demo shows it |
|---|---|---|---|
| 1 | **Translates a business question into an estimand** — KPI registry, domain vocabulary, brand / region / window resolution, typo tolerance, and an *ask-back* when the question is underspecified instead of a guess | A data scientist hand-picks treatment, outcome, confounders, window; the brand lead cannot ask at all | Beats 1–2 |
| 2 | **Runs the full validity stack on every estimate**, not only when someone remembers: guided DAG discovery with bootstrap corroboration and per-edge provenance, energy-score estimator selection, a six-check refutation suite (two critical tests that block, two that warn, an E-value sensitivity *reading*, a negative-control outcome check), and a PROCEED / REVIEW / BLOCK gate — and the gate itself is calibrated against planted truth | DoWhy *offers* refuters; nothing forces them to run, be recorded, be calibrated, or gate anything | Beats 2–3 |
| 3 | **Routes structures to a human domain expert**, records the sign-off with a 90-day validity, halts on a rejection, and keeps *structural* approval separate from *statistical* robustness (an approval cannot launder a fragile estimate) | No workflow; approval is a Slack thread | Beat 3 |
| 4 | **Fails closed and says so** — no fabricated allocation when the allocation substrate is missing, no forecast without a forecasting model, premise checks, window-coverage warnings, a deterministic post-check on superlatives that contradict the answer's own table | A notebook produces a number regardless of whether the question was well-posed | Beats 1, 5 |
| 5 | **Turns an effect into a decision object** — CATE by segment with a cross-library agreement gate and a bounded targeting policy, gap-to-ROI with Monte Carlo CI, attribution level and risk adjustment, constrained resource allocation, RCT design with power and feasibility warnings, a digital-twin pre-screen with a DEPLOY / REFINE / SKIP rule | Each is a separate hand-built analysis; none carries its assumptions with it | Beats 4–6 |
| 6 | **Closes the loop and remembers** — rep accept / override becomes ground-truth labels that drive concept-drift alerts and retraining; every refutation run emits a validation outcome to the feedback learner; findings live in episodic memory with staleness sentinels that queue re-analysis; every agent run writes a tamper-evident audit chain | Findings live in decks; nothing invalidates them when the data moves | Beats 6–7 |

The demo's job is to make each row *visible on screen*, once, inside one story.

---

## 2. Logistics (read before the day)

- **Surfaces.** The story alternates between the **copilot chat** (what a brand
  lead sees; AG-UI surface `POST /api/copilotkit/agent/default`) and five
  **workbench pages**: `/causal-analysis` (+ `/expert-reviews`),
  `/segment-analysis`, `/gap-analysis`, `/experiments` + `/digital-twin`, closing
  on `/audit-chain`. Chat gives the narrative; the pages show the machinery the
  chat is standing on.
- **Pre-warm (mandatory).** The Causal Analysis "Discover causal effects" job for
  Remibrutinib × `patient_journeys` (11 questions) took **1,590–1,887 s** end to
  end in the two recorded runs (`results/2026-09-09_expert_review_loop/`). Run
  it *before* the session on the **current image** and demo the results; a
  single-pair re-run after an expert decision takes ~2–3 min and can be shown
  live. Do **not** reuse the 2026-09-09 band table as "expected" — the
  sensitivity gate was recalibrated on 2026-09-10 (§4 Beat 3).
- **Latency you will see in chat** (measured p90 budgets, `V2.md` "Latency
  budgets"): simple KPI turns < 18 s, single-agent analytical turns < 25 s,
  composite / orchestrated turns < 90 s (experiment design is the long pole at
  ~60–80 s). First visible progress arrives in < 1 s. Narrate while it streams;
  never wait in silence.
- **Champion models must be promoted.** The propensity beat (Beat 4) serves
  `hcp_adoption_remibrutinib_goldstd_lr_v1` (holdout AUC 0.7945); if the weekly
  retrain demoted the champion the tool fails closed with *"no production
  champion registered"*. Check the registry the morning of (the 2026-08-18
  morning run lost three turns to exactly this; the restore script is #1384).
- **Region filters.** Prefer brand-level phrasing for *causal* questions and let
  the KPI tool carry the region (it returns `region_status: applied`). A defect
  where `causal_analysis_tool` echoed an unapplied `region` was filed from the
  2026-08-18 run; verify against the payload's `region_status` before promising
  regional causal specificity on stage.
- **Two things chat will honestly decline** — and the script uses both as
  beats, not surprises: a quantified rep-capacity re-allocation (needs
  per-entity response coefficients + a budget; use `/resource-optimization`
  with inputs), and a digital-twin lift figure (the twin lives at
  `/digital-twin` and inside the experiment designer, not in the chat tool set).
- **Have the one-slide close (§7) ready** as the last screen.

---

## 3. Run of show (30:00)

| Clock | Beat | Surface | Decision it supports |
|---|---|---|---|
| 0:00–3:00 | 0. Frame the challenge | slide + `/documentation` | — |
| 3:00–6:00 | 1. Pulse — is there actually a problem? | chat | Do I react to this quarter's NRx at all? |
| 6:00–10:00 | 2. Why — the causal drivers, and can I trust them | chat | Which lever do I pull? |
| 10:00–16:00 | 3. Inside the gate — refutation, provenance, expert review | `/causal-analysis` + `/expert-reviews` | Which effects are allowed to reach a plan? |
| 16:00–20:00 | 4. For whom — heterogeneity and targeting | `/segment-analysis` + chat | Where do I concentrate effort? |
| 20:00–24:00 | 5. Is it worth it — gap, ROI, allocation | `/gap-analysis` + chat (+ `/resource-optimization`) | Do I fund it, and at what confidence? |
| 24:00–27:30 | 6. Test before you spend, then close the loop | `/experiments` + `/digital-twin` + chat | Do I run the experiment, and did last quarter's triggers work? |
| 27:30–30:00 | 7. What the system remembers and governs; the one-slide close | `/audit-chain` + slide | Can I defend this in an audit? |

---

## 4. Beat-by-beat script

Each beat lists: the exact question or click, the **expected answer** (recorded,
with source), **what it proves**, the **bare-ML contrast** to say out loud, and a
**fallback** if the turn misfires. Persona throughout: Remibrutinib brand lead.

### Beat 0 — Frame (0:00–3:00)

Say: *"I'm going to ask the platform the questions I'd ask my analytics team on a
Monday, in order, about one launch brand. Watch for three things: it translates
my question into a causal estimand, it refuses to give me a number it cannot
defend, and every number it does give me carries its own evidence. Everything
you will see runs on synthetic data — deliberately, so we can show you what the
system does when an estimate is *wrong*."*

Open `/documentation` for 20 seconds: the six-section "How E2I Works" page
(purpose → causal impact → quality gate → methodology → practices → impact).
Point at the **Quality Gate** section: the six refutation checks and the three
bands. This is the contract the rest of the demo is held to.

### Beat 1 — Pulse: is there actually a problem? (3:00–6:00)

**Q1.1** `What is driving the drop in Remibrutinib NRx in the Northeast this quarter?`

Deliberately loaded — the question presupposes a drop.

**Expected answer** (recorded 2026-08-18, `grades_n1n2.json` 2.4, PASS; also
`2026-07-29 transcripts.md` 2.4):

- Leads with **NRx = 448, Northeast, quarter-to-date**, window 2026-07-01 →
  2026-10-01 with **48 days elapsed**; trailing-30-day = 172 = **38.4 % of the
  quarter-to-date total vs 62.5 % uniform expectation**.
- **Refuses to size "the drop"** without a same-length prior period, and offers
  to pull it.
- Flags that a trigger filter did not apply and *declines to use those rows*.
- Then gives the brand-level causal drivers of NRx (three validated paths —
  used again in Beat 2).

**What it proves.** Rows 1 and 4: KPI resolution with an explicit window,
coverage honesty, and a premise check before attribution.

**Bare-ML contrast.** *"A notebook would have been handed 'the Q3 drop' as a
fact and regressed on it. The platform first asked whether there is a drop."*

**Fallback.** If the answer sizes a drop anyway, ask `Compare that to the same
48 days of last quarter` — the KPI tool applies non-overlapping windows and the
coverage warning surfaces.

Optional 20-second aside on translation (row 1): `Show me converson rate for
Remibrutnib` (typos) → same answer as the clean phrasing (A.4, 5.1 s recorded).
And the underspecified case: a cold `Why did it drop?` gets an ask-back for
brand and metric instead of a guess (A.5, measured 2026-08-03).

### Beat 2 — Why, and can I trust it (6:00–10:00)

**Q2.1** `What are the causal drivers of Remibrutinib NRx?`

**Expected answer** (recorded 2026-08-18, 2.4 payload; identical values in the
2026-07-29 transcript):

| Cause → NRx (via) | Effect | Confidence | Lag | Est. impact |
|---|---|---|---|---|
| treatment_initiated (patient onboarding) | **+0.41** | 0.816 | 73 d | $72,247 |
| intent_to_prescribe (new patient starts) | +0.27 | 0.816 | 68 d | $111,256 |
| sample_dropped (trial experience) | +0.092 | 0.841 | 87 d | $30,580 |

All three carry `validation_status: validated`; the answer states they are
brand-level, not regional, and that lags of 2–3 months mean a dip *this*
quarter traces to engagement *last* quarter.

**Q2.2** `How confident are we in the rep-detailing effect on Remibrutinib TRx — did it pass refutation tests?`

**Expected answer** (recorded 2026-08-18, 4.7 and 5.7 grounded values):

- `rep_detailing_frequency → trx_volume` via `hcp_engagement`: **+0.298**,
  confidence **0.897**, lag **86 days**, business impact **+$91,289**.
- **"5/5 refutation tests passed, 0 failed, 0 warning; gate decision:
  proceed"** — read straight from `refutation_evidence` in the payload.
- Method named (`backdoor.linear_regression`), `evidence_is_synthetic: true`,
  and the date it was last tested.
- Ranks it explicitly among the brand's six paths (the 08-18 run's
  "4th-strongest of 6, ranked by confidence" was verified exact against the
  payload).

**What it proves.** Rows 1–2: the chat answer is not a narration of a model; it
is a narration of a *validated* model, and the validation evidence travels with
the number.

**Bare-ML contrast.** *"In a notebook, 'did it pass refutation' is a question to
the analyst's memory. Here it is a field on the object."*

**Fallback.** If the model answers with confidence scores but says refutation
fields are absent (the 07-29 behaviour), go straight to Beat 3 — the workbench
shows the same evidence per test.

### Beat 3 — Inside the gate: refutation, provenance, expert review (10:00–16:00)

This is the beat that answers the challenge. Open `/causal-analysis`, brand
Remibrutinib, dataset `patient_journeys`, pre-run results loaded (§2).

**Click 3.1 — the ranked table.** 11 questions from the brand's question
registry, each validated (guided DAG discovery → energy-score estimator
selection → refutation gate) and ranked by confidence then impact. The page's
own copy: *"The agent validates each one and ranks the effects by confidence
(robustness gate + significance) and impact (effect size)."*

**Expected screen on the current image.** ATEs and CIs are the production
configuration's values from the live-frame sweep
(`results/2026-09-12_estimator_calibration_2031/summary_live.md`, leaf 5,
seed 42, n = 5,000); readings from the 2026-09-11 re-band
(`results/2026-09-10_sensitivity_calibration/reband.md`):

| Question (Remibrutinib, patient_journeys) | ATE [95 % CI] | Sensitivity reading | Expected band |
|---|---|---|---|
| treatment_arm → treatment_initiated | **0.193 [0.158, 0.228]** | Robust to confounding at measured strength (RR 1.61 vs benchmark 1.24) | PROCEED |
| urticaria_severity_uas7 → persistent_180d | 0.151 [0.124, 0.177] | Robust (1.32 vs 1.00) | PROCEED |
| copay_support → low_gap_180d | 0.104 [0.076, 0.131] | Robust (1.39 vs 1.04) | PROCEED |
| copay_support → adherent_180d | 0.099 [0.070, 0.127] | Robust (1.32 vs 1.05) | PROCEED |
| trigger_accepted → treatment_initiated | 0.096 [0.070, 0.122] | Robust (1.37 vs 1.19) | PROCEED |
| psp_enrolled → adherent_180d | 0.086 [0.057, 0.114] | Robust (1.28 vs 1.04) | PROCEED |
| psp_enrolled → persistent_180d | 0.083 [0.054, 0.112] | Robust (1.13 vs 1.03) | PROCEED |
| rep_detailing_high → treatment_initiated | 0.072 [0.046, 0.099] | Robust (1.24 vs 1.05) | PROCEED |
| copay_support → persistent_180d | 0.066 [0.037, 0.095] | Robust (1.14 vs 1.12) — the thinnest margin | PROCEED |
| sample_dropped → treatment_initiated | 0.042 [0.015, 0.070] | Robust (1.13 vs 1.03) | PROCEED |
| treatment_arm → persistent_180d | **0.034 [−0.005, 0.072]** | **"No detectable effect at this sample size"** (CI includes zero) | REVIEW / BLOCK — see below |

Say: *"Every one of these eleven is a positive, plausible-looking number a
notebook would print. Each carries an E-value reading against the confounding we
could actually measure; the last one is a null finding the platform refuses to
headline. Two weeks ago six of these rows were BLOCKED. They are not blocked
today — and the reason is the most important thing on this screen."*

Then tell the calibration story (30 s, `refutation_runner.py:1136-1144`,
`reband.md`, `disproof.md`):

- On 2026-09-10 the team measured the sensitivity gate against **planted
  truth** and found the old E-value cutoffs had **blocked 7 of 11 correctly
  recovered effects** and could not tell an omitted-confounder fit from a
  correct one. The E-value became a *reading* benchmarked to the confounding
  the run measured; **56 stored estimates moved block → proceed, 6 stayed
  blocked, 59 stayed proceed**, and the reading was checked to be stable
  across frames (0 flips in 103 comparisons).
- On 2026-09-12 the one remaining null finding on this brand
  (`treatment_arm → persistent_180d`) was traced to a **seed artefact of the
  nuisance model**: seed 42 gives 0.034, the six-seed mean is 0.066; at a
  larger leaf it reads 0.065 [0.027, 0.104]. The same sweep found a truth
  defect in the synthetic generator itself (an unweighted segment mean
  overstating three planted ATEs by ≈ 0.02).

*"A notebook's gate is whatever the analyst set on the day. This gate has a
benchmark, a changelog, and a measured false-block rate."*

**Click 3.2 — drill into one row** (`treatment_arm → treatment_initiated`).
The deep view shows: the DAG with **per-edge provenance** (`required_prior` /
`discovered` / `curated`), the **`dag_source`** label (`discovered` /
`prior_asserted` / `augmented` / `domain_knowledge`), the selected estimator,
and the refutation checks with three-state results. Name them as the
`/documentation` page states them (`content.ts`):

| Check | Pass rule | Role |
|---|---|---|
| Placebo treatment | placebo p-value > 0.05 | **critical — a failure blocks** |
| Random common cause | effect moves ≤ 1 SE (1–2 SE warns, more fails) | **critical — a failure blocks** |
| Data subset | ≥ 80 % of subset effects inside the original CI | warns / scores |
| Bootstrap | bootstrap CI ≤ 1.5× the original width | warns / scores |
| Sensitivity (E-value) | point risk ratio above the measured-confounding benchmark; a CI including zero is a null finding; no benchmark → "not benchmarked" | **a reading, never a block** (since 2026-09-10) |
| Negative-control outcome | the control's CI includes 0; excludes 0 but smaller than the claimed effect warns; at least as large fails | weight 0 for the first live period — *the one refuter that can detect confounding* (#2007) |

Bands (`refutation_runner.py:1195-1199`): **Proceed** = confidence ≥ 0.70 and no
critical failure; **Review** = 0.50–0.70, surfaced with a caveat and queued for
expert review; **Block** = any critical failure or confidence < 0.50 — *"marked
refuted and never reaches a decision."* Confidence is a weighted score
(placebo 0.25, random common cause 0.25, sensitivity 0.25, subset 0.125,
bootstrap 0.125); skipped tests leave the denominator, and all-skipped fails
closed to 0.0.

Then point at `dag_source`. Say: *"Before ADR-017 this label said 'discovered'
for a graph the priors had fully determined — the same DAG came back on real
data and on pure noise. Now the platform reports what the data contributed at
three grains — the label, the discovered confounders, and per-edge provenance —
and they cannot disagree. That is a correction a notebook never makes, because
nobody is auditing the notebook."*

**Click 3.3 — `/expert-reviews`, the linked review card.** Open a pending
review for a Remibrutinib structure (the queue de-duplicates by DAG hash: the
2026-09-09 job re-used six pre-existing pending rows and minted none). Show the
agent's advisory assessment (checklist: confounders included / no forbidden
edges / mediators positioned). **Approve** it live as a domain expert.

**Expected result** (recorded 2026-09-09 `adjudications.md`, Step 2, live UI):

- Card flips to `approved` · reviewer · decided date · **valid until +90 days**
  (renewal warning at 14 days, `expert_review_gate.py:222-224`).
- Re-run the pair (~2 min). The record carries `decision: proceed` and
  `review_id: <the approval>` — and the **statistical band is computed
  independently**: in the recording the estimate still failed its (then
  critical) sensitivity test, so the band stayed BLOCK while the structure was
  approved.

Say the sentence the platform itself builds (`nodes/refutation.py`,
`_review_note`): *"That approval covers the DAG structure, not this estimate's
statistical robustness."*

**Then Reject** a second pending row (Step 3 in the recording): the card flips
to `rejected` with no expiry, pending rows for that hash go to **0 and stay 0**
after the re-run, and the re-run comes back `decision: rejected` with the
review id — a human rejection **always halts** the run, on every band
(`expert_review_gate.py:213-224`; `nodes/refutation.py:1546`).

**What it proves.** Rows 2, 3, 4 in one screen: a calibrated validity stack,
human-in-the-loop with expiry, and the separation that stops a sign-off from
laundering a fragile number. Also row 6: every run wrote 5 rows to
`causal_validations` and one to `discovered_dags` (+55 / +11 measured).

**Bare-ML contrast.** *"DoWhy has `refute_estimate()`. It does not have a gate, a
calibration record, a review queue, an expiry, provenance per edge, or a
database that remembers this structure was rejected on 3 September. We built
the part that makes the refuter matter."*

**Fallback.** If the live approve → re-run cannot be shown in time, show the
recorded screenshots (`linked_card_pending_4eab7033.png`,
`linked_card_approved_4eab7033.png`, `linked_card_rejected_2f79f11f.png`).

### Beat 4 — For whom: heterogeneity and targeting (16:00–20:00)

**Click 4.1 — `/segment-analysis`**, brand cohort Remibrutinib, curated pair
`treatment_arm → treatment_initiated` (the top PROCEED row from Beat 3). The
backend fixes the clinical contract server-side (segment variables, effect
modifiers, confounders, substrate) and loads the gold-standard
`patient_journeys` frame; the page renders CATE across **every** clinical
segment dimension ordered by spread, high / mid / low responder cards, uplift
metrics (AUUC / Qini; honest empty state when a run lacks them), a **targeting
policy** (who / why / expected lift), and between-segment heterogeneity (I²).

What the agent did to get there (`heterogeneous_optimizer/`): CausalForest
CATE → segment analysis → hierarchical nested CIs → uplift → policy learning —
with a **cross-library agreement gate** (EconML vs CausalML sign and ordering
agreement must reach 0.7, `cross_validation.py:59-69`), a latent-confounder
warning when > 30 % of segments carry one, and a **bounded policy**: the
recommended treatment rate can rise at most +20 pp and never above 90 %,
policy confidence grows with n and caps at 0.95 (`policy_learner.py:314-336`).

Say: *"The ATE from Beat 3 was 0.19. This is the same effect split by who the
patient is, with the second library checking the first. The decision is not
'does the treatment work' — it is 'for whom is it worth the effort'."*

**Q4.2 (chat)** `Where are the biggest untapped opportunities to grow Remibrutinib market share?`

**Expected answer** (recorded 2026-08-18, 2.5, PASS):

- Current position: **TRx share 33.4 %** of the tracked portfolio (KPI
  `WS3-BI-008`; the answer states it is portfolio share, *not* share vs Xolair
  / Dupixent — lifted from the payload's semantic note).
- Regional gaps, e.g. Northeast HCP engagement **10.0 vs 14.65 target
  (68.3 %)** — the weakest attainment metric — mapped onto the strongest causal
  lever (detailing → engagement → volume, +0.298, 89.7 %).
- A propensity ranking from the **named champion model**
  `hcp_adoption_remibrutinib_goldstd_lr_v1`, holdout **AUC 0.7945**, n = 5,000
  scored: Dermatology **55.0 % (n=854)**, Allergy/Immunology 51.6 % (n=577),
  Rheumatology 48.4 % (n=257) … Oncology 31.6 %; by region West 41.0 %.
- Bottom line: Northeast is simultaneously the weakest-engagement region *and*
  sits on the highest-confidence causal path — the priority for incremental
  detailing.

**What it proves.** Row 5 (effect → decision object) and row 1 (the propensity
model is named, with its AUC and n — provenance, not a black box).

**Bare-ML contrast.** *"EconML will give you a CATE. It will not check it
against a second library, bound the policy, or join it to the KPI gap, the
propensity model and the causal lag to tell you which region and which
specialty. That join is the harness."*

**Fallback.** If `heterogeneous_optimizer` reports `missing_required_inputs`
in chat (the 4.4 behaviour), the answer says so and refuses to fabricate a
segment ranking — show that honesty for ten seconds, then rely on the
`/segment-analysis` page, which fixes the inputs server-side.

### Beat 5 — Is it worth it: gap, ROI, allocation (20:00–24:00)

**Click 5.1 — `/gap-analysis`**, Remibrutinib. The Tier-2 gap analyzer runs
gap detection → ROI → instrument analysis → prioritisation → narrative
(`gap_analyzer/graph.py:52-58`). Point at the uncertainty clause the narrative
renders (`docs/roi_methodology.md` §11):

> "…at 4.0× ROI (risk-adjusted 2.4×, 95 % CI 1.1×–4.0×; P(ROI > 1×) = 78 %)"

and explain the moving parts in one breath: gaps below **5 %** are not
reported; only **30 %** of a gap is assumed closable (`DEFAULT_CAPTURE_RATE`,
`roi_calculator.py:44-78`); the **95 % CI is a 1,000-draw Monte Carlo** (normal
value drivers, gamma costs, beta acceptance rates); the **attribution level**
(FULL 1.00 / PARTIAL 0.65 / SHARED 0.35 / MINIMAL 0.10) is the coarsest step and
is chosen deliberately; the **risk adjustment** compounds multiplicatively
(all-HIGH ≈ 84 %, never > 100 %). The prioritiser demotes — never deletes —
opportunities whose driver has no causal evidence (× 0.7,
`prioritizer.py:32-40`) and buckets the rest into quick wins / steady plays /
strategic bets. The unit economics ($850 / incremental TRx, $1,200 / identified
patient, …) are documented assumptions, brand-overridable, not hidden constants.

**Q5.2 (chat)** `Which regions are underperforming on Remibrutinib conversion rate, and for those regions, what would be the ROI of shifting 20% more rep capacity there?`

**Expected answer** (recorded 2026-08-18, 6.2, PASS):

- Conversion by region, trailing 30 days through the data frontier: West
  **64.12 %**, Northeast 63.59 %, Midwest 63.45 %, South **62.43 %**; 4-region
  mean 63.40 %; each with its 12-month temporal-variability band and per-region
  ROI (Northeast 2.142 … West 1.626), every figure exact to the payload.
- **Refuses to compute the 20 %-capacity ROI**: no capacity-elasticity model is
  present, "I don't want to fabricate one"; recommends a causal pass on the
  underperformers first.

Say: *"This is the answer I want from an analyst: the numbers it has, the
number it doesn't, and what it would take to get it. Now let's give it what it
takes."*

**Click 5.3 — `/resource-optimization`** (optional, 60 s): with entities,
response coefficients and a budget constraint supplied, the Tier-4 optimizer
(scipy backend) solves the constrained allocation with before / after
comparison, scenario comparison and sensitivity. Chat declined precisely
because these inputs were absent (recorded 2.6: `missing_required_inputs`,
"Failing closed — no values were fabricated").

**What it proves.** Rows 4 and 5: decision-grade uncertainty, explicit
assumptions, and fail-closed rather than plausible-but-fake.

**Bare-ML contrast.** *"An ATE has no cost side. ROI with a CI, a capture
rate, an attribution choice and a risk haircut is what a CFO signs. None of it
comes with the estimator."*

### Beat 6 — Test before you spend, then close the loop (24:00–27:30)

**Q6.1 (chat, start it streaming, ~60–80 s)** `Design an experiment to measure whether speaker programs increase Remibrutinib NRx`

While it streams, **click `/experiments` and `/digital-twin`**: experiment
health, enrollment, SRM checks, interim analyses, **digital-twin fidelity
tracking**; the twin page runs a simulation, browses history, shows fidelity,
and renders *only* what the backend returns (no static stat cards — the page's
own honesty contract).

What the designer does (`experiment_designer/graph.py:232-317`): context →
**twin simulation** → design reasoning → power analysis → validity audit →
(bounded redesign loop) → template. The twin pre-screen simulates 10,000 twins
and applies a **CI-based three-way rule** (`digital_twin/effect/recommendation.py:40-56`):
CI lower bound above the 0.05 minimum effect → **DEPLOY**; CI upper bound
below it → **SKIP**; straddling → **REFINE**. It returns `simulated_ate`, the CI,
`recommended_sample_size`, and a confidence that is 40 % twin fidelity; a twin
with fidelity < 0.7 is flagged unreliable, and a failed simulation returns
REFINE with zeros — *"a fabricated ATE is never emitted"*. A SKIP ends the run
early: no design is produced for an experiment the twin says cannot win.

**Expected answer** (recorded 2026-08-18, 3.4, Fabhalta variant; expect the
same shape for Remibrutinib): the agent completes (~64 s) with a full protocol
— **RCT, individual randomization, stratified block** on baseline-Rx decile /
specialty / region / prior engagement tier, blocked by territory, **required
n = 78,490 (39,245 per arm)** at 80 % power, α = 0.05 for the effect it chose —
and a **feasibility warning**: at 50 HCPs / week the accrual is ~30 years; the
effect is too small for the addressable pool. The answer leads with *"not
executable as specified"* and offers three remedies (relax the MDE,
cluster-randomize, change the outcome). The template also carries a
pre-registration document, an analysis-code template and monitoring
checkpoints.

Say: *"That is the experiment we would have funded. The platform sized it, then
told us not to run it as specified. The twin's job is to pre-screen the
alternatives before any HCP is enrolled."* (Watch the units: the payload
declares Cohen's d; the recorded run mislabelled it as percentage points once —
if the answer quotes a "+2 pp lift", correct it to *d = 0.02* on stage.)

**Q6.2 (chat)** `Did rep actions driven by triggers actually lift prescriptions for Remibrutinib?`

**Expected answer** (recorded 2026-08-18, 4.7, PASS; 4.5 / 4.6 for the rates):

- Trigger funnel from the KPI engine (`WS2-TR-009`): **Delivered 6,179 → Viewed
  5,633 → Accepted …**; trigger precision 64.2 %, acceptance 55.1 %,
  false-alert 11.2 % (target 0.10), override 12.4 % if asked.
- Detailing → TRx (+0.298, 5/5 refutation passed) — and the answer **refuses
  to claim** that general detailing evidence answers the *trigger-driven*
  question: "the causal registry doesn't isolate trigger-prompted rep actions
  as a separate cause".
- Phrased as `trigger action prescription lift`, the registry returns the
  direct path **trigger_accepted → treatment_initiated: +0.059, confidence
  0.864, $21,480** — the same pair that reads 0.096 [0.070, 0.122] on the
  patient-journey frame in Beat 3.

**What it proves — the loop, as wired** (code-verified):

1. A rep's **accept / override / false-alert** on a trigger is a KPI
   (`WS2-TR-004/005/006`, `trigger_performance.py:77-85`) *and* a ground-truth
   label: an accepted trigger with downstream treatment activity labels the
   prediction POSITIVE at truth-confidence 0.90 (`006_feedback_loop_infrastructure.sql:505-545`).
2. Those labels drive **concept-drift alerts** — accuracy drop > 0.05,
   calibration error > baseline + 0.10, class-rate shift > 0.15 — and
   retraining (`006:837-871`); `/feedback-learning` shows the Tier-5 cycle.
3. `acceptance_status` becomes a real binary treatment for the conversion-rate
   frame so the heterogeneity agent can estimate *who* responds to an accepted
   trigger (`kpi_resolution.py:636-660`).
4. Every refutation run emits a **validation outcome** (passed / failed /
   needs_review / blocked, with a failure category) to the feedback learner
   (`nodes/refutation.py:40-53`), which proposes baseline / config / prompt /
   threshold updates — **propose-only by default**, applied only with a human
   in the loop (`knowledge_updater.py:54-88`), and scored by a five-criterion
   rubric (causal validity 0.25, actionability 0.25, evidence chain 0.20,
   regulatory awareness 0.15, uncertainty communication 0.15).

**Bare-ML contrast.** *"The notebook's output is a slide. Here the output is a
trigger, the trigger has a funnel, the funnel has a causal effect, the
acceptance is a label, and the label retrains the model. That loop is the
product."*

### Beat 7 — What the system remembers and governs; close (27:30–30:00)

**Click 7.1 — `/audit-chain`.** Every agent graph starts with an `audit_init`
genesis block and wraps each node in a timed, tamper-evident entry
(`causal_impact/graph.py:490-503`). Pick the Beat-3 run: graph builder →
adjustment-set policy → estimation → refutation → sensitivity →
interpretation, each an entry.

**Say (30 s) on memory.** The causal agent writes each analysis to episodic
memory and each validated path to the semantic knowledge graph
(`causal_impact/memory_hooks.py:415, 524`); the orchestrator reads the last
five relevant episodes and a RAG pass over them before routing
(`orchestrator/memory_hooks.py:146`). Findings are de-duplicated, promoted, and
crystallised into `executive_insights`. When an upstream finding is overturned,
the invalidator marks dependents stale, a **sentinel** (Celery, every 5 min,
cooled-down) fires a `staleness_alert` and **queues re-analysis** of the top-5
stale findings (`ARCHITECTURE.md` §5.5). *"The deck from last quarter does not
know it is wrong. This does."*

**Say (30 s) on routing honesty.** The orchestrator's 4-stage classifier
(features → domain → dependencies → pattern) abstains to a clarification when
its top domain is below 0.5 confidence; synthesis confidence is the *mean* of
the contributing agents', never inflated; a missing agent fails closed rather
than fabricating a result (`orchestrator/graph.py:99-106`).

**Say (30 s) on measurement.** The platform is graded against itself: 51
scripted turns, four independent graders tracing **every numeric claim to a
tool payload**; the last recorded run is **37 PASS / 14 PARTIAL / 0 FAIL**
(`results/2026-08-18_post1690_copilot_chat_perf/SUMMARY.md`). The PARTIALs are
mostly units and prose-vs-table slips — named, filed, and being fixed with
deterministic post-checks, not prompt tweaks.

**Close on the one slide (§7).**

---

## 5. Value statements per beat (for the speaker notes)

| Beat | Strategic decision | Value the harness adds over bare causal ML | Evidence on screen |
|---|---|---|---|
| 1 | React to a KPI move or not | Premise and window honesty before attribution; a business user can ask directly | 448 QTD / 48 days / "no prior window → won't size the drop" |
| 2 | Which lever to pull | Validated drivers with lag and $ impact; refutation evidence travels with the number | +0.298 / 0.897 / 86 d / 5-of-5 / proceed |
| 3 | Which effects may enter a plan | Always-on gate, *calibrated against planted truth*; provenance per edge; expert sign-off with expiry; rejection halts; approval cannot launder robustness | 11 ranked rows with E-value readings; 56 block→proceed after recalibration; approve → decision proceed, band computed independently |
| 4 | Where to concentrate effort | CATE with a cross-library gate, a bounded policy, joined to KPI gaps and a named propensity model | AUC 0.7945, Dermatology 55.0 % n=854, Northeast 68.3 % engagement |
| 5 | Fund it or not | ROI with Monte Carlo CI, capture rate, explicit attribution and risk; refusal to fabricate elasticity | "4.0× (risk-adj 2.4×, 95 % CI 1.1–4.0×, P>1× 78 %)"; 6.2 refusal |
| 6 | Run the test; did the last one work | Twin DEPLOY / REFINE / SKIP; power + feasibility warning; trigger funnel → causal effect → ground-truth labels → drift → retrain | n = 78,490 / "not executable as specified"; trigger_accepted → initiation +0.059 |
| 7 | Defend it in audit; keep it true | Tamper-evident chain; memory that invalidates itself; fail-closed routing; graded evaluation | audit chain; 37/14/0 |

---

## 6. Objections you will get, and the honest answer

1. **"It's slow."** 7–35 s for analytical turns, 60–80 s for a full experiment
   design, ~30 min for an 11-question discovery job (bootstrap corroboration
   is ~20× a single PC fit — the price of any corroboration signal on a
   single-algorithm run, ADR-017). Compare it to the human turnaround for the
   same validated output (days), not to a dashboard filter. Progress streams
   in < 1 s.
2. **"It's synthetic data."** Yes — and the synthetic generator plants known
   causal structure, which is the only way to *prove* the gate separates
   signal from noise: that is how the sensitivity gate's 7-of-11 false-block
   rate was found and fixed, how a null finding was traced to a seed artefact,
   and how a defect in the generator's own truth table was caught. The harness
   is data-source agnostic; the RWD path (`docs/RWD_PIPELINE.md`, Optum
   conversion) is the same pipeline.
3. **"The LLM makes things up."** It has, in measurable ways — units,
   superlatives contradicting the answer's own table, an invented citation
   once. The platform's answer is *not* "trust the model": grounded tools, a
   synthesis honesty guard that appends deterministic corrections, fail-closed
   tools, and a graded eval suite that catches the families. The 0-FAIL figure
   is a measured floor; the PARTIALs are the named backlog.
4. **"We can do this with DoWhy in a notebook."** You can compute the ATE. You
   cannot get, for free, the estimand translation, the always-on and
   calibrated gate, the review workflow, the provenance, the ROI object, the
   twin, the trigger loop, the memory, or the audit chain — and each of those
   is where a causal number either becomes a decision or quietly stops being
   true.
5. **"Chat couldn't give me the reallocation / the twin lift."** Correct, by
   design: it declined rather than invent. Both exist on the workbench pages
   with their inputs; the chat surface fails closed when the inputs are absent.
6. **"Your gate changed two weeks ago — how do I trust it?"** Because it changed
   for a measured reason, with the before / after recorded per estimate
   (`reband.md`: 56 moved, 6 stayed blocked, 59 unchanged, 0 frame flips), and
   the negative-control check that *can* detect confounding was added at weight
   0 so it is observed before it is trusted. That is what a governed gate looks
   like.

---

## 7. The one-slide close

> **Bare causal ML answers "what is the effect?"**
> **The E2I harness answers "what should we do, how sure are we, who signed off, and is it still true?"**
>
> - Every estimate: discovered DAG with provenance → energy-scored estimator → six refutation checks → PROCEED / REVIEW / BLOCK, calibrated against planted truth
> - Every structure: a domain expert, with an expiry, whose rejection halts and whose approval cannot launder robustness
> - Every effect: for whom (CATE, two libraries agreeing), worth what (ROI + CI + capture + attribution + risk), tested how (twin DEPLOY / REFINE / SKIP, then RCT + power + feasibility)
> - Every recommendation: a trigger, a funnel, a ground-truth label, a drift alert, a retrain
> - Every run: a tamper-evident audit chain, a memory that invalidates itself, a graded evaluation

---

## Appendix A — Sources for every recorded number

| Beat / item | Source |
|---|---|
| 1.1, 2.1 | `results/2026-08-18_post1690_copilot_chat_perf/grades_n1n2.json` (2.4); `results/2026-07-29_copilot_chat_perf/transcripts.md` (2.4) |
| A.4 / A.5 asides | `COPILOT_CHAT_DEMO_SCENARIOS_V2.md` Appendix A; `results/2026-08-03_clarify_probe_1407/` |
| 2.2 | `grades_n3n4.json` (4.7), `grades_n5n6.json` (5.7) |
| 3.1 ATEs / CIs | `results/2026-09-12_estimator_calibration_2031/summary_live.md` (leaf 5, seed 42 column), `disproof.md` |
| 3.1 readings, gate moves | `results/2026-09-10_sensitivity_calibration/reband.md` |
| 3.1 job timing, +55 / +11 rows | `results/2026-09-09_expert_review_loop/baseline.md`, `impact.md` |
| 3.2 checks, bands, weights | `frontend/src/components/documentation/content.ts:755-872`; `src/causal_engine/refutation_runner.py:1113-1199, 2990-3072`; `docs/decisions/adr-017-…md` |
| 3.3 approve / reject loop | `results/2026-09-09_expert_review_loop/adjudications.md` + PNGs; `src/causal_engine/expert_review_gate.py:95-251` |
| 4.1 agent internals | `src/agents/heterogeneous_optimizer/cross_validation.py:59-69`, `nodes/policy_learner.py:314-336`, `nodes/segment_analyzer.py:16` |
| 4.2 | `grades_n1n2.json` (2.5) |
| 5.1 ROI clause, constants, prioritiser | `docs/roi_methodology.md` §3–§6, §11; `src/agents/gap_analyzer/nodes/roi_calculator.py:44-78`; `nodes/prioritizer.py:32-40` |
| 5.2 | `grades_n5n6.json` (6.2); 2.6 for the fail-closed allocation |
| 6.1 | `grades_n3n4.json` (3.4, 3.5); `src/agents/experiment_designer/graph.py:232-317`; `src/digital_twin/effect/recommendation.py:22-56`; `src/digital_twin/simulation_engine.py:260-264, 456-476` |
| 6.2 | `grades_n3n4.json` (4.5, 4.6, 4.7); `src/kpi/calculators/trigger_performance.py:77-85`; `database/migrations/006_feedback_loop_infrastructure.sql:505-545, 837-871`; `src/services/kpi_resolution.py:636-660`; `src/agents/feedback_learner/nodes/knowledge_updater.py:54-88`; `evaluation/criteria.py:27-147` |
| 7 memory, routing, eval | `src/agents/causal_impact/memory_hooks.py:415, 524`; `src/agents/orchestrator/memory_hooks.py:146`, `graph.py:99-106`, `classifier/pattern_selector.py:47, 86`; `docs/ARCHITECTURE.md` §5.5; `results/2026-08-18_post1690_copilot_chat_perf/SUMMARY.md` |
| Latency budgets, surfaces | `docs/demos/COPILOT_CHAT_DEMO_SCENARIOS_V2.md` |
