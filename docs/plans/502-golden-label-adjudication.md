# #502 — Golden-label adjudication of unanimous frontier-model disagreement

**Refs #358 #502.** Status: ANALYSIS COMPLETE. Verdict: **11 LABEL-CORRECT, 0 LABEL-WRONG, 2
LABEL-AMBIGUOUS** (the two treatment-switch features). Recommendation: **0 label corrections** —
no golden label is *wrong*; the 11 correct labels are clearly preferred over the models' reading,
and the 2 ambiguous labels (`descendant` for the treatment-switch features) are defensible but not
strictly preferable over the models' `collider` reading, so they are flagged for human sign-off
rather than corrected. The 30–40 % cross-vendor correlated-failure rate observed in #242 is **real,
task-structural model failure, not label noise.** Deliverable is this doc; no JSON mutated; #502
stays OPEN pending human sign-off on (a) the 2 ambiguous switch-feature labels and (b) the
compile-set hardening recommendation. (The "real model failure" framing is exact for the 11 clear
cases; the 2 ambiguous switch cases are defensible-disagreement, not clear error — see §3.)

---

## 0. Method (independent re-derivation, not trust)

Per CHEAPEST-DISPROOF-FIRST, the per-entry assumption to disprove is one of two:

- **H_label-right:** "the golden label is correct, the three models slipped" — disproved by tracing
  the DAG edge and finding the models' role is the one the strict definition actually picks out.
- **H_label-wrong:** "the golden label is wrong" — disproved by tracing the DAG edge and finding
  the golden role is the one the strict definition picks out.

For each entry I traced the actual DAG path from the entry's own `derivation_pseudocode` +
`rationale` + `provenance`, applied the project's role definitions verbatim, and only then wrote
the verdict. I did **not** start from the prior read-only note on the `feat/242-multi-model-ensemble`
branch (`docs/plans/242-label-adjudication-502.md`); I re-derived from primary evidence and then
cross-checked. The re-derivation independently surfaced 5 additional unanimous-disagreement cases
the prior note had not measured (the zero-shot survival/response colliders, §1); on the two
treatment-switch cases (12, 13) it converges with the prior note's "ambiguous" call after an earlier
draft of this doc over-reached toward LABEL-CORRECT (see §2 cases 12–13 and §3).

### Evidence sources

- Golden set: `tests/fixtures/causal_role_golden_set.json` (issue #358, schema 1.0, 91 entries).
- Role definitions (authoritative): `src/data/causal_role_classifier.py:80-87` and the design
  notes at `:282-298` (the confounder-collider vs `T AND post-T-event` discriminator).
- Per-entry model votes: `/tmp/242_ab.json` (compiled few-shot, n=42) and
  `/tmp/242_ab_zeroshot.json` (zero-shot, n=42), both from `scripts/measure_ensemble_ab.py`.
- #242 findings: `docs/plans/242-p8-ab-findings.md` and `docs/plans/242-zeroshot-deconfound-findings.md`
  (on the `feat/242-multi-model-ensemble` branch).

### The role definitions applied (verbatim from `causal_role_classifier.py`)

| Role | Definition (`:80-87`) |
|------|------------------------|
| `ancestor` | Pre-prediction-time, causally upstream of target |
| `confounder` | Influences both prediction-time-knowable factors and target |
| `instrument` | Affects target only through known mediators |
| `mediator` | **On the causal path FROM treatment TO outcome** |
| `collider` | **Influenced by both target and other features** (common effect; conditioning opens a backdoor) |
| `descendant` | **Causally downstream of treatment** (typical leak pattern), **NOT on the path to Y** |

**The load-bearing discriminator (`:295-298`), in the project's own words:** a feature that
reduces to `T AND (post-T event)` is a **descendant, not a collider**, because the second "arrow"
is itself downstream of T rather than an independent cause. A **collider** requires a *second
arrowhead from an independent parent* — canonically a T–Y confounder U (baseline severity /
frailty / disease trajectory) with an arrow into BOTH the feature and Y, so that conditioning
opens `T → V ← U → Y`. A **mediator** must lie *on the path to THIS outcome* (it transmits part of
the treatment effect to Y).

These three boundaries — (mediator = on-path-to-Y), (descendant = off-path downstream of T),
(collider = common effect with an independent second parent into Y) — are the entire adjudication.

---

## 1. The unanimous-disagreement set (independently re-derived)

I did not take the candidate list on faith. I scanned both A/B output files for every
non-contaminated row where `sonnet == opus == gpt5 != ground_truth_role`:

| feature | cohort | gt | all-3 said | compiled? | zeroshot? |
|---------|--------|----|-----------|-----------|-----------|
| `ctdna_clearance_90d_flag_given_baseline_positive` | BC | collider | mediator | ✅ | ✅ |
| `dose_reduced_after_grade3_neutropenia_flag` | BC | collider | mediator | ✅ | ✅ |
| `hepatotoxicity_grade3_post_index_flag` | BC | descendant | mediator | ✅ | ✅ |
| `qtc_prolongation_grade2_post_index_flag` | BC | descendant | mediator | ✅ | ✅ |
| `best_recist_response_180d` | BC | mediator | descendant | ✅ | — |
| `switch_ai_to_fulvestrant_within_365d_flag` | BC | descendant | collider | ✅ | — |
| `post_index_thrombocytopenia_event_180d_flag` | CSU | descendant | mediator | ✅ | — |
| `post_index_switch_to_alternative_biologic_180d_flag` | CSU | descendant | collider | ✅ | — |
| `on_treatment_at_12m_flag` | BC | collider | descendant | — | ✅ |
| `alive_at_24m_flag` | BC | collider | descendant | — | ✅ |
| `alive_and_enrolled_at_180d_post_index_flag` | CSU | collider | descendant | — | ✅ |
| `any_rbc_transfusion_during_followup_flag` | PNH | collider | descendant | — | ✅ |
| `facit_fatigue_response_180d_postindex_flag` | PNH | collider | descendant | — | ✅ |

**13 distinct entries** (8 unanimous-wrong in compiled, 9 in zero-shot; 4 overlap, so 13 distinct).
This is a SUPERSET of the 4 named candidates and of the prior note's 8 cases: the zero-shot run
(which the prior note had not yet run) surfaced 5 additional `collider → descendant` survival/response
cases. Adjudicating only the 4 named cases would have missed the single most-populated slip pattern.
All 13 are adjudicated below; the §2 case numbers (1–13) are the adjudication order, NOT the row
order in this table. The 5 zero-shot-only `collider → descendant` survival/response cases are
adjudicated as **Cases 7–11** in §2.

Those 5 `collider → descendant` cases appear ONLY in zero-shot; in compiled mode the shared
Sonnet demos pulled the models toward the correct collider on these. That the zero-shot models
revert to `descendant` is itself diagnostic (see §3, slip pattern C).

---

## 2. Per-entry adjudication

For each entry: (a) the DAG traced from the entry's own derivation, (b) the strict definition
applied, (c) which role that definition selects, (d) verdict, (e) for LABEL-CORRECT, why the
models slip (the compile-set-hardening payoff).

### Case 1 — `ctdna_clearance_90d_flag_given_baseline_positive` (gt=collider → all said mediator)

- **DAG.** The feature name encodes a **conditioning restriction**: `_given_baseline_positive`
  selects the stratum with detectable baseline ctDNA. Within that stratum, ctDNA clearance by 90d
  is a common effect of (i) treatment effectiveness `T → clearance` and (ii) intrinsic tumor
  shedding biology `U_shed → clearance`, where `U_shed` (slow-vs-fast-shedding tumor biology) also
  affects PFS `U_shed → Y`. Conditioning on clearance within the baseline-positive stratum is
  selection on a common effect: `T → clearance ← U_shed → Y`.
- **Definition applied.** This is the textbook confounder-collider M-structure (`:282-298`):
  second arrowhead from an independent parent (`U_shed`) that also points at Y. Conditioning opens
  a backdoor. → **collider.**
- **Why NOT mediator (the models' answer):** a mediator must lie on `T → … → Y` and transmit the
  effect. Clearance is downstream of treatment effectiveness, but the variable as DEFINED is a
  *within-stratum selection gate*, not a node that passes the treatment effect forward to PFS. The
  models judged the raw construct ("clearance is caused by treatment ⇒ on the path ⇒ mediator")
  and **dropped the conditioning operator in the name.**
- **Provenance.** PADA-1 design, Berger 2022, PMID 35241469; collider-conditioning per
  Greenland-Pearl-Robins 1999, PMID 9888278.
- **Verdict: LABEL-CORRECT.** Slip pattern A (conditioning-operator blindness).

### Case 2 — `dose_reduced_after_grade3_neutropenia_flag` (gt=collider → all said mediator)

- **DAG.** Two parents: `T → neutropenia → (dose reduction)` AND
  `(oncologist management style + patient frailty) → dose reduction`. The second parent
  (style/frailty) is a T–Y confounder — frailty drives both dose-reduction propensity and PFS.
  Conditioning opens `T → V ← frailty → Y`.
- **Definition applied.** Independent second parent into both V and Y ⇒ confounder-collider
  (`:282-298`). → **collider.**
- **Why NOT mediator:** the models told a partial-exposure-reduction story (dose reduction lowers
  drug exposure ⇒ affects PFS ⇒ mediator). That captures only the `T → neutropenia → V` arm and
  **omits the second, independent style/frailty parent** — exactly the enumerate-the-second-parent
  failure the discriminator at `:291-294` was written to catch.
- **Provenance.** Parati 2022 (ribociclib neutropenia / dose-mod profile), PMID 35440873.
- **Verdict: LABEL-CORRECT.** Slip pattern B (missing-second-parent ⇒ collider misread as mediator).

### Case 3 — `hepatotoxicity_grade3_post_index_flag` (gt=descendant → all said mediator)

- **DAG.** `T → CYP3A4 hepatic metabolism → LFT elevation`. LFT elevation triggers dose-mods
  (modeled separately) but does **not** transmit the CDK4/6 efficacy signal to PFS. There is no
  edge `hepatotoxicity → PFS-efficacy`. So the feature is downstream of T and **off the path to Y**.
- **Definition applied.** Downstream of T, not on the path to Y ⇒ **descendant** (`:86`). It is NOT
  a collider: there is no independent second parent with an arrow into Y (no `U → hepatotox` that
  also `→ Y`); it reduces to `T → V`.
- **Why NOT mediator:** the models applied "treatment-caused intermediate ⇒ mediator" without the
  on-path-to-THIS-outcome test. An AE caused by the drug is not automatically on the efficacy path.
- **Provenance.** Parati 2022 (ribociclib hepatotoxicity is a recognized AE), PMID 35440873.
- **Verdict: LABEL-CORRECT.** Slip pattern D (treatment-caused-intermediate ⇒ off-path AE misread
  as mediator).

### Case 4 — `qtc_prolongation_grade2_post_index_flag` (gt=descendant → all said mediator)

- **DAG.** `T → hERG channel interaction → QTc prolongation`. Off-target cardiac signal; no edge to
  PFS efficacy. `T → V`, off-path to Y.
- **Definition applied.** Downstream of T, off-path ⇒ **descendant.** No independent second parent
  into Y ⇒ not collider.
- **Why NOT mediator:** identical to Case 3 — treatment-caused ≠ on-path.
- **Provenance.** Parati 2022 (ribociclib QTc cardiac safety signal), PMID 35440873.
- **Verdict: LABEL-CORRECT.** Slip pattern D.

### Case 5 — `post_index_thrombocytopenia_event_180d_flag` (gt=descendant → all said mediator)

- **DAG.** BTK-inhibitor class AE: `T → BTK-mediated platelet effect → thrombocytopenia`. Off the
  path to UAS7 (urticaria-activity) reduction — there is no mechanism by which low platelets
  transmit the anti-urticaria effect. The entry's own rationale notes a *possible* M-structure if
  AE-prone subgroups share genetic susceptibility with worse CSU prognosis, but that would make it
  a collider, NOT a mediator; the conservative label takes the dominant `T → V` off-path edge ⇒
  descendant.
- **Definition applied.** Downstream of T, off-path to Y ⇒ **descendant.**
- **Why NOT mediator:** same treatment-caused-intermediate slip (pattern D). The models had to
  assert an edge `thrombocytopenia → UAS7-response`; none exists in the BTKi pharmacology.
- **Provenance.** Lin-Suresh-Dispenza 2024 (BTKi allergy safety review), PMID 38492772.
- **Verdict: LABEL-CORRECT.** Slip pattern D.

### Case 6 — `best_recist_response_180d` (gt=mediator → all said descendant)

- **DAG.** `T (ribociclib) → tumor regression → measurable RECIST response → delayed progression →
  PFS`. Early objective response is literally a node on the treatment→outcome path; it transmits
  part of the treatment effect to PFS (most PR/CR-at-6-months patients have not yet progressed).
- **Definition applied.** On the path FROM treatment TO outcome ⇒ **mediator** (`:85`). Adjusting
  for it as a covariate would block part of the very effect being estimated (VanderWeele 2015
  mediation) — the defining harm of conditioning on a mediator.
- **Why NOT descendant (the models' answer):** the models conflated *temporal* downstream-ness
  ("response is measured at 180d, after index ⇒ downstream ⇒ descendant") with *structural*
  off-path-ness. A descendant is downstream AND off-path; RECIST response is downstream AND
  **on-path**. The models collapsed the two — the canonical temporal-downstream → descendant slip.
- **Provenance.** MONALEESA-2 final OS, Hortobagyi 2022, PMID 35263519 (objective response
  predicts survival benefit ⇒ response is on the efficacy path).
- **Verdict: LABEL-CORRECT.** Slip pattern E (temporal-downstream ⇒ on-path mediator misread as
  descendant).

### Case 7 — `on_treatment_at_12m_flag` (gt=collider → all said descendant) [zero-shot]

- **DAG.** Staying on ribociclib at 12m has two independent parents: `T → effectiveness → stay-on`
  (responders persist) AND `T → tolerability/frailty → stay-on` where **frailty/tolerance is a T–Y
  confounder** (frail patients drop out early AND have worse PFS). So `T → V ← frailty → Y`.
  This is also a special case of immortal-time / survivor selection (Suissa 2008).
- **Definition applied.** Independent second parent (frailty) into both V and Y ⇒ confounder-
  collider (`:282-298`). Conditioning on V=1 (the "still-on-treatment" cohort filter) opens the
  backdoor. → **collider.** This is DIRECTLY corroborated by the project's own compile-set exemplar
  `on_treatment_remibrutinib_at_90d_postindex_alive_flag_csu` (`causal_role_classifier.py:1697-1723`),
  labeled collider with a "why-not-DESCENDANT" rationale: a pure descendant carries only `T → V`;
  the second `Y/U → V` arrow makes it a common-descendant collider.
- **Why NOT descendant (the models' answer):** the models saw "post-index treatment-status event"
  and applied the `T AND (post-T event) ⇒ descendant` heuristic (`:295-298`) — but that heuristic
  is for features whose *second arrow is itself downstream of T*. Here the second arrow is the
  **independent frailty/U arm**, which is exactly what promotes it from descendant to collider.
  The models **dropped the second (independent) arrowhead.**
- **Provenance.** Suissa 2008 immortal-time bias, PMID 18056625; Greenland-Pearl-Robins 1999.
- **Verdict: LABEL-CORRECT.** Slip pattern C (survivor/persistence collider misread as descendant —
  dropped independent second parent).

### Case 8 — `alive_at_24m_flag` (gt=collider → all said descendant) [zero-shot]

- **DAG.** `T → survival → V` AND `frailty/comorbidity (U) → survival → V`, where `U → Y`. Survivor
  collider: `T → V ← U → Y`. PFS-among-survivors is biased.
- **Definition applied.** Independent second parent ⇒ collider (`:282-298`); mirrors the
  `alive_at_180d_observation_window` compile-set collider exemplar (`:666-711`), a binary
  sample-inclusion survivor-collider.
- **Why NOT descendant:** same dropped-second-arrowhead slip as Case 7.
- **Provenance.** MONALEESA-2 OS benefit, Hortobagyi 2022, PMID 35263519; survivor collider per
  Greenland 2003.
- **Verdict: LABEL-CORRECT.** Slip pattern C.

### Case 9 — `alive_and_enrolled_at_180d_post_index_flag` (gt=collider → all said descendant) [zero-shot]

- **DAG.** Classic survivor-collider: `T → survival/retention → V` AND `U (prognostic factors) →
  survival/retention → V`, `U → Y`. Restricting to V=1 (180d-survivors) opens `T → V ← U → Y`.
- **Definition applied.** ⇒ collider; identical structure to the `alive_at_180d_observation_window`
  exemplar (`:682-711`).
- **Why NOT descendant:** dropped second arrowhead.
- **Provenance.** Greenland 2003 collider-stratification bias, PMID 12859030.
- **Verdict: LABEL-CORRECT.** Slip pattern C.

### Case 10 — `any_rbc_transfusion_during_followup_flag` (gt=collider → all said descendant) [zero-shot]

- **DAG.** A transfusion is administered precisely when hemoglobin falls below threshold. So
  `T (iptacopan) → suppressed hemolysis → higher Hb → fewer transfusions` AND
  `Y (Hb response) → fewer transfusions` — i.e. BOTH the treatment AND the outcome point into the
  transfusion node. `T → V ← Y`. Conditioning on a variable affected by both prior exposure and the
  outcome is Hernán 2004's canonical selection-bias structure.
- **Definition applied.** Common effect of T and Y (the literal `collider` definition at `:85`,
  "influenced by both target and other features"). → **collider.**
- **Why NOT descendant:** the transfusion is not merely `T → V`; the outcome Hb-response is a second
  parent (`Y → V`). Models dropped the `Y → V` arrow.
- **Provenance.** Hernán 2004 structural selection-bias, PMID 15308962.
- **Verdict: LABEL-CORRECT.** Slip pattern C.

### Case 11 — `facit_fatigue_response_180d_postindex_flag` (gt=collider → all said descendant) [zero-shot]

- **DAG.** Fatigue improvement is a common effect: `T → hemolysis-suppression/NO-biology →
  fatigue↓` AND `Y (Hb rise) → fatigue↓`. Both treatment (direct anti-inflammatory/NO pathway) and
  the hemoglobin outcome cause fatigue improvement ⇒ `T → V ← Y`.
- **Definition applied.** Common effect of T and Y ⇒ collider (`:85`).
- **Why NOT descendant:** dropped the `Y → V` arrow (the Hb-outcome's direct effect on fatigue).
- **Provenance.** Hill TRIUMPH 2010 (hemolysis/NO biology); Risitano 2025 iptacopan PRO analysis,
  PMID 39774762; Hernán 2004.
- **Verdict: LABEL-CORRECT.** Slip pattern C.

### Case 12 — `switch_ai_to_fulvestrant_within_365d_flag` (gt=descendant → all said collider)

This is one of the two boundary disputes (the models said collider where the label says descendant
— the OPPOSITE direction from cases 7–11). It deserves the most scrutiny.

- **DAG (golden label's reading).** `T → treatment-history → switch`, and the switch is triggered by
  *suspected progression / ESR1 emergence / AI intolerance* — i.e. it is a response TO impending
  failure of the index regimen. Crucially, the golden rationale places the switch **downstream of
  the outcome trajectory** but frames it as a descendant: it is caused by treatment history and by
  early signs of the outcome, but is *not on the original-treatment → 24m-PFS path* (it is a
  reaction to failure, not a transmitter of the treatment effect).
- **The models' reading (collider).** `T → switch ← Y_nonresponse`: switch is a common effect of
  treatment assignment and disease non-response, so conditioning opens a backdoor.
- **Adjudication — both readings are defensible; this is genuinely under-determined.** The crux is
  the nature of the switch node's second parent.
  - *Descendant reading (golden label):* if the switch's trigger reduces to the patient's own
    disease trajectory toward the outcome (`T → poor-response → switch`), then the second "arrow" is
    itself downstream of T, and the feature reduces to the `T AND (post-T event)` shape the project
    classes as **descendant, not collider** (`:295-298`, `discontinuation_flag` precedent).
  - *Collider reading (the models):* but the fixture rationale (`causal_role_golden_set.json`,
    `switch_ai_to_fulvestrant` entry) names the triggers as *suspected progression / ESR1 emergence /
    AI intolerance*. ESR1-mutation emergence and progression biology are plausibly **independent
    prognostic drivers** that point into BOTH the switch decision AND 24m-PFS — i.e. a genuine
    second parent with an arrow into Y, which is the `T → switch ← U_prognostic → Y` collider
    structure. The derivation does NOT prove the trigger reduces to a pure T-downstream chain; it
    leaves the second parent's independence **unspecified**.
  - I cannot eliminate the collider reading from the derivation as written. The descendant label is
    defensible (and is what a conservative drop-as-predictor policy wants), but it is **not strictly
    preferred** over collider on the evidence in the fixture. This is the same conclusion the prior
    read-only note reached, and on re-examination it is the more rigorous one.
  - The pipeline impact is identical either way (both descendant and collider route to
    drop-as-predictor), so no estimation harm rides on the choice — but the *adjudication* should not
    overstate certainty.
- **Provenance.** MONALEESA-3 (ribociclib + fulvestrant), Neven 2023, PMID 37653397 — corroborates
  switching as a downstream clinical decision (supports the descendant reading), but the fixture
  provides no citation establishing whether ESR1/progression is an *independent* Y-bound parent
  (which would support collider). The provenance is therefore **insufficient to break the tie.**
- **Verdict: LABEL-AMBIGUOUS** (descendant defensible; collider defensible; the derivation under-
  determines the second parent's independence). NOT a label error — the golden `descendant` is a
  reasonable, conservative choice — but NOT strictly correct over collider either. Slip pattern F
  (the descendant↔collider boundary on treatment-reactive switches is genuinely hard, which is why
  all three models converged on the alternative reading).

  *Internal-consistency note:* Case 2 (`dose_reduced_after_g3_neutropenia`) IS labeled collider
  because its second parent (oncologist style + frailty) is explicitly named as an independent T–Y
  confounder. The switch features differ only in that their second parent (ESR1/progression) is
  named but its independence is left implicit. Making the second-parent classification explicit in
  the rationale for the switch features — exogenous-prognostic-U (⇒ collider) vs pure-outcome-
  trajectory (⇒ descendant) — would resolve the ambiguity. Flagged to #502 as the documentation
  improvement that would let a future reviewer settle the label.

### Case 13 — `post_index_switch_to_alternative_biologic_180d_flag` (gt=descendant → all said collider)

- **DAG.** `T → index-drug → switch`, triggered by non-response Y. Same structure as Case 12 in the
  CSU cohort. The fixture rationale is even more explicit than Case 12: it states switching is
  downstream of **both** treatment AND the outcome — "non-response Y drives switching" — and calls
  it a **descendant of Y** (per VanderWeele 2015 / EAACI 2021 treatment-ladder).
- **Adjudication — genuinely under-determined, leaning more toward collider than Case 12.** The
  fixture's own wording ("non-response Y drives switching") describes `Y → switch`, and treatment
  also drives the index drug, so `T → switch ← Y`. That is the literal collider shape (`:85`, "common
  effect"). The descendant defense is that a switch is a *reaction to* the realized outcome rather
  than a transmitter of the treatment effect, and VanderWeele's "descendant of Y" framing supports
  treating it as off-path-and-droppable. Both are defensible; the derivation does not force one. As
  with Case 10/11 (transfusion, FACIT — also `T → V ← Y`, labeled collider), the bare common-effect
  shape would point to collider, but the golden set chose descendant here because the switch is a
  discretionary clinical decision rather than a mechanistic outcome surrogate. That distinction is
  reasonable but not strictly compelled.
- **Provenance.** Giménez-Arnau 2025 REMIX switch-pathway, PMID 41115533, documents the switch
  pathway but does NOT establish whether the switch is a pure descendant-of-Y vs a common-effect
  collider; VanderWeele 2015 / EAACI 2021 are cited in the rationale prose but are not in the
  fixture provenance block. Insufficient to break the tie.
- **Verdict: LABEL-AMBIGUOUS** (descendant defensible; collider arguably the more literal reading of
  the fixture's own "T and Y both drive switch" wording). NOT a label error. Slip pattern F. Same
  documentation-improvement note as Case 12.

---

## 3. Tally and the systematic-slip taxonomy (compile-set hardening payoff)

**11 LABEL-CORRECT, 0 LABEL-WRONG, 2 LABEL-AMBIGUOUS** (cases 12, 13 — the treatment-switch
features).

No golden label requires **correction**: the 2 ambiguous cases are NOT wrong (the golden
`descendant` is a defensible, conservative choice), they are merely not strictly preferable over the
models' `collider` reading on the evidence in the fixture. This matches the prior read-only note's
"GENUINELY-AMBIGUOUS" call for these two — on re-examination that was the more rigorous conclusion,
and an earlier draft of this doc over-reached in promoting them to LABEL-CORRECT. The
descendant↔collider boundary for treatment-reactive switches is genuinely under-determined by the
derivations as written; both readings route to drop-as-predictor, so no estimation harm rides on
the choice, but the adjudication should not overstate certainty.

**Why the collider semantics are consistent across cases 10/11 (collider-CORRECT) and cases 12/13
(ambiguous), per HIGH-finding reconciliation:** all four have the bare `T → V ← Y` common-effect
shape. The difference is the *mechanistic determinism* of the second arrow. For transfusion
(case 10) and FACIT-fatigue (case 11), the `Y → V` arrow is **mechanistically forced**: a
transfusion is administered *because* hemoglobin crossed a threshold, and fatigue improves *because*
hemoglobin rose — the outcome physiologically causes the feature, so collider is not in dispute. For
the treatment switches (cases 12/13), the `Y → V` arrow is a **discretionary clinical decision** that
*reacts to* the outcome rather than being mechanistically caused by it, which is precisely why
VanderWeele's "descendant of Y" framing is a legitimate alternative reading and why the label is
defensible-but-not-forced. This is a real, statable distinction — not a post-hoc rescue — but it is
subtle enough that it belongs explicitly in the fixture rationale (see §4 rec 2).

**⇒ The #242 cross-vendor correlated failure (30 % compiled, 40 % zero-shot, rising under
de-confounding) is REAL, task-structural model failure — not label noise.** This is the
decision-grade answer to the experiment's open question, and it independently corroborates the
#242 zero-shot conclusion (correlation is intrinsic to the frontier models, not a prompt artifact).

The 13 cases collapse into **six systematic slip patterns**, all compile-set-fixable:

| Pattern | Cases | What the models do wrong | Discriminator example to add |
|---------|-------|--------------------------|------------------------------|
| **A. Conditioning-operator blindness** | 1 | Ignore a conditioning restriction encoded in the name (`_given_baseline_positive`) ⇒ call a selection-collider a mediator | An entry whose NAME carries a `_given_X` conditioning operator ⇒ collider-by-conditioning |
| **B. Missing-second-parent (collider→mediator)** | 2 | Tell the single-arm exposure story, omit the independent T–Y-confounder parent | Already partially covered by `dose_reduced` exemplars; add the explicit "enumerate the second parent" instruction |
| **C. Dropped-second-arrowhead (collider→descendant)** | 7,8,9,10,11 | See a post-index event, apply `T AND post-T ⇒ descendant`, but MISS the independent survival/frailty/outcome second arrow | Survivor/persistence/transfusion/PRO-response colliders with explicit `T → V ← U/Y` rationale (the `on_treatment_..._alive_flag` and `alive_at_180d` exemplars already model this — extend to OS/transfusion/PRO shapes) |
| **D. Treatment-caused-intermediate (descendant→mediator)** | 3,4,5 | Call any drug-caused AE a mediator without the on-path-to-THIS-outcome test | Off-path AE exemplars (hepatotox / QTc / thrombocytopenia) ⇒ descendant with explicit "no edge to efficacy Y" |
| **E. Temporal-downstream (mediator→descendant)** | 6 | Conflate "measured later" with "off-path" | Early on-path surrogate (RECIST / biomarker response) ⇒ mediator despite late measurement |
| **F. Genuinely-hard descendant↔collider boundary (descendant↔collider)** | 12,13 (LABEL-AMBIGUOUS) | The treatment-reactive switch sits ON the descendant↔collider boundary; the derivation under-determines whether the `Y → switch` arrow is mechanistic (collider) or discretionary-reactive (descendant). The models read collider; the label reads descendant; both defensible. | A treatment-switch exemplar that makes the second-parent classification explicit — mechanistic-outcome-cause ⇒ collider vs discretionary-reaction-to-outcome ⇒ descendant — so the boundary is teachable rather than left implicit |

**Why all three vendors fail correlated (and why GPT-5 did not help):** every slip is a *single
shared reasoning shortcut* about the collider/mediator/descendant boundary, not a vendor-specific
quirk. Patterns C and F both center on the SECOND PARENT of a post-index node: pattern C is a clear
*error* (dropping a mechanistically-forced second arrowhead ⇒ collider-misread-as-descendant);
pattern F is the genuinely *hard boundary* (whether a discretionary treatment-switch's outcome-
reactive arrow makes it a collider or a descendant is under-determined by the derivation, so all
three models reading collider is a defensible-disagreement, not a clear error). Because the
collider/descendant/mediator boundary is the *task structure's* hardest region, all three frontier
models converge on the same reading, so agreement-or-escalate ensembling **ratifies** it by majority
vote whether it is wrong (patterns A–E) or merely contested (pattern F). This is exactly
why multi-vendor diversity did not deliver the asymmetric-failure signal #240 AC3.5 assumed — and
it is the empirical justification for the non-LLM structural check direction (#501 Step C): a
deterministic "post-index timing ⇒ examine second parent" prior cannot share the LLM blind spot.

---

## 4. Recommendations

1. **#358 golden set — NO label corrections.** 11 of the 13 unanimous-disagreement labels are
   *correct* (clearly preferred over the models' reading); the remaining 2 (the treatment-switch
   features, cases 12/13) are *defensible but ambiguous* — the golden `descendant` is reasonable but
   not strictly preferable over the models' `collider`. No label is *wrong*, so no JSON mutated. Per
   the methodology, a doc-only outcome with zero corrections is a valid result; inventing a test for
   a non-existent code change would be lazy programming.
2. **#358 documentation (the action item from the 2 ambiguous cases — for human sign-off via #502).**
   For the treatment-switch features (cases 12, 13), make the **second-parent classification
   explicit** in the rationale: is the `Y → switch` arrow a *mechanistic* outcome-cause (⇒ collider,
   like transfusion/FACIT) or a *discretionary clinical reaction* to the outcome (⇒ descendant)? The
   golden set chose descendant on the discretionary-reaction reading; stating that choice removes the
   ambiguity for a future reviewer. (The dose-reduction feature, case 2, already names its
   independent second parent explicitly and needs no change.) Whether to (a) annotate the rationale,
   (b) keep descendant as-is, or (c) flip either to collider is a human product decision — no pipeline
   impact (descendant and collider both ⇒ drop-as-predictor), but it is a real label-semantics call.
3. **Compile-set hardening (the high-value lever — separate work, separate issue).** Add the six
   discriminator example types in §3. This is the payoff of the adjudication: the correlated failure
   is fixable by teaching the boundary, not by ensembling. (Filing/execution is a separate
   authorized task — this doc only recommends.)
4. **#240 AC3.5 / #501.** The correlated failure is intrinsic and shared across vendors, so
   multi-vendor agreement cannot be the gate's independence signal for these case types. This
   strengthens the case for a non-LLM structural/temporal check (#501 Step C). This doc is one of
   three independent lines of evidence (compiled A/B, zero-shot de-confound, this label
   adjudication) all reaching the same conclusion.

## 5. Why #502 stays OPEN

Per the task framing, label changes need human sign-off before #502 closes. Here there are **no
label changes proposed** — but two items need a human decision: (a) the 2 LABEL-AMBIGUOUS
treatment-switch features (keep `descendant`, flip to `collider`, or annotate the rationale with the
mechanistic-vs-discretionary second-parent distinction), and (b) the compile-set hardening + the
#240/#501 implication. `Refs #358 #502`, no closing keyword; leave #502 open for the user to act on
the recommendations.
