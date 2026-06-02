# Optum CSU biologic-initiation — structural causal-role attestations (research record)

**Track-2B-v3 Phase 2.** Literature-grounded causal-structure attestations for every
SAFE (pre/at-index) feature of the Optum chronic-spontaneous-urticaria (CSU)
biologic-**initiation** cohort, authored so the deterministic Layer-4 structural
decider (`src/ml/causal_role_dgp/extractor.py::extract_role`) can classify each
feature's causal role. The decider **stays DARK** — these attestations are inert
until an explicit, cohort-scoped ramp.

## Causal frame

- **Y (outcome)** = `initiated_biologic_180d` — initiated a biologic within 180 days post-index.
- **T (treatment)** = `biologic_initiation` — the clinical decision/event of starting a biologic; edge `T→Y`.
- This is a **prediction** cohort (binary classification; `run_optum_tier0_test.py`
  sets `prediction_target=initiated_biologic_180d`, no randomized treatment). So the
  treatment-effect taxonomy partially degenerates: every legitimate pre-index feature
  is a **confounder** or **instrument** of the initiation decision (all **ACCEPT**).
  The role is *derived from the authored edges*, never declared.

## Headline verdict: all 110 SAFE features ACCEPT — leakage is *structurally precluded*

Three independent literature-research agents (access/demographics, disease/comorbidity,
pharma/labs) authored edges for all 110 SAFE features. Result (independently re-derived
via `derive_structural_role`):

| bucket | role | count |
|---|---|---|
| ACCEPT | confounder | 93 |
| ACCEPT | instrument | 17 |
| LEAK | — | 0 |
| unclassifiable | — | 0 |

**No SAFE feature is a leak, and it cannot be**, verified against `scripts/convert_optum_rwd.py`:
- Feature windows are strictly pre-index: `[index_date−180d, index_date−1]` (the `<= lb_end` filter excludes the index date) — for diagnoses, drug fills, and labs.
- `*_days_since_last_fill = index_date − max(fill_date < index_date)` ⇒ always ≥1; mathematically cannot capture a post-index fill.
- `*_result_last` / `*_abnormal_flag` take the last lab `< index_date`.
- **Biologic rows are stripped** from the non-target drug-class features (`bio_mask`), so no prior-drug column can encode the index biologic or a post-index fill — the one outcome-derived path of concern is impossible.

The cohort is **dx-anchored** (index = qualifying L50.x diagnosis, not the biologic fill)
and **treatment-naive** (`_had_biologic_pre_index` enforced). T and Y both occur *after*
every SAFE feature is measured ⇒ none can be a mediator/collider/descendant.

**Implication:** on Optum the structural decider is a *second-line* safety check (it would
correctly keep all 110, and would only catch a feature *mislabeled* pre-index whose true
derivation peeked post-index). The leak-**catching** capability across all six roles was
proven separately on the labeled CSU_remibrutinib golden cohort (Phase 1, PR #542:
0 missed leaks, 31/31 leak-decision).

## The two authored edge patterns

| role | edges | applies to |
|---|---|---|
| **confounder** | `feature→T`, `feature→Y`, `T→Y` | disease-severity / treatment-burden / comorbidity / baseline-lab proxies (common cause of the decision AND the outcome) |
| **instrument** | `feature→T`, `T→Y` | access drivers of the *decision only* (no direct disease-biology path to Y) |

## Role assignments by family (with grounding)

**Instruments (17)** — drive *whether/which* biologic is initiated, not the disease course:
- Geography/residence — `zip5`, `zip3`, `zip_code`, `geographic_region`, `urban_rural_code`: specialist proximity & regional adoption shift omalizumab use (PMID 36481046 — use varied by patient ZIP, higher nearer the allergist practice; PMID 40169378 — BRIT registry, deprivation/distance affect access).
- Coverage/payer — `insurance_product`, `plan_type`, `payer_category`: prior-auth / step-therapy / formulary gate initiation (Aetna CPB 0670; Cigna & UHC Xolair PA policies; PMID 40004611 cost/access disparities).
- Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934).
- Calendar anchors — `index_date`, `lookback_start_date`: temporal adoption of biologics (PMID 32382379; PMID 40004611).

**Confounders (93)** — common causes of the escalation decision and the outcome:
- Demographics — `age_at_index`, `age_group`, `gender`, `primary_diagnosis_code` (PMID 34622498; PMID 32382379).
- Utilization severity proxies — `office_visits_total`, `office_visits_pcp`, `ed_visits_total`, `ed_visits_urticaria_angio`, `hospitalizations_total`, `unique_providers` (PMID 29429043; PMID 34622498).
- CSU dx burden — `dx_l50_1/8/9_count`, `dx_total_csu`, `dx_angioedema_count`, `csu_dx_intensity` (PMID 39325444; PMID 34984792).
- Comorbidities — atopic dermatitis / asthma / allergic rhinitis / thyroid-autoimmune / angioedema / nsaid-hypersensitivity / anxiety / depression families (`has_*` + `*_claim_count`), `atopy_score`, `mental_health_flag`, `elixhauser_score`, `charlson_score`, `comorbidity_load_total` (PMID 42050840; PMID 40220911; PMID 37378208; PMID 39949534).
- Disease duration/chronicity — `months_since_first_dx`, `csu_chronicity` (authored to the true clinical meaning; see caveat).
- Prior pharmacotherapy (the EAACI/GA²LEN CSU ladder: 2nd-gen H1 → up-dose → +H2/+LTRA → omalizumab → cyclosporine) — `h1_1g`, `h1_2g`, `h2`, `ltra`, `sys_steroid`, `top_steroid`, `immunosupp` families (×4 cols), `polypharmacy_breadth` (all.15090 EAACI guideline; PMC6735630 steroid-as-severity).
- Baseline labs / endotype markers — `ige_total`, `eosinophil`, `crp`, `tpo_ab`, `free_t4`, `tsh`, `ana`, `cbc` families (×3 cols), `lab_workup_completeness` (WAO 1939-4551(24)00036-X IgE/eos predictors; falgy.2025.1706705 autoimmune endotype; PMID 37634502 CRP).

## Honest caveats (all stay ACCEPT — no leak-decision impact)

1. **T ≈ Y near-deterministically.** T (the initiation decision) and Y (the realized 180d-initiation flag) are essentially the same post-index event, so `T→Y` is near-deterministic. This does not change any role — every SAFE feature is a pre-index parent of both — but the "outcome" is the treatment event itself, not a downstream response (unlike the CSU_remibrutinib golden cohort where Y = UAS7 response).
2. **Confounder-vs-instrument ambiguity (21 access + the 4 `top_steroid_*` features flagged by the agents).** Specialist-access/payer features could instead proxy disease severity (confounder); `top_steroid` is not on the CSU systemic ladder. Both readings are ACCEPT, so the leak-vs-accept bucket is unchanged. The extractor also safely demotes a non-exogenous "instrument" to ancestor (still ACCEPT).
3. **`months_since_first_dx` and `csu_chronicity` are degenerate constants** in the current converter (no patient-level variation). Authored to their true clinical meaning (duration/chronicity drive escalation) per the authoring guide's "author the real mechanism" rule; ACCEPT regardless.
4. **`has_nsaid_hypersensitivity`** — Asero (PMID 34284571) found NSAID hypersensitivity is independent of CSU severity/pathogenesis, weakening its `feature→Y` arm (confounder vs near-null ancestor). ACCEPT either way.

## D4 empirical crosscheck — MEASURED (2026-06-02)

The faithful crosscheck that was previously deferred as an "argued structural certainty"
has now been **run on the real cohort** — converting the argument into measured evidence
(cheapest-disproof: the structural-preclusion claim was a hypothesis until the empirical
signal confirmed it).

**Run:** real Optum CSU biologic-initiation cohort, n=1294 (37 positive, 2.9%), faithful
Tier-0 `--step 2` data-prep (`run_optum_tier0_test.py --cohort initiation --step 2`,
215 s). Structural roles re-derived offline via `derive_structural_role` at current `main`;
empirical per-feature severities taken from the run's `adaptive_verdicts`
(Layer-1 / Layer-3 FDR); paired by `compare_structural_vs_empirical`.

**Result — `gate_passed = True`, `missed_leaks = 0`:**

| quantity | value |
|---|---|
| structural roles (110 SAFE) | 93 confounder + 17 instrument; **0 leak, 0 unclassifiable** (all ACCEPT) |
| empirical verdicts | 84 total — 79 `info`, 1 `moderate`, **4 `high`** |
| empirical `high` features | `discontinued_180d`, `persistent_at_180d`, `treatment_initiated`, `discontinuation_flag` — **all forbidden post-index targets; NONE in the SAFE set** |
| SAFE features flagged high/critical | **0** |
| `missed_leaks` (structural ACCEPT ∧ empirical leak) | **0** |
| `disagreements` | 0 |

So **no SAFE attested feature is empirically a leak** — the structural decider's ACCEPT of
all 110 agrees with the data-driven signal.

**Honesty caveats (the measurement is conservative, not a false green):**
1. **This run was the manifest-OFF / stricter regime.** The `--step 2`-alone path did not
   propagate the `optum` manifest into the declared-safe path (artifact
   `feature_manifest_source = None`; the SAFE confounders `atopy_score` / `charlson_score` /
   `*_tested` were FDR-over-dropped and the QC completeness gate blocked — the documented
   #594/#604 manifest-OFF signature). Declared-safe σ-inflation (manifest-ON, the production
   path #545) can only **reduce** SAFE-feature flagging, so `missed_leaks == 0` holds *a
   fortiori* under production manifest-ON. The stricter test passing is stronger evidence,
   not weaker.
2. **Coverage 80/110.** 80 SAFE features received an empirical verdict (all `info`/`moderate`);
   ~30 were dropped pre-scan as degenerate/all-null columns (`pearsonr` is undefined on
   constant input) and are **incapable of leaking**, so the coverage gap hides no leak.
3. **The QC block is a data-quality artifact** (completeness, from the manifest-OFF over-drop),
   orthogonal to the leakage crosscheck — the leakage verdicts were produced by the scan
   before/independent of the completeness gate.
4. **Undetectable case (unchanged):** a feature where structural ACCEPTs *and* the empirical
   signal also misses a real leak is not detectable by this crosscheck; it is mitigated by
   the per-feature domain review (D3) and by EnsembleVoter precedence (empirical-high wins
   before the structural rule fires). A production manifest-ON confirmatory run would cover
   all 110 SAFE features and match the production config exactly — optional, as it can only
   confirm the gate.

## Still deferred (NOT in this PR)

The **cohort-scoped activation ramp** (D5.1 flip `adaptive_structural_decider_enabled` for
Optum only, D5.2 monitoring, D5.3 rollback runbook) and the **D3/D6 human domain sign-off**.
Activation on this all-ACCEPT cohort is functionally a no-op for leak-catching (its only
effect is replacing the LLM with the deterministic decision for these features). The decider
stays **dark**; the wiring (`adaptive_structural_decider_enabled`, PR #541) and the no-label
crosscheck gate are in place, and the gate is now **measured**, not argued.
