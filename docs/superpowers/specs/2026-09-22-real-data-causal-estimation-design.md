# Real-data causal estimation for CSU: design (2026-09-22)

**Owner decision (2026-09-22):** the goal is causal estimation exclusively. The
prediction frames stay as they are; this program adds a *causal* path over the same
real data. Proposal accepted as scoped below.

## 1. Problem

Today no causal DAG is built and no causal effect is estimated on real claims data.
The measured reasons (`docs/demos/results/2026-09-22_discovery_real_claims_disproof/`):

- The causal agent (`src/agents/causal_impact`, routes under `src/api/routes/causal/`)
  is reachable only for the three synthetic tables in
  `datasets.py::_CAUSAL_DATASET_SPECS`.
- The Optum mart converter (`scripts/convert_optum_mart.py`) selects the treated
  cohorts on `index_biologic_brand` and then drops that column, because the mart
  manifest (`optum_mart_feature_manifest.py`) declares it a post-index
  `mart_treatment` column that would leak the prediction target. Correct for
  prediction; wrong for causal estimation, where that column **is the treatment**.
- Guided structure discovery fails on real claims frames (singular correlation
  matrix, rank 45/59) and is too slow when pruned (230 s per PC fit on 43 covariates).
- The authored causal structure (Layer-4 attestations) exists for one Optum pair only,
  was authored by research agents, was never human-validated, and is not a DAG.

What the real data holds (`data/rwd/Optum_Parquet/Optum_enriched.parquet`, 814,587
patients, index dates 2016-06-01 to 2024-12-31, last observed 2025-09-30):

| Fact | Value |
|---|---|
| Treatment | `index_biologic_brand`: XOLAIR 17,765 / DUPIXENT 6,664 / no_treatment 790,158; `treatment_start_date` for all 24,429 initiators |
| Treated cohorts (mart persistence / discontinuation) | n = 15,209; XOLAIR 11,009 / DUPIXENT 4,200 (joined on `PAT_<patid>`) |
| Outcomes observed for both arms | `persistent_at_180d` (raw 52.3 % vs 34.7 %), `discontinued_180d` (10.6 % vs 12.2 %), `biologic_switch_180d_flag` (0.96 % vs 1.24 %) |
| Confounders | the mart manifest's 64 safe baseline features, measured at the diagnosis index, which precedes treatment start |
| Remibrutinib | absent from every real drop (Rhapsido approved 2025-09-30, the drop's last day; the matcher recognises only omalizumab/dupilumab) |
| Initiation vs none | blocked: non-initiators carry no post-index follow-up (all outcome columns zero, `last_observed_date` null) |

## 2. Goal and non-goals

**Goal.** A validated causal DAG and an adjusted causal estimate on real claims data for
the CSU escalation-therapy decision, built so that remibrutinib vs competitor becomes a
registry entry when post-launch data arrives.

**Non-goals.** Changing the prediction frames or the tier-0 pipeline; CRM data;
initiation-vs-none (needs a re-pull with follow-up for the untreated).

## 3. Lanes

Five lanes, each its own plan → implementation → PR. Order: A, then E, then B and D
(independent of each other; both need A's frame, B needs E's voters), then C. Lane A is
usable on its own (curated-confounder DAG, discovery off); Lane E turns the four
feature-role voters on for the causal cohorts and makes their verdicts an input to DAG
construction; Lane B upgrades A's structure with a validated prior authored over that
evidence; Lane D makes guided discovery run on claims frames so it can corroborate or
challenge that prior; Lane C is pre-wiring.

### Lane A — real-data causal pipeline, rehearsed on Dupixent vs Xolair → persistence

1. **Causal cohort export.** New cohort `persistence_causal` in
   `convert_optum_mart.py` (registry entries `COHORT_TARGETS`, `_SELECTOR_BY_COHORT`,
   `_ANCHOR_BY_COHORT`, `_OUTPUT_BY_COHORT`): the persistence selector's rows, the 64
   `MART_SAFE_FEATURES`, the ids, **plus** `index_biologic_brand` (as-is) and a binary
   `treatment_dupixent` (1 = DUPIXENT, 0 = XOLAIR), `treatment_start_date`, and four
   outcomes: `persistent_at_180d_g28` (primary; see §7), `discontinued_180d`,
   `biologic_switch_180d_flag`, and the shipped `persistent_at_180d`.
   Output `data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet`
   with `is_synthetic = false` on every row. The prediction cohorts are untouched; the
   manifest's forbidden list still applies to them. A unit test asserts the causal
   frame carries the treatment and the prediction frame does not.
2. **Table + load.** Migration `148_optum_biologic_persistence_causal.sql` creates
   `public.optum_biologic_persistence_causal` (one column per exported field,
   `is_synthetic BOOLEAN NOT NULL DEFAULT false`, indexes on treatment and outcomes).
   `scripts/load_optum_causal_cohort.py` loads the parquet (idempotent upsert on
   `patient_id`), with `--dry-run` and a row-count + arm-split verification printed
   after the load. Loading production is an owner-GO step; the migration rides the
   deploy as usual (`scripts/run_migrations.sh`).
3. **Registry.** `datasets.py`: `optum_biologic_persistence` in `_CAUSAL_DATASET_SPECS`
   (treatment `treatment_dupixent`; outcomes the four above, `persistent_at_180d_g28`
   first; covariates the 64
   baseline features), `_CAUSAL_PHYSICAL_TABLE`, `_CAUSAL_NUMERIC_COLUMNS`,
   `_CAUSAL_CATEGORICAL_COLUMNS` (payer/geography/gender), `_CAUSAL_BRAND_COLUMN`
   (`index_biologic_brand`), `_CAUSAL_NEGATIVE_CONTROL_OUTCOMES`: **none declared**
   until the omitted-confounder experiment the file mandates is run on this source; the
   runner then emits SKIPPED `no_negative_control_declared`, never a fabricated PASS.
   `_load_agent_estimation_frame` (`loaders.py`) reads it like any single-table
   dataset with the real-mode provenance filter.
4. **Run shape.** Until Lane D lands, the API default for this dataset is
   `auto_discover=False` (the measured failure mode) and the curated DAG: every covariate a common cause of T
   and Y, the estimand edge, adjustment set = the covariates (the graph builder's
   existing manual path). The agent's `limit` cap (20,000) covers the full cohort.
   Randomized design false. Refutation suite, E-value and expert-review consult run
   unchanged; a REVIEW band queues a review, `CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL`
   stays advisory.
5. **Evidence.** A cert under `docs/demos/results/<date>_optum_biologic_persistence_cert/`:
   the run's response (ATE with interval, refutation verdicts, gate decisions) for
   all three outcomes, plus the two caveats a reviewer must weigh, recorded as data:
   `treatment_response` is brand-coupled (4,109 "controlled" on Xolair vs 30 on
   Dupixent) and is **not** used; median coverage runs are 14 d (Dupixent) vs 45 d
   (Xolair), so the persistence definition's 60-day gap threshold is dosing-interval
   sensitive.

Gates: unit tests for export, loader, registry and spec consistency (the existing
registry-consistency tests must include the new key); a real-DB probe of the loaded
table's arm split against the parquet; the API run recorded as the cert.

### Lane E — the four voters, live and wired into DAG construction

Verified state (2026-09-22). Four independent processes assign each feature a causal
role in the ML data-preparer node (`adaptive_validity_check`, voter in
`src/data/kg/ensemble_voter.py`): Layer 1 declarative contract (live), Layer 3
adversarial probe (live), Layer 2 knowledge graph and Layer 4 LLM classifier. The last
two are dark, for three different reasons:

| Voter | Why it is dark | Evidence |
|---|---|---|
| Layer 4 LLM | flag off (`adaptive_layer4_enabled` defaults False; no script or config sets it); audit-only (`ADAPTIVE_LAYER4_LLM_DECIDES` unset); the compiled artifact `artifacts/dspy/causal_role_classifier.json` is committed but excluded from the image by `.dockerignore`'s `*.json` (only `config/**/*.json` and `data/kg_cache/**` are un-ignored) | `adaptive_validity_check.py:3776`, `ensemble_voter.py:195-204`, `.dockerignore:103-124` |
| Layer 2 KG | activation bound only for manifest `optum` (`src/data/kg/activation.py::KG_ACTIVATIONS`), shadow mode, one cache (`data/kg_cache/1cdaa038__96bfd2e0.json`, target omalizumab RXNORM:302379); nothing for `optum_mart`, `csu`, `optum_hcp`, so those cohorts emit `no_signal` by construction; promotion is operator-driven | `activation.py:34-39`, memory note 2026-09-22 |
| Causal agent | `graph_builder` consumes none of the four; every registry covariate is adjusted blind | grep: no voter import under `src/agents/causal_impact/` |

1. **Ship the voters.** Un-ignore `artifacts/dspy/causal_role_classifier.json` (a
   packaging guard test mirroring `tests/unit/test_data/test_kg/test_kg_cache_packaging.py`
   proves it is in the image). Build and commit KG caches for `optum_mart` and `csu`
   with `scripts/build_kg_cache.py --live` against both treatment concepts of the Lane A
   contrast (omalizumab RXNORM:302379 and dupilumab, whose RxCUI is resolved through
   RxNav in-lane and pinned by a test); add their `KG_ACTIVATIONS` entries in shadow.
   Define a cohort-scoped **causal activation profile** on the existing per-run config
   (`adaptive_layer4_enabled=True`, KG shadow, structural decider on) that the causal
   path applies; no global flag flips, `ADAPTIVE_LAYER4_LLM_DECIDES` stays off — under
   this design the voters inform, the author and the human decide.
2. **Feature-role panel.** `src/causal_engine/feature_role_panel.py`: runs the existing
   node's four layers (reused, not re-implemented) over a causal frame's covariates for
   a (manifest source, T, Y) and returns, per feature: the Layer 1 verdict, the Layer 3
   statistic and severity, the Layer 2 signal with its supporting edges, the Layer 4
   role, mechanism and citation verdicts, and the ensemble verdict (`decided_by`,
   `final_role`, confidence, disagreements). Serialisable; recorded as evidence.
3. **Panel → DAG construction.** (a) The panel is part of every feature's brief to the
   structural author (Lane B) and is written per feature into `agent_assessment_json`
   for the reviewer. (b) Hard constraints on authored structure: a Layer 1 high veto
   (post-index) forbids `feature → T`; a Layer 3 high veto (leak) excludes the feature
   from any adjustment set whatever the authored edges say. (c) Cross-check: the
   author's derived role vs the ensemble `final_role`; disagreement sets
   `ambiguous=true` and the review shows both. (d) The causal agent reads the panel:
   `anchored_confounders` = features whose approved structure derives confounder or
   instrument and that carry no leak verdict; leak-verdict covariates are removed from
   `modeled_confounders` with a named warning in the response, instead of being
   adjusted for blind as today.
4. **Measure.** Re-run `scripts/measure_layer4_precision.py` with the artifact as
   packaged (the ≥0.95 instrument-precision gate); run the panel on the real persistence
   frame's 64 covariates and record which layers fired, how many features each decided,
   and the abstain rate — a null is a finding. KG promotion from shadow follows the
   existing `compute_promotion_eligibility`, decided by the owner on that measurement.

### Lane B — structural author with human validation

Authors the DAG for the CSU escalation decision from the guide
(`docs/layer4/structural_attestation_authoring.md`), validates it in the expert-review
queue, and feeds the approved structure back to Lane A as its structural prior.

1. `src/data/kg/structural_author.py`: a DSPy program (OpenAI via
   `src/optimization/dspy_lm.py::ensure_dspy_configured`) whose instructions are the
   guide's sections 0–6 verbatim. Inputs per feature: the brief the Layer-4 classifier
   already gets (`adaptive_validity_check._build_layer_4_inputs`) plus the Lane E
   feature-role panel for that feature. Outputs: edges over
   `{feature, T, Y, U_*}`, cited rationale per non-obvious edge, `ambiguous`, expected
   role. Post-processing: `extract_role` derives the role; citations go through
   `src/data/kg/citation_resolver.CitationResolver.verify_citation`; per-edge grade
   `direct` / `family` / `unsupported`. Every record is stamped with model id, prompt
   hash and guide hash. `CausalStructureAttestation` gains `provenance`
   (`machine` | `machine_reviewed` | `human`). The 110 existing Optum entries are
   labelled `machine`: they are research-agent output with no human sign-off. The
   owner may relabel them after reading the Optum diff (run (b) below).
2. `src/ml/causal_role_dgp/assembler.py`: unions the fragments for one (T, Y) into
   a cohort DAG; adjustment set via the graph builder's backdoor finder; per-edge
   provenance; emits the `dag_structure_json` shape `DagPanel` renders and a
   per-feature evidence table for `agent_assessment_json`.
3. `scripts/measure_structural_author.py`: author once on the 91 blind briefs
   (`tests/fixtures/causal_role_csu_blind_briefs.json` and the golden set), score
   once against `ground_truth_role`; reports per-role precision/recall and the
   missed-leak rate. **Gate: zero missed leaks**, else the report is the deliverable.
4. `scripts/author_cohort_dag.py --manifest {optum_mart,optum} --treatment --outcome
   [--review]`: writes `docs/layer4/generated/<manifest>_<T>_<Y>/{attestations.json,
   dag.json, review.md}`; with `--review` opens an `expert_reviews` row
   (`review_type='initial_dag'`, `dag_structure_json`, `agent_assessment_json`).
   Two runs: (a) `optum_mart`, T = the CSU escalation-therapy choice framed as
   *remibrutinib vs competitor biologic*, Y = persistence at 180 d, over the 64
   baseline features → the review queue (the human gate). Stated assumption, which
   the reviewer is asked to confirm as a checklist item: remibrutinib enters at the
   same decision point as the biologics (second line after H1-antihistamine failure),
   so the confounders of *which* escalation therapy are the same set; Lane A's
   rehearsal therefore uses this DAG with `treatment_dupixent` on the same
   confounders. (b) `optum`, T =
   `biologic_initiation`, Y = `initiated_biologic_180d` → diff against the 110
   existing attestations (agreement per edge and role; disagreements listed with both
   rationales; no threshold).
5. Feedback to Lane A: an approved review's edges become the dataset's
   `anchored_confounders` (the structural-prior channel already in `agent.py`) via a
   loader that reads the approved `attestations.json` by review id; unapproved
   machine attestations are never used as priors and the structural decider treats
   them as audit-only.

### Lane D — guided discovery on claims frames

Measured starting point (`docs/demos/results/2026-09-22_discovery_real_claims_disproof/`):
production shape (57 numeric covariates) → singular correlation matrix (rank 45/59),
fisherz refuses, 0 edges; rank-pruned to 43 covariates → one un-bootstrapped PC fit
230.3 s (production's 20 resamples ≈ 81 min, over the 900 s agent timeout); pruned to
13–14 covariates → AUGMENT at 0.73 in 181–289 s with 20 resamples; the REQUIRED
estimand edge `(T, Y)` was absent from the ensemble on the real frame although the
prior declared it. Runtime is driven by the number of CI tests, not by rows, so the
15,209-row frame is not the constraint.

1. **Discovery frame pre-flight** (`src/causal_engine/discovery/preflight.py`, called
   from `graph_builder._run_discovery` before tiers are built): drop constant columns;
   greedy rank-preserving prune of exactly linearly dependent columns (a column is kept
   iff it raises the correlation-matrix rank), which removes the Charlson/Elixhauser
   duplicates and composite scores; then cap the DAG-learning frame at
   `discovery_max_covariates` (default 20) by a pre-treatment screening rule that does
   not peek at the outcome-treatment relation: the union of the top-k covariates by
   absolute association with T and the top-k by absolute association with Y, ties broken
   by manifest order. Every dropped or capped covariate stays in the adjustment
   guarantee (`modeled_confounders`), so the estimate still conditions on it. The
   pre-flight's decisions are recorded in `DiscoveryResult.metadata`
   (`preflight: {constant, collinear, capped, kept}`) and surfaced in the API response.
2. **Bootstrap under a budget.** `discovery_time_budget_s` (default derived from the
   agent timeout minus refutation's budget) bounds the resample loop; the achieved
   resample count is reported and the gate's corroboration is computed over the
   achieved count. A run that achieves fewer than `min_resamples` (default 10) is
   reported as uncorroborated, never as corroborated.
3. **Required-edge honesty.** Establish in-lane why the prior's required `(T, Y)` edge
   was missing from the ensemble on the real frame (causal-learn skeleton phase vs the
   bootstrap ensemble threshold), fix it if it is ours, and in every case assert the
   estimand edge on the shipped DAG with provenance `required_prior`, as the AUGMENT path
   already does.
4. **Independence test, measured not assumed.** Most claims covariates are binary
   flags. The lane measures fisherz vs `gsq`/`chisq` on the capped real frame (gate
   decision, runtime, resample stability, recovery of the planted synthetic structure)
   and picks per frame type with the measurement recorded as evidence; #2009's fisherz
   choice was measured on 10-level synthetic data, not on binary claims flags.
5. **Acceptance.** On the real persistence frame (T = `treatment_dupixent`): discovery
   reaches a gate decision inside the agent timeout with at least 10 resamples, the
   shipped adjustment set contains every declared covariate, and the response names
   what was pruned and capped. On the synthetic planted frame: unchanged (ACCEPT,
   `disease_severity` recovered as confounder). With Lane B's approved prior as
   `anchored_confounders`, discovery either corroborates the prior (ACCEPT/AUGMENT)
   or the gate says why not; both outcomes are recorded in the Lane A cert. After
   acceptance the real dataset's API default flips to guided discovery on.

### Lane C — remibrutinib pre-wiring

1. Matcher: `CSU_BIOLOGIC_*` in `convert_optum_rwd.py` and the mart converter gain
   remibrutinib (generic `remibrutinib`, brand `RHAPSIDO`, NDC `00078-1100-30` from
   `src/ml/synthetic/clinical_codes.py`); the CSU real-drop converter maps the
   brand instead of collapsing to `competitor`.
2. Registry template: `_CAUSAL_DATASET_SPECS` entry `csu_escalation_causal`
   (treatment `treatment_remibrutinib` vs competitor, the same outcomes and covariate
   contract as Lane A). Until a real table exists it is backed by the synthetic CSU
   cohort table, whose rows carry `is_synthetic=true`, so real mode returns no rows
   and only the planted-truth run below uses it; the real table replaces the backing
   at the post-launch refresh with no registry change.
3. Planted-truth end-to-end: the synthetic CSU cohort (`data/rwd/synthetic_CSU`,
   Remibrutinib brand, planted confounders) run through the whole Lane A + B path
   as a CI-runnable test of the wiring (the estimate must recover the planted ATE
   within the generator's tolerance).

## 4. Data flow

```
Optum_enriched.parquet ──convert_optum_mart --cohort persistence_causal──▶ parquet (T kept)
        │                                                                     │
        │                                                   load_optum_causal_cohort ──▶ public.optum_biologic_persistence_causal
        │                                                                     │
optum_mart manifest ─┐
causal frame ────────┴─feature_role_panel (L1 contract · L2 KG · L3 probe · L4 LLM → ensemble)─┐
                                                                                          ▼
                     structural_author (guide + panel evidence) ──▶ attestations.json ──assembler──▶ dag.json ──▶ expert_reviews (human)
                                                                              │ approved
                                                       anchored_confounders ◀─┘
                                                                              ▼
                       POST /api/causal/agent {dataset: optum_biologic_persistence, treatment_dupixent → persistent_at_180d}
                                                                              ▼
                        graph_builder (curated DAG / priors; Lane D: pre-flight → guided discovery → gate) → estimation → refutation → expert-review consult → response + cert
```

## 5. Error handling and honesty

- Discovery stays off for the real dataset until Lane D is accepted; if a caller turns
  it on before then, the fixed runner (PR #2203) reports "could not run: singular…"
  rather than an empty DAG. After Lane D, a frame the pre-flight cannot make full-rank
  still fails loudly with the same message.
- No negative control is fabricated; SKIPPED is the honest verdict until measured.
- Author failures route features to review; unresolved citations downgrade edges;
  nothing machine-authored decides without an approved review.
- `is_synthetic=false` on every real row; the real-mode provenance filter applies.
- Prod writes (migration apply, table load) are owner-GO steps; rehearsed with
  `--dry-run` and, for the load, inside `BEGIN … ROLLBACK` first.

## 6. Testing

Red-first per lane; `pytest -n 0`; no mocks in production paths. Lane A: export shape,
loader, registry consistency, real-DB arm-split probe, API run cert. Lane B: parser and
grader on fixed model outputs, assembler on hand-built fragments (latent, M-structure),
scorer on the golden fixtures, CLI with a fake LM and the dead-Supabase pin, provenance
handling in the decider; the benchmark itself is a real-LM run recorded as evidence.
Lane E: image-packaging guard for the DSPy artifact, KG activation entries against
their committed caches (fail-loud on a missing cache), the panel on a fixture frame with
a fake LM and the committed KG cache (real Layer 1 and Layer 3), the veto constraints and
the leak-covariate removal in the agent, and the precision re-measurement as evidence.
Lane C: matcher unit tests, synthetic planted-truth end-to-end. Lane D: pre-flight on
hand-built collinear frames (exact duplicate, composite = sum of parts, constant),
screening rule determinism and manifest-order tie-break, budgeted bootstrap with a fake
slow algorithm, required-edge assertion, and the fisherz-vs-gsq measurement recorded as
evidence with the planted synthetic frame as its control.

## 7. Owner decisions (2026-09-22, approved)

- **GO** for migration 148 apply and the production table load (Lane A step 2).
- The 110 existing Optum attestations are relabelled `provenance="machine"` (Lane B).
- KG stays in shadow and Layer 4 stays audit-only until Lane E's measurement on the
  causal cohorts; promotion of either is decided on that measurement.
- Persistence definition, ascertained by the cheapest disproof
  (`docs/demos/results/2026-09-22_persistence_definition_disproof/`): the shipped
  `persistent_at_180d` is days-supply sensitive (14-day Dupixent fills vs 28–45-day
  Xolair fills; the −17.6 pp raw gap collapses to 3.5 pp with a 14-day grace and inverts
  from 28 d). Lane A's primary outcome is `persistent_at_180d_g28` (covered through day
  152 and no internal gap > 60 d), exported as a new column; `discontinued_180d` is the
  brand-robust secondary; the shipped persistence is reported only alongside the sweep.

## 8. Still open

- Whether the raw Dupixent fills are 14-day pens or 28-day packs recorded as 14 days
  (needs the claim-level feed); the grace definition is the honest one until then.
- Layer 4 as a decider (`ADAPTIVE_LAYER4_LLM_DECIDES`): off under this design; the
  LLM informs the author and the reviewer. Say so if you want it to decide.
