# Lane B real runs — cert (2026-09-23)

## Verdict

**PASS with one blocked deliverable.** The paid benchmark passes the spec gate (zero missed leaks); the real Lane E panel and both real authoring runs completed and their files are committed; the relabel proposal is written. **No `expert_reviews` row exists for run (a) or run (b): both `--review` inserts were refused by production Postgres** (`22P02 invalid input value for enum expert_review_type: "initial_dag"`), files kept, exit 3. The re-run of `--review` waits on migration 151 (PR #2246, owner-applied). Nothing was approved; nothing was relabelled.

Spend: **USD 12.05 + run (a) at list price** (`spend_actual.json`; litellm-reported USD 8.76 + run (a)), under the USD 25 guard. RUN_A_SPEND_LINE

| step | ran | result (cited) | spend list / litellm |
|---|---|---|---|
| 1 benchmark, 91 blind briefs, `openai/gpt-5.6-terra`, live resolver | yes, exit 0 | `benchmark/summary.md:10` **PASS: gate missed_leaks == 0** (0 over 42 golden-leak features); `:11` exact role 60/91 (0.659), leak-decision 91/91, conservative errors 0 | 3.36 / 1.84 |
| 2 real Lane E panel, 64 covariates, Layer 4 `anthropic/claude-sonnet-4-6` | yes, exit 0 | `panel_real/summary.md:14` fired 19, roles confounder 18 / descendant 1; `:15` abstain 19 (rate 0.406), leak verdicts 0 | 5.06 / 5.06 |
| 3 run (a) `optum_mart` T=`treatment_dupixent` Y=`persistent_at_180d_g28`, real panel, `--review` | RUN_A_RAN | RUN_A_RESULT | RUN_A_COST |
| 4 run (b) `optum` T=`biologic_initiation` Y=`initiated_biologic_180d`, `--diff-manifest-attestations --review` | yes, **exit 3 at the review step**, files kept | `author_real/optum_biologic_initiation_initiated_biologic_180d/manifest_diff.json:3-6` compared 110, edge-exact 93, role-agree 93; `run_b_log.txt` `22P02 … "initial_dag"` | 3.60 / 1.85 |
| relabel PROPOSAL | yes | `relabel_proposal.md`: 93 → `machine_reviewed`, 17 unchanged; manifest NOT edited | — |
| smoke (1 feature, run (b) shape, no `--review`) | yes, exit 0 | `smoke_1_feature/token_usage.json`: 5,583 prompt + 912 completion tokens per brief — the cheapest disproof that the real author parses before the 110-brief spend | 0.03 / 0.01 |

## What ran, and how (tree `8615c0cadce9…`, `dirty_src_scripts_tests: false` on every `meta.tree`)

Worktree `.worktrees/lane-b-real-runs` off `origin/main` `f1bb9e36c`; real `.env`; the parquet read from the main checkout by absolute path (gitignored). Every paid step printed its estimate first (`budget_guard.json`: cumulative estimate USD 11.44, upper bound 20.92 < 25). Actual token usage is captured per step from dspy's call history (`*/token_usage.json`, the CLIs report none) and priced at list in `spend_actual.json` (gpt-5.6-terra 2.50/15.00 from `src/services/llm_pricing.py:30`; Sonnet 4.6 3.00/15.00 from the panel CLI). The CLIs' own `--i-accept-cost` gates were honoured; no other prod write was attempted.

## 1. Benchmark (spec §3 Lane B item 3) — gate PASS

`benchmark/score.json`: `gate_passed: true`, `missed_leaks: []`, `n_leak_truth 42`, `n_leak_scored 42`, `n_review 0`.

| | fake replay baseline (`…scaffold/measure_fake_all91/summary.md:10-11`) | **real author** (`benchmark/summary.md:10-19`) |
|---|---|---|
| missed leaks (gate) | 28 of 42 → FAIL (stand-in fragments) | **0 of 42 → PASS** |
| exact role agreement | 39/91 (0.429) | **60/91 (0.659)** |
| leak-decision agreement | 63/91 | **91/91** |
| instrument precision / recall | 1.00 / 0.32 | **1.00 / 0.74** |
| per cohort exact (BC / CSU / PNH) | 5 / 28 (replayed) / 6 | 17 / 23 / 20 |

Honest reads: (i) the real author scores **23/31 on CSU, below the committed CSU record's 28/31** (`measure_fake_csu` replays the committed blind edges; the real author re-authored them); (ii) weakest roles are descendant (recall 0.38: 7 of 13 called collider) and mediator (0.40: 6 of 15 called descendant) — all of these are leak-side confusions (collider/descendant/mediator ↔ each other), which is why the leak gate still passes; (iii) 5 ancestors and 4 instruments were called confounder (over-adjustment, conservative). Cited: `score.json` `confusion`. This is a literature-fixture benchmark (ConcertAI CSU/PNH/BC label sets); it says nothing about the Optum cohorts.

Cost reality: prompt tokens 515,868 vs the estimate's 531,375 (4 chars/token was slightly conservative); completion 138,241 vs 81,900 assumed (the author writes ~1,519 tokens/brief, 1.7× the 900 assumed) — `benchmark/token_usage.json`.

## 2. Real Lane E panel for run (a) (spec Lane E item 4)

`panel_real/summary.md:11-15` on the real 15,209-row persistence frame, 64 covariates, activation profile `{layer4 on, structural decider on, KG shadow}`:

- Layer 1: consulted 64, declared-safe 64, post-index 0. Layer 2 (shadow): signalled 2 (`leak_drug_treats_disease` on `cci_chronic_pulmonary`, `elx_chronic_pulmonary`). Layer 3: scored 57, high 9 / moderate 10 / info 38, FDR-confident 0, declared-safe immunity applied 9.
- Layer 4 (real Sonnet 4.6): **fired 19 — exactly the fake run's count** (`panel_fake/summary.md:14`, re-measured on this tree before paying); roles confounder 18, **descendant 1 = `charlson_score`**, whose mechanism says the comorbidity mart declares no lookback `window_days`, so post-index diagnoses could contaminate the score (a data-contract gap, not a clinical claim; `panel_real/panel.json` `records.charlson_score.layer_4.mechanism`). 2 of 19 calls cited PMIDs; the evaluator did not run (its fields are null).
- Ensemble: **unchanged from the fake run** — decided_by adversarial 36 / abstain 19 / kg 2 / none 7, abstain rate 0.406, leak verdicts 0; `promotion_eligibility.passes false` (non-abstain 0.667). The only per-feature differences are the audit-only evidence strings on three comorbidity features (Layer 4 is audit-only under the profile, spec Lane E item 1). A null is a finding: Layer 4 informs the author's brief (run (a)); it decides nothing.
- Cost reality: the real prompts were **85.5k tokens/call, not the probed 68k** (`layer4_prompt_size_probe.txt` estimated at 4 chars/token); USD 5.06 vs the 4.0 estimate.

## 3. Run (a) — `optum_mart`, dupilumab vs omalizumab → persistence at 180 d (28-day grace)

RUN_A_SECTION

## 4. Run (b) — `optum`, `biologic_initiation` → `initiated_biologic_180d`, diff vs the 110 machine attestations

`author_real/optum_biologic_initiation_initiated_biologic_180d/`: 110 authored (`dag.json` `is_dag true`, `adjustment_valid true`, admissible set 108 — the 2 authored descendants excluded), `review.md` (`## Review items (73)`), `manifest_diff.json`.

- **93/110 agree on the exact edge set AND role.** The 17 disagreements are **exactly the manifest's 17 instruments** — the same set the fake run disagreed on (`…scaffold/disproofs.txt:10`), but now with the real author's reasoning: 15 instrument → confounder (the author adds `feature -> Y`: geography `zip5/zip3/zip_code/geographic_region/urban_rural_code`, payer `insurance_product/plan_type/payer_category`, specialist access `office_visits_allergist/_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`), 2 instrument → descendant (`index_date`, `lookback_start_date`: the author reverses `date -> T` to `T -> date`, "the initiation event operationally defines the cohort index date"). Both rationales per row are in `relabel_proposal.md` (the manifest side cites `docs/layer4/optum_initiation_attestation_research.md` L59/L62 with PMIDs).
- **73/110 flagged `ambiguous` by the author itself**, with the stated reason (e.g. `age_group`): T=`biologic_initiation` and Y=`initiated_biologic_180d` are *overlapping initiation constructs*, so the operational meaning of Y matters — run (b) was briefed with the bare column names (the diff exercise has no labels). `review_required 0`; `expected_role == derived_role` on all 110. This is a finding about the run (b) estimand framing, not about the manifest.
- **All 218 non-estimand edges graded `unsupported`** by the citation grader, including edges whose PMID abstract resolved (`abstract_resolved true`, `entities_found []`, `overall_confidence 0.0` — e.g. `age_at_index -> T` citing PMID 24472253 in `smoke_1_feature/…/attestations.json`). The grader's entity match found nothing in any abstract; every record therefore carries `review_reasons: unsupported edges`. Worth the owner's eye: either the grader is stricter than the guide intends, or the author's citations are generic — the cert does not decide which.
- `--review`: refused, see Verdict. `review.json` absent by design (the CLI writes it only after a row exists).

## 5. Relabel PROPOSAL (owner item 4) — `relabel_proposal.md`

Rule: `machine_reviewed` only where the real author reproduces the manifest's edge set exactly AND derives the same role (two independent machine readings coincide); otherwise unchanged (`machine`). Result: **93 → `machine_reviewed`, 17 unchanged**, one table row per attested feature with both rationales, disagreements first, a machine-readable map at the end. **`src/data/manifests/optum_feature_manifest.py` is untouched**; the relabel is applied only after the owner reads the diff. `machine_reviewed` is not human sign-off and does not make an attestation decide (spec §7: only `human` / an approved review does).

## Blocked: the review rows (both runs)

`run_b_log.txt` / `run_a_log.txt`: `ERROR src.repositories.expert_review: Failed to create expert review: {'message': 'invalid input value for enum expert_review_type: "initial_dag"', 'code': '22P02'}` → `expert review NOT opened … (files kept)`, exit 3. Prod enum (read-only probe): `{dag_approval, methodology_review, quarterly_audit, ad_hoc_validation}` (`database/ml/010_causal_validation_tables.sql:53`); 41 `dag_approval` rows exist. The scaffold writes `initial_dag` and its item-5 loader fail-closes on `review_type == "initial_dag"` (`src/data/kg/structural_prior_loader.py:64`) — tested against an in-memory repository, never the real enum. Fix in flight: migration 151 (PR #2246, `ALTER TYPE expert_review_type ADD VALUE IF NOT EXISTS 'initial_dag'`), owner-applied; #2244 tracks the `uq_er_pending_estimand` index ignoring `review_type`.

Re-run commands after 151 lands (paid; the authored files are regenerated — the review row is minted from the in-memory DAG at the end of the run, there is no open-from-files mode):

    python -m scripts.author_cohort_dag --manifest optum_mart --treatment treatment_dupixent \
        --outcome persistent_at_180d_g28 \
        --treatment-label "dupilumab (Dupixent) vs omalizumab (Xolair) as the index CSU biologic" \
        --outcome-label "persistence at 180 days (28-day grace)" \
        --panel docs/demos/results/2026-09-23_lane_b_real_runs/panel_real/panel.json \
        --lm real --i-accept-cost --resolver live --review \
        --out-root docs/demos/results/<date>_lane_b_reviews          # ≈ USD RUN_A_RERUN_EST at list (this run's measured tokens)

    python -m scripts.author_cohort_dag --manifest optum --treatment biologic_initiation \
        --outcome initiated_biologic_180d --lm real --i-accept-cost --resolver live \
        --allow-no-panel --no-assumption --diff-manifest-attestations --review \
        --out-root docs/demos/results/<date>_lane_b_reviews          # ≈ USD 3.60 at list (this run's measured tokens)

## Not done (deliberately)

- No review approved; no attestation relabelled; `feature_manifest_source` not declared on the dataset spec (PR #2230 owner decision 4).
- No code change: the enum mismatch, the grader strictness and the panel-probe token undercount are reported, not fixed.
- The benchmark is not re-run for variance: n=1 real run (the gate is a floor, not a distribution).
