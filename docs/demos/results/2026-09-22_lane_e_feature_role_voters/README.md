# Lane E — the four voters, live and wired into DAG construction (2026-09-22)

Spec: `docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md`
§3 "Lane E" (on branch `claude/real-data-causal-estimation`). Every number below
cites the captured file:line in this directory.

## Cheapest disproofs run BEFORE building (and what they changed)

| Assumption the spec item rests on | Experiment | Result | Consequence |
|---|---|---|---|
| Item 1: "build a KG cache for `optum_mart`" produces Layer-2 signal on the causal cohort | count entity-bearing features per manifest (`src.data.manifests`; builder emits a record only for `kg_entity_codes` features, `scripts/build_kg_cache.py::build_cache_for_manifest`) | `optum_mart total 80 with_entity_codes 0`, `csu 28 → 1`, `optum 122 → 74` (probe output in the lane transcript; pinned by `tests/unit/test_data/test_optum_mart_kg_entity_codes.py`) | the cache would have been EMPTY → authored `MART_COMORBIDITY_KG_CODES` for the 48 Charlson/Elixhauser flags |
| The authored (ICD10CM, UMLS) codes name the intended concepts | `verify_mart_kg_codes.py` (live UTS `cui_lookup` + ICD10CM crosswalk, the builder's own resolution path) | `umls_code_verification.txt:102` `unresolved_or_raised 1`; `:89` C0042990 → UTS 404; `:53` C0522224 → `'Paralysed'` semtype Finding; `:42` I38 → `'Valvular regurgitation'` | those three dropped before commit; the other 95 (system, code) rows resolved to the intended disease |
| A two-code `--target-entity-codes` gives both drugs a drug-disease pass | read `scripts/build_kg_cache.py::_resolve_target_drug` (pre-lane: `return chembl_id, code, errors` inside the loop → first resolved code only) | disproved by source read; pinned red-first by `tests/unit/test_scripts/test_build_kg_cache_multi_target.py` (3 failed on the old builder) | `_resolve_target_drugs` + per-drug pass |
| Dupilumab's RxCUI | `RxNavClient.rxcui_for_name("dupilumab")` | `rxnav_dupilumab_resolution.txt:2` → `RxCUIMatch(rxcui='1876376', approximate=False)`, `:3` TTY=IN; omalizumab `:4-5` → 302379 | `DUPILUMAB_RXCUI = "1876376"` pinned (unit + live test) |
| A real Layer-4 run is ≤ US$5 (the lane brief's threshold) | `DummyLM` capture of one `classify_feature` call with the packaged artifact, keys blanked | `layer4_prompt_size_probe.txt:3` `prompt_chars 273047 approx_tokens_at_4chars 68261`; `:4` `n_demos_on_predictor [192]` | ≈US$0.21/call at Sonnet list prices → 91-entry precision re-run ≈ US$19, a 64-covariate panel ≤ US$13.5 → both are OWNER decisions, not run |

## What was built and measured

### KG caches for the causal cohorts (item 1)

Built live (`kg_cache_build_optum_mart.log`, `kg_cache_build_csu.log`) with
`RXNORM:302379,RXNORM:1876376`; classified with the voter's own
`classify_kg_signal` (`kg_cache_signals.txt`):

- `optum_mart` (`data/kg_cache/0b4c5fdb__214e5b23.json`): 48 records, `:3`
  `status: {'queried_no_edges': 24, 'ok': 24}`; `:14` `signalled=2/48` —
  `cci_chronic_pulmonary` / `elx_chronic_pulmonary` carry `treats` edges from
  BOTH drugs for asthma (`:4-8`, `:9-13`: `drug=CHEMBL1201589 (OMALIZUMAB)` and
  `drug=CHEMBL2108675 (DUPILUMAB)`, `disease='asthma' MONDO_0004979`) and
  `associated_with` for COPD. The other 46 flags are honestly `no_signal`.
- `csu` (`data/kg_cache/2e1be83e__214e5b23.json`): 1 record, `:20` `signalled=1/1`;
  `:18` omalizumab `treats` urticaria, `:19` dupilumab only `associated_with`
  (not approved for CSU).

Both activated in **shadow** (`src/data/kg/activation.py::KG_ACTIVATIONS`);
promotion is the owner's call on this measurement (spec §7) and was not made.

### Feature-role panel (item 2) and the causal agent seam (item 3(d))

`src/causal_engine/feature_role_panel.py` — see the module docstring for the
three causal-use design points (only Layer-1 post-index and Layer-3 high without
declared-safe immunity are leak verdicts; Layer 2 informs; the activation profile
is per run). Return shape documented in the PR's `interfaces` field.

### Real-frame measurement (item 4)

`panel_layer4_fake/` — `scripts/measure_feature_role_panel.py --layer4 fake` on the
real persistence frame (n = 15,209, the 64 `MART_SAFE_FEATURES`, T =
`treatment_dupixent`, Y = `persistent_at_180d_g28`; provider keys blanked, `DummyLM`
answering every call; run under the box lock on the FINAL code — commit `bc0bc4ed3` —
log `run_log_panel_layer4_fake3.txt`; the identical-numbers run on the pre-round-4 code is
`run_log_panel_layer4_fake2.txt`, `maxrss_kb=833240 elapsed_s=237.21`). Numbers from
`panel_layer4_fake/summary.md`:

| Layer | What fired (`summary.md:11-15`) |
|---|---|
| Layer 1 contracts | consulted 64, contracted 64, declared-safe 64, post-index **0** |
| Layer 2 KG (shadow) | cache bound, 24 features with cached edges, **2 signalled** (`cci_chronic_pulmonary`, `elx_chronic_pulmonary`: `leak_drug_treats_disease` = indication evidence in the causal contrast; 62 `no_signal`) |
| Layer 3 adversarial | scored **57** (the 7 non-numeric columns — categoricals and risk bands — are not scored by the node and have no verdict), pre-joint severities high 9 / moderate 10 / info 38, FDR active at 569 permutations with **0** confident features (n = 15,209 makes every association significant, but no `|ΔAUC|` clears the 0.10 floor), declared-safe immunity applied 0 |
| Layer 4 LLM (fake) | classifier loaded, **fired on 19 features** (`summary.md:88`: the 10 moderate + 9 high-and-declared-safe) — this is the paid run's exact call count |
| Ensemble | decided_by adversarial 36 / abstain 19 / kg 2 / none 7; abstain rate **0.406** (19 LLM-informed features abstain because Layer 4 is audit-only and the σ-band is joint-clamped — the honest "route to a human"); **leak verdicts 0** (`summary.md:90,92`: proven post-index 0, pending temporal review 0) |

A null is a finding: on this cohort no covariate is excluded — all 64 are contracted
pre-index, the prediction-era Layer-3 rule is inert by construction, and the only
non-`no_signal` KG voice is indication evidence. The first fake run (before the
`DummyLM` exhaustion fix) under-reported Layer 4 as "fired 1"; `run_log_panel_layer4_fake.txt`
shows the 18 `Layer 4 skipped` lines that exposed it.

`promotion_eligibility` (`summary.md:17`, from `panel.json`): **`passes: false`** —
n = 15,209 ≥ 200 and kg_decided 2 and disagreement 0.0 pass, but `non_abstain_pct`
is 0.667 over the 57 scored features against the ≥ 0.95 gate: the 19 LLM-informed
features abstain under the audit-only profile. (The first, under-reporting fake run
had read 0.98 / `passes: true` because Layer 4 fired only once there — codex r4
caught the stale number.) Recorded for the owner; nothing was promoted.

The real Layer-4 run (19 calls) was queued once under the lock at the measured
≈ US$4 and was refused by the session's permission system as a real-world
transaction; it stays an owner decision (below).

## Owner decisions (not done here)

1. **Paid Layer-4 precision re-run** (spec item 4, "≥0.95 instrument-precision
   gate"), exact command:
   `python scripts/measure_layer4_precision.py --enable-evaluator --evaluator-gate both --report-path docs/demos/results/2026-09-22_lane_e_feature_role_voters/layer4_precision_report.json`
   — 91 golden entries × ~68k prompt tokens (`layer4_prompt_size_probe.txt:3`)
   ≈ US$19 + the Haiku evaluator. Above the US$5 gate → not run.
2. **Paid real-frame panel run**, exact command:
   `python scripts/measure_feature_role_panel.py --parquet data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet --manifest-source optum_mart --treatment treatment_dupixent --outcome persistent_at_180d_g28 --covariates mart-safe --out docs/demos/results/2026-09-22_lane_e_feature_role_voters/panel_layer4_real --layer4 real --i-accept-cost`
   — measured 19 calls (`panel_layer4_fake/summary.md:88`) × ~68k tokens ≈ US$4.0 at
   Sonnet list prices. Queued once by the lane (within the ≤ US$5 gate) and refused by
   the session's permission system as a real-world transaction — the owner runs it.
3. **KG promotion from shadow** for `optum_mart` / `csu` on the measurement above
   (`compute_promotion_eligibility` is recorded in `panel.json`).
4. **Spec deviation to confirm**: approved *instruments* are routed to the
   state's `instruments` channel (not adjusted for, not anchored — graph_builder
   forces `conf -> outcome` for anchored confounders; an instrument must not
   have that edge); estimation does not consume that channel yet.
5. **Layer-3-only exclusions** (an uncontracted column that confidently predicts
   Y): excluded per spec 3(b) and flagged `review_required` with `temporal_status=
   unknown`, or keep-and-flag? Cannot fire on this cohort (all 64 contracted).
6. **Panel provenance**: the public API accepts a caller-supplied panel whose
   identity is checked at submit (typed parse, strict invariants, question,
   coverage, manifest when declared) — consistency, not authenticity. A
   server-owned panel artifact bound to a data fingerprint belongs with Lane B's
   approved-review loader; the public `approved_structure_roles` field was
   removed for the same reason (codex r2/r3).
