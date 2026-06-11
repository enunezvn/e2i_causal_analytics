# Plan: Synthetic Claim-Level Generator for CSU (`synthetic-rwd-claims`)

**Date:** 2026-06-08 · **Status:** DRAFT rev2 (converged via ralph-loop + codex-rescue; rev2 addresses codex round-1 findings, all independently verified against source)
**Source assessment:** `docs/reports/synthetic-data-csu-assessment-20260608.md`

---

## Context

The real Optum drop is a pre-aggregated mart with **no claim-level rows**, which blocks the longitudinal patient-feature and HCP forward-causal use cases (`docs/reports/optum-mart-data-treatment-findings-20260608.md`). Our synthetic stack has the **same** limitation. But `scripts/convert_optum_rwd.py` *already* turns a **6-file raw claim layout** into pre-index windowed features + an HCP influence graph. So the highest-leverage enhancement is a generator that emits those 6 files **with embedded ground truth**, letting us build and validate the full pipeline end-to-end *now* — with a known true effect the real data can never give us — and swap onto the real re-extract unchanged when it lands.

## Goal & non-goals

**Goal:** a `synthetic-rwd-claims` generator that emits the 6 raw parquet files `convert_optum_rwd.py` consumes, encoding a **known DGP**, such that the existing converter → tier-0 pipeline **recovers** the embedded effects within stated tolerances.

**Non-goals:** replacing v1/v2/`rwd_realistic`; non-CSU indications; PHI; modeling the upstream Spark mart-build.

---

## CRITICAL mechanism facts (verified against `convert_optum_rwd.py`) — read before the DGP

These four facts, verified by codex review + direct source check, constrain the DGP. **The DGP must work *through* these, not around them.**

1. **The converter self-derives a CLAIM-ANCHORED index and IGNORES vendor `indexdt`.** `_derive_index_date:1837` ("never using vendor indexdt"). Cohort **A (initiation)** index = ≥2 distinct CSU-dx (L50.x) claim dates from inpatient `diag1..5` (`:1840`). Cohorts **B (discontinuation) / C (persistence)** are anchored at the **first CSU-biologic fill** (`init_date`). ⇒ **DGP timing must be encoded in claim event dates, never in a demographics `indexdt` field.**
2. **Features are STRICTLY PRE-INDEX.** `_compute_features:2260-2261` windows `(index − 180, index − 1]` (`LOOKBACK_DAYS=180 :68`). For B/C, since index = first biologic fill, this window is the **dx→treatment-start interval** — richer than the mart's dx-stale features (the genuine longitudinal win).
3. **The disc/persistence fill-gap trajectory is the TARGET, not a feature.** `_target_discontinued_180d:2696` / `_target_persistent_at_180d:2725` read post-index biologic fills in `(init_date, init_date+180]` to produce the **label** only. Pre-index biologic fills are **excluded** from features (`bio_mask:2415-2416`). ⇒ There is **no post-index fill-pattern feature** in `patient_journeys`. A DGP can only make the outcome **recoverable from pre-index features** unless the converter is extended (P1c, below).
4. **HCP graph = shared-patient CO-TREATMENT cliques, not referral.** `build HCP influence graph :1145-1155` collects treating HCPs from **`med.npi ∪ proc.npi`** within the per-patient lookback; edge weight = patients seen by both endpoints. Outputs `peer_influence_score` (eigenvector centrality) + `influence_network_size` (degree). ⇒ The HCP DGP must encode signal in **shared-patient topology**, and must avoid circularity (do not derive the network *from* the adoption it is meant to predict).

## Output contract — the 6 raw files (verified against `convert_optum_rwd.py` reads)

Emit `<out>/{demographics,medication,procedure,lab,inpatientdata,provider}.parquet`. **Required** = converter actually reads it; *optional* = pass-through the converter ignores (include only if a named downstream tool needs it).

| File | Grain | Required (converter reads) | Notes / removed phantom cols |
|---|---|---|---|
| `demographics` | 1/patient | **patid, eligeff, eligend, diagcode, age, gdr_cd, zipcode_5, bus, product, health_exch, lis_dual, continuous_enrollment** | `bus`→`insurance_product`+`payer_category` (`:2276,2290`). **Drop `indexdt`** (self-derived, ignored). **Drop `yrdob`, `family_id`** (zero reads); emit `age` directly. |
| `medication` | 1/fill | **patid, medication_date, npi, code (NDC), days_sup, strength, Brand_Name, Generic_Name** | `strength` read `:3321`. **Drop `ahfsclss`, `quantity`, `clmid`** (zero reads). |
| `procedure` | 1/proc | **patid, proc_date, proc_code, npi** | `npi` feeds the HCP graph via `med.npi ∪ proc.npi` (`:1145-1155`). `clmid`/`sourcetable` optional. |
| `lab` | 1/result | **patid, fst_dt, loinc_cd, rslt_nbr, abnl_cd** | `hi_nrml`/`low_nrml` optional; `tst_desc` optional for features but needed for `lab_values` in `treatment_events` (`:3352`). |
| `inpatientdata` | 1/admit | **patid, admit_date, disch_date, diag1..5, tos_cd** | `tos_cd` read `:2390`. **Drop `proc1..5`** (zero reads). `diag1..5` carry CSU L50.x (drive cohort-A index + comorbidities). |
| `provider` | 1/npi | **npi, taxonomy1** | `taxonomy2`/`specialty`/`prov_state` optional (only `npi→taxonomy1` read, `:1550-1553`). |

## DGP — what ground truth to embed (works *through* the mechanism facts)

1. **Latent patient state** → `severity`, `response_propensity`, `adherence_propensity` (correlated block, calibrated to `scenario_c`/`csu_rwd` marginals).
2. **Enrollment gate (hard):** set `eligeff ≤ claim_index − 360d` and `eligend ≥ claim_index + 180d` (production regime `pre_days=360`, `:117`, gate `:2015`). Patients violating this are **dropped by the converter** — use this deliberately to produce the ~50% panel-fragmentation attrition.
3. **Pre-index claims** in `(claim_index − 360, claim_index]`: comorbidity dx in inpatient `diag1..5` (→ `cci_*`/`elx_*`), prior-therapy `medication` fills (non-biologic classes → the `_fill_count`/`_days_supply_total`/`_days_since_last_fill` features), labs **only where claims-plausible** (see Provenance). For B/C the window is pre-first-biologic-fill, so encode the **dx→treatment-start utilization trajectory** here — this is the recoverable longitudinal signal.
4. **Cohort-A initiation timing:** encode via **inpatient L50.x `admit_date` dates** (≥2 distinct → claim index) + a first-biologic `medication_date` within 180d for positives. Calibrate signal so post-conversion tier-0 `val_AUC` sits in the honest band (see Calibration).
5. **Disc/persistence outcome (B/C):** encode via the **post-index biologic fill sequence** (`medication_date` + `days_sup` gaps) — this *defines the label* via `_target_*`. To make it **predictable**, make pre-index features (item 3) statistically associated with the adherence/gap outcome (e.g., low `adherence_propensity` ⇒ both sparser pre-index utilization AND a post-index gap). Recovery is "pre-index features predict the post-index target," NOT "a fill-gap feature."
6. **HCP shared-patient diffusion:** generate HCPs + a **shared-patient graph** (degree skew, communities); assign treating `npi` on `medication`/`procedure` so that **co-treatment centrality causes earlier adoption**. Encode the network exogenously (centrality drawn first, adoption timing a function of it) to avoid circularity. The converter's co-treatment graph then carries the signal directly.
7. **TRUE_ATE handle:** one designated treatment→outcome edge with a known ATE for causal-impact validation (parity with v1/v2).

## Calibration (a stated design choice, not just a cite)

The converter emits ~40 features (utilization, drug-class, lab, comorbidity, provider-mix), so its information ceiling is **higher** than the 6-feature patient-grain `rwd_realistic` band. **Decision:** deliberately inject weak-enough signal to hold post-conversion tier-0 `val_AUC ∈ [0.62, 0.68]` (`synthetic_rwd_realistic.py:276`) — matching the published claims-only ceiling — tuned via a `signal_scale` knob (mirroring `RwdRealisticConfig`). State the realized band per cohort; if a cohort's irreducible signal exceeds it, document the higher honest band rather than forcing it down. Inherit `rwd_realistic` prevalence `0.024` and `panel_fragmentation_rate 0.50` as targets.

## Reuse (do not rebuild)

- **`scripts/convert_optum_rwd.py`** — the transform + validation harness.
- **`src/repositories/synthetic_rwd_realistic.py`** — calibration anchors (`RwdRealisticConfig`).
- **`src/ml/synthetic_v2/scenarios/scenario_c.py`** — CSU clinical distributions.
- **`src/ml/synthetic_v2/rwd_loaders/csu_rwd.py`** — `RWD_PROVENANCE_TAGS` map ONLY (the `_load_from_excel`/`_load_from_json_outputs` loaders raise `NotImplementedError`, `:172-194` — see P2 caveat).
- **`src/ml/synthetic/validation/`** + **`validators/causal_validator.py`** — validation to extend into a gate.

## Provenance honesty (operationalized)

`csu_rwd.py` `RWD_PROVENANCE_TAGS` keys are **scenario-C feature names** (e.g. `total_serum_ige_iu_ml`), not converter columns. So add a **translation step**: for each `RWD-missing` tag, identify the claim type that would spuriously synthesize it (e.g. `total_serum_ige_iu_ml` → suppress that LOINC `lab` row; PRO scores → emit no claim at all since claims carry no PROs) and list the suppressed claim types in the DGP spec. Net rule: **do not emit lab/PRO claims for features real CSU claims cannot carry.**

## Phased plan

**P1 — Claim-level patient generator + round-trip (the unblock).** New `src/ml/synthetic/claims/` + CLI `scripts/generate_synthetic_claims.py`; DGP items 1–5; calibrate to `rwd_realistic`.
- **Acceptance:** 6 files load through `convert_optum_rwd.py` with zero schema errors; tier-0 `val_AUC` in the honest band; **the pre-index (dx→treatment-start) windowed features predict the disc/persistence target above a comorbidity-only baseline** by a pre-registered margin. (Reworded per mechanism fact #3.)

**P1b — HCP shared-patient network + forward-causal.** DGP item 6.
- **Acceptance:** converter's co-treatment graph is non-trivial (degree skew, centrality variance); an HCP forward-causal model recovers the embedded centrality→adoption effect (sign + rough magnitude) under a pre-index window, with no circularity.

**P1c (conditional) — post-index fill-pattern features (only if needed).** If we want the model to *use* the adherence trajectory (not just predict its label from pre-index features), this requires a **scoped converter extension**: add post-index biologic features (`biologic_n_fills_180d`, `biologic_pdc_180d`, `biologic_max_gap_days`) for the B/C cohorts to the feature manifest. This is a converter change, explicitly scoped here — NOT assumed by P1.

**P2 — Fidelity & causal gates (deficiencies #2,#3).** Promote KS distributional checks into `validation/pipeline.py`. **Caveat:** the real-data KS gate needs `csu_rwd._load_from_excel` (currently `NotImplementedError`) — so either implement that loader, or compare against **hardcoded marginals derived from `csu_data.xlsx` headers**. Add correlation-recovery check. Make `causal_validator` a generation-time gate; attach explicit DAG + causal-role tags + literature-anchored TRUE_ATE/CATE.

**P3 — Infra (deficiency #4).** Add scipy/statsmodels to `requirements-synthetic.txt`; remove the silent numpy-lstsq fallback (fail-fast). Add temporal-leakage + covariate-balance checks to the split validator.

## Validation strategy (the convergence target)

1. **Schema round-trip:** 6 files → `convert_optum_rwd.py` → 3 canonical parquets, no errors.
2. **Calibration:** realized prevalence within 1.5pp of 0.024; tier-0 `val_AUC` in the stated honest band; ~50% panel-fragmentation attrition reproduced via the enrollment gate.
3. **Longitudinal recovery:** pre-index (dx→treatment-start) windowed features lift disc/persistence AUC above a comorbidity-only baseline by a pre-registered margin. (If P1c is built, additionally verify the post-index fill-pattern features carry the signal.)
4. **Network recovery:** HCP forward-causal model recovers the embedded centrality→adoption effect via the co-treatment graph.
5. **ATE recovery:** causal-impact recovers TRUE_ATE within tolerance (reuse `causal_validator`).
6. **Distributional fidelity:** KS vs real marginals (via the P2 route above) passes for RWD-direct features; RWD-missing features stay absent (no false fidelity).

## Risks / open questions (for user)

- **Claim timing faithfulness:** minimal viable = enrollment window + dx/fill dates + gap trajectory; defer seasonality/dose-escalation/censoring to a later phase?
- **Leakage patterns:** inherit `rwd_realistic`'s 7 variants now, or P2?
- **P1c scope:** do we want the model to *use* the adherence trajectory (converter extension), or is predicting the disc/persistence label from pre-index features sufficient for now?
- **Scale/compute:** claim rows ≈ 10–50× patients; cap default n and stream to parquet (droplet memory pressure — full-width reads OOM).
- **Placement/naming:** `src/ml/synthetic/claims/` vs a new top-level module.

## Verification

- `scripts/generate_synthetic_claims.py --n <small>` → `scripts/convert_optum_rwd.py --data-root <out>` → `scripts/run_tier0_test.py`; confirm gates 1–6.
- New tests under `tests/ml/synthetic/claims/` mirroring existing synthetic test conventions; CI via `synthetic-benchmarks.yml`.
- P1/P1b are additive (new generator + CLI). P1c and P2 touch the converter/validation — gated, reviewed separately.
