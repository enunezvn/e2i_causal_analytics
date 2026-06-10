# Synthetic Causal Data — How It Works and How to Validate With It

**Audience:** anyone running e2e validation of the causal pipeline (cohorts, estimators,
agents) against data with *known* ground-truth effects.
**Current dataset:** `data/rwd/synthetic/` (generated 2026-06-10, parquet-only, seed 42,
`confounded` DGP, FULL_SIZES; see its `README.md` for provenance).
**Generator stack:** `src/ml/synthetic/generators/` → `scripts/load_synthetic_data.py`
(plan: `.claude/plans/synthetic-causal-validation/`, merged via PR #850; cohort-frame
fan-out fix in PR #860).

---

## 1. What this is

A synthetic dataset that encodes **known ground-truth treatment effects (ATE/CATE) per
(cohort × brand)** so the full causal-analytics pipeline — KPIs, propensity/uplift
estimation, and the agents — can be validated to *recover* those effects. Every row
carries `is_synthetic=True` and is **excluded by default** from all real analyses.

It exists in two consumption modes:

| Mode | Produced by | Consumed by | Use for |
|---|---|---|---|
| **Parquet snapshots** (`data/rwd/synthetic/`) | `load_synthetic_data.py --parquet-only --parquet-out <dir>` | offline cohort/estimator checks; gate 10's hcp_adoption cells | estimator validation without touching the DB |
| **DB substrate** (docker-Supabase tables) | `load_synthetic_data.py --anchor-to-now` (no `--parquet-only`) | the runtime: KPIs, dispatcher, agents, the 11-gate harness | full e2e pipeline validation |

> The runtime in `src/` **never reads the patient-cell cohort-frame parquets** — its
> twin of those frames is the `patient_journeys` table, resolved through
> `cohort_resolution._PJ_COHORTS` (`src/services/cohort_resolution.py:259`). The parquet
> files are the offline/pollution-free equivalent of what the runtime reads from the DB.

**Distinct track:** the claim-level CSU/Optum converter (`scripts/convert_optum_rwd.py`,
`data/rwd/csu/`, `run_csu_tier0_test.py`) feeds tier-0 ML AUC only and writes no DB rows.
This guide covers the DB-substrate/causal-validation track.

---

## 2. How the data is generated (the DGP)

### 2.1 Patient grain — confounding, treatment arm, per-unit tau

Per patient (`patient_generator.py`, `src/ml/synthetic/dgp/treatment_arm.py`):

1. **Confounders** — `disease_severity ~ Normal(5,2)` clipped to [0,10];
   `academic_hcp ~ Bernoulli(0.30)` (`patient_generator.py:262-287`).
2. **Propensity & arm** — `e(X) = sigmoid(-2.0 + 0.30·(severity-5) + 0.80·academic)`,
   clipped to [0.01, 0.99] (overlap guaranteed); `treatment_arm ~ Bernoulli(e(X))`
   (`treatment_arm.py:27-47`). Both confounders also enter the outcome, so the arm is
   genuinely confounded and the naive contrast is biased (measured: naive 0.269 vs
   truth 0.172 on the current Remibrutinib initiation cell). The realized propensity is
   **stored** per row as `propensity_score`.
3. **Segments & heterogeneous tau** — severity >7 → `high_severity`, >4 → `medium`,
   else `low`. Latent CATE base map `{high: 0.50, medium: 0.30, low: 0.15}` × brand
   scale `_BRAND_CATE_SCALE = {Remibrutinib: 1.00, Kisqali: 1.40, Fabhalta: 0.70}`
   (`treatment_arm.py:53-57`).
4. **Initiation outcome** — latent score
   `0.10·(severity-5) + 0.15·academic + arm·tau + N(0, 0.6)`; the threshold is set at
   the (1 − 0.35) quantile so marginal prevalence is **exactly 0.35**, clamped to the
   [0.20, 0.50] design band (`treatment_arm.py:85-157`). This is what makes the label
   *recoverable* instead of degenerate.
5. **Ground truth stamped per unit** — `treatment_effect_estimate` = the per-unit
   risk-difference tau_i (`patient_generator.py:245`); `TRUE_ATE = mean(tau_i)`.

> ⚠️ **Mixed-run caveat:** when the loader generates all brands in one frame (the
> default), `config.brand` is unset and the CATE map falls back to the **Remibrutinib
> scale for every row** (`patient_generator.py:110`). Per-brand TRUE_ATEs then differ
> only by sampling noise (current run: 0.1718 / 0.1737 / 0.1714). Designed cross-brand
> CATE differences in the *patient* cohorts require per-brand generation runs. The
> hcp_adoption artifacts (generated per brand) DO carry the designed brand differences.

### 2.2 Discontinuation & persistence (initiators only)

`generate_discontinuation_outcomes` (`src/ml/synthetic/generators/cohort_outcomes.py:51-88`,
wired at `patient_generator.py:127-134`):

- `logit(disc) = -0.85 + scale·seg_effect·arm + 0.18·severity − 0.40·academic + N(0,0.5)`
  with `seg_effect = {high: −1.20, medium: −0.70, low: −0.35}` — **treatment lowers
  discontinuation**, most strongly for high-severity patients.
- `persistent_180d = 1 − discontinued_180d` **by construction** — the two cohorts share
  one outcome; persistence prevalence is the complement (current run: 0.467 / 0.533).
- Prevalence is soft-tuned to ~0.30 within [0.05, 0.60] (not quantile-enforced like
  initiation).

### 2.3 HCP adoption (HCP grain, per-brand)

`src/ml/synthetic/generators/hcp_adoption_artifact.py` — an exogenous-centrality causal
chain, deliberately leak-safe:

- `network_size ~ lognormal(3.0, 1.1)` → `centrality_z` (fully exogenous);
- `treatment_arm ~ Bernoulli(sigmoid(0.8·z + noise))` — central HCPs get more rep
  attention (confounded, propensity estimable);
- `logit(adopt) = -0.95 + 0.95·z + brand_scale·tau(segment)·T + N(0,0.6)` with tau-logit
  `{high_influence: 1.30, medium: 0.80, low: 0.40}` and brand scale
  `{Remibrutinib: 1.0, Kisqali: 0.8, Fabhalta: 1.2}` (lines 43-57);
- `cate_estimate` = the **designed probability-scale CATE**:
  `P(adopt|T=1) − P(adopt|T=0)` at fixed centrality (lines 92-94).
- None of the adoption-derived leaky columns (`adopter_rank`, `days_to_first`, …) are
  emitted, so topology features carry no label information beyond the designed path.

Measured on the current run: mean `cate_estimate` Fabhalta 0.194 > Remibrutinib 0.164 >
Kisqali 0.133 — exactly the designed 1.2 / 1.0 / 0.8 ordering.

### 2.4 Provenance and dates

- Every row in every table is stamped `is_synthetic=True` centrally
  (`load_synthetic_data.py:380-387`).
- `--anchor-to-now` remaps all dates onto a rolling window ending at run time (current
  run: 60.8% of treatment events within NOW()−30d, zero future-dated) so windowed KPIs
  read non-zero; re-anchored per run, not a one-off backfill.

---

## 3. File inventory (`data/rwd/synthetic/`)

| File | Grain | What it is |
|---|---|---|
| `<table>.parquet` × 24 | per table | snapshots of every loader table (hcp_profiles, patient_journeys, treatment_events, ml_predictions, triggers, business_metrics, feature_values, ml_experiments, …) |
| `cohort_frames/<cohort>__<brand>.parquet` (12) | patient (or HCP) | the resolved causal frame per (cohort, brand): `treatment_arm, outcome, disease_severity, age_at_diagnosis, segment_assignment, propensity_score, treatment_effect_estimate, brand, is_synthetic`; disc/persist cells filter `treatment_initiated==1`; hcp_adoption cells are `[hcp_id, cate_estimate, is_synthetic]` |
| `per_hcp_cate_hcp_adoption_<brand>.parquet` (3) | HCP | Shard-08 allocation-builder input (same content as the hcp_adoption cohort frames) |
| `ground_truth_<run>.json` | (brand, dgp_type) | TRUE_ATE + `cate_by_segment` + split counts; tolerance 0.10; written by `scripts/write_ground_truth_sidecar.py` (the loader itself does NOT write it) |
| `manifest.json` | run | table list, row counts, timestamp, `is_synthetic: true` |
| `README.md` | run | provenance of the current dataset |

---

## 4. How the platform consumes it

**Provenance enforcement (real analyses never see synthetic rows):**

- SSOT helper `apply_provenance_filter` (`src/repositories/provenance.py:21`) appends
  `.eq('is_synthetic', False)` on every tagged PostgREST read unless the caller passes
  `include_synthetic=True` (**default `False` everywhere**) — threaded through
  `BaseRepository`, `kpi_resolution`, `cohort_resolution`, the gap_analyzer connectors.
- DB-side, migrations 063/066/067/069 (`database/migrations/`) add the column to ~26
  tables and rewrite the `kpi_query` RPC's taggable statements as
  `(SELECT * FROM <t> WHERE is_synthetic = false)` — with a parallel
  `*_include_synthetic` statement family for validation runs.
- Estimator-side, `PROVENANCE_DROP_COLS` keeps `is_synthetic` out of every design matrix
  (`causal_impact/nodes/estimation.py:158-170`, `heterogeneous_optimizer/nodes/cate_estimator.py:212-220`).

**Who gets real frames at dispatch** (`src/agents/orchestrator/nodes/dispatcher.py:581-586`):
`tool_composer` and `heterogeneous_optimizer` resolve real KPI/cohort frames
(`triggers ⋈ treatment_events`, or `patient_journeys` via `cohort_resolution`);
`resource_optimizer` and `prediction_synthesizer` fail closed without structured params;
`causal_impact` receives `data` frames via the chatbot/API cohort resolution
(`src/api/routes/chatbot_tools.py:1115-1135`).

**Estimators** consume the frame as: `treatment_arm`→treatment, cohort outcome→outcome,
`disease_severity`/`age_at_diagnosis`/segment-ordinal as confounders/modifiers, routed to
LinearDML / CausalForestDML / DRLearner / OLS. `treatment_effect_estimate` and
`propensity_score` are *ground-truth columns for validation* — they must NOT be handed
to an estimator as covariates (they'd leak the answer).

---

## 5. Step-by-step validation

### 5.0 The lean ground-truth core (run these first, always)

Six checks carry almost all the information. 1–4 run **offline from the parquet**; 5–6
need the DB substrate. Expected values are from the current `data/rwd/synthetic/` run.

| # | Check | Pass criterion | Measured (Remibrutinib) |
|---|---|---|---|
| 1 | **ATE recovery** — adjusted estimate on `initiation__<brand>` vs sidecar TRUE_ATE | abs err < 0.10 (sidecar tolerance) | IPW w/ stored e: 0.1634 vs 0.1718 ✓ |
| 2 | **Confounding contrast** — naive diff-in-means must be *more* biased than the adjusted estimate | adjusted abs err < naive abs err | naive 0.2690 (err +0.097) vs IPW err −0.008 ✓ |
| 3 | **CATE segment ordering** — recovered CATE high > medium > low, vs sidecar `cate_by_segment` | strict ordering preserved | 0.294 > 0.191 > 0.074 ✓ |
| 4 | **Propensity & overlap** — AUC of arm ~ (severity, academic); stored `propensity_score` strictly inside (0,1) | 0.55 < AUC < 0.95; e ∈ [0.01, 0.99] | AUC 0.682; e ∈ [0.029, 0.574] ✓ |
| 5 | **Provenance leakage zero** (DB) — gate 9: no untagged synthetic rows; real-mode reads exclude them | 0 untagged + real-mode RPC returns | gate 9 |
| 6 | **Date freshness** (DB) — windowed KPIs non-zero after `--anchor-to-now` load | per-brand TRx>0 in NOW()−30d | gate 1 |

> **Why #2 is non-negotiable:** on this draw the naive estimate (0.2690) is 0.0972 from
> truth — *just inside* the ±0.10 tolerance. Tolerance alone (check #1) cannot
> distinguish a broken estimator that ignores confounding from a working one; the
> naive-vs-adjusted contrast can. Structural invariants worth asserting alongside:
> one row per patient in every cohort frame, `treatment_effect_estimate` present,
> disc/persist cells initiators-only with `persistent = 1 − discontinued`, prevalence
> in band (initiation [0.20, 0.50]; disc/persist [0.05, 0.60]).
> Optional 7th (cross-brand, hcp_adoption only): mean `cate_estimate` ordering
> Fabhalta > Remibrutinib > Kisqali (designed 1.2/1.0/0.8) — the patient cohorts do NOT
> carry designed brand differences in a mixed run (§2.1 caveat).

Offline snippet for checks 1–4 (memory-safe, <1 GB):

```python
import pandas as pd, numpy as np, json, glob
base = "data/rwd/synthetic"
f = pd.read_parquet(f"{base}/cohort_frames/initiation__Remibrutinib.parquet")
y, t, e = f["outcome"].values, f["treatment_arm"].values, f["propensity_score"].values

naive = y[t==1].mean() - y[t==0].mean()
ipw = (np.average(y[t==1], weights=1/e[t==1])
       - np.average(y[t==0], weights=1/(1-e[t==0])))          # checks 1+2
gt = json.load(open(sorted(glob.glob(f"{base}/ground_truth_*.json"))[-1]))
true_ate = next(g["true_ate"] for g in gt
                if g["brand"]=="Remibrutinib" and g["dgp_type"]=="confounded")
assert abs(ipw - true_ate) < 0.10 and abs(ipw - true_ate) < abs(naive - true_ate)

seg = f.groupby("segment_assignment")["treatment_effect_estimate"].mean()  # check 3
assert seg["high_severity"] > seg["medium_severity"] > seg["low_severity"]
assert 0.01 <= e.min() and e.max() <= 0.99                                  # check 4
```

To validate *your* estimator instead of IPW: fit it on
`treatment_arm / outcome / disease_severity / age_at_diagnosis / segment_assignment`
**only** (never feed it `propensity_score` or `treatment_effect_estimate` — those are
the answer key), then apply checks 1–3 to its estimates.

### 5.1 Mode A — offline validation from the parquet (no DB, droplet-safe)

1. `export LOKY_MAX_CPU_COUNT=1` (OOM discipline — econml/sklearn CV will otherwise
   fork per core).
2. Run §5.0 checks 1–4 per brand × cohort cell you care about. For disc/persistence,
   outcome semantics flip (treatment *reduces* discontinuation → expect negative
   effect on `discontinued_180d`, positive on `persistent_180d`).
3. For hcp_adoption: the frames are control-less designed-CATE artifacts —
   validate `mean(cate_estimate)` ordering across brands and heterogeneity (std > 0),
   not a treatment contrast.
4. Optionally run a real estimator (LinearDML/CausalForestDML) per cell and compare to
   the sidecar — `scripts/validate_synthetic_causal.py` gate 3 is the reference
   implementation of exactly this (`_resolve_synthetic_frame` + tolerance 0.10).

### 5.2 Mode B — full pipeline validation (DB substrate + 11-gate ladder)

From a clean docker-Supabase (the prod DB is the **local docker stack**, not the cloud
mirror). Canonical runbook (`scripts/validate_synthetic_causal.py:1159-1220`):

```bash
export LOKY_MAX_CPU_COUNT=1 E2I_DB_INTEGRATION=1   # + SUPABASE_URL / key in .env

# 1) migrations (idempotent) — provenance columns + kpi_query rewrite
bash scripts/run_migrations.sh

# 2) clean prior synthetic rows (loader appends namespaced ids per run)
#    DELETE FROM <t> WHERE is_synthetic=true;  (reverse-FK order, per taggable table)

# 3) generate + load the substrate (all 3 brands x 4 cohorts)
python scripts/load_synthetic_data.py --anchor-to-now

# 4) build the gate inputs
python scripts/write_ground_truth_sidecar.py        # gate 3 TRUE_ATE sidecar
python scripts/build_synthetic_ensemble_manifest.py # gate 7: >=2 real models/cell
#    (hcp_adoption cohort_frames for gate 10 come from the loader's --parquet-out)

# 5) the full ladder — eleven "[PASS]" lines, exit 0
python scripts/validate_synthetic_causal.py --all
#    or one gate: --gate 3   (exit 2 = no Supabase client; crash = FAIL, never silent)

# 6) staleness re-check: re-run the loader on a LATER date, then --gate 1 must still pass
```

The eleven gates (`validate_synthetic_causal.py`, registry at `GATES`):

| Gate | Name | Validates | Source |
|---|---|---|---|
| 1 | DATE-FRESHNESS | per-brand synthetic TRx > 0 in NOW()−30d + conversion_rate > 0 | DB |
| 2 | KPI→DASHBOARD | `get_kpi_summary` reads `data_source=database`; ≥4 non-zero metrics opt-in | DB |
| 3 | **ATE/CATE RECOVERY** | CausalImpactAgent recovers sidecar TRUE_ATE within ±0.10 | DB + sidecar |
| 4 | TRIGGER EFFECTIVENESS | `kpi_query` RPC uplift: treatment_rate > control_rate, uplift > 0 | DB (RPC) |
| 5 | gap_analyzer | ≥3 prioritized opportunities, addressable value > 0 | DB |
| 6 | heterogeneous_optimizer | production dispatcher resolver → heterogeneity_score > 0.4, responders, CATE segments | DB |
| 7 | prediction_synthesizer | ≥2 real fitted models load; model_agreement > 0.5 | files + DB |
| 8 | resource_optimizer | `/api/resources/optimize` solver_status=optimal (in-process) | none (DB-free) |
| 9 | **PROVENANCE LEAKAGE** | zero untagged rows on all taggable tables; real-mode RPC excludes synthetic | DB |
| 10 | 4-cohort × 3-brand | all 12 cells: label rate in [0.05, 0.60] + recovered ATE not None | DB + hcp parquet |
| 11 | **CHAT-PATH e2e** | real chat query → orchestrator → heterogeneous_optimizer → ≥2 CATE segments + numeric ATE (~96s) | DB |

If you only have budget for three gates: **3, 9, 11** (truth recovery, pollution
safety, full chat-path reachability).

---

## 6. Gotchas

- **Tolerance vs naive bias** — see §5.0 #2; always pair gate 3 with the naive contrast.
- **Sidecar is regenerated, not read from the load** — the loader does not call
  `GroundTruthStore.to_json_file` (parquet drops `df.attrs`);
  `write_ground_truth_sidecar.py` regenerates with the loader's seed/DGP. Pass
  `--n <loader patient count>` (25000 for FULL_SIZES); the regenerated frame is not
  byte-identical (per-brand n differs ~0.4%) but realized ATE agrees within ~0.0003.
- **Mixed-run brand scaling** (§2.1) — patient-cohort brand differences are sampling
  noise unless generated per brand; hcp_adoption differences are designed.
- **No per-brand/per-cohort CLI on the harness** — `--gate N` / `--all` only; gates
  2/3/5/6/7/11 hardcode Kisqali internally; gate 10 covers all 12 cells.
- **Cohort-frame writer fix (PR #860)** — frames written before it carry duplicated
  patient rows and no `treatment_effect_estimate`; regenerate rather than reuse old
  `cohort_frames/` output.
- **OOM discipline** (droplet) — `LOKY_MAX_CPU_COUNT=1`; read parquet with explicit
  `columns=`; never full-tree mypy/pytest; gate 11 takes ~96s serialized (inside the
  120s heterogeneous_optimizer SLA).
- **DB row cleanup** — synthetic rows are safe to leave (default-excluded everywhere)
  and cleanly removable with `DELETE FROM <t> WHERE is_synthetic=true` per taggable
  table in reverse-FK order.
