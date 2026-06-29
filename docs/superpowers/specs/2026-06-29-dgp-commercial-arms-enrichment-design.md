# DGP Commercial-Arms Enrichment — Design

**Date:** 2026-06-29
**Status:** Design — approved for implementation planning
**Author:** session 6ad36ed5 (brainstorming flow)
**Related:** [`2026-06-20-segment-analysis-clinical-hte-design.md`](2026-06-20-segment-analysis-clinical-hte-design.md), [`2026-06-21-persistence-dgp-enrichment-design.md`](2026-06-21-persistence-dgp-enrichment-design.md)

## 1. Problem & intent

The `/segment-analysis` and `/causal-discovery` pages offer a deliberately narrow set of treatment/outcome
options because the synthetic gold-standard cohort (`patient_journeys`) only wires a single causal
relationship: the confounded binary `treatment_arm` → initiation/persistence outcomes, with a known,
recoverable `TRUE_ATE`. Offering any other treatment/outcome pair would return meaningless results (the DGP
plants no effect there) or fail closed. This is correct behaviour, not a bug — but it limits the analyst to
one lever.

**Goal:** enrich the DGP so the cohort wires **four additional commercial/clinical levers**, each as an
independent, observationally-identified arm with a planted, recoverable effect on a causally-sensible
outcome. Then expand the allowlist so the analyst can study them. The four levers (user-prioritised):

1. **Financial / copay support** (`copay_support`)
2. **Patient support program** (`psp_enrolled`)
3. **Rep detailing / sampling** (`rep_detailing_high`, `sample_dropped`)
4. **Adherence / refill outcomes** (binarized: `adherent_180d`, `low_gap_180d`)

**Non-goals (YAGNI):** no continuous-treatment arms; no interaction/synergy effects between arms (each arm
is additive and independent); no new physical tables (all columns land on `patient_journeys`); no change to
the existing `treatment_arm` arm's tuned behaviour.

### Approved approach decisions (from brainstorming)

- **Identification:** observational + confounding (each arm has a designed propensity on its own confounder
  set; the estimator must adjust to recover the honest effect). Not RCT-style.
- **Outcome type:** binarize adherence to clinical thresholds (keep raw continuous columns as covariates).
- **Rollout:** Approach A — additive independent arms, phased, with the binarized-adherence outcomes on the
  *existing* arm as **Phase 0**.

## 2. Causal structure & curated pair map

Each new arm is a **binary** treatment with its own confounder set, an estimable propensity (overlap
guaranteed), and a **brand-scaled** planted CATE carrying a known per-arm `TRUE_ATE`. Pairs are curated to
causally-sensible outcomes — not a cross-product (e.g. copay support acts *after* a script is written, so it
is not paired with initiation).

| Arm (binary T) | Confounders (→ backdoor adjustment set) | Curated outcome(s) | Planted ATE direction (brand-scaled, RD scale) |
|---|---|---|---|
| `copay_support` | `insurance_access_score`, `disease_severity` (support skews to high-OOP / sicker) | `persistent_180d`, `adherent_180d`, `low_gap_180d` | ↑ persistence/adherence (+8–12 pp) |
| `psp_enrolled` | `disease_severity`, `engagement_score`, `academic_hcp` | `adherent_180d`, `persistent_180d` | ↑ adherence/persistence (+5–10 pp) |
| `rep_detailing_high` | `academic_hcp`, `engagement_score` | `treatment_initiated` | ↑ initiation (+3–6 pp) |
| `sample_dropped` | `academic_hcp`, `engagement_score` | `treatment_initiated` | ↑ initiation (+2–5 pp) |

**New binarized outcomes** (Phase 0, wired onto the *existing* `treatment_arm` first, then reused by the arms
above):

- `adherent_180d` = `1{adherence_rate ≥ 0.8}` (PDC threshold)
- `low_gap_180d` = `1{gap_days ≤ 30}`

The raw `adherence_rate` (PDC) and `gap_days` stay populated as covariates / feature-store inputs.

**Categorical confounders are confounded via a numeric proxy.** `copay_support`'s real-world confounder is
insurance coverage (a categorical column the EconML/DoWhy executors cannot adjust on directly). The DGP
instead confounds copay on a **numeric** `insurance_access_score` — the access gradient already encoded by the
module's `_INIT_INS_ACCESS` map (commercial best → uninsured worst) — materialized as a persisted, allowlisted
covariate. This keeps the backdoor adjustment set numeric and the identification contract (§2) cleanly
satisfiable. The raw categorical `insurance_type` remains a cohort filter only.

**Effect sizes are illustrative starting points.** They become configurable constants, brand-scaled like the
current DGP. Final coefficients are whatever makes the recovery gate (§5) green across seeds — the probe is
the tuning instrument, not these numbers.

### Identification contract (corrected MED-2 lesson)

The binding contract — locked by `tests/unit/test_synthetic/test_arm_confounder_contract.py` — is
**adjustment-set completeness**: *every covariate an arm is confounded on MUST appear in the analysis
covariate allowlist* so the estimator can adjust and recover the honest effect. Otherwise the estimator
silently reports the confounded naive difference-in-means (a plausible-but-wrong value — the anti-mocking
harm).

Confounders and effect-modifiers **may overlap** — the existing arm confounds on `disease_severity` *and*
uses it as the segment/effect-modifier, and that is causally valid. (An earlier draft of this design asserted
an "X ∩ W = ∅" guardrail; that was wrong and is corrected here. The real constraint is the inverse:
W ⊆ analysis-covariate-allowlist.)

## 3. Schema, columns & materialization

A new column must be registered in **three** places or it is silently dropped at load:
(1) migration DDL, (2) `PatientGenerator.generate()` emits it, (3) `batch_loader.py`'s per-table column
allowlist (`batch_loader.py:125`, which gates anything unlisted). Then the read-side allowlists in `causal.py`
expose it.

### Migration `088_synthetic_commercial_arms.sql`

Follows the migration-064 precedent: additive, idempotent (`ADD COLUMN IF NOT EXISTS`), canonical names only,
default NULL until the generator backfills, ends with `NOTIFY pgrst, 'reload schema'` and no inner `COMMIT`.
All DDL is front-loaded in this one migration; columns are invisible/harmless while un-allowlisted, so
front-loading does not couple the phases.

| Column | Type | Role |
|---|---|---|
| `copay_support` | SMALLINT | arm T (0/1) |
| `psp_enrolled` | SMALLINT | arm T (0/1) |
| `rep_detailing_high` | SMALLINT | arm T (0/1) |
| `sample_dropped` | SMALLINT | arm T (0/1) |
| `copay_support_propensity` | DOUBLE PRECISION | per-arm e(X) for backdoor |
| `psp_enrolled_propensity` | DOUBLE PRECISION | per-arm e(X) |
| `rep_detailing_high_propensity` | DOUBLE PRECISION | per-arm e(X) |
| `sample_dropped_propensity` | DOUBLE PRECISION | per-arm e(X) |
| `adherent_180d` | SMALLINT | binarized outcome (PDC ≥ 0.8) |
| `low_gap_180d` | SMALLINT | binarized outcome (gap ≤ 30d) |
| `insurance_access_score` | DOUBLE PRECISION | numeric access gradient from `insurance_type` (copay backdoor covariate) |

Each arm gets its **own** propensity column: each arm has a different confounder set and target population, so
a shared propensity would not identify the per-arm backdoor. `adherence_rate` / `gap_days` (added NULL by
migration 033) get **populated** by the generator so the binarized outcomes are derivable and the raw columns
become usable covariates.

### Ground-truth metadata

Today `df.attrs["true_ate"]` is a single scalar (the `treatment_arm` arm). Add
`df.attrs["true_ate_by_arm"]` = `{arm: {outcome: {"ate": float, "cate_by_segment": {seg: rd}}}}` so the
recovery harness validates each arm/outcome pair independently. The existing scalar stays for backward-compat
with the current arm and its persisted KPIs.

### Three-place registration per new column

1. Migration DDL (above).
2. `PatientGenerator.generate()` emits the column (NULL-safe before its phase populates it).
3. Add to `batch_loader.py`'s `patient_journeys` registered list (`batch_loader.py:125`), else gated out.

## 4. DGP wiring

Two existing primitives in `src/ml/synthetic/dgp/treatment_arm.py` are generalised; the existing arm and the
initiation outcome are refactored to **delegate** to the generalised forms with their current tuned
coefficients, so their behaviour is byte-identical and the recovery-probe / calibration tests stay green.

### 4.1 Arm assignment

`assign_treatment_arm` is hardcoded to `ARM_CONFOUNDERS = (disease_severity, academic_hcp)`. Add a generalised
`assign_arm_from_spec(covariates, spec, rng)` that reads a per-arm confounder→coefficient map + intercept and
returns `(arm, propensity)` with the same `clip(0.01, 0.99)` overlap guarantee. Refactor the existing function
to delegate to it. `test_arm_confounder_contract` stays green.

**Categorical confounders → numeric proxy.** Rather than confound on the raw categorical `insurance_type`
(which the executors cannot adjust on), the generator emits a numeric `insurance_access_score` from
`insurance_type` using the module's existing `_INIT_INS_ACCESS` gradient, and `copay_support`'s `ArmSpec`
confounds on that numeric column. So every `ArmSpec` confounder is numeric and carries a scalar coef; the
categorical→numeric mapping happens once at generation, not inside `assign_arm_from_spec`. This keeps the
backdoor adjustment set numeric end-to-end (DGP, probe, and production estimator all adjust on the same
allowlisted numeric column).

### 4.2 Outcome core

Extract the latent-score → quantile-threshold → analytic counterfactual-RD machinery
(`binary_outcome_with_cate` + `_counterfactual_rd`) into a general
`binary_outcome_rd(arm, baseline, segment, cate_map, rng, *, target_prevalence, noise_std)`. It returns
`(y, tau_i)` where `tau_i` carries exactly 3 distinct per-segment RD values (de-confounded, monotone
high>medium>low), exactly as today.

The existing initiation outcome (`binary_outcome_with_cate`) becomes a thin wrapper that builds the initiation
baseline + `_INIT_LATENT_CATE_BOOST` + prognostic offset and delegates to the core. Its tuned coefficients
(`baseline_severity_coef=0.10`, `baseline_academic_coef=0.15`, `noise_std=0.6`, the T11 boost) are preserved,
so `test_dgp_recovery_probe` and `test_initiation_calibration` stay green.

New arms call `binary_outcome_rd` with their **own** baseline coefficients and brand-scaled segment CATE — they
do **not** inherit the initiation-specific `_INIT_LATENT_CATE_BOOST` or prognostic offset.

### 4.3 Per-arm `ArmSpec` registry

One dataclass per arm:

```python
@dataclass(frozen=True)
class ArmSpec:
    name: str                       # column name, e.g. "copay_support"
    confounders: dict               # name -> scalar coef OR name -> {category: coef}
    intercept: float                # sets base treatment share
    segment_var: str = "disease_severity"   # 3-band assign_segment (uniform, proven)
    baseline_coefs: dict            # outcome-baseline coefficients (X -> coef)
    target_outcomes: tuple          # curated outcome columns this arm wires
    # brand-scaled CATE map derived via the existing brand_scaled_cate pattern
```

The segment / effect-modifier reuses the 3-band `assign_segment(disease_severity)` for **all** arms in this
iteration — keeping the proven recovery machinery uniform. Per-arm effect modifiers (e.g. `engagement_score`
for the rep arms) are a possible later refinement, explicitly out of scope here.

### 4.4 Adherence outcomes (single-latent approach)

`adherent_180d` / `low_gap_180d` are generated by the **recoverable** `binary_outcome_rd` core (authoritative,
known RD). The raw `adherence_rate` (PDC, [0,1]) and `gap_days` (count) are then drawn as **noisy continuous
proxies of the same latent score**, so:

- `adherence_rate ≥ 0.8` ≈ `adherent_180d` (consistency, validated in §5),
- the raw columns are clinically coherent covariates,
- the **binary** stays the recovery-probe target — no two-threshold disagreement.

`gap_days` is drawn inversely to the adherence latent (higher adherence → fewer gap days), thresholded at 30
for `low_gap_180d`.

### 4.5 Ground truth

Each arm/outcome's per-segment RD map (via the existing `rd_map_from_tau`) and mean ATE land in
`df.attrs["true_ate_by_arm"][arm][outcome]`.

## 5. Validation & recovery harness

Cheapest-disproof-first, made permanent, per arm: no arm reaches the page until the probe proves its planted
effect is recoverable in a faithful in-process run.

### 5.1 Generalise the probe

`recover_ate_and_cate(df)` → `recover_ate_and_cate(df, *, treatment_col, outcome_col, confounders,
segment_col, true_ate, cate_map)`, with defaults preserving today's call
(`treatment_arm` / `treatment_initiated` / `ARM_CONFOUNDERS` / `segment_assignment`) so the current
integration test (`tests/integration/test_dgp_recovery_probe.py`) stays green. Each arm's `confounders` (from
its `ArmSpec`) **is** its backdoor adjustment set — exactly as `ARM_CONFOUNDERS` is today.

### 5.2 New parametrized recovery gate

Over `{copay_support, psp_enrolled, rep_detailing_high, sample_dropped} × {curated outcomes} × {3 brands}`,
asserting per pair (same thresholds as today):

- propensity estimable: `propensity_auc > 0.5`, both arms populated (`n_treated >= 30`, `n_control >= 100`) →
  overlap holds;
- ATE recovery: `|linear_dml_ate − true_ate_by_arm[arm][outcome]| < 0.15` (RD-scale tolerance);
- CATE ordering: `high_severity > medium_severity > low_severity`.

An arm whose effect is not recoverable fails CI **before** the allowlist exposes it — the structural guarantee
that we never offer a meaningless pair. Marked `@pytest.mark.heavy_ml` (groups econml on one worker).

### 5.3 Confounder-contract guard (corrected MED-2 lesson)

Extend `test_arm_confounder_contract` so **each** new arm's confounder set is a subset of the analysis
covariate allowlist (`_CAUSAL_DATASET_SPECS` + the segment route). Offering an arm without its adjustment set
→ CI red. This is the durable defense against the silent-confounded-naive-diff harm. Because every arm
confounds on numeric columns (the copay arm uses `insurance_access_score`, not the raw categorical
`insurance_type` — §4.1), each arm's full confounder set is allowlistable as covariates and the contract is
cleanly satisfiable.

### 5.4 Prevalence / consistency guards

- Each new binary outcome lands in the `[0.20, 0.50]` prevalence band (guaranteed by the quantile-threshold
  construction; asserted).
- `adherence_rate ≥ 0.8` agrees with `adherent_180d` within a tolerance (the single-latent consistency check
  from §4.4).

### 5.5 Tuning reality (honest note)

The §2 effect sizes are starting points. Like Fabhalta's fragile med/low gap today, some arms may need per-arm
coefficient tuning. The recovery probe is the tuning instrument; final coefficients are whatever makes the
gate green across seeds (21 / 7 / 99 / 123, as the existing probe uses), not the illustrative numbers.

## 6. Allowlist wiring & page integration

The `/segment-analysis` and `/causal-discovery` pages are fully data-driven from a **single SSOT**:
`/segments/datasets` reads `_CAUSAL_DATASET_SPECS["patient_journeys"]` from `causal.py`
(`segments.py:1243-1258`). Extending that one allowlist flows to both surfaces automatically.

Per phase:

- Extend `_CAUSAL_DATASET_SPECS["patient_journeys"]` — add the arm to `treatment`, the new outcomes to
  `outcome`, and (once populated) `adherence_rate` / `gap_days` / `insurance_access_score` to `covariate`.
- Extend `_CAUSAL_NUMERIC_COLUMNS["patient_journeys"]` so the new columns coerce to float for the executors.
- The raw categorical `insurance_type` stays a brand-style cohort *filter*, **not** a covariate (matching the
  existing categorical-exclusion comment in `causal.py`). Its causal role is carried by the numeric
  `insurance_access_score` covariate (§4.1).

`/segments/datasets`, `/causal/variables`, and the segment-route validation all pick these up automatically —
no per-surface code change.

**Display labels (light polish, Phase 0).** Add a `column → label` map returned in the `/segments/datasets`
(and `/causal/variables`) response so `copay_support` renders as "Copay support", `adherent_180d` as "Adherent
at 180d", etc. Keeps the frontend data-driven (no FE humanizer). Not a blocker if deferred to a later phase.

## 7. Phased rollout (Approach A — additive independent arms)

Migration `088` front-loads all DDL. Each phase then ships independently = generator wiring + recovery-probe
gate + allowlist entry. A column that exists but is NULL and un-allowlisted is invisible and harmless, so
phases do not couple.

- **Phase 0** — binarized adherence outcomes (`adherent_180d`, `low_gap_180d`) + raw `adherence_rate` /
  `gap_days` on the **existing** `treatment_arm`. Smallest increment; proves the binarize-and-recover
  machinery end-to-end. Includes the display-label map.
- **Phase 1** — `copay_support` → `persistent_180d` / `adherent_180d` / `low_gap_180d`.
- **Phase 2** — `psp_enrolled` → `adherent_180d` / `persistent_180d`.
- **Phase 3** — `rep_detailing_high` + `sample_dropped` → `treatment_initiated`.

Each phase is its own PR, recovery-gated, deployed, and live-verified before the next begins.

## 8. Testing strategy (TDD, RED→GREEN per phase)

- **Unit** (`src/ml/synthetic/...`):
  - `ArmSpec` registry shape + per-arm config.
  - `assign_arm_from_spec`: propensity moves with each confounder (per-arm contract guard); overlap clip
    holds.
  - `insurance_access_score`: numeric gradient derived from `insurance_type` via `_INIT_INS_ACCESS`
    (commercial > medicare > medicaid > uninsured); emitted and allowlisted.
  - `binary_outcome_rd` core: exactly 3 distinct τ, monotone ordering, prevalence in band.
  - adherence-proxy consistency (`adherence_rate ≥ 0.8` ⇔ `adherent_180d` within tolerance; `gap_days`
    inverse).
  - generator emits new columns NULL-safe before populated; loader carries them (registered-column test).
  - `df.attrs["true_ate_by_arm"]` populated and self-consistent with persisted per-unit τ.
- **Integration** (`heavy_ml`): recovery probe per `(arm, outcome, brand)` (§5.2).
- **Route**: `/segments/datasets` + `/causal/variables` expose new options; segment-route validation accepts
  new treatment/outcome and rejects unknown columns; display-label map present.
- **Contract**: extended `test_arm_confounder_contract` per arm (§5.3).
- **Calibration**: prevalence-band tests per new binary outcome.

## 9. Deploy

Per established droplet mechanics (prod == dev == this host), phase by phase:

1. Apply migration `088` on the droplet (additive/idempotent — safe to run before any phase populates).
2. Merge each phase PR preserving history (`--merge`; `--admin` if branch protection requires review).
3. Enable + dispatch `deploy.yml` (`gh workflow enable deploy.yml`; `gh workflow run deploy.yml --ref main`),
   then re-disable after success (the workflow is normally `disabled_manually`).
4. Re-seed the synthetic cohort so the new columns populate (the generator + batch loader run).
5. **Live-verify on eznomics.site**: run a `/segment-analysis` on a new `(treatment, outcome)` pair; confirm
   `status=completed` with a sane recovered ATE and populated results — verify by rendered content, not just
   API 200.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| New arm's planted effect not recoverable off-seed (Fabhalta-style fragility). | Recovery gate across seeds 21/7/99/123 blocks the phase; probe is the tuning instrument (§5.5). |
| Refactor-to-delegate perturbs the existing tuned arm/initiation behaviour. | Existing recovery-probe + calibration tests are the guard; delegation must be byte-identical. They run RED→GREEN unchanged. |
| Two thresholds (continuous PDC vs binary outcome) disagree → analyst confusion. | Single-latent approach: binary is authoritative, raw PDC is a consistent proxy; consistency asserted (§5.4). |
| Categorical `insurance_type` breaks the EconML/DoWhy executors. | Confounding is carried by a numeric `insurance_access_score` proxy (allowlisted covariate); the raw categorical stays a cohort filter only (§4.1, §6). |
| Numpy scalars leak into the response (the bug just fixed). | `_to_native` coercion in `segments.py` already covers the response build; new numeric columns coerce via `_CAUSAL_NUMERIC_COLUMNS`. |

## 11. Open questions

None blocking. Effect-size constants and per-arm baseline coefficients are intentionally left to the
recovery-probe tuning step (§5.5) rather than fixed here.
