# Persistence/Discontinuation DGP Enrichment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the persistence/discontinuation gold-standard outcome depend on 7 leakage-safe covariates (not 3) with a higher signal-to-noise ratio, lifting achievable model AUC from ~0.70 to a realistic ~0.78–0.82, so `/feature-importance` legitimately ranks 7 features and the 6 persistence/discontinuation models improve.

**Architecture:** The outcome is generated in `src/ml/synthetic/generators/cohort_outcomes.py` as `discontinued = Bernoulli(sigmoid(logit))`, `persistent = 1 - discontinued`. We extend `logit` **additively** with 4 prognostic terms (insurance, comorbidity burden, age, prior-therapy lines) drawn **independently of `treatment_arm`** (so the true ATE/CATE is mathematically preserved), re-tune the intercept/noise to keep prevalence in [0.05,0.60] and hit the AUC target, lock coefficients via a measure-don't-assume calibration harness, re-lock `KEEP_COLUMNS` for the two cohorts, then retrain + reseed.

**Tech Stack:** Python 3.12, NumPy, scipy.special.expit, scikit-learn (LogisticRegression/roc_auc_score), pytest. DB: Supabase/Postgres migration. Generators under `src/ml/synthetic`.

**Spec:** `docs/superpowers/specs/2026-06-21-persistence-dgp-enrichment-design.md`

**Spec correction (discovered during planning):** insurance_type (`patient_generator.py:220`) and age_at_diagnosis (`:225`) are already generated and reusable, but `comorbidities[]` is unpopulated by this generator and `previous_treatment` lives on `treatment_events`, not `patient_journeys`. So 2 of the 4 drivers (`comorbidity_burden`, `prior_therapy_lines`) require one small additive migration + generation, not pure reuse. The 7 covariates are: `disease_severity, academic_hcp, geographic_region, insurance_type, age_at_diagnosis, comorbidity_burden, prior_therapy_lines`.

**Branch:** `feat/t9-persistence-dgp-enrichment` (already created).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/ml/synthetic/generators/cohort_outcomes.py` | The outcome structural equation | Add 4 driver params + coefficient constants + insurance map; re-tune intercept/noise |
| `tests/unit/test_synthetic/test_cohort_outcomes.py` | Hermetic DGP math tests | Update `_inputs()`, add per-driver + treatment-independence tests |
| `database/migrations/0NN_persistence_drivers.sql` | New driver columns | Additive `comorbidity_burden`, `prior_therapy_lines` on `patient_journeys` |
| `src/ml/synthetic/generators/patient_generator.py` | Patient frame assembly | Hoist driver generation above the outcome call; pass drivers in; emit 2 new columns |
| `tests/unit/test_synthetic/test_patient_generator_cohorts.py` | Generator output tests | Add columns-populated + treatment-independence + signal-present tests |
| `src/ml/synthetic/dgp/recovery_probe.py` | Causal recovery + (NEW) predictive calibration probe | Add `measure_persistence_signal()` |
| `tests/unit/test_synthetic/test_persistence_calibration.py` (new) | AUC + prevalence calibration gate | Asserted AUC∈band, prevalence∈band per brand |
| `src/mlops/gold_standard_eval/cohort_spec.py` | Per-cohort covariate sets | Add `_BASE7`, per-cohort map (persistence/discontinuation→7, initiation→3) |
| `tests/unit/test_mlops/test_gold_standard_eval/test_patient_cohort_factory.py` | Cohort spec tests | Assert persistence/discontinuation use the 7-covariate set |

**Retrain entrypoint:** `python -m src.mlops.gold_standard_eval.run_patient_cohorts` runs all 9 patient slots; the 6 persistence/discontinuation slots get the lift, the 3 initiation slots are unchanged (their outcome is untouched this round).

---

## Task 1: Extend the outcome equation with 4 prognostic drivers

**Files:**
- Modify: `src/ml/synthetic/generators/cohort_outcomes.py`
- Test: `tests/unit/test_synthetic/test_cohort_outcomes.py`

The new drivers enter the **discontinuation** logit. Sign convention (matches existing severity `+`, academic `−`): a driver that *improves persistence* gets a **negative** pull on the discontinuation logit.

- [ ] **Step 1: Update the test `_inputs()` helper to supply the 4 new driver arrays**

Replace the `_inputs` function in `tests/unit/test_synthetic/test_cohort_outcomes.py` (lines 18-29) with:

```python
def _inputs(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    treatment_arm = rng.integers(0, 2, n)  # Shard 03 per-unit arm
    disease_severity = np.clip(rng.normal(5.0, 2.0, n), 0, 10)
    academic_hcp = (rng.random(n) < 0.30).astype(int)
    segment = np.where(
        disease_severity > 7,
        "high_severity",
        np.where(disease_severity > 4, "medium_severity", "low_severity"),
    )
    geographic_region = rng.choice(["midwest", "northeast", "south", "west"], n)
    # NEW prognostic drivers — drawn INDEPENDENTLY of treatment_arm.
    insurance_type = rng.choice(["commercial", "medicare", "medicaid"], n, p=[0.6, 0.3, 0.1])
    age_at_diagnosis = rng.integers(18, 85, n)
    comorbidity_burden = rng.poisson(1.3, n).clip(0, 5)
    prior_therapy_lines = rng.integers(0, 4, n)
    return {
        "rng": rng,
        "treatment_arm": treatment_arm,
        "disease_severity": disease_severity,
        "academic_hcp": academic_hcp,
        "segment": segment,
        "geographic_region": geographic_region,
        "insurance_type": insurance_type,
        "age_at_diagnosis": age_at_diagnosis,
        "comorbidity_burden": comorbidity_burden,
        "prior_therapy_lines": prior_therapy_lines,
        "brand_cate_scale": 1.0,
    }
```

Then update every existing call in the file from the positional/tuple form to `generate_discontinuation_outcomes(**{**_inputs(), "brand_cate_scale": ...})`. Concretely, replace each test body's call with the dict-splat form. Example for `test_discontinuation_prevalence_in_band`:

```python
def test_discontinuation_prevalence_in_band():
    out = generate_discontinuation_outcomes(**_inputs())
    prev = out["discontinued_180d"].mean()
    assert 0.05 <= prev <= 0.60, f"disc prevalence {prev} out of [0.05,0.60]"
    assert np.array_equal(out["persistent_180d"], 1 - out["discontinued_180d"])
    assert 0.05 <= out["persistent_180d"].mean() <= 0.60
```

Apply the same `**_inputs()` splat to `test_treatment_reduces_discontinuation_recoverable`, `test_retention_benefit_is_non_negative`, and `test_region_drives_discontinuation`. For `test_brand_scale_changes_structure`, build two input dicts with different `brand_cate_scale`:

```python
def test_brand_scale_changes_structure():
    a_in = _inputs(seed=11); a_in["brand_cate_scale"] = 0.6
    b_in = _inputs(seed=11); b_in["brand_cate_scale"] = 1.4
    a = generate_discontinuation_outcomes(**a_in)
    b = generate_discontinuation_outcomes(**b_in)
    assert a["discontinued_180d"].mean() != b["discontinued_180d"].mean()
```

- [ ] **Step 2: Add the new per-driver + treatment-independence tests**

Append to `tests/unit/test_synthetic/test_cohort_outcomes.py`:

```python
def test_commercial_insurance_improves_persistence():
    inp = _inputs(n=9000)
    out = generate_discontinuation_outcomes(**inp)
    disc, ins = out["discontinued_180d"], inp["insurance_type"]
    # commercial = best access => lowest discontinuation; medicaid the highest.
    assert disc[ins == "commercial"].mean() < disc[ins == "medicaid"].mean()


def test_comorbidity_burden_increases_discontinuation():
    inp = _inputs(n=9000)
    out = generate_discontinuation_outcomes(**inp)
    disc, com = out["discontinued_180d"], inp["comorbidity_burden"]
    assert disc[com >= 3].mean() > disc[com == 0].mean()


def test_prior_therapy_increases_discontinuation():
    inp = _inputs(n=9000)
    out = generate_discontinuation_outcomes(**inp)
    disc, pr = out["discontinued_180d"], inp["prior_therapy_lines"]
    assert disc[pr >= 2].mean() > disc[pr == 0].mean()


def test_drivers_do_not_disturb_treatment_effect():
    # Drivers are prognostic-only: treatment must still lower discontinuation.
    inp = _inputs(n=9000)
    out = generate_discontinuation_outcomes(**inp)
    disc, t = out["discontinued_180d"], inp["treatment_arm"]
    diff = disc[t == 1].mean() - disc[t == 0].mean()
    assert diff < -0.05, f"treatment must still lower discontinuation; got {diff}"
```

- [ ] **Step 3: Run the new tests to verify they FAIL**

Run: `.venv/bin/python -m pytest tests/unit/test_synthetic/test_cohort_outcomes.py -v`
Expected: FAIL — `generate_discontinuation_outcomes()` got unexpected keyword arguments `insurance_type`/`age_at_diagnosis`/`comorbidity_burden`/`prior_therapy_lines`.

- [ ] **Step 4: Extend the equation in `cohort_outcomes.py`**

Add coefficient constants after `_DISC_REGION_LOGIT` (after line 58):

```python
# --- NEW prognostic drivers (Task T9, 2026-06-21) -------------------------
# Prognostic-only: drawn independently of treatment_arm in patient_generator, so
# they raise predictive AUC WITHOUT changing the true ATE/CATE. Signs are on the
# discontinuation logit (negative = improves persistence).
_INS_DISC_PULL = {       # access gradient: commercial best, medicaid worst
    "commercial": -0.65,
    "medicare": 0.10,
    "medicaid": 0.75,
}
_COMORBIDITY_COEF = 0.28   # per comorbidity: more burden -> more discontinuation
_PRIOR_THERAPY_COEF = 0.32  # per prior line: harder-to-treat -> more discontinuation
_AGE_CENTER = 50.0
_AGE_COEF = 0.018          # per year above center -> slightly more discontinuation
# Re-tuned for the richer equation (was -2.4 / 0.35). Calibration (Task 3) locks
# these so per-brand AUC lands in [0.78,0.82] and prevalence stays in [0.05,0.60].
_DISC_INTERCEPT = -1.35
_DISC_NOISE_SD = 0.25
```

Update the function signature (after `geographic_region: np.ndarray,`, before `segment:`):

```python
    geographic_region: np.ndarray,
    insurance_type: np.ndarray,
    age_at_diagnosis: np.ndarray,
    comorbidity_burden: np.ndarray,
    prior_therapy_lines: np.ndarray,
    segment: np.ndarray,
    brand_cate_scale: float,
```

Replace the `logit = (...)` block (current lines 91-98) with:

```python
    ins_pull = np.array(
        [_INS_DISC_PULL.get(str(i), 0.0) for i in insurance_type], dtype=float
    )
    logit = (
        _DISC_INTERCEPT
        + brand_cate_scale * seg_treat * treatment_arm  # causal effect — UNCHANGED
        + _DISC_SEVERITY_COEF * disease_severity
        + _DISC_ACADEMIC_COEF * academic_hcp
        + region_pull
        + ins_pull
        + _COMORBIDITY_COEF * np.asarray(comorbidity_burden, dtype=float)
        + _PRIOR_THERAPY_COEF * np.asarray(prior_therapy_lines, dtype=float)
        + _AGE_COEF * (np.asarray(age_at_diagnosis, dtype=float) - _AGE_CENTER)
        + rng.normal(0.0, _DISC_NOISE_SD, n)
    )
```

Update the module docstring (lines 14-17) to note the four added prognostic drivers and the re-tuned intercept/noise.

- [ ] **Step 5: Run the tests to verify they PASS**

Run: `.venv/bin/python -m pytest tests/unit/test_synthetic/test_cohort_outcomes.py -v`
Expected: PASS (all, including prevalence-in-band). If prevalence drifts out of [0.05,0.60], nudge `_DISC_INTERCEPT` (more negative → lower discontinuation) and re-run; Task 3 finalizes it.

- [ ] **Step 6: Commit**

```bash
git add src/ml/synthetic/generators/cohort_outcomes.py tests/unit/test_synthetic/test_cohort_outcomes.py
git commit -m "feat(t9): add 4 prognostic drivers to persistence/discontinuation DGP"
```

---

## Task 2: Generate the new driver columns + hoist driver generation above the outcome

**Files:**
- Create: `database/migrations/0NN_persistence_drivers.sql` (use the next free migration number; current latest is in `database/migrations/`)
- Modify: `src/ml/synthetic/generators/patient_generator.py`
- Test: `tests/unit/test_synthetic/test_patient_generator_cohorts.py`

- [ ] **Step 1: Write the additive migration**

Determine the next number: `ls database/migrations/ | tail -3` → use the next integer. Create `database/migrations/0NN_persistence_drivers.sql`:

```sql
-- T9 (2026-06-21): prognostic persistence drivers on patient_journeys.
-- Additive, nullable, pre-index (independent of treatment_arm) => leakage-safe.
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS comorbidity_burden  SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS prior_therapy_lines SMALLINT;

COMMENT ON COLUMN patient_journeys.comorbidity_burden  IS 'Pre-index comorbidity count (0-5); prognostic driver of 180d persistence.';
COMMENT ON COLUMN patient_journeys.prior_therapy_lines IS 'Pre-index prior therapy lines (0-3); prognostic driver of 180d persistence.';
```

- [ ] **Step 2: Write the failing generator test**

Add to `tests/unit/test_synthetic/test_patient_generator_cohorts.py` (follow the file's existing import/build pattern for `PatientGenerator`; if it already builds a generator via a helper, reuse it):

```python
import numpy as np
from src.ml.synthetic.generators.patient_generator import PatientGenerator
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.config import Brand


def _gen(n=6000, seed=3):
    cfg = GeneratorConfig(n_records=n, seed=seed, brand=Brand.REMIBRUTINIB)
    return PatientGenerator(cfg).generate()


def test_new_driver_columns_present_and_varied():
    df = _gen()
    for col in ("comorbidity_burden", "prior_therapy_lines"):
        assert col in df.columns
        assert df[col].notna().all()
        assert df[col].nunique() > 1  # real per-patient variance


def test_drivers_independent_of_treatment_arm():
    df = _gen()
    # Prognostic-only contract: |corr(driver, treatment_arm)| must be ~0.
    for col in ("comorbidity_burden", "prior_therapy_lines", "age_at_diagnosis"):
        corr = np.corrcoef(df[col].to_numpy(float), df["treatment_arm"].to_numpy(float))[0, 1]
        assert abs(corr) < 0.05, f"{col} must be independent of treatment_arm; corr={corr}"


def test_persistence_carries_driver_signal():
    df = _gen()
    # Commercial insurance should persist more than medicaid (signal wired through).
    p = df["persistent_180d"]
    assert p[df["insurance_type"] == "commercial"].mean() > p[df["insurance_type"] == "medicaid"].mean()
```

- [ ] **Step 3: Run the test to verify it FAILS**

Run: `.venv/bin/python -m pytest tests/unit/test_synthetic/test_patient_generator_cohorts.py -k "driver or persistence_carries" -v`
Expected: FAIL — KeyError/assert: `comorbidity_burden` not in columns; outcome call missing kwargs.

- [ ] **Step 4: Hoist driver generation above the outcome call and pass drivers in**

In `patient_generator.py`, BEFORE the `_coh = generate_discontinuation_outcomes(` call (currently line 132), add the driver draws (all independent of `treatment_arm`):

```python
        # T9: prognostic persistence drivers — drawn INDEPENDENTLY of treatment_arm
        # (so they raise predictive AUC without changing the true ATE/CATE).
        insurance_type = self._random_choice(
            [i.value for i in InsuranceTypeEnum],
            n,
            p=[self.INSURANCE_DIST[i] for i in InsuranceTypeEnum],
        )
        age_at_diagnosis = self._random_int(18, 85, n)
        comorbidity_burden = self._rng.poisson(1.3, n).clip(0, 5)
        prior_therapy_lines = self._rng.integers(0, 4, n)
```

Pass them into the outcome call (insert the 4 kwargs before `segment=`):

```python
        _coh = generate_discontinuation_outcomes(
            rng=self._rng,
            treatment_arm=np.asarray(treatment_arm, dtype=int),
            disease_severity=confounders["disease_severity"],
            academic_hcp=confounders["academic_hcp"],
            geographic_region=np.asarray(geographic_region),
            insurance_type=np.asarray(insurance_type),
            age_at_diagnosis=np.asarray(age_at_diagnosis),
            comorbidity_burden=np.asarray(comorbidity_burden),
            prior_therapy_lines=np.asarray(prior_therapy_lines),
            segment=np.asarray(segment),
            brand_cate_scale=_BRAND_CATE_SCALE.get(brand_enum, 1.0),
        )
```

In the DataFrame dict (lines 206-252), REPLACE the inline `insurance_type`/`age_at_diagnosis` generators (lines 220-225) with references to the hoisted variables, and ADD the two new columns:

```python
                "insurance_type": insurance_type,
                "age_at_diagnosis": age_at_diagnosis,
```
and near the causal-substrate columns (after `"persistent_180d": _coh["persistent_180d"],`):
```python
                "comorbidity_burden": comorbidity_burden,
                "prior_therapy_lines": prior_therapy_lines,
```

- [ ] **Step 5: Run the test to verify it PASSES**

Run: `.venv/bin/python -m pytest tests/unit/test_synthetic/test_patient_generator_cohorts.py -v`
Expected: PASS. Also run the full synthetic suite to catch schema/column-list assertions elsewhere: `.venv/bin/python -m pytest tests/unit/test_synthetic -q`.

- [ ] **Step 6: Commit**

```bash
git add database/migrations/0NN_persistence_drivers.sql src/ml/synthetic/generators/patient_generator.py tests/unit/test_synthetic/test_patient_generator_cohorts.py
git commit -m "feat(t9): generate + wire persistence drivers (migration + generator)"
```

---

## Task 3: Calibration harness — measure achieved AUC + prevalence, lock coefficients

**Files:**
- Modify: `src/ml/synthetic/dgp/recovery_probe.py`
- Test: `tests/unit/test_synthetic/test_persistence_calibration.py` (new)

- [ ] **Step 1: Add `measure_persistence_signal()` to `recovery_probe.py`**

Append to `src/ml/synthetic/dgp/recovery_probe.py`:

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

_PERSIST_KEEP = [
    "disease_severity", "academic_hcp", "geographic_region",
    "insurance_type", "age_at_diagnosis", "comorbidity_burden", "prior_therapy_lines",
]


def measure_persistence_signal(df: pd.DataFrame, label: str = "persistent_180d") -> Dict[str, Any]:
    """Fit an LR on the 7 KEEP_COLUMNS (encoded like FeatureBuilder) and return the
    holdout AUC + marginal prevalence. This is the measure-don't-assume gate: it
    reports the ACHIEVED Bayes-proxy AUC of the generated data, not an assumption."""
    cats = [c for c in ("geographic_region", "insurance_type") if c in df.columns]
    nums = [c for c in _PERSIST_KEEP if c not in cats and c in df.columns]
    X = df[cats + nums]
    y = df[label].to_numpy(dtype=int)
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cats)],
        remainder="passthrough",
    )
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))
    return {"holdout_auc": auc, "prevalence": float(y.mean()), "n": int(len(y))}
```

- [ ] **Step 2: Write the failing calibration gate test**

Create `tests/unit/test_synthetic/test_persistence_calibration.py`:

```python
"""Measure-don't-assume gate: the enriched persistence DGP must achieve a
realistic ~0.78-0.82 holdout AUC per brand with prevalence in band. These
asserted numbers LOCK the cohort_outcomes coefficients (like the 2026-06-14
feature experiment locked KEEP_COLUMNS)."""
from __future__ import annotations

import pytest

from src.ml.synthetic.config import Brand
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.patient_generator import PatientGenerator
from src.ml.synthetic.dgp.recovery_probe import measure_persistence_signal


@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
def test_persistence_auc_in_target_band(brand):
    df = PatientGenerator(GeneratorConfig(n_records=12000, seed=42, brand=brand)).generate()
    m = measure_persistence_signal(df)
    assert 0.05 <= m["prevalence"] <= 0.60, f"{brand}: prevalence {m['prevalence']} out of band"
    assert 0.77 <= m["holdout_auc"] <= 0.83, f"{brand}: AUC {m['holdout_auc']} out of [0.77,0.83]"


def test_brands_vary():
    aucs = []
    for b in (Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI):
        df = PatientGenerator(GeneratorConfig(n_records=12000, seed=42, brand=b)).generate()
        aucs.append(measure_persistence_signal(df)["holdout_auc"])
    assert max(aucs) - min(aucs) > 0.005, f"brands should differ; got {aucs}"
```

- [ ] **Step 3: Run the gate to see the ACHIEVED AUC**

Run: `.venv/bin/python -m pytest tests/unit/test_synthetic/test_persistence_calibration.py -v`
Expected: PASS or FAIL with the printed AUC. This is the calibration loop.

- [ ] **Step 4: Tune coefficients in `cohort_outcomes.py` until the gate passes**

If AUC < 0.77: increase signal — raise `_COMORBIDITY_COEF`/`_PRIOR_THERAPY_COEF`/`_INS_DISC_PULL` spread and/or lower `_DISC_NOISE_SD` (e.g. 0.25 → 0.20). If AUC > 0.83: do the inverse. After each change, re-run Step 3 AND `tests/unit/test_synthetic/test_cohort_outcomes.py` (prevalence must stay in band; nudge `_DISC_INTERCEPT` if it drifts). Iterate until both files are green. Do NOT hand-wave — the asserted AUC band is the lock.

- [ ] **Step 5: Commit**

```bash
git add src/ml/synthetic/dgp/recovery_probe.py tests/unit/test_synthetic/test_persistence_calibration.py src/ml/synthetic/generators/cohort_outcomes.py
git commit -m "feat(t9): calibration harness locks persistence AUC to ~0.78-0.82"
```

---

## Task 4: Re-lock KEEP_COLUMNS for persistence/discontinuation

**Files:**
- Modify: `src/mlops/gold_standard_eval/cohort_spec.py`
- Test: `tests/unit/test_mlops/test_gold_standard_eval/test_patient_cohort_factory.py`

- [ ] **Step 1: Write the failing cohort-spec test**

Add to `tests/unit/test_mlops/test_gold_standard_eval/test_patient_cohort_factory.py`:

```python
from src.mlops.gold_standard_eval.cohort_spec import make_patient_spec

_SEVEN = (
    "disease_severity", "academic_hcp", "geographic_region",
    "insurance_type", "age_at_diagnosis", "comorbidity_burden", "prior_therapy_lines",
)


def test_persistence_cohorts_use_seven_covariates():
    for brand in ("Remibrutinib", "Fabhalta", "Kisqali"):
        for cohort in ("persistence", "discontinuation"):
            spec = make_patient_spec(cohort, brand)
            assert spec.base_covariates == _SEVEN, f"{cohort}/{brand}"


def test_initiation_stays_three_covariates():
    spec = make_patient_spec("initiation", "Remibrutinib")
    assert spec.base_covariates == ("disease_severity", "academic_hcp", "geographic_region")
```

- [ ] **Step 2: Run it to verify FAIL**

Run: `.venv/bin/python -m pytest tests/unit/test_mlops/test_gold_standard_eval/test_patient_cohort_factory.py -k "covariates" -v`
Expected: FAIL — persistence currently returns the 3-tuple `_BASE3`.

- [ ] **Step 3: Add `_BASE7` and a per-cohort covariate map**

In `cohort_spec.py`, after the existing `_BASE3` definition (find it near the `PATIENT_COHORTS` block; it equals `("disease_severity", "academic_hcp", "geographic_region")`), add:

```python
_BASE7 = _BASE3 + (
    "insurance_type",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
)
# T9: persistence/discontinuation depend on 7 covariates after the DGP enrichment;
# initiation is unchanged this round (3 covariates).
_PATIENT_COVARIATES: dict[str, tuple[str, ...]] = {
    "initiation": _BASE3,
    "persistence": _BASE7,
    "discontinuation": _BASE7,
}
```

In `make_patient_spec` (line 95-102), change `base_covariates=_BASE3` to:

```python
        base_covariates=_PATIENT_COVARIATES[cohort],
```

Also update the standalone `PERSISTENCE` and `DISCONTINUATION` constants (lines 52-72) `base_covariates=...` to `_BASE7` for consistency (they are the superseded pooled specs but should not contradict the factory).

- [ ] **Step 4: Run it to verify PASS**

Run: `.venv/bin/python -m pytest tests/unit/test_mlops/test_gold_standard_eval/test_patient_cohort_factory.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops/gold_standard_eval/cohort_spec.py tests/unit/test_mlops/test_gold_standard_eval/test_patient_cohort_factory.py
git commit -m "feat(t9): re-lock KEEP_COLUMNS to 7 for persistence/discontinuation cohorts"
```

---

## Task 5: Invariant gate — ATE/CATE recovery still holds

**Files:**
- Test: `tests/unit/test_synthetic/test_persistence_calibration.py` (extend)

The prognostic-only contract claims the true ATE/CATE is unchanged. Prove it on the enriched generator output.

- [ ] **Step 1: Write the failing invariant test**

Append to `tests/unit/test_synthetic/test_persistence_calibration.py`:

```python
from src.ml.synthetic.dgp.recovery_probe import recover_ate_and_cate


def test_ate_cate_recovery_unchanged_by_drivers():
    df = PatientGenerator(GeneratorConfig(n_records=12000, seed=42, brand=Brand.REMIBRUTINIB)).generate()
    rec = recover_ate_and_cate(df)
    true_ate = float(df.attrs["true_ate"])
    assert abs(rec["linear_dml_ate"] - true_ate) < 0.10, (
        f"ATE drifted: recovered {rec['linear_dml_ate']} vs true {true_ate}"
    )
    cate = rec["cate_by_segment_estimate"]
    assert cate["high_severity"] >= cate["medium_severity"] >= cate["low_severity"], cate
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/unit/test_synthetic/test_persistence_calibration.py::test_ate_cate_recovery_unchanged_by_drivers -v`
Expected: PASS (drivers are independent of `treatment_arm`, so recovery is unaffected). If it FAILS, a driver leaked into treatment assignment — re-check Task 2 Step 4 draws use `self._rng` independently and are NOT derived from `treatment_arm`/`propensity`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_synthetic/test_persistence_calibration.py
git commit -m "test(t9): assert ATE/CATE recovery is unchanged by the new prognostic drivers"
```

---

## Task 6: Retrain the 6 models + reseed the substrate (operational)

**Files:** none (runs existing entrypoints against the DB). Requires `E2I_DB_INTEGRATION=1` and DB env (`.env`).

- [ ] **Step 1: Apply the migration**

Apply `database/migrations/0NN_persistence_drivers.sql` via the project's migration path (e.g. the Supabase MCP `apply_migration`, or the repo's migration runner). Verify the columns exist: the two new columns must appear on `patient_journeys`.

- [ ] **Step 2: Reseed the patient substrate**

Run the synthetic reseed that regenerates `patient_journeys` (the project's batch loader / reseed entrypoint, e.g. `src/ml/synthetic/loaders/batch_loader.py`; **load `.env` first** — known gotcha). Confirm `comorbidity_burden`/`prior_therapy_lines` are populated and `persistent_180d` reflects the new signal.

- [ ] **Step 3: Retrain the patient cohorts**

Run: `E2I_DB_INTEGRATION=1 .venv/bin/python -m src.mlops.gold_standard_eval.run_patient_cohorts`
Expected: the 6 persistence/discontinuation models report holdout AUC ~0.78–0.82 (varied); the 3 initiation models are unchanged (~0.64–0.69). **Landmine:** `register_cohort_model` delete+reinsert can hit the `ml_drift_history` FK RESTRICT (error 23503) once metric rows exist — if so, record vs the existing `model_id` by name rather than re-registering (see memory `goldstd_eval_confusion_roc_drifthistory_fk`).

- [ ] **Step 4: Verify the live metrics**

Query `ml_performance_metrics` (or the `/performance/{model}` API) for the 6 models; confirm AUC landed in band and the feature count is 7. Record the achieved per-brand AUCs in the spec's "achieved" note.

- [ ] **Step 5: Commit any operational notes** (no code change; if a retrain helper needed a fix, commit it with `fix(t9): ...`).

---

## Task 7: Downstream validation

**Files:** none (verification).

- [ ] **Step 1: Feature-importance shows 7**

Hit `GET /explain/models` (or the page `/feature-importance`) for a persistence cohort; confirm `keep_columns` now lists the 7 covariates and `/explain/global` returns the enriched encoded set (region + insurance one-hot + 5 numerics). The page's `groupByCovariate` should render 7 grouped rows.

- [ ] **Step 2: Predictive scoring still works**

Confirm "Score holdout cohort" for a persistence model returns predictions (depends on T5 BentoML restart being done; if T5 is unmet, expect the known 400 and treat separately).

- [ ] **Step 3: Segment-analysis + causal unaffected**

Run a segment analysis (`persistent_180d` outcome) and a causal estimate; confirm HTE is still populated and the treatment effect still recovers — the invariant gate in Task 5 predicts this, Step 3 confirms it live.

- [ ] **Step 4: Final full-suite check (targeted) + push**

Run: `.venv/bin/python -m pytest tests/unit/test_synthetic tests/unit/test_mlops/test_gold_standard_eval -q`
Run: `ruff check src/ tests/ && ruff format --check src/ tests/` (scope src+tests, not scripts).
Then push the branch and open the PR (merge non-squash). Rely on CI's MyPy gate (do not run whole-tree mypy on the droplet).

---

## Self-Review (completed during planning)

- **Spec coverage:** structural-equation change (Task 1), 4 drivers + population (Tasks 1–2), prognostic-only/independence (Task 2 test + Task 5), measure-don't-assume calibration (Task 3), invariant gates ATE/CATE/prevalence/leakage (Tasks 3,5 + KEEP_COLUMNS Task 4), re-lock→retrain→reseed (Tasks 4,6), downstream validation (Task 7), scope boundary persistence/discontinuation only (initiation/HCP untouched — Task 4 `_PATIENT_COVARIATES`). All covered.
- **Spec correction recorded:** 2 drivers need a migration (not pure reuse) — noted in header and Task 2.
- **Type/name consistency:** the 4 driver names (`insurance_type, age_at_diagnosis, comorbidity_burden, prior_therapy_lines`) and `_BASE7`/`measure_persistence_signal` are used identically across Tasks 1–7.
- **Placeholder scan:** the only `0NN` is the migration number (the engineer must read `ls database/migrations/` — explicitly instructed), not a content placeholder. Coefficient values are concrete starting points the calibration loop (Task 3) tightens against an asserted gate.
