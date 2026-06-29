# DGP Commercial-Arms Enrichment — Foundation + Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add binarized adherence outcomes (`adherent_180d`, `low_gap_180d`) — plus their raw `adherence_rate`/`gap_days` proxies — to the synthetic `patient_journeys` cohort as **recoverable** outcomes of the *existing* `treatment_arm`, and expose them on the `/segment-analysis` + `/causal-discovery` pages. This is Phase 0 of the four-phase commercial-arms enrichment (spec: `docs/superpowers/specs/2026-06-29-dgp-commercial-arms-enrichment-design.md`).

**Architecture:** Extract the existing latent-score → quantile-threshold → analytic-counterfactual-RD machinery into a reusable `binary_outcome_rd` core (the existing initiation outcome delegates to it, byte-identical). Generate the two new binary outcomes via that core (authoritative, known RD) on the existing arm + segment, then draw `adherence_rate`/`gap_days` as noisy proxies of the same latent. Generalize the recovery probe to validate any `(treatment, outcome, confounders, segment)` tuple, and gate the new outcomes through it. Extend the single allowlist SSOT (`_CAUSAL_DATASET_SPECS`) so both pages pick the options up automatically.

**Tech Stack:** Python 3.12, numpy/scipy, pandas, EconML (LinearDML/CausalForestDML), pytest, FastAPI/Pydantic v2, Supabase (Postgres) migrations.

**Scope / deferral note (YAGNI):** Phase 0 wires outcomes onto the existing arm only — it does **not** build `ArmSpec`, `assign_arm_from_spec`, or `insurance_access_score`. Those are consumed only by the new arms and are built in the Phase 1 plan. Migration `088` front-loads *all* commercial-arms columns (NULL until a later phase populates them) so Phases 1–3 require no further DDL.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `database/migrations/088_synthetic_commercial_arms.sql` | Create | Additive, idempotent DDL for all commercial-arms columns (front-loaded; NULL until populated). |
| `src/ml/synthetic/dgp/treatment_arm.py` | Modify | Extract `binary_outcome_rd` core; refactor `binary_outcome_with_cate` to delegate. |
| `src/ml/synthetic/dgp/adherence_outcomes.py` | Create | `generate_adherence_outcomes()` — recoverable binary outcomes + raw proxies on the existing arm/segment. |
| `src/ml/synthetic/generators/patient_generator.py` | Modify | Call the adherence generator; emit new columns (+ NULL placeholders for later phases); populate `df.attrs["true_ate_by_arm"]`. |
| `src/ml/synthetic/loaders/batch_loader.py` | Modify | Register new `patient_journeys` columns so the loader carries them. |
| `src/ml/synthetic/dgp/recovery_probe.py` | Modify | Generalize `recover_ate_and_cate` to any treatment/outcome/confounder/segment tuple (defaults unchanged). |
| `src/api/routes/causal.py` | Modify | Extend `_CAUSAL_DATASET_SPECS` + `_CAUSAL_NUMERIC_COLUMNS`; add `_COLUMN_LABELS`. |
| `src/api/routes/segments.py` | Modify | Add `labels` to `SegmentDatasetsResponse` + `/datasets` response. |
| `tests/unit/test_synthetic/test_dgp/test_binary_outcome_rd.py` | Create | Core outcome contract (3 distinct τ, ordering, prevalence band). |
| `tests/unit/test_synthetic/test_dgp/test_adherence_outcomes.py` | Create | Recoverable binary + proxy consistency. |
| `tests/unit/test_synthetic/test_dgp/test_patient_generator_adherence.py` | Create | Generator emits columns + `true_ate_by_arm`. |
| `tests/integration/test_dgp_recovery_probe.py` | Modify | Add the adherence recovery gate. |
| `tests/unit/test_synthetic/test_loaders/test_batch_loader_columns.py` | Create/Modify | Loader registers the new columns. |
| `tests/unit/test_api/test_routes/test_segments.py` | Modify | `/datasets` exposes new outcomes + labels. |
| `tests/unit/test_api/test_segment_hte_route.py` | Modify | Segment validation accepts `adherent_180d`. |

---

## Task 1: Migration — front-load all commercial-arms columns

DDL/migration files are a TDD exception (validated by the downstream contract/route tests in Tasks 8–10, not a unit test). Additive + idempotent, following the migration-064 precedent.

**Files:**
- Create: `database/migrations/088_synthetic_commercial_arms.sql`

- [ ] **Step 1: Verify `088` is the next free number**

Run: `ls database/migrations/ | grep -E '^08[0-9]' | sort | tail -3`
Expected: highest is `087_persistence_drivers.sql` (so `088` is free). If not, use the next free number and update all references in this plan.

- [ ] **Step 2: Write the migration**

```sql
-- ============================================================================
-- Migration 088: commercial-arms + binarized-adherence columns on
-- patient_journeys (dgp-commercial-arms-enrichment). Additive + idempotent.
-- All columns front-loaded; NULL until the generator's per-phase wiring fills
-- them. Phase 0 populates adherent_180d/low_gap_180d/adherence_rate/gap_days;
-- the arm + per-arm-propensity + insurance_access_score columns stay NULL until
-- Phases 1-3. Canonical names only. Mirrors migration 064's contract.
-- ----------------------------------------------------------------------------

-- Phase 0: binarized adherence outcomes + raw proxies
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS adherent_180d   SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS low_gap_180d    SMALLINT;
-- adherence_rate / gap_days were added NULL by migration 033; ensure-exists here
-- so this migration is self-contained on a fresh DB.
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS adherence_rate  DOUBLE PRECISION;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS gap_days        DOUBLE PRECISION;

-- Phases 1-3: new arms + per-arm propensity + numeric insurance proxy (NULL now)
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS copay_support                 SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS psp_enrolled                  SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS rep_detailing_high            SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS sample_dropped                SMALLINT;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS copay_support_propensity      DOUBLE PRECISION;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS psp_enrolled_propensity       DOUBLE PRECISION;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS rep_detailing_high_propensity DOUBLE PRECISION;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS sample_dropped_propensity     DOUBLE PRECISION;
ALTER TABLE patient_journeys ADD COLUMN IF NOT EXISTS insurance_access_score        DOUBLE PRECISION;

COMMENT ON COLUMN patient_journeys.adherent_180d IS
    'Binarized adherence outcome (1 = PDC adherence_rate >= 0.8 at 180d). Recoverable effect of treatment_arm. Added by migration 088 (Phase 0).';
COMMENT ON COLUMN patient_journeys.low_gap_180d IS
    'Binarized refill-gap outcome (1 = gap_days <= 30 at 180d). Recoverable effect of treatment_arm. Added by migration 088 (Phase 0).';
COMMENT ON COLUMN patient_journeys.adherence_rate IS
    'Continuous PDC proxy of the adherence latent (covariate). Populated by the generator. Migration 088 (Phase 0).';
COMMENT ON COLUMN patient_journeys.gap_days IS
    'Continuous refill-gap-days proxy (inverse of adherence latent; covariate). Migration 088 (Phase 0).';

NOTIFY pgrst, 'reload schema';
-- (No COMMIT; run_migrations.sh owns the outer --single-transaction.)
```

- [ ] **Step 3: Verify it parses (psql dry-check or syntax read)**

Run: `grep -c 'ADD COLUMN IF NOT EXISTS' database/migrations/088_synthetic_commercial_arms.sql`
Expected: `13` (4 Phase-0 columns + 4 arms + 4 per-arm propensity + insurance_access_score)

- [ ] **Step 4: Commit**

```bash
git add database/migrations/088_synthetic_commercial_arms.sql
git commit -m "feat(dgp): migration 088 front-loads commercial-arms columns (Phase 0 DDL)"
```

---

## Task 2: Extract `binary_outcome_rd` core; delegate the initiation outcome

**Files:**
- Modify: `src/ml/synthetic/dgp/treatment_arm.py`
- Test: `tests/unit/test_synthetic/test_dgp/test_binary_outcome_rd.py`

- [ ] **Step 1: Write the failing test**

```python
"""Phase 0: the general binary-outcome-with-recoverable-RD core."""
import numpy as np
import pytest

from src.ml.synthetic.dgp.treatment_arm import binary_outcome_rd


@pytest.mark.unit
def test_binary_outcome_rd_three_distinct_ordered_tau_in_band():
    rng = np.random.default_rng(7)
    n = 4000
    severity = rng.uniform(0, 10, n)
    segment = np.where(severity > 7, "high_severity",
                       np.where(severity > 4, "medium_severity", "low_severity"))
    arm = (rng.random(n) < 0.5).astype(int)
    baseline = 0.10 * (severity - 5.0)
    cate_map = {"high_severity": 0.9, "medium_severity": 0.5, "low_severity": 0.2}

    y, tau_i = binary_outcome_rd(
        arm, baseline, segment, cate_map, rng,
        target_prevalence=0.35, noise_std=0.6,
    )

    # exactly 3 distinct per-segment RD values, monotone high>medium>low>0
    distinct = sorted(set(np.round(tau_i, 6)))
    assert len(distinct) == 3
    hi = tau_i[segment == "high_severity"][0]
    md = tau_i[segment == "medium_severity"][0]
    lo = tau_i[segment == "low_severity"][0]
    assert hi > md > lo > 0
    # prevalence in the INDEX band by the quantile-threshold construction
    assert 0.20 <= y.mean() <= 0.50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_synthetic/test_dgp/test_binary_outcome_rd.py -v`
Expected: FAIL with `ImportError: cannot import name 'binary_outcome_rd'`.

- [ ] **Step 3: Extract the core and delegate**

In `src/ml/synthetic/dgp/treatment_arm.py`, add the general core (place it directly above `binary_outcome_with_cate`):

```python
def binary_outcome_rd(
    arm: np.ndarray,
    baseline: np.ndarray,
    segment: np.ndarray,
    cate_map: Dict[str, float],
    rng: np.random.Generator,
    *,
    target_prevalence: float = 0.35,
    noise_std: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """General binary outcome Y + per-unit RECOVERABLE segment RD-scale CATE.

    latent = baseline(X) + arm * tau_latent(segment) + N(0, noise_std);
    Y = 1{latent >= q}, q = (1 - target_prevalence) sample quantile (=> marginal
    prevalence ~= target_prevalence, clamped to [0.20, 0.50]). Returns (y, tau_i)
    where tau_i is the per-segment counterfactual risk difference (exactly 3
    distinct values, de-confounded, RD scale) — the quantity LinearDML/
    CausalForestDML recover. ``baseline`` is the caller-built latent baseline
    (so callers own their own confounding / prognostic structure); ``cate_map``
    is the brand-scaled segment CATE on the latent score scale.
    """
    if not (0.20 <= target_prevalence <= 0.50):
        target_prevalence = float(np.clip(target_prevalence, 0.20, 0.50))
    baseline = np.asarray(baseline, dtype=float)
    tau_latent = np.array([cate_map[str(s)] for s in segment], dtype=float)
    noise = rng.normal(0.0, noise_std, len(arm))
    score = baseline + arm.astype(float) * tau_latent + noise
    q = float(np.quantile(score, 1.0 - target_prevalence))
    y = (score >= q).astype(int)
    rd_unit = _counterfactual_rd(baseline, tau_latent, q, noise_std)
    rd_map = {str(s): float(np.mean(rd_unit[segment == s])) for s in np.unique(segment)}
    tau_i = np.array([rd_map[str(s)] for s in segment], dtype=float)
    return y, tau_i
```

Then refactor `binary_outcome_with_cate` to build the initiation baseline + boost and delegate, replacing its body from the `severity = ...` line through the `return y, tau_i` with:

```python
    if not (0.20 <= target_prevalence <= 0.50):
        target_prevalence = float(np.clip(target_prevalence, 0.20, 0.50))

    severity = np.asarray(covariates["disease_severity"], dtype=float)
    academic = np.asarray(covariates["academic_hcp"], dtype=float)
    baseline = baseline_severity_coef * (severity - 5.0) + baseline_academic_coef * academic
    if prognostic_offset is not None:
        baseline = baseline + np.asarray(prognostic_offset, dtype=float)
    # initiation keeps its tuned latent-CATE boost (T11) — applied to the map
    # BEFORE delegation so the core stays boost-agnostic.
    boosted_map = {str(s): float(v) * _INIT_LATENT_CATE_BOOST for s, v in cate_map.items()}
    return binary_outcome_rd(
        arm, baseline, segment, boosted_map, rng,
        target_prevalence=target_prevalence, noise_std=noise_std,
    )
```

- [ ] **Step 4: Run the new test + the existing initiation contracts**

Run: `pytest tests/unit/test_synthetic/test_dgp/test_binary_outcome_rd.py tests/unit/test_synthetic/test_initiation_calibration.py tests/unit/test_synthetic/test_dgp/test_outcome_band.py -v`
Expected: all PASS (the initiation outcome is byte-identical because the boosted map + baseline + quantile construction are unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/ml/synthetic/dgp/treatment_arm.py tests/unit/test_synthetic/test_dgp/test_binary_outcome_rd.py
git commit -m "refactor(dgp): extract binary_outcome_rd core; initiation outcome delegates"
```

---

## Task 3: Adherence outcome generator (single-latent: recoverable binary + proxies)

**Files:**
- Create: `src/ml/synthetic/dgp/adherence_outcomes.py`
- Test: `tests/unit/test_synthetic/test_dgp/test_adherence_outcomes.py`

- [ ] **Step 1: Write the failing test**

```python
"""Phase 0: adherence outcomes — recoverable binary + consistent raw proxies."""
import numpy as np
import pytest

from src.ml.synthetic.dgp.adherence_outcomes import generate_adherence_outcomes


@pytest.mark.unit
def test_adherence_outcomes_recoverable_and_proxy_consistent():
    rng = np.random.default_rng(21)
    n = 5000
    severity = rng.uniform(0, 10, n)
    academic = (rng.random(n) < 0.3).astype(int)
    segment = np.where(severity > 7, "high_severity",
                       np.where(severity > 4, "medium_severity", "low_severity"))
    arm = (rng.random(n) < 0.5).astype(int)
    cate_map = {"high_severity": 0.9, "medium_severity": 0.5, "low_severity": 0.2}

    out = generate_adherence_outcomes(
        treatment_arm=arm,
        disease_severity=severity,
        academic_hcp=academic,
        segment=segment,
        cate_map=cate_map,
        rng=rng,
    )

    # binary outcomes present + in band
    assert set(np.unique(out["adherent_180d"])) <= {0, 1}
    assert set(np.unique(out["low_gap_180d"])) <= {0, 1}
    assert 0.20 <= out["adherent_180d"].mean() <= 0.50

    # raw proxies in plausible clinical ranges
    assert out["adherence_rate"].min() >= 0.0 and out["adherence_rate"].max() <= 1.0
    assert out["gap_days"].min() >= 0.0

    # proxy CONSISTENCY: adherence_rate>=0.8 agrees with adherent_180d for most rows
    agree = np.mean((out["adherence_rate"] >= 0.8) == (out["adherent_180d"] == 1))
    assert agree >= 0.80

    # per-segment RD ground truth: 3 distinct, ordered
    rd = out["adherent_rd_by_segment"]
    assert rd["high_severity"] > rd["medium_severity"] > rd["low_severity"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_synthetic/test_dgp/test_adherence_outcomes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ml.synthetic.dgp.adherence_outcomes'`.

- [ ] **Step 3: Implement the generator**

```python
"""Phase 0 of commercial-arms enrichment: binarized adherence outcomes for the
EXISTING treatment_arm, plus clinically-coherent raw proxies drawn from the same
latent score (so adherence_rate>=0.8 agrees with the authoritative binary).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from src.ml.synthetic.dgp.treatment_arm import binary_outcome_rd, rd_map_from_tau

# Adherence baseline: sicker patients are modestly LESS adherent; academic-HCP
# patients modestly MORE (kept small so the arm effect is the dominant signal).
_ADH_SEVERITY_COEF = -0.08
_ADH_ACADEMIC_COEF = 0.12
_ADH_NOISE_STD = 0.6
# Map the (standardized) adherence latent to a PDC in [0,1] via a logistic squash
# centered so the marginal mean PDC is ~0.7 (typical real-world chronic-Rx PDC).
_PDC_CENTER = 0.0
_PDC_SCALE = 1.1
# gap_days: inverse of adherence; ~ (1 - PDC) * 180-day window, floored at 0.
_GAP_WINDOW_DAYS = 180.0


def generate_adherence_outcomes(
    *,
    treatment_arm: np.ndarray,
    disease_severity: np.ndarray,
    academic_hcp: np.ndarray,
    segment: np.ndarray,
    cate_map: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Return adherent_180d / low_gap_180d (recoverable binaries) + adherence_rate
    / gap_days (raw proxies) + the per-segment RD ground-truth maps.

    The binary outcomes are AUTHORITATIVE (generated by binary_outcome_rd with a
    known counterfactual RD). adherence_rate is a noisy continuous proxy of the
    SAME latent score, so adherence_rate>=0.8 ~= adherent_180d; gap_days is the
    inverse proxy thresholded at 30 for low_gap_180d.
    """
    arm = np.asarray(treatment_arm, dtype=int)
    severity = np.asarray(disease_severity, dtype=float)
    academic = np.asarray(academic_hcp, dtype=float)
    baseline = _ADH_SEVERITY_COEF * (severity - 5.0) + _ADH_ACADEMIC_COEF * academic

    # Authoritative binary: PDC>=0.8 adherence at 180d (prevalence ~0.35 in-band).
    adherent_180d, tau_adherent = binary_outcome_rd(
        arm, baseline, segment, cate_map, rng,
        target_prevalence=0.35, noise_std=_ADH_NOISE_STD,
    )

    # Raw PDC proxy from the SAME latent (recompute the latent w/o re-thresholding):
    tau_latent = np.array([cate_map[str(s)] for s in segment], dtype=float)
    latent = baseline + arm.astype(float) * tau_latent + rng.normal(0.0, _ADH_NOISE_STD, len(arm))
    adherence_rate = np.clip(
        1.0 / (1.0 + np.exp(-(latent - _PDC_CENTER) * _PDC_SCALE)), 0.0, 1.0
    )
    # Calibrate the proxy so adherence_rate>=0.8 lands on ~the same rows as the
    # authoritative binary: shift PDC so its (1 - prevalence) quantile sits at 0.8.
    shift = 0.8 - float(np.quantile(adherence_rate, 1.0 - float(adherent_180d.mean())))
    adherence_rate = np.clip(adherence_rate + shift, 0.0, 1.0)

    gap_days = np.clip((1.0 - adherence_rate) * _GAP_WINDOW_DAYS, 0.0, _GAP_WINDOW_DAYS)
    low_gap_180d = (gap_days <= 30.0).astype(int)

    return {
        "adherent_180d": adherent_180d,
        "low_gap_180d": low_gap_180d,
        "adherence_rate": np.round(adherence_rate, 4),
        "gap_days": np.round(gap_days, 1),
        "adherent_rd_by_segment": rd_map_from_tau(np.asarray(segment), tau_adherent),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_synthetic/test_dgp/test_adherence_outcomes.py -v`
Expected: PASS. If `agree < 0.80`, widen the calibration (the `shift` step) before changing the threshold — the binary stays authoritative.

- [ ] **Step 5: Commit**

```bash
git add src/ml/synthetic/dgp/adherence_outcomes.py tests/unit/test_synthetic/test_dgp/test_adherence_outcomes.py
git commit -m "feat(dgp): recoverable adherence outcomes + consistent raw proxies (Phase 0)"
```

---

## Task 4: Wire adherence outcomes into `PatientGenerator.generate()`

**Files:**
- Modify: `src/ml/synthetic/generators/patient_generator.py` (call site ~line 168; DataFrame build ~line 234-278; attrs ~line 280-288)
- Test: `tests/unit/test_synthetic/test_dgp/test_patient_generator_adherence.py`

- [ ] **Step 1: Write the failing test**

```python
"""Phase 0: PatientGenerator emits adherence columns + true_ate_by_arm."""
import numpy as np
import pytest

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator


@pytest.mark.unit
def test_generator_emits_adherence_columns_and_true_ate_by_arm():
    cfg = GeneratorConfig(seed=21, n_records=2000, brand=Brand.REMIBRUTINIB,
                          dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()

    for col in ("adherent_180d", "low_gap_180d", "adherence_rate", "gap_days"):
        assert col in df.columns, f"{col} missing from generated frame"
    assert df["adherent_180d"].notna().all()
    assert set(np.unique(df["adherent_180d"])) <= {0, 1}

    # later-phase columns exist as NULL placeholders (so the loader carries them)
    for col in ("copay_support", "psp_enrolled", "insurance_access_score"):
        assert col in df.columns
        assert df[col].isna().all()

    # per-arm ground truth for the adherence outcomes
    tba = df.attrs["true_ate_by_arm"]
    assert "treatment_arm" in tba
    assert "adherent_180d" in tba["treatment_arm"]
    assert tba["treatment_arm"]["adherent_180d"]["ate"] > 0
    assert set(tba["treatment_arm"]["adherent_180d"]["cate_by_segment"]) == {
        "high_severity", "medium_severity", "low_severity"
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_synthetic/test_dgp/test_patient_generator_adherence.py -v`
Expected: FAIL with `KeyError: 'adherent_180d'` (column not in frame).

- [ ] **Step 3: Wire the generator**

In `patient_generator.py`, add the import near the existing dgp imports:

```python
from src.ml.synthetic.dgp.adherence_outcomes import generate_adherence_outcomes
```

After the `_coh = generate_discontinuation_outcomes(...)` block (ends ~line 168), add:

```python
        # Phase 0 (commercial-arms enrichment): binarized adherence outcomes of the
        # EXISTING treatment_arm, on the SAME segment/CATE map (single SSOT). The
        # binary is authoritative + recoverable; adherence_rate/gap_days are proxies.
        _adh = generate_adherence_outcomes(
            treatment_arm=np.asarray(treatment_arm, dtype=int),
            disease_severity=confounders["disease_severity"],
            academic_hcp=confounders["academic_hcp"],
            segment=np.asarray(segment),
            cate_map=latent_cate_map,
            rng=self._rng,
        )
```

In the `pd.DataFrame({...})` build (after the `"persistent_180d": _coh["persistent_180d"],` line ~274), add the Phase 0 columns and the NULL placeholders for later phases:

```python
                # Phase 0 adherence outcomes + raw proxies (migration 088).
                "adherent_180d": _adh["adherent_180d"],
                "low_gap_180d": _adh["low_gap_180d"],
                "adherence_rate": _adh["adherence_rate"],
                "gap_days": _adh["gap_days"],
                # Phases 1-3 commercial arms — NULL placeholders so the loader
                # carries them; populated by their phase's generator wiring.
                "copay_support": np.nan,
                "psp_enrolled": np.nan,
                "rep_detailing_high": np.nan,
                "sample_dropped": np.nan,
                "copay_support_propensity": np.nan,
                "psp_enrolled_propensity": np.nan,
                "rep_detailing_high_propensity": np.nan,
                "sample_dropped_propensity": np.nan,
                "insurance_access_score": np.nan,
```

After the existing `df.attrs[...]` block (~line 281-286), add the per-arm ground-truth map:

```python
        # Per-arm/outcome recoverable ground truth (commercial-arms enrichment).
        # Existing arm: treatment_initiated (scalar above) + the Phase 0 adherence
        # outcomes. Later phases extend this dict with their arm keys.
        df.attrs["true_ate_by_arm"] = {
            "treatment_arm": {
                "treatment_initiated": {
                    "ate": float(np.mean(tau_i)),
                    "cate_by_segment": cate_map,
                },
                "adherent_180d": {
                    "ate": float(np.mean([_adh["adherent_rd_by_segment"][str(s)] for s in segment])),
                    "cate_by_segment": _adh["adherent_rd_by_segment"],
                },
            }
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_synthetic/test_dgp/test_patient_generator_adherence.py -v`
Expected: PASS.

- [ ] **Step 5: Run the existing generator/arm tests for regressions**

Run: `pytest tests/unit/test_synthetic/test_dgp/ tests/unit/test_synthetic/test_initiation_calibration.py -q`
Expected: all PASS (existing columns and attrs unchanged; we only added).

- [ ] **Step 6: Commit**

```bash
git add src/ml/synthetic/generators/patient_generator.py tests/unit/test_synthetic/test_dgp/test_patient_generator_adherence.py
git commit -m "feat(dgp): emit adherence outcomes + true_ate_by_arm; NULL placeholders for later phases"
```

---

## Task 5: Register the new columns in `batch_loader`

**Files:**
- Modify: `src/ml/synthetic/loaders/batch_loader.py` (the `patient_journeys` registered-column list, ~line 125-165)
- Test: `tests/unit/test_synthetic/test_loaders/test_batch_loader_columns.py`

- [ ] **Step 1: Write the failing test**

```python
"""Phase 0: the loader's patient_journeys registered columns include the new ones."""
import pytest

from src.ml.synthetic.loaders import batch_loader


@pytest.mark.unit
def test_patient_journeys_registers_commercial_arms_columns():
    registered = set(batch_loader._REGISTERED_COLUMNS["patient_journeys"])
    for col in (
        "adherent_180d", "low_gap_180d", "adherence_rate", "gap_days",
        "copay_support", "psp_enrolled", "rep_detailing_high", "sample_dropped",
        "copay_support_propensity", "psp_enrolled_propensity",
        "rep_detailing_high_propensity", "sample_dropped_propensity",
        "insurance_access_score",
    ):
        assert col in registered, f"{col} not registered -> loader will drop it"
```

- [ ] **Step 2: Confirm the registered-columns symbol name, then run the test**

Run: `grep -n '"patient_journeys": \[' src/ml/synthetic/loaders/batch_loader.py`
Then confirm the enclosing dict name (expected `_REGISTERED_COLUMNS`):
Run: `grep -n '_REGISTERED_COLUMNS\|REGISTERED_COLUMNS' src/ml/synthetic/loaders/batch_loader.py | head -3`
If the symbol differs, update the test's `batch_loader._REGISTERED_COLUMNS` reference to match.

Run: `pytest tests/unit/test_synthetic/test_loaders/test_batch_loader_columns.py -v`
Expected: FAIL (new columns absent from the registered list).

- [ ] **Step 3: Add the columns to the registered list**

In `batch_loader.py`, inside the `"patient_journeys": [ ... ]` list, after the `"persistent_180d",` entry (~line 160), add:

```python
        # Commercial-arms enrichment (migration 088). Phase 0 fills the adherence
        # columns; the arm + propensity + insurance_access_score columns load as
        # NULL until their phase populates them (nullable in DB).
        "adherent_180d",
        "low_gap_180d",
        "adherence_rate",
        "gap_days",
        "copay_support",
        "psp_enrolled",
        "rep_detailing_high",
        "sample_dropped",
        "copay_support_propensity",
        "psp_enrolled_propensity",
        "rep_detailing_high_propensity",
        "sample_dropped_propensity",
        "insurance_access_score",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_synthetic/test_loaders/test_batch_loader_columns.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ml/synthetic/loaders/batch_loader.py tests/unit/test_synthetic/test_loaders/test_batch_loader_columns.py
git commit -m "feat(dgp): register commercial-arms columns in batch_loader (Phase 0)"
```

---

## Task 6: Generalize the recovery probe (defaults unchanged)

**Files:**
- Modify: `src/ml/synthetic/dgp/recovery_probe.py`
- Test: covered by the existing `tests/integration/test_dgp_recovery_probe.py` (defaults) + the new gate in Task 7.

- [ ] **Step 1: Add a focused unit test for the new signature**

Append to `tests/unit/test_synthetic/test_dgp/test_binary_outcome_rd.py`:

```python
@pytest.mark.unit
def test_recovery_probe_accepts_explicit_tuple_signature():
    # Signature-only smoke (no econml fit): the function must accept the new
    # keyword args without TypeError. We monkeypatch the heavy estimators out.
    import inspect
    from src.ml.synthetic.dgp import recovery_probe

    sig = inspect.signature(recovery_probe.recover_ate_and_cate)
    for p in ("treatment_col", "outcome_col", "confounders", "segment_col",
              "true_ate", "cate_map"):
        assert p in sig.parameters, f"{p} missing from recover_ate_and_cate signature"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest "tests/unit/test_synthetic/test_dgp/test_binary_outcome_rd.py::test_recovery_probe_accepts_explicit_tuple_signature" -v`
Expected: FAIL (`treatment_col missing from ... signature`).

- [ ] **Step 3: Generalize the function**

Replace the `recover_ate_and_cate` signature + the data-extraction lines (29-35) and the return's `true_ate` line (69) with:

```python
def recover_ate_and_cate(
    df: pd.DataFrame,
    *,
    treatment_col: str = "treatment_arm",
    outcome_col: str = "treatment_initiated",
    confounders: list | None = None,
    segment_col: str = "segment_assignment",
    true_ate: float | None = None,
    cate_map: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Recover ATE (LinearDML) + per-segment CATE (CausalForestDML) from a
    synthetic patient frame. Defaults reproduce the original treatment_arm ->
    treatment_initiated probe; pass the keyword args to validate any other arm/
    outcome/confounder/segment tuple (commercial-arms enrichment)."""
    covars = list(confounders) if confounders is not None else _COVARS
    Y = df[outcome_col].to_numpy(dtype=float)
    T = df[treatment_col].to_numpy(dtype=int)
    X = df[covars].to_numpy(dtype=float)
    seg = df[segment_col].to_numpy()
```

And change the returned `true_ate` line to prefer the explicit arg:

```python
        "true_ate": float(true_ate) if true_ate is not None else float(df.attrs.get("true_ate", np.mean(eff))),
```

(`cate_map` is accepted for caller symmetry/forward-use; the probe's recovered ordering is asserted by the caller against it. Leave the body otherwise unchanged.)

- [ ] **Step 4: Run the unit smoke + the existing integration gate (defaults unchanged)**

Run: `pytest "tests/unit/test_synthetic/test_dgp/test_binary_outcome_rd.py::test_recovery_probe_accepts_explicit_tuple_signature" -v`
Expected: PASS.
Run: `pytest tests/integration/test_dgp_recovery_probe.py -v -m heavy_ml`
Expected: PASS (defaults reproduce the original behaviour).

- [ ] **Step 5: Commit**

```bash
git add src/ml/synthetic/dgp/recovery_probe.py tests/unit/test_synthetic/test_dgp/test_binary_outcome_rd.py
git commit -m "refactor(dgp): generalize recover_ate_and_cate to any tuple (defaults unchanged)"
```

---

## Task 7: Adherence recovery gate (the cheapest-disproof, made permanent)

**Files:**
- Modify: `tests/integration/test_dgp_recovery_probe.py`

- [ ] **Step 1: Write the failing gate**

Append to `tests/integration/test_dgp_recovery_probe.py`:

```python
@pytest.mark.parametrize("brand", [Brand.REMIBRUTINIB, Brand.FABHALTA, Brand.KISQALI])
def test_adherence_outcome_recoverable_on_existing_arm(brand):
    """Phase 0: treatment_arm -> adherent_180d must be recoverable (ATE within
    tolerance + CATE ordering), proving the binarized adherence outcome carries
    the planted effect BEFORE the allowlist exposes it."""
    cfg = GeneratorConfig(seed=21, n_records=3000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()
    truth = df.attrs["true_ate_by_arm"]["treatment_arm"]["adherent_180d"]

    out = recover_ate_and_cate(
        df,
        treatment_col="treatment_arm",
        outcome_col="adherent_180d",
        confounders=list(ARM_CONFOUNDERS),
        segment_col="segment_assignment",
        true_ate=truth["ate"],
        cate_map=truth["cate_by_segment"],
    )

    assert out["propensity_auc"] > 0.5
    assert out["n_treated"] >= 30 and out["n_control"] >= 100
    assert abs(out["linear_dml_ate"] - out["true_ate"]) < 0.15, out
    cate = out["cate_by_segment_estimate"]
    assert cate["high_severity"] > cate["medium_severity"] > cate["low_severity"]
```

Add the `ARM_CONFOUNDERS` import at the top of the file:

```python
from src.ml.synthetic.dgp.treatment_arm import ARM_CONFOUNDERS
```

- [ ] **Step 2: Run the gate**

Run: `pytest "tests/integration/test_dgp_recovery_probe.py::test_adherence_outcome_recoverable_on_existing_arm" -v -m heavy_ml`
Expected: PASS across all 3 brands. If a brand fails ATE tolerance or CATE ordering, tune `_ADH_NOISE_STD` / `_ADH_SEVERITY_COEF` in `adherence_outcomes.py` (the probe is the tuning instrument — §5.5 of the spec); do NOT loosen the assertion thresholds.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_dgp_recovery_probe.py
git commit -m "test(dgp): recovery gate for adherence outcomes on existing arm (Phase 0)"
```

---

## Task 8: Extend the allowlist SSOT (`causal.py`)

**Files:**
- Modify: `src/api/routes/causal.py` (`_CAUSAL_DATASET_SPECS` ~line 811-827; `_CAUSAL_NUMERIC_COLUMNS` ~line 863-878)
- Test: `tests/unit/test_synthetic/test_arm_confounder_contract.py` (must stay green) + new assertion below.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_api/test_routes/test_segments.py` (a focused allowlist assertion):

```python
@pytest.mark.unit
def test_patient_journeys_allowlist_exposes_adherence_outcomes():
    from src.api.routes.causal import _CAUSAL_DATASET_SPECS, _CAUSAL_NUMERIC_COLUMNS

    spec = _CAUSAL_DATASET_SPECS["patient_journeys"]
    assert "adherent_180d" in spec["outcome"]
    assert "low_gap_180d" in spec["outcome"]
    assert "adherence_rate" in spec["covariate"]
    assert "gap_days" in spec["covariate"]

    numeric = _CAUSAL_NUMERIC_COLUMNS["patient_journeys"]
    for col in ("adherent_180d", "low_gap_180d", "adherence_rate", "gap_days"):
        assert col in numeric, f"{col} must coerce to float for the executors"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest "tests/unit/test_api/test_routes/test_segments.py::test_patient_journeys_allowlist_exposes_adherence_outcomes" -v`
Expected: FAIL (`adherent_180d` not in `spec["outcome"]`).

- [ ] **Step 3: Extend the allowlist**

In `_CAUSAL_DATASET_SPECS["patient_journeys"]`, add the new outcomes + covariates:

```python
    "patient_journeys": {
        "treatment": ["treatment_arm", "treatment_initiated"],
        "outcome": [
            "persistent_180d",
            "discontinued_180d",
            "treatment_initiated",
            "adherent_180d",
            "low_gap_180d",
        ],
        "covariate": [
            "disease_severity",
            "engagement_score",
            "age_at_diagnosis",
            "academic_hcp",
            "geographic_region",
            "egfr",
            "proteinuria_g_day",
            "ldh_ratio",
            "urticaria_severity_uas7",
            "ecog_performance_status",
            "adherence_rate",
            "gap_days",
        ],
    },
```

In `_CAUSAL_NUMERIC_COLUMNS["patient_journeys"]`, add (inside the set):

```python
        "adherent_180d",
        "low_gap_180d",
        "adherence_rate",
        "gap_days",
```

Update the read-only comment block above `_CAUSAL_DATASET_SPECS` (line 807-810): `adherence_rate`/`gap_days` are no longer "100% NULL — deliberately NOT offered"; note Phase 0 now populates them.

- [ ] **Step 4: Run the new test + the contract guard**

Run: `pytest "tests/unit/test_api/test_routes/test_segments.py::test_patient_journeys_allowlist_exposes_adherence_outcomes" tests/unit/test_synthetic/test_arm_confounder_contract.py -v`
Expected: all PASS (the contract guard stays green — the existing arm's confounders are still in the covariate list).

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/causal.py tests/unit/test_api/test_routes/test_segments.py
git commit -m "feat(api): allowlist adherence outcomes + raw proxies for patient_journeys (Phase 0)"
```

---

## Task 9: Display labels in the `/datasets` response

**Files:**
- Modify: `src/api/routes/causal.py` (add `_COLUMN_LABELS` near the allowlist)
- Modify: `src/api/routes/segments.py` (`SegmentDatasetsResponse` ~line 1215; `get_segment_datasets` ~line 1256)
- Test: `tests/unit/test_api/test_routes/test_segments.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_api/test_routes/test_segments.py`:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_segment_datasets_returns_human_labels():
    from src.api.routes.segments import get_segment_datasets

    resp = await get_segment_datasets()
    assert resp.labels.get("adherent_180d") == "Adherent at 180d"
    assert resp.labels.get("treatment_arm") == "Treatment arm"
    # every offered treatment/outcome has a label
    for col in resp.treatments + resp.outcomes:
        assert col in resp.labels, f"{col} has no display label"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest "tests/unit/test_api/test_routes/test_segments.py::test_get_segment_datasets_returns_human_labels" -v`
Expected: FAIL (`AttributeError: 'SegmentDatasetsResponse' object has no attribute 'labels'`).

- [ ] **Step 3a: Add the label map in `causal.py`**

Below `_CAUSAL_DATASET_SPECS`, add:

```python
# Human-readable display labels for the curated columns (data-driven FE; keeps
# the frontend free of a humanizer). Columns absent here fall back to the raw
# name title-cased by the caller.
_COLUMN_LABELS: Dict[str, str] = {
    "treatment_arm": "Treatment arm",
    "treatment_initiated": "Treatment initiated",
    "persistent_180d": "Persistent at 180d",
    "discontinued_180d": "Discontinued at 180d",
    "adherent_180d": "Adherent at 180d",
    "low_gap_180d": "Low refill gap (<=30d)",
    "adherence_rate": "Adherence rate (PDC)",
    "gap_days": "Refill gap (days)",
}
```

- [ ] **Step 3b: Add `labels` to the response model + populate it**

In `segments.py`, add the field to `SegmentDatasetsResponse`:

```python
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Human-readable display labels keyed by column name",
    )
```

(Ensure `from typing import Dict` is imported — it is, alongside `List`.)

In `get_segment_datasets`, build the labels for the offered columns and pass them:

```python
    from src.api.routes.causal import _CAUSAL_DATASET_SPECS, _COLUMN_LABELS

    spec = _CAUSAL_DATASET_SPECS[_SEGMENT_HTE_DATASET]
    offered = list(spec["treatment"]) + list(spec["outcome"])
    labels = {c: _COLUMN_LABELS.get(c, c.replace("_", " ").capitalize()) for c in offered}
```

```python
    return SegmentDatasetsResponse(
        treatments=list(spec["treatment"]),
        outcomes=list(spec["outcome"]),
        brands=brands,
        labels=labels,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest "tests/unit/test_api/test_routes/test_segments.py::test_get_segment_datasets_returns_human_labels" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api/routes/causal.py src/api/routes/segments.py tests/unit/test_api/test_routes/test_segments.py
git commit -m "feat(api): human-readable display labels in /segments/datasets (Phase 0)"
```

---

## Task 10: Segment-route accepts the new outcome (end-to-end allowlist validation)

**Files:**
- Modify: `tests/unit/test_api/test_segment_hte_route.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_api/test_segment_hte_route.py` (follow the file's existing TestClient/validation pattern; adapt the request helper to the one already used there):

```python
@pytest.mark.unit
def test_segment_route_accepts_adherent_180d_outcome():
    """adherent_180d is now an allowlisted outcome -> the route must validate it
    (no 400). Unknown columns must still 400."""
    from src.api.routes.causal import _CAUSAL_DATASET_SPECS

    spec = _CAUSAL_DATASET_SPECS["patient_journeys"]
    assert "adherent_180d" in spec["outcome"]            # allowlisted
    assert "made_up_outcome" not in spec["outcome"]      # still rejected
```

(If the file already has a request-level validation helper that exercises the 400 path, prefer asserting through it — e.g. posting `outcome_var="adherent_180d"` returns non-400 and `outcome_var="made_up_outcome"` returns 400. Use the established helper rather than duplicating TestClient setup.)

- [ ] **Step 2: Run test to verify it fails / passes**

Run: `pytest "tests/unit/test_api/test_segment_hte_route.py::test_segment_route_accepts_adherent_180d_outcome" -v`
Expected: PASS once Task 8 is merged (this test pins the contract; if Task 8 is not yet applied it FAILS on the allowlist assertion).

- [ ] **Step 3: Run the full segment-route + causal-route suites for regressions**

Run: `pytest tests/unit/test_api/test_segment_hte_route.py tests/unit/test_api/test_routes/test_segments.py -q`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_api/test_segment_hte_route.py
git commit -m "test(api): segment route accepts adherent_180d, rejects unknown (Phase 0)"
```

---

## Task 11: Deploy + live-verify (manual checklist — not TDD)

**Files:** none (operational).

- [ ] **Step 1: Run the focused suites locally (targeted, per droplet policy — do NOT run whole-tree mypy/pytest here)**

Run:
```bash
pytest tests/unit/test_synthetic/test_dgp/ tests/unit/test_synthetic/test_loaders/ \
       tests/unit/test_api/test_routes/test_segments.py tests/unit/test_api/test_segment_hte_route.py -q
ruff check src/
mypy src/ml/synthetic/dgp/adherence_outcomes.py src/ml/synthetic/dgp/treatment_arm.py
```
Expected: green. (Whole-tree mypy/pytest are CI's job — see CLAUDE.md.)

- [ ] **Step 2: Open the PR (preserve history — never squash)**

```bash
git config --global http.https://github.com.proxy ""
git push -u origin <branch>
gh pr create --title "feat(dgp): Phase 0 — binarized adherence outcomes on patient_journeys" \
  --body "Implements Phase 0 of docs/superpowers/specs/2026-06-29-dgp-commercial-arms-enrichment-design.md. Recoverable adherent_180d/low_gap_180d outcomes of the existing treatment_arm + raw adherence_rate/gap_days proxies; allowlist + labels; recovery-gated."
```
Wait for CI green (Type Check / Unit / Heavy Unit / Integration / Agents gates).

- [ ] **Step 3: Apply migration 088 on the droplet**

Per `reference-supabase-droplet-migration-apply` memory: apply `088_synthetic_commercial_arms.sql` (additive/idempotent — safe before any data populates).

- [ ] **Step 4: Merge (admin if branch protection requires), then deploy**

```bash
gh pr merge <PR#> --merge --admin --delete-branch
gh workflow enable deploy.yml
gh workflow run deploy.yml --ref main
# watch to success, then re-disable (deploy.yml is normally disabled_manually)
gh workflow disable deploy.yml
```

- [ ] **Step 5: Re-seed the synthetic cohort so the adherence columns populate**

Run the synthetic generation + batch load path that materializes `patient_journeys` (the same path the prior DGP-enrichment rounds used). Confirm `adherent_180d` is non-NULL in the live table.

- [ ] **Step 6: Live-verify on eznomics.site (rendered content, not just API 200)**

On `/segment-analysis`: select treatment `treatment_arm`, outcome `Adherent at 180d`, run the analysis. Confirm `status=completed` with a sane recovered ATE (positive, plausible pp magnitude) and populated results. Repeat for `Low refill gap (<=30d)`.

---

## Self-Review

**1. Spec coverage (Phase 0 + the foundation it needs):**
- §2 binarized adherence outcomes (`adherent_180d` PDC≥0.8, `low_gap_180d` gap≤30) → Tasks 3, 4. ✅
- §2 identification contract (existing arm's confounders stay allowlisted) → Task 8 keeps `test_arm_confounder_contract` green. ✅
- §3 migration `088` (all columns front-loaded, three-place registration) → Tasks 1, 4, 5. ✅
- §3 `df.attrs["true_ate_by_arm"]` → Task 4. ✅
- §4.2 `binary_outcome_rd` core + delegate (existing behaviour frozen) → Task 2. ✅
- §4.4 single-latent adherence (binary authoritative, raw proxies consistent) → Task 3. ✅
- §5.1 generalized probe → Task 6. §5.2 recovery gate (per-brand) → Task 7. §5.4 prevalence + proxy consistency → Tasks 3, 7. ✅
- §6 allowlist SSOT + numeric coercion + display labels → Tasks 8, 9. ✅
- §7 Phase 0 increment; §9 deploy/live-verify → Task 11. ✅
- **Deferred to Phase 1 plan (correct per §7):** `ArmSpec`, `assign_arm_from_spec`, `insurance_access_score` population, the new arms, the per-arm contract-guard extension, `_INIT_INS_ACCESS`-based gradient. Migration `088` already front-loads their columns. Documented in the scope note. ✅

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N" — every code step shows full code; the one parametric reference (batch_loader symbol name in Task 5) includes a grep to confirm and an instruction to adjust. ✅

**3. Type/name consistency:**
- `binary_outcome_rd(arm, baseline, segment, cate_map, rng, *, target_prevalence, noise_std)` — defined Task 2, called Task 3 with matching kwargs. ✅
- `generate_adherence_outcomes(*, treatment_arm, disease_severity, academic_hcp, segment, cate_map, rng)` returning keys `adherent_180d/low_gap_180d/adherence_rate/gap_days/adherent_rd_by_segment` — defined Task 3, consumed Task 4 with exactly those keys. ✅
- `recover_ate_and_cate(df, *, treatment_col, outcome_col, confounders, segment_col, true_ate, cate_map)` — defined Task 6, called Task 7 with matching kwargs. ✅
- `df.attrs["true_ate_by_arm"]["treatment_arm"]["adherent_180d"]["ate" | "cate_by_segment"]` — written Task 4, read Tasks 4-test, 7. ✅
- `_COLUMN_LABELS` (causal.py) + `SegmentDatasetsResponse.labels` — defined Task 9, asserted Task 9 test. ✅
- Column names identical across migration (Task 1), generator (Task 4), loader (Task 5), allowlist (Task 8): `adherent_180d`, `low_gap_180d`, `adherence_rate`, `gap_days`. ✅

No issues outstanding.
