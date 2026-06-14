# Gold-Standard Model Eval — P1 (Initiation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Stand up the initiation cohort end-to-end on the synthetic claims gold-standard — engineer leakage-safe features, train a real model that consumes them, register a loadable champion, run a walk-forward + holdout evaluation that records **real** metrics into `ml_performance_metrics`, and make the Time-Series "Model performance" page render that real trend.

**Architecture:** New package `src/mlops/gold_standard_eval/` (FeatureBuilder → Scorer → WalkForwardRunner → MetricRecorder) + a parametrized cohort deployer (refactor of `prediction_synthesizer_deploy.py`) that trains on gold-standard features instead of `generate_scenario`. Small additive extensions to the metrics writer (`source`/`measured_at` passthrough), the trend read route (raise the day cap + summary window), and the frontend (default model + range + metric-name alignment). No DB migration (the `source` column already exists; idempotency is delete-by-source-then-insert).

**Tech Stack:** Python 3.12, scikit-learn (LogisticRegression, calibration), pandas, FastAPI, Supabase (async), pytest (+ real-DB integration), React/TS frontend.

---

## EXECUTION PROTOCOL (read first — applies to every task)

These are user-mandated and override convenience:

1. **Worktree isolation.** Each implementation task (or independent task group) runs in its own git worktree off this branch (`feat/claims-gold-standard-model-eval`), e.g. `.claude/worktrees/gse-p1-<task>`. Use `superpowers:using-git-worktrees`. Never edit the shared checkout. Reap worktrees after merge-back to the feature branch.
2. **No mocking — real results.** Integration tests hit the real prod Supabase DB (`docker exec supabase-db psql ...` for setup/asserts; the app's async client for code paths). Unit tests may stub *boundaries* (e.g. the Supabase client) but MUST NOT fake model outputs, metrics, or DB rows. A test that asserts a fabricated number is a failure.
3. **Cheap-disproof over theorizing.** Every empirical decision (feature set, window mode, algorithm, n_min) is settled by a small experiment whose result is shown — not by assertion. Each such task names the single assumption + the experiment that would falsify it. Run the experiment; proceed only if the assumption survives.
4. **Codebase-intent first.** Before choosing features/labels, read the authoritative source (`src/services/cohort_resolution.py:259 _PJ_COHORTS`, the generators, PR/issue history). When intent is unclear or a library behavior is uncertain, do **web research** (WebSearch/WebFetch) rather than guess.
5. **Convergence harness.** For any task whose answer is non-obvious or whose first attempt fails, drive it with `ralph-wiggum:ralph-loop` + `codex:codex-rescue` to converge — iterate until tests/experiments pass, escalating to codex for a second diagnosis/implementation pass.
6. **Memory (OOM history) — monitor + batch.** Before any heavy step (training, walk-forward over many months, pytest module, any build) run `free -h`; if **available < 2 GiB, STOP and wait/retry serially** — never run two heavy steps concurrently. Training LR on ≤25k rows is light (~hundreds MB); the walk-forward loop trains ~once/month — keep it serial. **NO local `vite build`, no whole-tree `mypy`/`pytest`** (CI is the arbiter for those).
7. **CI batched at the very end.** Do **not** push per task or trigger CI per task. Commit locally on the feature branch, run **targeted local tests** (memory-guarded) as you go, and run the **single CI batch only at the end** (push the feature branch / open the PR once, after all P1 tasks are green locally). This avoids the concurrent-deploy/OOM churn.
8. **No omitted scope.** Every component in the File Structure below ships in P1 (incl. the holdout headline, the read-path fixes, and the frontend default). If something must be deferred, it is called out explicitly here — nothing is silently dropped.

---

## File Structure

**Create (new package `src/mlops/gold_standard_eval/`):**
- `cohort_spec.py` — `CohortSpec` dataclass + `INITIATION` instance (target/brand/label/covariates), sourced from `_PJ_COHORTS`.
- `feature_builder.py` — `FeatureBuilder`: `patient_ids → (X: DataFrame[feature_columns], y: Series)` from the gold-standard, leakage-safe.
- `scorer.py` — `score(y_true, y_score) -> dict` with page-aligned metric names.
- `walk_forward.py` — `WalkForwardRunner`: expanding-origin train<M / eval=M loop → list of `(month, metrics, n)`.
- `cohort_deployer.py` — parametrized train→serialize→**merge-manifest**→register (refactor reusing `prediction_synthesizer_deploy.py` helpers) for a `CohortSpec` on gold-standard features.
- `run_initiation_eval.py` — CLI wiring: deploy champion → walk-forward + holdout → record.
**Modify:**
- `src/repositories/drift_monitoring.py` — add `source` to `PerformanceMetricRecord`/`to_db_row`; extend `record_metrics(..., measured_at=None, source=None)`; add `delete_metrics(model_id, source, split_version=None)`.
- `src/services/performance_tracking.py` — emit `f1` (not `f1_score`); raise `PerformanceTrackingConfig.trend_window_days`.
- `src/api/routes/monitoring.py:1118` — raise the trend `days` cap.
- `src/mlops/prediction_synthesizer_deploy.py` — extract the manifest-merge + register helpers so `cohort_deployer` reuses them (no behavior change to the CSU go-live path).
- `frontend/src/pages/TimeSeries.tsx` — set `DEFAULT_MODEL_ID`; add a multi-year `TIME_RANGES` entry; confirm `METRIC_OPTIONS` strings match recorded names.
**Test:**
- `tests/unit/test_mlops/test_gold_standard_eval/` (feature_builder, scorer, walk_forward, cohort_deployer) + `tests/unit/test_repositories/test_drift_monitoring_source.py` + `tests/unit/test_services/test_performance_tracking_metric_names.py`.
- `tests/integration/test_gold_standard_initiation_eval.py` (real DB).

---

## Task 0: Worktree + clean baseline

- [ ] **Step 1: Create the P1 worktree**

```bash
REPO=/home/enunez/Projects/e2i_causal_analytics
git -C "$REPO" worktree add "$REPO/.claude/worktrees/gse-p1" -b feat/gse-p1-initiation feat/claims-gold-standard-model-eval
```

- [ ] **Step 2: Verify memory headroom before any Python**

Run: `free -h`
Expected: `available` ≥ 2 GiB. If not, wait/retry — do not proceed.

- [ ] **Step 3: Confirm the cohort intent source exists**

Run: `grep -n "_PJ_COHORTS" src/services/cohort_resolution.py`
Expected: the dict definition (~line 259). Read it; the `initiation` entry's label + covariates are authoritative inputs to Task 1.

---

## Task 1: CohortSpec (initiation) — grounded in `_PJ_COHORTS`

**Files:** Create `src/mlops/gold_standard_eval/cohort_spec.py`; Test `tests/unit/test_mlops/test_gold_standard_eval/test_cohort_spec.py`

- [ ] **Step 1: Write the failing test**

```python
from src.mlops.gold_standard_eval.cohort_spec import INITIATION
def test_initiation_spec_matches_codebase_intent():
    assert INITIATION.target == "csu_treatment_initiation"
    assert INITIATION.brand == "Remibrutinib"
    assert INITIATION.label_column == "treatment_initiated"
    assert INITIATION.grain == "patient"
    # covariates must be a non-empty subset of patient_journeys columns, no label leakage
    assert INITIATION.base_covariates
    assert INITIATION.label_column not in INITIATION.base_covariates
    for leak in ("days_to_treatment", "discontinued_180d", "persistent_180d", "adherence_rate"):
        assert leak not in INITIATION.base_covariates
```

- [ ] **Step 2: Run it — expect ImportError/FAIL**

Run: `cd /home/enunez/Projects/e2i_causal_analytics/.claude/worktrees/gse-p1 && PYTHONPATH=$PWD python -m pytest tests/unit/test_mlops/test_gold_standard_eval/test_cohort_spec.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `cohort_spec.py`**

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class CohortSpec:
    name: str
    target: str          # ml_experiments.prediction_target
    brand: str           # patient_journeys.brand partition
    label_column: str    # ground-truth column in patient_journeys
    grain: str           # "patient" | "hcp"
    base_covariates: tuple[str, ...]  # leakage-safe seed features (from _PJ_COHORTS)

# Seed covariates copied from cohort_resolution._PJ_COHORTS['initiation'] (codebase intent),
# pruned of any post-anchor/outcome columns. Final feature set is finalized empirically in Task 3.
INITIATION = CohortSpec(
    name="initiation",
    target="csu_treatment_initiation",
    brand="Remibrutinib",
    label_column="treatment_initiated",
    grain="patient",
    base_covariates=("disease_severity", "academic_hcp", "geographic_region",
                     "age_group", "gender", "insurance_type", "risk_score"),
)
```

(Worker: open `cohort_resolution.py:259` and reconcile `base_covariates` to the actual `initiation` entry; the list above is the seed, not gospel.)

- [ ] **Step 4: Run test — expect PASS**

Run: same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops/gold_standard_eval/cohort_spec.py tests/unit/test_mlops/test_gold_standard_eval/test_cohort_spec.py
git commit -m "feat(gse): initiation CohortSpec grounded in _PJ_COHORTS"
```

---

## Task 2: FeatureBuilder — leakage-safe gold-standard features (real DB)

**Files:** Create `src/mlops/gold_standard_eval/feature_builder.py`; Test `tests/unit/test_mlops/test_gold_standard_eval/test_feature_builder.py` + assertions in the Task 8 integration test.

**Cheap-disproof framing:** Assumption = "the cohort's covariates are derivable for holdout patients from `patient_journeys` (+ the 8 patient `feature_values`) without label leakage." Falsify by (a) a leakage scan and (b) coverage count on real holdout patients.

- [ ] **Step 1: Write the failing unit test (no-leakage + column completeness, with an injected fake frame)**

```python
import pandas as pd
from src.mlops.gold_standard_eval.cohort_spec import INITIATION
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder, LEAKAGE_DENYLIST

def test_feature_builder_is_leakage_safe_and_complete():
    fb = FeatureBuilder(INITIATION)
    raw = pd.DataFrame({
        "patient_id": ["scvpt_1", "scvpt_2"],
        "treatment_initiated": [1, 0],
        "days_to_treatment": [10, None],      # post-anchor → must be dropped
        "disease_severity": ["high", "low"],
        "age_group": ["45-54", "65-74"],
        "risk_score": [0.7, 0.3],
    })
    X, y = fb.build_from_frame(raw)
    assert list(y) == [1, 0]
    assert "treatment_initiated" not in X.columns
    for col in LEAKAGE_DENYLIST:
        assert col not in X.columns
    assert not X.isnull().any().any()        # imputed, no NaNs reach the model
    assert len(fb.feature_columns) == X.shape[1]
```

- [ ] **Step 2: Run — expect FAIL**

Run: `PYTHONPATH=$PWD python -m pytest tests/unit/test_mlops/test_gold_standard_eval/test_feature_builder.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `feature_builder.py`**

```python
from __future__ import annotations
import pandas as pd

# Columns knowable only AFTER the initiation decision → never features (anti-leakage).
LEAKAGE_DENYLIST = (
    "treatment_initiated", "days_to_treatment", "discontinued_180d", "persistent_180d",
    "adherence_rate", "refill_count", "gap_days", "is_churned", "treatment_arm",
    "outcome_probability", "treatment_propensity",  # outcome-derived feature_values (verify w/ feature_contract)
)

class FeatureBuilder:
    def __init__(self, spec):
        self.spec = spec
        self.feature_columns: list[str] = []

    def build_from_frame(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        y = raw[self.spec.label_column].astype(int)
        drop = set(LEAKAGE_DENYLIST) | {"patient_id", "patient_hash", "data_split", "split_config_id"}
        feats = raw.drop(columns=[c for c in raw.columns if c in drop], errors="ignore")
        # categorical → one-hot; numeric → median-impute + missingness flag
        feats = self._encode(feats)
        self.feature_columns = list(feats.columns)
        return feats, y

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in df.columns:
            s = df[col]
            if s.dtype == object or str(s.dtype) == "category":
                for val, dummy in pd.get_dummies(s, prefix=col, dummy_na=True).items():
                    out[val] = dummy.astype(float)
            else:
                out[f"{col}__isna"] = s.isnull().astype(float)
                out[col] = s.fillna(s.median()).astype(float)
        return pd.DataFrame(out, index=df.index)

    async def build_for_split(self, db, split: str, *, before_month=None):
        """Load patient_journeys rows for the brand+split (include_synthetic=True),
        optionally journey_start_date < before_month, → build_from_frame.
        Uses the app async client; SELECT-only."""
        # implemented against the async Supabase client; see Task 8 for the live shape.
        ...
```

(Worker: `_encode` here is the minimal contract; finalize feature scope in Task 3. The `build_for_split` DB body is filled in Task 4 once the live column set is confirmed against `patient_journeys`.)

- [ ] **Step 4: Run unit test — expect PASS**

Run: same pytest command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlops/gold_standard_eval/feature_builder.py tests/unit/test_mlops/test_gold_standard_eval/test_feature_builder.py
git commit -m "feat(gse): leakage-safe FeatureBuilder (frame contract + denylist)"
```

---

## Task 3: EXPERIMENT — finalize the initiation feature set (cheap disproof, real data)

**This is an experiment task, not a code task. Use `ralph-wiggum:ralph-loop` + `codex:codex-rescue` to converge.**

- [ ] **Step 1: State the assumption + falsifier**

Assumption: "a leakage-safe feature set drawn from `patient_journeys` (+ patient `feature_values`) trains an initiation model with **holdout AUC materially > 0.5** (real signal)." Falsifier: AUC ≤ ~0.55 on the real holdout, or any feature flagged leaky by `feature_contract`.

- [ ] **Step 2: Run the leakage scan**

For each candidate column, check `src/data/feature_contract.py` `knowable_at` semantics; drop any not knowable pre-initiation. Record the kept/dropped lists in the experiment log.

- [ ] **Step 3: Memory check, then run the experiment script (real DB, one fit)**

Run: `free -h` (need ≥2 GiB). Then a throwaway script: load `data_split='train'` (brand=Remibrutinib, `include_synthetic=True`) via FeatureBuilder, fit `LogisticRegression(class_weight='balanced')`, score on `data_split='holdout'`. Print AUC + n + positive rate.
Expected (assumption survives): holdout AUC > 0.6 (record the actual number — do not assume).

- [ ] **Step 4: Decide + record**

If AUC > ~0.6 → lock the feature list into `FeatureBuilder` (a `KEEP` constant) + commit the experiment log to `docs/superpowers/plans/experiments/2026-06-14-initiation-features.md`. If not → loop (codex-rescue: revisit features/encoding) before any downstream task.

- [ ] **Step 5: Commit**

```bash
git add src/mlops/gold_standard_eval/feature_builder.py docs/superpowers/plans/experiments/2026-06-14-initiation-features.md
git commit -m "exp(gse): lock initiation feature set (holdout AUC=<measured>)"
```

---

## Task 4: FeatureBuilder live DB loader (`build_for_split`) — real DB

**Files:** Modify `feature_builder.py`; Test: covered by Task 8 integration (real DB).

- [ ] **Step 1: Confirm the live column set + patient-id format**

Run: `docker exec supabase-db psql -U postgres -d postgres -At -c "SELECT string_agg(column_name,',') FROM information_schema.columns WHERE table_name='patient_journeys'"`
Expected: includes the locked features + `data_split`, `brand`, `journey_start_date`, `patient_id` (`scvpt_…`).

- [ ] **Step 2: Implement `build_for_split` against the async client**

Query `patient_journeys` filtered by `brand=spec.brand`, `data_split=split`, `is_synthetic=True` (opt-in — the provenance filter default-excludes synthetic), optional `journey_start_date < before_month`; paginate (PostGREST 1000-row cap — reuse the project's pagination helper); → `build_from_frame`. Join the 8 patient `feature_values` via `get_historical_features` where present (impute+flag the rest).

- [ ] **Step 3: Smoke it (real DB, memory-guarded)**

Run: `free -h` then a one-liner that calls `build_for_split(db,'holdout')` and prints `X.shape`, `y.mean()`.
Expected: ~15,211 rows total across the brand splits; holdout brand subset ~5k; `y.mean()` ≈ 0.35; **no NaNs**.

- [ ] **Step 4: Commit**

```bash
git add src/mlops/gold_standard_eval/feature_builder.py
git commit -m "feat(gse): FeatureBuilder live patient_journeys loader (synthetic opt-in, paged)"
```

---

## Task 5: Scorer — page-aligned metric names

**Files:** Create `scorer.py`; Test `tests/unit/test_mlops/test_gold_standard_eval/test_scorer.py`

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.mlops.gold_standard_eval.scorer import score, METRIC_NAMES
def test_score_emits_page_aligned_names_and_real_values():
    y = np.array([0,0,1,1]); s = np.array([0.1,0.4,0.6,0.9])
    out = score(y, s)
    assert set(out) == set(METRIC_NAMES) == {"accuracy","precision","recall","f1","auc_roc"}
    assert abs(out["auc_roc"] - 1.0) < 1e-9      # perfectly separable
    assert all(0.0 <= v <= 1.0 for v in out.values())
```

- [ ] **Step 2: Run — FAIL.** `PYTHONPATH=$PWD python -m pytest .../test_scorer.py -q`

- [ ] **Step 3: Implement `scorer.py`**

```python
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "auc_roc")
def score(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
    }
```

- [ ] **Step 4: Run — PASS.** **Step 5: Commit** `feat(gse): Scorer with page-aligned metric names`.

---

## Task 6: Writer extension — `source` + `measured_at` passthrough (additive, no migration)

**Files:** Modify `src/repositories/drift_monitoring.py`; Test `tests/unit/test_repositories/test_drift_monitoring_source.py`

- [ ] **Step 1: Failing test (stub the async client boundary only — assert the ROW we'd write)**

```python
import datetime as dt
from src.repositories.drift_monitoring import PerformanceMetricRecord
def test_record_carries_source_and_measured_at():
    rec = PerformanceMetricRecord(model_id="m1", metric_name="auc_roc", metric_value=0.83,
        measured_at=dt.datetime(2026,5,1,tzinfo=dt.timezone.utc), source="backtest_wf",
        metadata={"split_version": "e2i_pilot_v3"})
    row = rec.to_db_row()
    assert row["source"] == "backtest_wf"
    assert row["measured_at"].startswith("2026-05-01")
    assert row["metadata"]["split_version"] == "e2i_pilot_v3"
```

- [ ] **Step 2: Run — FAIL** (`to_db_row` omits `source`).

- [ ] **Step 3: Implement (add `source` field + emit it)**

In `PerformanceMetricRecord`: add `source: Optional[str] = None`. In `to_db_row` add (only when set, so existing callers keep the DB default `'mlflow'`):

```python
        row = {
            "id": self.id, "model_id": self.model_id,
            "metric_name": self.metric_name, "metric_value": self.metric_value,
            "sample_size": self.sample_size,
            "measurement_window_start": _iso(self.measurement_window_start),
            "measurement_window_end": _iso(self.measurement_window_end),
            "measured_at": _iso(self.measured_at), "metadata": meta,
        }
        if self.source is not None:
            row["source"] = self.source
        return row
```

Extend `record_metrics(self, model_version, metrics, sample_size, window_start, window_end, *, measured_at=None, source=None)` → pass `measured_at` (default keeps `now()`) and `source` into each `PerformanceMetricRecord`. Add:

```python
    async def delete_metrics(self, model_id: str, source: str, split_version: str | None = None) -> int:
        if not self.client: return 0
        q = self.client.table(self.table_name).delete().eq("model_id", model_id).eq("source", source)
        if split_version is not None:
            q = q.eq("metadata->>split_version", split_version)
        res = await q.execute()
        return len(res.data or [])
```

- [ ] **Step 4: Run — PASS.** **Step 5: Commit** `feat(metrics): record_metrics accepts source+measured_at; add delete_metrics (idempotency)`.

---

## Task 7: MetricRecorder — idempotent delete-then-insert

**Files:** Create `src/mlops/gold_standard_eval/recorder.py`; Test `tests/unit/test_mlops/test_gold_standard_eval/test_recorder.py`

- [ ] **Step 1: Failing test** — given a fake repo capturing calls, `MetricRecorder.record_run(model_version, points, source, split_version)` calls `delete_metrics(model_id, source, split_version)` exactly once **before** any insert, then one `record_metrics` per `(month, metrics)` with `measured_at=month, source=source`.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `MetricRecorder` wrapping `PerformanceMetricRepository`: resolve model_id once, `await repo.delete_metrics(...)`, then loop `await repo.record_metrics(model_version, metrics, n, month_start, month_end, measured_at=month, source=source)`.
- [ ] **Step 4: Run — PASS.** **Step 5: Commit** `feat(gse): idempotent MetricRecorder (delete-by-source then insert)`.

---

## Task 8: WalkForwardRunner — expanding-origin out-of-sample (+ EXPERIMENT: window mode)

**Files:** Create `walk_forward.py`; Test `tests/unit/test_mlops/test_gold_standard_eval/test_walk_forward.py`

- [ ] **Step 1: Failing unit test (synthetic monotone data, injected frame)** — `WalkForwardRunner(min_train_n=50, n_min=30).run(frame_by_month)` returns one result per qualifying month, each strictly **out-of-sample** (assert the runner never includes month M's rows in month M's training set), and **skips** months with `n < n_min` or `train_n < min_train_n` (assert skipped months are logged, not emitted).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** expanding-origin loop: for each ordered month M, `train = rows[start < M]`, `eval = rows[month == M]`; if guards pass, fit (Task 9 trainer), `score()` (Task 5), yield `(M, metrics, len(eval))`.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: EXPERIMENT (cheap disproof, real data, memory-guarded)** — Assumption: "expanding-origin gives a stable trend (no degenerate months)." Run the real walk-forward over the brand timeline; if early months are too thin/unstable (AUC swings > ~0.2 month-to-month from sample noise), compare a **rolling 3-month** window and pick the more stable; record both in the experiment log. Use ralph-loop + codex to converge.
- [ ] **Step 6: Commit** `feat(gse): WalkForwardRunner (expanding-origin, guarded) + window experiment`.

---

## Task 9: CohortDeployer — train on GOLD-STANDARD features + register loadable champion (real DB)

**Files:** Create `cohort_deployer.py`; refactor reusable helpers out of `src/mlops/prediction_synthesizer_deploy.py`; Test `tests/unit/test_mlops/test_gold_standard_eval/test_cohort_deployer.py` + Task 11 integration.

**Key change vs the existing deployer:** training data = `FeatureBuilder.build_for_split('train')` (gold-standard), **not** `generate_scenario(...)`. Manifest is **merged**, not overwritten (the shared `deployment_manifest.json` already holds the CSU go-live models).

- [ ] **Step 1: Failing unit test** — `train_cohort_model(spec, X, y)` returns a fitted estimator whose `feature_names_in_` == `FeatureBuilder.feature_columns`; `merge_manifest(existing, new)` preserves prior models and adds the new one (assert CSU keys survive).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — `train_cohort_model` (calibrated `LogisticRegression`, fit on the named-column gold-standard DataFrame so `feature_names_in_` serializes); `serialize_and_write_manifest` reused from the refactored module but **reading-merging** the existing manifest; `register_cohort_models` reusing `register_deployed_models`/`_get_or_create_experiment` parametrized by `CohortSpec` (target/brand/experiment), writing `stage='production'`, `is_synthetic=False`, real `artifact_path`, into `data/ml_artifacts/<target>/`.
- [ ] **Step 4: Run — PASS.** **Step 5: Commit** `feat(gse): CohortDeployer trains on gold-standard features + merges manifest`.

---

## Task 10: Read-path + frontend (raise caps, align names, default model)

**Files:** Modify `src/api/routes/monitoring.py:1118`, `src/services/performance_tracking.py:60-67,191`, `frontend/src/pages/TimeSeries.tsx:65,70,78`.

- [ ] **Step 1: Failing test (route cap)** — `tests/unit/test_api/test_monitoring_trend_window.py`: assert the `get_performance_trend` route accepts `days=1825` (5y) without a validation error. (FastAPI `Query(le=...)` — assert via the OpenAPI schema or a direct call.)
- [ ] **Step 2: Run — FAIL** (current `le=90`).
- [ ] **Step 3: Implement** — `monitoring.py:1118` → `days: int = Query(default=365, ge=1, le=1825, ...)`; `PerformanceTrackingConfig.trend_window_days = 365`; in `performance_tracking._calculate_metrics` rename emitted `f1_score` → `f1` (align to the page; update `tests/unit/test_services/test_performance_tracking.py` expectations in the same commit); `TimeSeries.tsx` add a `{label:'5 Years', value:'1825d', days:1825}` `TIME_RANGES` entry and set `DEFAULT_MODEL_ID = 'csu_treatment_initiation_lr_balanced_v1'` (the resolvable initiation champion).
- [ ] **Step 4: Run — PASS** (route test + the updated performance_tracking unit test).
- [ ] **Step 5: Commit** `feat(monitoring): widen trend window; align f1; default time-series to initiation champion`.

---

## Task 11: Integration — initiation end-to-end on the REAL DB (no mocks)

**Files:** `tests/integration/test_gold_standard_initiation_eval.py`; CLI `src/mlops/gold_standard_eval/run_initiation_eval.py`

- [ ] **Step 1: Write the CLI** `run_initiation_eval.py`: `deploy champion (Task 9) → WalkForwardRunner over train→holdout timeline → MetricRecorder(source='backtest_wf') → holdout headline eval (champion on test+holdout, source='holdout')`. Memory-guarded (free -h before training).
- [ ] **Step 2: Write the failing integration test** (gated like the repo's other real-DB tests via `conftest.py` `SERVICES_AVAILABLE`):

```python
import pytest
pytestmark = pytest.mark.skipif(not SERVICES_AVAILABLE.get("supabase"), reason="real DB required")
@pytest.mark.asyncio
async def test_initiation_eval_records_real_trend(async_db):
    from src.mlops.gold_standard_eval.run_initiation_eval import run
    report = await run()                       # deploys + evals, no mocks
    assert report["champion_registered"]
    rows = await async_db.table("ml_performance_metrics").select("*") \
        .eq("source","backtest_wf").eq("metric_name","auc_roc").execute()
    months = sorted({r["measured_at"][:7] for r in rows.data})
    assert len(months) >= 3                    # real multi-month trend
    assert all(0.5 <= r["metric_value"] <= 1.0 for r in rows.data)  # real, sane AUCs
    # idempotency: a second run does not duplicate
    before = len(rows.data); await run()
    rows2 = await async_db.table("ml_performance_metrics").select("id") \
        .eq("source","backtest_wf").eq("metric_name","auc_roc").execute()
    assert len(rows2.data) == before
```

- [ ] **Step 3: Run it (memory-guarded, serial)** — `free -h` ≥ 2 GiB, then `PYTHONPATH=$PWD python -m pytest tests/integration/test_gold_standard_initiation_eval.py -q`. Converge with ralph-loop + codex if red.
- [ ] **Step 4: Faithful endpoint check** — after the run, call `get_performance_trend` for the champion (`days=1825`, `metric_name=auc_roc`) and assert non-empty multi-month `history`.
- [ ] **Step 5: Commit** `test(gse): real-DB initiation end-to-end (walk-forward + holdout + idempotency)`.

---

## Task 12: Live render verification + activation

- [ ] **Step 1:** Run the deploy CLI on the box (writes champion artifact to the `e2i_ml_artifacts` volume + merges manifest + registers). `free -h` first.
- [ ] **Step 2:** Faithful frontend check (per the project's headless-Playwright/reviewer-JWT pattern) that `/time-series` Model-performance mode renders a real multi-month trend + populated "Trend Summary" for the default initiation model.
- [ ] **Step 3:** Record the live evidence (rootChildren, trend point count, AUC range) in the PR body. Commit any doc.

---

## Task 13: FINAL CI BATCH (once, at the very end)

- [ ] **Step 1:** Merge all `gse-p1*` task worktrees back to `feat/claims-gold-standard-model-eval`; reap worktrees.
- [ ] **Step 2:** `free -h`; run the **targeted** suites locally once (`tests/unit/test_mlops/test_gold_standard_eval`, `test_repositories/test_drift_monitoring_source.py`, `test_services/test_performance_tracking*`, the integration test). Do NOT run whole-tree pytest/mypy locally (CI arbiter).
- [ ] **Step 3:** Push the feature branch **once** and open the PR → this is the single CI batch. Address CI as one pass (ralph-loop + codex if red). Hold merge for user (prod-affecting + deploy.yml).

---

## Self-Review (against the v2 spec)

- **Spec coverage:** FeatureBuilder→§4.1 (T2/T3/T4); Trainer/Registrar→§4.2 (T9); walk-forward→§4.3 (T8); holdout headline→§4.4 (T11 CLI); MetricRecorder/writer→§4.5 (T6/T7); read-path+frontend→§4.6 (T10); ongoing eval §4.7 = **P4, explicitly deferred** (not P1); HCP-adoption §4.8 = **P3, deferred**; honesty guardrails §5 → leakage denylist (T2/T3), no-mock integration (T11), n_min/min_train_n (T8), include_synthetic (T4). **No P1 requirement is unmapped.**
- **Placeholder scan:** the only `...` bodies (FeatureBuilder.build_for_split, deployer internals) are filled by their own later tasks (T4, T9) with explicit DB/contract steps — not vague "implement later." Experiment tasks (T3, T8.5) are concrete (assumption + falsifier + command + decision), not placeholders.
- **Type consistency:** `METRIC_NAMES`/recorded names = `{accuracy,precision,recall,f1,auc_roc}` everywhere (Scorer, writer, route, frontend `METRIC_OPTIONS`); `CohortSpec` fields consistent across T1/T2/T9; `source='backtest_wf'|'holdout'` consistent T6/T7/T11; `delete_metrics`/`record_metrics(...,measured_at,source)` signatures match T6↔T7.

## Deferred (explicitly, so nothing is fished-for later)
- **P2:** persistence cohort (+ discontinuation as its `1−persistent` complement view) — reruns T1–T11 with a new `CohortSpec`; the deployer/runner are already cohort-parametrized.
- **P3:** HCP-adoption — `hcp_profiles` grain, requires designating a holdout split there + an HCP FeatureBuilder variant.
- **P4:** ongoing eval — event-driven on champion promotion + monthly dedup backstop (`source='scheduled'|'event'`).
