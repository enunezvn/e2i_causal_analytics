# Gold-Standard Eval P2 — Persistence + Discontinuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train + register loadable gold-standard models for the **persistence** cohort (`persistent_180d`) and its **discontinuation** complement (`discontinued_180d`), and record real walk-forward + holdout metrics into `ml_performance_metrics` so the Model-Perf / Time-Series / Monitoring pages render real trends for both cohorts.

**Architecture:** Reuse the P1 machinery (`FeatureBuilder`, `WalkForwardRunner`, `scorer`, `recorder`, `cohort_deployer`) — all already `CohortSpec`-parametrized. P2 adds two `CohortSpec`s, makes the brand filter optional (all-brands persistence), generalizes the experiment/target resolver beyond the initiation target, and adds one end-to-end run script that processes BOTH cohorts from a single loaded frame.

**Tech Stack:** Python 3.12, scikit-learn (calibrated LogisticRegression), pandas/numpy, async supabase-py against the self-hosted docker Supabase DB.

**Data facts established by read-only prod probes (2026-06-14) — DO NOT re-derive, but DO re-verify if anything looks off:**
- `persistent_180d` / `discontinued_180d` are `smallint` (0/1); populated for **synthetic rows only** (real rows are NULL → dropped). 25,000 synthetic rows.
- Positive rate ~**0.55** persistent (well-balanced). `discontinued_180d ≡ 1 − persistent_180d` **exactly** (cross-tab has only (0,1) and (1,0); zero (0,0)/(1,1)).
- Brand-agnostic: pos rate Remibrutinib 0.542 / Fabhalta 0.552 / Kisqali 0.545 → **train all-brands** (no brand omitted).
- ~37-month span (2023-06 → 2026-06): train 22mo / val 9mo / test 6mo / holdout 3mo (newest, 15,211 rows). Walk-forward ignores `data_split` and re-windows by month → full monthly trend; holdout headline = newest-3mo.
- `ml_experiments` targets: `csu_treatment_initiation` (P1) and `pnh_persistence` (120 exps, 12+ "production" champions — **all `is_synthetic=true`, all artifact-less** → no serving collision). **No discontinuation target exists** (created on first registration).
- Base covariates (from `cohort_resolution._PJ_COHORTS["persistence"]`): `disease_severity`, `academic_hcp`, `geographic_region` (same base-3 as initiation).

**Design decisions (reasoned):**
- **Two separately-registered models**, one per cohort (`pnh_persistence` + new `pnh_discontinuation`), each genuinely trained on its own label via the existing pipeline. We do NOT exploit the complement in code (it's a validation check: the two AUCs should match) — separate honest models are simpler and avoid inversion bugs.
- **`stage='staging'`** for both (parity with P1; `register_cohort_model` already hard-refuses `production`).
- **Feature set decided by a measured experiment** (Task 3), exactly like P1.

---

## File Structure

- Modify `src/mlops/gold_standard_eval/cohort_spec.py` — add `PERSISTENCE`, `DISCONTINUATION`; allow `brand: str | None`.
- Modify `src/mlops/gold_standard_eval/feature_builder.py` — `load_frame` skips the brand filter when `spec.brand is None`.
- Modify `src/mlops/prediction_synthesizer_deploy.py` — `_get_or_create_experiment` accepts a `prediction_target` arg (default = module `PREDICTION_TARGET`, back-compat).
- Modify `src/mlops/gold_standard_eval/cohort_deployer.py` — `_resolve_goldstd_experiment` uses `spec.target` (drop the initiation-only refusal); per-cohort model/experiment names.
- Create `src/mlops/gold_standard_eval/run_persistence_eval.py` — end-to-end run for BOTH cohorts from one loaded frame.
- Create `docs/superpowers/plans/experiments/2026-06-14-persistence-features.md` — the Task-3 measured feature-lock log.
- Tests: extend `tests/unit/test_mlops/test_gold_standard_eval/` (cohort_spec, feature_builder all-brands, cohort_deployer target-generalization) + `tests/integration/test_gold_standard_persistence_eval.py`.

---

## Task 1: PERSISTENCE + DISCONTINUATION CohortSpecs

**Files:**
- Modify: `src/mlops/gold_standard_eval/cohort_spec.py`
- Test: `tests/unit/test_mlops/test_gold_standard_eval/test_cohort_spec.py`

- [ ] **Step 1: Write the failing test**

```python
def test_persistence_and_discontinuation_specs():
    from src.mlops.gold_standard_eval.cohort_spec import PERSISTENCE, DISCONTINUATION
    # Persistence: all-brands (brand=None), label persistent_180d, base-3 covariates.
    assert PERSISTENCE.target == "pnh_persistence"
    assert PERSISTENCE.label_column == "persistent_180d"
    assert PERSISTENCE.brand is None
    assert PERSISTENCE.grain == "patient"
    assert PERSISTENCE.base_covariates == ("disease_severity", "academic_hcp", "geographic_region")
    # Discontinuation: separate target + label, same all-brands base-3.
    assert DISCONTINUATION.target == "pnh_discontinuation"
    assert DISCONTINUATION.label_column == "discontinued_180d"
    assert DISCONTINUATION.brand is None
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/unit/test_mlops/test_gold_standard_eval/test_cohort_spec.py::test_persistence_and_discontinuation_specs -v` → FAIL (ImportError).

- [ ] **Step 3: Implement** — in `cohort_spec.py`, change `brand: str` to `brand: str | None` in the dataclass (update the comment to note `None` = all brands), then add:

```python
# Grounded in _PJ_COHORTS["persistence"]/["discontinuation"] in
# src/services/cohort_resolution.py. Both labels are 180-day post-index outcomes
# (knowable only AFTER the index decision) → each is the OTHER cohort's leakage
# column (already in feature_builder.LEAKAGE_DENYLIST). brand=None: persistence is
# brand-agnostic in the synthetic DGP (pos rate ~0.55 across all 3 brands), so we
# train on ALL brands; discontinued_180d == 1 - persistent_180d exactly in-data.
PERSISTENCE = CohortSpec(
    name="persistence",
    target="pnh_persistence",
    brand=None,
    label_column="persistent_180d",
    grain="patient",
    base_covariates=("disease_severity", "academic_hcp", "geographic_region"),
)

DISCONTINUATION = CohortSpec(
    name="discontinuation",
    target="pnh_discontinuation",
    brand=None,
    label_column="discontinued_180d",
    grain="patient",
    base_covariates=("disease_severity", "academic_hcp", "geographic_region"),
)
```

- [ ] **Step 4: Run test to verify it passes.**

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(gse-p2): add PERSISTENCE + DISCONTINUATION CohortSpecs (all-brands)"`

---

## Task 2: FeatureBuilder all-brands loading

**Files:**
- Modify: `src/mlops/gold_standard_eval/feature_builder.py` (`load_frame`, ~line 274-283)
- Test: `tests/unit/test_mlops/test_gold_standard_eval/test_feature_builder.py`

- [ ] **Step 1: Write the failing test** (fake async client capturing query filters):

```python
def test_load_frame_omits_brand_filter_when_brand_none(monkeypatch):
    import asyncio
    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder
    from src.mlops.gold_standard_eval.cohort_spec import PERSISTENCE

    calls = {"eq": []}

    class _Q:
        def select(self, *a, **k): return self
        def eq(self, col, val): calls["eq"].append(col); return self
        def in_(self, *a, **k): return self
        def lt(self, *a, **k): return self
        def order(self, *a, **k): return self
        def range(self, *a, **k): return self
        async def execute(self):
            class R: data = []
            return R()
    class _DB:
        def table(self, *a, **k): return _Q()

    fb = FeatureBuilder(PERSISTENCE)
    asyncio.run(fb.load_frame(_DB()))
    assert "brand" not in calls["eq"]          # all-brands: no brand filter
    assert "is_synthetic" in calls["eq"]       # synthetic provenance still enforced
```

- [ ] **Step 2: Run test → FAIL** (current code always calls `.eq("brand", ...)`).

- [ ] **Step 3: Implement** — in `load_frame`, replace the unconditional `.eq("brand", self.spec.brand)` chain with a conditional. Build the base query, then:

```python
            query = (
                db.table("patient_journeys")
                .select(select_expr)
                .eq("is_synthetic", True)
            )
            if self.spec.brand is not None:
                query = query.eq("brand", self.spec.brand)
```

  Also update the runaway-guard log (line ~309) that references `self.spec.brand` — it already prints the value, so `None` is fine.

- [ ] **Step 4: Run test → PASS.** Also re-run the existing initiation load_frame test to confirm brand="Remibrutinib" still filters.

- [ ] **Step 5: Commit** — `feat(gse-p2): load_frame supports all-brands (brand=None)`

---

## Task 3: EXPERIMENT — lock the persistence feature set (measured)

**Files:**
- Create: `docs/superpowers/plans/experiments/2026-06-14-persistence-features.md`
- (No production code unless the experiment says base-3 is NOT best.)

- [ ] **Step 1:** Write a throwaway probe script (NOT committed) that, against the docker DB, loads all-brands synthetic persistence rows, fits `LogisticRegression(class_weight='balanced', max_iter=1000)` on `train`+`validation`, scores AUC on `holdout` for three candidate feature tiers:
  - **A** base-3 covariates only,
  - **B** A + brand (one-hot) — tests brand heterogeneity,
  - **C** B + leakage-safe `patient_journeys` extras present in the frame (exclude every column in `LEAKAGE_DENYLIST`).

- [ ] **Step 2:** Run it. Record the three measured holdout AUCs in the experiment log with n_train / n_holdout / positive rate.

- [ ] **Step 3:** Lock `KEEP_COLUMNS` for persistence = the winning tier. If A wins (expected, mirroring P1), persistence reuses the existing module `KEEP_COLUMNS` (base-3) and NO code change is needed — pass `keep_columns=None` so `FeatureBuilder` uses the module default. If B/C wins, set `keep_columns=(...)` explicitly in the run script (Task 4) and document why. **The data decides — do not assume A.**

- [ ] **Step 4: Commit** the experiment log — `docs(gse-p2): persistence feature-lock experiment (measured holdout AUC)`

---

## Task 4: Generalize the experiment/target resolver beyond initiation

**Files:**
- Modify: `src/mlops/prediction_synthesizer_deploy.py` (`_get_or_create_experiment`)
- Modify: `src/mlops/gold_standard_eval/cohort_deployer.py` (`_resolve_goldstd_experiment`)
- Test: `tests/unit/test_mlops/test_gold_standard_eval/test_cohort_deployer.py`

- [ ] **Step 1: Write the failing test** — `register_cohort_model` for a persistence spec resolves/creates an experiment under `pnh_persistence` (not the initiation target). Use a fake client that records the `prediction_target` written on experiment-create and the `stage`/`model_name` on the registry row; assert `prediction_target == "pnh_persistence"`, `stage == "staging"`, and that `stage='production'` still raises.

- [ ] **Step 2: Run test → FAIL** (current `_resolve_goldstd_experiment` raises for any target ≠ `csu_treatment_initiation`).

- [ ] **Step 3: Implement.**
  - In `prediction_synthesizer_deploy._get_or_create_experiment`, add a keyword arg `prediction_target: str | None = None` and use `prediction_target or PREDICTION_TARGET` everywhere the function currently reads the module constant (match-on-create AND the inserted row). Back-compat: existing callers omit it and keep the initiation target.
  - In `cohort_deployer._resolve_goldstd_experiment`, DELETE the `target != _deploy.PREDICTION_TARGET` refusal; instead read `target = spec.target` (fail closed only if `target` is falsy) and pass `prediction_target=target` + a spec-derived `description` to `_get_or_create_experiment`.

- [ ] **Step 4: Run test → PASS.** Re-run the initiation cohort_deployer tests (back-compat).

- [ ] **Step 5: Commit** — `feat(gse-p2): generalize gold-standard experiment resolver to any cohort target`

---

## Task 5: run_persistence_eval — end-to-end, BOTH cohorts, real DB

**Files:**
- Create: `src/mlops/gold_standard_eval/run_persistence_eval.py`
- Test: `tests/integration/test_gold_standard_persistence_eval.py` (gated `E2I_DB_INTEGRATION=1`)

- [ ] **Step 1: Write the failing integration test** (skip unless `E2I_DB_INTEGRATION=1`): call `run()`; assert it returns a dict with non-zero `persistence` and `discontinuation` sub-results, each having `backtest_points > 0`, a finite `holdout_auc`, and that a second `run()` (idempotency) leaves the recorded row counts unchanged.

- [ ] **Step 2: Run → FAIL** (module doesn't exist).

- [ ] **Step 3: Implement** — mirror `run_initiation_eval.run()` (read it for the exact train→holdout-headline→clear-dependent-metrics→register→walk-forward→record sequence and the FK-ordering re-run-safety block). Differences:
  - Per-cohort constants: `PERSISTENCE_MODEL_NAME="pnh_persistence_goldstd_lr_v1"`, `DISCONTINUATION_MODEL_NAME="pnh_discontinuation_goldstd_lr_v1"`, distinct experiment names `persistence_goldstd_eval_v1` / `discontinuation_goldstd_eval_v1`.
  - Load the all-brands frame ONCE with `FeatureBuilder(PERSISTENCE).load_frame(client, splits=None)` (discontinuation reuses the same rows; its label column is also present).
  - For EACH cohort spec in `(PERSISTENCE, DISCONTINUATION)`: fit a fresh `FeatureBuilder(spec, keep_columns=<locked in Task 3>)`, `train_cohort_model`, holdout-headline `score`, clear-dependent-metrics for that model_name, `serialize_model` + `register_cohort_model(..., model_name=<cohort name>, experiment_name=<cohort exp>)` at staging, `WalkForwardRunner(spec).run(frame)`, `recorder.record_run(model_handle, points, source='backtest_wf')` + the holdout point (`source='holdout'`).
  - Empty-frame guard raises (no fabrication), matching run_initiation_eval.
  - `argparse` CLI entrypoint mirroring run_initiation_eval; runnable as `E2I_DB_INTEGRATION=1 python -m src.mlops.gold_standard_eval.run_persistence_eval`.
  - **Validation assertion (the complement check as code):** after both runs, log persistence vs discontinuation holdout AUC; they should be within a small tolerance (mirror models). Log a WARNING if they diverge > 0.05 (signals a data/label problem) — do not fail the run.

- [ ] **Step 4: Run the integration test on the box** (`E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD pytest tests/integration/test_gold_standard_persistence_eval.py -v -n0`) → PASS.

- [ ] **Step 5: Commit** — `feat(gse-p2): run_persistence_eval end-to-end (persistence + discontinuation), real DB`

---

## Task 6: Live activation + render verification (prod)

**Files:** none (operational), but capture results in the experiment log.

- [ ] **Step 1:** Run `E2I_DB_INTEGRATION=1 python -m src.mlops.gold_standard_eval.run_persistence_eval` against the prod docker DB (the box IS prod). Confirm rows land in `ml_performance_metrics` for both model handles (psql count by `source` + `model_id`).
- [ ] **Step 2:** Hit the deployed trend endpoint for both handles (`/api/monitoring/performance/{handle}/trend?days=1825`) with the reviewer JWT; assert a multi-point series returns for each.
- [ ] **Step 3:** Headless-Playwright render check of `/time-series` entering each model handle → assert the Trend Summary card + KPI tiles populate (mirror the P1 verification how-to).
- [ ] **Step 4:** Record measured AUCs + point counts in the experiment log. Commit the log update.

> NOTE: requires the P2 code to be deployed. If P2 is merged into an atomic batch with deploy disabled, run Step 1 (DB populate) locally against prod DB after merge, and do Steps 2-3 after the batch deploy lands.

---

## Task 7: FINAL CI batch (once, at end)

- [ ] **Step 1:** Run the CI-faithful local gates from the worktree root BEFORE pushing: `ruff check src/ tests/` AND `ruff format --check src/ tests/` (ruff 0.14.10), and **scoped** `mypy --config-file pyproject.toml <changed src files>` (memory-safe; the gate is `src/`-only + ceiling-count). Fix locally to zero. Run the affected unit tests `-n0`.
- [ ] **Step 2:** Push the branch ONCE; let CI run the full suite. Converge on green (codex:codex-rescue if stuck).
- [ ] **Step 3:** Open the PR (`Closes` the P2 tracking item if one exists). HOLD merge + deploy for explicit user direction (deploy is currently managed by the parallel session / atomic batch).

---

## Self-Review

- **Spec coverage:** persistence model ✓ (T1-T5), discontinuation model ✓ (T1,T4,T5), all-brands ✓ (T2), measured feature lock ✓ (T3), target generalization ✓ (T4), real metrics recorded + verified ✓ (T5-T6), idempotency ✓ (T5), no serving collision ✓ (staging, T4-T5). HCP-adoption is **out of scope** (P3, different grain).
- **Type consistency:** `register_cohort_model(model_name=, experiment_name=)` already exist as kwargs; `_get_or_create_experiment(prediction_target=)` added in T4 and consumed in T5. `CohortSpec.brand: str | None` consumed by `load_frame` (T2). `keep_columns` from T3 passed to `FeatureBuilder` in T5.
- **No placeholders:** every code step shows the change or references the exact P1 source to mirror.
