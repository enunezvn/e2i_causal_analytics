# Design Spec: Digital Twin Causal Effect Engine

- **Date:** 2026-06-04
- **Status:** Approved (brainstorming) → pending implementation plan
- **Author:** enunezvn + Claude (with Codex use-case review)
- **Audit reference:** `docs/reports/digital-twin-audit-20260604.md` (finding **H5**)
- **Decision provenance:** owner confirmed H5 = REWIRE (effects must be data-derived causal, not a hardcoded heuristic); data source = synthetic-known-effect first with an RWD-ready interface; estimator architecture = Codex use-case review → **uplift-based blend** (not "most production-ready").

---

## 1. Problem & Goal

The Digital Twin is an A/B-test **pre-screening** subsystem: train ML twins of HCPs/patients/territories from historical data, generate a synthetic population, simulate an intervention's effect on it in seconds, and emit a **DEPLOY / REFINE / SKIP** recommendation plus a recommended real-experiment sample size — so unviable experiments are skipped before spending real-world resources. It is a hypothesis *pre-filter*, not a replacement for the downstream real A/B test.

**Today the headline effect is fabricated.** `SimulationEngine._calculate_individual_effect` (`src/digital_twin/simulation_engine.py:344-448`) computes the per-twin treatment effect from a hardcoded `INTERVENTION_EFFECTS` dict (`:71-100`) × deterministic multipliers + `np.random.normal` noise. The trained model (`TwinGenerator.train`) is **baseline-only by construction** (`model.predict` sets only `baseline_outcome`, `twin_generator.py:271-273`); it never reaches the effect math. The documented intended mechanism — **"Counterfactual analysis / Uplift estimation"** (`docs/Archive/mlops/phase-15-ab-testing.md:34-38`) — was **never implemented** (`docs/Archive/v4.2_IMPLEMENTATION_TODO.md` §1 "Digital Twin ML Algorithms", all unchecked).

**Goal:** Replace the heuristic with a real **uplift-based causal effect engine** that:
1. Recovers known effects on synthetic data within the documented ≤20% ATE error (`docs/Archive/digital_twin_component_update_list.md` §9.3, §12).
2. Produces **confidence-interval-based** DEPLOY/REFINE/SKIP + heterogeneity ("top responding segments") + a recommended sample size.
3. **Fails closed** — no fabricated values, no silent heuristic fallback (CLAUDE.md anti-mocking discipline).
4. Is validated synthetic-first with a **pluggable, RWD-ready** data interface.

### 1.1 Why uplift (use-case reasoning, not reuse-convenience)

A pre-screen is a **decision gate**, not a measurement instrument. Its rigor asymmetry is the opposite of the downstream real experiment: **false negatives are expensive** (a good intervention gets suppressed), **false positives are cheap** (the real A/B test finds null and stops). So it needs sensitivity + honest uncertainty, not maximum-precision multi-library inference. DoWhy refutation / multi-library selection belong **downstream** (`src/agents/causal_impact/`, on real data) — running them in the pre-screen would violate the <2s latency target (`causal_engine` pipeline timeouts are 30s/120s, `src/causal_engine/pipeline/orchestrator.py:97`) for no decision value. Uplift modeling (CausalML `UpliftRandomForestClassifier`, KL-divergence splits) directly targets treatment-effect heterogeneity and natively emits per-unit scores, ATE, CIs, ATT/ATC, AUUC/Qini and top/bottom responding segments — exactly the pre-screen's outputs, and exactly the "simulate the intervention on N twins" framing (score the fitted model over the twin population, average → ATE).

## 2. Scope

**v1 (this spec — buildable & fully testable in isolation; NO dependency on audit findings H4/H6):**
the estimator, the synthetic data provider, heterogeneity drill-down, the CI-based recommendation policy, the calibration/validation harness, provenance labeling, and the phased test suite.

**v2 (designed here, deferred):** RWD `CohortEffectDataProvider`; propensity-matched twin pairs (SMD < 0.1) in `TwinGenerator.generate_twin()`; the fidelity feedback loop (predicted vs real ATE); live-path wiring (depends on **H4** model-load + **H6** repo-client); agent-tool rewire (**H3**).

**Out of scope:** the twin *generation* model (`TwinGenerator.train` stays real sklearn baseline); the downstream real `causal_impact` run; the frontend fabrication (H1/H2, tracked separately in the audit).

## 3. Architecture & Components

Each unit has one purpose, a narrow interface, and is independently testable. New code lives under `src/digital_twin/effect/`.

### 3.1 `EffectDataProvider` (interface) — supplies the labeled training frame
```python
class TrainingFrame(BaseModel):
    df: pd.DataFrame                 # labeled rows
    treatment_var: str
    outcome_var: str
    confounders: list[str]
    effect_modifiers: list[str]
    ground_truth_ate: float | None   # known for synthetic; None for RWD

class EffectDataProvider(Protocol):
    def get_training_frame(
        self, intervention_type: str, brand: str, twin_type: str
    ) -> TrainingFrame: ...           # raises EffectDataUnavailable (fail-closed)
```
- **`SyntheticEffectDataProvider` (v1):** wraps the platform synthetic DGP (`src/ml/synthetic*`), keyed by `intervention_type`; generates a complete labeled `(X, T, y)` frame with a **known** ground-truth ATE (used by calibration). Balanced by construction.
- **`CohortEffectDataProvider` (v2 stub):** pulls a real RWD cohort (reusing the tier0 `cohort_constructor`/`data_preparer` output, or `treatment_events` + `business_metrics`), maps each intervention to an observed treatment column. `ground_truth_ate=None`.
- *Depends on:* the synthetic generator. Fail-closed via `EffectDataUnavailable` on missing/malformed/empty/declared-column-absent (mirrors `ExecutorDataUnavailable`, `src/causal_engine/pipeline/executors/causalml.py:57`).

### 3.2 `TwinEffectEstimator` (core) — fit uplift, score the twin population
```python
class EffectEstimate(BaseModel):
    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    att: float | None
    atc: float | None
    per_twin_uplift: np.ndarray       # one score per twin (kept in-process; summarized for persistence)
    auuc: float | None
    qini: float | None
    feature_importances: dict[str, float]
    n_train: int
    estimator_type: str               # e.g. "uplift_random_forest"
    data_provenance: str              # "synthetic_uplift_v1" | "rwd_uplift"

def estimate(training: TrainingFrame, twin_population: pd.DataFrame) -> EffectEstimate
```
- Fits `causal_engine.uplift.UpliftRandomForest` (reused; already prod-wired via `src/agents/heterogeneous_optimizer/nodes/uplift_analyzer.py`) on the **training frame**, then scores the **twin population covariates** (distinct frames — training ≠ scoring). Average of `per_twin_uplift` = population ATE; CIs from the uplift result.
- *Depends on:* `src/causal_engine/uplift`. **Fail-closed** on fit failure / `n < MIN_TRAINING_SAMPLES` (raise `EstimationError` from `src/causal_engine/errors.py`). AUUC/Qini helper failure → field `None` + warning, estimate still returned (non-fatal), mirroring `CausalMLExecutor.execute`.
- Fit may be **cached/pre-fit per (intervention_type, brand)** to hold the <2s SLA; scoring is a vectorized `.predict`.

### 3.3 `HeterogeneityAnalyzer` — "top responding segments"
```python
def segments(training: TrainingFrame, per_twin_uplift: np.ndarray) -> list[SegmentEffect]
```
- Uses `src/causal_engine/hierarchical/segment_cate.py` on top/bottom uplift quantiles → per-segment CATE + uncertainty for the REFINE path. Additive (~1–2s).

### 3.4 `RecommendationPolicy` (pure function) — CI-based three-way decision
```python
def decide(estimate: EffectEstimate, thresholds: PolicyThresholds)
        -> tuple[Recommendation, str, int]   # (DEPLOY|REFINE|SKIP, rationale, recommended_sample_size)
```
- **DEPLOY** if `ate_ci_lower > min_effect`; **SKIP** if `ate_ci_upper < min_effect`; **REFINE** if the CI straddles `min_effect`.
- `recommended_sample_size` from the existing power calc (`SimulationEngine._calculate_recommended_sample_size`) fed the **estimated** effect + variance.
- `min_effect` / confidence come from §3.6 calibration, **not** the inherited `DEFAULT_MIN_EFFECT_THRESHOLD=0.05` / `DEFAULT_CONFIDENCE_THRESHOLD=0.70` (those were tuned to the fake table).

### 3.5 `EffectProvenance` — labeling
Every result carries `data_provenance` + `estimator_type` (mirrors the #689 `data_provenance` pattern). A synthetic-derived estimate can never be mistaken for an empirical one. This also discharges the H5 interim-safety requirement.

### 3.6 Calibration (validation-phase only — NOT runtime)
- `src/causal_engine/energy_score/estimator_selector.py` picks the best uplift learner (RF / GB / Base-S,T,X meta-learners) **once** on synthetic benchmarks.
- Thresholds calibrated so `|ate − ground_truth_ate| / ground_truth_ate < 0.20` holds across a distribution of effect sizes **including near-zero** (confirming the SKIP path fires).
- Output = locked config (`config/digital_twin_config.yaml`), consumed at runtime. Never run per-simulation (that is the rejected Option B and the reason it's slow).

### 3.7 Propensity matching (v2) — upstream of the estimator
`TwinGenerator.generate_twin()` gains an SMD < 0.1 balance gate (`docs/Archive/v4.2_IMPLEMENTATION_TODO.md` §1.1). Load-bearing for unbiased RWD effect recovery. The synthetic DGP is balanced by construction, so v1 only **asserts** balance; full matching is v2 (RWD path).

## 4. Data Flow

```
SIMULATE(intervention_type, brand, twin_type, twin_count)
  ├─ TwinGenerator.generate(n=twin_count)        → twin population (covariates only)   [SCORING frame]
  ├─ EffectDataProvider.get_training_frame(...)  → labeled (X, T, y) (+known ATE)        [TRAINING frame]
  │       └─ fail-closed → EffectDataUnavailable → SimulationResult(status="failed")  (NO fake ATE)
  ├─ TwinEffectEstimator.estimate(training, population)
  │       └─ fit UpliftRandomForest(training) → score population → EffectEstimate
  ├─ HeterogeneityAnalyzer.segments(...)         → top/bottom responding segments
  ├─ RecommendationPolicy.decide(...)            → DEPLOY/REFINE/SKIP + sample_size
  └─ SimulationResult(ate, ci, recommendation, sample_size, segments, data_provenance)
          → persist (v2 live path)
```
The **training frame** (labeled) is distinct from the **scoring population** (twin covariates) throughout.

## 5. Error Handling (fail-closed)

- **Data missing/malformed/empty/declared-column-absent** → `EffectDataUnavailable` → `SimulationResult(status="failed", error=...)`; **no ATE emitted**.
- **Fit failure / `n < MIN_TRAINING_SAMPLES`** → fail-closed `EstimationError` (`src/causal_engine/errors.py`).
- **AUUC/Qini helper failure** → field `None` + warning; estimate still returned (non-fatal), matching `CausalMLExecutor`.
- **The `INTERVENTION_EFFECTS` heuristic is deleted**, not kept as a fallback (keeping it re-introduces the fabrication). The orphaned/drifted `config/digital_twin_config.yaml: intervention_effects` block is removed in the same change.
- Every success result carries provenance; no unlabeled estimate leaves the engine.

## 6. Testing & Phased CI (memory-safe)

The uplift fit + calibration sweep (CausalML forests × DGPs × seeds) is the memory-heavy part; the platform has hit `[Errno 28]`/OOM tail-hangs. Tests are split into two lanes reusing the established mitigations (`pytest-split` sharding, `jlumbroso/free-disk-space`, the isolated `slow-tests` lane, no torch).

### Phase 1 — Light lane (required, runs in `backend-tests`)
Fast, memory-bounded unit tests on tiny frames (n≈200, capped `n_estimators`/`max_depth`, fixed seeds):
- `RecommendationPolicy` — all three CI branches (DEPLOY/REFINE/SKIP) as a pure function.
- `EffectDataProvider` contract + every fail-closed path.
- Provenance labeling; `EffectEstimate` schema; estimator interface contract with the heavy fit mocked.
- Latency assertion (<2s scoring) against a small pre-fit model.
No full-size fit → no OOM.

### Phase 2 — Heavy lane (isolated `slow-tests`, sharded)
The real recovery/calibration tests:
- Fit `UpliftRandomForest` on full synthetic DGPs (n≥1000); assert `|ate−truth|/truth < 0.20` across effect sizes incl. near-zero.
- `energy_score` learner selection on synthetic benchmarks.
- **Sharded via `pytest-split` matrix** (one estimator/DGP family per shard — mirrors the `test_agents` 3-way split), each shard with `jlumbroso/free-disk-space`, explicit per-test teardown/`gc`, bounded forest sizes, **no torch/heavy deps loaded** (CausalML only). Marked `@pytest.mark.slow` so they never run in the light lane.

This caps peak memory per job and prevents the single-fat-job OOM. Determinism via fixed seeds.

## 7. Integration & Sequencing

- The estimator is a **drop-in for `SimulationEngine._simulate_effects` / `_calculate_individual_effect`**; the public `simulate() → SimulationResult` surface is unchanged, so API/agent/FE contracts don't move.
- **v1 lands and is fully validated without H4/H6** (estimator + synthetic provider; unit + calibration tested in isolation).
- **Live path** (`/simulate` returning real numbers end-to-end) additionally needs **H4** (load/fit a model before `generate()`) and **H6** (inject the repo client) — sequenced after v1.
- Once the engine is real, the **agent tool mock (H3)** calls it and the mock is removed — immediate follow-up, not v1.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Synthetic DGP doesn't resemble real intervention response | v1 is explicitly synthetic-validated; RWD provider (v2) is the faithfulness step; provenance label makes the synthetic basis explicit |
| Uplift fit too slow for <2s SLA at large `twin_count` | Pre-fit/cache per (intervention_type, brand); scoring is vectorized; bound forest size |
| CI too wide on small training frames → everything REFINE | calibration sweep tunes thresholds; `MIN_TRAINING_SAMPLES` floor; surface n_train in result |
| Heavy tests OOM CI | Phase-2 sharding + free-disk-space + no-torch + per-test teardown (§6) |
| Removing the heuristic breaks tests that asserted its constants | those tests assert fabricated values and must be rewritten to the recovery/CI contracts (Phase 1/2) |

## 9. Success Criteria

1. `UpliftRandomForest`-based estimator recovers synthetic ground-truth ATE within ≤20% across effect sizes incl. near-zero (the SKIP path fires correctly).
2. DEPLOY/REFINE/SKIP is CI-based; thresholds are calibration outputs, not literals.
3. Engine fails closed on bad/absent data; no path emits an unlabeled or fabricated ATE; `INTERVENTION_EFFECTS` is gone.
4. Phase-1 tests run in the light lane within the existing memory budget; Phase-2 heavy tests pass sharded without OOM.
5. `simulate()` contract unchanged; v1 needs no H4/H6.

## 10. File-level change sketch (for the implementation plan)

- **New:** `src/digital_twin/effect/__init__.py`, `data_provider.py` (`EffectDataProvider`, `SyntheticEffectDataProvider`, `EffectDataUnavailable`), `estimator.py` (`TwinEffectEstimator`, `EffectEstimate`), `heterogeneity.py`, `recommendation.py` (`RecommendationPolicy`, `PolicyThresholds`), `provenance.py`, `calibration.py` (validation-phase harness).
- **Modify:** `src/digital_twin/simulation_engine.py` (replace `_calculate_individual_effect`/`_simulate_effects`; delete `INTERVENTION_EFFECTS`); `config/digital_twin_config.yaml` (remove orphaned `intervention_effects`; add calibrated thresholds + selected learner).
- **Tests:** `tests/unit/test_digital_twin/effect/` (Phase 1), `tests/ml/.../test_twin_effect_calibration.py` (Phase 2, slow-marked, sharded); CI matrix entries for the heavy lane.
- **v2 (separate plans):** `CohortEffectDataProvider`; `TwinGenerator.generate_twin()` SMD<0.1 matching; fidelity feedback; live wiring (H4/H6); agent-tool rewire (H3).
