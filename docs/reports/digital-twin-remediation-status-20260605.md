# Digital Twin Feature — Remediation Status Audit

**Date:** 2026-06-05
**Audit verified:** `docs/reports/digital-twin-audit-20260604.md` (the 17-finding audit, H1–H17)
**Against:** `main` @ `d19fc2c6` (deployed; running image `ghcr.io/enunezvn/e2i-api:d19fc2c6…`, created 2026-06-05 15:58 UTC). Twin remediation PRs landed since the audit: **#704** (H5 effect engine), **#709** (H1/H2 frontend), **#711** (H4 + H6/H7 persistence/wiring), **#724** (deploy follow-up). PRs #712–#723 are the unrelated *causal* remediation.
**Method:** 36-agent verify→adversarial-refute workflow (one agent reads the *current* code at each finding's location; a second independently re-reads to refute, hunting docstring-only relabels / rename-still-fake / wired-only-in-test) + a regression critic + an E2E-completeness critic. **PLUS a cheapest-disproof faithful prod check**: live `supabase-db` row counts and an in-container run of the real `/simulate` handler in `e2i_api`. All 17 source verdicts were independently confirmed (`agrees: true`); the faithful runtime check **overturned the one verdict source review got wrong** (H4).

> **Headline:** The frontend fabrications (H1/H2) and the persistence/wiring layer (H6/H7/H10/H12) are genuinely closed. **But the flagship is still down: `/simulate` and `/simulations/compare` return HTTP 500 on every call in prod** — the H5 engine rewrite dropped `model_id` from `SimulationEngine.__init__`, the route still passes it, and a `# type: ignore[call-arg]` plus engine-mocking tests hid it from mypy and CI. Live `twin_simulations = 0` confirms zero throughput. **H4 is the #1 open item, not a closed one.** Net: 7 closed · 2 kept-by-design · 2 partial · 6 open · 1 new regression.

---

## 1. Status tally (17 audit findings + 1 new)

| Status | Count | IDs |
|---|---|---|
| **ADDRESSED** | 6 | H1, H2, H6, H7, H10, H12 |
| **KEPT-AS-DESIGNED** (not defects) | 2 | H13, H14 |
| **PARTIAL** | 2 | H5, H15 |
| **NOT_ADDRESSED** | 6 | H3, H8, H9, H11, H16, H17 |
| **RUNTIME-BROKEN** (source looked ADDRESSED; faithfully disproven) | 1 | **H4** |
| **NEW regression** (introduced by the fix set) | 1 | **N1** |

The faithful runtime smoke overturned **H4** (both the source verifier and refuter rated it "ADDRESSED, high confidence"). The regression critic surfaced **N1**.

---

## 2. Fully closed — no action

| ID | Was | Verified fix (current code) |
|---|---|---|
| **H1** | FE Results panel rendered hardcoded `SAMPLE_SIMULATION` (ate 0.18/ROI 3.2×/DEPLOY); real result discarded | `DigitalTwin.tsx:421-423` `displayed = selectedId ? selectedDetail : runResult` (real `useRunSimulation`/`useSimulation(id)` data); `onSuccess` clears selection so the fresh run shows; history-click routes through `useSimulation(id)`. `grep SAMPLE_SIMULATION frontend/src/` → only a docstring + tests asserting absence. Honest loading/error/empty states. (#709) |
| **H2** | FE `SAMPLE_HISTORY` fallback + static "2.4s/68%/87%" cards | `historyItems = historyData?.simulations ?? []` (`:437`) with a real "No simulations yet" empty state; stat cards computed from data (`deployRate`, `fidelityPct`, `models_available`) showing `—` when null; the unbacked "2.4s" card dropped. `grep` for the literals → 0. (#709) |
| **H6** | All route handlers built `TwinRepository()` with no client → saves/reads no-op | Every handler now `repo = await _get_twin_repo()` (`digital_twin.py:64-79`) which resolves a **real service-role** async client via `factories.get_async_supabase_client` (raises `ServiceConnectionError` if unset — never returns None); facade propagates the client to all 3 sub-repos (`twin_repository.py:838-844`). Zero bare `TwinRepository()`. (#711) |
| **H7** | `FidelityTracker.validate` called nonexistent `update_fidelity_record` + missing `await` | `fidelity_tracker.py:203` now `await self.repository.update_fidelity_validation(...)` (correct method/signature, awaited); `get_fidelity_by_simulation` also awaited (`:395`). Only surviving `update_fidelity_record` token is an explanatory comment. (#711) |
| **H10** | `/simulations/compare` ran N×twin_count inline with no `heavy_compute_slot` | `compare_scenarios` now wraps the fan-out in `async with heavy_compute_slot():` + `run_in_bounded_executor` (`digital_twin.py:1089-1096`), matching `/simulate`; `HeavyComputeSaturated`→fast 503. (One slot per N-scenario fan-out = tighter bound, no OOM regression.) *Note: shares the H4 runtime bug — see §3.* |
| **H12** | `_save_to_mlflow` stub returned synthetic `models:/twin_<type>_<brand>/latest` | Stub **deleted**; `twin_persistence.py:108` real `mlflow.sklearn.log_model` + pickled preprocessor bundle, fail-closed on untrained model (`:99-100`). Reachable from `train_and_persist_twin`, the Celery retrain task, and `scripts/train_twin_model.py`. Exceeded the KEEP-placeholder bar. **Faithfully confirmed live:** model `5141ba73` carries a real MLflow URI `models:/m-f9220c58…`, not the phantom format. (#711) |

**Kept-as-designed (correctly unchanged — not defects):**
- **H13** — `get_recent_fidelity_records` computes a cutoff and ignores `days`; byte-identical, documented placeholder, zero consumers. As the audit expected.
- **H14** — `worker_heavy replicas:0` (`docker/docker-compose.yml:805-811`) + `HEAVY_OFFLOAD_ENABLED` default false (`compute.py:253`, set nowhere). **Faithfully confirmed:** no `worker_heavy` container on the droplet (only `worker_medium` + 2× `worker_light`). The inline P1 path serves; documented OOM-headroom decision.

---

## 3. PENDING inventory

Legend — **Reach:** PROD = user-reachable now · DARK = code exists, not reachable on current prod config · TEST = test-only · GATED = reachable only behind an off-by-default flag.

### 3a. RUNTIME-BROKEN HIGH — the flagship is down *(do first)*

| ID | Sev | Reach | Residual (current code + faithful evidence) |
|---|---|---|---|
| **H4** | **CRITICAL** | PROD | `/simulate` (`digital_twin.py:778-781`), `/simulations/compare` (`:1045-1047`), and the dark offload `simulation_runner.py:96-98` all build `SimulationEngine(population=…, model_id=model_id)  # type: ignore[call-arg]`. The H5-rewritten engine **does not accept `model_id`** (sig: `population, min_effect_threshold, confidence_threshold, model_fidelity_score, cache, effect_provider, effect_estimator`; it derives `self.model_id = population.model_id` at `simulation_engine.py:110`). → `TypeError` → `except Exception` → **HTTP 500 "Simulation failed"** (`:834-836`). **Faithfully disproven** by running the real `run_simulation` handler in `e2i_api` (HEAD image) — traceback lands in the route's own `_do_sim`. **Live `twin_simulations = 0`** (vs `digital_twin_models = 1`) confirms every call dies before persist. The original H4 (untrained-generator→500) is gone; this is a *new* 500 cause that shipped because the `# type: ignore` silenced mypy's `call-arg` error and every route test mocks `SimulationEngine`. |

**Fix:** drop the `model_id=` kwarg at all 3 sites; set the resolved DB id explicitly so the `twin_simulations.model_id` FK holds — `engine.model_id = model_id` after construction (the route already computes `model_id = UUID(model_row["model_id"])` at `:717`), or set `population.model_id` pre-construction. Remove the `# type: ignore` (do not re-suppress). `hydrate_generator` already restores a model_id (`twin_persistence.py:181-182`) but set it explicitly to be FK-safe.

### 3b. NEW regression — fabricated-value seam reintroduced

| ID | Sev | Reach | Residual |
|---|---|---|---|
| **N1** | HIGH | PROD (once H4 fixed) | A FAILED simulation returns **HTTP 200** with `simulated_ate=0.0`, `recommendation=REFINE`, `status=FAILED` and is **persisted to history**. The engine's `_create_error_result` (`simulation_engine.py:417-440`) returns these on `n_twins<100` after filtering or estimation error; the route copies `status`/`error_message` but **never gates on `result.status==FAILED`** (`digital_twin.py:792-821`, `save_simulation` at `:792` runs regardless); the FE renders it as a real "ATE 0.000 / Refine" success with a "Simulation Complete" toast and **never inspects `status`** (`use-digital-twin.ts:160-174`, `DigitalTwin.tsx:261-358`). History has no status column, so a failed run shows as a real row. Same harm-class as the original H1/H5 (plausible-but-fake user-facing value). **Faithfully reproduced** engine-level (50-twin pop → `status=FAILED`, `ate=0.0`). |

**Fix:** route raises (422/503) when `result.status==FAILED` and does **not** persist a failed row; or the FE branches on `status==='failed'`. Pair with H5 (below).

### 3c. PARTIAL — finish the started work

| ID | Sev | Reach | Residual |
|---|---|---|---|
| **H5** | HIGH | PROD | Core fabrication **fixed**: `INTERVENTION_EFFECTS` deleted (only docstring mentions remain), ATE now from a real `TwinEffectEstimator` uplift fit (`simulation_engine.py:205-221`, `effect/estimator.py:75-81`), fail-closed (`EffectDataUnavailable`/`EstimationError`). **Residual = the report's required transparency half:** `data_provenance` is computed on the internal `SimulationResult` (`simulation_models.py:186`, set `:273`, persisted on the *model* row) but **the HTTP `SimulationResponse`/`SimulationDetailResponse` omit it** (`digital_twin.py:335-372`, `grep provenance` over the route = 0), the `save_simulation` row omits it, and the FE has no provenance field. The audit's "do not fix H4 without H5's provenance" precondition is now live (H4 will return a real *synthetic-DGP* ATE — `provider.py:59,77-88` recovers ~`true_ate=0.15` for every intervention/brand — with no label distinguishing it from a brand-specific causal estimate). |
| **H15** | LOW | mixed | `train_twin_model` route is **resolved** — now a real wired task (`heavy_offload_tasks.py:103` → `train_and_persist_twin`; registered + producer `scripts/train_twin_model.py`). **`generate_twins` route (`celery_app.py:154`) is still dead** — no task body, no producer (only the route entry + its test mirror). DELETE it (real population work is `simulate_population`). |

### 3d. NOT_ADDRESSED

| ID | Sev | Reach | Residual |
|---|---|---|---|
| **H3** | HIGH | GATED | `simulate_intervention` tool still fabricates: `_get_or_create_twins` `return None` (`simulate_intervention_tool.py:144`, real generate() commented), falling to `_create_mock_result` → `ate = base + random.gauss(0,0.02)` (`:339`), `simulation_confidence:0.75`, **`fidelity_warning:False`**, `"deploy"` (`:360,375`). Unchanged. **Unreachable in prod** only because H8 keeps the gate off — a fix-before-flip landmine. |
| **H8** | HIGH | — | `enable_twin_simulation` never set by the prod entrypoint: `ExperimentDesignerInput` (`agent.py:40-106`) lacks the field, `_create_initial_state` (`:416-446`) omits it, the factory default (`graph.py:134`) is inert. Node short-circuits (`twin_simulation.py:94`) → designer pre-screen **dark**. The module-level `graph.py:create_initial_state` defaults it True but has no prod caller. Output state contract exists; the input plumbing does not. |
| **H9** | MED | DARK | `fidelity_tracking_update` (`ab_testing_tasks.py:694-822`) still orphaned (no producer, not in beat schedule). Two bugs: (1) unbound `predicted_effect`/`predicted_ci` on the `if twin_simulation_id:` branch (`:752-783`); (2) **new** — `results_analysis.compare_with_twin_prediction` (`:516-524`) requires `predicted_ci_lower`/`predicted_ci_upper` and has no `predicted_ci` param, but the task passes `predicted_ci=` and omits the required args → `TypeError` on both branches (swallowed into `status=failed`). The wrapped service body is real. |
| **H11** | MED | PROD | Read GETs (`list_simulations` `:845`, `get_simulation_history` `:921`, `get_simulation/{id}` `:1135`, `list_models` `:1340`) have no per-tenant scope — `brand` is a caller-controlled filter, `get_simulation/{id}` has no ownership check. Only `require_operator` (RBAC role) was added, on the **write** routes. The canonical `get_user_brands` helper exists (`auth.py:180`, admin/`'all'` bypass) but is never imported. Low-pri (single-operator tenancy); proactive hardening. |
| **H16** | LOW | — | `TwinPopulation.validate_size` (`models/twin_models.py:177-181`) still `return v` — no `v==len(twins)` assert despite the docstring. No current caller deliberately violates it; the validator falsely advertises enforcement. |
| **H17** | LOW | TEST | `_get_fidelity_from_simulation_summary` (`experiment_monitor/nodes/fidelity_checker.py:157`) still never called (`execute()` → only `_check_fidelity`); hardcoded `actual_effect=0.0` (`:203`) unreachable. File byte-identical to audit HEAD. Wire as the `v_simulation_summary` fallback (real read) or delete. |

---

## 4. Recommended execution sequence

Mirrors the team's proven method (one worktree/branch per phase, red-first TDD + `codex:codex-rescue` → fixed point, faithful in-container verify, serial deploy). Ordered by present-harm × reachability.

### R1 — Restore the flagship + re-close the honesty surface on the now-live path *(CRITICAL; small, surgical)*
- **H4:** remove the invalid `model_id=` kwarg at `digital_twin.py:780`, `:1047`, `simulation_runner.py:98`; set `engine.model_id = model_id` (FK-safe) from the resolved active-model id; delete the `# type: ignore[call-arg]`.
- **N1:** gate `result.status==FAILED` → 422/503 and do not persist a failed row (or FE branches on `status==='failed'`).
- **H5 (provenance half):** add `data_provenance` to `SimulationResponse`/`SimulationDetailResponse`, the `save_simulation` row, and the FE (mirror the #689 `/health-score` provenance pattern); label the synthetic-DGP origin.
- *Red-first test:* a test that exercises the real handler/engine (not a mocked `SimulationEngine`) and asserts a 200 + a persisted `twin_simulations` row on success, a non-2xx (or explicit failed-state) on a sub-threshold population, and `data_provenance` present on the response.
- *Verify:* re-run the in-container smoke → expect 200 + 1 persisted row; force a sub-100 population → expect the N1 gate. **This is the cheapest disproof that the path is actually functional E2E.**

### R2 — Fail-close the gated agent mock *(prerequisite before any agent enablement)*
- **H3:** make `simulate_intervention` raise/skip (matching the #548 fail-closed discipline) instead of `_create_mock_result`; never emit `fidelity_warning:False`/`"deploy"` from a mock. Must land before H8.

### R3 — Activate the designer-agent pre-screen *(feature completion; gated on product readiness)*
- **H8:** add `enable_twin_simulation`/`intervention_type` to `ExperimentDesignerInput` + `_create_initial_state` (thread into node state). Only after R2. Consider gating on the owner's RWD effect-model intent — the synthetic DGP returns a constant ~0.15 regardless of intervention/brand, so an agent pre-screen on it is not yet decision-useful.

### R4 — Correctness / robustness cleanup *(independent, parallelizable)*
- **H9:** fix both bugs (unbound `predicted_effect`/`predicted_ci`; the `compare_with_twin_prediction` signature mismatch), then wire a trigger (beat/post-experiment hook) **or** delete the orphaned task.
- **H17:** wire `_get_fidelity_from_simulation_summary` as the real `v_simulation_summary` fallback (replace `actual_effect=0.0` with a real read) **or** delete.
- **H15-residual:** delete the dead `generate_twins` route entry.
- **H16:** enforce `size==len(twins)` or drop the misleading validator/docstring.
- **H11:** add fail-closed `get_user_brands` scoping to `list_simulations`/`list_models` (+ regression test); no behavior change for the admin/`['all']` user.

### R5 — Structural gaps *(larger; schedule deliberately)*
- **All-real e2e test:** generate→simulate→real-persist→fidelity against a real store (none exists; `tests/integration/test_digital_twin_e2e.py` mocks the repo and hand-derives the "actual"). The real-world fidelity leg is inherently untestable without a ground-truth outcome feed.
- **Fidelity boundary:** reconcile twin-side (`twin_fidelity_tracking`/`v_simulation_summary`) vs A/B-side (`ab_fidelity_comparisons`/`vw_fidelity_summary`).
- Leave `worker_heavy`/auto-retrain dark (H14) unless product needs automatic retraining.

---

## 5. Faithful prod evidence (this host == prod droplet)

| Check | Result |
|---|---|
| `digital_twin_models` rows | **1** — real trained model `5141ba73` (twin_type=hcp, brand=Remibrutinib, r²=0.81, 2000 samples, real MLflow run `019a94dd…` + URI `models:/m-f9220c58…`). Training/persist path works E2E. |
| `twin_simulations` rows | **0** — no simulation ever persisted (consistent with H4: every call 500s before `save_simulation`). |
| `twin_fidelity_tracking` / `twin_retraining_jobs` | **0 / 0** — downstream / no live cohort feed. |
| `worker_heavy` container | **absent** (only `worker_medium` + 2× `worker_light`) — H14 dark, consistent. |
| In-container `/simulate` (real handler, HEAD image) | **HTTP 500** — `TypeError: SimulationEngine.__init__() got an unexpected keyword argument 'model_id'`. |
| Auth posture | **enabled** — 401 on every twin endpoint without a JWT (so the faithful test ran the handler in-container, bypassing only the proven auth `Depends`, not extracting secrets). |

---

## 6. Honesty / method notes
- Verdicts are against the *current source* + *faithful runtime*, not commit messages. The lone over-claim (**H4**) passed both a source verifier and an adversarial refuter and was only caught by running the real handler — a `# type: ignore[call-arg]` plus engine-mocking tests produced a **green CI over a prod-down flagship**. This is the canonical cheapest-disproof lesson: mechanism-validation (tests/CI/mypy green) ≠ premise-validation (the endpoint actually serves a real value).
- The prior memory claim "H4 FULLY FUNCTIONAL E2E ON PROD / `/simulate`→200" is **corrected** by this report.
- Reachability is explicit (PROD/DARK/TEST/GATED) so effort isn't spent on dark paths (H3/H9 latent; H8 gates H3) before the prod-down flagship (H4).
- Nothing was changed by this audit except: this report, and the memory index/entry recording its findings.
