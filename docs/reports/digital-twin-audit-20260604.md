# Digital Twin Feature — Deep Audit: Intended Functionality vs. Codebase Status

*Audit date: 2026-06-04 · HEAD `defd8dec` · This host IS the prod droplet (PROD == DEV == `e2i-analytics-prod`) — live-DB checks are faithful.*

---

## 1. Executive Summary

**Verdict: PARTIAL with three HARMFUL-NOW fabrication surfaces (two ACTIVE on the live frontend, one LATENT/gated in the agent), plus a broken-but-fail-closed compute path. The Digital Twin is a substantial, genuinely-built subsystem whose compute core, persistence schema, retraining lifecycle, and a real orchestrator-reachable fidelity-monitor node are REAL and test-proven — but it is NOT functional end-to-end on any user-reachable path, and the only fabricated values that reach a user today come from the frontend page.**

The ML engine (`TwinGenerator.train` → sklearn RandomForest/GradientBoosting), the statistics machinery (`SimulationEngine` ATE/CI/power), the Supabase-backed persistence (4 tables + 4 views, all confirmed applied on the live droplet), the durable cross-process retraining store (#549), the fail-closed Celery retraining task (#548), and the experiment_monitor `FidelityCheckerNode` (a real, orchestrator-reachable reader of `twin_fidelity_tracking`) are all real, wired, and covered by tests that would fail against a stub. The DB schema is applied to prod and column-faithful to the repository code.

However, the **headline pre-screening value never reaches a user as a real number**, and the **fabrication surfaces are concentrated in the frontend**:

- The **HTTP `/simulate` route** (and `/simulations/compare`) builds a `TwinGenerator` and calls `generate()` *without ever training or loading a model* → `RuntimeError("Model not trained")` → **HTTP 500** (runtime-disproven, not just static). It fails **closed** (honest 500, no fake data) — this is a **REWIRE/broken-fails-closed** path, **not** HARMFUL-NOW. The feature is simply non-functional here.
- The **experiment_designer agent tool** (`simulate_intervention`) unconditionally returns a **mock** — `_get_or_create_twins` always returns `None`, forcing `_create_mock_result` to fabricate an ATE from a hardcoded dict + Gaussian noise with `fidelity_warning: False` and a "deploy" verdict. This is a **LATENT (gated) HARMFUL-NOW** mock: it is reachable **only** when `enable_twin_simulation` is forced True, which **no production entrypoint does** (test-only today). It must be fixed *before* the gate is ever flipped, but it is **not actively firing in prod traffic**.
- The **frontend Digital Twin page** renders a hardcoded `SAMPLE_SIMULATION` (ate 0.18 / ROI 3.2× / fidelity 0.87 / DEPLOY) on **every** load, after **every** run, and after **every** history-row click — the real mutation result is discarded. This, plus the `SAMPLE_HISTORY` fallback and static stat cards, are the **only ACTIVELY-reachable fabrications in prod** (no gate, every page load).

So the engine's hardcoded `INTERVENTION_EFFECTS` heuristic (a "simplified model", not a learned causal effect) is a *latent* C8-style risk behind the broken compute path, while the **frontend SAMPLE_SIMULATION/SAMPLE_HISTORY** is the *active* fabricated-value surface and the **agent tool mock** is a *gated/latent* one.

### Reachable twin consumer inventory (corrected)

The draft's "three live consumers, two fabricated" framing was inaccurate. The corrected inventory of reachable twin surfaces:

| Consumer | Reachable in prod? | Serves fabricated values? | Status |
|---|---|---|---|
| HTTP `/simulate`, `/simulations/compare` route | Yes (OPERATOR) | **No** — 500s (fails closed) | REWIRE/broken |
| experiment_designer `simulate_intervention` tool | **No** — gated off (test-only) | Would (mock) **if** gate flipped | LATENT HARMFUL-NOW |
| Frontend Digital Twin page | **Yes** (every load) | **Yes** — `SAMPLE_SIMULATION`/`SAMPLE_HISTORY`/static cards | **ACTIVE HARMFUL-NOW** |
| experiment_monitor `FidelityCheckerNode` | **Yes** (orchestrator-routable) | **No** — real read of empty `twin_fidelity_tracking` | REAL (inert: empty data) |

### Status at a glance

| Layer | Status | One-line verdict |
|---|---|---|
| Engine — `TwinGenerator.train` | **REAL** | Real sklearn fit/predict/CV; only called by the retraining task |
| Engine — `SimulationEngine` stats | **REAL** | Real `np.mean`/`scipy.sem`/z-CI/power; sound over its input |
| Engine — effect model (`INTERVENTION_EFFECTS`) | **REWIRE** (owner-confirmed stopgap, 2026-06-04) | Hardcoded base-effects + noise, NOT ML/causal; owner intends data-derived causal effects → redesign through a real uplift/CATE estimator |
| Engine — `/simulate` wiring (train/load before generate) | **REWIRE** | Missing model-load step → `RuntimeError` → 500 (fails closed) |
| Persistence — repos & fidelity math | **REAL** | Real postgrest CRUD + real error/grade math |
| Persistence — live API client wiring | **REWIRE** | `TwinRepository()` built with no client → all core reads/writes no-op |
| Persistence — `FidelityTracker.validate` | **REWIRE** | Calls non-existent `update_fidelity_record` + missing `await` (latent) |
| Retraining — #548/#549 task + durable store | **REAL** | Real fail-closed sklearn retrain; durable job round-trip proven |
| Retraining — auto-trigger | **KEEP-AS-PLACEHOLDER** | `auto_trigger_retraining` defaults False; no live cohort feed (by design) |
| Agent (designer) — `TwinSimulationNode` (graph) | **REAL** node, **REWIRE** wiring | Real node, edged, but prod agent never sets `enable_twin_simulation` |
| Agent (designer) — `simulate_intervention` tool | **HARMFUL-NOW (latent/gated)** | Always returns fabricated ATE/"deploy"; `fidelity_warning:False`; reachable test-only |
| Agent (monitor) — `FidelityCheckerNode` | **REAL & wired** (inert data) | Orchestrator-routable; real read of (empty) `twin_fidelity_tracking` |
| API Route — router + 12 endpoints | **REAL & wired** (2 compute endpoints **REWIRE/broken-fails-closed**) | Reachable, OPERATOR-gated writes; reads lack brand authZ |
| Celery — `simulate_population`, `execute_twin_retraining` | **REAL** | Real compute, fail-closed; worker_heavy is dark (replicas 0) |
| Celery — `generate_twins`, `train_twin_model` routes | **DELETE** | Route entries with no task and no producer (dead) |
| Celery — `fidelity_tracking_update` | **REWIRE** | Real body (wraps real `compare_with_twin_prediction`), orphaned (no producer) + latent NameError |
| DB Schema — 4 tables + 4 views on live droplet | **REAL & applied** | Tables present, **0 rows each**, columns 1:1, triggers + views live, ledger-recorded |
| Frontend — page Results panel | **HARMFUL-NOW (active)** | Hardcoded `SAMPLE_SIMULATION`; real result discarded |
| Frontend — history fallback + stat cards | **HARMFUL-NOW (active)** | `SAMPLE_HISTORY` masks empty fetch; "2.4s/68%/87%" literals |
| Frontend — client/hooks/contract test | **REAL** | Correctly wired to `/api/digital-twin/*`; CI-run contract test |
| Tests — engine/generator unit | **REAL** | 56 engine tests, 0 mocks, behavioral invariants; real sklearn |
| Tests — full real e2e (generate→sim→persist→fidelity) | **GAP** | No single all-real e2e; persist + real-world fidelity always mocked |

---

## 1.5 Owner Decisions & Resolved Intent (2026-06-04)

The two product-intent questions the code audit could not resolve (H5, H11) were investigated via git/PR/issue/doc archaeology + cross-route comparison (workflow `wf_95d7b4a2-c44`) and then **decided by the feature owner**. Recorded here so the next audit does not re-open them.

### H5 — effect-model: **REWIRE (owner-confirmed stopgap)**
- **Investigation finding (medium confidence):** the `INTERVENTION_EFFECTS` heuristic looked like a *deliberate v1 parametric pre-screen* — born in the first commit, self-labeled "simplified model" (no replacement-TODO), documented as the mechanism in `SYNTHETIC_DATA.md:523-549`, with the trained model scoped to `baseline_outcome` only and **no effect seam** for a model. No ticket ever proposed replacing it.
- **Owner decision (overrides the lean):** *"This is meant for causal analytics… the digital twin is supposed to generate synthetic populations from historical data and simulate intervention **effects** for A/B-test pre-screening."* → The intended design is **data-derived causal effects**, so the hardcoded table **is a stopgap**. **H5 reclassified KEEP-AS-PLACEHOLDER → REWIRE.**
- **Implication:** this is a **redesign, not a drop-in** — the trained generator is baseline-only by construction; there is no treated-vs-untreated counterfactual today. The rewire must plumb a real treatment-effect/uplift estimator (the in-repo `causal_engine` CausalML uplift, or EconML `CausalForestDML`/`LinearDML` already used by `causal_impact`) through `SimulationEngine._calculate_individual_effect`, replacing the dict-derived `base_effect`. Until then, the heuristic ATE must carry a `data_provenance` label so it is never mistaken for a causal estimate (still required even in the interim).

### H11 — read-side brand authZ: **REWIRE (owner-confirmed; low priority, single-tenant)**
- **Investigation finding (high confidence):** the platform's intended posture **is** to brand-scope tenant reads (fail-closed `get_user_brands` on `memory`/`graph`/`sentinels`/`explain`, hardened in #657). `digital_twin` is the lone brand-bearing read route that never got it — auth came from the generic RBAC commit, no brand-scoping commit since, no carve-out comment → **oversight, not intent**.
- **Owner decision:** **add brand-scoping to match `memory`/`sentinels`/`graph`.** Tenancy confirmed **single-operator (no brand-limited users)** → H11 is **latent/low-severity**; the fix is **proactive hardening** (remove the leak before any brand-limited operator is ever provisioned), not urgent. ~15-line fail-closed `get_user_brands` scope on `list_simulations`/`list_models` + a regression test; no behavior change for the admin/`['all']` user.

---

## 2. Intended Functionality

The Digital Twin Engine is a pharmaceutical A/B-test **pre-screening** subsystem: train ML "twins" of HCPs/patients/territories on historical data, simulate intervention effects on a synthetic population in seconds, and emit a DEPLOY / REFINE / SKIP recommendation so unviable experiments are skipped before spending real-world resources.

**Stated KEY PRINCIPLE (the "validation paradox" answer):** twins are a hypothesis-*refinement / pre-filter*, NOT a replacement for real A/B tests; promising sims proceed to real tests, fidelity is tracked over time. (`docs/Archive/digital_twin_component_update_list.md:10,360-379`; `docs/Archive/digital_twin_implementation.html:706-712`; `README.md:26,140-147`.)

### Intended capabilities (with source refs)
- **Twin Generation** — sklearn RF/GB on ≥1,000 historical rows; synthesize 100–100,000 twins preserving feature distributions; return R²/RMSE/MAE/5-fold CV; each twin carries `baseline_outcome` + `baseline_propensity`. (`README.md:142-143`; `docs/SYNTHETIC_DATA.md:469-499`; `config/digital_twin_config.yaml:6-57`.)
- **Intervention Simulation** — filter population → per-twin effect from base-effect × decile/engagement/adoption/duration/channel/propensity multipliers × noise → ATE/CI/SE → heterogeneity → recommendation → recommended real-experiment sample size. (`docs/SYNTHETIC_DATA.md:501-553`; `config/digital_twin_config.yaml:59-107`.) **Note:** config base-effects (`email_campaign 0.08`…) and `SYNTHETIC_DATA.md` (`email_campaign 0.05`…) *disagree on names and values* — both are **modeled priors, not measured outcomes**.
- **Smart Recommendations** — DEPLOY ≥5%, REFINE 2–5%, SKIP <2%, with CI qualifiers. (`README.md:145`; `config/.../digital_twin_config.yaml:60-65`.)
- **Recommended sample-size** at power 0.80 / α 0.05. (`config:105-107`.)
- **Fidelity Tracking & grading** — `prediction_error=(sim−actual)/actual`; grades excellent <10% / good 10-20% / fair 20-35% / poor >35% / unvalidated; DB trigger auto-grades on actual recorded; `confidence = 30% N + 30% precision + 40% fidelity`. (`docs/Archive/...:23,132-176`; `config:109-130`.)
- **Degradation detection + auto-retraining** (Phase 6, `enabled:false` by default). (`config:181-223`.)
- **Twin retraining durable store (#548/#549)** — `twin_retraining_jobs` table + repository so an API-created job is completed by a separate Celery worker; `fidelity_after` written ONLY on certified completion (#548 fail-closed). (`database/ml/029_twin_retraining_jobs.sql:1-84`.)
- **MLflow lineage**, **Redis result caching (Phase 5)**, **memory-system integration** (design intent).

### Intended data model
`digital_twin_models`, `twin_simulations`, `twin_fidelity_tracking` (the original v4.2 "3 New Tables", `database/ml/012`), **plus** `twin_retraining_jobs` (`029`, #549) and `ab_fidelity_comparisons` (A/B side). **Plus 4 SQL views defined in `012`** — `v_active_twin_models` (`012:296`), `v_simulation_summary` (`012:319`), `v_model_fidelity_history` (`012:348`), `v_fidelity_degradation_alerts` (`012:371`) — part of the intended schema and an actual live read surface (`v_simulation_summary` is referenced by `experiment_monitor/fidelity_checker.py:174-184`). Custom enums: `twin_type`, `simulation_status`, `simulation_recommendation`, `fidelity_grade`.

### Intended wiring
Twin = a **new tool on the existing Experiment Designer agent** (Tier-3), inserting a `[TWIN SIMULATION]` pre-screen before the Design node (skip on low effect); **and a fidelity-monitoring node on the Experiment Monitor agent** that reads recorded twin-vs-actual fidelity. REST group `/api/digital-twin/*` at OPERATOR auth. Dedicated Celery `twins` queue on the Heavy Worker tier. (`docs/ARCHITECTURE.md:26,57,313,352`; `v4.2_integration.md:102-110`.)

### Version / phase evolution
- **v4.2 / v4.2.0** headline version (README, component-list, implementation HTML).
- **Phase 15** = the flagship milestone (commit `b32ea757`, 2025-12-26): `TwinSimulationNode` pre-screening, Supabase/BaseRepository rewrite, 1002-line API route, config.
- **Phases 1/5/6** (commit `52ac5419`): `SimulationCache` (5), `TwinRetrainingService` (6).
- **#548 → #552** (2026-05-29): fixed a dangling-import no-op; implemented the real fail-closed Celery retrain task.
- **#549 → #564** (2026-05-30): durable `twin_retraining_jobs` cross-process store.
- **#572** (P2): OOM heavy-offload to `worker_heavy` (DARK).
- **#666** (2026-06-03): replaced `/health` hardcoded stats with real counts, added `/simulations/history` + `/simulations/compare`, stopped echoing raw exceptions.

**Critical status caveat from the docs themselves:** `v4.2_integration.md` has an explicit "Implementation Status" split — ✅ Completed = *scaffolding only* (files copied, config, docs); ⏳ Pending = "Implement twin generation ML algorithms / Connect to MLflow / Build fidelity workflows / Create API endpoints / Write tests." Per REASON-BEFORE-RULES, this is an **early-state artifact**; later work (#548/#549, SYNTHETIC_DATA runtime descriptions, the OPERATOR-gated route) built much of it out. This audit resolves which capabilities are now real vs still placeholder.

---

## 3. Implementation Status by Layer

### 3.1 Engine Core

**REAL:**
- `TwinGenerator.train` (`twin_generator.py:143-233`) — real `self.model.fit(X_train,y_train)` (:199), `predict` (:202), `cross_val_score(cv=5)` (:203), real R²/RMSE/MAE (:216-218); `_create_model` returns `RandomForestRegressor(n_estimators=100,max_depth=10)` / `GradientBoostingRegressor` (:344-358); `MIN_TRAINING_SAMPLES=1000` enforced. **Verified confirmed-high.** Only called from `ab_testing_tasks.py:1152` (retraining) + tests.
- `SimulationEngine._calculate_statistics` / `_calculate_recommended_sample_size` (`:450-586`) — `np.mean` (:458), `scipy.stats.sem` (:459), z-CI (:462-464), `stats.norm.ppf` power formula (:575-581). Sound stats over whatever vector it's given.
- Pydantic models — real validators (`DigitalTwin` features-non-empty `twin_models.py:150-156`; `SimulationResult` CI-ordering `simulation_models.py:195-200`; `FidelityRecord.calculate_fidelity` real error/grade math :276-302). *Minor:* `TwinPopulation.validate_size` (`twin_models.py:177-181`) is a no-op validator (returns `v` without asserting `size==len(twins)`).

**KEEP-AS-INTENTIONAL-PLACEHOLDER (latent HARMFUL):**
- `SimulationEngine._calculate_individual_effect` (`simulation_engine.py:344-448`) — the per-twin treatment effect is a **deterministic heuristic, NOT ML and NOT from the trained model**: `base_effect` from the hardcoded `INTERVENTION_EFFECTS` table (`:71-100`) × hand-tuned multipliers, then `effect += np.random.normal(0, variance)` (`:441`). `model.predict()` appears **exactly once** in the package — at `twin_generator.py:273` for `baseline_outcome` — and is never consulted in the ATE math. The headline ATE is **mathematically independent of the fitted regressor**. Self-labeled "simplified model" (`:70`). *(The added Gaussian noise sits on top of a fully feature-derived multiplier and is legitimate stochastic simulation noise, not a canned return — celery-async verification correctly did not flag it as fabrication.)*

**REWIRE (the central wiring bug):**
- `/simulate` inline path (`digital_twin.py:616` builds a fresh `TwinGenerator`, `:671` calls `generate()`) and `simulation_runner.py:74-77` (Celery path) **never train or load a model** before `generate()`. `generate()` raises `RuntimeError("Model not trained. Call train() first.")` (`twin_generator.py:255-256`). **Runtime-disproven** in the project venv. Caught by `except Exception` (`:729-731`) → **HTTP 500**. The retraining task *does* persist a real model (`ab_testing_tasks.py:1185`), but **nothing in the simulate path loads it** (grep for `joblib|pickle|mlflow|load_model` in the simulate path = zero hits). The two halves are disconnected. Intent for "load before generate" is explicit in the agent-tool comments (`simulate_intervention_tool.py:127,131,134`).

> **Adversarial note:** the engine-core reader called the overall status "mixed" and said claim-11 returned "no fabricated value." Verification **confirmed** this for the HTTP route (it fails closed, true). A *second* consumer (the agent tool) **would** return fabricated values, but is gated off in prod (test-only) — see §3.3.

### 3.2 Persistence / Fidelity / Retraining

**REAL:**
- All four repositories issue genuine postgrest calls against real tables: `TwinModelRepository` (`twin_repository.py:117,151-156,233-244`), `SimulationRepository` (`:411,432-437,517-522`), `FidelityRepository` (`:609,661-667,728-739`), `TwinRetrainingJobRepository` (`:1018-1052`). Guarded by `if self.client`.
- `FidelityRecord.calculate_fidelity` (`simulation_models.py:282-300`) — real `(sim−actual)/actual`, abs error, CI coverage, grade thresholds.
- `execute_twin_retraining` (`ab_testing_tasks.py:1234-1266` → `_execute_real_twin_retraining:1043-1231`) — real `TwinGenerator.train`, finite-R² extraction (`_extract_validation_r2` uses `math.isfinite`), `repo.save_model`, `complete_retraining`. Every failure branch → `_mark_failed` (no metric written). **Fail-closed, no fabricated metric.**
- Durable retraining wiring proven end-to-end: `trigger_retraining` persists a job row *before* enqueue (`retraining_service.py:336-345`), `.delay()` to queue `ml` (`:355-361`), `complete_retraining` confirms the durable write *before* mutating in-process state (`:483-505`), `get_job_status` reads the durable record first (`:443-447`). `fidelity_after` written only on success, cleared on fail (`twin_repository.py:1050-1052,1068-1069`).
- `SimulationCache` (`simulation_cache.py`) — real async Redis `get/setex/hset/hincrby/scan_iter/delete`, pickle-over-latin1 round-trip; no-op when Redis down.

**REWIRE (live-API inertness — confirmed-high):**
- **All 10 route handlers construct `TwinRepository()` with no client** (`digital_twin.py:489,686,764,838,1006,1096,1216,1273,1328,1417`). `TwinRepository.__init__` passes `supabase_client=None` to the three core sub-repos, and **only `TwinRetrainingJobRepository` self-resolves** a client (`_ensure_async_client`, `:967-987`). So in the live API, `self.client is None` for model/simulation/fidelity → **every save no-ops (returns the input UUID), every read returns None/[]**. This is a **wiring defect, not a fake-value defect** — `/health` honestly degrades to zeroed counts (`:494-502`) rather than fabricating.
- `FidelityTracker.validate` (`fidelity_tracker.py:196`) calls `self.repository.update_fidelity_record(record)` — **a method that does not exist** (real name `update_fidelity_validation`, different signature) — and calls async repo methods **without `await`** (`:136,196`). Currently *shadowed* because the no-client repo makes `get_simulation` return None → the `/validate` route 404s first (`digital_twin.py:1102-1106`). If a client were ever wired, this raises `AttributeError` + leaks un-awaited coroutines. **Latent crash.**

**KEEP-AS-INTENTIONAL-PLACEHOLDER (intent-documented, currently unreachable/non-harmful):**
- `_save_to_mlflow` (`twin_repository.py:300-310`) — returns synthetic `models:/twin_<type>_<brand>/latest`, real `mlflow.sklearn.log_model` commented at :309. **Unreachable in the wired path**: the retraining task builds `TwinModelRepository(supabase_client=...)` with no `mlflow_client`, so the `if self.mlflow_client and model_artifact` guard (:85) is False; `mlflow_model_uri` persists as `None`.
- `_invalidate_model_cache` (`:433-437`) — clears the entire cache (over-invalidates, never stale; conservative).
- `get_recent_fidelity_records` (`:766-776`) — computes a midnight cutoff then **never applies the date filter** (inline comment admits it); returns real rows but ignores `days`.

### 3.3 Agent Integration

The Digital Twin has **two** agent surfaces — a pre-screening node + tool on **experiment_designer**, and a fidelity-monitoring node on **experiment_monitor**. The draft audited only the first; both are covered here.

#### 3.3a experiment_designer (pre-screening)

**REAL but DARK in prod (REWIRE):**
- `TwinSimulationNode.execute` (`twin_simulation.py:78-234`) is a real async node, edged `context_loader → twin_simulation → design_reasoning/END/error_handler` (`graph.py:200,217-225`), gated by `state.get("enable_twin_simulation", False)` (`:94`).
- The **state contract is fully scaffolded on the OUTPUT side**: `experiment_designer/state.py:212-234` declares `enable_twin_simulation` (:214), `intervention_type` (:215), `twin_simulation_result` (:229), `twin_recommendation` (:230), `twin_simulated_ate` (:231), `twin_recommended_sample_size` (:232), `twin_top_segments` (:233), `skip_experiment` (:234), plus status literals `'simulating_twins'` (:20) and `'skipped'` (:27).
- **But the production agent entrypoint never enables it.** `ExperimentDesignerAgent._create_initial_state` (`agent.py:425-444`) omits `enable_twin_simulation`, and `ExperimentDesignerInput` (`agent.py:40-83`) has no such field (verified fields: `business_question`, `constraints`, `available_data`, `preregistration_formality`, `max_redesign_iterations`, `enable_validity_audit`, `brand` — no twin fields). The graph-factory default `enable_twin_simulation=True` (`graph.py:134`) is **inert** — the factory param is *never threaded into the node or state*. Net: **in prod traffic the node short-circuits to `status="reasoning"` and runs no simulation.** (Orchestrator → `ExperimentDesignerAgent.run` IS wired via `_agent_method_map.py:123-134` + `dispatcher.py:268-271`.) The gap is precisely the **Input→state plumbing**: the *output* contract exists in `state.py`, the *input* contract does not.

**HARMFUL-NOW — LATENT/GATED (the worst non-active finding):**
- `simulate_intervention` tool → `_get_or_create_twins` (`simulate_intervention_tool.py:115-138`) **unconditionally returns `None`** (the real `generator.generate()` is commented out at :135). So the tool always falls through to `_create_mock_result` (`:244,309-368`), which fabricates `ate = base_effects[type] + random.gauss(0,0.02)` (email 0.06 … speaker 0.14), `simulation_confidence=0.75`, **`fidelity_warning: False`**, and a **"deploy"** recommendation when `ate≥0.05`. These flow into the agent's `twin_simulated_ate` state. *(Confirmed against source this session: `:138 return None`; comments `:127` "In production, this would load a pre-trained model", `:131` "replaced with actual model loading".)* **Reachability qualifier (corrected):** this mock is reachable **only** when `enable_twin_simulation` is forced True — which **no prod entrypoint does** (`Input` lacks the field; `_create_initial_state` never sets it). So it is a **HARMFUL-NOW-CAPABLE mock that is currently UNREACHABLE in prod traffic (test-only)** — a "fix-before-flip" item, **not** an actively-firing fabrication like the frontend findings. It is **shielded by the dark gate (H8)**.

#### 3.3b experiment_monitor (fidelity monitoring) — **REAL & WIRED (newly audited)**

This consumer was **missing from the draft** and is material: it is a fully-wired, orchestrator-reachable live twin consumer.

- `FidelityCheckerNode.execute` (`experiment_monitor/nodes/fidelity_checker.py:73`) is a real async node, **added and edged** in the graph: `interim_analyzer → fidelity_checker → alert_generator` (`graph.py:50,58,66-67`).
- It is **reachable in prod via the orchestrator**: registered (`factory.py:127-129`), method-mapped (`orchestrator/_agent_method_map.py:135-145`, `experiment_monitor → run`), and routable (`orchestrator/nodes/router.py:131-133`, `intent_classifier.py:63,157,302`). `ExperimentMonitorAgent.run_async` builds the initial state (incl. `fidelity_threshold`) and calls `self.graph.ainvoke` (`agent.py:125-170`) → reaches `fidelity_checker`.
- **Unlike the route's no-op `TwinRepository()`, this node SELF-RESOLVES a real Supabase client** (`fidelity_checker.py:39-41`, `get_supabase_client`) and issues a **real postgrest read** of `twin_fidelity_tracking` (`:113-121`: `.table('twin_fidelity_tracking').select('simulation_id,simulated_ate,actual_ate,prediction_error,fidelity_grade').eq('actual_experiment_id', …)`), producing real `FidelityIssue` objects (`:141-150`).
- **Classification: REAL node, REWIRE/inert-data.** On the empty prod `twin_fidelity_tracking` it **honestly returns no fidelity issues** — **no fabrication**. It only becomes useful once real fidelity rows exist (which requires the persist path to work — see §3.2 H6).
- **DELETE-or-WIRE candidate:** `_get_fidelity_from_simulation_summary` (`fidelity_checker.py:157-215`) is a documented "alternative/fallback" that is **never called** (`execute()` at :73 invokes only `_check_fidelity`; grep finds zero call sites). It also contains a hardcoded `actual_effect=0.0` placeholder (`:203`, labeled "Not available in summary") — currently **unreachable** because nothing calls the method. Surface as a new finding (H17).

> **Adversarial corrections applied (designer side):** (a) the agent-integration reader cited a nonexistent `factory.py:115` for the *designer* — designer registration is via the injected `agent_registry` dict (`dispatcher.py:179-185`); the *mechanism* is real, the *citation* was fabricated. (b) The reader's "no production code calls `TwinGenerator.train()`" is **FALSE** — `ab_testing_tasks.py:1150-1157` does; the accurate statement is "no *simulate* path *loads* a trained model." (c) `validate_twin_fidelity` tool (designer) is real but **not graph-wired** (zero refs in designer `graph.py`/`agent.py`).

### 3.4 API Route (`src/api/routes/digital_twin.py`, mounted `/api/digital-twin/*`)

The 1462-LOC router is **REAL and WIRED** (`main.py:75` import, `:1033` `app.include_router(digital_twin_router, prefix="/api")`), with no canned-JSON handlers. **9 of 12 endpoints are genuinely real reads/fidelity tracking** (health honest-degrade; list/history/{id}; models/{id}/fidelity/report; validate). RBAC is real: `/simulate`, `/simulations/compare`, `/validate` require OPERATOR (`Depends(require_operator)` at `:550,876,1075` → `auth.py:538-542` → 403). Middleware JWT-gates every digital-twin path (none in `PUBLIC_PATHS`), with the documented intentional fail-open when auth is disabled.

**The two compute endpoints are broken but FAIL CLOSED — REWIRE, not HARMFUL-NOW** (correcting the draft's status-table/header over-label, which was inherited from an over-eager api-route verifier): `/simulate` (`:616,671`) and `/simulations/compare` (`:918,920`) hit the **same untrained-generator `RuntimeError`** → caught at `:730-731` (`except Exception as e: … raise HTTPException(status_code=500, detail="Simulation failed")`) → **HTTP 500 with no fabricated value reaching the client** (runtime-disproven). A path that 500s with no fake data is **not** HARMFUL-NOW by the project's own definition; it is a broken-fails-closed **REWIRE**. Every API test masks the breakage by patching `TwinGenerator`/`SimulationEngine` — green tests do **not** prove the route works.

**Confirmed gaps (all confirmed-high):**
- **No `data_provenance`/mock label on any response model** (`:238-460`) — if the heuristic ATE were ever returned, it would be indistinguishable from an empirical estimate (unlike the `/health-score/components` pattern adopted in #689).
- **Read-only GETs have no per-tenant/brand authZ** — `brand` is only a query *filter*, never a scope; any authenticated VIEWER+ can read any brand's simulations/models. (Verification rated the "intentional" hypothesis **uncertain/low** — no intent evidence; this is *not* the same as the documented "monitoring/analytics PUBLIC" decision, which concerned fully-unauthenticated endpoints.)
- **`/simulations/compare` runs N×twin_count INLINE with no `heavy_compute_slot` guard and no offload** (`:956-958`) — unlike `/simulate` (`:682-683`). A genuine OOM/DoS-risk asymmetry; verification rated this an oversight (medium confidence).
- **Offload is dark on multiple axes**: `HEAVY_OFFLOAD_ENABLED` defaults false (`compute.py:253`) AND `worker_heavy` ships `replicas: 0` (`docker-compose.yml:811`).

### 3.5 Celery / Async

**REAL:** `simulate_population` (`heavy_offload_tasks.py:57-78` → real `run_population_simulation`) and `execute_twin_retraining` (`ab_testing_tasks.py:1234`, fail-closed real sklearn). Both correctly named/routed (`celery_app.py:157,161`).

**DELETE (vestigial routes):** `src.tasks.generate_twins` (`celery_app.py:154`) and `src.tasks.train_twin_model` (`:156`) are **route entries with NO task definition and NO producer anywhere** (grep rc=1 for both `def …` and any `.delay/.apply_async/send_task`). Aspirational/dead.

**REWIRE:** `fidelity_tracking_update` (`ab_testing_tasks.py:694-822`) — real body that wraps a **real, audited** twin-fidelity comparison: `ResultsAnalysisService.compare_with_twin_prediction` (`src/services/results_analysis.py:516-586`; real `TwinPredictionComparison` build at `:104-107`, real fidelity scoring `:600-643`) — **but** the task is **orphaned** (no producer, not in the beat schedule) **and carries a latent NameError**: in the `if twin_simulation_id:` branch (`:752-753`) `predicted_effect`/`predicted_ci` are never bound, yet referenced unconditionally at `:782-783`. The wrapped service body is REAL; the *task wiring* is the defect.

**KEEP-AS-INTENTIONAL-PLACEHOLDER:** `worker_heavy replicas:0` + dark `simulate_population` offload — documented (`docker-compose.yml:805-811` cites the 2026-05-30 16GB-droplet memory audit; the API uses the inline P1 path regardless of the flag).

> **Faithful live check:** `docker ps` on this prod droplet shows `worker_medium-1`, `worker_light-1/2` running and **NO `worker_heavy` container at all** — stronger than "replicas:0 in the file." So *zero* workers consume `twins`/`ml` in prod today; an enqueued retrain job would sit unconsumed.

### 3.6 DB Schema + live droplet state — **REAL & APPLIED**

**Faithful check (`docker exec -i supabase-db psql`):** all four tables EXIST on the live prod droplet — `digital_twin_models`, `twin_simulations`, `twin_fidelity_tracking` (`database/ml/012`) and `twin_retraining_jobs` (`029`). **Row counts (re-run this session): 0 / 0 / 0 / 0 — "real-but-empty."** No twin model has ever been persisted, no simulation run, no retraining job created on this DB (consistent with §3.1–3.4: every live compute/persist path is broken or no-op).

**Views (newly audited — the draft only counted "4 tables"):** `012` also defines **4 SQL views** — `v_active_twin_models` (`012:296`), `v_simulation_summary` (`012:319`), `v_model_fidelity_history` (`012:348`), `v_fidelity_degradation_alerts` (`012:371`). The faithful `pg_views` check on the droplet confirms **`v_active_twin_models`, `v_simulation_summary`, `v_model_fidelity_history`, `v_fidelity_degradation_alerts` are LIVE**. `v_simulation_summary` is a code-referenced read surface (the `experiment_monitor` fallback `_get_fidelity_from_simulation_summary`, `fidelity_checker.py:174-184` — though that fallback is itself never called, see §3.3b/H17). These views are part of the intended data model and were not audited in the draft.

- Column cross-check vs `twin_repository.py` is **CLEAN** — every column the repos write/read exists in the live tables (≈40 cols across the 3 core tables + 12 on `twin_retraining_jobs`, verified 1:1 against `TwinRetrainingJobRecord`).
- **Triggers live (verified):** `trg_auto_grade_fidelity` (BEFORE INSERT/UPDATE OF actual_ate) and `trg_update_model_fidelity` exist; `pg_get_functiondef` shows the live `trigger_auto_grade_fidelity` body is byte-for-byte the SQL file. `update_fidelity_validation` correctly delegates grade computation to this trigger.
- **Migration ledger:** `public.schema_migrations` records `ml/012_…` (applied 2026-06-04 01:34:20) and `ml/029_…` (01:34:24) — resolving the #549 prod-application open question: **029 IS applied to prod.**
- **No RLS** on any twin table (`relrowsecurity=false`, 0 policies) — but this **matches the migration files** (012 GRANTs commented out, 029 defines none), so it is intent-consistent, not drift. It does mean no DB-layer tenant isolation on these PHI-adjacent tables (empty today → no exposure).
- *Cosmetic:* `012`'s internal header comment says "Migration: 011_digital_twin_tables.sql" — a stale rename artifact (filename + ledger are correctly `012`), harmless. This resolves the docs' "Migration 011 vs file 012" inconsistency: **012 is canonical; 011 does not exist as a separate twin migration.**

### 3.7 Frontend

The page is routed, lazy-loaded, auth-gated (`routes.tsx:388-397`; nav entry also at `routes.tsx:157`), and the client/hooks/types are **REAL and correctly wired** to `/api/digital-twin/*`. The CI-run contract test (`digital-twin.contract.test.ts`) pins history/compare paths.

**REAL & wired:** `useDigitalTwinHealth` → `GET /health` (benign `unknown` fallback); the simulate-mutation *request* path (form → `useRunSimulation` → `POST /simulate`); `useSimulationHistory` → `GET /simulations/history`.

**HARMFUL-NOW — ACTIVE (the only fabrications actually reaching prod users; confirmed against source this session):**
- **Results panel** — `selectedSimulation` is initialized to the hardcoded `SAMPLE_SIMULATION` (`DigitalTwin.tsx:319`, ate 0.18 / ROI 3.2× / fidelity 0.87 / DEPLOY / projections), and `onSuccess` **never calls `setSelectedSimulation(data)`** — it only switches to the history tab + toasts. History-row click also resets to `SAMPLE_SIMULATION` (`:561-563`, with inline comment `// In real app, would fetch full simulation`). **A real user always sees the same fabricated DEPLOY result, even after a real run.**
- **`SAMPLE_HISTORY` fallback** (`:354` `historyData?.simulations || SAMPLE_HISTORY`) silently substitutes 4 fake rows on empty/loading/error — and on a fresh/empty prod DB (which it is), the History tab shows fabricated rows by default.
- **Top stat cards** "Avg Execution 2.4s / Deploy Rate 68% / Model Fidelity 87%" (`:399-414`) are static literals with no data source.

> **Adversarial correction:** the frontend reader claimed `SimulationPanel`/`ScenarioResults`/`RecommendationCards` are "dead/unused in the app." **REFUTED** — they ARE imported and rendered by `InterventionImpact.tsx:53,973,983,988`. Correct narrower claim: not used by the *Digital Twin page* (which reimplements inline). Do **not** delete them.

### 3.8 Tests — what is actually PROVEN

- **REAL:** 56 `SimulationEngine` unit tests, **0 mocks**, 57 real `engine.simulate()` calls asserting behavioral invariants a stub would fail (intensity/duration/decile ordering, SKIP rationale text, determinism, confidence-vs-N, NaN/Inf guards) — **re-run green this session** ("56 passed"). `TwinGenerator` tests run real sklearn. `test_retraining_service_durable.py` uses a *faithful* in-memory `FakeSupabaseClient` shared across two repo instances to prove the cross-process metric round-trip + fail-closed-when-inert. One genuinely-faithful retrain test (`test_twin_retraining_end_to_end_real_trainer:252`) runs the **real trainer** on a 1200-row learnable cohort and persists a real held-out R².
- **GAP:** **No single test proves generate→simulate→persist→fidelity end-to-end against all-real components.** The "e2e" test (`test_digital_twin_e2e.py`) mocks the repository and fabricates the "actual" outcome (`simulated_ate*0.92/*0.95`) — inherent, because the platform has **no live cohort/outcome feed** for the real-world fidelity leg. API/route tests mock engine+generator+repo (so they'd pass against a stub). 7 of 8 retraining tests mock the trainer (testing fail-closed wiring).

---

## 4. End-to-End Reachability

**Path A — REST `/api/digital-twin/simulate` (FE-reachable, OPERATOR):**
`frontend/.../digital-twin.ts:81` POST → `main.py:1033` router → `digital_twin.py:616` build `TwinGenerator` → `:671` `generate()` → **`RuntimeError("Model not trained")` (twin_generator.py:255-256) → except (`:730`) → HTTP 500.** **Reachable in prod, fails closed (honest 500), persists nothing.** The FE never surfaces this anyway because it discards the response and shows `SAMPLE_SIMULATION`.

**Path B — `/simulations/compare`:** same untrained-generator → 500, additionally **unbounded** (no heavy-slot guard).

**Path C — experiment_designer agent (orchestrator-reachable):**
dispatcher → `ExperimentDesignerAgent.run` → `_create_initial_state` (no `enable_twin_simulation`) → graph → `twin_simulation.py:94` gate **False** → node short-circuits, **no simulation runs**. *If* the gate were forced True (tests only), the tool returns the **fabricated** `_create_mock_result` (deploy/fake-ATE/`fidelity_warning:False`). **Not reachable in prod traffic.**

**Path D — experiment_monitor agent (orchestrator-reachable, NEW):**
intent_classifier/router → `_agent_method_map.py:135` (`experiment_monitor → run`) → `ExperimentMonitorAgent.run_async` builds state (incl. `fidelity_threshold`) → `graph.ainvoke` → `interim_analyzer → fidelity_checker (:73) → alert_generator`. `FidelityCheckerNode` **self-resolves a real Supabase client** and **really reads** `twin_fidelity_tracking`. **Reachable in prod; returns real results — but the table is empty, so it honestly reports no fidelity issues (no fabrication).** This is the one twin consumer that is *both* prod-reachable *and* non-fabricating; it is simply starved of data because the persist path (H6) is broken.

**Path E — Celery offload (`simulate_population`, queue `twins`):** only enqueued when `HEAVY_OFFLOAD_ENABLED` truthy (dark) AND no `worker_heavy` exists → unreachable; *even if reached*, the runner has the same untrained-generator bug.

**Path F — Retraining (`execute_twin_retraining`, queue `ml`):** real, fail-closed, durable — but (i) `auto_trigger_retraining` defaults False with no live caller, (ii) no live cohort feed (must supply a `data_source` path via `config_overrides`), (iii) no `worker_heavy` to consume `ml`. **Wired and real, but never fires end-to-end in prod.**

**Net:** **No path reaches real twin computation + persistence in production.** The only *fully-real* compute that runs is the retraining trainer (tests only); the only *prod-reachable, non-fabricating* twin read is the experiment_monitor fidelity node, which reads empty data. All 4 DB tables are empty, confirming zero live throughput. The only fabricated values that reach a user come from the **frontend page** (Path A's FE shell), since the agent-tool mock (Path C) is gated off.

---

## 5. Findings & Classification

Priority order: **ACTIVE HARMFUL-NOW first, then LATENT/gated HARMFUL-NOW, then REWIRE, then placeholders/cleanup.**

| # | Finding | File:line | Classification | Reasoning (intent + harm) |
|---|---|---|---|---|
| **H1** | FE Results panel renders hardcoded `SAMPLE_SIMULATION`; real mutation result discarded | `DigitalTwin.tsx:319,325-334,561-563` | **HARMFUL-NOW (ACTIVE)** | Every user sees ate 0.18/ROI 3.2×/DEPLOY on load, after any run, and on history-click — no provenance/disclaimer. Directly user-visible fabricated decision, **no gate**. Highest current blast radius. |
| **H2** | FE `SAMPLE_HISTORY` fallback + static stat cards (2.4s/68%/87%) | `DigitalTwin.tsx:354,399-414` | **HARMFUL-NOW (ACTIVE)** | On the (currently empty) prod DB, fake history rows + vanity metrics render as if live, every page load. |
| **H3** | Agent tool `simulate_intervention` always returns fabricated ATE / "deploy" / `fidelity_warning:False` | `simulate_intervention_tool.py:138,244,309-368` | **HARMFUL-NOW (LATENT/GATED)** | Plausible pharma uplift (0.06–0.14) into agent state; intent comments confirm "would load a pre-trained model." **Reachable test-only** (gated by `enable_twin_simulation`, which no prod entrypoint sets — H8). A "fix-before-flip" item: it is **not** firing in prod traffic, but must precede any agent-pre-screen enablement. |
| **H4** | `/simulate` & `/simulations/compare` build an untrained `TwinGenerator`, never load a model → `RuntimeError` → 500 | `digital_twin.py:616,671,918,920,730-731`; `simulation_runner.py:74-77` vs guard `twin_generator.py:255-256` | **REWIRE (broken, fails closed)** | Functionality requested, persistence/`get_model` already exist; missing the model-load step. Fails *closed* (honest 500, **no fabricated value** — confirmed `:730-731`), so **not** HARMFUL-NOW; the feature is simply non-functional. **Fixing H4 without also fixing the heuristic (H5)/adding provenance would convert H5 into HARMFUL-NOW.** |
| **H5** | `INTERVENTION_EFFECTS` hardcoded base-effects + noise drive the ATE (not the ML model); no `data_provenance` label | `simulation_engine.py:71-100,441`; responses `digital_twin.py:238-460` | **REWIRE** (owner-confirmed 2026-06-04 — see §1.5) | **Owner intent = data-derived causal effects** for A/B pre-screening, so the hardcoded table is a stopgap. Redesign through a real uplift/CATE estimator (`causal_engine` / EconML) — no effect seam today, so this is architecture work, not a swap. *Interim:* add `data_provenance` label (still HARMFUL-NOW the moment H4 is fixed without it). |
| H6 | Live API: `TwinRepository()` built with no client → all core save/read no-op | `digital_twin.py:489…1417`; `twin_repository.py:839-845` | **REWIRE** | Persistence code + tables are real; the route just never injects a client. No fake values (honest degrade). Also starves the experiment_monitor fidelity node (H-mon) of data. |
| H7 | `FidelityTracker.validate` calls nonexistent `update_fidelity_record` + missing `await` | `fidelity_tracker.py:136,196` | **REWIRE** | Latent `AttributeError`/un-awaited coroutine; shadowed only by H6's 404. Must fix before H6. |
| H8 | `enable_twin_simulation` never set by prod agent entrypoint (Input→state plumbing gap) | `agent.py:40-83,425-444`; `state.py:214-215`; `graph.py:134`; `twin_simulation.py:94` | **REWIRE** | Real node, real edges, **output** state contract scaffolded (`state.py:214-234`); the missing piece is the **input** plumbing (no `Input` field, factory param inert). Keeps H3 dark. |
| H9 | `fidelity_tracking_update` orphaned + latent NameError (wraps real `compare_with_twin_prediction`) | `ab_testing_tasks.py:694-822` (bug `752-783`); body `results_analysis.py:516-586` | **REWIRE** | Real, audited service body; requested calibration feedback; but no producer + crashes on the `twin_simulation_id` branch (`predicted_effect`/`predicted_ci` unbound). |
| H10 | `/simulations/compare` lacks `heavy_compute_slot`/offload | `digital_twin.py:956-958` | **REWIRE** | OOM/DoS asymmetry vs `/simulate`; likely an oversight (medium conf). |
| H11 | Read-only GETs have no per-tenant/brand authZ | `digital_twin.py:740-1399`; `twin_repository.py:443-476,175-206` | **REWIRE** (owner-confirmed 2026-06-04 — see §1.5) | Oversight, not intent (lone unscoped brand-bearing read route vs. the `memory`/`graph`/`sentinels`/`explain` pattern). **Owner: add fail-closed `get_user_brands` scoping.** Single-operator tenancy → **low priority / proactive**; ~15-line fix + regression test, no change for admin/`['all']`. |
| H12 | `_save_to_mlflow` stub (real `log_model` commented) | `twin_repository.py:300-310` | **KEEP-AS-INTENTIONAL-PLACEHOLDER** | Intent-documented; unreachable in wired path (no `mlflow_client`); `mlflow_model_uri` persists None. |
| H13 | `get_recent_fidelity_records` computes cutoff, never filters | `twin_repository.py:766-776` | **KEEP-AS-INTENTIONAL-PLACEHOLDER** | Self-documented incomplete filter; returns real rows, ignores `days`. |
| H14 | `worker_heavy replicas:0` / dark offload | `docker-compose.yml:805-811` | **KEEP-AS-INTENTIONAL-PLACEHOLDER** | Documented OOM-headroom decision; inline path serves. |
| H15 | Celery routes `generate_twins`, `train_twin_model` (no task, no producer) | `celery_app.py:154,156` | **DELETE** | Vestigial/aspirational; no recoverable in-flight intent in code. *Investigate roadmap before deleting.* |
| H16 | `TwinPopulation.validate_size` no-op validator | `twin_models.py:177-181` | **REWIRE** (minor) | Docstring claims an invariant it doesn't enforce. |
| **H17** | experiment_monitor `_get_fidelity_from_simulation_summary` defined but never called; contains hardcoded `actual_effect=0.0` placeholder | `fidelity_checker.py:157-215` (placeholder `:203`) | **DELETE-or-WIRE** | Documented "alternative/fallback" with zero call sites (`execute()` calls only `_check_fidelity`). The `actual_effect=0.0` placeholder is currently **unreachable**. Either wire it as the intended `v_simulation_summary` fallback or remove. |

**Newly-surfaced REAL component (not a defect — recorded for completeness):** the experiment_monitor `FidelityCheckerNode` itself (`fidelity_checker.py:73,113-150`) is **REAL & WIRED**, orchestrator-reachable, self-resolves a real client, and reads `twin_fidelity_tracking` honestly. Its only limitation is **inert data** (empty table, downstream of H6). No fix needed on the node; it is gated functionally by the broken persist path.

---

## 6. Gaps vs Intent

1. **The flagship pre-screen never runs for a real user.** Intent (Phase-15: pre-screen experiments) vs reality: HTTP `/simulate` 500s; the designer agent node is dark; the FE shows a constant. The capability exists in pieces but is not connected into any live flow.
2. **The "ML-based" claim overstates the effect model.** Docs/route docstrings advertise "ML-based twin generation" and "results are real (not fabricated)" (`digital_twin.py:882`), but the ATE is a hardcoded-table heuristic; the sklearn model only sets `baseline_outcome`. The twins are ML; the *effect* is a hand-tuned formula. **Config and `SYNTHETIC_DATA.md` even disagree on the base-effect names/values** — neither is a measured outcome.
3. **No provenance labeling** anywhere on twin responses — diverges from the repo's own `/health-score` provenance pattern (#689).
4. **The monitoring half is wired but data-starved.** experiment_monitor's `FidelityCheckerNode` is real and prod-reachable but reads an empty `twin_fidelity_tracking` (downstream of the broken persist path, H6); its documented `v_simulation_summary` fallback is dead (H17). So fidelity monitoring cannot fire until simulations actually persist.
5. **Durable retraining is real but never fires end-to-end in prod** — auto-trigger default-off, no live cohort feed, no `worker_heavy`. (#549's durable store is genuinely correct and now *confirmed applied to prod*.)
6. **No all-real e2e test** for generate→simulate→persist→fidelity; the real-world fidelity leg is inherently untestable without a ground-truth outcome feed the platform doesn't ingest.
7. **Orphaned/dead Celery + fallback surfaces** (`generate_twins`/`train_twin_model` routes; `fidelity_tracking_update` task; the never-called `_get_fidelity_from_simulation_summary`) and the `_save_to_mlflow` stub mean the documented MLflow-lineage + post-experiment-calibration loops are not live.
8. **Fidelity surfaces span two sides and the boundary is unaudited:** twin-side views (`v_simulation_summary`, `v_model_fidelity_history`, `v_fidelity_degradation_alerts`) vs the A/B-side `ab_fidelity_comparisons` + its view `vw_fidelity_summary` (`database/ml/021_ab_results_tables.sql`, live on the droplet). The relationship between `twin_fidelity_tracking` and `ab_fidelity_comparisons` (dedup/superseded) remains **out of scope/open**.

---

## 7. Recommendations (prioritized, reason-first)

**P0 — Stop serving fabricated values on the ACTIVELY-reachable path (H1, H2), then close the gated one (H3).**
- **H1/H2 (FE — active, highest priority):** Wire the Results panel to the real `useRunSimulation` result and the selected history row to `useSimulation(id)` (the hook already exists, unused). Replace `SAMPLE_HISTORY` with a real empty-state. Remove or data-bind the static stat cards. *Until then,* given H4 means the backend 500s, the honest interim is to surface the backend error state rather than a fake DEPLOY — the current page actively *hides* a broken backend behind a plausible success. **These are the only fabrications reaching prod users today.**
- **H3 (agent tool — gated, fix-before-flip):** Either (a) gate the tool to raise/skip explicitly instead of fabricating (fail closed, matching the rest of the subsystem's #548 discipline), or (b) implement real model-load+`generate()` (intent comments request this). Do **not** ship `fidelity_warning:False` + "deploy" from a mock. Because H4/H8 keep it dark in prod today, this is lower current blast-radius than H1/H2, but it **must precede any agent-pre-screen enablement (H8)**.

**P1 — Make the compute path functional + safe (H4, H5, H6, H7).**
- **H4:** Add the missing model-load step in `/simulate`, `/simulations/compare`, and `simulation_runner` — load the persisted active model via `TwinModelRepository.get_model`/the saved artifact before `generate()`. The persistence half already exists.
- **H5 (gate on owner intent):** *Before* H4 ships, decide with the owner whether the `INTERVENTION_EFFECTS` heuristic is the permanent pre-screen or a stopgap. Either way, **add a `data_provenance`/`simulation_estimate` field** to the response models (mirror #689) so the heuristic ATE can never be mistaken for an empirical effect. **Do not fix H4 without H5's provenance**, or you convert a latent risk into a live HARMFUL-NOW.
- **H6 + H7:** Inject a Supabase client into `TwinRepository()` at the route layer (or give the core sub-repos the same `_ensure_async_client` lazy resolution the retraining repo has). This *also* unblocks the experiment_monitor fidelity node (which is otherwise correct but data-starved). Fix `FidelityTracker.validate` to call `update_fidelity_validation` with the right signature and `await` it — H7 must land *with or before* H6, or wiring a client surfaces the latent `AttributeError`.

**P2 — Close wiring/robustness gaps (H8, H9, H10, H11, H16, H17).**
- H8: add `enable_twin_simulation`/`intervention_type` to `ExperimentDesignerInput` + `_create_initial_state` (only after H3). The output state contract (`state.py:214-234`) already exists; this is the input half.
- H9: fix the NameError (`predicted_effect`/`predicted_ci` unbound on the `twin_simulation_id` branch) and decide a trigger (beat schedule or post-experiment hook) — or remove if not on the roadmap. The wrapped `compare_with_twin_prediction` body is real and reusable.
- H10: wrap `compare_scenarios` in `heavy_compute_slot()` + bounded executor (or offload), matching `/simulate`.
- H11: add brand/tenant scoping to the read GETs (or get an explicit owner decision recorded — current "intentional" is unverified).
- H16: enforce the `size==len(twins)` invariant or drop the misleading validator.
- H17: either wire `_get_fidelity_from_simulation_summary` as the intended `v_simulation_summary` fallback (replacing its `actual_effect=0.0` placeholder with a real read) or delete the dead method.

**P3 — Cleanup, intent permitting (H12, H13, H14, H15).**
- H15: confirm `generate_twins`/`train_twin_model` are not on the roadmap, then **delete** the dead routes (real training is `execute_twin_retraining`/`simulate_population`).
- H12/H13: leave as documented placeholders; implement `_save_to_mlflow` + the `days` filter when MLflow lineage / fidelity-window reporting is prioritized.
- H14: leave `worker_heavy` dark per the memory-documented OOM decision.

---

## 8. Confidence & Unverified

**High-confidence, faithfully verified (this host = prod droplet):**
- All 4 DB tables present + **0 rows** (re-run this session) + columns 1:1 + triggers live + **4 views live** (`v_active_twin_models`, `v_simulation_summary`, `v_model_fidelity_history`, `v_fidelity_degradation_alerts`) + ledger-recorded `012`/`029` (resolves the #549 prod-application open question — **029 is applied**).
- No `worker_heavy` container running (stronger than file `replicas:0`).
- `/simulate` `RuntimeError` runtime-disproven in the project venv; the route **fails closed** to HTTP 500 (`digital_twin.py:730-731`, no fabricated value) — so it is REWIRE, not HARMFUL-NOW. 56 engine tests re-run green.
- H1/H2 (FE) and H3 (agent tool) mock bodies re-read against source this session; H3's gate (`enable_twin_simulation` absent from `ExperimentDesignerInput`/`_create_initial_state`) confirmed → H3 is **gated/test-only**, not actively firing.
- experiment_monitor `FidelityCheckerNode` confirmed REAL & WIRED end-to-end (registered/mapped/routable; self-resolves real client; real read of `twin_fidelity_tracking`); returns no fidelity issues on the empty table (no fabrication).

**Unverified / residual uncertainty (faithful-env limits):**
- **Live HTTP exercise of `/simulate`** against the running container was not performed (traced statically + venv repro); a DI override injecting `generator.model` outside the files read is *very unlikely* but not exhaustively ruled out.
- **Whether the heuristic effect model is permanent or a stopgap (H5)** and **whether cross-brand reads (H11) are sanctioned** — both are **product-intent questions** that code cannot answer; git history shows only bulk feat/format commits, no targeted intent commit. **Recommend asking the owner** before any KEEP-vs-REWIRE action on these.
- **`HEAVY_OFFLOAD_ENABLED` live env value** confirmed only from compose defaults + the absent worker, not from the running container's env.
- **`ab_fidelity_comparisons` / `vw_fidelity_summary` (A/B side) vs `twin_fidelity_tracking` / `v_simulation_summary` (twin side)** dedup/overlap was out of scope; `vw_fidelity_summary` is confirmed live but traces to `database/ml/021` (A/B), not the twin migration.
- **No all-real e2e** exists, so the real-world *accuracy* of any twin prediction is unproven by construction — there is no ground-truth A/B-outcome feed; the "actual" in the e2e is hand-derived from the prediction (`*0.92/*0.95`). This is an inherent limitation, not a test bug.

---

## Audit method & confidence

This audit used a multi-agent, evidence-first method: per-layer reader agents fanned out over the eight implementation layers (engine, persistence, designer-agent, monitor-agent, route, celery, DB, frontend, tests), each emitting file:line-cited claims, followed by an adversarial **critic** pass that independently re-verified claims, hunted for missed components, and corrected severities. The critic caught four material issues the first-pass draft got wrong or missed, all incorporated here:

1. **A fourth live consumer was missing** — the experiment_monitor `FidelityCheckerNode` (orchestrator-reachable, self-resolves a real client, really reads `twin_fidelity_tracking`). Added to §3.3b, §4 (Path D), §5, §6.
2. **Severity correction on the compute route** — the draft's status table/header inherited a "HARMFUL-NOW" label for `/simulate` + `/simulations/compare`, contradicting its own H4/§8. The path **fails closed** (HTTP 500, no fabricated value, `digital_twin.py:730-731`) → corrected to **REWIRE/broken-fails-closed** throughout.
3. **Consumer taxonomy corrected** — the draft's "three live consumers, two fabricated" was inaccurate. Restated as an explicit inventory: FE = **active** fabrication; agent tool = **gated/latent** fabrication (test-only, not firing in prod); HTTP route = fails closed; monitor node = real-but-empty. H3 re-tiered below the active FE findings (H1/H2) to avoid overstating its current blast radius.
4. **Schema audit was incomplete** — `012` defines **4 SQL views** (all confirmed live on the droplet) that the draft never counted; added to §2, §3.6, §8. Also surfaced the never-called `_get_fidelity_from_simulation_summary` fallback (H17) and grounded H9 in the real `compare_with_twin_prediction` body it wraps.

**Faithful-environment strengths:** this host **is** the prod droplet, so all DB checks (table/view existence, 0-row counts, triggers, migration ledger) and the `docker ps` worker check are faithful to production. The 56-test engine suite and the `RuntimeError` repro were run in the project venv.

**Faithful-environment limitations:** the running API container was **not** exercised over live HTTP (the route bug was confirmed statically + via venv repro); `HEAVY_OFFLOAD_ENABLED`'s live container env value was not read directly; and two findings (H5 heuristic-permanence, H11 cross-brand-read sanction) are **product-intent questions** that code cannot resolve — both are flagged for owner confirmation rather than asserted. The real-world *accuracy* of twin predictions is unprovable in this environment by construction: there is no ground-truth A/B-outcome feed, so no all-real generate→simulate→persist→fidelity e2e can exist today.
