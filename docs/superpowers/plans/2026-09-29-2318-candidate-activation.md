# #2318 Retrain Candidate Activation (candidate → served) and Rollback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. One lane = one worktree = one PR. Every pytest is `pytest -n 0` and targeted. **Never run mypy on the droplet** (CLAUDE.md). No prod DB / MLflow / Redis / FalkorDB writes outside the owner-approved steps of Lane 8.

**Goal:** An owner can run `promote-candidate <registry id>` to make a retrained gold-standard model the served model — serving the exact trained artifact (no re-fit), gated on non-inferiority on the same holdout — and can roll it back by recorded ids, with DB, MLflow, the bundle store, the sidecar and the SHAP cache reconciled idempotently.

**Architecture:** A retrain logs a self-contained *serving bundle* (fitted preprocessor + model + column contract) to its MLflow run, with a sha256 tag. Activation downloads that exact file, verifies the hash, scores it and the currently served bundle on the same test ∪ holdout rows (DeLong non-inferiority), then atomically swaps the served bundle file, restarts the sidecar, verifies the sidecar now reports the new hash, and applies the registry/deployment/ledger changes in one Postgres transaction (RPC), then MLflow tags, then the SHAP cache slot. A ledger table (`ml_model_activations`, migration 165) records every id needed to roll back and is the desired state a re-run reconciles towards. The weekly reseed/retrain and the bundle re-materializer learn to leave an activated slot alone.

**Tech Stack:** Python 3.12, scikit-learn 1.6.1, MLflow 3.15.1 (REST, aliases + tags), Supabase/PostgREST + a SQL RPC, BentoML sidecar (`scripts/bentoml/e2i_serving_service.py`), scipy (DeLong normal approximation).

---

## 0. OWNER DECISIONS (read first)

Each item: my recommendation, why, and the single fact that would reverse it. Lanes that depend on a decision say so; everything else can start now.

**OD-1 — Which preprocessing does a servable retrain carry? (blocks Lane 2 and the adapter half of Lane 3)**
- **Recommendation: B — serve the trainer's own preprocessing.** The retrain logs the fitted sklearn `ColumnTransformer` it trained with (not the custom `ModelTrainerPreprocessor` wrapper, which the sidecar cannot import) plus the estimator, as one bundle. The sidecar gains a small raw-covariate adapter for a `ColumnTransformer` bundle.
- Why: the owner approved serving *the retrained model*, and the retrain's model is fitted on this encoding (F9, F10). A cheap probe showed this encoding is not worse on the goldstd harness. It reproduced the candidate's exact 15-feature shape (`ModelTrainerPreprocessor` over the 10 raw keep-columns plus the candidate's l1-LR hyper-parameters, fit on `train`) and scored **AUC 0.8517 vs the v1.0 artifact's 0.8509** on the same 1,766 test ∪ holdout rows (DeLong r = 0.991, SE(Δ) = 0.0012; scratch probe `p2318_delong_probe.py`, read-only). So the candidate's registry AUC of 0.8359 reflects a different split, not a worse model.
- Alternative A: make the trainer encode goldstd retrains with `FeatureBuilder` so the sidecar stays unchanged. This changes the trainer's preprocessing/HPO path for goldstd contracts (a larger blast radius in `model_trainer/nodes/preprocessor.py:206-236`, whose "already preprocessed" heuristic would not detect un-scaled FeatureBuilder output).
- **Reverses it:** the owner requires the Feature-Importance page to keep FeatureBuilder's `__isna` missingness features, or wants exactly one encoder contract in the sidecar. Then choose A.

**OD-2 — Non-inferiority margin and method (Lane 5)**
- **Recommendation:** gate on all four of the following.
  - **Primary — AUC non-inferiority.** H0: AUC_cand − AUC_served ≤ −δ, with **δ = 0.010 AUC**, tested by the **DeLong paired test for correlated ROC curves** (DeLong, DeLong & Clarke-Pearson 1988, fast algorithm of Sun & Xu 2014) on the identical rows at one-sided α = 0.05. Pass iff the one-sided 95% lower confidence bound of Δ > −δ.
  - **Brier non-inferiority** by paired bootstrap (B = 2000, fixed seed): the upper 95% bound of Brier_cand − Brier_served < **0.005**.
  - **Calibration slope** of the candidate in **[0.8, 1.25]**.
  - **Refuse** when the holdout has fewer than 100 positives or 100 negatives.
- Why DeLong: both models score the same rows, so the AUCs are strongly correlated (measured r = 0.991–0.999). An unpaired comparison throws that correlation away. DeLong is the standard nonparametric paired AUC test, and is closed-form, deterministic and cheap. Paired bootstrap is reported alongside as a cross-check; it agreed on the probe (bootstrap SE 0.0006 vs DeLong 0.0006).
- Why δ = 0.01: on this harness SE(Δ) is 0.0004–0.0012 (measured), so power is not the constraint; any δ ≥ 0.005 has power ≈ 1 at a true Δ = 0. The margin is therefore a *practical-relevance* choice: 0.01 AUC is below the week-to-week refit movement of the v1.0 row and far below the 0.05 "honest band" lift the trainer already enforces.
- **Reverses it:** the owner wants "no measurable loss" (then δ = 0.005), or wants a superiority requirement (then Δ's lower bound > 0).

**OD-3 — Comparator: what "the served model" means for the gate (Lane 5, Lane 6)**
- **Recommendation: compare against the bundle the sidecar is actually serving**, identified by its sha256 (Lane 3 makes the sidecar report it). Also report the v1.0 registry artifact's score.
- Why: the thing being replaced is the served bundle, and it is **not** the v1.0 registry artifact (F3, F4).
  - The served bundle was re-fit on *all* rows on 2026-08-12, so it is in-sample on the holdout. On the harness it scores 0.8521.
  - The registry artifact is re-fit weekly on train+validation and scores 0.8509.
  - The measured in-sample optimism is +0.0012 AUC, small against δ, and it biases the gate *against* the candidate (conservative).
- **Reverses it:** the owner wants the gate to compare to the honest registry number (0.8509). The code supports both via `--comparator served|registry`; only the default changes.

**OD-4 — CLI-only or also an API endpoint (Lane 6)**
- **Recommendation: CLI-only** (`python -m scripts.model_activation promote-candidate <id> --approved-by <name> [--execute]`), dry-run by default, run on the droplet host.
- Why:
  - Activation must `docker restart` the sidecar and write the host bind-mounted bundle dir. The API container has neither the docker socket nor a writable mount (`docker/docker-compose.yml:733-750` mounts the bundles `:ro` into the sidecar; the API does not mount them).
  - An HTTP endpoint would add an authz surface for a rare, owner-only action.
- **Reverses it:** the owner wants activation from the UI. That needs a job queue with a host-side executor, which is a separate design.

**OD-5 — Weekly reseed vs an activated slot (Lane 7)**
- The weekly cron will **silently revert** an activation (F14–F16). **Recommendation: the weekly goldstd retrain SKIPS any slot with an active activation.** It logs `SKIPPED (served by activated candidate <id>, activation <id>)` and leaves the archived v1.0 row, its metrics and the served bundle untouched. The bundle re-materializer refuses the slot the same way.
- Add a defensive guard as well: `register_model_row` refuses to upsert over a row whose stage is `archived` by an active activation.
- **Reverses it:** the owner wants the weekly job to keep re-fitting v1.0 as a standing challenger. Then use a stage-preserving upsert (keep `archived`, do not bump `registered_at`) instead of a skip. The re-materializer must refuse either way.

**OD-6 — hcp_adoption serves at `production`; the #968 gate refuses `synthetic_gold` → production (Lane 4, Lane 6)**
- **Recommendation:** extend the existing, owner-ruled hcp_adoption exemption (`scripts/promote_hcp_adoption_champions.py:38-62`) to activation, scoped to the three `hcp_adoption_*_goldstd_lr_v1` names, with the same calibration pathology gate. All other goldstd slots activate to `staging` (their served stage today, F12).
- Chat propensity also requires the champion row to have an `artifact_path` (`src/services/hcp_segment_likelihood.py:219-236`), so activation sets `artifact_path` to the versioned bundle path.
- **Reverses it:** the owner does not want a retrained synthetic_gold model at production. Then hcp_adoption slots are refused by `promote-candidate`.

**OD-7 — What a rolled-back candidate becomes (Lane 4, Lane 6)**
- **Recommendation: `archived`** (as the issue says), with the ledger row `rolled_back`. It cannot be re-activated; a new retrain is required.
- **Reverses it:** the owner wants to re-try the same artifact. Then roll back to `candidate` instead; the RPC takes the target stage as a parameter, so this is a one-line change.

**OD-8 — Existing candidates (rows `cff4f2b5…`, `faf4ed1d…`, `9376e88d…`, `a4d6f219…`, `44e1fe04…`)**
- **Recommendation: they are not activatable and should stay `candidate` (evidence only).** Their MLflow artifacts contain only the estimator; the fitted preprocessor was never persisted (F9), so serving them would need a re-fit, which the issue forbids.
- Lane 8 needs a fresh retrain after Lane 2 deploys.
- **Reverses it:** nothing. Reconstructing a preprocessor by re-fitting on today's rows is not the trained artifact, because the substrate grows weekly.

---

## 1. Established facts (every claim cited; all queries were read-only)

**Serving today**
- **F1.** The sidecar discovers goldstd bundles first from the BentoML store, then from the filesystem (`scripts/bentoml/e2i_serving_service.py:585-603`).
  - The BentoML store on prod is **empty** (`docker exec e2i_bentoml bentoml models list` → header only), so FS discovery serves.
  - FS discovery walks `data/ml_artifacts/shap_serving/**/<name>.bundle.pkl` and keeps the **first** file seen per name via `setdefault` (`:552-582`). `os.walk` order is not a contract, so **two files with the same name under that root are served nondeterministically**. A versioned archive must therefore live **outside** `shap_serving/`.
- **F2.** Bundles are keyed by `model_name` only, never by registry version (`:518-549`, `:565-573`). The container mounts the dir read-only and needs `src/mlops/gold_standard_eval` mounted to unpickle `FeatureBuilder` (`docker/docker-compose.yml:736-750`). **Anything pickled into a bundle must be importable in the sidecar**: sklearn and numpy classes, or `gold_standard_eval`. `src.agents.*` is not importable there.
- **F3.** The 12 served bundle files are all dated **2026-08-12 01:24** (`find /home/bentoml/data/ml_artifacts/shap_serving -name '*.pkl'` in the container). `scripts/rematerialize_goldstd_bundles.py:119-128` re-fits each one from `load_frame(db)` with default `splits=None`, i.e. on **all rows, including test/holdout**.
- **F4.** The weekly retrain does **not** refresh serving: `scripts/retrain_goldstd.sh:27-29` ("NOT included here … SHAP serving-bundle re-materialize + bentoml restart + SHAP cache refresh"). The v1.0 registry artifact `data/ml_artifacts/initiation_kisqali/initiation_kisqali_goldstd_lr_v1.pkl` is dated 2026-09-28 03:03. **So today the served model is not the registry row's artifact.**
- **F5.** The sidecar runs **scikit-learn 1.9.1** (`docker exec e2i_bentoml python -c 'import sklearn'`); the API and workers run **1.6.1**. The sidecar logs `InconsistentVersionWarning: Trying to unpickle estimator LogisticRegression from version 1.6.1 when using version 1.9.1` (3 lines in `docker logs e2i_bentoml`). `docker/bentoml/requirements-bentoml.txt` pins only `scikit-learn>=1.3.0`. "Serve the exact artifact" is not provable across a version skew, so Lane 3 pins it.
- **F6.** The sidecar's raw-covariate contract is duck-typed on `keep_columns` + `transform` (`e2i_serving_service.py:733-744`). SHAP needs a linear estimator: it unwraps `calibrated_classifiers_[0].estimator` (`:896-921`). Numeric-vs-categorical validation reads `preprocessor._numeric_medians` (`:972`).
  - Verified in the venv: a sklearn 1.6.1 `CalibratedClassifierCV(FrozenEstimator(LR))`'s inner `FrozenEstimator` exposes `coef_`, and `shap.LinearExplainer` accepts it.
- **F7.** API resolution:
  - predict and batch go by `model_name` to the sidecar (`src/api/routes/predictions.py:38-86` lists names from `stage IN ('production','staging')`).
  - `/explain/global` keys its SHAP cache (`ml_shap_analyses.model_registry_id`) on the **newest canonical** row for the name (`src/api/routes/explain.py:2239-2261`, `:2707-2720`). After activation, with the predecessor archived, that is the candidate's id: a cache miss, so it recomputes against the sidecar.
  - The KPI selector reads `stage='staging'` rows (`src/kpi/goldstd_model_perf.py:148-154`) and averages each one's `ml_performance_metrics source='holdout'` rows (`:157-166`). It does **not** de-duplicate by name (`:39-56`), so two staging rows for one name double-count.
- **F8.** Chat propensity requires the `hcp_adoption` row to be `stage='production' AND is_champion AND artifact_path IS NOT NULL AND NOT is_synthetic` (`src/services/hcp_segment_likelihood.py:205-240`). It then scores by `model_name` through the sidecar (`:256-290`).

**What a completed retrain leaves behind** (initiation_kisqali, `SELECT … FROM ml_model_registry WHERE model_name='initiation_kisqali_goldstd_lr_v1'`)
- **F9.** There are 1 predecessor row, 5 `candidate` rows and 1 `archived` row. Candidate `44e1fe04-a19d-458b-a3d9-cfbe0c1edced` looks like this:
  - Registry fields: version `1.0_retrained_20260929_2109_d4777d`, `retrain_of_id=4ec55d13…`, `mlflow_model_version=8`, `mlflow_run_id=142ea87f…`, `mlflow_model_uri=models:/m-2b475f52…`, `auc=0.8359`, `artifact_path=NULL`, `preprocessing_pipeline_path=NULL`.
  - MLflow logged model `m-2b475f52…`: `MLmodel`, `model.skops` (41,505 B), `conda.yaml`, `python_env.yaml`, `requirements.txt`. `serialization_format: skops`, `sklearn_version: 1.6.1`.
  - The skops tree is `CalibratedClassifierCV(method="sigmoid") → FrozenEstimator → LogisticRegression` with **`n_features_in_=15` and no `feature_names_in_`**. The run's own artifacts are four JSON dirs only.
  - **No preprocessor is persisted anywhere.** `_log_model_artifact` logs only `trained_model` (`src/agents/ml_foundation/model_trainer/nodes/mlflow_logger.py:344`, `:512-570`); the fitted preprocessor lives only in memory (`model_trainer/agent.py:455`, `:600`). **The artifact is not sufficient to serve without a re-fit.**
- **F10.** The 15 features come from `ModelTrainerPreprocessor`: mean-impute + StandardScaler + OneHot over the 10 raw keep-columns (`model_trainer/nodes/preprocessor.py:43-160`). The probe reproduced exactly 15 columns from the same 10 raw columns. The served v1.0 model is instead 25 FeatureBuilder columns with `__isna` indicators (loaded from the served bundle: `feature_columns` has 25 entries; `keep_columns` has the same 10 raw names). The two models are different encoders over **the same raw contract**.
- **F11.** MLflow version 8 is `current_stage='None'` with tags `e2i.role=candidate` and `e2i.retrain_of=4ec55d13…`. The registered model's `aliases` is `None`. The predecessor v1.0 row has **no MLflow version** (`mlflow_model_version`, `mlflow_run_id`, `mlflow_model_uri` all NULL).
- **F12.** Stage counts across goldstd names: `staging` 9, `production` 3 (the hcp_adoption champions, `training_provenance='synthetic_gold'`, `is_champion=true`), `candidate` 5, `archived` 1.
- **F13.** `ml_deployments` for the five candidate rows:
  - All are `status='registered'` with `environment` `staging` (2 older) or `candidate` (3). `environment` is `varchar`, and `deployment_status_enum` = `{pending,deploying,active,draining,rolled_back,failed,registered}`.
  - The **v1.0 predecessor has no deployment row at all**, so rollback cannot rely on `previous_deployment_id`.
  - There is no "one active per model" index; `idx_deployments_active` is a plain partial index.

**Reseed interaction**
- **F14.** The cron `0 3 * * 1 …/scripts/reseed_synthetic.sh` (crontab) runs `stage_goldstd_retrain` → `scripts/retrain_goldstd.sh` (`reseed_synthetic.sh:163-170`, `:214-220`). That re-registers all 12 v1.0 rows through `register_cohort_model` → `register_model_row`, an **upsert on `(model_name, model_version)` that writes `stage` and `registered_at=now`** (`src/mlops/prediction_synthesizer_deploy.py:348-387`; `src/mlops/gold_standard_eval/cohort_deployer.py:140-202`).
- **F15.** Consequence after an activation: the archived v1.0 row is written back to `staging` with the newest `registered_at`. It becomes the newest canonical row (`src/repositories/model_registry_roles.py:75-119`), which flips the SHAP-cache key and the Model-Performance reads back to v1.0. The KPI selector would count two `staging` rows for the name (F7). The sidecar would still serve the candidate bundle. **Silent split-brain.**
- **F16.** For hcp_adoption, `promote_hcp_adoption_champions.py --execute` runs next (`retrain_goldstd.sh:82-86`). It reads canonical rows and **raises on >1** (`scripts/promote_hcp_adoption_champions.py:245-270`). After F15 that is candidate (production) + v1.0 (staging), so it fails with a WARNING only. Also `run_persistence_eval`/`run_initiation_eval` delete the v1.0 row's holdout/backtest metrics before re-recording (`run_persistence_eval.py:250-271`).

**Stage machinery**
- **F17.** `ModelStage` exists twice without `candidate`: `src/repositories/ml_experiment.py:33-41` and `src/mlops/mlflow_connector.py:75-84`. The DB enum already has it: `model_stage_enum = {development,staging,shadow,production,archived,deprecated,candidate}` (migration 159). `normalize_stage("candidate")` therefore raises `ValueError` (`ml_experiment.py:1215-1237`).
- **F18.** There is **no model-stage transition table** in the repository layer. `transition_stage` (`ml_experiment.py:1239-1320`):
  - does not check the from-stage;
  - applies the #968/#2259 provenance gate only for `production`;
  - archives same-name predecessors **only when `new_stage == 'production'`** (`:1309-1318`).
  The only transition map is MLflow-stage-shaped: `ALLOWED_PROMOTIONS` in `src/agents/ml_foundation/model_deployer/nodes/registry_manager.py:1196-1202`.
- **F19.** `rollback_retraining` only sets `ml_retraining_history.status='rolled_back'` (`src/repositories/drift_monitoring.py:1414-1425`).
- **F20.** DB triggers:
  - `ensure_single_champion` demotes other champions **with the same `experiment_id`** (`SELECT prosrc … 'ensure_single_champion'`). The initiation_kisqali candidate and v1.0 share experiment `35a2cd41…`, so the trigger would demote v1.0 automatically. The RPC still sets it explicitly.
  - `tr_ml_model_registry_retrain_of_immutable` blocks updates of `retrain_of_id`.
- **F21.** The census test `tests/unit/test_repositories/test_registry_reader_census_2310.py` pins every `ml_model_registry` accessor in `src/` + `scripts/`, AST- and SQL-classified (`ACCESSORS` at `:40-…`; checks at `:310-540`). A new reader or writer fails CI until classified.

**Holdout harness**
- **F22.** The v1.0 registry `auc` is the holdout AUC of a model trained on `data_split IN ('train','validation')` and scored on `data_split IN ('test','holdout')` (`src/mlops/gold_standard_eval/run_persistence_eval.py:94-95`, `:197-241`; the same in `run_initiation_eval.py:66-67`). **Reproduced exactly (0.8509)** by the probe on 1,766 OOS rows (609 positives). Split sizes for Kisqali: train 5330, validation 1800, test 876, holdout 890.
- **F23.** The candidate trained on `train` only. Its `validation_*` metrics are over 1,800 rows (validation_accuracy 0.76444 = 1376/1800) and its `test_*` over 876, so `test ∪ holdout` is out-of-sample for it too. An artifact-scoring harness already exists: `scripts/promote_hcp_adoption_champions.py:295-340` (`_score_artifact`, `positive_class_scores` `:110`, `calibration_intercept` `:123`).
- **F24.** Frontier appends add rows to every split each week (max `journey_start_date` 2026-09-27/28 in all four). The OOS set therefore changes weekly. The gate must score both models on one snapshot and record `n`, positives and a hash of the sorted row ids.

**Migrations**
- **F25.** The highest file is `database/migrations/164_backfill_model_selector_episodic_selection.sql`. `public.schema_migrations` (`filename, applied_at`) latest row is `164_…` (2026-09-29 20:06). **Next free number: 165.** Peer worktrees `issue-2328-procedural-memory` and `issue-2331-test-prod-guard` are open and may also add a migration. Lane 4 re-checks `ls database/migrations | sort -V | tail -3` on `origin/main` immediately before opening its PR and renumbers if 165 is taken; the rollback file is renumbered too.

---

## 2. Contradictions with the design direction in the issue comment

1. **"Serve the exact artifact + preprocessing bundle"** — no retrain today persists a preprocessing bundle (F9). Activation of existing candidates is impossible without a re-fit (OD-8). **New prerequisite: Lane 2.**
2. **"served v1.0 registry AUC 0.8509"** — the served model is not the v1.0 registry artifact (F3, F4). The gate's comparator must be chosen explicitly (OD-3).
3. **"predecessor → archived"** — the weekly cron un-archives it within a week (F14–F16). **New lane: Lane 7**, which must deploy before the first prod activation.
4. **"MLflow tags `e2i.role` updated"** — the predecessor has no MLflow version (F11), so only the candidate side exists in MLflow. The plan uses MLflow 3 **aliases** (`served`) plus the `e2i.role` tag rather than deprecated stages.
5. **"`ml_deployments` row active; rollback by `deployment_id`"** — the predecessor has no deployment row (F13), so rollback needs a ledger of prior state (migration 165).
6. **"goldstd: staging; else production"** — hcp_adoption goldstd *is* production and blocked by #968 (OD-6).
7. Not in the comment but load-bearing:
   - sidecar sklearn skew (F5);
   - nondeterministic duplicate-name FS discovery (F1);
   - the KPI page needs `source='holdout'` metrics under the new row id (F7), which the gate computes and activation records;
   - the Time-Series walk-forward trend is not re-computed for the candidate. It shows only the holdout point until a follow-up; noted, not in scope.

---

## 3. Lanes, ordering, dependencies

| Lane | Title | Depends on | Migration | Prod write at merge? |
|---|---|---|---|---|
| **L1** | `candidate` in `ModelStage` + registry stage-transition rules | — | no | no |
| **L2** | Retrain logs a serving bundle + sha256 to MLflow | OD-1 | no | no (next retrain produces it) |
| **L3** | Sidecar: sklearn pin, bundle identity (`bundle_sha256`), ColumnTransformer raw adapter | OD-1 (adapter part only) | no | sidecar rebuild at deploy |
| **L4** | Migration 165: `ml_model_activations` ledger + `activate_model_candidate` / `rollback_model_activation` RPCs | L1 | **165** | owner applies migration |
| **L5** | Holdout harness + non-inferiority gate (DeLong + bootstrap) | OD-2, OD-3 | no | no |
| **L6** | `scripts/model_activation.py` CLI + reconciler (promote, rollback, status) | L1–L5 | no | no |
| **L7** | Reseed/re-materializer/hcp-promote guards for activated slots | L4 | no | no |
| **L8** | Live verification script + prod procedure | L1–L7 deployed | no | **yes — owner-approved in-turn** |

- L1, L3 and L5 start immediately in parallel.
- L2 starts as soon as OD-1 is answered.
- L4 after L1 merges, because it uses the enum member in Python tests.
- L6 after L1–L5 merge.
- L7 after L4 merges; it runs in parallel with L6.
- **L7 must be deployed before L8's first activation** (otherwise the next Monday 03:00Z cron reverts it).
- L8 last. It also needs a fresh retrain run after L2 is deployed.

File map (created or modified):

```
L1  src/repositories/ml_experiment.py              (ModelStage, STAGE_TRANSITIONS, transition_stage guard)
    src/mlops/mlflow_connector.py                  (ModelStage.CANDIDATE)
    tests/unit/test_repositories/test_model_stage_transitions_2318.py   (new)
L2  src/agents/ml_foundation/model_trainer/nodes/serving_bundle.py      (new: build + log bundle)
    src/agents/ml_foundation/model_trainer/nodes/mlflow_logger.py       (call it after _log_model_artifact)
    tests/unit/test_agents/test_ml_foundation/test_model_trainer/test_serving_bundle_2318.py (new)
L3  docker/bentoml/requirements-bentoml.txt        (scikit-learn==1.6.1)
    scripts/bentoml/e2i_serving_service.py          (_ColumnTransformerRawEncoder, sha256 identity)
    tests/unit/test_serving/test_bentoml_bundle_identity_2318.py        (new)
    tests/unit/test_docker/test_compose_runtime_invariants.py           (sklearn parity pin)
L4  database/migrations/165_model_activation_ledger.sql                 (new)
    database/migrations/rollback_165_model_activation_ledger.sql        (new)
    tests/unit/test_database/test_model_activation_rpc_realdb_2318.py   (new, ThrowawayPg)
    tests/unit/test_repositories/test_registry_reader_census_2310.py    (classify the RPCs)
L5  src/mlops/activation/__init__.py, holdout_gate.py                   (new)
    tests/unit/test_mlops/test_activation_holdout_gate_2318.py          (new)
L6  src/mlops/activation/bundle_store.py, reconcile.py                  (new)
    scripts/model_activation.py                                         (new CLI)
    tests/unit/test_mlops/test_activation_bundle_store_2318.py          (new)
    tests/unit/test_mlops/test_activation_reconcile_2318.py             (new)
    tests/unit/test_repositories/test_registry_reader_census_2310.py    (classify new accessors)
L7  src/mlops/activation/guards.py                                      (new: active_activation_for)
    src/mlops/gold_standard_eval/run_persistence_eval.py, run_initiation_eval.py, run_hcp_cohorts.py (skip slot)
    scripts/rematerialize_goldstd_bundles.py, scripts/promote_hcp_adoption_champions.py (refuse slot)
    src/mlops/prediction_synthesizer_deploy.py::register_model_row      (defensive refusal)
    tests/unit/test_mlops/test_activation_guards_2318.py                (new)
L8  scripts/verify_model_activation_live.py                             (new)
    docs/demos/results/<date>_2318_activation_live/                    (evidence)
```

---

## Lane 1 — `candidate` in `ModelStage` + transition rules

**Why:** `normalize_stage("candidate")` raises (F17). `transition_stage` would let any caller promote a candidate straight to `staging`, archiving nothing (F18). That is exactly the silent path the owner wants gated. The rule: **a `candidate` row may only become `archived` through the generic path. `candidate → staging|production` happens only inside `activate_model_candidate` (L4).**

### Task 1.1: `ModelStage.CANDIDATE` in both enums

**Files:** Modify `src/repositories/ml_experiment.py:33-41`, `src/mlops/mlflow_connector.py:75-84`. Test `tests/unit/test_repositories/test_model_stage_transitions_2318.py`.

- [ ] **Step 1: Write the failing test**

```python
"""#2318: `candidate` is a first-class model stage, and only activation may promote it."""

import pytest

from src.mlops.mlflow_connector import ModelStage as ConnectorStage
from src.repositories.ml_experiment import MLModelRegistryRepository, ModelStage


def test_candidate_is_a_model_stage_in_both_enums():
    assert ModelStage.CANDIDATE.value == "candidate"
    assert ConnectorStage.CANDIDATE.value == "candidate"
    # Both enums mirror model_stage_enum (migration 159) exactly.
    assert {s.value for s in ModelStage} == {s.value for s in ConnectorStage} == {
        "development", "staging", "shadow", "production", "archived", "deprecated", "candidate",
    }


def test_normalize_stage_accepts_candidate():
    assert MLModelRegistryRepository.normalize_stage("candidate") == "candidate"
    assert MLModelRegistryRepository.normalize_stage("Candidate") == "candidate"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest -n 0 tests/unit/test_repositories/test_model_stage_transitions_2318.py -q`
Expected: FAIL, `AttributeError: CANDIDATE`.

- [ ] **Step 3: Implement.** In both enums add after `DEPRECATED`:

```python
    CANDIDATE = "candidate"  # #2310 retrain awaiting review; promoted only by activation (#2318)
```

- [ ] **Step 4: Run to verify it passes.** Same command. Expected: PASS.

### Task 1.2: Stage-transition rules; `transition_stage` refuses candidate promotion

**Files:** Modify `src/repositories/ml_experiment.py` (add `STAGE_TRANSITIONS` near `ModelStage`; guard at the top of `transition_stage` after `get_by_id`, `:1287-1291`). Same test file.

- [ ] **Step 1: Write the failing tests** (append):

```python
from unittest.mock import AsyncMock, MagicMock

from src.repositories.ml_experiment import STAGE_TRANSITIONS, StageTransitionRefused


def _repo_with_current(stage: str):
    repo = MLModelRegistryRepository.__new__(MLModelRegistryRepository)
    repo.client = MagicMock()
    repo.table_name = "ml_model_registry"
    current = MagicMock(stage=stage, model_name="m_goldstd_lr_v1", training_provenance="real")
    repo.get_by_id = AsyncMock(return_value=current)
    return repo


def test_candidate_may_only_be_archived_generically():
    assert STAGE_TRANSITIONS["candidate"] == frozenset({"archived"})


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["staging", "production", "shadow", "development"])
async def test_transition_stage_refuses_promoting_a_candidate(target):
    repo = _repo_with_current("candidate")
    with pytest.raises(StageTransitionRefused, match="promote-candidate"):
        await repo.transition_stage("00000000-0000-0000-0000-000000000001", target)
    repo.client.table.assert_not_called()  # refused before any write


@pytest.mark.asyncio
async def test_transition_stage_still_archives_a_candidate():
    repo = _repo_with_current("candidate")
    repo.client.table.return_value.update.return_value.eq.return_value.execute = AsyncMock(
        return_value=MagicMock(data=[{"id": "x"}])
    )
    assert await repo.transition_stage("00000000-0000-0000-0000-000000000001", "archived")


@pytest.mark.asyncio
async def test_nobody_transitions_into_candidate_generically():
    repo = _repo_with_current("staging")
    with pytest.raises(StageTransitionRefused):
        await repo.transition_stage("00000000-0000-0000-0000-000000000001", "candidate")
```

- [ ] **Step 2: Run.** Expected: FAIL (`ImportError: STAGE_TRANSITIONS`).

- [ ] **Step 3: Implement** (after the enums in `ml_experiment.py`):

```python
class StageTransitionRefused(ValueError):
    """A registry stage change the generic path must not make (#2318)."""


#: Stage changes ``transition_stage`` may make, by current stage. Only the rows that need a
#: rule are listed; any stage not listed keeps today's behaviour. A ``candidate`` enters service
#: only through ``activate_model_candidate`` (migration 165, scripts/model_activation.py), which
#: gates it on the holdout and records how to undo it; the generic path may only retire it.
#: Nothing enters ``candidate`` generically: the retrain deployer inserts it (#2310).
STAGE_TRANSITIONS: dict[str, frozenset[str]] = {
    "candidate": frozenset({"archived"}),
}
```

In `transition_stage`, immediately after `current = await self.get_by_id(...)` / `if not current: return False`:

```python
        from_stage = (current.stage.value if hasattr(current.stage, "value") else current.stage) or ""
        allowed = STAGE_TRANSITIONS.get(from_stage)
        if allowed is not None and new_stage not in allowed:
            raise StageTransitionRefused(
                f"model {model_id} is a '{from_stage}'; the generic path may only move it to "
                f"{sorted(allowed)}. Use scripts/model_activation.py promote-candidate (#2318)."
            )
        if new_stage == "candidate":
            raise StageTransitionRefused(
                "no generic transition into 'candidate'; the retrain deployer registers it (#2310)"
            )
```

- [ ] **Step 4: Run** the file plus the neighbouring suites that exercise `transition_stage`:

`pytest -n 0 tests/unit/test_repositories/test_model_stage_transitions_2318.py tests/unit/test_repositories -k "transition or stage or promot" -q`
Expected: PASS. If an existing test transitions from `candidate`, read it: a test promoting a candidate generically is a finding to report, not to delete.

- [ ] **Step 5: Teeth.** Temporarily change `STAGE_TRANSITIONS["candidate"]` to include `"staging"`, confirm `test_transition_stage_refuses_promoting_a_candidate[staging]` fails, then restore. Verify the plant landed: the test must go red.

- [ ] **Step 6: Commit** — `feat(registry): candidate stage + generic transition refuses candidate promotion (Part of #2318)`.

**Gates:** `ruff check --no-cache` on changed files; `ruff format --check`; CI mypy is the type arbiter (do not run it locally).

---

## Lane 2 — Retrain logs a serving bundle (requires OD-1; written for recommendation B)

**Why:** F9 — the preprocessor is never persisted, so no candidate can be served without a re-fit.

**Bundle contract (`bundle_format = "sklearn_ct_v1"`).** A plain dict of sklearn/numpy/builtins only (F2: loadable in the sidecar):

```python
{
  "bundle_format": "sklearn_ct_v1",
  "model": <fitted estimator, e.g. CalibratedClassifierCV>,
  "preprocessor": <fitted sklearn ColumnTransformer = ModelTrainerPreprocessor._pipeline>,
  "keep_columns": [<raw input column names, in fit order>],
  "numeric_columns": [<subset of keep_columns treated as numeric>],
  "feature_columns": [<encoded names, 'num__'/'cat__' prefixes stripped>],
  "sklearn_version": sklearn.__version__,
}
```

- `feature_columns` has its prefixes stripped so the API's covariate collapse (`scripts/sync_goldstd_serving.py:167-176`, `split("__")[0]` then the region/specialty/insurance prefix rule) maps `geographic_region_south` → `geographic_region`.
- The bundle is logged as run artifact `serving_bundle/bundle.pkl`, with run tag `e2i.serving_bundle_sha256=<hex>`.

### Task 2.1: `build_serving_bundle` (pure)

**Files:** Create `src/agents/ml_foundation/model_trainer/nodes/serving_bundle.py`. Test `tests/unit/test_agents/test_ml_foundation/test_model_trainer/test_serving_bundle_2318.py`.

- [ ] **Step 1: Failing test** — fit a real `ModelTrainerPreprocessor` + calibrated LR on a small frame with the 10 goldstd raw columns (dtypes as measured: `disease_severity` float64, `geographic_region`/`insurance_type` object, the rest int64):

```python
import hashlib
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from src.agents.ml_foundation.model_trainer.nodes.preprocessor import ModelTrainerPreprocessor
from src.agents.ml_foundation.model_trainer.nodes.serving_bundle import (
    BUNDLE_FORMAT,
    build_serving_bundle,
    serialize_bundle,
)

RAW = ["disease_severity", "academic_hcp", "geographic_region", "insurance_type",
       "age_at_diagnosis", "comorbidity_burden", "prior_therapy_lines",
       "rep_detailing_high", "sample_dropped", "trigger_accepted"]


def _frame(n=300, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "disease_severity": r.normal(5, 2, n), "academic_hcp": r.integers(0, 2, n),
        "geographic_region": r.choice(["northeast", "south", "midwest", "west"], n),
        "insurance_type": r.choice(["commercial", "medicare", "medicaid"], n),
        "age_at_diagnosis": r.integers(20, 80, n), "comorbidity_burden": r.integers(0, 5, n),
        "prior_therapy_lines": r.integers(0, 4, n), "rep_detailing_high": r.integers(0, 2, n),
        "sample_dropped": r.integers(0, 2, n), "trigger_accepted": r.integers(0, 2, n),
    })


def _fitted():
    X = _frame(); y = (X["disease_severity"] + X["academic_hcp"] > 5.5).astype(int)
    pre = ModelTrainerPreprocessor().fit(X)
    model = CalibratedClassifierCV(LogisticRegression(max_iter=500), method="sigmoid").fit(pre.transform(X), y)
    return pre, model, X


def test_bundle_is_sklearn_only_and_predicts_identically():
    pre, model, X = _fitted()
    bundle = build_serving_bundle(model=model, preprocessor=pre)
    assert bundle["bundle_format"] == BUNDLE_FORMAT
    assert isinstance(bundle["preprocessor"], ColumnTransformer)  # not the src.agents wrapper
    assert bundle["keep_columns"] == RAW
    assert set(bundle["numeric_columns"]) == set(RAW) - {"geographic_region", "insurance_type"}
    assert not any(c.startswith(("num__", "cat__")) for c in bundle["feature_columns"])
    assert "geographic_region_south" in bundle["feature_columns"]
    assert len(bundle["feature_columns"]) == pre.transform(X).shape[1]
    # Round-trip through bytes: predictions are bit-identical to the in-memory pair.
    blob, sha = serialize_bundle(bundle)
    assert sha == hashlib.sha256(blob).hexdigest()
    loaded = pickle.loads(blob)  # noqa: S301 - test of our own artifact
    np.testing.assert_array_equal(
        loaded["model"].predict_proba(loaded["preprocessor"].transform(X[RAW])),
        model.predict_proba(pre.transform(X)),
    )


def test_bundle_pickle_references_no_src_module():
    pre, model, _ = _fitted()
    blob, _ = serialize_bundle(build_serving_bundle(model=model, preprocessor=pre))
    assert b"src.agents" not in blob and b"src.mlops" not in blob  # F2: sidecar cannot import them


def test_refuses_an_unfitted_or_foreign_preprocessor():
    import pytest
    _, model, _ = _fitted()
    with pytest.raises(ValueError, match="fitted ModelTrainerPreprocessor"):
        build_serving_bundle(model=model, preprocessor=ModelTrainerPreprocessor())
    with pytest.raises(ValueError, match="fitted ModelTrainerPreprocessor"):
        build_serving_bundle(model=model, preprocessor=object())
```

- [ ] **Step 2: Run.** `pytest -n 0 tests/unit/test_agents/test_ml_foundation/test_model_trainer/test_serving_bundle_2318.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
"""Serving bundle for a trained model (#2318): the exact fitted preprocessing + estimator.

A retrain candidate can only be served as-trained if its preprocessing is persisted with it
(before #2318 only the estimator was logged, so serving it needed a re-fit). The bundle holds
sklearn/numpy/builtins only, so the BentoML sidecar, which cannot import ``src.*``, can load it.
"""

from __future__ import annotations

import hashlib
import pickle
from typing import Any

import sklearn

BUNDLE_FORMAT = "sklearn_ct_v1"
BUNDLE_ARTIFACT_DIR = "serving_bundle"
BUNDLE_FILENAME = "bundle.pkl"
BUNDLE_SHA_TAG = "e2i.serving_bundle_sha256"


def _strip(name: str) -> str:
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def build_serving_bundle(*, model: Any, preprocessor: Any) -> dict[str, Any]:
    ct = getattr(preprocessor, "_pipeline", None)
    if ct is None or not getattr(preprocessor, "_is_fitted", False):
        raise ValueError("serving bundle needs a fitted ModelTrainerPreprocessor")
    return {
        "bundle_format": BUNDLE_FORMAT,
        "model": model,
        "preprocessor": ct,
        "keep_columns": [str(c) for c in ct.feature_names_in_],
        "numeric_columns": [str(c) for c in preprocessor.numeric_features],
        "feature_columns": [_strip(str(n)) for n in ct.get_feature_names_out()],
        "sklearn_version": sklearn.__version__,
    }


def serialize_bundle(bundle: dict[str, Any]) -> tuple[bytes, str]:
    blob = pickle.dumps(bundle, protocol=5)
    return blob, hashlib.sha256(blob).hexdigest()
```

  (If `ModelTrainerPreprocessor` exposes the fitted flag under another name, use it. Read `preprocessor.py:99-160` first; do not guess.)

- [ ] **Step 4: Run.** Expected: PASS.

### Task 2.2: Log the bundle from the trainer's MLflow logger

**Files:** Modify `src/agents/ml_foundation/model_trainer/nodes/mlflow_logger.py` (right after `_log_model_artifact`, `:344`). Test (same file) with a fake `run` that records `log_artifact` / `set_tags` calls. This fakes the MLflow boundary, not business logic.

- [ ] **Step 1: Failing test**

```python
import pytest

from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import _log_serving_bundle


class _Run:
    def __init__(self):
        self.artifacts, self.tags = [], {}

    async def log_artifact(self, local_path, artifact_path=None):
        self.artifacts.append((open(local_path, "rb").read(), artifact_path, local_path))

    async def set_tags(self, tags):
        self.tags.update(tags)


@pytest.mark.asyncio
async def test_logs_bundle_and_its_sha():
    pre, model, _ = _fitted()
    run = _Run()
    sha = await _log_serving_bundle(run, {"preprocessor": pre}, model)
    (blob, art_dir, local), = run.artifacts
    assert art_dir == "serving_bundle" and local.endswith("/bundle.pkl")
    assert run.tags["e2i.serving_bundle_sha256"] == sha == hashlib.sha256(blob).hexdigest()


@pytest.mark.asyncio
async def test_no_preprocessor_logs_nothing_and_says_why(caplog):
    _, model, _ = _fitted()
    run = _Run()
    assert await _log_serving_bundle(run, {}, model) is None
    assert run.artifacts == [] and "e2i.serving_bundle_sha256" not in run.tags
    assert "not servable as-trained" in caplog.text
```

- [ ] **Step 2: Run** → FAIL (import).

- [ ] **Step 3: Implement** in `mlflow_logger.py`:

```python
async def _log_serving_bundle(run: Any, state: Dict[str, Any], model: Any) -> Optional[str]:
    """Log the exact preprocessing + estimator as one servable bundle (#2318); return its sha256.

    Without it a candidate cannot be activated: serving it would need a re-fit. A failure is
    logged, never raised: the training run itself succeeded, and activation refuses a candidate
    whose run carries no bundle tag.
    """
    from .serving_bundle import (
        BUNDLE_ARTIFACT_DIR, BUNDLE_FILENAME, BUNDLE_SHA_TAG, build_serving_bundle, serialize_bundle,
    )

    preprocessor = state.get("preprocessor")
    if preprocessor is None:
        logger.warning("No fitted preprocessor in state: this model is not servable as-trained (#2318)")
        return None
    try:
        blob, sha = serialize_bundle(build_serving_bundle(model=model, preprocessor=preprocessor))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, BUNDLE_FILENAME)
            with open(path, "wb") as fh:
                fh.write(blob)
            await run.log_artifact(path, BUNDLE_ARTIFACT_DIR)
        await run.set_tags({BUNDLE_SHA_TAG: sha})
        return sha
    except Exception as e:  # noqa: BLE001
        logger.error("Serving bundle NOT logged (%s): the candidate will not be activatable", e)
        return None
```

  Call it right after `model_uri = await _log_model_artifact(...)` (`:344`): `await _log_serving_bundle(run, state, trained_model)`. Add `import os` if absent.

- [ ] **Step 4: Run** the new file plus `pytest -n 0 tests/unit/test_agents/test_ml_foundation/test_model_trainer -k mlflow_logger -q` → PASS.

- [ ] **Step 5: Faithfulness probe (read-only, no MLflow write).** In a scratch script, run `build_serving_bundle` on a `ModelTrainerPreprocessor` fit over the live Kisqali `train` rows (`FeatureBuilder(spec).load_frame(db, splits=("train",))[keep_columns]`). Assert `len(feature_columns) == 15` (F10). This proves the bundle matches what the live retrain produces.

- [ ] **Step 6: Commit** — `feat(trainer): log the exact preprocessing+estimator as a serving bundle with sha256 (Part of #2318)`.

---

## Lane 3 — Sidecar: version parity, bundle identity, ColumnTransformer adapter

**Why:** F5 (skew), F1 (no identity: nothing proves which file is served), F6 (raw contract only for FeatureBuilder). The adapter half depends on OD-1 = B. The pin and identity halves are needed regardless.

### Task 3.1: Pin sidecar scikit-learn to the training version

**Files:** `docker/bentoml/requirements-bentoml.txt`; test `tests/unit/test_docker/test_compose_runtime_invariants.py` (add a test).

- [ ] **Step 1: Failing test**

```python
def test_sidecar_sklearn_matches_the_training_pin():
    """#2318: bundles are pickled by workers on the repo's sklearn; the sidecar must unpickle on
    the SAME version (it ran 1.9.1 vs 1.6.1 and logged InconsistentVersionWarning)."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    side = (root / "docker/bentoml/requirements-bentoml.txt").read_text()
    m = re.search(r"^scikit-learn==([\d.]+)\s*$", side, re.M)
    assert m, "sidecar must pin scikit-learn exactly"
    repo_pins = (root / "requirements.txt").read_text()  # confirm the repo's pin file name first
    assert re.search(rf"^scikit-learn==({re.escape(m.group(1))})\b", repo_pins, re.M)
```

  (Confirm where the repo pins sklearn — `grep -rn "scikit-learn" requirements*.txt pyproject.toml` — and point the test at that file.)

- [ ] **Step 2: Run** → FAIL. **Step 3:** set `scikit-learn==1.6.1`. **Step 4:** PASS.

- [ ] **Step 5: Cheapest disproof before merge** (read-only, local image build is heavy — prefer CI's image build). Confirm `shap==<sidecar pin>` supports sklearn 1.6.1: `pip download --no-deps` metadata, or the CI build log. If shap requires newer sklearn, stop and report; do not unpin.

### Task 3.2: Bundle identity — the sidecar reports each bundle's sha256

**Files:** `scripts/bentoml/e2i_serving_service.py` (`_discover_goldstd_bundles_from_fs` `:552-582`, `_unwrap_bundle` `:507-515`, `model_info` `:1470-1543`, `PredictionOutput.model_id` construction in the routed paths). Test `tests/unit/test_serving/test_bentoml_bundle_identity_2318.py` (uses the `serving_module` fixture, `tests/unit/test_serving/conftest.py:82-101`).

- [ ] **Step 1: Failing tests**

```python
import hashlib
import pickle


def _write(tmp_path, name, bundle):
    d = tmp_path / "initiation"; d.mkdir(exist_ok=True)
    p = d / f"{name}.bundle.pkl"; p.write_bytes(pickle.dumps(bundle))
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_fs_discovery_records_the_file_sha(serving_module, tmp_path, fitted_goldstd_bundle):
    sha = _write(tmp_path, "initiation_kisqali_goldstd_lr_v1", fitted_goldstd_bundle)
    found = serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))
    assert found["initiation_kisqali_goldstd_lr_v1"]["bundle_sha256"] == sha


def test_duplicate_names_under_the_root_fail_closed(serving_module, tmp_path, fitted_goldstd_bundle):
    """F1: os.walk order is not a contract; two files for one name must not be served at random."""
    _write(tmp_path, "initiation_kisqali_goldstd_lr_v1", fitted_goldstd_bundle)
    other = tmp_path / "archive"; other.mkdir()
    (other / "initiation_kisqali_goldstd_lr_v1.bundle.pkl").write_bytes(pickle.dumps(fitted_goldstd_bundle))
    found = serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))
    assert "initiation_kisqali_goldstd_lr_v1" not in found
```

  Add a third test: `model_info(ModelInfoInput(model_name=...))` returns `bundle_sha256`, and `/predict` routed to that name returns `model_id == f"{name}@{sha[:12]}"`. Build `fitted_goldstd_bundle` as a fixture exactly like `_fit_bundle` in `tests/unit/test_serving/test_bentoml_multimodel.py:26-60`.

- [ ] **Step 2: Run** `pytest -n 0 tests/unit/test_serving/test_bentoml_bundle_identity_2318.py -q` → FAIL.

- [ ] **Step 3: Implement.**
  - In the FS walk, read bytes once: `data = fh.read(); obj = pickle.loads(data)`, then attach `entry["bundle_sha256"] = hashlib.sha256(data).hexdigest()`.
  - Collect paths per name. When a name is seen twice, log `ERROR` naming both paths and **drop the name** (fail closed: `/predict` for it then 4xx's through the existing unknown-name path).
  - `_unwrap_bundle` keeps `bundle_sha256` when present.
  - `model_info` adds `"bundle_sha256"`, and the routed `PredictionOutput.model_id` becomes `f"{name}@{sha[:12]}"` when a sha is known. Check `src/api/dependencies/bentoml_client.py` and `predictions.py:646,731` read `model_id` as an opaque string. Grep for any equality check on it first; if one exists, keep `model_id` and add a separate `bundle_sha256` field instead.

- [ ] **Step 4: Run** the new file plus the whole serving dir: `pytest -n 0 tests/unit/test_serving -q` → PASS (the multimodel/raw/SHAP suites must not regress).

### Task 3.3 (OD-1 = B): `_ColumnTransformerRawEncoder` adapter

**Files:** same service module; test (same new file).

- [ ] **Step 1: Failing test.** Load an `sklearn_ct_v1` bundle built by Lane 2's `build_serving_bundle` (import it in the test; the *test* may import `src`, the sidecar may not). Then assert:
  - `/predict` with `raw_features` equals `bundle["model"].predict_proba(bundle["preprocessor"].transform(df))[:, 1]` **exactly**;
  - `/shap` returns `encoded_feature_columns == bundle["feature_columns"]` and satisfies additivity (`base + sum(shap) ≈ inner-LR margin`, as the existing `test_bentoml_shap.py` does);
  - `model_info.keep_columns == bundle["keep_columns"]`;
  - a string for a numeric column fails closed.

- [ ] **Step 2: Run** → FAIL (raw path raises "no FeatureBuilder preprocessor").

- [ ] **Step 3: Implement** (self-contained in the service; sklearn/pandas only):

```python
class _ColumnTransformerRawEncoder:
    """Raw-covariate contract for an ``sklearn_ct_v1`` bundle (#2318).

    Presents the duck-type the serving paths already use for FeatureBuilder: ``keep_columns``,
    ``transform(raw_df) -> DataFrame[feature_columns]`` and ``_numeric_medians`` (only its KEYS
    are read, for numeric-vs-categorical validation). Built at load time, never pickled.
    """

    def __init__(self, ct: Any, keep_columns: List[str], numeric_columns: List[str], feature_columns: List[str]):
        self._ct = ct
        self.keep_columns = tuple(keep_columns)
        self._numeric_medians = {c: None for c in numeric_columns}
        self._feature_columns = list(feature_columns)

    def transform(self, raw: Any) -> Any:
        import pandas as pd
        missing = [c for c in self.keep_columns if c not in raw.columns]
        if missing:
            raise ValueError(f"raw_features missing covariate(s): {missing}")
        out = self._ct.transform(raw[list(self.keep_columns)])
        return pd.DataFrame(out, columns=self._feature_columns, index=raw.index)
```

  In `_unwrap_bundle`, when `obj.get("bundle_format") == "sklearn_ct_v1"`, return the entry with `preprocessor=_ColumnTransformerRawEncoder(obj["preprocessor"], obj["keep_columns"], obj["numeric_columns"], obj["feature_columns"])`. `_is_goldstd_bundle_dict` already accepts the dict (it has `model`, `preprocessor`, `feature_columns`).

- [ ] **Step 4: Run** `pytest -n 0 tests/unit/test_serving -q` → PASS.
- [ ] **Step 5: Teeth.** Break the adapter's column order (reverse `keep_columns` in `transform`), confirm the exact-equality test fails, then restore.
- [ ] **Step 6: Commit** — `feat(serving): sklearn parity pin, per-bundle sha256 identity, ColumnTransformer raw contract (Part of #2318)`.

**Deploy note:** the sidecar image is rebuilt by the deploy (Dockerfile copies `scripts/bentoml/e2i_serving_service.py`). After deploy, check `docker logs e2i_bentoml | grep -c InconsistentVersionWarning` → 0. Also check `/model_info` for each of the 12 names now returns a `bundle_sha256` (read-only POST).

---

## Lane 4 — Migration 165: activation ledger + transactional RPCs

**Why:** the DB side of the switch must be one transaction (registry roles, deployments, ledger), and rollback needs the predecessor's prior stage, champion flag and artifact path recorded. Neither `ml_deployments` nor MLflow holds them (F11, F13).

### Task 4.1: Migration file

**Files:** Create `database/migrations/165_model_activation_ledger.sql`, `database/migrations/rollback_165_model_activation_ledger.sql`. Re-check the number first: `git -C <worktree> fetch origin && git -C <worktree> ls-tree --name-only origin/main database/migrations/ | sort -V | tail -3`.

```sql
-- 165: model activation ledger + transactional activate/rollback (#2318).
-- One row per activation. It is the desired state scripts/model_activation.py reconciles
-- towards, and holds every id and prior value needed to undo it (the predecessor may have no
-- MLflow version and no ml_deployments row).
BEGIN;

CREATE TABLE IF NOT EXISTS public.ml_model_activations (
    id                              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name                      varchar(255) NOT NULL,
    candidate_registry_id           uuid NOT NULL REFERENCES public.ml_model_registry(id),
    predecessor_registry_id         uuid NOT NULL REFERENCES public.ml_model_registry(id),
    served_stage                    model_stage_enum NOT NULL,
    predecessor_prior_stage         model_stage_enum NOT NULL,
    predecessor_prior_is_champion   boolean NOT NULL,
    predecessor_prior_artifact_path text,
    candidate_deployment_id         uuid REFERENCES public.ml_deployments(id),
    candidate_mlflow_model_version  integer,
    candidate_bundle_sha256         char(64) NOT NULL,
    candidate_bundle_path           text NOT NULL,
    predecessor_bundle_sha256       char(64) NOT NULL,
    predecessor_bundle_path         text NOT NULL,
    gate_report                     jsonb NOT NULL,
    status                          text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'active', 'rolled_back', 'failed')),
    approved_by                     text NOT NULL CHECK (length(btrim(approved_by)) > 0),
    created_at                      timestamptz NOT NULL DEFAULT now(),
    activated_at                    timestamptz,
    rolled_back_at                  timestamptz,
    rollback_reason                 text,
    CONSTRAINT ml_model_activations_distinct CHECK (candidate_registry_id <> predecessor_registry_id),
    CONSTRAINT ml_model_activations_served_stage CHECK (served_stage IN ('staging', 'production'))
);

-- At most one in-flight or active activation per served name.
CREATE UNIQUE INDEX IF NOT EXISTS uq_ml_model_activations_one_live
    ON public.ml_model_activations (model_name) WHERE status IN ('pending', 'active');

CREATE OR REPLACE FUNCTION public.activate_model_candidate(p_activation_id uuid)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE a public.ml_model_activations%ROWTYPE; cand public.ml_model_registry%ROWTYPE;
BEGIN
    SELECT * INTO a FROM public.ml_model_activations WHERE id = p_activation_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'activation % not found', p_activation_id; END IF;
    IF a.status = 'active' THEN RETURN; END IF;                         -- idempotent re-run
    IF a.status <> 'pending' THEN RAISE EXCEPTION 'activation % is %', a.id, a.status; END IF;
    SELECT * INTO cand FROM public.ml_model_registry WHERE id = a.candidate_registry_id FOR UPDATE;
    IF cand.stage <> 'candidate' OR cand.model_name <> a.model_name
       OR cand.retrain_of_id IS DISTINCT FROM a.predecessor_registry_id THEN
        RAISE EXCEPTION 'registry row % is not a candidate retrain of %', cand.id, a.predecessor_registry_id;
    END IF;
    PERFORM 1 FROM public.ml_model_registry WHERE id = a.predecessor_registry_id
        AND stage = a.predecessor_prior_stage FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'predecessor % changed stage since the gate ran', a.predecessor_registry_id; END IF;

    UPDATE public.ml_model_registry SET stage = 'archived', is_champion = false
     WHERE id = a.predecessor_registry_id;
    UPDATE public.ml_model_registry
       SET stage = a.served_stage, is_champion = a.predecessor_prior_is_champion,
           artifact_path = a.candidate_bundle_path, preprocessing_pipeline_path = a.candidate_bundle_path,
           promoted_at = now()
     WHERE id = a.candidate_registry_id;
    UPDATE public.ml_deployments
       SET status = 'active', environment = a.served_stage::text, deployed_at = now(),
           endpoint_name = a.model_name, endpoint_url = 'bentoml://e2i_bentoml/' || a.model_name
     WHERE id = a.candidate_deployment_id;
    UPDATE public.ml_model_activations SET status = 'active', activated_at = now() WHERE id = a.id;
END $$;

CREATE OR REPLACE FUNCTION public.rollback_model_activation(p_activation_id uuid, p_reason text)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE a public.ml_model_activations%ROWTYPE;
BEGIN
    SELECT * INTO a FROM public.ml_model_activations WHERE id = p_activation_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'activation % not found', p_activation_id; END IF;
    IF a.status = 'rolled_back' THEN RETURN; END IF;                    -- idempotent re-run
    IF a.status NOT IN ('active', 'pending') THEN RAISE EXCEPTION 'activation % is %', a.id, a.status; END IF;
    IF p_reason IS NULL OR length(btrim(p_reason)) = 0 THEN RAISE EXCEPTION 'rollback needs a reason'; END IF;

    UPDATE public.ml_model_registry SET stage = 'archived', is_champion = false
     WHERE id = a.candidate_registry_id;                                -- OD-7
    UPDATE public.ml_model_registry
       SET stage = a.predecessor_prior_stage, is_champion = a.predecessor_prior_is_champion,
           artifact_path = a.predecessor_prior_artifact_path
     WHERE id = a.predecessor_registry_id;
    UPDATE public.ml_deployments
       SET status = 'rolled_back', rolled_back_at = now(), rollback_reason = p_reason, deactivated_at = now()
     WHERE id = a.candidate_deployment_id;
    UPDATE public.ml_model_activations
       SET status = 'rolled_back', rolled_back_at = now(), rollback_reason = p_reason
     WHERE id = a.id;
END $$;

REVOKE ALL ON FUNCTION public.activate_model_candidate(uuid) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.rollback_model_activation(uuid, text) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.activate_model_candidate(uuid) TO service_role;
GRANT EXECUTE ON FUNCTION public.rollback_model_activation(uuid, text) TO service_role;
ALTER TABLE public.ml_model_activations ENABLE ROW LEVEL SECURITY;
CREATE POLICY ml_model_activations_service ON public.ml_model_activations
    FOR ALL TO service_role USING (true) WITH CHECK (true);
COMMIT;
```

- `tr_single_champion` (F20) fires on the candidate update. Setting `is_champion` from the predecessor's prior flag keeps hcp_adoption's champion semantics (OD-6); archiving the predecessor first means the trigger finds nothing to demote.
- Rollback file: `DROP FUNCTION IF EXISTS …` for both functions, then `DROP TABLE IF EXISTS public.ml_model_activations;`.
- Before writing the RLS/grant lines, read migrations 160 and 163 for this repo's role/RLS idiom. Match it and do not invent one.

### Task 4.2: Real-Postgres tests (prod-free)

**Files:** `tests/unit/test_database/test_model_activation_rpc_realdb_2318.py`. Use the `ThrowawayPg(image=docker inspect supabase-db)` fixture exactly as `tests/unit/test_database/test_registry_candidate_readers_realdb_2310.py:96-180` does: apply the table DDL of `ml_model_registry`/`ml_deployments` + migrations 159–161 + 165.

- [ ] **Step 1: Failing tests** (psycopg against the throwaway). Seed predecessor P (`staging`, champion false, artifact `/a/p.pkl`), candidate C (`candidate`, `retrain_of_id=P`), a deployment D (`registered`/`candidate`), and a `pending` ledger row A.
  - `test_activate_switches_roles_in_one_transaction` — after `SELECT activate_model_candidate(A)`: P is `archived`; C is `staging` with `artifact_path = candidate_bundle_path`; D is `active` and `environment='staging'`; A is `active`.
  - `test_activate_is_idempotent` — a second call is a no-op, with no error and an unchanged `activated_at`.
  - `test_activate_refuses_when_predecessor_moved` — set P to `production` after creating A → exception, and **nothing changed** (C still `candidate`, D still `registered`).
  - `test_activate_refuses_a_non_candidate` and `test_activate_refuses_a_candidate_of_another_parent`.
  - `test_one_live_activation_per_name` — a second `pending` row for the same `model_name` → unique violation.
  - `test_rollback_restores_prior_state` — P back to `staging`, champion and artifact exactly as recorded; C `archived`; D `rolled_back` with the reason; A `rolled_back`. A second call is a no-op.
  - `test_rollback_restores_a_production_champion` — P seeded `production`/champion (hcp shape): after activate then rollback, P is `production` + champion again. `ensure_single_champion` must not leave two champions: `SELECT count(*) … is_champion` → 1.
  - `test_anon_cannot_execute` — `SET ROLE anon; SELECT activate_model_candidate(...)` → permission denied.
- [ ] **Step 2: Run** `pytest -n 0 tests/unit/test_database/test_model_activation_rpc_realdb_2318.py -q` → FAIL (function missing).
- [ ] **Step 3:** write the migration (Task 4.1). **Step 4:** PASS.
- [ ] **Step 5: Census.** Add the two SQL functions to the SQL-reader classification in `tests/unit/test_repositories/test_registry_reader_census_2310.py` as `EXACT` (by id), with the reason `"#2318 activation/rollback RPC: rows named by the ledger's ids"`. Run `pytest -n 0 tests/unit/test_repositories/test_registry_reader_census_2310.py -q` → PASS.
- [ ] **Step 6: Commit** — `feat(db): 165 model activation ledger + transactional activate/rollback RPCs (Part of #2318)`.

**Prod apply:** owner-approved, via the repo's migration procedure (memory `reference-supabase-droplet-migration-apply`). Record the `schema_migrations` row. No data backfill.

---

## Lane 5 — Holdout harness + non-inferiority gate

**Why:** F22–F24; OD-2, OD-3. The harness must score **each model through its own persisted preprocessor**. No encoder re-fit for bundles: that is the exact-artifact principle applied to evaluation.

### Task 5.1: DeLong + bootstrap primitives

**Files:** Create `src/mlops/activation/__init__.py` (empty), `src/mlops/activation/holdout_gate.py`. Test `tests/unit/test_mlops/test_activation_holdout_gate_2318.py`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.mlops.activation.holdout_gate import GateConfig, delong_paired, evaluate_gate


def _scores(seed=0, n=2000, noise=0.0):
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, n)
    base = y + r.normal(0, 1.0, n)
    return y, 1 / (1 + np.exp(-base)), 1 / (1 + np.exp(-(base + r.normal(0, noise, n))))


def test_delong_aucs_match_sklearn():
    y, a, b = _scores(noise=0.5)
    aucs, cov = delong_paired(y, np.vstack([a, b]))
    assert aucs[0] == pytest.approx(roc_auc_score(y, a), abs=1e-12)
    assert aucs[1] == pytest.approx(roc_auc_score(y, b), abs=1e-12)
    assert cov.shape == (2, 2) and cov[0, 1] > 0


def test_identical_models_pass_and_report_zero_delta():
    y, a, _ = _scores()
    rep = evaluate_gate(y, served=a, candidate=a.copy(), cfg=GateConfig())
    assert rep["passed"] and rep["auc_delta"] == 0.0


def test_a_clearly_worse_candidate_fails_on_auc():
    y, a, _ = _scores()
    worse = 0.5 * a + 0.5 * np.random.default_rng(1).random(len(a))
    rep = evaluate_gate(y, served=a, candidate=worse, cfg=GateConfig())
    assert not rep["passed"] and "auc_noninferiority" in rep["failed_checks"]


def test_miscalibrated_candidate_fails_on_slope():
    y, a, _ = _scores()
    logit = np.log(a / (1 - a))
    sharp = 1 / (1 + np.exp(-3 * logit))  # same ranking, slope ~ 1/3
    rep = evaluate_gate(y, served=a, candidate=sharp, cfg=GateConfig())
    assert rep["auc_delta"] == pytest.approx(0.0, abs=1e-12)
    assert "calibration_slope" in rep["failed_checks"]


def test_refuses_a_thin_holdout():
    y = np.array([0] * 500 + [1] * 50)
    s = np.linspace(0, 1, len(y))
    with pytest.raises(ValueError, match="at least 100"):
        evaluate_gate(y, served=s, candidate=s, cfg=GateConfig())
```

- [ ] **Step 2: Run** `pytest -n 0 tests/unit/test_mlops/test_activation_holdout_gate_2318.py -q` → FAIL.

- [ ] **Step 3: Implement.**
  - `delong_paired`: the midrank/structural-component algorithm from the probe `p2318_delong_probe.py`, i.e. Sun & Xu 2014.
  - `GateConfig(auc_margin=0.010, alpha=0.05, brier_margin=0.005, slope_band=(0.8, 1.25), bootstrap_b=2000, seed=0, min_class_n=100)`.
  - `evaluate_gate(y, served, candidate, cfg)` returns a JSON-serialisable dict:
    - `n`, `n_pos`, `auc_served`, `auc_candidate`, `auc_delta`, `se_delta`, `auc_lower_bound`;
    - `brier_served`, `brier_candidate`, `brier_delta_upper` (paired bootstrap, fixed seed);
    - `calibration_slope`, `calibration_intercept` (reuse `positive_class_scores` / `calibration_intercept` semantics from `scripts/promote_hcp_adoption_champions.py:110-160` — move the shared helpers into `src/mlops/activation/holdout_gate.py` and import them back into the script; do not duplicate);
    - `bootstrap_auc_delta_p05` (cross-check, reported only);
    - `failed_checks: list[str]`, `passed: bool`, `config: asdict(cfg)`.
  - The lower bound is `auc_delta - norm.ppf(1 - alpha) * se_delta`, and the check passes iff it is `> -auc_margin`.

- [ ] **Step 4: Run** → PASS.

### Task 5.2: Score both artifacts on one holdout snapshot

**Files:** `src/mlops/activation/holdout_gate.py` (add `load_holdout_snapshot`, `score_bundle`). Test (same file).

- [ ] **Step 1: Failing tests.**
  - `score_bundle(bundle, frame)` for a FeatureBuilder bundle equals `bundle["model"].predict_proba(bundle["preprocessor"].transform(frame)[feature_columns])`.
  - For an `sklearn_ct_v1` bundle it equals `model.predict_proba(ct.transform(frame[keep_columns]))`.
  - Neither calls any `fit`: wrap the preprocessor's `fit` to raise and assert no error.
  - `load_holdout_snapshot` (with a fake async client returning a fixed frame) returns `(frame, y, snapshot)`, where `snapshot = {"splits": ["test","holdout"], "n", "n_pos", "row_ids_sha256"}`. The hash is over sorted `patient_journey_id` (or the cohort's id column; read `FeatureBuilder.load_frame`'s select list at `feature_builder.py:260-300` for the id column name). Two calls over the same rows give the same hash.
  - `test_keep_columns_must_match`: a candidate whose `keep_columns` set ≠ the served set is refused (`ValueError("raw contract differs")`), because the API and Feast supply the served contract (F7, `sync_goldstd_serving.py:135-164`).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** Snapshot via `FeatureBuilder(spec).load_frame(db, splits=None)` filtered to `("test","holdout")`, as `run_persistence_eval.py:180-229` does, and resolve `spec` from the model name with `rematerialize_goldstd_bundles.SPEC_REGISTRY` (`:64-96`). **Step 4:** PASS.
- [ ] **Step 5: Faithful rehearsal (read-only).** Run a scratch script against prod that scores the **served bundle** and the **v1.0 registry artifact** with `score_bundle`/`evaluate_gate` (served as comparator, registry as "candidate"). Expected, reproducing the probe: `auc_served≈0.8521`, `auc_candidate≈0.8509`, and pass. Paste the report into the PR body.
- [ ] **Step 6: Commit** — `feat(mlops): paired DeLong non-inferiority gate on the goldstd holdout (Part of #2318)`.

---

## Lane 6 — `promote-candidate` / `rollback-activation` CLI + reconciler

**Why:** the owner-approved command. It never runs automatically, defaults to dry-run, and requires `--approved-by`.

**Reconcile order** (each step idempotent; the ledger row is the desired state):

1. **Resolve.**
   - Candidate row by id: it must be `stage='candidate'` with `retrain_of_id` = the current canonical row of the name.
   - Refuse if a `pending`/`active` activation exists for the name.
   - Refuse hcp_adoption unless OD-6 is granted (`--allow-hcp-production`).
2. **Fetch the exact artifact.**
   - MLflow run `mlflow_run_id` → tag `e2i.serving_bundle_sha256`. Refuse if absent: "not servable as-trained; retrain after #2318 Lane 2".
   - GET `serving_bundle/bundle.pkl` via `http://localhost:5000/api/2.0/mlflow-artifacts/artifacts/<run>/artifacts/serving_bundle/bundle.pkl`.
   - Verify its sha256 **before unpickling**.
3. **Snapshot the served bundle** (predecessor). Read `shap_serving/<cohort>/<name>.bundle.pkl` and hash it. Its hash must equal the sidecar's `/model_info` `bundle_sha256` for the name (Lane 3); otherwise refuse, because disk and sidecar disagree.
4. **Gate** (Lane 5) on one snapshot. Print the report. **Dry-run stops here** and prints the exact planned writes.
5. **`--execute`:**
   - (a) Write both bundles into the versioned store `data/ml_artifacts/serving_versions/<name>/<registry_id>.<sha12>.bundle.pkl`. This is **outside** `shap_serving` (F1); the dir is created 0750, and each file is written to a temp file, fsynced, then renamed.
   - (b) Insert the ledger row (`pending`) with every prior value, `gate_report` and `approved_by`.
   - (c) Record the holdout metrics under the candidate id: `ml_performance_metrics source='holdout'` (auc_roc, brier_score, calibration_slope, pr_auc; `sample_size=n`), through `MetricRecorder`/`PerformanceMetricRepository`, as `run_persistence_eval.py:300-330` does. The KPI page needs them (F7).
   - (d) `os.replace` the candidate bundle onto `shap_serving/<cohort>/<name>.bundle.pkl` (atomic on one filesystem; both paths are under `data/ml_artifacts`).
   - (e) `docker restart <sidecar>`, resolving the name with `docker ps --format '{{.Names}}' | grep -E '^e2i_bentoml(_dev)?$'`. Then poll `/healthz` and `/model_info` until `bundle_sha256 == candidate sha` (timeout 180 s). On timeout: restore the predecessor file, restart, set the ledger to `failed`, exit 3.
   - (f) RPC `activate_model_candidate(ledger_id)`.
   - (g) MLflow on the candidate version:
     - `POST /api/2.0/mlflow/registered-models/alias` `{name, alias: "served", version}`;
     - `POST /api/2.0/mlflow/model-versions/set-tag` `e2i.role=served`;
     - `DELETE` the old `e2i.role=candidate` by overwrite.
   - (h) SHAP cache: `GET /explain/global?model_type=<cohort>&brand=<brand>&sample_size=20&refresh=true` for that one slot, with the admin token as in `sync_goldstd_serving.py:103-116`. Assert 200 and that the response's features collapse to the served `keep_columns` (`_raw_covariates`, `sync_goldstd_serving.py:167-176`).
6. **Re-run safety.** `promote-candidate` with an existing `pending` ledger row for the same candidate resumes at the first unfinished step (each step checks its end state first). `status` prints the ledger plus every surface's observed state and flags drift.

**Rollback (`rollback-activation <ledger id> --reason ... --approved-by ...`):**

1. `os.replace` the predecessor file from the versioned store, after verifying its sha256.
2. Restart the sidecar and verify `/model_info` sha == `predecessor_bundle_sha256`.
3. RPC `rollback_model_activation`.
4. MLflow: delete alias `served` and set tag `e2i.role=rolled_back`.
5. SHAP refresh for the slot.
6. Metrics: the candidate's holdout rows stay (history); the predecessor's are untouched.

### Task 6.1: Versioned bundle store

**Files:** `src/mlops/activation/bundle_store.py`; test `tests/unit/test_mlops/test_activation_bundle_store_2318.py` (uses `tmp_path` only).

- [ ] **Step 1: Failing tests.**
  - `stash(name, registry_id, blob)` writes `serving_versions/<name>/<id>.<sha12>.bundle.pkl` and returns `(path, sha)`. Re-stashing the same bytes is a no-op; the same id with different bytes raises.
  - `swap_in(name, cohort, path, expected_sha)` refuses on a sha mismatch, and otherwise leaves exactly one file named `<name>.bundle.pkl` under `shap_serving/` with that sha.
  - `test_store_is_outside_the_serving_root`: `serving_versions` is not under `shap_serving`, so the sidecar's walk never sees duplicates (F1).
  - `served_sha(name, cohort)` hashes the served file.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** with `tempfile.NamedTemporaryFile(dir=target.parent, delete=False)` + `os.fsync` + `os.replace`. **Step 4:** PASS. **Commit.**

### Task 6.2: Reconciler with injectable surfaces

**Files:** `src/mlops/activation/reconcile.py`; test `tests/unit/test_mlops/test_activation_reconcile_2318.py`.

Design: `Surfaces` is a small protocol with methods `registry_row(id)`, `live_activation(name)`, `insert_ledger(row)`, `rpc_activate(id)`, `rpc_rollback(id, reason)`, `mlflow_bundle(run_id) -> (blob, tag_sha)`, `mlflow_set_served(name, version)`, `mlflow_clear_served(name, version)`, `sidecar_sha(name)`, `restart_sidecar()`, `refresh_shap(cohort, brand)`, `record_holdout_metrics(model_id, report)`.
- Production binds them to Supabase / MLflow REST / docker / HTTP.
- Tests bind in-memory fakes of these **external boundaries**; business logic (gate, ordering, idempotency, refusal) is real.

- [ ] **Step 1: Failing tests.**
  - `test_dry_run_writes_nothing` — every write-method fake records zero calls; the plan printed lists the steps in order.
  - `test_refuses_candidate_without_bundle_tag` (the OD-8 case).
  - `test_refuses_hash_mismatch_before_unpickle` — the fake returns bytes whose sha ≠ tag. Assert that `pickle.loads` is never called by patching it to raise.
  - `test_refuses_when_disk_and_sidecar_disagree`.
  - `test_refuses_failed_gate` — no ledger row is written.
  - `test_execute_order` — calls happen in the order stash → ledger → metrics → swap → restart → sidecar verify → rpc → mlflow → shap.
  - `test_sidecar_timeout_restores_predecessor_and_marks_failed`.
  - `test_resume_from_pending_skips_done_steps` — the swap already happened, so the second run does not restart twice before the RPC.
  - `test_rollback_order_and_idempotency`.
  - `test_second_live_activation_refused`.
  - `test_hcp_requires_explicit_production_grant`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4:** PASS. **Commit.**

### Task 6.3: CLI + production bindings

**Files:** `scripts/model_activation.py` (argparse: `promote-candidate <registry_id> --approved-by NAME [--execute] [--comparator served|registry] [--allow-hcp-production]`, `rollback-activation <ledger_id> --reason TEXT --approved-by NAME [--execute]`, `status [<model_name>]`).

- [ ] **Step 1: Failing test.** `tests/unit/test_scripts/test_model_activation_cli_2318.py`:
  - `--help` lists the three subcommands;
  - `promote-candidate` without `--approved-by` exits 2;
  - the default is dry-run: construct with fake surfaces and assert no write-method calls.
- [ ] **Step 2–4:** implement, then PASS.
- [ ] **Step 5: Census.** Classify every new `ml_model_registry` accessor in `src/mlops/activation/*.py` and `scripts/model_activation.py` in `test_registry_reader_census_2310.py`:
  - the candidate lookup is `EXACT` (by id);
  - the "current canonical row of the name" check is `CANONICAL` via `resolve_canonical_model_id`.
  Then run the census → PASS.
- [ ] **Step 6: Read-only rehearsal on prod.** Run `python -m scripts.model_activation promote-candidate 44e1fe04-a19d-458b-a3d9-cfbe0c1edced --approved-by rehearsal` (dry-run). Expected: refused with the "not servable as-trained" message (OD-8). That proves the refusal on a real pre-#2318 candidate.
- [ ] **Step 7: Commit** — `feat(mlops): owner-gated promote-candidate / rollback-activation with DB+MLflow+bundle+sidecar reconciliation (Part of #2318)`.

---

## Lane 7 — Keep the weekly reseed and the re-materializer from reverting an activation

**Why:** F14–F16 (OD-5). This lane must be deployed before any prod activation.

### Task 7.1: `active_activation_for(model_name)`

**Files:** `src/mlops/activation/guards.py`; test `tests/unit/test_mlops/test_activation_guards_2318.py`.

- [ ] **Step 1: Failing tests.**
  - Returns the ledger row with `status='active'` for the name, else `None`.
  - A `pending` row also counts: fail safe while an activation is in flight.
  - A lookup error **raises**: the callers are writers and must not proceed blind. This is the opposite of the reader fail-open in `model_registry_roles.py:114-118`, stated in the docstring.
- [ ] **Step 2–4: implement → PASS.**

### Task 7.2: Retrain slots skip activated names

**Files:** `src/mlops/gold_standard_eval/run_persistence_eval.py` and `run_initiation_eval.py` (at the top of the per-slot function, before the metric delete at `run_persistence_eval.py:250-271`), `run_hcp_cohorts.py` (same). Tests in the guards test file, with a fake client.

- [ ] **Step 1: Failing test.** With an active activation for `initiation_kisqali_goldstd_lr_v1`, the slot returns `{"model": name, "skipped": "served by activated candidate <cand_id> (activation <id>)"}`. `register_model_row`, `delete_metrics` and `serialize_model` are **not called**. Other slots run.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** the early return. `run_patient_cohorts._print_report` (`:110-125`) must render `skipped` rows without `KeyError` (the complement-validation block at `:83-107` must skip a pair with a skipped member). **Step 4:** PASS.
- [ ] **Step 5:** extend `tests/unit/test_scripts/test_reseed_retrain_hookup.py` if it asserts the slot count, so a skipped slot is not a failure.

### Task 7.3: Re-materializer, hcp promote and `register_model_row` refuse activated slots

**Files:** `scripts/rematerialize_goldstd_bundles.py` (`_amain` `:157-186`), `scripts/sync_goldstd_serving.py` (it calls `_amain`), `scripts/promote_hcp_adoption_champions.py` (before `_fetch_registry_row` use), `src/mlops/prediction_synthesizer_deploy.py::register_model_row` (before the upsert, `:380-387`).

- [ ] **Step 1: Failing tests.**
  - `_amain([name])` with an active activation prints `SKIP <name>: served by activation <id>` and does **not** write the bundle; rc stays 0.
  - hcp promote HOLDs the brand with the same reason.
  - `register_model_row(..., model_name=n, model_version="1.0")`, when the existing `(n, "1.0")` row is the `predecessor_registry_id` of a live activation, raises `ValueError("… archived by activation …")` before the upsert. This is the defensive half of OD-5.
- [ ] **Step 2–4: implement → PASS.** Run `pytest -n 0 tests/unit/test_scripts/test_reseed_retrain_hookup.py tests/unit/test_mlops/test_activation_guards_2318.py -q`.
- [ ] **Step 5: Census** — classify the new ledger reads (they read `ml_model_activations`, not the registry; add them only if the census flags them).
- [ ] **Step 6: Commit** — `fix(goldstd): weekly retrain, re-materializer and hcp promote leave an activated slot alone (Part of #2318)`.

---

## Lane 8 — Live verification on prod (owner approval in-turn for every ⚠ step)

**Prerequisites:**
- L1–L7 merged and deployed.
- Migration 165 applied.
- The sidecar reports `bundle_sha256` for all 12 names.
- `InconsistentVersionWarning` count is 0.

Before the first paid or prod step, run `pgrep -fa` for peer sessions and `ls -d docs/demos/results/$(date +%F)*` (memory: concurrent-session duplicate-cert check).

**Script:** `scripts/verify_model_activation_live.py` (read-only except where it calls the CLI with `--execute`, which it does only when passed `--i-have-owner-approval`). The fixed probe rows are the first 20 test ∪ holdout rows by sorted id for the slot; their ids are written to the evidence dir.

| # | Step | Write? |
|---|---|---|
| 0 | Record the baseline: sidecar `/model_info` sha (P_sha) for the slot; registry rows; ledger (empty); MLflow aliases; `/explain/global` (cached) response; the sidecar's `/predict` probabilities for the 20 fixed rows (`p_before`). | read-only |
| 1 | ⚠ Produce a Lane-2 candidate for **one non-hcp slot** (recommend `initiation_kisqali_goldstd_lr_v1`) using the same retrain trigger as the #2157 live cert. | **prod write (registry candidate row + MLflow run/version)** |
| 2 | Dry-run `promote-candidate <new id> --approved-by <owner>` and paste the gate report. | read-only |
| 3 | ⚠ `promote-candidate … --execute`. | **prod write (FS bundle, sidecar restart, registry/deployments/ledger, MLflow alias/tag, metrics, SHAP cache)** |
| 4 | Verify serving is the exact artifact. Download `serving_bundle/bundle.pkl` from MLflow, check its sha against the run tag, and compute `p_local = model.predict_proba(ct.transform(rows[keep]))[:,1]` **in the worker image** (`docker exec -i e2i-causal-analytics-worker_medium-1 python -`, sklearn 1.6.1). Assert: sidecar `/predict` for the same rows returns `p_sidecar == p_local` (max abs diff ≤ 1e-12); sidecar sha == ledger `candidate_bundle_sha256`; registry candidate stage `staging`, predecessor `archived`; deployment `active`; MLflow alias `served` → the version; `/explain/global` 200 with `cached=false` then `cached=true` under the candidate id; KPI page `summarize` includes the candidate's holdout metrics exactly once. | read-only |
| 5 | Verify the reseed guard **without waiting for Monday**: run `python -m scripts.rematerialize_goldstd_bundles --model <name>` → `SKIP`. Also run the slot's retrain dry path, if one exists; otherwise rely on the unit tests and state that. | read-only (the SKIP path writes nothing) |
| 6 | ⚠ `rollback-activation <ledger id> --reason "2318 live verification" --approved-by <owner> --execute`. | **prod write** |
| 7 | Verify the predecessor is restored: sidecar sha == P_sha; `/predict` for the 20 rows == `p_before` exactly; predecessor stage/champion/artifact_path equal the baseline; candidate `archived`; deployment `rolled_back`; ledger `rolled_back`; alias `served` absent; `/explain/global` recomputed under the predecessor id. | read-only |
| 8 | Re-run `rollback-activation` (idempotency) → no-op, exit 0. Run `status` → no drift. | read-only |

**Evidence:**
- Save all outputs to `docs/demos/results/<date>_2318_activation_live/`: JSON per step, the gate report, prediction CSVs, and a verdict line first.
- Evidence PR titled `docs(evidence): #2318 live activation + rollback`, body "Part of #2318". Do not write "close"; check `closingIssuesReferences` before the merge.
- A FAIL at step 4 or 7 is a finding: stop, roll back if active, report. Never patch around it.

---

## 4. Risks and follow-ups (not in scope)

- **Time-Series walk-forward trend** for an activated candidate: only the holdout point is recorded. A backtest re-run under the new id is a follow-up issue.
- **Feast** is untouched: the raw contract must be identical (enforced by Task 5.2).
- **BentoML store path** (`_discover_goldstd_bundles_from_store`) stays unused on prod (F1). Activation writes FS only. If the store is ever populated it wins over FS; Lane 3's identity check (`/model_info` sha) would expose that mismatch at step 5e.
- **Pickle trust.** Bundles are unpickled only after sha256 verification against the MLflow run tag written by our own trainer. The sidecar already unpickles FS bundles (`e2i_serving_service.py:574`). The peer lane `claude/skops-trust-boosters` changes skops trusted types for `model.skops`, which this plan does not load.
- **Migration number** collision: see F25.
