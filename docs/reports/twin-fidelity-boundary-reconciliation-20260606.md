# Twin↔Experiment Fidelity Boundary — Reconciliation (R5, 2026-06-06)

Deliverable for R5 Task 2 of the digital-twin remediation. The fidelity domain is
split across **two boundaries** with overlapping-but-distinct schemas. This note
maps them, names the canonical join key per side, lists every consumer, and
records the reconciliation already applied (H17) plus the remaining boundary that
is intentionally left as-is.

## The two sides

### 1. Twin-side (digital-twin prediction tracking)
Source: `database/ml/012_digital_twin_tables.sql`.

| Object | Kind | Join key to experiment | Key columns |
|---|---|---|---|
| `twin_simulations` | table | `experiment_design_id` (FK ml_experiments, nullable) | `simulation_id`, `simulated_ate`, `simulated_ci_lower/_upper`, `simulation_status`, `data_provenance` (migration 030) |
| `twin_fidelity_tracking` | table | `actual_experiment_id` (FK ml_experiments, nullable) | `simulation_id`, `simulated_ate`, `actual_ate`, `prediction_error`, `fidelity_grade` |
| `v_simulation_summary` | view | `experiment_design_id` | LEFT JOINs `twin_simulations` → `twin_fidelity_tracking`; exposes `actual_ate`, `prediction_error`, `fidelity_grade` |

**Two different experiment keys exist on this side**: a simulation is linked to the
design it pre-screens via `twin_simulations.experiment_design_id`; a *validated*
prediction is linked to the run that produced the outcome via
`twin_fidelity_tracking.actual_experiment_id`. They are equal when a design becomes
a single executed experiment, but the schema keeps them distinct (a design may map
to 0..n runs). `v_simulation_summary` is keyed on `experiment_design_id` (it starts
from `twin_simulations`).

Writers: `POST /digital-twin/validate` + `FidelityTracker` (`src/digital_twin/fidelity_tracker.py`) write `twin_fidelity_tracking`; `save_simulation` (`src/digital_twin/twin_repository.py`) writes `twin_simulations`.

### 2. A/B-side (experiment results fidelity)
Source: `database/ml/021_ab_results_tables.sql`.

| Object | Kind | Join key | Notes |
|---|---|---|---|
| `ab_fidelity_comparisons` | table | `experiment_id` | written by the A/B results services (`ResultsAnalysisService._persist_fidelity_comparison`) |
| `vw_fidelity_summary` | view | `experiment_id` | A/B-side rollup |

`ResultsAnalysisService.compare_with_twin_prediction` (and the R1 convenience
`compare_experiment_to_twin`) compute a `FidelityComparison` and persist to the
A/B-side `ab_fidelity_comparisons`.

## Consumers

| Consumer | Reads/Writes | Key used |
|---|---|---|
| `FidelityCheckerNode` (`src/agents/experiment_monitor/nodes/fidelity_checker.py`) | reads `twin_fidelity_tracking` then falls back to `v_simulation_summary` | `actual_experiment_id` (primary) + `experiment_design_id` (fallback) — **standardized on BOTH by H17** |
| `ResultsAnalysisService` (`src/services/results_analysis.py`) | writes `ab_fidelity_comparisons` | `experiment_id` |
| `fidelity_tracking_update` task (`src/tasks/ab_testing_tasks.py`) | via `compare_experiment_to_twin` → A/B-side | `experiment_id` (H9) |
| `POST /experiments/{id}/fidelity/{sim}` (`src/api/routes/experiments.py`) | via `compare_experiment_to_twin` → A/B-side | `experiment_id` (N2) |
| `TwinRepository` / `SimulationRepository` (`src/digital_twin/twin_repository.py`) | writes `twin_simulations`, reads `v_simulation_summary` | `experiment_design_id` |

## Canonical-key decision

- **Twin-side prediction tracking**: the canonical key is `actual_experiment_id`
  on `twin_fidelity_tracking` (a *validated* prediction is tied to the run that
  produced the actual). `v_simulation_summary.experiment_design_id` is a legitimate
  **fallback** for designs whose fidelity row hasn't been keyed by run yet (this is
  exactly what `FidelityCheckerNode` does after H17 — primary on
  `actual_experiment_id`, fallback on `experiment_design_id`).
- **A/B-side**: `experiment_id` is canonical (single source: the executed
  experiment). The route + task converge here via `compare_experiment_to_twin`.

The two sides are **intentionally distinct**, not duplicates: the twin-side tracks
prediction-vs-actual calibration of the *twin model*; the A/B-side records the
fidelity comparison as part of *experiment results analysis*. They share the
`FidelityComparison` shape but persist to different tables for different lifecycles.

## Outcome

No additional code change falls out of this reconciliation beyond H17 (which already
standardized `FidelityCheckerNode` to read both twin-side keys). The remaining
boundary (twin-side vs A/B-side) is by-design and documented here. The real-world
fidelity leg — twin prediction vs an actual A/B outcome — still requires a
ground-truth outcome feed (the per-unit metric-observation schema, #422) before the
automatic producer (R4b/H9) yields non-skipped rows; the on-demand `POST /validate`
and `POST /experiments/{id}/fidelity/{sim}` paths are the live triggers today.
