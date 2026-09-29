"""READ-ONLY live verify for #2320: the Pandera schema gate on the three real Kisqali goldstd
patient cohorts, in the PRODUCTION state shape.

Production shape (measured, not assumed): tier_0 passes the table cohort contract dict as the
top-level ``data_source``; ``scope_spec`` comes from scope_definer, which never sets
``scope_spec.data_source`` / ``table_name`` (``ScopeSpecSchema.data_source`` is Optional[str], so
a compiled graph rejects a dict there). Earlier probes (#2207, #2294) put the dict in
``scope_spec.data_source`` too — a shape production cannot produce.

Walks the real node sequence load_data -> ... -> finalize_output, reads prod Supabase only
(SELECTs). Never runs leakage_remediation (LLM). The Layer-4 LLM evaluator and the verdict
sidecar are disabled by unsetting their env vars. Run with PYTHONPATH pointing at the code
under test; the header prints which tree was imported.
"""
import asyncio, json, os, sys, time
sys.path.insert(0, os.getcwd())
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("src.mlops.pandera_schemas").setLevel(logging.INFO)

import src
from src.agents.ml_foundation.data_preparer import graph as G
from src.mlops.gold_standard_eval.cohort_spec import _PATIENT_COVARIATES, _PATIENT_LABELS

for var in ("ADAPTIVE_VALIDITY_EVALUATOR_ENABLED", "ADAPTIVE_VALIDITY_ARTIFACTS_DIR"):
    os.environ.pop(var, None)
print(f"code under test: {os.path.dirname(src.__file__)}")


def fold(state, update):
    for k, v in (update or {}).items():
        if v is not None:
            state[k] = v


async def run(cohort, brand):
    label = _PATIENT_LABELS[cohort]
    contract = {"type": "table", "table": "patient_journeys",
                "filters": {"brand": brand, "is_synthetic": True},
                "columns": list(_PATIENT_COVARIATES[cohort]) + [label]}
    print(f"\n=== {cohort} / {brand} === columns={contract['columns']}")
    state = {
        "experiment_id": f"probe2320-{cohort}-{brand.lower()}",
        "data_source": contract,
        # production shape: no scope_spec.data_source / table_name
        "scope_spec": {"prediction_target": label,
                       "experiment_id": f"probe2320-{cohort}-{brand.lower()}",
                       "problem_type": "binary_classification",
                       "feature_manifest_source": "synthetic_csu"},
        "blocking_issues": [],
    }
    pre = [("load_data", G.load_data), ("audit_sampling_frame", G.audit_sampling_frame),
           ("run_schema_validation", G.run_schema_validation), ("run_quality_checks", G.run_quality_checks),
           ("run_ge_validation", G.run_ge_validation), ("engineer_features", G.engineer_features_node),
           ("detect_leakage", G.detect_leakage), ("adaptive_validity_check", G.adaptive_validity_check)]
    for name, node in pre:
        t0 = time.time(); upd = await node(state); fold(state, upd)
        keys = {k: upd.get(k) for k in ("error", "schema_validation_status", "schema_splits_validated",
                                        "qc_status", "overall_score", "ge_validation_status",
                                        "ge_expectations_passed", "ge_expectations_evaluated",
                                        "leakage_severity", "leaked_features") if k in (upd or {})}
        print(f"[{name} {time.time()-t0:.1f}s] {json.dumps(keys, default=str)} blocking={state.get('blocking_issues')}")
        if name == "load_data":
            tr = state.get("train_df")
            print(f"  rows train/val/test = {len(tr)}/{len(state.get('validation_df'))}/{len(state.get('test_df'))}; "
                  f"frame columns = {list(tr.columns)}")
        if name == "run_schema_validation" and state.get("schema_validation_errors"):
            print(f"  schema errors (first 5): {state['schema_validation_errors'][:5]}")
    route = G._route_after_leakage_detection(state)
    print(f"[route_after_leakage_detection] -> {route}")
    if route == "remediate":
        print("leakage_remediation (LLM) would run — NOT executed by this probe")
    else:
        for name, node in [("transform_data", G.transform_data), ("compute_baseline_metrics", G.compute_baseline_metrics),
                           ("sufficiency_check", G.run_sufficiency_check), ("finalize_output", G.finalize_output)]:
            t0 = time.time(); upd = await node(state); fold(state, upd)
            keys = {k: upd.get(k) for k in ("error", "gate_passed", "qc_passed", "is_ready") if k in (upd or {})}
            print(f"[{name} {time.time()-t0:.1f}s] {json.dumps(keys, default=str)} blocking={state.get('blocking_issues')}")
    schema_blockers = [b for b in state.get("blocking_issues") or [] if b.startswith("schema: ")]
    print(f"RESULT {cohort}/{brand}: schema_validation_status={state.get('schema_validation_status')} "
          f"splits={state.get('schema_splits_validated')} schema_blockers={schema_blockers} "
          f"route={route} gate_passed={state.get('gate_passed')} blocking_issues={state.get('blocking_issues')}")
    return state.get("schema_validation_status")


async def main():
    res = {c: await run(c, "Kisqali") for c in ("initiation", "persistence", "discontinuation")}
    print("\nSUMMARY schema_validation_status:", res)

asyncio.run(main())
