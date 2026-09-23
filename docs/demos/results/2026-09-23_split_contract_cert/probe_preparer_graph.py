"""READ-ONLY disproof: does the DGP's own manifest (synthetic_csu) clear the Layer-3 flag?

Same graph walk as split_contract_probe_graph.py, with scope_spec["feature_manifest_source"]
resolved through the real resolve_manifest_source(data_source_dict, "synthetic_csu") first
(proves the override resolves without M1/M2). Records per feature severity / z / remediation /
decided_by after adaptive_validity_check, the leakage route, and whether finalize_output
reports an empty blocking_issues.
"""
import asyncio, json, os, sys, time
sys.path.insert(0, os.getcwd())
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check").setLevel(logging.INFO)

from src.agents.ml_foundation.data_preparer import graph as G
from src.data.manifests.resolution import resolve_manifest_source
from src.mlops.gold_standard_eval.cohort_spec import _PATIENT_COVARIATES, _PATIENT_LABELS


def fold(state, update):
    for k, v in (update or {}).items():
        if v is not None:
            state[k] = v


async def run(cohort, brand):
    label = _PATIENT_LABELS[cohort]
    contract = {"type": "table", "table": "patient_journeys",
                "filters": {"brand": brand, "is_synthetic": True},
                "columns": list(_PATIENT_COVARIATES[cohort]) + [label]}
    resolved = resolve_manifest_source(contract, "synthetic_csu")
    print(f"\n=== {cohort} / {brand} === resolve_manifest_source(dict, 'synthetic_csu') -> {resolved!r}")
    state = {
        "experiment_id": f"probe-{cohort}-{brand.lower()}",
        "data_source": contract,
        "scope_spec": {"prediction_target": label, "data_source": contract, "filters": {},
                       "experiment_id": f"probe-{cohort}-{brand.lower()}",
                       "problem_type": "binary_classification",
                       "feature_manifest_source": resolved},
        "blocking_issues": [],
    }
    pre = [("load_data", G.load_data), ("audit_sampling_frame", G.audit_sampling_frame),
           ("run_schema_validation", G.run_schema_validation), ("run_quality_checks", G.run_quality_checks),
           ("run_ge_validation", G.run_ge_validation), ("engineer_features", G.engineer_features_node),
           ("detect_leakage", G.detect_leakage), ("adaptive_validity_check", G.adaptive_validity_check)]
    for name, node in pre:
        t0 = time.time(); upd = await node(state); fold(state, upd)
        keys = {k: upd.get(k) for k in ("error", "qc_status", "overall_score", "ge_validation_status", "leakage_severity", "leaked_features") if k in (upd or {})}
        print(f"[{name} {time.time()-t0:.1f}s] {json.dumps(keys, default=str)} blocking={state.get('blocking_issues')}")
        if name == "adaptive_validity_check":
            cols = [c for c in state["train_df"].columns if c not in (label, "data_split")]
            seen = set()
            for f in state.get("leakage_findings", []) or []:
                if f.get("layer") in ("1", "3", "4", 1, 3, 4) or f.get("decided_by"):
                    feat = f.get("feature"); seen.add(feat)
                    z = f.get("z_score"); z = f"{z:.2f}" if isinstance(z, (int, float)) else z
                    print(f"  feature={feat:<22} severity={f.get('severity'):<9} z={z:<8} remediation={f.get('remediation'):<10} decided_by={f.get('decided_by')} layer={f.get('layer')} declared_safe={f.get('layer_1_declared_safe')}")
            undeclared = [c for c in cols if c not in seen]
            print(f"  columns with no adaptive finding (Layer 1 / not scored): {undeclared}")
            print(f"  adaptive_flagged_features={state.get('adaptive_flagged_features')} leaked_features={state.get('leaked_features')}")
    route = G._route_after_leakage_detection(state)
    print(f"[route_after_leakage_detection] -> {route}")
    if route == "remediate":
        print("leakage_remediation (LLM) would run — NOT executed; VERDICT FAIL")
        return False
    for name, node in [("transform_data", G.transform_data), ("compute_baseline_metrics", G.compute_baseline_metrics),
                       ("sufficiency_check", G.run_sufficiency_check), ("finalize_output", G.finalize_output)]:
        t0 = time.time(); upd = await node(state); fold(state, upd)
        keys = {k: upd.get(k) for k in ("error", "transformation_status", "baseline_status", "sufficiency_status", "gate_passed", "qc_passed", "is_ready", "qc_score") if k in (upd or {})}
        print(f"[{name} {time.time()-t0:.1f}s] {json.dumps(keys, default=str)} blocking={state.get('blocking_issues')}")
    ok = state.get("gate_passed") is True and state.get("qc_status") in ("passed", "warning") and not state.get("blocking_issues")
    print(f"GATE {cohort}/{brand}: gate_passed={state.get('gate_passed')} qc_status={state.get('qc_status')} qc_passed={state.get('qc_passed')} is_ready={state.get('is_ready')} qc_score={state.get('qc_score')} missing_required_features={state.get('missing_required_features')} blocking_issues={state.get('blocking_issues')}")
    print(f"VERDICT {cohort}/{brand}: {'PASS' if ok else 'FAIL'}")
    return ok


async def main():
    res = {c: await run(c, "Kisqali") for c in ("initiation", "persistence", "discontinuation")}
    print("\nSUMMARY:", res, "| ALL PASS" if all(res.values()) else "| SOME FAIL")

asyncio.run(main())
