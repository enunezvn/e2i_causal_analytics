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
    """Post-#2294 live walk (read-only): same node sequence as the 09-23 probe, but when routing
    says remediate, the LLM remediation is NOT run and transform..finalize still execute so the
    gate's blocking_issues on real data are observed end to end."""
    label = _PATIENT_LABELS[cohort]
    contract = {"type": "table", "table": "patient_journeys",
                "filters": {"brand": brand, "is_synthetic": True},
                "columns": list(_PATIENT_COVARIATES[cohort]) + [label]}
    resolved = resolve_manifest_source(contract, "synthetic_csu")
    print(f"\n=== {cohort} / {brand} ===")
    state = {"experiment_id": f"probe2294-{cohort}-{brand.lower()}", "data_source": contract,
             "scope_spec": {"prediction_target": label, "data_source": contract, "filters": {},
                            "experiment_id": f"probe2294-{cohort}-{brand.lower()}",
                            "problem_type": "binary_classification",
                            "feature_manifest_source": resolved},
             "blocking_issues": []}
    pre = [("load_data", G.load_data), ("audit_sampling_frame", G.audit_sampling_frame),
           ("run_schema_validation", G.run_schema_validation), ("run_quality_checks", G.run_quality_checks),
           ("run_ge_validation", G.run_ge_validation), ("engineer_features", G.engineer_features_node),
           ("detect_leakage", G.detect_leakage), ("adaptive_validity_check", G.adaptive_validity_check)]
    for name, node in pre:
        t0 = time.time(); upd = await node(state); fold(state, upd)
        keys = {k: upd.get(k) for k in ("error", "schema_validation_status", "qc_status", "ge_validation_status", "leakage_severity", "leaked_features") if k in (upd or {})}
        print(f"[{name} {time.time()-t0:.1f}s] {json.dumps(keys, default=str)} blocking={state.get('blocking_issues')}")
    route = G._route_after_leakage_detection(state)
    print(f"[route_after_leakage_detection] -> {route}" + ("  (LLM remediation NOT run by this probe)" if route == "remediate" else ""))
    for name, node in [("transform_data", G.transform_data), ("compute_baseline_metrics", G.compute_baseline_metrics),
                       ("sufficiency_check", G.run_sufficiency_check), ("finalize_output", G.finalize_output)]:
        t0 = time.time(); upd = await node(state); fold(state, upd)
        keys = {k: upd.get(k) for k in ("error", "sufficiency_status", "gate_passed", "qc_passed", "is_ready") if k in (upd or {})}
        print(f"[{name} {time.time()-t0:.1f}s] {json.dumps(keys, default=str)} blocking={state.get('blocking_issues')}")
    bi = state.get("blocking_issues") or []
    dup = len(bi) != len(set(map(str, bi)))
    print(f"GATE {cohort}/{brand}: gate_passed={state.get('gate_passed')} n_blocking={len(bi)} duplicates={dup}")
    return {"gate_passed": state.get("gate_passed"), "blocking": bi, "duplicates": dup, "route": route}


async def main():
    res = {c: await run(c, "Kisqali") for c in ("initiation", "persistence", "discontinuation")}
    print("\nSUMMARY:", json.dumps(res, default=str))
    ok = all(r["gate_passed"] is False and r["blocking"] and not r["duplicates"] for r in res.values() if r["route"] == "remediate" or r["blocking"])
    print("VERDICT:", "PASS" if ok else "FAIL", "(every cohort with a blocking condition reaches finalize with a non-empty, duplicate-free blocking_issues and gate_passed False)")


asyncio.run(main())
