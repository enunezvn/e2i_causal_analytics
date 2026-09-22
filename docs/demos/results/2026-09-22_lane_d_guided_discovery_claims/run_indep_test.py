"""Lane D item 4: fisherz vs gsq vs chisq, MEASURED on (a) the capped real Optum
persistence frame and (b) the planted synthetic patient_journeys frame as control.

Production entry point ``GraphBuilderNode._run_discovery`` (pre-flight -> guided PC
-> bootstrap under budget -> gate), latent diagnostic OFF so the arms compare the
skeleton test alone. gsq/chisq need discrete data: for those arms the non-binary
columns are quantile-binned to <= 10 levels (the #2009 measurement's convention);
fisherz runs on the production (unbinned) frame. Run from the worktree root:

    python - real chisq < docs/demos/results/2026-09-22_lane_d_guided_discovery_claims/run_indep_test.py
    (argv: [real|synthetic|both] [comma-separated tests, default fisherz,gsq,chisq])
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "docs/demos/results/2026-09-22_lane_d_guided_discovery_claims"))
for key in ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_SERVICE_ROLE_KEY", "REDIS_URL"):
    os.environ[key] = ""
import src  # noqa: E402

assert ".worktrees/lane-d-guided-discovery-claims" in src.__file__, src.__file__
print("src resolves to:", src.__file__)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from frame_resolver import OUTCOME, TREATMENT, resolve  # noqa: E402

from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode  # noqa: E402
from src.causal_engine.discovery import DiscoveryRunner  # noqa: E402
from src.data.manifests import MART_SAFE_FEATURES  # noqa: E402
from src.ml.causal_role_dgp.extractor import extract_role  # noqa: E402

DATA_ROOT = "/home/enunez/Projects/e2i_causal_analytics/data"


def quantile_bin(frame: pd.DataFrame, cols: list, levels: int = 10) -> pd.DataFrame:
    out = frame.copy()
    for c in cols:
        if out[c].nunique() > levels:
            out[c] = pd.qcut(out[c].rank(method="first"), q=levels, labels=False).astype(float)
    return out


def roles(dag, T, Y, covs):
    out = {}
    for c in covs:
        if dag is None or c not in dag.nodes():
            out[c] = "absent"
            continue
        try:
            out[c] = extract_role(c, T, Y, dag)
        except ValueError:
            out[c] = "unrelated"
    return out


async def run_arm(label, frame, T, Y, covs, indep_test, planted=None):
    node = GraphBuilderNode()
    node._discovery_runner = DiscoveryRunner(enable_tracing=False)
    state = {
        "treatment_var": T, "outcome_var": Y, "confounders": covs, "modeled_confounders": covs,
        "anchored_confounders": [], "auto_discover": True, "discovery_guided": True,
        "data_cache": {"estimation_data": frame[[T, Y] + covs]}, "session_id": None,
        "discovery_latent_diagnostic": False, "discovery_indep_test": indep_test,
        "query": f"{T} -> {Y}",
    }
    t0 = time.time()
    result, gate = await node._run_discovery(state, T, Y)
    wall = time.time() - t0
    pc = result.algorithm_results[0]
    boot = result.metadata.get("bootstrap") or {}
    pf = result.metadata.get("preflight") or {}
    ens = result.ensemble_dag
    beyond = [e for e in result.edges if (e.source, e.target) != (T, Y)]
    stab = [e.bootstrap_stability for e in beyond if e.bootstrap_stability is not None]
    line = {
        "arm": label, "indep_test": pc.metadata.get("indep_test"), "converged": pc.converged,
        "error": pc.metadata.get("error"), "wall_s": round(wall, 1),
        "primary_fit_s": round(pc.runtime_seconds, 1), "n_edges": result.n_edges,
        "gate": gate["decision"], "confidence": round(gate["confidence"], 3),
        "corroboration_basis": gate.get("metadata", {}).get("corroboration_basis"),
        "n_attempted": boot.get("n_attempted"), "n_succeeded": boot.get("n_succeeded"),
        "budget_exhausted": boot.get("budget_exhausted"),
        "mean_stability_beyond_prior": round(float(np.mean(stab)), 3) if stab else None,
        "share_edges_stability_ge_0.9": round(float(np.mean([s >= 0.9 for s in stab])), 3) if stab else None,
        "preflight_kept": pf.get("n_kept"),
        "T_to_Y_in_ensemble": bool(ens is not None and ens.has_edge(T, Y)),
        "parents_T": sorted(ens.predecessors(T)) if ens is not None and T in ens else None,
        "parents_Y": sorted(ens.predecessors(Y)) if ens is not None and Y in ens else None,
    }
    if planted:
        r = roles(ens, T, Y, covs)
        line["planted_roles"] = {c: r.get(c) for c in planted}
    print("ARM", json.dumps(line, default=str))
    return line


async def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    # Optional second argument: a comma-separated subset of the tests, so ONE
    # arm can run under its own hard ``timeout`` (a resample fit that does not
    # return cannot be interrupted from inside the process; the real/gsq arm
    # ran > 53 min on one fit before it was stopped by hand -- d7_gsq_arm_stopped.txt).
    tests = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("fisherz", "gsq", "chisq")
    if which in ("both", "real"):
        frame, covs = resolve(list(MART_SAFE_FEATURES))
        nonbinary = [c for c in covs if frame[c].nunique() > 2]
        print(f"REAL frame n={len(frame)} k={len(covs)} non-binary={nonbinary}")
        for test in tests:
            arm_frame = frame if test == "fisherz" else quantile_bin(frame, nonbinary)
            await run_arm(f"real/{test}", arm_frame, TREATMENT, OUTCOME, covs, test)
    if which in ("both", "synthetic"):
        dfb = pd.read_parquet(DATA_ROOT + "/rwd/synthetic_CSU/patient_journeys.parquet")
        dfb = dfb[dfb["brand"] == "Remibrutinib"] if "brand" in dfb else dfb
        T2, Y2 = "treatment_arm", "treatment_initiated"
        leak = {"propensity_score", "treatment_effect_estimate", "days_to_treatment", "is_synthetic",
                "data_split", "persistent_180d", "discontinued_180d", "outcome"}
        ids2 = {c for c in dfb.columns if c.endswith("_id") or c.endswith("_date") or c.endswith("_at")}
        cols = []
        for c in dfb.columns:
            if c in leak | ids2 | {T2, Y2}:
                continue
            s = dfb[c]
            if not (np.issubdtype(s.dtype, np.number) or s.dtype == bool) or s.nunique(dropna=True) < 2:
                continue
            cols.append((s.astype(float).var(), c))
        cols.sort(reverse=True)
        covs2 = [c for _, c in cols[:25]]
        dfb = dfb[[T2, Y2] + covs2].dropna().sample(n=min(4000, len(dfb)), random_state=0).astype(float)
        nonbinary2 = [c for c in covs2 if dfb[c].nunique() > 2]
        print(f"SYNTHETIC frame n={len(dfb)} k={len(covs2)} non-binary={len(nonbinary2)}")
        for test in tests:
            arm_frame = dfb if test == "fisherz" else quantile_bin(dfb, nonbinary2)
            await run_arm(f"synthetic/{test}", arm_frame, T2, Y2, covs2, test,
                          planted=["disease_severity", "academic_hcp"])


asyncio.run(main())
