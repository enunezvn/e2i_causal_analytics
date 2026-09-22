"""Cheapest disproof: does guided discovery (production graph_builder path) yield a
gate-acceptable, role-sane DAG on (A) a REAL Optum claims cohort frame and (B) a
synthetic frame with planted confounders?  No DB / Redis / Opik writes:
_run_discovery + gate only (never GraphBuilderNode.execute, which persists)."""
import os
import asyncio, sys, time, json, os
import numpy as np, pandas as pd
DATA_ROOT = os.environ.get("E2I_DATA_ROOT", ".")  # data/ is gitignored; point at the checkout that holds it
import src; print("src resolves to:", src.__file__)  # a script run BY PATH imports src from the MAIN checkout (editable .pth); use a cwd-first import to test a worktree
from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
from src.causal_engine.discovery import DiscoveryRunner
from src.ml.causal_role_dgp.extractor import extract_role

def roles(dag, T, Y, covs):
    out = {}
    for c in covs:
        if c not in dag.nodes(): out[c] = "absent"; continue
        try: out[c] = extract_role(c, T, Y, dag)
        except ValueError: out[c] = "unrelated"
    return out

async def run(label, df, T, Y, covs, planted=None):
    node = GraphBuilderNode()
    node._discovery_runner = DiscoveryRunner(enable_tracing=False)
    state = {"treatment_var": T, "outcome_var": Y, "confounders": covs, "modeled_confounders": covs,
             "anchored_confounders": [], "auto_discover": True, "discovery_guided": True,
             "data_cache": {"estimation_data": df[[T, Y] + covs]}, "session_id": None,
             "discovery_bootstrap_resamples": 20, "query": f"{T} -> {Y}"}
    t0 = time.time()
    result, gate = await node._run_discovery(state, T, Y)
    wall = time.time() - t0
    dag, aug, overridden = node._build_dag_with_discovery(T, Y, covs, result, gate, anchored_confounders=[])
    adj = node._find_adjustment_sets(dag, T, Y)
    ens = result.ensemble_dag
    r = roles(ens, T, Y, covs) if ens is not None else {}
    hist = {}
    for v in r.values(): hist[v] = hist.get(v, 0) + 1
    print(f"\n===== {label}: n={len(df)} T={T} Y={Y} k={len(covs)} wall={wall:.1f}s")
    print("gate:", gate["decision"], f"conf={gate['confidence']:.3f}", "| reasons:", gate.get("reasons"))
    print("ensemble edges:", result.n_edges, "| shipped DAG edges:", dag.number_of_edges(), "| overridden:", overridden, "| augmented:", aug)
    print("edges into T:", sorted(ens.predecessors(T)) if ens is not None and T in ens else None)
    print("edges into Y:", sorted(ens.predecessors(Y)) if ens is not None and Y in ens else None)
    print("T->Y in ensemble:", bool(ens is not None and ens.has_edge(T, Y)), "| Y->T:", bool(ens is not None and ens.has_edge(Y, T)))
    print("role histogram (discovered DAG):", hist)
    print("adjustment sets (shipped DAG, pre-guarantee):", adj[:3])
    print("latent_diagnostic:", json.dumps(result.metadata.get("latent_diagnostic"), default=str)[:300])
    if planted:
        print("PLANTED confounders:", {c: r.get(c) for c in planted})
    return gate["decision"]

def numeric_covs(df, exclude, cap=20):
    cols = []
    for c in df.columns:
        if c in exclude: continue
        s = df[c]
        if not (np.issubdtype(s.dtype, np.number) or s.dtype == bool): continue
        if s.nunique(dropna=True) < 2: continue
        cols.append((s.astype(float).var(), c))
    cols.sort(reverse=True)
    return [c for _, c in cols[:cap]]

async def main():
    df0 = pd.read_parquet(DATA_ROOT + "/data/rwd/mart/persistence/e2i_ml_v3_patient_journeys.parquet")
    df0 = df0[df0["data_split"] == "train"].sample(n=4000, random_state=0).reset_index(drop=True)
    T, Y = "lis_dual_flag", "persistent_at_180d"
    ids = {"patient_journey_id","patient_id","patient_hash","index_date","journey_start_date","journey_status","discontinuation_flag","data_split", T, Y}
    # A0: PRODUCTION shape = every numeric non-id column (graph_builder tiers ALL frame columns)
    covs0 = numeric_covs(df0, ids, cap=200)
    d0 = df0[[T, Y] + covs0].dropna().astype(float)
    print("A0 k=", len(covs0))
    rk = np.linalg.matrix_rank(np.corrcoef(d0.values.T))
    print("A0 corr-matrix rank", rk, "of", d0.shape[1])
    dA0 = await run("A0_REAL_production_shape_all_numeric", d0, T, Y, covs0)
    # A1: collinearity-pruned (one comorbidity family, no composite scores/derived flags)
    keep = ["age_at_index","enrollment_duration_days","comorbidity_diag_distinct_count","charlson_score"]
    cci = [c for c in df0.columns if c.startswith("cci_") and df0[c].mean() >= 0.02]
    covs1 = keep + cci
    d1 = df0[[T, Y] + covs1].dropna().astype(float)
    rk = np.linalg.matrix_rank(np.corrcoef(d1.values.T))
    print("A1 covs", covs1, "| corr rank", rk, "of", d1.shape[1])
    dA1 = await run("A1_REAL_pruned_cci_family", d1, T, Y, covs1)
    # A2: pruned, drop charlson_score too (it is a function of the cci_* flags)
    covs2 = [c for c in covs1 if c != "charlson_score"]
    d2 = df0[[T, Y] + covs2].dropna().astype(float)
    rk = np.linalg.matrix_rank(np.corrcoef(d2.values.T))
    print("A2 corr rank", rk, "of", d2.shape[1])
    dA2 = await run("A2_REAL_pruned_no_composite", d2, T, Y, covs2)
    # B: synthetic patient_journeys (the prod causal dataset shape) with planted truth
    dfb = pd.read_parquet(DATA_ROOT + "/data/rwd/synthetic_CSU/patient_journeys.parquet")
    dfb = dfb[dfb["brand"] == "Remibrutinib"] if "brand" in dfb else dfb
    T2, Y2 = "treatment_arm", "treatment_initiated"
    leak = {"propensity_score","treatment_effect_estimate","days_to_treatment","is_synthetic","data_split",
            "persistent_180d","discontinued_180d","outcome"}
    ids2 = {c for c in dfb.columns if c.endswith("_id") or c.endswith("_date") or c.endswith("_at")}
    covs2b = numeric_covs(dfb, leak | ids2 | {T2, Y2}, cap=25)
    dfb = dfb[[T2, Y2] + covs2b].dropna().sample(n=min(4000, len(dfb)), random_state=0).astype(float)
    print("B covariates:", covs2b)
    dB = await run("B_SYNTH_patient_journeys_planted", dfb, T2, Y2, covs2b, planted=["disease_severity","academic_hcp"])
    print("\nSUMMARY: A0(prod shape)=", dA0, " A1(pruned)=", dA1, " A2(pruned,no composite)=", dA2, " B(synthetic)=", dB)

asyncio.run(main())

