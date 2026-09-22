"""Lane D item 5: acceptance measurement on the real persistence frame and the
planted synthetic frame, through the production node ``GraphBuilderNode.execute``
(pre-flight -> guided PC -> bootstrap under budget -> latent diagnostic under budget
-> gate -> shipped DAG + adjustment guarantee -> warnings) with Supabase/Redis
blanked so nothing persists. Defaults exactly as production wires them
(cap 20, budget 180 s, B = 20, min_resamples 10, latent diagnostic ON).

    python - < docs/demos/results/2026-09-22_lane_d_guided_discovery_claims/run_acceptance.py [real|synthetic|both]
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

from src.agents.causal_impact.nodes.graph_builder import (  # noqa: E402
    DISCOVERY_MAX_COVARIATES,
    DISCOVERY_MIN_RESAMPLES,
    DISCOVERY_TIME_BUDGET_S,
    GraphBuilderNode,
)
from src.causal_engine.discovery import DiscoveryRunner  # noqa: E402
from src.data.manifests import MART_SAFE_FEATURES  # noqa: E402
from src.ml.causal_role_dgp.extractor import extract_role  # noqa: E402

DATA_ROOT = "/home/enunez/Projects/e2i_causal_analytics/data"
print(f"defaults: cap={DISCOVERY_MAX_COVARIATES} budget_s={DISCOVERY_TIME_BUDGET_S} min_resamples={DISCOVERY_MIN_RESAMPLES}")

# Phase timers: the first real run's node wall (504.6 s) was 300 s more than
# its discovery latency (202.7 s); wrap the two adjustment-set steps to see
# where the rest goes (read-only wrappers, behaviour unchanged).
PHASES: dict = {}


def _timed(name):
    raw = GraphBuilderNode.__dict__[name]
    is_static = isinstance(raw, staticmethod)
    original = raw.__func__ if is_static else raw

    def wrapper(*args, **kwargs):
        t0 = time.time()
        try:
            return original(*args, **kwargs)
        finally:
            PHASES[name] = round(time.time() - t0, 1)

    setattr(GraphBuilderNode, name, staticmethod(wrapper) if is_static else wrapper)


for _name in ("_find_adjustment_sets", "_apply_adjustment_guarantee", "_compute_edge_provenance"):
    _timed(_name)


async def run(label, frame, T, Y, covs, planted=None, **overrides):
    node = GraphBuilderNode()
    node._discovery_runner = DiscoveryRunner(enable_tracing=False)
    state = {
        "query": f"{T} -> {Y}", "treatment_var": T, "outcome_var": Y, "confounders": covs,
        "modeled_confounders": covs, "anchored_confounders": [], "auto_discover": True,
        "discovery_guided": True, "data_cache": {"estimation_data": frame[[T, Y] + covs]},
        "session_id": None, "warnings": [],
    }
    state.update(overrides)
    t0 = time.time()
    out = await node.execute(state)
    wall = time.time() - t0
    graph = out["causal_graph"]
    res = out.get("discovery_result") or {}
    meta = res.get("metadata") or {}
    boot = meta.get("bootstrap") or {}
    pf = meta.get("preflight") or {}
    latent = meta.get("latent_diagnostic") or {}
    adj = graph.get("adjustment_sets") or [[]]
    declared = set(covs)
    shipped_adj = set(adj[0]) if adj else set()
    line = {
        "run": label, "n": len(frame), "k_declared": len(covs), "wall_s": round(wall, 1),
        "discovery_latency_s": round(out.get("discovery_latency_ms", 0) / 1000, 1),
        "gate": graph.get("discovery_gate_decision"), "gate_confidence": round(graph.get("discovery_confidence", 0), 3),
        "dag_overridden": graph.get("discovery_dag_overridden"), "n_edges_ensemble": res.get("n_edges"),
        "shipped_edges": len(graph.get("edges") or []), "shipped_nodes": len(graph.get("nodes") or []),
        "preflight": {k: (len(v) if isinstance(v, list) else v) for k, v in pf.items() if k != "screening"},
        "preflight_k": (pf.get("screening") or {}).get("k"),
        "bootstrap": boot, "latent_diagnostic": {k: v for k, v in latent.items() if k != "bidirected_edges"},
        "required_edges_missing": meta.get("required_edges_missing"),
        "adjustment_set_size": len(shipped_adj),
        "declared_covariates_in_adjustment_set": len(declared & shipped_adj),
        "declared_missing_from_adjustment_set": sorted(declared - shipped_adj),
        "estimand_edge_shipped": [T, Y] in [list(e) for e in graph.get("edges") or []],
        "estimand_provenance": next((e["provenance"] for e in graph.get("edge_provenance") or [] if e["source"] == T and e["target"] == Y), None),
        "skip_reason": out.get("discovery_skip_reason"),
        "phase_seconds": dict(PHASES),
    }
    PHASES.clear()
    if planted:
        import networkx as nx
        dag = nx.DiGraph()
        dag.add_nodes_from(graph["nodes"])
        dag.add_edges_from(graph["edges"])
        r = {}
        for c in planted:
            try:
                r[c] = extract_role(c, T, Y, dag) if c in dag else "absent"
            except ValueError:
                r[c] = "unrelated"
        line["planted_roles_shipped_dag"] = r
    print("RUN", json.dumps(line, default=str))
    print("WARNINGS:")
    for w in out.get("warnings", []):
        print("  -", w[:600])
    return line


async def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "real"):
        frame, covs = resolve(list(MART_SAFE_FEATURES))
        print(f"REAL frame n={len(frame)} k={len(covs)}")
        await run("real/persistence_g28/defaults", frame, TREATMENT, OUTCOME, covs)
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
        print(f"SYNTHETIC frame n={len(dfb)} k={len(covs2)}")
        await run("synthetic/patient_journeys/defaults", dfb, T2, Y2, covs2,
                  planted=["disease_severity", "academic_hcp"])


asyncio.run(main())
