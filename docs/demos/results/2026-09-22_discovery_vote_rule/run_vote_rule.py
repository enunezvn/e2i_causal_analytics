"""Discovery ensemble vote rule -- measurement script (lane: discovery-vote-rule).

Two steps, each run from the tree under test with a cwd-first import
(``cd <tree> && python - --step ... < this_file``); a script run BY PATH inside a
worktree imports ``src`` from the MAIN checkout (editable .pth), so every capture
prints ``src resolves to:``.

  --step algo --frame {synthetic,real} --algo <name>
      One UNGUIDED single-algorithm run (no priors, no bootstrap, no latent
      diagnostic) through the production ``DiscoveryRunner`` with the runner's own
      per-algorithm timeout; writes ``<frame>_<algo>.json`` (edge list, edge types,
      runtime, convergence, error) next to this script.

  --step ensemble --frame {synthetic,real}
      Reads every ``<frame>_<algo>.json``, rebuilds ``AlgorithmResult`` objects and
      votes them through ``DiscoveryRunner._build_ensemble`` AS SHIPPED in the tree
      under test, next to two reference rules computed here (union: an edge needs
      >= 1 vote; agreement: >= max(2, ceil(n * threshold)) votes), then grades every
      variant with ``DiscoveryGate``. On the synthetic frame it scores planted-edge
      recovery.

No DB / Redis / Opik writes: the runner is built with ``enable_tracing=False`` and
nothing here calls ``GraphBuilderNode.execute``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import src  # noqa: E402

print("src resolves to:", src.__file__)

from src.causal_engine.discovery.base import (  # noqa: E402
    AlgorithmResult,
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
    EdgeType,
)
from src.causal_engine.discovery.gate import DiscoveryGate  # noqa: E402
from src.causal_engine.discovery.runner import DiscoveryRunner  # noqa: E402

DATA_ROOT = os.environ.get("E2I_DATA_ROOT", "/home/enunez/Projects/e2i_causal_analytics")
OUT_DIR = Path(os.environ.get("VOTE_RULE_OUT", Path(__file__).resolve().parent if "__file__" in globals() else "."))

# Planted structure of the synthetic patient_journeys frame, read from the generator:
#   treatment_arm  <- disease_severity (beta 0.30), academic_hcp (beta 0.80)
#       src/ml/synthetic/dgp/treatment_arm.py:295-301 (assign_treatment_arm)
#   treatment_initiated <- treatment_arm, disease_severity, academic_hcp
#       src/ml/synthetic/generators/patient_generator.py:320-335 (generate_initiation_outcome)
#   treatment_initiated <- age_at_diagnosis (prognostic offset)
#       src/ml/synthetic/dgp/treatment_arm.py:455-478 (initiation_prognostic_offset)
#   engagement_score <- disease_severity (0.3), academic_hcp (2.0)
#       src/ml/synthetic/generators/patient_generator.py:1080-1087
SYNTH_T, SYNTH_Y = "treatment_arm", "treatment_initiated"
SYNTH_PLANTED: List[Tuple[str, str]] = [
    ("disease_severity", SYNTH_T),
    ("academic_hcp", SYNTH_T),
    (SYNTH_T, SYNTH_Y),
    ("disease_severity", SYNTH_Y),
    ("academic_hcp", SYNTH_Y),
    ("age_at_diagnosis", SYNTH_Y),
    ("disease_severity", "engagement_score"),
    ("academic_hcp", "engagement_score"),
]

REAL_T, REAL_Y = "treatment_dupixent", "persistent_at_180d_g28"
REAL_CAP_PER_SIDE = 7  # top-7 by |corr T| union top-7 by |corr Y| -> <= 14 covariates


# --------------------------------------------------------------------------- frames
def numeric_covs(df: pd.DataFrame, exclude: set, cap: int = 20) -> List[str]:
    """Same covariate selection as the discovery-disproof lane (run_disproof.py)."""
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if not (np.issubdtype(s.dtype, np.number) or s.dtype == bool):
            continue
        if s.nunique(dropna=True) < 2:
            continue
        cols.append((s.astype(float).var(), c))
    cols.sort(reverse=True)
    return [c for _, c in cols[:cap]]


def synthetic_frame() -> Tuple[pd.DataFrame, str, str, List[str]]:
    """Byte-for-byte the disproof lane's frame B (run_disproof.py:95-103)."""
    dfb = pd.read_parquet(DATA_ROOT + "/data/rwd/synthetic_CSU/patient_journeys.parquet")
    dfb = dfb[dfb["brand"] == "Remibrutinib"] if "brand" in dfb else dfb
    leak = {
        "propensity_score", "treatment_effect_estimate", "days_to_treatment", "is_synthetic",
        "data_split", "persistent_180d", "discontinued_180d", "outcome",
    }
    ids = {c for c in dfb.columns if c.endswith("_id") or c.endswith("_date") or c.endswith("_at")}
    covs = numeric_covs(dfb, leak | ids | {SYNTH_T, SYNTH_Y}, cap=25)
    dfb = dfb[[SYNTH_T, SYNTH_Y] + covs].dropna().sample(n=min(4000, len(dfb)), random_state=0)
    return dfb.astype(float).reset_index(drop=True), SYNTH_T, SYNTH_Y, covs


def greedy_rank_prune(df: pd.DataFrame, fixed: List[str], candidates: List[str]) -> Tuple[List[str], List[str]]:
    """Keep a candidate iff it raises the correlation-matrix rank (Lane D's rule)."""
    kept: List[str] = list(fixed)
    dropped: List[str] = []
    rank = np.linalg.matrix_rank(np.corrcoef(df[kept].values.T))
    for c in candidates:
        trial = kept + [c]
        r = np.linalg.matrix_rank(np.corrcoef(df[trial].values.T))
        if r > rank:
            kept, rank = trial, r
        else:
            dropped.append(c)
    return [c for c in kept if c not in fixed], dropped


def real_frame() -> Tuple[pd.DataFrame, str, str, List[str]]:
    from src.data.manifests import MART_SAFE_FEATURES

    raw = pd.read_parquet(
        DATA_ROOT + "/data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet"
    )
    numeric = [
        c
        for c in MART_SAFE_FEATURES
        if np.issubdtype(raw[c].dtype, np.number) or raw[c].dtype == bool
    ]
    df = raw[[REAL_T, REAL_Y] + numeric].dropna().astype(float).reset_index(drop=True)
    constant = [c for c in numeric if df[c].nunique() < 2]
    numeric = [c for c in numeric if c not in constant]
    kept, collinear = greedy_rank_prune(df, [REAL_T, REAL_Y], numeric)
    corr_t = df[kept].corrwith(df[REAL_T]).abs().sort_values(ascending=False, kind="stable")
    corr_y = df[kept].corrwith(df[REAL_Y]).abs().sort_values(ascending=False, kind="stable")
    top_t = list(corr_t.index[:REAL_CAP_PER_SIDE])
    top_y = list(corr_y.index[:REAL_CAP_PER_SIDE])
    capped = [c for c in kept if c in set(top_t) | set(top_y)]  # manifest order
    print(f"REAL rows={len(df)} numeric_manifest_covs={len(numeric) + len(constant)} constant={constant}")
    print(f"REAL collinear_dropped({len(collinear)})={collinear}")
    print(f"REAL rank_kept={len(kept)} top|corr T|={top_t}")
    print(f"REAL top|corr Y|={top_y}")
    print(f"REAL capped_covs({len(capped)})={capped}")
    print(f"REAL binary_among_capped={sum(int(df[c].nunique() == 2) for c in capped)}")
    return df[[REAL_T, REAL_Y] + capped], REAL_T, REAL_Y, capped


def load_frame(name: str):
    return synthetic_frame() if name == "synthetic" else real_frame()


# --------------------------------------------------------------------------- metrics
def skeleton(edges: Sequence[Tuple[str, str]]) -> set:
    return {frozenset(e) for e in edges}


def planted_metrics(edges: Sequence[Tuple[str, str]], planted: Sequence[Tuple[str, str]]) -> Dict[str, object]:
    e = {tuple(x) for x in edges}
    sk = skeleton(e)
    psk = skeleton(planted)
    oriented = [p for p in planted if p in e and (p[1], p[0]) not in e]
    reversed_ = [p for p in planted if (p[1], p[0]) in e and p not in e]
    undirected_hit = [p for p in planted if p in e and (p[1], p[0]) in e]
    missed = [p for p in planted if frozenset(p) not in sk]
    false_pairs = sorted(tuple(sorted(x)) for x in sk - psk)
    reciprocal = sum(1 for (a, b) in e if (b, a) in e) // 2
    return {
        "n_edges": len(e),
        "skeleton_recall": f"{len(planted) - len(missed)}/{len(planted)}",
        "oriented_recall": f"{len(oriented)}/{len(planted)}",
        "reversed": reversed_,
        "undirected_hit": undirected_hit,
        "missed": missed,
        "false_pairs": false_pairs,
        "n_false_pairs": len(false_pairs),
        "reciprocal_pairs": reciprocal,
    }


def plain_metrics(edges: Sequence[Tuple[str, str]], t: str, y: str) -> Dict[str, object]:
    e = {tuple(x) for x in edges}
    reciprocal = sum(1 for (a, b) in e if (b, a) in e) // 2
    return {
        "n_edges": len(e),
        "n_pairs": len(skeleton(e)),
        "reciprocal_pairs": reciprocal,
        "T->Y": (t, y) in e,
        "Y->T": (y, t) in e,
        "into_T": sorted(a for (a, b) in e if b == t),
        "into_Y": sorted(a for (a, b) in e if b == y),
    }


# --------------------------------------------------------------------------- step: algo
def step_algo(frame: str, algo: str, cap: float) -> None:
    df, t, y, covs = load_frame(frame)
    algo_type = DiscoveryAlgorithmType(algo)
    runner = DiscoveryRunner(enable_tracing=False, timeout_seconds=cap)
    config = DiscoveryConfig(algorithms=[algo_type], bootstrap_resamples=0, latent_diagnostic=False)
    t0 = time.time()
    result = runner.discover_dag_sync(df, config)
    wall = time.time() - t0
    (run,) = result.algorithm_results
    payload = {
        "frame": frame,
        "algorithm": algo,
        "n": int(len(df)),
        "columns": list(df.columns),
        "wall_seconds": round(wall, 2),
        "runtime_seconds": round(run.runtime_seconds, 2),
        "converged": bool(run.converged),
        "error": run.metadata.get("error"),
        "edge_list": [list(e) for e in run.edge_list],
        "edge_types": run.metadata.get("edge_types"),
        "n_bidirected": run.metadata.get("n_bidirected_edges"),
        "n_undirected": run.metadata.get("n_undirected_edges"),
        "indep_test": run.metadata.get("indep_test"),
    }
    if frame == "synthetic":
        payload["planted"] = planted_metrics(run.edge_list, SYNTH_PLANTED) if run.converged else None
    else:
        payload["summary"] = plain_metrics(run.edge_list, t, y) if run.converged else None
    out = OUT_DIR / f"{frame}_{algo}.json"
    out.write_text(json.dumps(payload, indent=1, default=str))
    print(
        f"ALGO {frame} {algo}: converged={payload['converged']} n_edges={len(run.edge_list)} "
        f"runtime={payload['runtime_seconds']}s wall={payload['wall_seconds']}s error={payload['error']!r}"
    )
    if payload.get("planted"):
        print(f"ALGO {frame} {algo} planted: {json.dumps(payload['planted'])}")
    if payload.get("summary"):
        print(f"ALGO {frame} {algo} summary: {json.dumps(payload['summary'])}")
    if payload["edge_types"]:
        counts: Dict[str, int] = {}
        for v in payload["edge_types"].values():
            counts[v] = counts.get(v, 0) + 1
        print(f"ALGO {frame} {algo} edge_type_counts: {counts}")
    print(f"ALGO wrote {out}")


# --------------------------------------------------------------------------- step: ensemble
def reference_vote(results: List[AlgorithmResult], node_names: List[str], threshold: float, rule: str):
    """Reference vote rules, computed independently of the tree's _build_ensemble."""
    import networkx as nx

    converged = [r for r in results if r.converged]
    n = len(converged)
    votes: Dict[Tuple[str, str], List[str]] = {}
    for r in converged:
        for s, d in r.edge_list:
            votes.setdefault((s, d), []).append(r.algorithm.value)
    if rule == "union":
        min_votes = max(1, int(n * threshold))
    elif rule == "agreement":
        min_votes = 1 if n < 2 else max(2, math.ceil(n * threshold - 1e-9))
    else:
        raise ValueError(rule)
    edges = [
        DiscoveredEdge(source=s, target=d, edge_type=EdgeType.DIRECTED, confidence=len(a) / n,
                       algorithm_votes=len(a), algorithms=a)
        for (s, d), a in votes.items()
        if len(a) >= min_votes
    ]
    dag = nx.DiGraph()
    dag.add_nodes_from(node_names)
    for e in edges:
        dag.add_edge(e.source, e.target, confidence=e.confidence, votes=e.algorithm_votes)
    dag = DiscoveryRunner()._remove_cycles(dag)
    return edges, dag, {"min_votes": min_votes, "n_candidates": len(votes), "n_kept": len(edges)}


def gate_line(label: str, edges, dag, results, config) -> None:
    res = DiscoveryResult(success=True, config=config, ensemble_dag=dag, edges=edges,
                          algorithm_results=results, metadata={})
    ev = DiscoveryGate().evaluate(res)
    meta = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.metadata.items()
            if k in ("corroboration_score", "corroboration_basis", "edge_confidence_score", "structure_score")}
    confs = sorted({round(e.confidence, 3) for e in edges})
    print(f"GATE {label}: decision={ev.decision.value} confidence={ev.confidence:.3f} "
          f"n_edges={len(edges)} dag_edges={dag.number_of_edges()} edge_confidences={confs} {json.dumps(meta)}")


def step_ensemble(frame: str, combos: List[List[str]], threshold: float) -> None:
    loaded: Dict[str, dict] = {}
    for p in sorted(OUT_DIR.glob(f"{frame}_*.json")):
        d = json.loads(p.read_text())
        loaded[d["algorithm"]] = d
    print(f"ENSEMBLE {frame}: loaded runs={sorted(loaded)}")
    if not loaded:
        sys.exit("no per-algorithm captures found")
    columns = next(iter(loaded.values()))["columns"]
    t, y = (SYNTH_T, SYNTH_Y) if frame == "synthetic" else (REAL_T, REAL_Y)

    def as_result(d: dict) -> AlgorithmResult:
        k = len(columns)
        return AlgorithmResult(
            algorithm=DiscoveryAlgorithmType(d["algorithm"]),
            adjacency_matrix=np.zeros((k, k), dtype=int),
            edge_list=[tuple(e) for e in d["edge_list"]],
            runtime_seconds=d["runtime_seconds"],
            converged=d["converged"],
            metadata={"error": d["error"]} if d["error"] else {},
        )

    runner = DiscoveryRunner(enable_tracing=False)
    for combo in combos:
        missing = [a for a in combo if a not in loaded]
        if missing:
            print(f"ENSEMBLE {frame} {'+'.join(combo)}: SKIPPED, no capture for {missing}")
            continue
        results = [as_result(loaded[a]) for a in combo]
        config = DiscoveryConfig(algorithms=[r.algorithm for r in results], ensemble_threshold=threshold)
        n_conv = sum(r.converged for r in results)
        label = "+".join(combo)
        print(f"ENSEMBLE {label}: converged={n_conv}/{len(results)} threshold={threshold}")
        variants = {}
        shipped_edges, shipped_dag = runner._build_ensemble(results, columns, threshold)
        variants["tree_build_ensemble"] = (shipped_edges, shipped_dag, dict(shipped_dag.graph))
        for rule in ("union", "agreement"):
            e, g, info = reference_vote(results, columns, threshold, rule)
            variants[f"ref_{rule}"] = (e, g, info)
        for name, (edges, dag, info) in variants.items():
            dag_edges = list(dag.edges())
            if frame == "synthetic":
                m_votes = planted_metrics([(e.source, e.target) for e in edges], SYNTH_PLANTED)
                m_dag = planted_metrics(dag_edges, SYNTH_PLANTED)
            else:
                m_votes = plain_metrics([(e.source, e.target) for e in edges], t, y)
                m_dag = plain_metrics(dag_edges, t, y)
            print(f"VOTES {label} {name}: info={json.dumps(info, default=str)}")
            print(f"VOTES {label} {name} voted_edges: {json.dumps(m_votes)}")
            print(f"VOTES {label} {name} final_dag: {json.dumps(m_dag)}")
            gate_line(f"{label} {name}", edges, dag, results, config)
        # Single-algorithm gate lines for reference (uncorroborated by construction: no bootstrap here).
    for a, d in loaded.items():
        r = as_result(d)
        if not r.converged:
            print(f"GATE {a} alone: not converged ({d['error']!r})")
            continue
        edges, dag = runner._build_ensemble([r], columns, threshold)
        config = DiscoveryConfig(algorithms=[r.algorithm], ensemble_threshold=threshold)
        gate_line(f"{a} alone (no bootstrap)", edges, dag, [r], config)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["algo", "ensemble"], required=True)
    ap.add_argument("--frame", choices=["synthetic", "real"], required=True)
    ap.add_argument("--algo", default=None)
    ap.add_argument("--cap", type=float, default=300.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--combos", default="ges,pc;ges,pc,fci")
    args = ap.parse_args()
    if args.step == "algo":
        if not args.algo:
            sys.exit("--algo required")
        step_algo(args.frame, args.algo, args.cap)
    else:
        combos = [c.split(",") for c in args.combos.split(";") if c]
        step_ensemble(args.frame, combos, args.threshold)


if __name__ == "__main__":
    main()
