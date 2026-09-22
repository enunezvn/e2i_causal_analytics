import os
import asyncio, time, numpy as np, pandas as pd
DATA_ROOT = os.environ.get("E2I_DATA_ROOT", ".")  # data/ is gitignored; point at the checkout that holds it
import src; print("src resolves to:", src.__file__)  # a script run BY PATH imports src from the MAIN checkout (editable .pth); use a cwd-first import to test a worktree
from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
from src.causal_engine.discovery import DiscoveryRunner
df = pd.read_parquet(DATA_ROOT + "/data/rwd/mart/persistence/e2i_ml_v3_patient_journeys.parquet")
df = df[df["data_split"]=="train"].sample(n=4000, random_state=0).reset_index(drop=True)
T,Y = "lis_dual_flag","persistent_at_180d"
ids = {"patient_journey_id","patient_id","patient_hash","index_date","journey_start_date","journey_status","discontinuation_flag","data_split",T,Y}
num = [c for c in df.columns if c not in ids and np.issubdtype(df[c].dtype, np.number) and df[c].nunique()>1]
X = df[[T,Y]+num].astype(float).dropna()
# greedy rank-preserving prune: keep a column only if it raises the rank
kept, dropped = [T,Y], []
M = X[kept].values
for c in num:
    cand = np.column_stack([M, X[c].values])
    if np.linalg.matrix_rank(np.corrcoef(cand.T)) > np.linalg.matrix_rank(np.corrcoef(M.T)):
        kept.append(c); M = cand
    else:
        dropped.append(c)
covs = kept[2:]
print("kept k=", len(covs), "| dropped (linearly dependent):", dropped)
async def main():
    node = GraphBuilderNode(); node._discovery_runner = DiscoveryRunner(enable_tracing=False)
    for boots in (0,):
        st = {"treatment_var":T,"outcome_var":Y,"confounders":covs,"modeled_confounders":covs,"anchored_confounders":[],
              "auto_discover":True,"discovery_guided":True,"data_cache":{"estimation_data":X[[T,Y]+covs]},
              "session_id":None,"discovery_bootstrap_resamples":boots,"discovery_latent_diagnostic":False}
        t0=time.time(); res, gate = await node._run_discovery(st,T,Y); w=time.time()-t0
        print(f"bootstrap={boots}: wall={w:.1f}s edges={res.n_edges} gate={gate['decision']} conf={gate['confidence']:.2f} reasons={gate['reasons'][:2]}")
        print("T->Y:", res.ensemble_dag.has_edge(T,Y) if res.ensemble_dag is not None else None, "| parents(T):", sorted(res.ensemble_dag.predecessors(T)) if res.ensemble_dag is not None else None)
asyncio.run(main())
