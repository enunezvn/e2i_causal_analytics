"""Run the production estimator tournament on the LIVE synthetic-dataset frames (read-only DB
load through the route's own loader) and record winner / ATE / energy scores per served pair.
Run once BEFORE and once AFTER an engine change; diff the two JSONs. Usage: <out.json> [per_scope]"""

import asyncio, json, sys, time, logging, warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
import src

assert ".worktrees/real-data-causal" in src.__file__, src.__file__
from dotenv import load_dotenv

load_dotenv("/home/enunez/Projects/e2i_causal_analytics/.env")
import numpy as np
from src.api.routes.causal.datasets import _DISCOVERY_ROW_CAP
from src.api.routes.causal.discovery import _resolve_discovery_scope, _list_dataset_brands
from src.api.routes.causal.loaders import _load_agent_estimation_frame
from src.agents.causal_impact.nodes.estimation import _encode_categorical_covariates
from src.causal_engine.energy_score.estimator_selector import (
    EstimatorSelector,
    EstimatorSelectorConfig,
)
from src.repositories.provenance import PROVENANCE_DROP_COLS


async def main(out_path: str, per_scope: int) -> None:
    scopes = []
    for ds in ("patient_journeys", "hcp_adoption", "nba_triggers"):
        try:
            brands = await _list_dataset_brands(ds)
        except Exception as e:  # noqa: BLE001
            print("brands fail", ds, type(e).__name__, e, flush=True)
            brands = []
        scopes += [(ds, b) for b in (list(brands)[:2] or [None])]
    print("scopes", scopes, flush=True)
    rows = []
    for ds, b in scopes:
        try:
            brand, qs = await _resolve_discovery_scope(ds, b)
        except Exception as e:  # noqa: BLE001
            print("scope fail", ds, b, type(e).__name__, str(e)[:200], flush=True)
            continue
        for q in qs[:per_scope]:
            try:
                df, _ = await _load_agent_estimation_frame(
                    dataset=ds,
                    treatment_var=q.treatment,
                    outcome_var=q.outcome,
                    covariates=list(q.adjustment_set),
                    limit=_DISCOVERY_ROW_CAP,
                    brand=q.brand or brand,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    "load fail",
                    ds,
                    b,
                    q.treatment,
                    q.outcome,
                    type(e).__name__,
                    str(e)[:200],
                    flush=True,
                )
                continue
            t = df[q.treatment].values
            y = df[q.outcome].values.astype(float)
            cov_cols = [
                c for c in q.adjustment_set if c in df.columns and c not in PROVENANCE_DROP_COLS
            ]
            X = _encode_categorical_covariates(df[cov_cols] if cov_cols else df.iloc[:, :0])
            tb = (
                (t > np.median(t)).astype(int)
                if not np.array_equal(t, t.astype(int))
                else t.astype(int)
            )
            t0 = time.perf_counter()
            sel = EstimatorSelector(EstimatorSelectorConfig()).select(tb, y, X)
            dt = time.perf_counter() - t0
            row = {
                "dataset": ds,
                "brand": q.brand or brand,
                "treatment": q.treatment,
                "outcome": q.outcome,
                "n": int(len(df)),
                "k": int(X.shape[1]),
                "selected": sel.selected.estimator_type.value,
                "ate": sel.selected.ate,
                "ci": [sel.selected.ate_ci_lower, sel.selected.ate_ci_upper],
                "energy_scores": sel.energy_scores,
                "gap": sel.energy_score_gap,
                "subsampled": getattr(sel, "selection_subsampled", None),
                "wall_s": round(dt, 1),
                "per_estimator": {
                    r.estimator_type.value: {
                        "success": r.success,
                        "ate": r.ate,
                        "energy": (r.energy_score if r.success else None),
                        "ms": round(r.estimation_time_ms or 0),
                        "ps_mean": (
                            float(np.mean(r.propensity_scores))
                            if r.propensity_scores is not None
                            else None
                        ),
                        "ps_sd": (
                            float(np.std(r.propensity_scores))
                            if r.propensity_scores is not None
                            else None
                        ),
                    }
                    for r in sel.all_results
                },
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "dataset",
                            "brand",
                            "treatment",
                            "outcome",
                            "n",
                            "k",
                            "selected",
                            "ate",
                            "gap",
                            "wall_s",
                        )
                    }
                ),
                flush=True,
            )
            Path(out_path).write_text(json.dumps(rows, indent=1, default=str))
    print("done", len(rows), flush=True)


asyncio.run(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3))
