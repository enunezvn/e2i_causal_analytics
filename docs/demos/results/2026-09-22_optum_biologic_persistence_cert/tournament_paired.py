"""Paired before/after on ONE loaded frame per served pair: the selector runs twice on the same
rows, once with the bare LogisticRegressionCV propensity (pre-change) and once with the
StandardScaler pipeline (post-change). Removes the loader's row-subset nondeterminism."""

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
from src.causal_engine.energy_score import estimator_selector as es, dml_learner as dl
from src.causal_engine.energy_score.estimator_selector import (
    EstimatorSelector,
    EstimatorSelectorConfig,
)
from src.causal_engine.nuisance_config import propensity_model as NEW
from src.repositories.provenance import PROVENANCE_DROP_COLS


def OLD():
    from sklearn.linear_model import LogisticRegressionCV

    return LogisticRegressionCV(cv=3, max_iter=500)


def run(tb, y, X):
    sel = EstimatorSelector(EstimatorSelectorConfig()).select(tb, y, X)
    return {
        "selected": sel.selected.estimator_type.value,
        "ate": sel.selected.ate,
        "energy_scores": sel.energy_scores,
        "gap": sel.energy_score_gap,
        "per": {
            r.estimator_type.value: {
                "ate": r.ate,
                "energy": (r.energy_score if r.success else None),
            }
            for r in sel.all_results
        },
    }


async def main(out_path, datasets, per_scope):
    rows = []
    for ds in datasets:
        brands = list(await _list_dataset_brands(ds))[:2] or [None]
        for b in brands:
            brand, qs = await _resolve_discovery_scope(ds, b)
            for q in qs[:per_scope]:
                df, _ = await _load_agent_estimation_frame(
                    dataset=ds,
                    treatment_var=q.treatment,
                    outcome_var=q.outcome,
                    covariates=list(q.adjustment_set),
                    limit=_DISCOVERY_ROW_CAP,
                    brand=q.brand or brand,
                )
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
                es.propensity_model = OLD
                dl.propensity_model = OLD
                before = run(tb, y, X)
                es.propensity_model = NEW
                dl.propensity_model = NEW
                after = run(tb, y, X)
                row = {
                    "dataset": ds,
                    "brand": q.brand or brand,
                    "treatment": q.treatment,
                    "outcome": q.outcome,
                    "n": int(len(df)),
                    "k": int(X.shape[1]),
                    "before": before,
                    "after": after,
                }
                rows.append(row)
                Path(out_path).write_text(json.dumps(rows, indent=1, default=str))
                print(
                    json.dumps(
                        {
                            "pair": [ds, row["brand"], q.treatment, q.outcome],
                            "k": row["k"],
                            "win": [before["selected"], after["selected"]],
                            "ate_same": before["ate"] == after["ate"],
                            "dE": max(
                                abs(
                                    (before["energy_scores"].get(e) or 0)
                                    - (after["energy_scores"].get(e) or 0)
                                )
                                for e in before["energy_scores"]
                            ),
                        }
                    ),
                    flush=True,
                )
    print("done", len(rows), flush=True)


asyncio.run(main(sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])))
