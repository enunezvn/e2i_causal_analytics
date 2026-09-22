#!/usr/bin/env python3
"""Lane A pre-merge pre-flight: run the causal_impact agent IN-PROCESS on the
exported causal frame through the production coercion + one-hot helpers and the
route's own background task. Measures wall time and records the estimate so the
live cert (post-deploy) is not the first time the run shape is exercised.

Suggestive, not decisive: this is the host venv, not the e2i_api container; the
live API cert after deploy is the faithful measurement.

Run from the lane worktree:
  cd .worktrees/real-data-causal && $PY docs/demos/results/2026-09-22_optum_biologic_persistence_cert/preflight_agent.py <outcome> [estimator]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import resource
import sys
import time
import uuid
from pathlib import Path

# WRITE-FREE GUARD -- set BEFORE any ``src`` import. This process is not the
# production API, but the graph it runs IS production code: the refutation node
# persists its suite to ``causal_validations`` / ``validation_outcomes`` and the
# tracker logs a run to MLflow. The worktree sits inside the main checkout, so a
# lazy ``load_dotenv()`` (``src/ml/data_loader.py``) walks up to the main
# checkout's ``.env`` and finds the PROD Supabase URL + service-role key; on
# 2026-09-22 07:24 the first pre-flight that reached the refutation node wrote
# 6 + 1 rows to prod and one MLflow run (see preflight.md). ``load_dotenv``
# never overrides a variable that already exists, so blanking these here keeps
# the run on the fail-closed "persistence unavailable" path (a WARNING in the
# log, no rows) and MLflow on a scratch file store.
for _var in (
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_ANON_KEY",
):
    os.environ[_var] = ""
os.environ["MLFLOW_TRACKING_URI"] = (
    "file:///tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/3c437b64-e40a-4bb9-ba13-d7b7674bd2ab/scratchpad/preflight_mlruns"
)

import pandas as pd

# Node-level INFO logs carry the per-stage timings (graph_builder / estimation /
# refutation) that a timed-out run's failed response discards.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "httpcore", "urllib3", "opik", "sentry_sdk"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

sys.path.insert(0, str(Path.cwd()))
import src  # noqa: E402

assert ".worktrees/real-data-causal" in src.__file__, src.__file__

from src.api.routes.causal import agent as causal_routes  # noqa: E402
from src.api.routes.causal.datasets import (  # noqa: E402
    _CAUSAL_CATEGORICAL_COLUMNS,
    _CAUSAL_DATASET_SPECS,
    _CAUSAL_NUMERIC_COLUMNS,
)
from src.api.routes.causal.loaders import (  # noqa: E402
    _coerce_estimation_row,
    _one_hot_categoricals,
)
from src.api.schemas.causal import AgentCausalAnalysisRequest  # noqa: E402

DATASET = "optum_biologic_persistence"
TREATMENT = "treatment_dupixent"
PARQUET = Path(
    "/home/enunez/Projects/e2i_causal_analytics/data/rwd/mart/persistence_causal/"
    "e2i_causal_v1_biologic_persistence.parquet"
)
OUT = Path(__file__).resolve().parent


class _MemStore:
    def __init__(self) -> None:
        self.d: dict = {}

    async def get(self, key):
        return self.d.get(key)

    async def set(self, key, value):
        self.d[key] = value


def build_frame(outcome: str):
    spec = _CAUSAL_DATASET_SPECS[DATASET]
    covariates = list(spec["covariate"])
    select_cols = [TREATMENT, outcome, *covariates]
    raw = pd.read_parquet(PARQUET, columns=select_cols)
    raw = raw.astype(object).where(raw.notna(), None)
    records = []
    for row in raw.to_dict(orient="records"):
        rec = _coerce_estimation_row(
            row,
            select_cols=select_cols,
            treatment_var=TREATMENT,
            outcome_var=outcome,
            numeric_cols=_CAUSAL_NUMERIC_COLUMNS[DATASET],
            categorical_cols=frozenset(_CAUSAL_CATEGORICAL_COLUMNS[DATASET]),
        )
        if rec is not None:
            records.append(rec)
    frame = pd.DataFrame(records)
    cats = [c for c in select_cols if c in _CAUSAL_CATEGORICAL_COLUMNS[DATASET]]
    frame, dummies = _one_hot_categoricals(frame, cats)
    resolved = [c for c in select_cols if c not in (TREATMENT, outcome) and c not in cats] + dummies
    return frame, resolved


async def run(outcome: str, estimator: str | None) -> dict:
    store = _MemStore()
    causal_routes._agent_analysis_store = store
    frame, covariates = build_frame(outcome)
    req = AgentCausalAnalysisRequest(
        treatment_var=TREATMENT,
        outcome_var=outcome,
        dataset=DATASET,
        limit=20000,
        estimator=estimator,
    )
    req = req.model_copy(update={"auto_discover": False})
    aid = str(uuid.uuid4())
    t0 = time.monotonic()
    await causal_routes._run_agent_analysis_task(aid, req, frame, covariates, "database")
    wall = time.monotonic() - t0
    result = store.d[aid]
    payload = result.model_dump(mode="json")
    payload["_preflight"] = {
        "wall_s": round(wall, 1),
        "n_rows": int(frame.shape[0]),
        "n_covariates_resolved": len(covariates),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "estimator_forced": estimator,
        "auto_discover": False,
    }
    return payload


if __name__ == "__main__":
    outcome = sys.argv[1]
    estimator = sys.argv[2] if len(sys.argv) > 2 else None
    payload = asyncio.run(run(outcome, estimator))
    path = OUT / f"preflight_{outcome}{'_' + estimator if estimator else ''}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    p = payload["_preflight"]
    print(
        f"{outcome}: status={payload['status']} wall={p['wall_s']}s rss={p['max_rss_mb']}MB "
        f"n={p['n_rows']} k={p['n_covariates_resolved']} "
        f"ate={payload.get('ate')} ci=[{payload.get('ate_ci_lower')}, {payload.get('ate_ci_upper')}] "
        f"p={payload.get('p_value')} estimator={payload.get('selected_estimator')} "
        f"dag_source={payload.get('dag_source')} refutation_passed={payload['refutation'].get('passed')}"
    )
