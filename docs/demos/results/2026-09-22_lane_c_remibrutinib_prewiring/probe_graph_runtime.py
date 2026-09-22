#!/usr/bin/env python3
"""Lane C disproof D2: is the full causal_impact graph CI-runnable on MAIN for a
planted csu_escalation_causal frame? Measures wall time at (a) planted-confounder
width and (b) the full 64-feature contract width (after one-hot).

Run from the lane worktree root:
  cd .worktrees/lane-c-remibrutinib-prewiring && \
    LANE_C_PROBE_OUT=<scratch dir> $PY <this file> <n> <narrow|wide> [estimator]
Writes <scratch dir>/lane_c_probe_<width>_<n>[_<estimator>].json and prints one PROBE line.
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

# WRITE-FREE GUARD -- before any ``src`` import (prod-write trap, Lane A 07:24).
for _var in (
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_ANON_KEY",
):
    os.environ[_var] = ""
# Output + MLflow file store: LANE_C_PROBE_OUT (a scratch dir; never the repo).
SCRATCH = Path(os.environ.get("LANE_C_PROBE_OUT", "/tmp/lane_c_probe_out")).resolve()
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = f"file://{SCRATCH}/lane_c_mlruns"
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "httpcore", "urllib3", "opik", "sentry_sdk"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

sys.path.insert(0, str(Path.cwd()))
import src  # noqa: E402

assert ".worktrees/lane-c-remibrutinib-prewiring" in src.__file__, src.__file__

from src.api.routes.causal import agent as causal_routes  # noqa: E402
from src.api.routes.causal.loaders import _one_hot_categoricals  # noqa: E402
from src.api.schemas.causal import AgentCausalAnalysisRequest  # noqa: E402
from src.data.manifests import MART_SAFE_FEATURES  # noqa: E402
from src.ml.synthetic.dgp.treatment_arm import (  # noqa: E402
    ArmSpec,
    assign_arm_from_spec,
    binary_outcome_rd,
)

TREATMENT = "treatment_remibrutinib"
OUTCOME = "persistent_at_180d_g28"
CATEGORICALS = [
    "gdr_cd",
    "payer_category",
    "payer_product",
    "payer_bus",
    "charlson_risk_band",
    "elixhauser_risk_band",
    "geographic_region",
]


class _MemStore:
    def __init__(self) -> None:
        self.d: dict = {}

    async def get(self, key):
        return self.d.get(key)

    async def set(self, key, value):
        self.d[key] = value


def plant(n: int, seed: int):
    rng = np.random.default_rng(seed)
    cov: dict = {}
    cov["age_at_index"] = np.clip(rng.normal(48, 15, n), 18, 90).round()
    cov["charlson_score"] = rng.poisson(0.8, n)
    payer = rng.choice(
        ["commercial", "medicare", "medicaid", "other"], n, p=[0.55, 0.25, 0.15, 0.05]
    )
    cov["is_commercial"] = (payer == "commercial").astype(float)
    spec = ArmSpec(
        name=TREATMENT,
        confounders={"age_at_index": -0.03, "charlson_score": -0.35, "is_commercial": 0.9},
        intercept=-0.6,
        cate_by_segment={},
        target_outcomes=(OUTCOME,),
        center={"age_at_index": 48.0},
    )
    arm, prop = assign_arm_from_spec(spec, cov, rng)
    seg = np.where(
        cov["age_at_index"] > 60,
        "high_severity",
        np.where(cov["age_at_index"] > 40, "medium_severity", "low_severity"),
    )
    baseline = (
        -0.02 * (cov["age_at_index"] - 48)
        - 0.25 * cov["charlson_score"]
        + 0.4 * cov["is_commercial"]
    )
    y, tau = binary_outcome_rd(
        arm,
        baseline,
        seg,
        {"high_severity": 0.9, "medium_severity": 0.7, "low_severity": 0.5},
        rng,
        target_prevalence=0.45,
    )
    df = pd.DataFrame({TREATMENT: arm.astype(float), OUTCOME: y.astype(float)})
    for c in MART_SAFE_FEATURES:
        if c in ("age_at_index", "charlson_score"):
            df[c] = cov[c].astype(float)
        elif c == "payer_category":
            df[c] = payer
        elif c == "gdr_cd":
            df[c] = rng.choice(["F", "M"], n, p=[0.67, 0.33])
        elif c == "payer_product":
            df[c] = rng.choice(["HMO", "PPO", "POS", "EPO"], n)
        elif c == "payer_bus":
            df[c] = rng.choice(["COM", "MCR", "MCD"], n)
        elif c == "charlson_risk_band":
            df[c] = np.where(
                cov["charlson_score"] >= 3,
                "high",
                np.where(cov["charlson_score"] >= 1, "medium", "low"),
            )
        elif c == "elixhauser_risk_band":
            df[c] = rng.choice(["low", "medium", "high"], n, p=[0.6, 0.3, 0.1])
        elif c == "geographic_region":
            df[c] = rng.choice(["midwest", "south", "northeast", "west"], n)
        elif c == "enrollment_duration_days":
            df[c] = rng.integers(365, 3650, n).astype(float)
        elif c in (
            "elixhauser_van_walraven_score",
            "comorbidity_diag_distinct_count",
            "comorbidity_diag_claim_count",
        ):
            df[c] = rng.poisson(2.0, n).astype(float)
        else:
            df[c] = rng.binomial(1, 0.08, n).astype(float)
    return df, float(np.mean(tau))


async def run(n: int, width: str, estimator: str | None) -> dict:
    store = _MemStore()
    causal_routes._agent_analysis_store = store
    df, true_ate = plant(n, 7)
    if width == "narrow":
        keep = [TREATMENT, OUTCOME, "age_at_index", "charlson_score", "payer_category"]
        df = df[keep]
        frame, dummies = _one_hot_categoricals(df, ["payer_category"])
    else:
        frame, dummies = _one_hot_categoricals(df, CATEGORICALS)
    covariates = [c for c in frame.columns if c not in (TREATMENT, OUTCOME)]
    req = AgentCausalAnalysisRequest(
        treatment_var=TREATMENT,
        outcome_var=OUTCOME,
        dataset="patient_journeys",  # the task only reads dataset for negative-control lookup
        limit=20000,
        estimator=estimator,
    )
    req = req.model_copy(update={"auto_discover": False})
    aid = str(uuid.uuid4())
    t0 = time.monotonic()
    await causal_routes._run_agent_analysis_task(aid, req, frame, covariates, "synthetic")
    wall = time.monotonic() - t0
    result = store.d[aid]
    payload = result.model_dump(mode="json")
    payload["_probe"] = {
        "wall_s": round(wall, 1),
        "n_rows": int(frame.shape[0]),
        "k": len(covariates),
        "true_ate": round(true_ate, 4),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "estimator_forced": estimator,
        "width": width,
    }
    return payload


if __name__ == "__main__":
    n = int(sys.argv[1])
    width = sys.argv[2]
    estimator = sys.argv[3] if len(sys.argv) > 3 else None
    payload = asyncio.run(run(n, width, estimator))
    out = SCRATCH / f"lane_c_probe_{width}_{n}{'_' + estimator if estimator else ''}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    p = payload["_probe"]
    print(
        f"PROBE {width} n={p['n_rows']} k={p['k']} wall={p['wall_s']}s rss={p['max_rss_mb']}MB "
        f"status={payload['status']} ate={payload.get('ate')} true={p['true_ate']} "
        f"ci=[{payload.get('ate_ci_lower')}, {payload.get('ate_ci_upper')}] "
        f"estimator={payload.get('selected_estimator')} dag_source={payload.get('dag_source')} "
        f"refutation_passed={payload['refutation'].get('passed')} warnings={payload.get('warnings')}"
    )
