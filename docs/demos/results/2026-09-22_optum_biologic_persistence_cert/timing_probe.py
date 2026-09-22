#!/usr/bin/env python3
"""Lane A budget attribution: time the production pieces the agent runs on the
real causal frame, standalone, so the 900 s timeouts of the full-graph
pre-flights can be attributed to a stage (cheapest disproof before touching the
run shape).

Pieces (same classes the estimation / refutation nodes call):
  1. econml LinearDML fit with the production RF nuisances (nuisance_config)
  2. the production LinearDMLWrapper.fit (adds the honest CI + LogisticRegressionCV propensity)
  3. the refutation node's DoWhy reconstruction (_build_dowhy_estimate)

Run from the lane worktree:
  $PY docs/demos/results/2026-09-22_optum_biologic_persistence_cert/timing_probe.py <outcome>
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
import src  # noqa: E402

assert ".worktrees/real-data-causal" in src.__file__, src.__file__

from preflight_agent import DATASET, TREATMENT, build_frame  # noqa: E402

OUT = Path(__file__).resolve().parent


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def main(outcome: str) -> None:
    timings: dict = {"dataset": DATASET, "outcome": outcome}
    frame, covariates = build_frame(outcome)
    T = frame[TREATMENT].astype(int).to_numpy()
    Y = frame[outcome].astype(float).to_numpy()
    X = frame[covariates]
    timings["n_rows"] = int(len(frame))
    timings["n_covariates"] = len(covariates)
    print(f"frame n={len(frame)} k={len(covariates)} rss={_rss_mb()}MB", flush=True)

    # 1. bare econml LinearDML with the production nuisances
    from econml.dml import LinearDML

    from src.causal_engine.nuisance_config import linear_dml_model_t, linear_dml_model_y

    t0 = time.perf_counter()
    model = LinearDML(
        model_y=linear_dml_model_y(),
        model_t=linear_dml_model_t(),
        discrete_treatment=True,
        random_state=42,
    )
    Xv = X.to_numpy()
    model.fit(Y, T, X=Xv, W=Xv)
    ate_bare = float(np.mean(model.effect(Xv)))
    timings["econml_lineardml_fit_s"] = round(time.perf_counter() - t0, 1)
    timings["econml_lineardml_ate"] = ate_bare
    print(
        f"1. econml LinearDML fit: {timings['econml_lineardml_fit_s']} s ate={ate_bare:.4f} rss={_rss_mb()}MB",
        flush=True,
    )

    # 2. the production wrapper (fit + honest CI + LogisticRegressionCV propensity)
    from src.causal_engine.energy_score.estimator_selector import (
        EstimatorConfig,
        EstimatorType,
        LinearDMLWrapper,
    )

    t0 = time.perf_counter()
    res = LinearDMLWrapper(EstimatorConfig(estimator_type=EstimatorType.LINEAR_DML)).fit(T, Y, X)
    timings["wrapper_fit_s"] = round(time.perf_counter() - t0, 1)
    timings["wrapper"] = {
        "success": res.success,
        "ate": res.ate,
        "ate_ci": [res.ate_ci_lower, res.ate_ci_upper],
        "ate_std": res.ate_std,
        "estimation_time_ms": res.estimation_time_ms,
    }
    print(
        f"2. LinearDMLWrapper.fit: {timings['wrapper_fit_s']} s {timings['wrapper']} rss={_rss_mb()}MB",
        flush=True,
    )

    # 3. the refutation node's DoWhy reconstruction
    from src.agents.causal_impact.nodes.refutation import _build_dowhy_estimate

    t0 = time.perf_counter()
    try:
        _model, _estimand, estimate, dowhy_method = _build_dowhy_estimate(
            data=frame,
            treatment=TREATMENT,
            outcome=outcome,
            common_causes=list(covariates),
            estimation_result={
                "selected_estimator": "LinearDML",
                "method": "LinearDML",
                "ate": res.ate,
            },
        )
        timings["reconstruction_s"] = round(time.perf_counter() - t0, 1)
        timings["reconstruction"] = {"dowhy_method": dowhy_method, "ate": float(estimate.value)}
    except Exception as exc:  # noqa: BLE001 — the point is to record what happened
        timings["reconstruction_s"] = round(time.perf_counter() - t0, 1)
        timings["reconstruction"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    print(
        f"3. DoWhy reconstruction: {timings['reconstruction_s']} s {timings['reconstruction']} rss={_rss_mb()}MB",
        flush=True,
    )

    timings["max_rss_mb"] = _rss_mb()
    (OUT / f"timing_probe_{outcome}.json").write_text(json.dumps(timings, indent=2, default=str))
    print(json.dumps(timings, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
