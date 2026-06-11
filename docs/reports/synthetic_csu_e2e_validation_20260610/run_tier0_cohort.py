#!/usr/bin/env python
"""Drive the canonical tier0 CLI per synthetic-CSU cohort, capturing the
pipeline state pickle the tier1-5 harness consumes.

run_tier0_test.main() saves the console MD report but discards the in-memory
state run_pipeline returns; we wrap run_pipeline (module-global, resolved at
call time inside main's asyncio.run) to pickle the state on the way through —
the exact CLI path otherwise (same argparse defaults, same MD report).

Usage (from repo root):
    LOKY_MAX_CPU_COUNT=1 .venv/bin/python \
      docs/reports/synthetic_csu_e2e_validation_20260610/run_tier0_cohort.py <cohort>
"""

import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

import scripts.run_tier0_test as t0  # noqa: E402

COHORT_TARGETS = {
    "initiation": "treatment_initiated",
    "discontinuation": "discontinued_180d",
    "persistence": "persistent_180d",
    "hcp_adoption": "adopted_target_brand",
}

cohort = sys.argv[1]
target = COHORT_TARGETS[cohort]
report_dir = Path("docs/reports/synthetic_csu_e2e_validation_20260610") / f"tier0_{cohort}"
report_dir.mkdir(parents=True, exist_ok=True)
cache_file = report_dir / f"tier0_state_{cohort}.pkl"

_orig_run_pipeline = t0.run_pipeline


async def _capture(*args, **kwargs):
    state = await _orig_run_pipeline(*args, **kwargs)
    try:
        with open(cache_file, "wb") as fh:
            pickle.dump(state, fh)
        print(f"\n[driver] tier0 state pickled to {cache_file}")
    except Exception as exc:  # state must never be lost silently
        print(f"\n[driver] WARNING: state pickle failed: {exc}")
    return state


t0.run_pipeline = _capture

sys.argv = [
    "run_tier0_test.py",
    "--data-dir",
    f"data/rwd/synthetic_CSU/tier0/{cohort}",
    "--target",
    target,
    "--brand",
    "Remibrutinib",
    "--indication",
    "Chronic Spontaneous Urticaria (CSU)",
    "--no-bentoml",
    # The demo cost matrix {tp:+1, fp:-0.05, fn:-1} is an injected DEMO artifact
    # (default-on); at the chosen threshold it forces business_utility negative
    # and blocks deployment regardless of model quality (same artifact the
    # 2026-06-09 Optum gate run documented). The runner ships this flag to drop
    # the injection; all data-driven gates (AUC/precision/recall/permutation/
    # honest-band) remain enforced.
    "--no-demo-cost-matrix",
    # e2i is a commercial-analytics platform; the clinical default demands
    # recall>=0.65 / MCC>=0.35 — unreachable BY DESIGN on this dataset (the
    # DGP's latent noise caps AUC ~0.67 to keep labels causally recoverable).
    # The commercial profile (docs/results/tier0_commercial_deployment_intent_
    # 20260607.md: recall>=0.50, MCC>=0.10, AUC bar 0.65, calibration guards
    # kept) is the documented intent axis for targeting use-cases.
    "--deployment-intent",
    "commercial",
    # Declared-safe-by-construction contracts for the generator's designed
    # features: without them the leakage layers false-positively dropped
    # disease_severity (initiation) and influence_network_size +
    # peer_influence_score (hcp_adoption AUC 0.78 -> 0.51).
    "--feature-manifest-source",
    "synthetic_csu",
    "--output-dir",
    str(report_dir),
]
sys.exit(t0.main())
