"""Read-only LIVE probe of the unit-outcome feed after --refresh-ab (2026-09-23).

Runs the REAL ExperimentOutcomeRepository.load_arrays against the live DB for every
synthetic experiment (mirroring src/tasks/ab_testing_tasks.py:292: prediction_target as
primary_metric, brand passed), and checks (1) last_outcome_source == 'unit_outcomes',
(2) mean(t) - mean(c) equals the stored ab_experiment_results.effect_estimate, (3) the array
sizes equal the stored control_n / treatment_n.  No writes."""
import asyncio
import os
import sys
from uuid import UUID

import numpy as np

sys.path.insert(0, "/home/enunez/Projects/e2i_causal_analytics")
import logging

logging.disable(logging.INFO)
from src.repositories.experiment_outcome import ExperimentOutcomeRepository  # noqa: E402
from src.repositories import get_supabase_client  # noqa: E402

client = get_supabase_client()
exps = (
    client.table("ml_experiments")
    .select("id,brand,prediction_target")
    .eq("is_synthetic", True)
    .range(0, 999)
    .execute()
    .data
)
res = (
    client.table("ab_experiment_results")
    .select("experiment_id,primary_metric,effect_estimate,control_n,treatment_n")
    .eq("is_synthetic", True)
    .range(0, 999)
    .execute()
    .data
)
res_by = {r["experiment_id"]: r for r in res}
print(f"experiments {len(exps)}, results {len(res)}")
assert len(exps) == 360 and len(res) == 360


async def main() -> int:
    repo = ExperimentOutcomeRepository(supabase_client=client)
    sources, eq, n_ok, bad = {}, 0, 0, []
    for e in exps:
        r = res_by[e["id"]]
        c, t = await repo.load_arrays(
            UUID(e["id"]), e["prediction_target"] or "", brand=e.get("brand"), include_synthetic=True
        )
        src = repo.last_outcome_source
        sources[src] = sources.get(src, 0) + 1
        eff = float(np.mean(t) - np.mean(c)) if len(c) and len(t) else float("nan")
        d = abs(eff - float(r["effect_estimate"]))
        if d < 1e-9:
            eq += 1
        else:
            bad.append((e["id"], eff, r["effect_estimate"], len(c), len(t)))
        if len(c) == int(r["control_n"]) and len(t) == int(r["treatment_n"]):
            n_ok += 1
    print(f"outcome_source counts: {sources}")
    print(f"effect equal within 1e-9: {eq}/{len(exps)}; sizes equal stored n: {n_ok}/{len(exps)}")
    for b in bad[:5]:
        print("  MISMATCH", b)
    ok = sources.get("unit_outcomes") == len(exps) and eq == len(exps) and n_ok == len(exps)
    print("VERDICT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


sys.exit(asyncio.run(main()))
