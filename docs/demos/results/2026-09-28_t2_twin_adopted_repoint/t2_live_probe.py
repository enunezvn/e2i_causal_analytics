"""Lane T2 live check — READ-ONLY, in-process, through the lane's OWN code paths.

What it runs, against live prod Supabase, with this worktree's ``src`` (asserted):

  1. ``cohort_loader.load_cohort_frame`` per brand — the two-read merge (collapsed
     business_metrics rollups LEFT-joined to paged hcp_brand_adoption.adopted). Timed.
  2. ``cohort_loader.cohort_treatment_availability`` per brand — the new joined-coverage
     availability gate (one merged read for all eight channels). Timed, to state its cost.
  3. For 3 brands x 5 interventions: ``assess_cohort_frame`` -> provider ->
     ``CohortEffectDataProvider.get_training_frame`` -> ``estimate_cohort_effect`` exactly as
     ``CohortCausalEstimator`` calls it. Records the DR ATE, its SE, the DR CI and whether it
     excludes 0. Fits run SEQUENTIALLY (~0.7 GiB each).
  4. One region-targeted estimate (the ``target_regions`` DR subset path) per brand.

READ-ONLY: every DB call is a ``.select()`` issued by the loader. No Redis, no MLflow, no writes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import resource
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

WORKTREE = "/home/enunez/Projects/e2i_causal_analytics/.worktrees/twin-adopted-repoint"
sys.path.insert(0, WORKTREE)
logging.disable(logging.INFO)
warnings.filterwarnings("ignore")

import src  # noqa: E402

assert src.__file__.startswith(WORKTREE), f"WRONG TREE: {src.__file__}"

from src.digital_twin.effect import cohort_loader  # noqa: E402
from src.digital_twin.effect.cohort_causal_estimator import estimate_cohort_effect  # noqa: E402
from src.digital_twin.effect.provider import COHORT_MIN_ROWS  # noqa: E402

BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")
INTERVENTIONS = (
    "digital_engagement",
    "speaker_program_invitation",
    "peer_influence_activation",
    "patient_support_program",
    "rep_training_quality",  # the planted NULL
)
NULL = "rep_training_quality"
TARGET_REGION = "northeast"
OUT = Path(WORKTREE) / "docs/demos/results/2026-09-28_t2_twin_adopted_repoint"


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def say(*a: Any) -> None:
    print(*a, flush=True)


async def load_all() -> Dict[str, Any]:
    from src.memory.services.factories import loop_scoped_async_supabase_client

    out: Dict[str, Any] = {"frames": {}, "load": {}, "availability": {}}
    async with loop_scoped_async_supabase_client() as client:
        for brand in BRANDS:
            t0 = time.perf_counter()
            frame = await cohort_loader.load_cohort_frame(client, brand)
            load_s = time.perf_counter() - t0
            joined = int(frame["adopted"].notna().sum()) if "adopted" in frame else 0
            out["frames"][brand] = frame
            out["load"][brand] = {
                "seconds": round(load_s, 2),
                "pairs": int(len(frame)),
                "joined_to_adopted": joined,
                "unjoined_pairs": int(len(frame) - joined),
                "adopted_rate": round(float(frame["adopted"].mean()), 4) if joined else None,
                "has_max_metric_date": "max_metric_date" in frame.columns,
                "columns": sorted(frame.columns.tolist()),
            }
            say(
                f"LOAD  {brand:<13} {load_s:5.2f}s pairs={len(frame)} joined={joined} "
                f"unjoined={len(frame) - joined}"
            )
            t0 = time.perf_counter()
            avail = await cohort_loader.cohort_treatment_availability(client, brand)
            avail_s = time.perf_counter() - t0
            out["availability"][brand] = {
                "seconds": round(avail_s, 2),
                "n_probe_errors": avail.n_probe_errors,
                "available": dict(avail),
            }
            say(
                f"AVAIL {brand:<13} {avail_s:5.2f}s errors={avail.n_probe_errors} "
                f"available={sum(avail.values())}/{len(avail)}"
            )
    return out


def fit_all(frames: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for brand, frame in frames.items():
        for intervention in INTERVENTIONS:
            rec: Dict[str, Any] = {"brand": brand, "intervention": intervention}
            usability = cohort_loader.assess_cohort_frame(frame, intervention)
            if usability.provider is None:
                rec["refused"] = str(usability.cause)
                rec["details"] = dict(usability.details)
                rows.append(rec)
                say(f"FIT   {brand:<13} {intervention:<27} REFUSED {usability.cause}")
                continue
            tf = usability.provider.get_training_frame(intervention, brand=brand, twin_type="hcp")
            t0 = time.perf_counter()
            eff = estimate_cohort_effect(
                tf.df,
                tf.treatment_var,
                outcome_col=tf.outcome_var,
                confounders=tuple(tf.confounders),
            )
            rec.update(
                {
                    "treatment": tf.treatment_var,
                    "outcome": tf.outcome_var,
                    "n": eff.n,
                    "ate": eff.ate,
                    "ate_stderr": eff.ate_stderr,
                    "ci_lower": eff.ate_ci_lower,
                    "ci_upper": eff.ate_ci_upper,
                    "ci_width": eff.ci_width(),
                    "excludes_0": bool(eff.ate_ci_lower > 0 or eff.ate_ci_upper < 0),
                    "interval_method": eff.interval_method,
                    "cate_by_region": eff.cate_by_region,
                    "seconds": round(time.perf_counter() - t0, 2),
                    "peak_rss_gib": round(rss_gib(), 3),
                }
            )
            rows.append(rec)
            say(
                f"FIT   {brand:<13} {intervention:<27} n={eff.n:<5} ate={eff.ate:+.4f} "
                f"se={eff.ate_stderr:.4f} CI=({eff.ate_ci_lower:+.4f},{eff.ate_ci_upper:+.4f}) "
                f"excl0={rec['excludes_0']!s:<5} {rec['seconds']}s rss={rec['peak_rss_gib']:.2f}G"
            )
        # The DR subset path, once per brand.
        tf = cohort_loader.assess_cohort_frame(frame, "digital_engagement").provider
        if tf is not None:
            training = tf.get_training_frame("digital_engagement", brand=brand, twin_type="hcp")
            eff = estimate_cohort_effect(
                training.df,
                training.treatment_var,
                outcome_col=training.outcome_var,
                confounders=tuple(training.confounders),
                target_regions=[TARGET_REGION],
            )
            rows.append(
                {
                    "brand": brand,
                    "intervention": "digital_engagement",
                    "target_regions": [TARGET_REGION],
                    "target_n": eff.target_n,
                    "target_ate": eff.target_ate,
                    "target_stderr": eff.target_stderr,
                    "target_ci_lower": eff.target_ci_lower,
                    "target_ci_upper": eff.target_ci_upper,
                    "declared_region_cate": eff.cate_by_region.get(TARGET_REGION),
                }
            )
            say(
                f"TGT   {brand:<13} digital_engagement@{TARGET_REGION} n={eff.target_n} "
                f"ate={eff.target_ate:+.4f} CI=({eff.target_ci_lower:+.4f},"
                f"{eff.target_ci_upper:+.4f}) region_cate={eff.cate_by_region.get(TARGET_REGION):+.4f}"
            )
    return rows


def main() -> int:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    say(f"started {started}  src={src.__file__}")
    loaded = asyncio.run(load_all())
    fits = fit_all(loaded.pop("frames"))

    cohort = [r for r in fits if "excludes_0" in r]
    planted = [r for r in cohort if r["intervention"] != NULL]
    nulls = [r for r in cohort if r["intervention"] == NULL]
    verdict = {
        "planted_exclude_0": f"{sum(r['excludes_0'] for r in planted)}/{len(planted)}",
        "null_covers_0": f"{sum(not r['excludes_0'] for r in nulls)}/{len(nulls)}",
        "null_max_abs_ate": max((abs(r["ate"]) for r in nulls), default=None),
        "rows_meet_min": all(
            v["joined_to_adopted"] >= COHORT_MIN_ROWS for v in loaded["load"].values()
        ),
        "refusals": [r for r in fits if "refused" in r],
    }
    say("\n===== VERDICT =====")
    for k, v in verdict.items():
        say(f"  {k}: {v}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "live_probe.json").write_text(
        json.dumps(
            {"started_at": started, **loaded, "fits": fits, "verdict": verdict},
            indent=2,
            default=str,
        )
    )
    say(f"wrote {OUT / 'live_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
