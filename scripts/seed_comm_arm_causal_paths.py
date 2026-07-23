#!/usr/bin/env python
"""Targeted, idempotent apply of the patient-grain commercial-arm causal_paths edges.

The synthetic substrate is FROZEN (full reseed = disaster recovery only), and the
patient_journeys arm data (copay_support / psp_enrolled / rep_detailing_high /
sample_dropped / trigger_accepted — COMM-ARMS Phases 1-4) ALREADY exists. This
script only adds the ``causal_paths`` edges that make those planted levers appear
on the discovery leaderboard: content-addressed path_ids and values (see
CausalPathsGenerator._COMM_ARM_EDGES), so re-runs are no-ops apart from the
freshness stamps, and a future disaster-recovery reseed emits the same rows and
upserts over them harmlessly.

Run inside the API container (it has the Supabase env):

    docker exec e2i_api python scripts/seed_comm_arm_causal_paths.py [--dry-run]

After applying, sync the KG so the arms appear as chains too:

    docker exec e2i_api python scripts/sync_causal_paths_to_falkordb.py --execute
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the repo root to sys.path so ``python scripts/seed_comm_arm_causal_paths.py``
# can import ``src``. Run that way, sys.path[0] is scripts/ (not the repo root), so
# the ``from src...`` import below would fail with ModuleNotFoundError; the app
# server only resolves ``src`` because gunicorn runs with cwd=/app on the path.
# Mirrors the bootstrap in sibling scripts (e.g. sample_ml_pipeline.py); harmless
# when run as a module or with PYTHONPATH already set.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.synthetic.generators.causal_paths_generator import comm_arm_rows_for_upsert


async def main(dry_run: bool) -> None:
    records = comm_arm_rows_for_upsert()
    ids = [r["path_id"] for r in records]
    print(f"{len(records)} patient-arm causal-path rows (content-addressed, scp_a*)")
    if dry_run:
        for r in records:
            print(
                f"  {r['path_id']}: {r['start_node']} -> {r['end_node']} "
                f"({r['brand']}, effect {r['causal_effect_size']:+.4f}, "
                f"confounders {r['confounders_controlled']})"
            )
        print("  ... dry run, nothing written")
        return

    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    before = (
        await client.table("causal_paths")
        .select("path_id", count="exact")
        .in_("path_id", ids)
        .limit(1)
        .execute()
    )
    await client.table("causal_paths").upsert(records, on_conflict="path_id").execute()
    after = (
        await client.table("causal_paths")
        .select("path_id", count="exact")
        .in_("path_id", ids)
        .limit(1)
        .execute()
    )
    print(f"upserted: {before.count} of these ids existed before, {after.count} exist now")

    # Faithful verification: confirm each arm now surfaces as a distinct leaderboard
    # question (the exact read the discovery route enumerates from).
    from src.repositories.causal_path import CausalPathRepository

    repo = CausalPathRepository(client)
    questions = await repo.get_distinct_questions(brand=None, include_synthetic=True)
    arms = {
        "copay_support",
        "psp_enrolled",
        "rep_detailing_high",
        "sample_dropped",
        "trigger_accepted",
    }
    surfaced = sorted({(q["treatment"], q["outcome"]) for q in questions if q["treatment"] in arms})
    print(f"verify: {len(surfaced)} arm (treatment -> outcome) questions now enumerable:")
    for t, o in surfaced:
        print(f"  {t} -> {o}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print rows, write nothing")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
