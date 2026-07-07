#!/usr/bin/env python
"""Targeted, idempotent apply of the commercial-KPI causal_paths grain.

The synthetic substrate is FROZEN (full reseed = disaster recovery only), so
this script upserts ONLY the commercial grain — content-addressed path_ids and
values (see CausalPathsGenerator._COMMERCIAL_EDGES), so re-runs are no-ops
apart from the freshness stamps. A future disaster-recovery reseed emits the
same rows from the same generator and upserts over them harmlessly.

Run inside the API container (it has the Supabase env):

    docker exec e2i_api python scripts/seed_commercial_causal_paths.py [--dry-run]

Verifies through the REAL chat read path (search_paths_for_outcome) afterward.
"""

import argparse
import asyncio

from src.ml.synthetic.generators.causal_paths_generator import commercial_rows_for_upsert


async def main(dry_run: bool) -> None:
    records = commercial_rows_for_upsert()
    ids = [r["path_id"] for r in records]
    print(f"{len(records)} commercial causal-path rows (content-addressed, scp_c*)")
    if dry_run:
        for r in records[:3]:
            print(
                f"  {r['path_id']}: {r['start_node']} -> {r['end_node']} "
                f"({r['brand']}, effect {r['causal_effect_size']:+.4f})"
            )
        print("  ... dry run, nothing written")
        return

    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.causal_path import CausalPathRepository

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

    # Faithful verification: the exact read path causal_analysis_tool uses.
    repo = CausalPathRepository(client)
    for term in ("TRx", "NRx", "NBRx", "TRx Share", "ROI"):
        hits = await repo.search_paths_for_outcome(
            term, min_confidence=0.7, limit=15, include_synthetic=True
        )
        print(f"verify: {term!r} -> {len(hits)} chains via chat read path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print rows, write nothing")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
