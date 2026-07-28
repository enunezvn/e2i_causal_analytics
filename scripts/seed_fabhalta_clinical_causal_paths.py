#!/usr/bin/env python
"""Targeted, idempotent apply of the Fabhalta-ONLY clinical-axis causal_paths edge.

The synthetic substrate is FROZEN (full reseed = disaster recovery only). The
patient_journeys ``complement_inhibitor_status`` column already exists (populated
"current"/"prior" for Fabhalta rows, NULL off-brand — BRAND_ELIGIBILITY_FIELDS).
This script only adds the ``causal_paths`` edge that makes the planted prior-C5
persistence effect appear on Fabhalta's discovery leaderboard AND (after the
FalkorDB sync) as a node on Fabhalta's /knowledge-graph — the FIRST brand-DISTINCT
causal variable in the gold standard (issue #1321 pilot).

Content-addressed path_id + values (see CausalPathsGenerator._FABHALTA_CLINICAL_EDGES,
namespace scp_f*), so re-runs are no-ops apart from the freshness stamps, and a
future disaster-recovery reseed emits the same row and upserts over it harmlessly.
Emitted for Fabhalta ONLY — Kisqali/Remibrutinib have no C5-inhibitor axis, so
their KG never gains this node.

Run inside the API container (it has the Supabase env):

    docker exec e2i_api python scripts/seed_fabhalta_clinical_causal_paths.py [--dry-run]

After applying, sync the KG so the node appears on Fabhalta's graph too:

    docker exec e2i_api python scripts/sync_causal_paths_to_falkordb.py --execute
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the repo root to sys.path so ``python scripts/seed_fabhalta_clinical_causal_paths.py``
# resolves ``src`` (sys.path[0] is scripts/, not the repo root). Mirrors the
# bootstrap in the sibling seed scripts; harmless as a module or with PYTHONPATH set.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.synthetic.generators.causal_paths_generator import fabhalta_clinical_rows_for_upsert


async def main(dry_run: bool) -> None:
    records = fabhalta_clinical_rows_for_upsert()
    ids = [r["path_id"] for r in records]
    print(f"{len(records)} Fabhalta clinical causal-path row(s) (content-addressed, scp_f*)")
    for r in records:
        print(
            f"  {r['path_id']}: {r['start_node']} -> {r['end_node']} "
            f"({r['brand']}, effect {r['causal_effect_size']:+.4f}, "
            f"confounders {r['confounders_controlled']})"
        )
    if dry_run:
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

    # Faithful verification: the edge must surface as a Fabhalta leaderboard
    # question AND must NOT surface for the other brands (brand-distinct guarantee).
    from src.repositories.causal_path import CausalPathRepository

    repo = CausalPathRepository(client)
    for brand in ("Fabhalta", "Kisqali", "Remibrutinib"):
        qs = await repo.get_distinct_questions(brand=brand, include_synthetic=True)
        hit = [
            (q["treatment"], q["outcome"])
            for q in qs
            if q["treatment"] == "complement_inhibitor_status"
        ]
        marker = "OK (present)" if (brand == "Fabhalta") == bool(hit) else "!! UNEXPECTED"
        print(f"verify {brand}: complement_inhibitor_status questions {hit} -> {marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print rows, write nothing")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
