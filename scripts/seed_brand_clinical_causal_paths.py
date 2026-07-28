#!/usr/bin/env python
"""Targeted, idempotent apply of the brand-distinct clinical-axis causal_paths edges.

The synthetic substrate is FROZEN (full reseed = disaster recovery only). Each brand's
axis column already exists on patient_journeys (populated for that brand's rows, NULL
off-brand — BRAND_ELIGIBILITY_FIELDS). This script only adds the ``causal_paths`` edge
that makes each brand's planted axis persistence effect appear on that brand's discovery
leaderboard AND (after the FalkorDB sync) as a node on that brand's /knowledge-graph —
the brand-DISTINCT causal variables in the gold standard (issue #1321):
    * Fabhalta       — complement_inhibitor_status (prior C5-inhibitor switch)  [pilot]
    * Kisqali        — disease_stage (advanced line: metastatic / stage_iv)
    * Remibrutinib   — urticaria_severity_uas7 (uncontrolled CSU, UAS7 >= 28)

Content-addressed path_id + values (CausalPathsGenerator._BRAND_CLINICAL_AXES, namespace
scp_f*), so re-runs are no-ops apart from freshness stamps, and a future disaster-recovery
reseed emits the same rows and upserts over them harmlessly. Each edge is emitted for its
brand ONLY — the others' KGs never gain the node.

By DEFAULT this seeds the ROLLOUT brands (Kisqali + Remibrutinib); Fabhalta's edge is
already live in prod (the pilot). Pass --brands to override.

Run inside the API container (it has the Supabase env):

    docker exec e2i_api python scripts/seed_brand_clinical_causal_paths.py [--dry-run]
    docker exec e2i_api python scripts/seed_brand_clinical_causal_paths.py --brands Kisqali Remibrutinib Fabhalta

After applying, sync the KG so the nodes appear on each brand's graph too:

    docker exec e2i_api python scripts/sync_causal_paths_to_falkordb.py --execute
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the repo root to sys.path so ``python scripts/seed_brand_clinical_causal_paths.py``
# resolves ``src`` (sys.path[0] is scripts/, not the repo root). Mirrors the bootstrap in
# the sibling seed scripts; harmless as a module or with PYTHONPATH set.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.synthetic.generators.causal_paths_generator import clinical_axis_rows_for_upsert

# The rollout defaults to the two NEW brands — Fabhalta's edge is already live (pilot).
_DEFAULT_BRANDS = ["Kisqali", "Remibrutinib"]
_AXIS_BY_BRAND = {
    "Fabhalta": "complement_inhibitor_status",
    "Kisqali": "disease_stage",
    "Remibrutinib": "urticaria_severity_uas7",
}


async def main(brands: list[str], dry_run: bool) -> None:
    records = clinical_axis_rows_for_upsert(brands)
    ids = [r["path_id"] for r in records]
    print(f"{len(records)} clinical causal-path row(s) for {brands} (content-addressed, scp_f*)")
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

    # Faithful verification: each seeded brand's edge must surface as a leaderboard
    # question for THAT brand ONLY (the brand-distinct guarantee).
    from src.repositories.causal_path import CausalPathRepository

    repo = CausalPathRepository(client)
    for brand in ("Fabhalta", "Kisqali", "Remibrutinib"):
        qs = await repo.get_distinct_questions(brand=brand, include_synthetic=True)
        for seeded in brands:
            axis = _AXIS_BY_BRAND[seeded]
            hit = [(q["treatment"], q["outcome"]) for q in qs if q["treatment"] == axis]
            marker = "OK (present)" if (brand == seeded) == bool(hit) else "!! UNEXPECTED"
            print(f"verify {brand}: {axis} questions {hit} -> {marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print rows, write nothing")
    parser.add_argument(
        "--brands",
        nargs="+",
        default=_DEFAULT_BRANDS,
        choices=sorted(_AXIS_BY_BRAND),
        help="Brands to seed (default: the rollout brands Kisqali + Remibrutinib).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.brands, args.dry_run))
