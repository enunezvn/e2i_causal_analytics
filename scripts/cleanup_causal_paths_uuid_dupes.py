#!/usr/bin/env python
"""One-off cleanup for #1725: collapse the uuid-family causal_paths duplicates.

CausalPathsGenerator used to mint ``scp_<uuid4.hex[:13]>`` path_ids for the
patient/hcp/trigger grains, so every reseed INSERTED a fresh copy of the same
logical path instead of updating in place (PK conflict never fired) — measured
2,657 rows across 21 logical identities before this cleanup. The generator now
mints content-addressed ids (``scp_p*``/``scp_h*``/``scp_t*``), which makes the
on-PK upsert idempotent going forward; this script deletes the legacy uuid-family
rows (and their linked causal_validations evidence) and applies the deterministic
replacement rows once.

Only the legacy family is touched: ``^scp_[0-9a-f]{13}$`` (17 chars) is disjoint
from every deterministic family (scp_c*/scp_a*/scp_f*/scp_p*/scp_h*/scp_t*, all
16 chars) and from the pre-synthetic real rows, and the delete is additionally
gated on ``is_synthetic = TRUE``.

causal_validations linkage: migration 119 keys evidence as
``uuid5(NAMESPACE_URL, "e2i:causal_paths:" + path_id)`` with
``estimate_source='causal_paths'`` — mirrored by
``derive_causal_path_estimate_id`` — so orphaned evidence is derivable from the
deleted path_ids. The DB trigger re-seeds evidence for the replacement rows on
insert, so no manual evidence writes are needed.

FalkorDB: the KG sync (scripts/sync_causal_paths_to_falkordb.py) MERGEs nodes
keyed by VARIABLE name, not path_id, and the replacement rows carry the same
variable vocabulary — deleting duplicate registry rows strands no KG nodes.

Run inside the API container (it has the Supabase env):

    docker exec e2i_api python scripts/cleanup_causal_paths_uuid_dupes.py [--dry-run]

Re-runs are safe: the legacy scan finds nothing and the upsert is a no-op by PK.
"""

import argparse
import asyncio
import re

from src.ml.synthetic.generators.causal_paths_generator import (
    cohort_hcp_trigger_rows_for_upsert,
)
from src.repositories.causal_validation import derive_causal_path_estimate_id

#: Legacy uuid4 family: "scp_" + 13 hex chars (17 total). Every deterministic
#: family is 16 chars ("scp_" + [cafpht] + 11 hex), so this cannot match one.
_LEGACY_UUID_FAMILY = re.compile(r"^scp_[0-9a-f]{13}$")

_PAGE = 1000  # PostgREST per-request row cap
_DELETE_CHUNK = 100


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def _scan_synthetic_path_ids(client) -> list[str]:
    """All synthetic path_ids, paginated past PostgREST's per-request cap."""
    ids: list[str] = []
    offset = 0
    while True:
        page = (
            await client.table("causal_paths")
            .select("path_id")
            .eq("is_synthetic", True)
            .order("path_id")
            .range(offset, offset + _PAGE - 1)
            .execute()
        )
        ids.extend(row["path_id"] for row in page.data)
        if len(page.data) < _PAGE:
            return ids
        offset += _PAGE


async def main(dry_run: bool) -> None:
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()

    all_ids = await _scan_synthetic_path_ids(client)
    legacy = [pid for pid in all_ids if _LEGACY_UUID_FAMILY.match(pid)]
    print(f"synthetic causal_paths rows: {len(all_ids)}; legacy uuid-family: {len(legacy)}")

    records = cohort_hcp_trigger_rows_for_upsert()
    print(f"deterministic replacement rows (scp_p/h/t): {len(records)}")

    if dry_run:
        for pid in legacy[:5]:
            print(f"  would delete {pid} (+ evidence {derive_causal_path_estimate_id(pid)})")
        print("  ... dry run, nothing written")
        return

    # 1) Orphan evidence first (no FK — nothing cascades for us).
    deleted_evidence = 0
    for chunk in _chunks([derive_causal_path_estimate_id(pid) for pid in legacy], _DELETE_CHUNK):
        result = (
            await client.table("causal_validations")
            .delete()
            .eq("estimate_source", "causal_paths")
            .in_("estimate_id", chunk)
            .execute()
        )
        deleted_evidence += len(result.data)
    print(f"deleted causal_validations evidence rows: {deleted_evidence}")

    # 2) The legacy paths themselves (is_synthetic re-checked on the delete).
    deleted_paths = 0
    for chunk in _chunks(legacy, _DELETE_CHUNK):
        result = (
            await client.table("causal_paths")
            .delete()
            .eq("is_synthetic", True)
            .in_("path_id", chunk)
            .execute()
        )
        deleted_paths += len(result.data)
    print(f"deleted legacy causal_paths rows: {deleted_paths}")

    # 3) Apply the deterministic replacement rows — twice, to PROVE idempotency
    #    (the second pass must leave the registry count unchanged).
    await client.table("causal_paths").upsert(records, on_conflict="path_id").execute()
    count_after_first = len(await _scan_synthetic_path_ids(client))
    await client.table("causal_paths").upsert(records, on_conflict="path_id").execute()
    ids_after_second = await _scan_synthetic_path_ids(client)
    print(
        f"registry after seed: {count_after_first} synthetic rows; "
        f"after re-seed: {len(ids_after_second)} "
        f"({'IDEMPOTENT' if len(ids_after_second) == count_after_first else 'NOT IDEMPOTENT'})"
    )

    leftovers = [pid for pid in ids_after_second if _LEGACY_UUID_FAMILY.match(pid)]
    print(f"legacy uuid-family remaining: {len(leftovers)}")
    if leftovers or len(ids_after_second) != count_after_first:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(main(parser.parse_args().dry_run))
