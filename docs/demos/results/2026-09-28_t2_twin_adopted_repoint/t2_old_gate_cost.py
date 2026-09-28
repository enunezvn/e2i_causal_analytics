"""READ-ONLY cost baseline: the availability gate this lane replaced (eight concurrent
PostgREST exact-count probes on business_metrics, one per channel), timed per brand on the
same box and server as t2_live_probe.py's new one-read gate. Every call is a .select()."""

import asyncio
import sys
import time

WORKTREE = "/home/enunez/Projects/e2i_causal_analytics/.worktrees/twin-adopted-repoint"
sys.path.insert(0, WORKTREE)

from src.data.per_hcp_cohort_collapse import CHANNEL_COLUMNS  # noqa: E402
from src.digital_twin.effect.provider import COHORT_CONFOUNDERS  # noqa: E402


async def _count(client, brand, col):
    q = (
        client.table("business_metrics")
        .select("metric_id", count="exact")
        .eq("metric_type", "per_hcp_rollup")
        .eq("brand", brand)
        .not_.is_(col, "null")
        .not_.is_("cohort_conversion_outcome", "null")
        .not_.is_("region", "null")
    )
    for c in COHORT_CONFOUNDERS:
        q = q.not_.is_(c, "null")
    return (await q.limit(1).execute()).count


async def main():
    from src.memory.services.factories import loop_scoped_async_supabase_client

    async with loop_scoped_async_supabase_client() as client:
        for brand in ("Remibrutinib", "Fabhalta", "Kisqali"):
            t0 = time.perf_counter()
            counts = await asyncio.gather(*(_count(client, brand, c) for c in CHANNEL_COLUMNS))
            print(f"OLD-GATE {brand:<13} {time.perf_counter() - t0:5.2f}s counts={counts}")


asyncio.run(main())
