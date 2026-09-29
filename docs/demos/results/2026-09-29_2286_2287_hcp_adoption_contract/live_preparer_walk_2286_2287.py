"""READ-ONLY live proof 3/3 (#2286 / #2287): the 3-brand data_preparer walk on live data.

Runs probe_preparer_hcp_adoption_2286_2287.run_brand per brand with the LIVE clients:
registry row as 163 sets it -> has_cohort_contract -> the sweep's retrain input -> the
compiled scope_definer graph -> data_preparer load_data .. finalize_output -> the
pipeline's trainer handoff, plus the frame comparison with the goldstd builder. Never runs
LLM remediation, kg_role_enrichment (FalkorDB) or the Feast registrar; SELECTs only.

Needs migration 162 applied (the view). Before that it prints VERDICT: NOT-RUNNABLE.

    PYTHONPATH=$PWD .venv/bin/python \
      docs/demos/results/2026-09-29_2286_2287_hcp_adoption_contract/live_preparer_walk_2286_2287.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _live_common as C  # noqa: E402


async def run(sync_client: Any, async_client: Any) -> str:
    import probe_preparer_hcp_adoption_2286_2287 as probe

    from src.repositories.ml_data_loader import MLDataLoader

    try:
        sync_client.table(C.VIEW).select("hcp_id").limit(1).execute()
    except Exception as e:  # noqa: BLE001 — the view is absent until 162 applies
        print(f"view probe failed: {e}")
        return "NOT-RUNNABLE (hcp_adoption_goldstd_v not served: apply migration 162 first)"

    results = [
        await probe.run_brand(b, lambda: MLDataLoader(supabase_client=sync_client), async_client)
        for b in C.BRANDS
    ]
    print("\nSUMMARY")
    failures: List[str] = []
    for r in results:
        print(json.dumps(r, default=str))
        why = []
        if not r.get("has_cohort_contract_after163"):
            why.append("has_cohort_contract False")
        if r.get("prediction_target") != "adopted":
            why.append(f"prediction_target={r.get('prediction_target')!r}")
        if r.get("route") != "continue":
            why.append(f"route={r.get('route')} (leakage_severity={r.get('leakage_severity')})")
        if r.get("gate_passed") is not True:
            why.append(f"gate_passed={r.get('gate_passed')}")
        if r.get("blocking_issues"):
            why.append(f"blocking={r.get('blocking_issues')}")
        if r.get("fidelity_row_multiset_equal") is not True:
            why.append("frame != goldstd builder frame")
        if why:
            failures.append(f"{r['brand']}: " + ", ".join(why))
    return "PASS" if not failures else "FAIL: " + "; ".join(failures)


async def _main() -> None:
    out = C.tee_to_out(__file__)
    C.header("live proof 3/3: data_preparer walk, all 3 brands (read-only)")
    print(f".env loaded from: {C.load_env()}")
    verdict = await run(C.live_sync_client(), await C.live_async_client())
    print(f"\n(output also written to {out})")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    asyncio.run(_main())
