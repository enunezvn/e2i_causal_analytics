"""READ-ONLY live proof 2/3 (#2287): the retrain frame == the frame the champions trained on.

Per brand, loads (a) FeatureBuilder(make_hcp_spec(brand))._load_hcp_frame -- the goldstd
builder's hcp_brand_adoption + hcp_profiles FK-embed frame -- and (b) the retrain frame:
migration 163's contract through the real data_loader.load_data node -> MLDataLoader ->
hcp_adoption_goldstd_v. Compares row count, label rate, data_split distribution and the
row multiset over the contract columns (covariates + adopted + data_split). SELECTs only.

Needs migration 162 applied (the view). Before that it prints VERDICT: NOT-RUNNABLE.

    PYTHONPATH=$PWD .venv/bin/python \
      docs/demos/results/2026-09-29_2286_2287_hcp_adoption_contract/live_frame_equivalence_2287.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from typing import Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _live_common as C  # noqa: E402


def _multiset(df: Any, cols: List[str]) -> Counter:
    import pandas as pd

    return Counter(
        tuple(None if pd.isna(v) else v for v in r)
        for r in df[cols].itertuples(index=False, name=None)
    )


async def run(sync_client: Any, async_client: Any) -> str:
    import pandas as pd

    from src.agents.ml_foundation.data_preparer.nodes import data_loader
    from src.mlops.gold_standard_eval.cohort_spec import make_hcp_spec
    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder
    from src.repositories.ml_data_loader import MLDataLoader
    from src.services.cohort_contract import contract_from_registry_row

    try:
        sync_client.table(C.VIEW).select("hcp_id").limit(1).execute()
    except Exception as e:  # noqa: BLE001 — the view is absent until 162 applies
        print(f"view probe failed: {e}")
        return "NOT-RUNNABLE (hcp_adoption_goldstd_v not served: apply migration 162 first)"

    loader = MLDataLoader(supabase_client=sync_client)
    data_loader.get_ml_data_loader = lambda: loader  # same client the node would build
    m163 = C.migration_163_rows()
    failures: List[str] = []
    for brand in C.BRANDS:
        spec = make_hcp_spec(brand)
        gold = await FeatureBuilder(spec)._load_hcp_frame(async_client)
        row = {"cohort_target_outcome": "adopted", **m163[C.model_name(brand)]["set"]}
        contract = contract_from_registry_row(row)
        out = await data_loader.load_data(
            {
                "experiment_id": f"live2287-{brand.lower()}",
                "data_source": contract["data_source"],
                "scope_spec": {"prediction_target": "adopted"},
            }
        )
        if out.get("error"):
            failures.append(f"{brand}: load_data error {out['error']}")
            print(f"{brand}: load_data error {out['error']}")
            continue
        frames = [out.get(k) for k in ("train_df", "validation_df", "test_df", "holdout_df")]
        loaded = pd.concat([f for f in frames if f is not None], ignore_index=True)
        cols = list(spec.base_covariates) + ["adopted", "data_split"]
        result = {
            "rows": (len(loaded), len(gold)),
            "label_rate": (
                round(float(loaded["adopted"].mean()), 4) if len(loaded) else None,
                round(float(gold["adopted"].mean()), 4) if len(gold) else None,
            ),
            "splits": (dict(Counter(loaded["data_split"])), dict(Counter(gold["data_split"]))),
            "holdout_rows": 0 if out.get("holdout_df") is None else len(out["holdout_df"]),
            "row_multiset_equal": bool(len(gold))
            and _multiset(loaded, cols) == _multiset(gold, cols),
        }
        print(f"{brand} (retrain load, goldstd builder): {json.dumps(result, default=str)}")
        if not len(gold):
            failures.append(f"{brand}: goldstd builder frame is empty")
        elif not result["row_multiset_equal"] or result["rows"][0] != result["rows"][1]:
            failures.append(f"{brand}: frames differ")
        elif not result["holdout_rows"]:
            failures.append(f"{brand}: no holdout (data_split did not resolve)")
    return "PASS" if not failures else "FAIL: " + "; ".join(failures)


async def _main() -> None:
    out = C.tee_to_out(__file__)
    C.header("live proof 2/3: frame equivalence, retrain contract vs goldstd builder (read-only)")
    print(f".env loaded from: {C.load_env()}")
    verdict = await run(C.live_sync_client(), await C.live_async_client())
    print(f"\n(output also written to {out})")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    asyncio.run(_main())
