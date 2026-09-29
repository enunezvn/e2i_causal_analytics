"""READ-ONLY live proof 1/3 (#2286 / #2287): what migration 163 will do to the registry rows.

Reads every ml_model_registry row named hcp_adoption_<brand>_goldstd_lr_v1 (one SELECT),
evaluates migration 163's WHERE for each in Python, applies its SET IN MEMORY to the rows
it would match, and runs the real has_cohort_contract / contract_from_registry_row /
recorded_calibration_method on the before and after shapes. Nothing is written.

Runnable before OR after deploy. VERDICT: PASS when, for every brand, exactly one
production real row exists, 163 would match it (or already has), today's row is refused
(no_cohort_contract) and the after-163 row passes has_cohort_contract.

    PYTHONPATH=$PWD .venv/bin/python \
      docs/demos/results/2026-09-29_2286_2287_hcp_adoption_contract/live_registry_rows_2286_2287.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _live_common as C  # noqa: E402

_COLUMNS = (
    "id, model_name, model_version, stage, is_champion, is_synthetic, cohort_data_source, "
    "cohort_target_outcome, cohort_feature_manifest_source, hyperparameters"
)


def _would_match(row: Dict[str, Any]) -> bool:
    """Migration 163's WHERE, evaluated on a fetched row."""
    return (
        row.get("stage") == "production"
        and row.get("is_synthetic") is False
        and row.get("cohort_data_source") is None
        and row.get("cohort_feature_manifest_source") is None
        and row.get("cohort_target_outcome") == "adopted"
    )


def run(sync_client: Any) -> str:
    from src.services.cohort_contract import (
        contract_from_registry_row,
        recorded_calibration_method,
    )
    from src.services.retraining_trigger import has_cohort_contract

    m163 = C.migration_163_rows()
    names = [C.model_name(b) for b in C.BRANDS]
    rows: List[Dict[str, Any]] = (
        sync_client.table("ml_model_registry")
        .select(_COLUMNS)
        .in_("model_name", names)
        .execute()
        .data
        or []
    )
    print(f"rows read: {len(rows)} (all stages, synthetic included)")
    failures: List[str] = []
    for brand in C.BRANDS:
        name = C.model_name(brand)
        written = m163[name]["set"]
        mine = [r for r in rows if r.get("model_name") == name]
        prod_real = [
            r for r in mine if r.get("stage") == "production" and r.get("is_synthetic") is False
        ]
        print(f"\n=== {name} === rows={len(mine)} production+real={len(prod_real)}")
        for r in mine:
            shown = {
                k: r.get(k) for k in ("id", "model_version", "stage", "is_champion", "is_synthetic")
            }
            shown["cohort_data_source"] = (r.get("cohort_data_source") or "<NULL>")[:60]
            shown["cohort_target_outcome"] = r.get("cohort_target_outcome")
            shown["cohort_feature_manifest_source"] = r.get("cohort_feature_manifest_source")
            shown["163_would_update"] = _would_match(r)
            print(f"  {json.dumps(shown, default=str)}")
        if len(prod_real) != 1:
            failures.append(f"{name}: {len(prod_real)} production real rows (expected 1)")
            continue
        row = prod_real[0]
        applied = all(row.get(col) == val for col, val in written.items())
        before = dict(row)
        if applied:
            print("  163 is ALREADY applied to this row (its pair is present)")
            before = {**row, **{col: None for col in written}}
        elif not _would_match(row):
            failures.append(f"{name}: 163's WHERE does not match the production row")
            print("  163 would NOT match this row -> it keeps no_cohort_contract")
            continue
        after = {**before, **written}
        ok_before = has_cohort_contract(contract_from_registry_row(before))
        contract_after = contract_from_registry_row(after)
        ok_after = has_cohort_contract(contract_after)
        print(
            f"  has_cohort_contract: before={ok_before} after={ok_after}; sweep branch "
            f"before={'blocked: no_cohort_contract' if not ok_before else 'enqueue'} "
            f"after={'enqueue' if ok_after else 'blocked: no_cohort_contract'}"
        )
        print(
            f"  after-163 contract: data_source.table={contract_after.get('data_source', {}).get('table')} "
            f"target={contract_after.get('target_outcome')} "
            f"manifest={contract_after.get('feature_manifest_source')} "
            f"calibration_method={recorded_calibration_method(row)}"
        )
        if ok_before:
            failures.append(f"{name}: today's row already passes has_cohort_contract")
        if not ok_after:
            failures.append(f"{name}: the after-163 row fails has_cohort_contract")
    verdict = "PASS" if not failures else "FAIL: " + "; ".join(failures)
    return verdict


def main() -> None:
    out = C.tee_to_out(__file__)
    C.header("live proof 1/3: registry rows + has_cohort_contract simulation (read-only)")
    print(f".env loaded from: {C.load_env()}")
    verdict = run(C.live_sync_client())
    print(f"\n(output also written to {out})")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
