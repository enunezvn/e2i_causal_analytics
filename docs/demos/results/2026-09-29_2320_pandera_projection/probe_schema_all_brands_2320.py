"""READ-ONLY: load_data + run_schema_validation on all 9 goldstd patient contracts
(3 cohorts x 3 brands) as the registry holds them (migration 151 shape), production state
shape (contract on the top-level data_source only). SELECTs only."""
import asyncio, os, sys
sys.path.insert(0, os.getcwd())
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
import src
from src.agents.ml_foundation.data_preparer import graph as G
from src.mlops.gold_standard_eval.cohort_spec import _PATIENT_COVARIATES, _PATIENT_LABELS
print(f"code under test: {os.path.dirname(src.__file__)}")


async def main():
    for cohort in ("initiation", "persistence", "discontinuation"):
        for brand in ("Kisqali", "Fabhalta", "Remibrutinib"):
            label = _PATIENT_LABELS[cohort]
            contract = {"type": "table", "table": "patient_journeys",
                        "filters": {"brand": brand, "is_synthetic": True},
                        "columns": list(_PATIENT_COVARIATES[cohort]) + [label]}
            state = {"experiment_id": f"p2320-{cohort}-{brand}", "data_source": contract,
                     "scope_spec": {"prediction_target": label, "experiment_id": "p2320"},
                     "blocking_issues": []}
            for node in (G.load_data, G.run_schema_validation):
                state.update({k: v for k, v in ((await node(state)) or {}).items() if v is not None})
            n = [len(state.get(k)) for k in ("train_df", "validation_df", "test_df")]
            print(f"{cohort:<16}{brand:<14} rows={n} schema={state.get('schema_validation_status')} "
                  f"splits={state.get('schema_splits_validated')} errors={len(state.get('schema_validation_errors') or [])} "
                  f"blocking={state.get('blocking_issues')}")

asyncio.run(main())
