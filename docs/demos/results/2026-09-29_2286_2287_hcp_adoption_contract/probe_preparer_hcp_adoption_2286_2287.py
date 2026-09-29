"""End-to-end walk of the HCP-adoption retrain contract (#2286 / #2287), PRODUCTION state shape.

For each brand it starts from the ml_model_registry row AS MIGRATION 163 SETS IT (simulated;
nothing is written), then runs the real code the retrain runs:

  registry row -> contract_from_registry_row -> has_cohort_contract (+ the sweep's
  blocked-reason branch) -> drift_monitoring_tasks._cohort_input_from_training_config ->
  the pipeline's scope_input -> the COMPILED scope_definer graph -> the data_preparer's
  initial state (top-level data_source = the contract dict; scope_spec from scope_definer)
  -> load_data ... finalize_output.

It never runs leakage_remediation or qc_remediation (LLM), kg_role_enrichment (FalkorDB) or
the Feast registrar, and it disables the Layer-4 LLM evaluator and the verdict sidecar.

--source throwaway (default, prod-free): a throwaway container of prod's Postgres image,
    fronted by PostgREST at prod's image tag, seeded from the committed generators
    (HCPGenerator seed 42 reproduces the live hcp_profiles; adoption seed 427 reproduces
    the live arm and prevalence, NOT the per-row labels), with migration 162 applied. The
    loader is handed that PostgREST. SUGGESTIVE, not decisive: same DGP, not the live rows.
--source live: the real get_ml_data_loader() against prod Supabase, SELECTs only. Needs
    migration 162 applied (post-deploy). This is the decisive run.

Both modes also compare the contract load with FeatureBuilder._load_hcp_frame (the frame
the champions were trained on): rows, label rate, data_split distribution, and the row
multiset over the contract columns.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from uuid import uuid4

sys.path.insert(0, os.getcwd())
import logging  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

import pandas as pd  # noqa: E402

import src  # noqa: E402

for var in ("ADAPTIVE_VALIDITY_EVALUATOR_ENABLED", "ADAPTIVE_VALIDITY_ARTIFACTS_DIR"):
    os.environ.pop(var, None)

from src.agents.ml_foundation.data_preparer import graph as G  # noqa: E402
from src.agents.ml_foundation.data_preparer.nodes import data_loader  # noqa: E402
from src.agents.ml_foundation.scope_definer.graph import create_scope_definer_graph  # noqa: E402
from src.data.manifests.resolution import resolve_manifest_source  # noqa: E402
from src.mlops.gold_standard_eval.cohort_spec import BRANDS, make_hcp_spec  # noqa: E402
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder  # noqa: E402
from src.services.cohort_contract import contract_from_registry_row  # noqa: E402
from src.services.retraining_trigger import has_cohort_contract  # noqa: E402
from src.tasks.drift_monitoring_tasks import _cohort_input_from_training_config  # noqa: E402

MIGRATION_163 = "database/migrations/163_registry_cohort_contract_hcp_adoption.sql"


def row_163(brand: str) -> dict:
    """The contract columns migration 163 writes for ``brand``, read from the file, over
    the label migration 151 already set."""
    import re

    text = open(MIGRATION_163).read()
    model = f"hcp_adoption_{brand.lower()}_goldstd_lr_v1"
    m = re.search(
        r"UPDATE ml_model_registry\s+SET (?P<set>.*?)\s+WHERE model_name = '"
        + re.escape(model)
        + "'",
        text,
        re.S,
    )
    assert m, model
    row = {"cohort_target_outcome": "adopted"}
    for col, val in re.findall(r"(cohort_\w+) = '([^']*)'", m.group("set")):
        row[col] = val
    return row


def sweep_decision(cohort: dict) -> str:
    """retraining_trigger.evaluate_and_trigger_retraining's branch for a wanted trigger."""
    wants_trigger = True  # drift/perf says retrain and approval is satisfied
    if wants_trigger and not has_cohort_contract(cohort):
        return "blocked: no_cohort_contract"
    return "enqueue: trigger_retraining(cohort=...)"


def fold(state: dict, update: dict) -> None:
    for k, v in (update or {}).items():
        if v is not None:
            state[k] = v


async def scope_for(input_data: dict) -> dict:
    """scope_spec exactly as the pipeline's scope stage builds it (compiled graph)."""
    scope_input = {
        "problem_description": input_data["problem_description"],
        "business_objective": input_data["business_objective"],
        "target_outcome": input_data["target_outcome"],
        "brand": input_data.get("brand", "unknown"),
        "region": input_data.get("region", "all"),
        "use_case": input_data.get("use_case", "commercial_targeting"),
        "problem_type_hint": input_data.get("problem_type_hint"),
        "target_variable_hint": input_data.get("target_variable_hint"),
        "target_variable": input_data.get("target_variable"),
        "candidate_features": input_data.get("candidate_features"),
        "feature_manifest_source": resolve_manifest_source(
            input_data.get("data_source"), input_data.get("feature_manifest_source")
        ),
        "deployment_intent": input_data.get("deployment_intent"),
        "audit_workflow_id": uuid4(),
        "performance_requirements": {},
    }
    out = await create_scope_definer_graph().ainvoke(scope_input)
    assert not out.get("error"), out.get("error")
    return dict(out["scope_spec"])


MANIFEST_OVERRIDE: str | None = None  # --manifest: experiment without editing 163


def row_163_for(brand: str) -> dict:
    row = row_163(brand)
    if MANIFEST_OVERRIDE is not None:
        if MANIFEST_OVERRIDE == "none":
            row.pop("cohort_feature_manifest_source", None)
        else:
            row["cohort_feature_manifest_source"] = MANIFEST_OVERRIDE
    return row


async def run_brand(brand: str, make_loader, async_db) -> dict:
    row_today = {"cohort_target_outcome": "adopted"}
    row_163 = row_163_for(brand)
    cohort = contract_from_registry_row(row_163)
    print(f"registry row after 163 (simulated, not written): {row_163}")
    print(f"\n=== hcp_adoption / {brand} ===")
    print(
        f"has_cohort_contract: today(151 label only)={has_cohort_contract(contract_from_registry_row(row_today))} "
        f"after163={has_cohort_contract(cohort)}"
    )
    print(
        f"sweep branch: today -> {sweep_decision(contract_from_registry_row(row_today))!r}; "
        f"after163 -> {sweep_decision(cohort)!r}"
    )

    input_data = _cohort_input_from_training_config(dict(cohort))
    scope_spec = await scope_for(input_data)
    scope_spec.setdefault("sufficiency", {})["force_low_power_run"] = False
    print(
        f"scope_spec: prediction_target={scope_spec.get('prediction_target')!r} "
        f"problem_type={scope_spec.get('problem_type')!r} "
        f"feature_manifest_source={scope_spec.get('feature_manifest_source')!r} "
        f"data_source-in-scope_spec={'data_source' in scope_spec}"
    )

    loader = make_loader()
    data_loader.get_ml_data_loader = lambda: loader  # the loader the node builds, same transport
    state = {
        "audit_workflow_id": uuid4(),
        "experiment_id": scope_spec.get("experiment_id") or f"probe2287-{brand.lower()}",
        "scope_spec": scope_spec,
        "data_source": input_data["data_source"],
        "split_id": None,
        "validation_suite": None,
        "skip_leakage_check": False,
        "adaptive_structural_decider_enabled": False,
        "adaptive_fdr_enabled": True,
        "adaptive_declared_safe_full_immunity": False,
        "qc_min_overall_score": None,
        "blocking_issues": [],
    }
    pre = [
        ("load_data", G.load_data),
        ("audit_sampling_frame", G.audit_sampling_frame),
        ("run_schema_validation", G.run_schema_validation),
        ("run_quality_checks", G.run_quality_checks),
        ("run_ge_validation", G.run_ge_validation),
        ("engineer_features", G.engineer_features_node),
        ("detect_leakage", G.detect_leakage),
        ("adaptive_validity_check", G.adaptive_validity_check),
    ]
    keys_of_interest = (
        "error",
        "schema_validation_status",
        "schema_splits_validated",
        "qc_status",
        "overall_score",
        "ge_validation_status",
        "ge_expectations_passed",
        "ge_expectations_evaluated",
        "leakage_severity",
        "leaked_features",
    )
    for name, node in pre:
        t0 = time.time()
        upd = await node(state)
        fold(state, upd)
        keys = {k: upd.get(k) for k in keys_of_interest if k in (upd or {})}
        print(
            f"[{name} {time.time() - t0:.1f}s] {json.dumps(keys, default=str)} blocking={state.get('blocking_issues')}"
        )
        if name == "load_data" and state.get("train_df") is not None:
            frames = {
                "train": state["train_df"],
                "validation": state["validation_df"],
                "test": state["test_df"],
                "holdout": state.get("holdout_df"),
            }
            sizes = {k: (0 if v is None else len(v)) for k, v in frames.items()}
            loaded = pd.concat([v for v in frames.values() if v is not None], ignore_index=True)
            print(f"  rows {sizes} total={len(loaded)} label_rate={loaded['adopted'].mean():.4f}")
            print(f"  frame columns = {list(state['train_df'].columns)}")
            state["_loaded"] = loaded
        if name == "run_schema_validation" and state.get("schema_validation_errors"):
            print(f"  schema errors (first 5): {state['schema_validation_errors'][:5]}")
    if state.get("adaptive_verdicts"):
        sev = Counter(v.get("severity") for v in state["adaptive_verdicts"])
        print(f"  adaptive verdict severities: {dict(sev)}")
        for v in state["adaptive_verdicts"]:
            if v.get("severity") in ("high", "critical"):
                print(
                    f"    {v.get('severity')}: {v.get('feature')} layer={v.get('layer')} {v.get('reason', '')[:160]}"
                )
    route = G._route_after_leakage_detection(state)
    print(f"[route_after_leakage_detection] -> {route}")
    if route == "remediate":
        print("leakage_remediation (LLM) would run -- NOT executed by this probe")
    else:
        for name, node in [
            ("transform_data", G.transform_data),
            ("compute_baseline_metrics", G.compute_baseline_metrics),
            ("sufficiency_check", G.run_sufficiency_check),
            ("finalize_output", G.finalize_output),
        ]:
            t0 = time.time()
            upd = await node(state)
            fold(state, upd)
            keys = {
                k: upd.get(k)
                for k in (
                    "error",
                    "gate_passed",
                    "qc_passed",
                    "is_ready",
                    "missing_required_features",
                )
                if k in (upd or {})
            }
            print(
                f"[{name} {time.time() - t0:.1f}s] {json.dumps(keys, default=str)} blocking={state.get('blocking_issues')}"
            )

    # Trainer handoff: the pipeline builds model_trainer's {X, y} splits from the frames
    # data_preparer returns, exactly like this (tier_0/pipeline.py frames_to_trainer_splits).
    handoff = None
    if state.get("gate_passed"):
        from src.agents.tier_0.split_handoff import (
            feature_columns_to_drop,
            frames_to_trainer_splits,
        )

        frames = {
            "train": state.get("train_df"),
            "validation": state.get("validation_df"),
            "test": state.get("test_df"),
            "holdout": state.get("holdout_df"),
        }
        drop_columns = (
            scope_spec.get("entity_column"),
            scope_spec.get("date_column"),
            *(scope_spec.get("excluded_features") or []),
        )
        plan = feature_columns_to_drop(
            frames["train"], scope_spec["prediction_target"], drop_columns
        )
        splits = frames_to_trainer_splits(
            frames, target_column=scope_spec.get("prediction_target"), drop_columns=drop_columns
        )
        handoff = {
            "X_columns": list(splits["train_data"]["X"].columns),
            "X_dtypes": {c: str(t) for c, t in splits["train_data"]["X"].dtypes.items()},
            "rows": {k: v["row_count"] for k, v in splits.items()},
            "dropped": {k: v for k, v in plan.items() if v and k != "target"},
        }
    print(f"TRAINER_HANDOFF {brand}: {json.dumps(handoff, default=str)}")

    # Frame fidelity: the contract load vs the goldstd builder's frame.
    spec = make_hcp_spec(brand)
    gold = await FeatureBuilder(spec)._load_hcp_frame(async_db)
    loaded = state.get("_loaded")
    cols = list(spec.base_covariates) + ["adopted", "data_split"]

    def multiset(df: pd.DataFrame) -> Counter:
        return Counter(
            tuple(None if pd.isna(v) else v for v in r)
            for r in df[cols].itertuples(index=False, name=None)
        )

    fidelity = None
    if loaded is not None and len(gold):
        fidelity = {
            "rows": (len(loaded), len(gold)),
            "label_rate": (
                round(float(loaded["adopted"].mean()), 4),
                round(float(gold["adopted"].mean()), 4),
            ),
            "splits": (dict(Counter(loaded["data_split"])), dict(Counter(gold["data_split"]))),
            "row_multiset_equal": multiset(loaded) == multiset(gold),
        }
    print(f"FIDELITY {brand} (contract load, goldstd builder): {fidelity}")
    result = {
        "brand": brand,
        "has_cohort_contract_after163": has_cohort_contract(cohort),
        "prediction_target": scope_spec.get("prediction_target"),
        "route": route,
        "gate_passed": state.get("gate_passed"),
        "blocking_issues": state.get("blocking_issues"),
        "schema_validation_status": state.get("schema_validation_status"),
        "ge_validation_status": state.get("ge_validation_status"),
        "leakage_severity": state.get("leakage_severity"),
        "fidelity_row_multiset_equal": (fidelity or {}).get("row_multiset_equal"),
        "trainer_X_columns": (handoff or {}).get("X_columns"),
    }
    print(f"RESULT {json.dumps(result, default=str)}")
    return result


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("throwaway", "live"), default="throwaway")
    ap.add_argument(
        "--manifest",
        default=None,
        help="override the row's cohort_feature_manifest_source ('none' drops it); "
        "default: exactly what migration 163 writes",
    )
    args = ap.parse_args()
    global MANIFEST_OVERRIDE
    MANIFEST_OVERRIDE = args.manifest
    print(
        f"code under test: {os.path.dirname(src.__file__)}  source={args.source} manifest_override={args.manifest}"
    )

    from src.repositories.ml_data_loader import MLDataLoader

    cleanup = []
    if args.source == "throwaway":
        from tests.unit.test_database._hcp_adoption_pg import (
            M162_PATH,
            PostgrestServer,
            SupabaseShim,
            apply_file,
            build_base,
            prod_image,
            seed,
        )
        from tests.unit.test_database.learning_loop._pg import PgConn, ThrowawayPg

        pg = ThrowawayPg(image=prod_image("supabase-db"))
        pg.start()
        cleanup.append(pg.stop)
        pg.rows("postgres", "CREATE DATABASE probe2287 OWNER postgres")
        conn = PgConn(pg, "probe2287")
        build_base(conn)
        print(f"seeded (generators, prod-free): {seed(conn)}; images db={pg.image}")
        apply_file(conn, M162_PATH)
        server = PostgrestServer(image=prod_image("supabase-rest"), pg=pg, db="probe2287")
        server.start()
        cleanup.insert(0, server.stop)
        print(f"postgrest image={server.image}")

        def make_loader():
            return MLDataLoader(supabase_client=SupabaseShim(server.client("service_role")))

        async_db = SupabaseShim(server.client("service_role", is_async=True))
    else:
        from src.memory.services.factories import get_async_supabase_client
        from src.repositories.ml_data_loader import get_ml_data_loader

        make_loader = get_ml_data_loader
        async_db = await get_async_supabase_client()

    try:
        results = [await run_brand(b, make_loader, async_db) for b in BRANDS]
    finally:
        for fn in cleanup:
            fn()
    print("\nSUMMARY")
    for r in results:
        print(json.dumps(r, default=str))


asyncio.run(main())
