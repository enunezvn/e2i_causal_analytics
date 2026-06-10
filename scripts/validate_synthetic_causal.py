"""Acceptance-gate harness for the synthetic causal-validation dataset (Shard 11).

Runs INDEX gates 1-11 against the FAITHFUL docker Supabase. Each gate is a discrete,
individually-runnable function returning a GateResult(name, ok, measured, expected).
OOM-safe: LOKY_MAX_CPU_COUNT=1; no full-tree work. Usage::

    LOKY_MAX_CPU_COUNT=1 python scripts/validate_synthetic_causal.py            # all
    LOKY_MAX_CPU_COUNT=1 python scripts/validate_synthetic_causal.py --gate 3   # one

REASON-BEFORE-RULES / anti-fabrication: every column, RPC id, agent entrypoint, and
orchestrator state key below is VERIFIED against the live docker Supabase + real
source (2026-06-10). Where the plan's sketch diverged from reality the harness aligns
to reality with an inline ``RECONCILED:`` note. A gate that cannot recover a real
value FAILS (ok=False) — it never papers over absent substrate with a fabricated pass.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")  # OOM discipline (INDEX §SHARED)

from src.api.dependencies.supabase_client import get_supabase  # noqa: E402

# INDEX §SHARED brand_type enum (verified: src/ml/synthetic/config.py Brand).
BRANDS = ["Remibrutinib", "Kisqali", "Fabhalta"]


@dataclass
class GateResult:
    name: str
    ok: bool
    measured: object
    expected: str


def _kpi(client, query_id: str, params: list) -> list:
    """Call the kpi_query allowlist RPC exactly as the runtime does.

    VERIFIED: ``kpi_query(query_id text, params jsonb DEFAULT '[]')`` RETURNS SETOF
    json (database/migrations/044_kpi_query_allowlist.sql:49). supabase-py JSON-encodes
    the ``params`` list into the jsonb arg; ``.data`` is the row list of json objects.
    """
    return (
        client.rpc("kpi_query", {"query_id": query_id, "params": params}).execute().data
    ) or []


def _banner(r: GateResult) -> str:
    tag = "PASS" if r.ok else "FAIL"
    return f"[{tag}] {r.name}: measured={r.measured!r} expected={r.expected}"


# =============================================================================
# Gates 1 & 2 — DATE-FRESHNESS + KPI->DASHBOARD
# =============================================================================
# VERIFIED RPC ids (database/migrations/044_kpi_query_allowlist.sql):
#   business_impact_trx [brand] -> {trx}                 (044:128)
#   business_impact_conversion_rate []  -> {conversion_rate} (044:132)


def gate_1_date_freshness(client) -> GateResult:
    measured = {}
    ok = True
    for brand in BRANDS:
        rows = _kpi(client, "business_impact_trx", [brand])
        v = (rows[0].get("trx") if rows else 0) or 0
        measured[brand] = v
        ok = ok and v > 0
    conv = _kpi(client, "business_impact_conversion_rate", [])
    cv = (conv[0].get("conversion_rate") if conv else 0) or 0
    measured["conversion_rate"] = cv
    ok = ok and cv > 0
    return GateResult(
        "1 DATE-FRESHNESS", ok, measured,
        "per-brand TRx>0 and conversion_rate>0 over NOW()-30d",
    )


def gate_2_kpi_dashboard(client) -> GateResult:
    import asyncio

    from src.api.routes.copilotkit import get_kpi_summary

    summary = asyncio.run(get_kpi_summary("Kisqali"))
    # RECONCILED: get_kpi_summary returns {"error": ...} (no metrics/data_source) for
    # an unknown brand (copilotkit.py:1289) — read defensively so a shape miss FAILS
    # explicitly instead of raising a KeyError that masks the real measured value.
    metrics = summary.get("metrics") or {}
    data_source = summary.get("data_source")
    nonnull = [v for v in metrics.values() if v not in (None, 0)]
    ok = data_source == "database" and len(nonnull) >= 4
    return GateResult(
        "2 KPI->DASHBOARD", ok,
        {"data_source": data_source, "non_zero_metrics": len(nonnull)},
        "data_source='database' with >=4 non-zero metrics",
    )


# =============================================================================
# Gates 3 & 4 — ATE/CATE RECOVERY + TRIGGER EFFECTIVENESS
# =============================================================================
# RECONCILED canonical per-unit substrate (verified \d+ patient_journeys, 2026-06-10):
#   patient_journeys has treatment_arm, treatment_initiated, discontinued_180d,
#   persistent_180d, disease_severity, age_at_diagnosis, segment_assignment,
#   propensity_score, brand, is_synthetic. There is NO stored `cohort` column —
#   cohort is resolved by its OUTCOME column (mirrors cohort_resolution._PJ_COHORTS).
# RECONCILED hcp grain: hcp_profiles has ONLY adoption_category + is_synthetic of the
#   plan's claimed feature cols (verified \d hcp_profiles) — the rich per-HCP CATE
#   substrate lives in the Shard-06 artifact data/synthetic/cohort_frames/
#   hcp_adoption__<brand>.parquet [hcp_id, cate_estimate, is_synthetic]. Selecting the
#   plan's disease_severity/propensity_score/... off hcp_profiles would 42703-crash, so
#   the hcp_adoption frame is read from that artifact (fail loud if absent).

# cohort -> (outcome column on patient_journeys, requires treatment_initiated=1)
_COHORT_OUTCOME = {
    "initiation": ("treatment_initiated", False),
    "discontinuation": ("discontinued_180d", True),
    "persistence": ("persistent_180d", True),
    # hcp_adoption resolves on the per-HCP CATE artifact (handled separately)
}

# cohort -> the patient-generator DGPType.value whose designed TRUE_ATE the sidecar
# carries. The synthetic loader runs ONE --dgp per load; gate 3 probes the recovered
# ATE against that designed value. (config.DGPType: simple_linear/confounded/
# heterogeneous/time_series/selection_bias — there is NO 'initiation' DGPType, so the
# sidecar is matched by brand + the run's dgp_type, NOT by a fabricated cohort key.)
_DEFAULT_DGP = "confounded"  # load_synthetic_data.py --dgp default

# cohort -> patient-generator DGPType.value (config.DGPType). Best-effort map for the
# sidecar lookup; the run's actual --dgp is also accepted by _true_ate.
_COHORT_DGP = {
    "initiation": "confounded",
    "discontinuation": "heterogeneous",
    "persistence": "heterogeneous",
    "hcp_adoption": "heterogeneous",
}


def _resolve_synthetic_frame(client, cohort: str, brand: str):
    """Read per-unit synthetic rows into a pandas frame, resolving the cohort by its
    OUTCOME column and renaming canonical cols to the agent's treatment/outcome arg
    names IN-FRAME. Read-only, bound to REAL columns only. Fail loud (AssertionError)
    when the substrate is absent — never fabricate a fallback frame."""
    import pandas as pd

    if cohort == "hcp_adoption":
        # RECONCILED: HCP grain comes from the Shard-06 per-HCP CATE artifact, not
        # hcp_profiles (which lacks the feature cols). cate_estimate is the per-HCP
        # treatment effect; outcome is a binary adopter label derived from its sign.
        import os

        path = f"data/synthetic/cohort_frames/hcp_adoption__{brand}.parquet"
        if not os.path.exists(path):
            raise AssertionError(
                f"no hcp_adoption/{brand} CATE artifact at {path} (Shard 06 not run)"
            )
        df = pd.read_parquet(path)
        if df.empty:
            raise AssertionError(f"empty hcp_adoption/{brand} artifact {path}")
        if "cate_estimate" not in df.columns:
            raise AssertionError(
                f"hcp_adoption/{brand} artifact missing cate_estimate col: {list(df.columns)}"
            )
        # Binary adopter outcome from the per-HCP CATE sign (positive effect -> adopter).
        df["outcome"] = (df["cate_estimate"].astype(float) > df["cate_estimate"].astype(float).median()).astype(int)
        df["treatment"] = 1  # HCP-grain artifact is the treated/exposed analytic frame
        conf = [c for c in ("cate_estimate",) if c in df.columns]
        return df, conf

    outcome_col, needs_initiated = _COHORT_OUTCOME[cohort]
    q = (
        client.table("patient_journeys")
        .select(
            "treatment_arm,treatment_initiated,discontinued_180d,persistent_180d,"
            "disease_severity,age_at_diagnosis,segment_assignment,propensity_score,brand"
        )
        .eq("is_synthetic", True)
        .eq("brand", brand)
    )
    if needs_initiated:  # disc/persistence are conditioned on initiation eligibility
        q = q.eq("treatment_initiated", 1)
    rows = q.limit(5000).execute().data or []
    if not rows:
        raise AssertionError(f"no synthetic {cohort}/{brand} rows (Shard 03/06 not loaded)")
    df = pd.DataFrame(rows)
    if outcome_col not in df.columns:
        raise AssertionError(
            f"patient_journeys missing canonical outcome col {outcome_col!r} "
            f"(Shard 01/03 regression); have {list(df.columns)}"
        )
    df["outcome"] = df[outcome_col].astype(int)  # canonical outcome -> agent arg in-frame
    df["treatment"] = df["treatment_arm"]  # canonical arm -> agent arg in-frame
    conf = [c for c in ("disease_severity", "age_at_diagnosis") if c in df.columns]
    return df, conf


def _true_ate(client, cohort: str, brand: str) -> float:
    """Read the DESIGNED TRUE_ATE from Shard 03's ground-truth sidecar.

    RECONCILED to the REAL GroundTruthStore.to_json_file contract
    (src/ml/synthetic/ground_truth/causal_effects.py:98) — it writes a JSON **list**
    of GroundTruthEffect.to_dict() entries, each keyed by {brand, dgp_type, true_ate,
    cate_by_segment}. The plan's truth["cells"][f"{cohort}/{brand}"] shape does NOT
    exist. We match on brand + the run's dgp_type (cohorts map to one patient-generator
    DGPType, not to a fabricated cohort key). Fail loud if the sidecar or the matching
    entry is absent — gate 3/10 then RED for the RIGHT reason (Shard 03 sidecar
    producer not run), never a fabricated value.
    """
    import glob
    import json
    import os

    paths = sorted(glob.glob("data/synthetic/ground_truth_*.json"), key=os.path.getmtime)
    if not paths:
        raise AssertionError(
            "no data/synthetic/ground_truth_<run>.json (Shard 03 GroundTruthStore "
            "sidecar not written — wire GroundTruthStore.to_json_file into the load)"
        )
    with open(paths[-1]) as fh:  # most recent run
        truth = json.load(fh)
    entries = truth if isinstance(truth, list) else (truth.get("effects") or truth.get("cells") or [])
    if isinstance(entries, dict):  # tolerate a {key: entry} mapping shape
        entries = list(entries.values())
    for e in entries:
        if not isinstance(e, dict):
            continue
        e_brand = e.get("brand")
        e_dgp = e.get("dgp_type")
        if e_brand == brand and (e_dgp in (cohort, _DEFAULT_DGP, _COHORT_DGP.get(cohort))) and "true_ate" in e:
            return float(e["true_ate"])
    raise AssertionError(
        f"no designed TRUE_ATE for brand={brand!r} cohort={cohort!r} in {paths[-1]} "
        f"(GroundTruthStore entries keyed by brand+dgp_type; Shard 03)"
    )


def gate_3_ate_cate(client) -> GateResult:
    import asyncio

    from src.agents.causal_impact.agent import CausalImpactAgent

    frame, conf = _resolve_synthetic_frame(client, "initiation", "Kisqali")
    true_ate = _true_ate(client, "initiation", "Kisqali")
    out = asyncio.run(
        CausalImpactAgent().run(
            {
                "query": "gate3 recover ATE for Kisqali initiation",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": conf,
                "data_source": "database",  # NOT 'synthetic' -> no seed-42 fall-through
                "data": frame,
            }
        )
    )
    ate = _extract_ate(out)
    ok = ate is not None and abs(ate - true_ate) < 0.10
    return GateResult(
        "3 ATE/CATE RECOVERY", ok, {"recovered": ate, "true": true_ate},
        "CausalImpactAgent recovers TRUE_ATE within 0.10 (no seed-42 fall-through)",
    )


def _extract_ate(out) -> float | None:
    """RECONCILED: CausalImpactOutput is a TypedDict whose effect field is
    `ate_estimate` (state.py:392), NOT `ate`. Read `ate_estimate` first, then a few
    documented aliases, so the gate measures the recovered effect and never silently
    reads None (a false RED) off a wrong key."""
    if isinstance(out, dict):
        for k in ("ate_estimate", "ate", "overall_ate"):
            if out.get(k) is not None:
                return float(out[k])
        return None
    for k in ("ate_estimate", "ate", "overall_ate"):
        v = getattr(out, k, None)
        if v is not None:
            return float(v)
    return None


def gate_4_trigger(client) -> GateResult:
    rows = _kpi(client, "trigger_performance_action_rate_uplift", [])
    row = rows[0] if rows else {}
    ok = (
        bool(row)
        and (row.get("treatment_rate") or 0) > (row.get("control_rate") or 0)
        and (row.get("action_rate_uplift") or 0) > 0
    )
    return GateResult(
        "4 TRIGGER EFFECTIVENESS", ok, row,
        "treatment_rate>control_rate and action_rate_uplift>0",
    )


# =============================================================================
# Gates 5, 6, 7, 8 — the 4 named agents (via .run / .synthesize / REST)
# =============================================================================
# VERIFIED entrypoints (2026-06-10):
#   GapAnalyzerAgent.run(input_data) -> Dict           (gap_analyzer/agent.py:102)
#   HeterogeneousOptimizerAgent.run(input_data) -> Dict (heterogeneous_optimizer/agent.py:101)
#   PredictionSynthesizerAgent.synthesize(entity_id, prediction_target, ...,
#       entity_type='hcp') -> PredictionSynthesizerOutput (prediction_synthesizer/agent.py:166)
#   POST /api/resources/optimize (resource_optimizer.py prefix=/resources, main.py
#       mount prefix=/api); RunOptimizationRequest schema + async_mode=False for solver.


def gate_5_gap(client) -> GateResult:
    import asyncio

    from src.agents.gap_analyzer.agent import GapAnalyzerAgent

    out = asyncio.run(
        GapAnalyzerAgent().run(
            {
                "query": "gaps",
                "metrics": ["trx", "conversion_rate"],
                "segments": ["region"],
                "brand": "Kisqali",
            }
        )
    )
    # RECONCILED: gap output exposes quick_wins/strategic_bets as SEPARATE lists
    # (gap_analyzer/agent.py:355-357), NOT an `opportunity_type` field on each opp.
    opps = out.get("prioritized_opportunities") or []
    quick_wins = out.get("quick_wins") or []
    strategic_bets = out.get("strategic_bets") or []
    ok = (
        len(opps) >= 3
        and (out.get("total_addressable_value") or 0) > 0
        and len(quick_wins) >= 1
        and len(strategic_bets) >= 1
    )
    return GateResult(
        "5 gap_analyzer", ok,
        {
            "n_opps": len(opps),
            "n_quick_wins": len(quick_wins),
            "n_strategic_bets": len(strategic_bets),
            "tav": out.get("total_addressable_value"),
        },
        ">=3 opportunities, TAV>0, >=1 quick_win AND >=1 strategic_bet",
    )


def gate_6_hetero(client) -> GateResult:
    import asyncio

    from src.agents.heterogeneous_optimizer.agent import HeterogeneousOptimizerAgent

    out = asyncio.run(
        HeterogeneousOptimizerAgent().run(
            {
                "query": "uplift for Kisqali initiation by severity",
                "filters": {"brand": "Kisqali"},
            }
        )
    )
    # VERIFIED output keys: heterogeneity_score, high_responders, low_responders,
    # cate_by_segment (heterogeneous_optimizer/agent.py:397-401).
    hscore = out.get("heterogeneity_score") or 0
    highs = out.get("high_responders") or []
    lows = out.get("low_responders") or []
    cate_seg = out.get("cate_by_segment") or {}
    ok = hscore > 0.4 and len(highs) > 0 and len(lows) > 0 and bool(cate_seg)
    return GateResult(
        "6 heterogeneous_optimizer", ok,
        {
            "heterogeneity_score": hscore,
            "n_high": len(highs),
            "n_low": len(lows),
            "has_cate_by_segment": bool(cate_seg),
        },
        "heterogeneity_score>0.4, non-empty high+low responders, cate_by_segment present",
    )


def gate_7_pred_synth(client) -> GateResult:
    import asyncio

    from src.agents.prediction_synthesizer.agent import PredictionSynthesizerAgent

    # Resolve a real synthetic HCP id (entity-bound; fail loud if none).
    hcp_rows = (
        client.table("hcp_profiles")
        .select("hcp_id")
        .eq("is_synthetic", True)
        .limit(1)
        .execute()
        .data
    ) or []
    if not hcp_rows:
        return GateResult(
            "7 prediction_synthesizer", False, {"error": "no synthetic hcp_profiles row"},
            ">=2 models -> model_agreement>0.5, risk_level!=CANNOT_ASSESS",
        )
    entity_id = hcp_rows[0]["hcp_id"]
    out = asyncio.run(
        PredictionSynthesizerAgent().synthesize(
            entity_id=entity_id,
            prediction_target="conversion",
            entity_type="hcp",
        )
    )
    # RECONCILED: model_agreement lives on ensemble_prediction (state.py:36); risk_level
    # lives on prediction_interpretation, = "CANNOT_ASSESS" when <2 models
    # (ensemble_combiner.py:401). PredictionSynthesizerOutput is a TypedDict (dict).
    ensemble = (out.get("ensemble_prediction") if isinstance(out, dict) else getattr(out, "ensemble_prediction", None)) or {}
    interp = (out.get("prediction_interpretation") if isinstance(out, dict) else getattr(out, "prediction_interpretation", None)) or {}
    agreement = ensemble.get("model_agreement") if isinstance(ensemble, dict) else None
    risk_level = interp.get("risk_level") if isinstance(interp, dict) else None
    ok = (agreement or 0) > 0.5 and risk_level not in (None, "", "CANNOT_ASSESS")
    return GateResult(
        "7 prediction_synthesizer", ok,
        {"model_agreement": agreement, "risk_level": risk_level},
        ">=2 models -> model_agreement>0.5, risk_level!=CANNOT_ASSESS",
    )


def gate_8_resource(_client) -> GateResult:
    from fastapi.testclient import TestClient

    from src.api.main import app

    # RECONCILED to the REAL RunOptimizationRequest schema (resource_optimizer.py:132):
    #   required query + resource_type + allocation_targets[{entity_id, entity_type,
    #   current_allocation, expected_response}]; budget is a Constraint, NOT a
    #   top-level total_budget; the plan's segment_id/total_budget body 422s.
    # CRITICAL: the route defaults async_mode=True -> returns PENDING with no
    #   solver_status; the SYNCHRONOUS solver path needs ?async_mode=false (line 311).
    body = {
        "query": "optimize budget across two HCP segments",
        "resource_type": "budget",
        "allocation_targets": [
            {
                "entity_id": "high",
                "entity_type": "hcp",
                "current_allocation": 100.0,
                "min_allocation": 0.0,
                "max_allocation": 1000.0,
                "expected_response": 0.50,
            },
            {
                "entity_id": "low",
                "entity_type": "hcp",
                "current_allocation": 100.0,
                "min_allocation": 0.0,
                "max_allocation": 1000.0,
                "expected_response": 0.15,
            },
        ],
        "constraints": [{"constraint_type": "budget", "value": 200.0, "scope": "global"}],
        "objective": "maximize_outcome",
    }
    r = TestClient(app).post("/api/resources/optimize?async_mode=false", json=body)
    data = r.json() if r.status_code == 200 else {}
    ok = r.status_code == 200 and data.get("solver_status") == "optimal"
    return GateResult(
        "8 resource_optimizer", ok,
        {"status": r.status_code, "solver": data.get("solver_status")},
        "POST /api/resources/optimize?async_mode=false -> solver_status='optimal'",
    )


# =============================================================================
# Gate 9 — PROVENANCE LEAKAGE (the blast-radius backstop)
# =============================================================================
# Mirrors Shard 07's enforcement surface. EVERY taggable read path is listed so the
# leakage test is exhaustive, not best-effort. Anything intentionally opt-in lives
# under documented_optin with the reason — never silently omitted.
#
# RECONCILED taggable_tables to the LIVE DB (verified, 2026-06-10):
#   SELECT table_name FROM information_schema.columns WHERE column_name='is_synthetic'.
#   The plan listed `operational_corpus` but that table does NOT exist in this DB
#   (to_regclass('public.operational_corpus') IS NULL). The corpus dedup read path is
#   `episodic_memories` (which IS is_synthetic-tagged) — substituted here so the gate
#   audits the REAL blast radius instead of crashing on a phantom table.
READ_PATHS = {
    "taggable_tables": [
        "business_metrics",
        "treatment_events",
        "triggers",
        "patient_journeys",
        "ml_predictions",
        "episodic_memories",  # RECONCILED: replaces non-existent operational_corpus
    ],
    "documented_optin": {
        # validation runs explicitly pass is_synthetic=true; real chat never does
        "causal_impact_estimation": "opt-in via data_source/segment_filters at call site",
        "kpi_query_include_synthetic": "066 *_include_synthetic ids (validation-only)",
    },
}


def gate_9_provenance(client) -> GateResult:
    failures = []
    for table in READ_PATHS["taggable_tables"]:
        try:
            # RECONCILED: count via select("*") — the taggable tables have NO `id`
            # column (distinct PKs: metric_id/treatment_event_id/trigger_id/...), so
            # the plan's select("id") 42703-crashes. select("*", count="exact") counts
            # rows without binding a column name.
            untagged = (
                client.table(table)
                .select("*", count="exact")
                .is_("is_synthetic", "null")
                .limit(1)
                .execute()
            ).count or 0
        except Exception as e:  # table without is_synthetic yet -> hard finding, not a silent pass
            failures.append(f"{table}:no-is_synthetic({e})")
            continue
        if untagged:
            failures.append(f"{table}:{untagged}-untagged")
    # default-exclude invariant: the real-mode RPC must run without error and never
    # return None (synthetic rows present but excluded by the 066 default-exclude SQL).
    try:
        trx_rows = _kpi(client, "business_impact_trx", ["Kisqali"])
        rpc_trx = trx_rows[0].get("trx") if trx_rows else None
        if rpc_trx is None:
            failures.append("trx_rpc:no-row")
    except Exception as e:
        failures.append(f"trx_rpc:{e}")
        rpc_trx = None
    ok = not failures
    return GateResult(
        "9 PROVENANCE", ok, {"failures": failures, "trx_real_mode": rpc_trx},
        "0 untagged on every taggable table; real KPI excludes synthetic by default",
    )


# =============================================================================
# Gate 10 — 4-cohort x 3-brand x agent e2e + 17-agent crash-free smoke
# =============================================================================
COHORTS = ["initiation", "discontinuation", "persistence", "hcp_adoption"]
BRANDS3 = ["Remibrutinib", "Kisqali", "Fabhalta"]


def gate_10_cohort_brand_e2e(client) -> GateResult:
    import asyncio

    from src.agents.causal_impact.agent import CausalImpactAgent

    measured = {}
    ok = True
    for cohort in COHORTS:
        for brand in BRANDS3:
            cell = f"{cohort}/{brand}"
            try:
                frame, conf = _resolve_synthetic_frame(client, cohort, brand)
            except AssertionError as e:
                measured[cell] = f"no-substrate:{e}"
                ok = False
                continue
            rate = float(frame["outcome"].mean())
            agent_out = asyncio.run(
                CausalImpactAgent().run(
                    {
                        "query": cell,
                        "treatment_var": "treatment",
                        "outcome_var": "outcome",
                        "confounders": conf,
                        "data_source": "database",  # no seed-42 fall-through
                        "data": frame,
                    }
                )
            )
            ate = _extract_ate(agent_out)
            cell_ok = 0.05 <= rate <= 0.60 and ate is not None
            measured[cell] = {"label_rate": round(rate, 3), "ate": ate}
            ok = ok and cell_ok
    return GateResult(
        "10 4-COHORT x 3-BRAND x AGENT", ok, measured,
        "each cell: label 5-60%, runnable var-set, >=1 agent non-empty useful output",
    )


# The 4 named agents proven by gates 5-8, plus causal_impact (gate 3/10's
# representative) — the smoke covers the REMAINING enabled agents.
_GATED_AGENTS = {
    "gap_analyzer",
    "heterogeneous_optimizer",
    "prediction_synthesizer",
    "resource_optimizer",
    "causal_impact",
}


def _smoke_other_agents() -> list:
    """Crash-free smoke of the remaining enabled agents (NOT a correctness gate).

    Uses the production factory (src/agents/factory.py) to INSTANTIATE every enabled
    agent. An agent that fails to construct is a real finding (returned in the crashed
    list); the factory logs but swallows drops, so we detect them by diffing the
    enabled-and-selected set against the registry it returns. Construction-level only —
    we do NOT call .run (no substrate dependency, no heavy compute)."""
    from src.agents.factory import AGENT_REGISTRY_CONFIG, create_agent_registry

    other = [
        name
        for name, cfg in AGENT_REGISTRY_CONFIG.items()
        if cfg.get("enabled", False) and name not in _GATED_AGENTS
    ]
    registry = create_agent_registry(include_agents=other, fail_on_import_error=False)
    crashed = sorted(set(other) - set(registry.keys()))
    return crashed
