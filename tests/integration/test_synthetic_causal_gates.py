"""Shard 11 — faithful acceptance-gate harness for the synthetic causal-validation
dataset. NO MOCKING. Each test wraps one INDEX global-gate (1-11) and asserts a
MEASURED value against the LIVE docker Supabase. Run gate: E2I_DB_INTEGRATION=1 -n0.

Many gates only go GREEN after the orchestrator runs the full synthetic load against
the faithful docker Supabase (rolling dates, recoverable DGP, two-arm triggers, the
input_model dispatcher bridges, the ground-truth sidecar). Before that load they fail
RED for the RIGHT reason — no-substrate / stale-window / missing-artifact — never on a
typo. The harness binds only to columns/RPCs/signatures verified to exist (2026-06-10).
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 to run (-n0)",
)

from scripts.validate_synthetic_causal import _kpi  # noqa: E402
from src.api.dependencies.supabase_client import get_supabase  # noqa: E402


@pytest.fixture(scope="module")
def client():
    c = get_supabase()
    if c is None:
        pytest.skip("no Supabase client (SUPABASE_URL/ANON_KEY unset)")
    return c


def test_disproof_synthetic_substrate_present(client):
    """CHEAPEST DISPROOF: synthetic rows must EXIST + be tagged before any gate
    can recover a known truth. Fail loud if a producing shard (02-09) regressed."""
    # RECONCILED: business_metrics has no `id` column (PK is `metric_id`); selecting
    # a non-existent column 42703-crashes. `select("*", count="exact")` counts rows
    # without binding a column name (verified \d business_metrics, 2026-06-10).
    resp = (
        client.table("business_metrics")
        .select("*", count="exact")
        .eq("is_synthetic", True)
        .limit(1)
        .execute()
    )
    assert (resp.count or 0) > 0, (
        "no synthetic business_metrics rows tagged is_synthetic=true; "
        "run scripts/load_synthetic_data.py (Shard 02) before the gates"
    )


# =============================================================================
# Gates 1 & 2 — DATE-FRESHNESS + KPI->DASHBOARD
# =============================================================================


def test_gate_1_date_freshness(client):
    for brand in ("Remibrutinib", "Kisqali", "Fabhalta"):
        trx = _kpi(client, "business_impact_trx", [brand])
        assert trx and (trx[0].get("trx") or 0) > 0, (
            f"{brand} TRx==0 over NOW()-30d (staleness; load with --anchor-to-now)"
        )
    conv = _kpi(client, "business_impact_conversion_rate", [])
    assert conv and (conv[0].get("conversion_rate") or 0) > 0, (
        "conversion_rate==0 over NOW()-30d window"
    )


def test_gate_2_kpi_dashboard(client):
    import asyncio

    from src.api.routes.copilotkit import get_kpi_summary

    summary = asyncio.run(get_kpi_summary("Kisqali"))
    assert summary.get("data_source") == "database", summary  # NOT 'unavailable'/'fallback'
    nonnull = [v for v in (summary.get("metrics") or {}).values() if v not in (None, 0)]
    assert len(nonnull) >= 4, f"too few non-zero metrics: {summary.get('metrics')}"


# =============================================================================
# Gates 3 & 4 — ATE/CATE RECOVERY + TRIGGER EFFECTIVENESS
# =============================================================================


def test_gate_3_ate_cate_recovery(client):
    import asyncio

    from scripts.validate_synthetic_causal import (
        _extract_ate,
        _resolve_synthetic_frame,
        _true_ate,
    )
    from src.agents.causal_impact.agent import CausalImpactAgent

    frame, conf = _resolve_synthetic_frame(client, cohort="initiation", brand="Kisqali")
    true_ate = _true_ate(client, "initiation", "Kisqali")  # RED here pre-sidecar
    out = asyncio.run(
        CausalImpactAgent().run(
            {
                "query": "recover ATE for Kisqali initiation",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": conf,
                "data_source": "database",  # NOT 'synthetic' -> no seed-42 fall-through
                "data": frame,
            }
        )
    )
    assert out.get("status") != "failed", out
    ate = _extract_ate(out)
    assert ate is not None, "recovered ATE is None"
    assert abs(ate - true_ate) < 0.10, f"ATE {ate} off TRUE_ATE {true_ate}"


def test_gate_4_trigger_effectiveness(client):
    rows = _kpi(client, "trigger_performance_action_rate_uplift", [])
    assert rows, "trigger_performance_action_rate_uplift returned no rows (no two-arm data)"
    row = rows[0]
    assert (row.get("treatment_rate") or 0) > (row.get("control_rate") or 0), row
    assert (row.get("action_rate_uplift") or 0) > 0, row


# =============================================================================
# Gates 5, 6, 7, 8 — the 4 named agents
# =============================================================================


def test_gate_5_gap_analyzer(client):
    from scripts.validate_synthetic_causal import gate_5_gap

    res = gate_5_gap(client)
    assert res.ok, res.measured


def test_gate_6_heterogeneous_optimizer(client):
    from scripts.validate_synthetic_causal import gate_6_hetero

    res = gate_6_hetero(client)
    assert res.ok, res.measured


def test_gate_7_prediction_synthesizer(client):
    from scripts.validate_synthetic_causal import gate_7_pred_synth

    res = gate_7_pred_synth(client)
    assert res.ok, res.measured


def test_gate_8_resource_optimizer():
    from scripts.validate_synthetic_causal import gate_8_resource

    res = gate_8_resource(None)
    assert res.ok, res.measured


# =============================================================================
# Gate 9 — PROVENANCE LEAKAGE
# =============================================================================


def test_gate_9_provenance_default_exclude(client):
    from scripts.validate_synthetic_causal import READ_PATHS

    # (a) tagging completeness on each blast-radius table from Shard 07
    for table in READ_PATHS["taggable_tables"]:
        untagged = (
            client.table(table)
            .select("*", count="exact")
            .is_("is_synthetic", "null")
            .limit(1)
            .execute()
        )
        assert (untagged.count or 0) == 0, f"{table} has untagged rows; partial-coverage leakage"
    # (b) real-mode KPI runs and excludes synthetic by default (066 default-exclude SQL);
    # it must return a row (non-None), proving the RPC path is wired, not fabricated.
    trx_rows = _kpi(client, "business_impact_trx", ["Kisqali"])
    assert trx_rows and trx_rows[0].get("trx") is not None, trx_rows


# =============================================================================
# Gate 10 — 4-cohort x 3-brand x agent e2e + 17-agent crash-free smoke
# =============================================================================


def test_gate_10_cohort_brand_agent_e2e(client):
    from scripts.validate_synthetic_causal import gate_10_cohort_brand_e2e

    res = gate_10_cohort_brand_e2e(client)
    assert res.ok, res.measured  # measured maps each cell -> {label_rate, ate}


def test_other_17_agents_smoke():
    from scripts.validate_synthetic_causal import _smoke_other_agents

    crashed = _smoke_other_agents()
    assert not crashed, f"agents crashed on smoke: {crashed}"


# =============================================================================
# Gate 11 — CHAT-PATH e2e (decision §3 proof)
# =============================================================================


def test_gate_11_chat_path_e2e(client):
    from scripts.validate_synthetic_causal import gate_11_chat_path

    res = gate_11_chat_path(client)
    assert res.ok, res.measured  # measured -> {dispatched, ordered_cate}
