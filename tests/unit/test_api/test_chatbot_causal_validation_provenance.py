"""Chat surfacing of causal-path validation provenance (#1352).

Pre-#1352, ``_format_causal_path`` DROPPED ``validation_status`` and no chat
path could cite refutation evidence at all (the q07 finding: the live answer
"the registry exposes no refutation results" was the most accurate statement
of the actual state). Post-migration-119 the semantics are pinned —
``validation_status='validated'`` asserts "RefutationSuite evidence exists and
passed" — so chat answers must surface BOTH the status and a summary of the
evidence behind it, with three honestly-distinct states per path:

* summary dict        -> evidence rows exist (counts, gate, synthetic flag);
* ``None``            -> lookup succeeded, genuinely no evidence on record;
* lookup-failed dict  -> the evidence query errored — never presented as
  "no evidence".
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.chatbot_tools import (
    _format_causal_path,
    _refutation_evidence_entry,
    _summarize_refutation_rows,
    causal_analysis_tool,
)

VALIDATED_PATH = {
    "path_id": "scp_001cdc6864504",
    "start_node": "peer_influence_score",
    "end_node": "treatment_initiated",
    "intermediate_nodes": ["hcp_engagement"],
    "causal_effect_size": 0.124,
    "confidence_level": 0.87,
    "method_used": "dowhy_backdoor",
    "time_lag_days": 14,
    "business_impact_estimate": 120000.0,
    "brand": "Fabhalta",
    "validation_status": "validated",
    "is_synthetic": True,
}


def _seeded_evidence_rows(n_passed=5, n_failed=0, n_warning=0, synthetic=True):
    """causal_validations-shaped rows as migration 119 seeds them."""
    rows = []
    statuses = ["passed"] * n_passed + ["failed"] * n_failed + ["warning"] * n_warning
    test_types = [
        "placebo_treatment",
        "random_common_cause",
        "data_subset",
        "bootstrap",
        "sensitivity_e_value",
    ]
    for i, status in enumerate(statuses):
        rows.append(
            {
                "estimate_id": "d28b51a1-775c-531d-9998-e7240741ab10",
                "estimate_source": "causal_paths",
                "test_type": test_types[i % len(test_types)],
                "status": status,
                "gate_decision": "proceed",
                "confidence_score": 0.87,
                "details_json": (
                    {"is_synthetic": True, "provenance": "dgp_backfill_migration_119"}
                    if synthetic
                    else {}
                ),
                "created_at": f"2026-07-30T00:00:0{i}+00:00",
            }
        )
    return rows


# ---------------------------------------------------------------- formatter


@pytest.mark.unit
def test_format_causal_path_surfaces_validation_status():
    out = _format_causal_path(VALIDATED_PATH)
    assert out["validation_status"] == "validated"


@pytest.mark.unit
def test_format_causal_path_carries_refutation_evidence_summary():
    summary = _summarize_refutation_rows(_seeded_evidence_rows())
    out = _format_causal_path(VALIDATED_PATH, refutation_evidence=summary)
    assert out["refutation_evidence"]["tests_total"] == 5
    assert out["refutation_evidence"]["tests_passed"] == 5
    # Pre-#1352 shape must survive (regression guard for existing consumers).
    assert out["cause"] == "peer_influence_score"
    assert out["confidence"] == 0.87


@pytest.mark.unit
def test_format_causal_path_defaults_refutation_evidence_none():
    out = _format_causal_path(VALIDATED_PATH)
    assert out["refutation_evidence"] is None


# ---------------------------------------------------------------- summarizer


@pytest.mark.unit
def test_summarize_counts_statuses_and_flags_synthetic():
    summary = _summarize_refutation_rows(_seeded_evidence_rows(n_passed=3, n_warning=1))
    assert summary["tests_total"] == 4
    assert summary["tests_passed"] == 3
    assert summary["tests_warning"] == 1
    assert summary["tests_failed"] == 0
    assert summary["gate_decision"] == "proceed"
    assert summary["evidence_is_synthetic"] is True
    assert summary["confidence_score"] == pytest.approx(0.87)
    assert summary["latest_test_at"] == "2026-07-30T00:00:03+00:00"


@pytest.mark.unit
def test_summarize_handles_details_json_as_string():
    rows = _seeded_evidence_rows(n_passed=1)
    rows[0]["details_json"] = '{"is_synthetic": true, "provenance": "dgp_backfill_migration_119"}'
    summary = _summarize_refutation_rows(rows)
    assert summary["evidence_is_synthetic"] is True


@pytest.mark.unit
def test_summarize_real_evidence_not_flagged_synthetic():
    summary = _summarize_refutation_rows(_seeded_evidence_rows(n_passed=2, synthetic=False))
    assert summary["evidence_is_synthetic"] is False


@pytest.mark.unit
def test_summarize_gate_priority_block_wins():
    rows = _seeded_evidence_rows(n_passed=4, n_failed=1)
    rows[-1]["gate_decision"] = "block"
    summary = _summarize_refutation_rows(rows)
    assert summary["gate_decision"] == "block"
    assert summary["tests_failed"] == 1


@pytest.mark.unit
def test_summarize_empty_returns_none():
    assert _summarize_refutation_rows([]) is None


# ------------------------------------------------------------- three states


@pytest.mark.unit
def test_refutation_entry_three_states():
    summaries = {"scp_001cdc6864504": {"tests_total": 5}}
    # evidence exists
    assert _refutation_evidence_entry("scp_001cdc6864504", summaries) == {"tests_total": 5}
    # lookup ok, no evidence on record -> honest None
    assert _refutation_evidence_entry("scp_other", summaries) is None
    # lookup failed -> marked, NOT presented as absence
    failed = _refutation_evidence_entry("scp_001cdc6864504", None)
    assert failed is not None
    assert failed.get("lookup_failed") is True


# ------------------------------------------------------------- tool wiring


def _mock_path_repo(paths, outcomes=None):
    repo = MagicMock()
    repo.search_paths_for_outcome = AsyncMock(return_value=paths)
    repo.get_distinct_outcomes = AsyncMock(return_value=outcomes or [])
    return repo


def _mock_validation_repo(rows_by_path):
    repo = MagicMock()
    repo.get_rows_for_paths = AsyncMock(return_value=rows_by_path)
    return repo


@pytest.mark.unit
@pytest.mark.asyncio
async def test_causal_analysis_tool_results_carry_validation_provenance():
    vrepo = _mock_validation_repo({"scp_001cdc6864504": _seeded_evidence_rows()})
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch(
            "src.api.routes.chatbot_tools.CausalPathRepository",
            return_value=_mock_path_repo([dict(VALIDATED_PATH)]),
        ),
        patch("src.api.routes.chatbot_tools.CausalValidationRepository", return_value=vrepo),
        patch("src.api.routes.chatbot_tools.kpi_include_synthetic", return_value=True),
    ):
        result = await causal_analysis_tool.ainvoke(
            {"kpi_name": "treatment initiation", "brand": "Fabhalta"}
        )
    assert result["success"] is True
    entry = result["results"][0]
    assert entry["validation_status"] == "validated"
    ev = entry["refutation_evidence"]
    assert ev["tests_total"] == 5
    assert ev["tests_passed"] == 5
    assert ev["evidence_is_synthetic"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_causal_analysis_tool_evidence_lookup_failure_degrades_honestly():
    vrepo = MagicMock()
    vrepo.get_rows_for_paths = AsyncMock(side_effect=RuntimeError("db down"))
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch(
            "src.api.routes.chatbot_tools.CausalPathRepository",
            return_value=_mock_path_repo([dict(VALIDATED_PATH)]),
        ),
        patch("src.api.routes.chatbot_tools.CausalValidationRepository", return_value=vrepo),
        patch("src.api.routes.chatbot_tools.kpi_include_synthetic", return_value=True),
    ):
        result = await causal_analysis_tool.ainvoke(
            {"kpi_name": "treatment initiation", "brand": "Fabhalta"}
        )
    # The tool must still answer (evidence is enrichment, not a gate) ...
    assert result["success"] is True
    entry = result["results"][0]
    assert entry["validation_status"] == "validated"
    # ... but the failed lookup must NOT masquerade as "no evidence".
    assert entry["refutation_evidence"].get("lookup_failed") is True
