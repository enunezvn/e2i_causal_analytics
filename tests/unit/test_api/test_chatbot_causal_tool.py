"""Unit coverage for ``causal_analysis_tool`` (post-rewire, 2026-07-07).

Old behavior (structurally broken): the tool ran a generic ``hybrid_search``
and filtered by ``metadata.get("confidence", r.score) >= min_confidence``.
No RAG result carries a ``confidence`` key, so the filter compared an RRF
rank-fusion score (ceiling ~0.03) against a causal-confidence threshold
(0.7) — guaranteed 0 chains for EVERY query, misreported to the user as
"no paths met the confidence threshold".

New contract:
- Query the ``causal_paths`` registry (real ``confidence_level`` 0-1 values)
  via ``CausalPathRepository.search_paths_for_outcome``.
- Synthetic provenance follows the platform gate (``kpi_include_synthetic``),
  labeled via ``data_source`` — same convention as the KPI tools (#893).
- When the registry has no paths for the requested KPI, say so honestly via
  ``substrate_coverage`` (the registry models patient-journey outcomes, not
  commercial KPIs) instead of implying an analysis ran and found nothing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes.chatbot_tools import _query_causal_chains, causal_analysis_tool

SAMPLE_PATH = {
    "path_id": "CP-0001",
    "start_node": "peer_influence_score",
    "end_node": "treatment_initiated",
    "intermediate_nodes": ["hcp_engagement"],
    "causal_effect_size": 0.124,
    "confidence_level": 0.87,
    "method_used": "dowhy_backdoor",
    "time_lag_days": 14,
    "business_impact_estimate": 120000.0,
    "brand": "Fabhalta",
    "is_synthetic": True,
}


def _mock_repo(paths, outcomes=None):
    repo = MagicMock()
    repo.search_paths_for_outcome = AsyncMock(return_value=paths)
    repo.get_distinct_outcomes = AsyncMock(return_value=outcomes or [])
    return repo


@pytest.mark.unit
@pytest.mark.asyncio
async def test_paths_found_carry_registry_confidence():
    repo = _mock_repo([SAMPLE_PATH])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.CausalPathRepository", return_value=repo),
        patch("src.api.routes.chatbot_tools.kpi_include_synthetic", return_value=True),
    ):
        result = await causal_analysis_tool.ainvoke(
            {"kpi_name": "treatment initiation", "brand": "Fabhalta"}
        )
    assert result["success"] is True
    assert result["causal_chains_found"] == 1
    assert result["results"][0]["confidence"] == 0.87
    assert result["results"][0]["cause"] == "peer_influence_score"
    assert result["results"][0]["effect"] == "treatment_initiated"
    assert result["analysis_type"] == "causal_paths_registry"
    assert result["data_source"] == "synthetic"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_matches_disclose_substrate_coverage():
    """TRx has no causal paths — the tool must say the SUBSTRATE doesn't cover it,
    not imply a real analysis found nothing above threshold."""
    repo = _mock_repo([], outcomes=["conversion_flag", "treatment_initiated", "persistent_180d"])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.CausalPathRepository", return_value=repo),
        patch("src.api.routes.chatbot_tools.kpi_include_synthetic", return_value=True),
    ):
        result = await causal_analysis_tool.ainvoke({"kpi_name": "TRx", "brand": "Fabhalta"})
    assert result["success"] is True
    assert result["causal_chains_found"] == 0
    coverage = result["substrate_coverage"]
    assert "TRx" in coverage["note"]
    assert "conversion_flag" in coverage["outcomes_covered"]
    # The note must frame this as a coverage gap, not a null finding.
    assert "does not" in coverage["note"] or "no causal paths" in coverage["note"].lower()
    # Post commercial-grain seed the registry DOES model commercial volume
    # KPIs, so the note must not claim they are uncovered as a class — the
    # honest note names the missing KPI and points at outcomes_covered.
    assert "not commercial" not in coverage["note"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_min_confidence_and_brand_forwarded_to_repo():
    repo = _mock_repo([SAMPLE_PATH])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.CausalPathRepository", return_value=repo),
        patch("src.api.routes.chatbot_tools.kpi_include_synthetic", return_value=True),
    ):
        await causal_analysis_tool.ainvoke(
            {"kpi_name": "persistence", "brand": "Kisqali", "min_confidence": 0.9}
        )
    kwargs = repo.search_paths_for_outcome.await_args.kwargs
    assert kwargs["min_confidence"] == 0.9
    assert kwargs["brand"] == "Kisqali"
    assert kwargs["include_synthetic"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_real_mode_excludes_synthetic_and_labels_database():
    repo = _mock_repo([{**SAMPLE_PATH, "is_synthetic": False}])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.CausalPathRepository", return_value=repo),
        patch("src.api.routes.chatbot_tools.kpi_include_synthetic", return_value=False),
    ):
        result = await causal_analysis_tool.ainvoke({"kpi_name": "persistence"})
    assert repo.search_paths_for_outcome.await_args.kwargs["include_synthetic"] is False
    assert result["data_source"] == "database"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_causal_chains_kpi_branch_uses_registry_not_rag():
    """e2i_data_query_tool's causal_chain branch had the same RRF detour — it must
    now go through the registry too, and never call hybrid_search."""
    from datetime import datetime, timezone

    repo = _mock_repo([SAMPLE_PATH])
    with (
        patch("src.api.routes.chatbot_tools.get_async_supabase_client", new=AsyncMock()),
        patch("src.api.routes.chatbot_tools.CausalPathRepository", return_value=repo),
        patch("src.api.routes.chatbot_tools.kpi_include_synthetic", return_value=True),
        patch("src.api.routes.chatbot_tools.hybrid_search", new=AsyncMock()) as rag,
    ):
        result = await _query_causal_chains(
            brand="Fabhalta",
            kpi_name="treatment initiation",
            since=datetime(2026, 6, 1, tzinfo=timezone.utc),
            limit=10,
        )
    rag.assert_not_awaited()
    assert result["success"] is True
    assert result["count"] == 1
    assert result["data"][0]["confidence"] == 0.87


@pytest.mark.unit
def test_outcome_match_tokens_normalizes_free_text():
    from src.repositories.causal_path import outcome_match_tokens

    # Free text tokenizes to 6-char stem prefixes so morphology bridges
    # ("initiation" must match node "treatment_initiated").
    assert outcome_match_tokens("treatment initiation") == ["treatm", "initia"]
    assert outcome_match_tokens("TRx") == ["trx"]
    # Short/empty tokens drop; morphological variants dedupe to one prefix.
    assert outcome_match_tokens("a b") == []
    assert outcome_match_tokens("Persistence & persistent") == ["persis"]
